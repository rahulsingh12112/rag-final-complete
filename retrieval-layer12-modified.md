## Layer 12: GitLab CI/CD + ArgoCD — Production RAG Retrieval Service Deployment

RAG Retrieval API — jo hybrid search (dense + sparse + reranking) karta hai — ko production mein deploy karna with full pipeline + retrieval quality gates + external access.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
rag-retrieval-service/
├── src/
│   ├── app.py                  # FastAPI — main RAG retrieval API
│   ├── retriever.py            # HybridRetriever (dense + sparse + RRF)
│   ├── reranker.py             # Cross-encoder reranking
│   ├── query_processor.py      # Query expansion, filter extraction
│   ├── config.py               # Environment config
│   ├── requirements.txt
│   └── tests/
│       ├── test_retriever.py
│       ├── test_reranker.py
│       ├── test_app.py
│       └── test_integration.py
├── eval/
│   ├── test_set.json           # 50+ queries with expected relevant docs
│   ├── eval_retrieval.py       # Recall@5, MRR, NDCG, latency checks
│   └── baseline_metrics.json   # Previous metrics (regression detection)
├── Dockerfile
├── k8s/
│   ├── deployment.yaml         # RAG API pods
│   ├── service.yaml
│   ├── ingress.yaml            # External access (cluster ke BAHAR se)
│   ├── configmap.yaml
│   ├── hpa.yaml                # Auto-scaling
│   ├── pdb.yaml                # Pod Disruption Budget
│   └── dependencies/
│       ├── qdrant-statefulset.yaml    # Dense search (vector DB)
│       ├── elasticsearch.yaml         # Sparse search (BM25)
│       ├── redis.yaml                 # Query cache
│       └── networkpolicy.yaml
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── scripts/
│   └── smoke-test.sh
├── .gitlab-ci.yml
└── README.md
```

### Step 1: RAG Retrieval API (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from retriever import HybridRetriever
from reranker import Reranker
from config import settings
import time
import os

app = FastAPI(title="RAG Retrieval Service", version="1.0.0")

# Metrics
RETRIEVE_COUNTER = Counter("rag_retrieve_total", "Total retrieve requests", ["status"])
RETRIEVE_LATENCY = Histogram("rag_retrieve_duration_seconds", "Retrieve latency",
                             buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
RERANK_LATENCY = Histogram("rag_rerank_duration_seconds", "Rerank latency")

# Init components
retriever = HybridRetriever(
    vector_db_url=settings.VECTOR_DB_URL,
    bm25_url=settings.BM25_URL,
    embedding_url=settings.EMBEDDING_SERVICE_URL
)
reranker = Reranker(model_url=settings.RERANKER_URL)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True
    filters: dict | None = None
    collection: str = "documents"


class RetrieveResponse(BaseModel):
    results: list[dict]
    latency_ms: float
    dense_candidates: int
    sparse_candidates: int
    reranked: bool


@app.get("/health")
def health():
    """Liveness probe"""
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    """Readiness probe — all dependencies connected?"""
    checks = {}
    try:
        checks["vector_db"] = retriever.check_vector_db()
        checks["bm25_index"] = retriever.check_bm25()
        checks["reranker"] = reranker.check_health()
        all_ready = all(checks.values())
        if not all_ready:
            raise HTTPException(status_code=503, detail=f"Not ready: {checks}")
        return {"status": "ready", "dependencies": checks}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    """Main retrieval endpoint — hybrid search + reranking"""
    start = time.time()
    try:
        # Stage 1: Hybrid search (dense + sparse + RRF)
        candidates, dense_count, sparse_count = retriever.hybrid_search(
            query=request.query,
            collection=request.collection,
            top_k=100,  # Get top-100 candidates
            filters=request.filters
        )

        # Stage 2: Reranking (if enabled)
        reranked = False
        if request.use_reranking and len(candidates) > 0:
            rerank_start = time.time()
            candidates = reranker.rerank(request.query, candidates, top_k=request.top_k)
            RERANK_LATENCY.observe(time.time() - rerank_start)
            reranked = True
        else:
            candidates = candidates[:request.top_k]

        latency = (time.time() - start) * 1000
        RETRIEVE_COUNTER.labels(status="success").inc()
        RETRIEVE_LATENCY.observe(time.time() - start)

        return RetrieveResponse(
            results=candidates,
            latency_ms=round(latency, 2),
            dense_candidates=dense_count,
            sparse_candidates=sparse_count,
            reranked=reranked
        )
    except Exception as e:
        RETRIEVE_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/dense")
def retrieve_dense_only(request: RetrieveRequest):
    """Dense-only retrieval (for comparison/fallback)"""
    results = retriever.dense_search(request.query, request.collection, request.top_k)
    return {"results": results, "method": "dense_only"}


@app.post("/retrieve/sparse")
def retrieve_sparse_only(request: RetrieveRequest):
    """Sparse-only retrieval (BM25)"""
    results = retriever.sparse_search(request.query, request.top_k)
    return {"results": results, "method": "sparse_bm25"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics"""
    return generate_latest()


@app.get("/stats")
def stats():
    """Retrieval pipeline stats"""
    return {
        "vector_db_collections": retriever.list_collections(),
        "bm25_index_count": retriever.bm25_doc_count(),
        "cache_hit_rate": retriever.cache_hit_rate()
    }
```

### Step 2: Hybrid Retriever (`src/retriever.py`)

```python
import requests
import hashlib
import json
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import settings


class HybridRetriever:
    """Production hybrid retrieval: Dense + BM25 + RRF + Redis cache"""

    def __init__(self, vector_db_url, bm25_url, embedding_url):
        self.qdrant = QdrantClient(url=vector_db_url, grpc_port=6334, prefer_grpc=True, timeout=30)
        self.bm25_url = bm25_url
        self.embedding_url = embedding_url
        self.cache = redis.Redis(host=settings.REDIS_HOST, port=6379, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour

    def hybrid_search(self, query: str, collection: str, top_k: int = 100,
                      filters: dict = None) -> tuple:
        """Run dense + sparse in parallel, fuse with RRF"""
        # Check cache
        cache_key = self._cache_key(query, collection, filters)
        cached = self.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            return data["results"], data["dense_count"], data["sparse_count"]

        # Dense search
        query_embedding = self._get_embedding(query)
        dense_results = self.dense_search_raw(query_embedding, collection, top_k, filters)

        # Sparse search (BM25)
        sparse_results = self.sparse_search(query, top_k)

        # RRF Fusion
        fused = self.reciprocal_rank_fusion(
            [dense_results, sparse_results], k=60
        )

        # Cache result
        result_data = {
            "results": fused[:top_k],
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results)
        }
        self.cache.setex(cache_key, self.cache_ttl, json.dumps(result_data))

        return fused[:top_k], len(dense_results), len(sparse_results)

    def dense_search(self, query: str, collection: str, top_k: int) -> list:
        """Dense-only search (public endpoint)"""
        query_embedding = self._get_embedding(query)
        return self.dense_search_raw(query_embedding, collection, top_k)

    def dense_search_raw(self, query_embedding, collection, top_k, filters=None):
        """Raw dense search with embedding vector"""
        query_filter = None
        if filters:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            query_filter = Filter(must=conditions)

        results = self.qdrant.search(
            collection_name=collection,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )
        return [{"id": r.id, "text": r.payload.get("text", ""), "score": r.score,
                 "metadata": r.payload} for r in results]

    def sparse_search(self, query: str, top_k: int) -> list:
        """BM25 search via Elasticsearch"""
        resp = requests.post(f"{self.bm25_url}/documents/_search", json={
            "query": {"match": {"text": query}},
            "size": top_k
        }, timeout=10)
        hits = resp.json().get("hits", {}).get("hits", [])
        return [{"id": h["_id"], "text": h["_source"].get("text", ""),
                 "score": h["_score"], "metadata": h["_source"]} for h in hits]

    def reciprocal_rank_fusion(self, result_lists: list, k: int = 60) -> list:
        """RRF: rank-based fusion (score-agnostic)"""
        scores = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = str(doc["id"])
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0, "doc": doc}
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked]

    def _get_embedding(self, text: str) -> list:
        resp = requests.post(f"{self.embedding_url}/embed",
                             json={"texts": [text], "is_query": True}, timeout=15)
        return resp.json()["embeddings"][0]

    def _cache_key(self, query, collection, filters):
        raw = f"{query}:{collection}:{json.dumps(filters or {}, sort_keys=True)}"
        return f"rag:search:{hashlib.md5(raw.encode()).hexdigest()}"

    def check_vector_db(self) -> bool:
        try:
            self.qdrant.get_collections()
            return True
        except:
            return False

    def check_bm25(self) -> bool:
        try:
            r = requests.get(f"{self.bm25_url}/_cluster/health", timeout=5)
            return r.status_code == 200
        except:
            return False

    def list_collections(self):
        return [c.name for c in self.qdrant.get_collections().collections]

    def bm25_doc_count(self):
        try:
            r = requests.get(f"{self.bm25_url}/documents/_count", timeout=5)
            return r.json().get("count", 0)
        except:
            return -1

    def cache_hit_rate(self):
        info = self.cache.info()
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return round(hits / total, 3) if total > 0 else 0.0
```

### Step 3: Reranker (`src/reranker.py`)

```python
import requests
from config import settings


class Reranker:
    """Cross-encoder reranking (managed API or self-hosted)"""

    def __init__(self, model_url: str):
        self.model_url = model_url

    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []

        texts = [doc.get("text", "") for doc in documents]

        # Call reranker service (could be Cohere API, self-hosted ms-marco, etc.)
        resp = requests.post(f"{self.model_url}/rerank", json={
            "query": query,
            "documents": texts,
            "top_k": top_k
        }, timeout=30)

        if resp.status_code != 200:
            # Fallback: return original order if reranker fails
            return documents[:top_k]

        reranked_indices = resp.json()["results"]  # [{index: 0, score: 0.95}, ...]
        reranked_docs = []
        for item in reranked_indices[:top_k]:
            doc = documents[item["index"]]
            doc["rerank_score"] = item["score"]
            reranked_docs.append(doc)

        return reranked_docs

    def check_health(self) -> bool:
        try:
            r = requests.get(f"{self.model_url}/health", timeout=5)
            return r.status_code == 200
        except:
            return False
```

### Step 4: Config (`src/config.py`)

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VECTOR_DB_URL: str = "http://qdrant:6333"
    BM25_URL: str = "http://elasticsearch:9200"
    EMBEDDING_SERVICE_URL: str = "http://embedding-service:8080"
    RERANKER_URL: str = "http://reranker-service:8080"
    REDIS_HOST: str = "redis"
    DEFAULT_COLLECTION: str = "documents"
    APP_VERSION: str = "unknown"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
```

### Step 5: Dockerfile (Production Multi-Stage)

```dockerfile
# ===== Builder =====
FROM python:3.11-slim AS builder
WORKDIR /build
COPY src/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== Production =====
FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ .
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
EXPOSE 8000
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "--access-logfile", "-"]
```

### Step 6: Requirements (`src/requirements.txt`)

```txt
fastapi==0.115.0
gunicorn==22.0.0
uvicorn==0.30.0
pydantic-settings==2.5.0
qdrant-client==1.12.0
grpcio==1.66.0
redis==5.2.0
requests==2.32.0
elasticsearch==8.15.0
prometheus-client==0.21.0
numpy==2.1.0
httpx==0.27.0
```

---

### Step 7: Full Production `.gitlab-ci.yml`

```yaml
# =============================================================================
# PRODUCTION PIPELINE:
# test → lint → security scan → build → push → staging deploy →
# smoke test → approval gate → prod deploy (canary 10%) → monitor →
# prod deploy (100%)
#
# BONUS: Post-deploy retrieval quality gate (Recall@5, MRR, latency)
# =============================================================================

stages:
  - test
  - lint
  - security_scan
  - build
  - push
  - staging_deploy
  - smoke_test
  - retrieval_quality_gate
  - approval_gate
  - prod_deploy_canary
  - monitor
  - prod_deploy_full

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-retrieval-service
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://rag-staging.yourdomain.com"
  PROD_URL: "https://rag.yourdomain.com"

# ─────────────── STAGE 1: TEST ───────────────
unit_tests:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install -r src/requirements.txt pytest pytest-cov pytest-asyncio httpx
  script:
    - pytest src/tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  coverage: '/TOTAL.*\s+(\d+%)$/'

integration_tests:
  stage: test
  image: python:3.11-slim
  services:
    - name: qdrant/qdrant:v1.12.0
      alias: qdrant
    - name: elasticsearch:8.15.0
      alias: elasticsearch
      variables:
        discovery.type: single-node
        xpack.security.enabled: "false"
    - name: redis:7-alpine
      alias: redis
  variables:
    VECTOR_DB_URL: "http://qdrant:6333"
    BM25_URL: "http://elasticsearch:9200"
    REDIS_HOST: "redis"
  before_script:
    - pip install -r src/requirements.txt pytest httpx
    - sleep 15  # Wait for ES to be ready
  script:
    - pytest src/tests/test_integration.py -v --timeout=120

# ─────────────── STAGE 2: LINT ───────────────
lint_python:
  stage: lint
  image: python:3.11-slim
  before_script:
    - pip install ruff mypy
  script:
    - ruff check src/ --output-format=gitlab
    - ruff format src/ --check
    - mypy src/ --ignore-missing-imports || true

lint_dockerfile:
  stage: lint
  image: hadolint/hadolint:latest-debian
  script:
    - hadolint Dockerfile --failure-threshold warning

lint_k8s:
  stage: lint
  image: garethr/kubeval:latest
  script:
    - kubeval k8s/*.yaml --strict
    - kubeval k8s/dependencies/*.yaml --strict

# ─────────────── STAGE 3: SECURITY SCAN ───────────────
sast:
  stage: security_scan
  image: python:3.11-slim
  before_script:
    - pip install bandit safety
  script:
    - bandit -r src/ -ll      # Fail on HIGH/CRITICAL
    - pip install -r src/requirements.txt
    - safety check --full-report

container_scan:
  stage: security_scan
  image: docker:24.0
  services: [docker:24.0-dind]
  before_script:
    - apk add --no-cache curl
    - curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
  script:
    - docker build -t ${DOCKER_IMAGE}:scan .
    - trivy image --exit-code 1 --severity CRITICAL,HIGH --ignore-unfixed ${DOCKER_IMAGE}:scan

secrets_scan:
  stage: security_scan
  image: zricethezav/gitleaks:latest
  script:
    - gitleaks detect --source . --report-format json --report-path gitleaks-report.json

# ─────────────── STAGE 4: BUILD ───────────────
build_image:
  stage: build
  image: docker:24.0
  services: [docker:24.0-dind]
  script:
    - docker build
        --label "org.opencontainers.image.revision=${CI_COMMIT_SHA}"
        --label "org.opencontainers.image.version=${DOCKER_TAG}"
        -t ${DOCKER_IMAGE}:${DOCKER_TAG}
        -t ${DOCKER_IMAGE}:latest .
    - docker save ${DOCKER_IMAGE}:${DOCKER_TAG} > image.tar
  artifacts:
    paths: [image.tar]
    expire_in: 1 hour

# ─────────────── STAGE 5: PUSH ───────────────
push_image:
  stage: push
  image: docker:24.0
  services: [docker:24.0-dind]
  before_script:
    - echo ${CI_REGISTRY_PASSWORD} | docker login -u ${CI_REGISTRY_USER} --password-stdin ${CI_REGISTRY}
  script:
    - docker load < image.tar
    - docker push ${DOCKER_IMAGE}:${DOCKER_TAG}
    - docker push ${DOCKER_IMAGE}:latest
  only: [main]

# ─────────────── STAGE 6: STAGING DEPLOY ───────────────
deploy_staging:
  stage: staging_deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: ${STAGING_URL}
  script:
    - kubectl set image deployment/rag-retrieval-service
        rag-retrieval-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/rag-retrieval-service -n rag-staging --timeout=300s
  only: [main]

# ─────────────── STAGE 7: SMOKE TEST ───────────────
smoke_test:
  stage: smoke_test
  image: python:3.11-slim
  before_script:
    - pip install requests
  script:
    - |
      echo "=== Health Check ==="
      python3 -c "
      import requests
      r = requests.get('${STAGING_URL}/health')
      assert r.status_code == 200, f'Health failed: {r.status_code}'
      print('PASSED:', r.json())
      "

    - |
      echo "=== Readiness Check (all dependencies) ==="
      python3 -c "
      import requests
      r = requests.get('${STAGING_URL}/ready')
      assert r.status_code == 200, f'Ready failed: {r.status_code}'
      data = r.json()
      print('PASSED:', data)
      assert data['dependencies']['vector_db'] == True
      assert data['dependencies']['bm25_index'] == True
      "

    - |
      echo "=== Functional: Hybrid Retrieve ==="
      python3 -c "
      import requests, json
      # Use a dummy query
      r = requests.post('${STAGING_URL}/retrieve', json={
          'query': 'How does VPC networking work in EKS?',
          'top_k': 5,
          'use_reranking': True
      })
      assert r.status_code == 200, f'Retrieve failed: {r.status_code}'
      data = r.json()
      assert 'results' in data
      assert data['latency_ms'] < 500, f'Too slow: {data[\"latency_ms\"]}ms'
      print(f'PASSED: {len(data[\"results\"])} results, {data[\"latency_ms\"]}ms, reranked={data[\"reranked\"]}')
      "

    - |
      echo "=== Dense-Only Endpoint ==="
      python3 -c "
      import requests
      r = requests.post('${STAGING_URL}/retrieve/dense', json={
          'query': 'kubernetes pod scheduling', 'top_k': 3
      })
      assert r.status_code == 200
      print('PASSED:', r.json()['method'])
      "

    - |
      echo "=== Sparse-Only Endpoint ==="
      python3 -c "
      import requests
      r = requests.post('${STAGING_URL}/retrieve/sparse', json={
          'query': 'EKS-AUTH-403 error', 'top_k': 3
      })
      assert r.status_code == 200
      print('PASSED:', r.json()['method'])
      "

    - |
      echo "=== Metrics Endpoint ==="
      python3 -c "
      import requests
      r = requests.get('${STAGING_URL}/metrics')
      assert r.status_code == 200
      assert 'rag_retrieve_total' in r.text
      print('PASSED: metrics available')
      "

    - echo "=== ALL SMOKE TESTS PASSED ==="
  only: [main]

# ─────────────── STAGE 7.5: RETRIEVAL QUALITY GATE ───────────────
retrieval_quality:
  stage: retrieval_quality_gate
  image: python:3.11-slim
  before_script:
    - pip install requests numpy
  script:
    - |
      python3 eval/eval_retrieval.py \
        --api-url ${STAGING_URL} \
        --test-set eval/test_set.json \
        --min-recall 0.75 \
        --min-mrr 0.65 \
        --max-latency-p99 250
  artifacts:
    paths:
      - eval/results.json
    when: always
  only: [main]
  allow_failure: false   # BLOCKS pipeline if quality drops

# ─────────────── STAGE 8: APPROVAL GATE ───────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging verified + Retrieval quality passed."
    - echo "Image → ${DOCKER_IMAGE}:${DOCKER_TAG}"
    - echo "Commit → ${CI_COMMIT_SHA}"
    - cat eval/results.json 2>/dev/null || echo "No eval results"
  when: manual             # ← Human clicks to approve
  allow_failure: false
  only: [main]

# ─────────────── STAGE 9: PROD DEPLOY — CANARY 10% ───────────────
deploy_prod_canary:
  stage: prod_deploy_canary
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: ${PROD_URL}
  script:
    - |
      cat <<EOF | kubectl apply -f -
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: rag-retrieval-service-canary
        namespace: rag-production
        labels:
          app: rag-retrieval-service
          track: canary
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: rag-retrieval-service
            track: canary
        template:
          metadata:
            labels:
              app: rag-retrieval-service
              track: canary
            annotations:
              prometheus.io/scrape: "true"
              prometheus.io/port: "8000"
          spec:
            containers:
            - name: rag-retrieval-service
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports:
              - containerPort: 8000
              envFrom:
              - configMapRef:
                  name: rag-retrieval-config
              resources:
                requests: { memory: "2Gi", cpu: "1" }
                limits: { memory: "4Gi", cpu: "2" }
              livenessProbe:
                httpGet: { path: /health, port: 8000 }
                initialDelaySeconds: 10
              readinessProbe:
                httpGet: { path: /ready, port: 8000 }
                initialDelaySeconds: 5
      EOF
    - kubectl rollout status deployment/rag-retrieval-service-canary -n rag-production --timeout=300s
    - echo "Canary deployed — 10% traffic (1/10 pods)"
  only: [main]

# ─────────────── STAGE 10: MONITOR ───────────────
monitor_canary:
  stage: monitor
  image: python:3.11-slim
  before_script:
    - pip install requests
  script:
    - |
      python3 -c "
      import requests, time, sys

      PROD_URL = '${PROD_URL}'
      FAILURES = 0
      CHECKS = 10
      INTERVAL = 30

      print('=== Monitoring canary for 5 minutes ===')
      for i in range(1, CHECKS + 1):
          print(f'--- Check {i}/{CHECKS} ---')

          # Health check
          try:
              r = requests.get(f'{PROD_URL}/health', timeout=10)
              if r.status_code != 200:
                  FAILURES += 1
                  print(f'WARN: Health failed ({r.status_code})')
              else:
                  print('OK: Health passed')
          except Exception as e:
              FAILURES += 1
              print(f'WARN: Health exception ({e})')

          # Latency check
          try:
              start = time.time()
              r = requests.post(f'{PROD_URL}/retrieve', json={
                  'query': 'test monitoring query', 'top_k': 3
              }, timeout=10)
              latency = (time.time() - start) * 1000
              print(f'Latency: {latency:.0f}ms')
              if latency > 300:
                  FAILURES += 1
                  print('WARN: Latency too high')
          except Exception as e:
              FAILURES += 1
              print(f'WARN: Retrieve failed ({e})')

          if FAILURES >= 3:
              print('=== CANARY FAILED: Too many failures ===')
              sys.exit(1)

          if i < CHECKS:
              time.sleep(INTERVAL)

      print(f'=== CANARY HEALTHY ({FAILURES} failures) ===')
      "
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        echo "Auto-rolling back canary..."
        kubectl delete deployment rag-retrieval-service-canary -n rag-production --ignore-not-found
      fi
  only: [main]

# ─────────────── STAGE 11: PROD DEPLOY — FULL 100% ───────────────
deploy_prod_full:
  stage: prod_deploy_full
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: ${PROD_URL}
  script:
    - kubectl set image deployment/rag-retrieval-service
        rag-retrieval-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/rag-retrieval-service -n rag-production --timeout=600s
    - kubectl delete deployment rag-retrieval-service-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100% DEPLOYED ==="
    - echo "External URL: ${PROD_URL}"
  only: [main]
```

---

### Step 8: Retrieval Quality Evaluator (`eval/eval_retrieval.py`)

```python
#!/usr/bin/env python3
"""
Automated retrieval quality evaluation — runs in CI pipeline.
Fails if metrics drop below thresholds (prevents quality regression).
"""
import json
import requests
import numpy as np
import argparse
import time
import sys


def evaluate(api_url: str, test_set_path: str, min_recall: float,
             min_mrr: float, max_latency_p99: float):
    with open(test_set_path) as f:
        test_set = json.load(f)

    recalls = []
    mrrs = []
    latencies = []

    print(f"Evaluating {len(test_set)} queries against {api_url}...")

    for item in test_set:
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])

        start = time.time()
        try:
            resp = requests.post(f"{api_url}/retrieve", json={
                "query": query, "top_k": 5, "use_reranking": True
            }, timeout=30)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            if resp.status_code != 200:
                print(f"  WARN: Query failed ({resp.status_code}): {query[:50]}")
                recalls.append(0.0)
                mrrs.append(0.0)
                continue

            results = resp.json()["results"]
            retrieved_ids = [str(r.get("id", "")) for r in results]

            # Recall@5
            recall = len(set(retrieved_ids) & relevant_ids) / len(relevant_ids) if relevant_ids else 0
            recalls.append(recall)

            # MRR
            mrr = 0.0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)

        except Exception as e:
            print(f"  ERROR: {e}")
            recalls.append(0.0)
            mrrs.append(0.0)
            latencies.append(9999)

    # Calculate metrics
    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mrrs)
    p99_latency = np.percentile(latencies, 99) if latencies else 9999

    results = {
        "recall_at_5": round(float(avg_recall), 4),
        "mrr": round(float(avg_mrr), 4),
        "latency_p99_ms": round(float(p99_latency), 1),
        "queries_evaluated": len(test_set),
        "thresholds": {
            "min_recall": min_recall,
            "min_mrr": min_mrr,
            "max_latency_p99": max_latency_p99
        },
        "passed": True
    }

    # Print results
    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY RESULTS")
    print("=" * 60)
    print(f"  Recall@5:      {avg_recall:.4f}  (threshold: >= {min_recall})")
    print(f"  MRR:           {avg_mrr:.4f}  (threshold: >= {min_mrr})")
    print(f"  Latency p99:   {p99_latency:.0f}ms  (threshold: <= {max_latency_p99}ms)")
    print("=" * 60)

    # Check thresholds
    failures = []
    if avg_recall < min_recall:
        failures.append(f"Recall@5 {avg_recall:.4f} < {min_recall}")
    if avg_mrr < min_mrr:
        failures.append(f"MRR {avg_mrr:.4f} < {min_mrr}")
    if p99_latency > max_latency_p99:
        failures.append(f"Latency p99 {p99_latency:.0f}ms > {max_latency_p99}ms")

    if failures:
        results["passed"] = False
        print("\n❌ QUALITY GATE FAILED:")
        for f in failures:
            print(f"   - {f}")
        with open("eval/results.json", "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)
    else:
        print("\n✅ QUALITY GATE PASSED")
        with open("eval/results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.65)
    parser.add_argument("--max-latency-p99", type=float, default=250)
    args = parser.parse_args()

    evaluate(args.api_url, args.test_set, args.min_recall, args.min_mrr, args.max_latency_p99)
```

### Step 9: Test Set (`eval/test_set.json`)

```json
[
  {
    "query": "How does VPC-CNI networking work in EKS?",
    "relevant_doc_ids": ["doc-vpc-cni-1", "doc-eks-networking-7"]
  },
  {
    "query": "Error EKS-AUTH-403 resolution steps",
    "relevant_doc_ids": ["doc-eks-auth-errors-12"]
  },
  {
    "query": "S3 durability eleven nines",
    "relevant_doc_ids": ["doc-s3-durability-3"]
  },
  {
    "query": "What is the pricing for t3.large instances?",
    "relevant_doc_ids": ["doc-ec2-pricing-5", "doc-ec2-pricing-6"]
  },
  {
    "query": "How to configure auto-scaling in Kubernetes?",
    "relevant_doc_ids": ["doc-k8s-hpa-2", "doc-k8s-scaling-9"]
  }
]
```

---

### Step 10: Kubernetes — RAG Service Deployment (`k8s/deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-retrieval-service
  namespace: rag-production
  labels:
    app: rag-retrieval-service
    track: stable
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: rag-retrieval-service
      track: stable
  template:
    metadata:
      labels:
        app: rag-retrieval-service
        track: stable
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: rag-retrieval-service
        image: registry.gitlab.com/yourgroup/rag-retrieval-service:latest
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: rag-retrieval-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: rag-retrieval-service
```

### Step 11: Service (`k8s/service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-retrieval-service
  namespace: rag-production
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: rag-retrieval-service    # Stable + canary dono get traffic
```

### Step 12: Ingress — External Access (Cluster ke BAHAR se)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-retrieval-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "500"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "5m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "Strict-Transport-Security: max-age=31536000";
spec:
  tls:
  - hosts:
    - rag.yourdomain.com
    secretName: rag-retrieval-tls
  rules:
  - host: rag.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-retrieval-service
            port:
              number: 80
```

**AWS EKS (ALB):**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-retrieval-alb
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-south-1:ACCOUNT:certificate/CERT-ID
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/healthcheck-path: /health
spec:
  rules:
  - host: rag.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-retrieval-service
            port:
              number: 80
```

### Step 13: ConfigMap (`k8s/configmap.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-retrieval-config
  namespace: rag-production
data:
  VECTOR_DB_URL: "http://qdrant.rag-production.svc.cluster.local:6333"
  BM25_URL: "http://elasticsearch.rag-production.svc.cluster.local:9200"
  EMBEDDING_SERVICE_URL: "http://embedding-service.rag-production.svc.cluster.local:8080"
  RERANKER_URL: "http://reranker-service.rag-production.svc.cluster.local:8080"
  REDIS_HOST: "redis.rag-production.svc.cluster.local"
  DEFAULT_COLLECTION: "documents"
  LOG_LEVEL: "INFO"
```

### Step 14: HPA (`k8s/hpa.yaml`)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-retrieval-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-retrieval-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 3
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### Step 15: Network Policy (`k8s/dependencies/networkpolicy.yaml`)

```yaml
# Qdrant — only RAG service can access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qdrant-access
  namespace: rag-production
spec:
  podSelector:
    matchLabels:
      app: qdrant
  policyTypes: [Ingress]
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: rag-retrieval-service
    ports:
    - port: 6333
    - port: 6334
  - from:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - port: 6335
---
# Elasticsearch — only RAG service can access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: elasticsearch-access
  namespace: rag-production
spec:
  podSelector:
    matchLabels:
      app: elasticsearch
  policyTypes: [Ingress]
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: rag-retrieval-service
    ports:
    - port: 9200
---
# Redis — only RAG service can access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: redis-access
  namespace: rag-production
spec:
  podSelector:
    matchLabels:
      app: redis
  policyTypes: [Ingress]
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: rag-retrieval-service
    ports:
    - port: 6379
```

---

### Step 16: ArgoCD Applications

**Staging (auto-sync):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rag-retrieval-staging
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/rag-retrieval-service.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-staging
  syncPolicy:
    automated: { prune: true, selfHeal: true }
    syncOptions: [CreateNamespace=true]
```

**Production (manual sync):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rag-retrieval-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/rag-retrieval-service.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-production
  syncPolicy:
    syncOptions: [CreateNamespace=true, PrunePropagationPolicy=foreground]
```

---

### Step 17: Operations Commands

```bash
# ─── External Access (from ANYWHERE) ───
curl https://rag.yourdomain.com/health
curl https://rag.yourdomain.com/ready
curl https://rag.yourdomain.com/stats

# Hybrid search from anywhere
curl -X POST https://rag.yourdomain.com/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "How does EKS networking work?", "top_k": 5, "use_reranking": true}'

# Dense-only search
curl -X POST https://rag.yourdomain.com/retrieve/dense \
  -d '{"query": "pod scheduling", "top_k": 3}'

# Sparse-only search (BM25)
curl -X POST https://rag.yourdomain.com/retrieve/sparse \
  -d '{"query": "EKS-AUTH-403", "top_k": 3}'

# Prometheus metrics
curl https://rag.yourdomain.com/metrics

# ─── Check Ingress ───
kubectl get ingress -n rag-production
# NAME                     HOSTS                ADDRESS                    PORTS
# rag-retrieval-ingress    rag.yourdomain.com   a1b2c3.elb.amazonaws.com   80,443

# ─── Rollback ───
kubectl rollout undo deployment/rag-retrieval-service -n rag-production

# ─── Scale ───
kubectl scale deployment/rag-retrieval-service --replicas=5 -n rag-production
```

---

### Architecture (Complete):

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE DEPLOYMENT + RETRIEVAL ARCHITECTURE                    │
│                                                                                  │
│  Developer pushes → GitLab CI Pipeline:                                          │
│  TEST → LINT → SECURITY → BUILD → PUSH → STAGING → SMOKE TEST                   │
│       → RETRIEVAL QUALITY GATE (Recall@5, MRR, latency) ← Unique to this service│
│       → APPROVAL → CANARY 10% → MONITOR 5min → FULL 100%                        │
│                                                                                  │
│  ┌────────────────────── K8s Cluster ──────────────────────────┐                 │
│  │                                                              │                │
│  │  Internet → LoadBalancer → Ingress (TLS + rate limit)        │                │
│  │                                  │                           │                │
│  │                          ┌───────┴───────┐                   │                │
│  │                          │ RAG Retrieval │ (3-20 pods, HPA)  │                │
│  │                          │ Service :8000 │                   │                │
│  │                          └───┬───┬───┬───┘                   │                │
│  │                              │   │   │                       │                │
│  │              ┌───────────────┘   │   └───────────────┐       │                │
│  │              ▼                   ▼                   ▼       │                │
│  │      ┌─────────────┐    ┌──────────────┐    ┌─────────┐    │                │
│  │      │   Qdrant    │    │Elasticsearch │    │  Redis  │    │                │
│  │      │  (Dense)    │    │   (BM25)     │    │ (Cache) │    │                │
│  │      │  gRPC:6334  │    │   :9200      │    │  :6379  │    │                │
│  │      │  3 replicas │    │  3 replicas  │    │ 1 node  │    │                │
│  │      └─────────────┘    └──────────────┘    └─────────┘    │                │
│  │           ↑                    ↑                  ↑         │                │
│  │      NetworkPolicy:       NetworkPolicy:     NetworkPolicy: │                │
│  │      Only RAG service     Only RAG service   Only RAG svc   │                │
│  └──────────────────────────────────────────────────────────────┘                │
│                                                                                  │
│  External: https://rag.yourdomain.com                                            │
│  Internal DBs: NEVER exposed externally (NetworkPolicy protected)                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Additions (vs basic Layer 12):

| Feature | Pehle (basic) | Ab (production) |
|---------|---------------|-----------------|
| CI stages | 4 (test, build, push, deploy) | **12** (full pipeline + quality gate) |
| Security | None | Bandit + Trivy + Gitleaks |
| External access | None (localhost) | **Ingress + TLS + rate limit** |
| Canary | None | **10% traffic, 5 min monitoring** |
| Retrieval quality gate | Post-deploy test | **CI stage — blocks pipeline if Recall/MRR drops** |
| Dependencies security | None | **NetworkPolicy on Qdrant, ES, Redis** |
| Auto-scaling | Fixed replicas | **HPA 3→20 pods** |
| Cache | None | **Redis query cache** |
| Monitoring | None | **Prometheus metrics + latency tracking** |
| Rollback | Manual | **Auto-rollback on canary failure** |
