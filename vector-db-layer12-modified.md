## Layer 12: GitLab CI/CD + ArgoCD — Vector DB + RAG API Production Deployment

Vector DB (Qdrant) infra hai — uspe ek **RAG API service** baith-ta hai jo queries handle karta hai. Dono ko production mein deploy karna hai with full pipeline.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
vector-db-rag-service/
├── src/
│   ├── app.py                  # FastAPI RAG service (query → vector search → response)
│   ├── vector_store.py         # Qdrant client wrapper (production code from Layer 4)
│   ├── config.py               # Environment config
│   ├── requirements.txt
│   └── tests/
│       ├── test_app.py
│       ├── test_vector_store.py
│       └── test_integration.py
├── Dockerfile
├── k8s/
│   ├── qdrant/
│   │   ├── statefulset.yaml    # Qdrant cluster (3 nodes HA)
│   │   ├── service.yaml        # ClusterIP (internal only)
│   │   ├── configmap.yaml
│   │   ├── pvc.yaml
│   │   ├── networkpolicy.yaml  # Only RAG service can access
│   │   └── backup-cronjob.yaml # Daily S3 snapshot
│   ├── rag-service/
│   │   ├── deployment.yaml     # RAG API pods
│   │   ├── service.yaml
│   │   ├── ingress.yaml        # External access (cluster ke BAHAR se)
│   │   ├── configmap.yaml
│   │   ├── hpa.yaml            # Auto-scaling
│   │   └── pdb.yaml            # Pod Disruption Budget
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── scripts/
│   └── smoke-test.sh
├── .gitlab-ci.yml
└── README.md
```

### Step 1: RAG API Service (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from vector_store import ProductionVectorDB
from config import settings
import time
import os

app = FastAPI(title="RAG Vector Search API", version="1.0.0")

# Metrics
SEARCH_COUNTER = Counter("rag_searches_total", "Total search requests", ["status"])
SEARCH_LATENCY = Histogram("rag_search_duration_seconds", "Search latency",
                           buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
VECTORS_TOTAL = Counter("rag_vectors_upserted_total", "Total vectors upserted")

# Init vector DB client
db = ProductionVectorDB(url=settings.VECTOR_DB_URL, grpc_port=settings.VECTOR_DB_GRPC_PORT)


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int = 5
    filters: dict | None = None
    collection: str = "documents"


class SearchResponse(BaseModel):
    results: list[dict]
    latency_ms: float
    total_results: int


class UpsertRequest(BaseModel):
    texts: list[str]
    embeddings: list[list[float]]
    metadata: list[dict]
    collection: str = "documents"


@app.get("/health")
def health():
    """Liveness probe"""
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    """Readiness probe — Qdrant connected?"""
    try:
        info = db.get_collection_info(settings.DEFAULT_COLLECTION)
        return {
            "status": "ready",
            "vector_db": "connected",
            "vectors_count": info["vectors_count"],
            "indexed": info["indexed_vectors"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """Similarity search — core RAG endpoint"""
    start = time.time()
    try:
        results = db.search(
            collection=request.collection,
            query_vector=request.query_vector,
            top_k=request.top_k,
            filters=request.filters
        )
        latency = (time.time() - start) * 1000
        SEARCH_COUNTER.labels(status="success").inc()
        SEARCH_LATENCY.observe(time.time() - start)
        return SearchResponse(results=results, latency_ms=round(latency, 2), total_results=len(results))
    except Exception as e:
        SEARCH_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upsert")
def upsert(request: UpsertRequest):
    """Batch upsert vectors"""
    try:
        db.upsert_batch(
            collection=request.collection,
            texts=request.texts,
            embeddings=request.embeddings,
            metadata=request.metadata
        )
        VECTORS_TOTAL.inc(len(request.texts))
        return {"status": "success", "vectors_upserted": len(request.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete")
def delete_by_source(collection: str = "documents", source: str = Query(...)):
    """Delete vectors by source document"""
    db.delete_by_filter(collection, "source", source)
    return {"status": "deleted", "source": source}


@app.get("/collections/{collection_name}/info")
def collection_info(collection_name: str):
    """Collection stats for monitoring"""
    try:
        return db.get_collection_info(collection_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics"""
    return generate_latest()
```

### Step 2: Config (`src/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    VECTOR_DB_URL: str = "http://qdrant:6333"
    VECTOR_DB_GRPC_PORT: int = 6334
    DEFAULT_COLLECTION: str = "documents"
    APP_VERSION: str = "unknown"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
```

### Step 3: Dockerfile (Production Multi-Stage)

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

### Step 4: Requirements (`src/requirements.txt`)

```txt
fastapi==0.115.0
gunicorn==22.0.0
uvicorn==0.30.0
pydantic-settings==2.5.0
qdrant-client==1.12.0
grpcio==1.66.0
prometheus-client==0.21.0
numpy==2.1.0
httpx==0.27.0
```

---

### Step 5: Full Production `.gitlab-ci.yml`

```yaml
# =============================================================================
# PRODUCTION PIPELINE:
# test → lint → security scan → build → push → staging deploy →
# smoke test → approval gate → prod deploy (canary 10%) → monitor →
# prod deploy (100%)
# =============================================================================

stages:
  - test
  - lint
  - security_scan
  - build
  - push
  - staging_deploy
  - smoke_test
  - approval_gate
  - prod_deploy_canary
  - monitor
  - prod_deploy_full

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-vector-service
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
    - pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
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
  variables:
    VECTOR_DB_URL: "http://qdrant:6333"
    VECTOR_DB_GRPC_PORT: "6334"
  before_script:
    - pip install -r src/requirements.txt pytest httpx
  script:
    - pytest tests/test_integration.py -v --timeout=60

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
    - kubeval k8s/rag-service/*.yaml --strict
    - kubeval k8s/qdrant/*.yaml --strict

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
    - kubectl set image deployment/rag-vector-service
        rag-vector-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/rag-vector-service -n rag-staging --timeout=300s
  only: [main]

# ─────────────── STAGE 7: SMOKE TEST ───────────────
smoke_test:
  stage: smoke_test
  image: curlimages/curl:latest
  script:
    - |
      echo "=== Health Check ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${STAGING_URL}/health)
      [ "$HTTP_CODE" = "200" ] || (echo "FAILED: $HTTP_CODE" && exit 1)
      echo "PASSED"

    - |
      echo "=== Readiness Check (Qdrant connected?) ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${STAGING_URL}/ready)
      [ "$HTTP_CODE" = "200" ] || (echo "FAILED: $HTTP_CODE" && exit 1)
      echo "PASSED"

    - |
      echo "=== Functional: Search Endpoint ==="
      # Create dummy vector (1024d zeros) for smoke test
      VECTOR=$(python3 -c "import json; print(json.dumps([0.1]*1024))")
      RESPONSE=$(curl -s -X POST ${STAGING_URL}/search \
        -H "Content-Type: application/json" \
        -d "{\"query_vector\": ${VECTOR}, \"top_k\": 3}")
      echo "Response: $RESPONSE"
      echo "$RESPONSE" | grep -q "results" || (echo "FAILED: No results field" && exit 1)
      echo "PASSED"

    - |
      echo "=== Collection Info ==="
      RESPONSE=$(curl -s ${STAGING_URL}/collections/documents/info)
      echo "Collection: $RESPONSE"
      echo "$RESPONSE" | grep -q "vectors_count" || (echo "FAILED" && exit 1)
      echo "PASSED"

    - |
      echo "=== Metrics Endpoint ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${STAGING_URL}/metrics)
      [ "$HTTP_CODE" = "200" ] || (echo "FAILED: $HTTP_CODE" && exit 1)
      echo "PASSED"

    - echo "=== ALL SMOKE TESTS PASSED ==="
  only: [main]

# ─────────────── STAGE 8: APPROVAL GATE ───────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging verified. Awaiting manual approval for production."
    - echo "Image → ${DOCKER_IMAGE}:${DOCKER_TAG}"
    - echo "Commit → ${CI_COMMIT_SHA}"
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
        name: rag-vector-service-canary
        namespace: rag-production
        labels:
          app: rag-vector-service
          track: canary
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: rag-vector-service
            track: canary
        template:
          metadata:
            labels:
              app: rag-vector-service
              track: canary
            annotations:
              prometheus.io/scrape: "true"
              prometheus.io/port: "8000"
          spec:
            containers:
            - name: rag-vector-service
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports:
              - containerPort: 8000
              envFrom:
              - configMapRef:
                  name: rag-service-config
              resources:
                requests: { memory: "1Gi", cpu: "500m" }
                limits: { memory: "2Gi", cpu: "1" }
              livenessProbe:
                httpGet: { path: /health, port: 8000 }
                initialDelaySeconds: 10
              readinessProbe:
                httpGet: { path: /ready, port: 8000 }
                initialDelaySeconds: 5
      EOF
    - kubectl rollout status deployment/rag-vector-service-canary -n rag-production --timeout=300s
    - echo "Canary deployed — 10% traffic (1/10 pods)"
  only: [main]

# ─────────────── STAGE 10: MONITOR ───────────────
monitor_canary:
  stage: monitor
  image: curlimages/curl:latest
  script:
    - |
      echo "=== Monitoring canary for 5 minutes ==="
      FAILURES=0
      for i in $(seq 1 10); do
        echo "--- Check $i/10 (every 30s) ---"
        
        # Health check
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${PROD_URL}/health)
        if [ "$HTTP_CODE" != "200" ]; then
          FAILURES=$((FAILURES + 1))
          echo "WARN: Health check failed ($HTTP_CODE)"
        else
          echo "OK: Health check passed"
        fi

        # Search latency check (should be < 100ms)
        VECTOR=$(python3 -c "import json; print(json.dumps([0.1]*1024))")
        START=$(date +%s%N)
        curl -s -X POST ${PROD_URL}/search \
          -H "Content-Type: application/json" \
          -d "{\"query_vector\": ${VECTOR}, \"top_k\": 3}" > /dev/null
        END=$(date +%s%N)
        LATENCY=$(( (END - START) / 1000000 ))
        echo "Search latency: ${LATENCY}ms"
        
        if [ $LATENCY -gt 200 ]; then
          FAILURES=$((FAILURES + 1))
          echo "WARN: Latency too high"
        fi

        [ $FAILURES -ge 3 ] && echo "=== CANARY FAILED ===" && exit 1
        sleep 30
      done
      echo "=== CANARY HEALTHY ==="
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        echo "Auto-rolling back canary..."
        kubectl delete deployment rag-vector-service-canary -n rag-production --ignore-not-found
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
    - kubectl set image deployment/rag-vector-service
        rag-vector-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/rag-vector-service -n rag-production --timeout=600s
    - kubectl delete deployment rag-vector-service-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100% DEPLOYED ==="
    - echo "External URL: ${PROD_URL}"
  only: [main]
```

---

### Step 6: Qdrant StatefulSet (`k8s/qdrant/statefulset.yaml`)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: rag-production
spec:
  serviceName: qdrant-headless
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "6333"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.12.0
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        - containerPort: 6335
          name: internal
        env:
        - name: QDRANT__CLUSTER__ENABLED
          value: "true"
        - name: QDRANT__SERVICE__GRPC_PORT
          value: "6334"
        - name: QDRANT__STORAGE__PERFORMANCE__MEMMAP_THRESHOLD_KB
          value: "20000"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 6333
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /readyz
            port: 6333
          initialDelaySeconds: 5
          periodSeconds: 10
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: gp3
      resources:
        requests:
          storage: 100Gi
```

### Step 7: Qdrant Service (`k8s/qdrant/service.yaml`)

```yaml
# Headless service — StatefulSet internal DNS
apiVersion: v1
kind: Service
metadata:
  name: qdrant-headless
  namespace: rag-production
spec:
  clusterIP: None
  selector:
    app: qdrant
  ports:
  - port: 6333
    name: http
  - port: 6334
    name: grpc
  - port: 6335
    name: internal
---
# Regular service — client access
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: rag-production
spec:
  type: ClusterIP          # Internal only — NEVER expose Qdrant externally
  selector:
    app: qdrant
  ports:
  - port: 6333
    name: http
    targetPort: 6333
  - port: 6334
    name: grpc
    targetPort: 6334
```

### Step 8: Network Policy (`k8s/qdrant/networkpolicy.yaml`)

```yaml
# Only RAG service pods can talk to Qdrant — everything else blocked
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
          app: rag-vector-service
    ports:
    - port: 6333
    - port: 6334
  # Allow cluster internal communication between Qdrant nodes
  - from:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - port: 6335
```

### Step 9: Qdrant Backup CronJob (`k8s/qdrant/backup-cronjob.yaml`)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qdrant-backup
  namespace: rag-production
spec:
  schedule: "0 3 * * *"         # Daily 3 AM
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: curlimages/curl:latest
            command:
            - /bin/sh
            - -c
            - |
              echo "Creating Qdrant snapshot..."
              SNAPSHOT=$(curl -s -X POST http://qdrant:6333/collections/documents/snapshots)
              echo "Snapshot: $SNAPSHOT"
              
              # Extract snapshot filename
              FILENAME=$(echo $SNAPSHOT | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
              
              # Download and upload to S3
              curl -s http://qdrant:6333/collections/documents/snapshots/$FILENAME \
                -o /tmp/$FILENAME
              
              # Upload to S3 (using AWS CLI)
              aws s3 cp /tmp/$FILENAME s3://qdrant-backups/$(date +%Y-%m-%d)/$FILENAME
              
              echo "Backup complete: s3://qdrant-backups/$(date +%Y-%m-%d)/$FILENAME"
          restartPolicy: OnFailure
```

---

### Step 10: RAG Service Deployment (`k8s/rag-service/deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-vector-service
  namespace: rag-production
  labels:
    app: rag-vector-service
    track: stable
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0       # Zero-downtime
  selector:
    matchLabels:
      app: rag-vector-service
      track: stable
  template:
    metadata:
      labels:
        app: rag-vector-service
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
      - name: rag-vector-service
        image: registry.gitlab.com/yourgroup/rag-vector-service:latest
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: rag-service-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
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
            app: rag-vector-service
```

### Step 11: RAG Service ConfigMap (`k8s/rag-service/configmap.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-service-config
  namespace: rag-production
data:
  VECTOR_DB_URL: "http://qdrant.rag-production.svc.cluster.local:6333"
  VECTOR_DB_GRPC_PORT: "6334"
  DEFAULT_COLLECTION: "documents"
  LOG_LEVEL: "INFO"
  APP_VERSION: "latest"
```

### Step 12: Service (`k8s/rag-service/service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-vector-service
  namespace: rag-production
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: rag-vector-service    # Stable + canary dono ko traffic
```

### Step 13: Ingress — External Access (Cluster ke BAHAR se)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-vector-service-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "200"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "Strict-Transport-Security: max-age=31536000";
spec:
  tls:
  - hosts:
    - rag.yourdomain.com
    secretName: rag-tls-cert
  rules:
  - host: rag.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-vector-service
            port:
              number: 80
```

**AWS EKS pe ALB use karo:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-vector-service-alb
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing       # ← EXTERNAL
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-south-1:ACCOUNT:certificate/CERT-ID
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "15"
spec:
  rules:
  - host: rag.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-vector-service
            port:
              number: 80
```

### Step 14: HPA — Auto Scaling (`k8s/rag-service/hpa.yaml`)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-vector-service-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-vector-service
  minReplicas: 3
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
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

### Step 15: Pod Disruption Budget (`k8s/rag-service/pdb.yaml`)

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rag-vector-service-pdb
  namespace: rag-production
spec:
  minAvailable: 2                  # At least 2 pods always running
  selector:
    matchLabels:
      app: rag-vector-service
```

---

### Step 16: ArgoCD Applications

**Staging (auto-sync):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rag-vector-service-staging
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/vector-db-rag-service.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-staging
  syncPolicy:
    automated: { prune: true, selfHeal: true }
    syncOptions: [CreateNamespace=true]
```

**Production (manual sync — safety):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rag-vector-service-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/vector-db-rag-service.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-production
  syncPolicy:
    syncOptions: [CreateNamespace=true, PrunePropagationPolicy=foreground]
    # NO automated — manual only for prod
```

---

### Step 17: One-Time Cluster Setup

```bash
# 1. Ingress Controller (if not already installed)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.service.type=LoadBalancer

# 2. Cert-Manager (auto TLS)
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace --set installCRDs=true

# 3. Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@company.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# 4. Deploy everything
kubectl apply -f k8s/qdrant/
kubectl apply -f k8s/rag-service/

# 5. DNS: rag.yourdomain.com → Ingress external IP
kubectl get ingress -n rag-production
```

### Step 18: Operations Commands

```bash
# ─── External Access (from ANYWHERE, not localhost) ───
curl https://rag.yourdomain.com/health
curl https://rag.yourdomain.com/ready
curl https://rag.yourdomain.com/collections/documents/info

# Search (from anywhere)
curl -X POST https://rag.yourdomain.com/search \
  -H "Content-Type: application/json" \
  -d '{"query_vector": [0.1, 0.2, ...], "top_k": 5}'

# ─── Check Ingress External IP ───
kubectl get ingress -n rag-production
# NAME                          HOSTS                ADDRESS                     PORTS
# rag-vector-service-ingress    rag.yourdomain.com   a1b2c3.elb.amazonaws.com    80,443

# ─── Qdrant Cluster Health (internal) ───
kubectl exec -it qdrant-0 -n rag-production -- \
  curl -s http://localhost:6333/cluster | python3 -m json.tool

# ─── Rollback ───
kubectl rollout undo deployment/rag-vector-service -n rag-production

# ─── Force Qdrant Snapshot ───
kubectl exec -it qdrant-0 -n rag-production -- \
  curl -X POST http://localhost:6333/collections/documents/snapshots

# ─── Scale RAG service ───
kubectl scale deployment/rag-vector-service --replicas=5 -n rag-production
```

---

### Architecture Flow (Complete):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE DEPLOYMENT ARCHITECTURE                            │
│                                                                                 │
│  Developer pushes to main                                                       │
│       │                                                                         │
│       ▼                                                                         │
│  TEST (pytest) → LINT (ruff+hadolint+kubeval) → SECURITY (bandit+trivy+gitleaks)│
│       │                                                                         │
│       ▼                                                                         │
│  BUILD (multi-stage docker) → PUSH (gitlab registry)                            │
│       │                                                                         │
│       ▼                                                                         │
│  STAGING DEPLOY → SMOKE TEST (health+ready+search+metrics)                      │
│       │                                                                         │
│       ▼                                                                         │
│  ⏸️  APPROVAL GATE (human approves)                                              │
│       │                                                                         │
│       ▼                                                                         │
│  CANARY 10% → MONITOR 5 min (health + latency < 200ms) → FULL 100%             │
│                                                                                 │
│                                                                                 │
│  ┌─────────────────── K8s Cluster ───────────────────────┐                      │
│  │                                                        │                     │
│  │  Internet → LoadBalancer → Ingress → Service           │                     │
│  │                                         │              │                     │
│  │                                    ┌────┴────┐         │                     │
│  │                                    │ RAG API │ (3 pods)│                     │
│  │                                    │ :8000   │         │                     │
│  │                                    └────┬────┘         │                     │
│  │                                         │ gRPC:6334    │                     │
│  │                                    ┌────┴────┐         │                     │
│  │                                    │ Qdrant  │ (3 node │                     │
│  │                                    │ Cluster │  HA)    │                     │
│  │                                    └─────────┘         │                     │
│  │                                         │              │                     │
│  │                                    [100Gi PVC each]    │                     │
│  └────────────────────────────────────────────────────────┘                     │
│                                                                                 │
│  External: https://rag.yourdomain.com ← accessible from ANYWHERE               │
│  Qdrant: Internal only (NetworkPolicy blocks external)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions:

| Decision | Choice | Why |
|----------|--------|-----|
| Qdrant external access | **NO — Internal only** | Security — vector DB should never be public |
| RAG API external access | **YES — Ingress + TLS** | Clients need to query from outside |
| Qdrant ↔ RAG communication | **gRPC (port 6334)** | 3-5x faster than REST for vectors |
| Canary traffic split | **Pod-based (1/10)** | Simple, no service mesh needed |
| Qdrant HA | **StatefulSet, 3 replicas** | Data persistence + ordered scaling |
| RAG API HA | **Deployment, 3-15 replicas (HPA)** | Stateless, auto-scale on load |
| Backup | **Daily CronJob → S3 snapshot** | Disaster recovery, avoid re-embedding |
| Network isolation | **NetworkPolicy** | Only RAG pods can reach Qdrant |
