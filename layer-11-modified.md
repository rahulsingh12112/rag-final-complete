## Layer 11: GitLab CI/CD + ArgoCD — Production Chunking Pipeline Deployment

Chunking ek **batch job** hai — real-time nahi. Documents S3 se aate hain, chunk hote hain, embed hote hain, vector DB mein store hote hain. Yeh Kubernetes CronJob ke through daily run hota hai, GitLab CI se image build hota hai, ArgoCD se deploy.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
chunking-pipeline/
├── src/
│   ├── chunker.py              # Chunking logic
│   ├── pipeline.py             # S3 → Chunk → Embed → Vector DB
│   ├── health.py               # Health/readiness/trigger API
│   ├── requirements.txt
│   └── tests/
│       ├── test_chunker.py
│       └── test_pipeline.py
├── Dockerfile
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml         # Long-running API service
│   ├── service.yaml
│   ├── ingress.yaml            # External access (cluster ke BAHAR se)
│   ├── cronjob.yaml            # Scheduled daily processing
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── hpa.yaml                # Auto-scaling
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── scripts/
│   └── smoke-test.sh
├── .gitlab-ci.yml
└── README.md
```

### Step 1: Pipeline Code (`src/pipeline.py`)

```python
import boto3
import hashlib
import os
import uuid
import requests
from chunker import RecursiveChunker
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

S3_BUCKET = os.getenv("S3_BUCKET", "raw-documents")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://qdrant:6333")
EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8080")
COLLECTION = os.getenv("COLLECTION_NAME", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

s3 = boto3.client("s3")
vector_db = QdrantClient(url=VECTOR_DB_URL)
chunker = RecursiveChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


def get_embeddings(texts):
    resp = requests.post(f"{EMBEDDING_URL}/embed", json={"texts": texts, "is_query": False})
    return resp.json()["embeddings"]


def process_document(key, content, metadata):
    doc_hash = hashlib.md5(content.encode()).hexdigest()

    # Delete old chunks
    vector_db.delete(collection_name=COLLECTION,
        points_selector={"filter": {"must": [{"key": "source", "match": {"value": key}}]}})

    # Chunk → Embed → Store
    chunks = chunker.split(content)
    embeddings = get_embeddings(chunks)

    points = [PointStruct(
        id=str(uuid.uuid4()), vector=emb,
        payload={"text": chunk, "source": key, "chunk_index": i, "doc_hash": doc_hash, **metadata}
    ) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]

    vector_db.upsert(collection_name=COLLECTION, points=points)
    print(f"  {key}: {len(points)} chunks stored")
    return len(points)


def run():
    total_docs = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="docs/"):
        for obj in page.get("Contents", []):
            content = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read().decode()
            process_document(obj["Key"], content, {"category": "general"})
            total_docs += 1
    return total_docs

if __name__ == "__main__":
    run()
```

### Step 2: Health/API Service (`src/health.py`)

Cluster ke bahar se access ke liye — health, trigger, status endpoints:

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
import time, os

app = FastAPI(title="Chunking Pipeline API", version="1.0.0")

# Metrics
CHUNKS_PROCESSED = Counter("chunks_processed_total", "Total chunks processed")
PIPELINE_DURATION = Histogram("pipeline_duration_seconds", "Pipeline execution time")
PIPELINE_STATUS = {"status": "idle", "last_run": None, "docs_processed": 0}


@app.get("/health")
def health():
    """Liveness probe"""
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    """Readiness probe — vector DB connected?"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("VECTOR_DB_URL", "http://qdrant:6333"), timeout=5)
        client.get_collections()
        return {"status": "ready", "vector_db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


@app.post("/trigger")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """Manual pipeline trigger — external se call kar sakte ho"""
    if PIPELINE_STATUS["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running")
    background_tasks.add_task(run_pipeline_async)
    return {"message": "Pipeline triggered", "status": "accepted"}


@app.get("/status")
def get_status():
    return PIPELINE_STATUS


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()


async def run_pipeline_async():
    from pipeline import run
    PIPELINE_STATUS["status"] = "running"
    start = time.time()
    try:
        docs_count = run()
        PIPELINE_STATUS["status"] = "completed"
        PIPELINE_STATUS["docs_processed"] = docs_count
        CHUNKS_PROCESSED.inc(docs_count)
    except Exception as e:
        PIPELINE_STATUS["status"] = f"failed: {str(e)}"
    finally:
        PIPELINE_STATUS["last_run"] = time.time()
        PIPELINE_DURATION.observe(time.time() - start)
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
CMD ["gunicorn", "health:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "300", "--access-logfile", "-"]
```

### Step 4: GitLab CI/CD — Full Production Pipeline (`.gitlab-ci.yml`)

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
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/chunking-pipeline
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://chunking-staging.yourdomain.com"
  PROD_URL: "https://chunking.yourdomain.com"

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
    - name: qdrant/qdrant:latest
      alias: qdrant
  variables:
    VECTOR_DB_URL: "http://qdrant:6333"
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
    - kubeval k8s/*.yaml --strict

# ─────────────── STAGE 3: SECURITY SCAN ───────────────
sast:
  stage: security_scan
  image: python:3.11-slim
  before_script:
    - pip install bandit safety
  script:
    - bandit -r src/ -ll    # Fail on HIGH/CRITICAL
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
    - kubectl set image deployment/chunking-pipeline
        chunking-pipeline=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/chunking-pipeline -n rag-staging --timeout=300s
  only: [main]

# ─────────────── STAGE 7: SMOKE TEST ───────────────
smoke_test:
  stage: smoke_test
  image: curlimages/curl:latest
  script:
    - |
      echo "=== Health Check ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${STAGING_URL}/health)
      [ "$HTTP_CODE" = "200" ] || exit 1
      echo "PASSED"

    - |
      echo "=== Readiness Check ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${STAGING_URL}/ready)
      [ "$HTTP_CODE" = "200" ] || exit 1
      echo "PASSED"

    - |
      echo "=== Functional: Trigger Pipeline ==="
      RESPONSE=$(curl -s -X POST ${STAGING_URL}/trigger)
      echo "$RESPONSE" | grep -q "accepted" || exit 1
      echo "PASSED"

    - |
      echo "=== Metrics Endpoint ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${STAGING_URL}/metrics)
      [ "$HTTP_CODE" = "200" ] || exit 1
      echo "PASSED"

    - echo "=== ALL SMOKE TESTS PASSED ==="
  only: [main]

# ─────────────── STAGE 8: APPROVAL GATE ───────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging verified. Awaiting manual approval for production."
    - echo "Image → ${DOCKER_IMAGE}:${DOCKER_TAG}"
  when: manual          # ← Human clicks to approve
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
        name: chunking-pipeline-canary
        namespace: rag-production
        labels:
          app: chunking-pipeline
          track: canary
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: chunking-pipeline
            track: canary
        template:
          metadata:
            labels:
              app: chunking-pipeline
              track: canary
          spec:
            containers:
            - name: chunking-pipeline
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports:
              - containerPort: 8000
              envFrom:
              - configMapRef:
                  name: chunking-config
              resources:
                requests: { memory: "2Gi", cpu: "1" }
                limits: { memory: "4Gi", cpu: "2" }
              livenessProbe:
                httpGet: { path: /health, port: 8000 }
                initialDelaySeconds: 15
              readinessProbe:
                httpGet: { path: /ready, port: 8000 }
                initialDelaySeconds: 10
      EOF
    - kubectl rollout status deployment/chunking-pipeline-canary -n rag-production --timeout=300s
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
        echo "--- Check $i/10 ---"
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${PROD_URL}/health)
        if [ "$HTTP_CODE" != "200" ]; then
          FAILURES=$((FAILURES + 1))
          echo "WARN: Health check failed"
        else
          echo "OK"
        fi
        [ $FAILURES -ge 3 ] && echo "CANARY FAILED" && exit 1
        sleep 30
      done
      echo "=== CANARY HEALTHY ==="
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        echo "Rolling back canary..."
        kubectl delete deployment chunking-pipeline-canary -n rag-production --ignore-not-found
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
    - kubectl set image deployment/chunking-pipeline
        chunking-pipeline=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/chunking-pipeline -n rag-production --timeout=600s
    - kubectl delete deployment chunking-pipeline-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100% DEPLOYED ==="
    - echo "URL: ${PROD_URL}"
  only: [main]
```

### Step 5: Kubernetes Deployment (`k8s/deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chunking-pipeline
  namespace: rag-production
  labels:
    app: chunking-pipeline
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
      app: chunking-pipeline
      track: stable
  template:
    metadata:
      labels:
        app: chunking-pipeline
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
      - name: chunking-pipeline
        image: registry.gitlab.com/yourgroup/chunking-pipeline:latest
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: chunking-config
        - secretRef:
            name: chunking-secrets
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
          initialDelaySeconds: 15
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Step 6: Service (`k8s/service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: chunking-pipeline
  namespace: rag-production
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: chunking-pipeline      # Stable + canary dono ko traffic milti hai
```

### Step 7: Ingress — External Access (Cluster ke BAHAR se)

**Yeh sabse important part hai — localhost nahi, proper external URL:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chunking-pipeline-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "Strict-Transport-Security: max-age=31536000";
spec:
  tls:
  - hosts:
    - chunking.yourdomain.com
    secretName: chunking-tls-cert
  rules:
  - host: chunking.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: chunking-pipeline
            port:
              number: 80
```

**AWS EKS pe ho toh ALB Ingress use karo:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chunking-pipeline-alb
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing      # ← EXTERNAL
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-south-1:ACCOUNT:certificate/CERT-ID
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/healthcheck-path: /health
spec:
  rules:
  - host: chunking.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: chunking-pipeline
            port:
              number: 80
```

### Step 8: CronJob — Daily Chunking (`k8s/cronjob.yaml`)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: chunking-scheduled
  namespace: rag-production
spec:
  schedule: "0 2 * * *"            # Daily 2 AM
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 5
  jobTemplate:
    spec:
      backoffLimit: 3
      activeDeadlineSeconds: 3600   # Max 1 hour
      template:
        spec:
          containers:
          - name: chunker-batch
            image: registry.gitlab.com/yourgroup/chunking-pipeline:latest
            command: ["python", "pipeline.py"]
            envFrom:
            - configMapRef:
                name: chunking-config
            resources:
              requests: { memory: "4Gi", cpu: "2" }
              limits: { memory: "8Gi", cpu: "4" }
          restartPolicy: OnFailure
```

### Step 9: ConfigMap (`k8s/configmap.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chunking-config
  namespace: rag-production
data:
  S3_BUCKET: "raw-documents-prod"
  VECTOR_DB_URL: "http://qdrant.rag-production.svc.cluster.local:6333"
  EMBEDDING_SERVICE_URL: "http://embedding-service.rag-production.svc.cluster.local:8080"
  COLLECTION_NAME: "documents_prod"
  CHUNK_SIZE: "512"
  CHUNK_OVERLAP: "50"
  LOG_LEVEL: "INFO"
```

### Step 10: HPA — Auto Scaling (`k8s/hpa.yaml`)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chunking-pipeline-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chunking-pipeline
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Step 11: ArgoCD Applications

**Staging (auto-sync):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: chunking-pipeline-staging
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/chunking-pipeline.git
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
  name: chunking-pipeline-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/chunking-pipeline.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-production
  syncPolicy:
    syncOptions: [CreateNamespace=true, PrunePropagationPolicy=foreground]
    # NO automated — manual sync only for prod safety
```

### Step 12: One-Time Cluster Setup (External Access Enable)

```bash
# 1. Ingress Controller (NGINX)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.service.type=LoadBalancer

# 2. Cert-Manager (auto TLS)
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true

# 3. Let's Encrypt ClusterIssuer
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

# 4. DNS point karo: chunking.yourdomain.com → Ingress External IP
kubectl get ingress -n rag-production
# Output me ADDRESS column me external IP/hostname milega

# 5. Deploy
kubectl apply -f k8s/
```

### Step 13: Commands (Operations)

```bash
# External se access test (from anywhere — not cluster)
curl https://chunking.yourdomain.com/health
curl https://chunking.yourdomain.com/status
curl -X POST https://chunking.yourdomain.com/trigger

# Check external IP
kubectl get ingress -n rag-production
# NAME                          HOSTS                     ADDRESS                          PORTS
# chunking-pipeline-ingress     chunking.yourdomain.com   a1b2c3.elb.amazonaws.com         80,443

# Manual CronJob trigger
kubectl create job --from=cronjob/chunking-scheduled manual-run -n rag-production
kubectl logs -f job/manual-run -n rag-production

# Rollback (agar kuch galat ho)
kubectl rollout undo deployment/chunking-pipeline -n rag-production
```

### Flow (Complete):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE DEPLOYMENT FLOW                               │
│                                                                             │
│  Developer pushes to main                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  TEST (pytest + coverage) → LINT (ruff + hadolint + kubeval)                │
│       │                                                                     │
│       ▼                                                                     │
│  SECURITY SCAN (bandit + trivy + gitleaks)                                  │
│       │                                                                     │
│       ▼                                                                     │
│  BUILD (docker multi-stage) → PUSH (gitlab registry)                        │
│       │                                                                     │
│       ▼                                                                     │
│  STAGING DEPLOY → SMOKE TEST (health + ready + trigger + metrics)           │
│       │                                                                     │
│       ▼                                                                     │
│  ⏸️  APPROVAL GATE (human clicks approve)                                   │
│       │                                                                     │
│       ▼                                                                     │
│  CANARY 10% (1 pod new version) → MONITOR (5 min, 10 health checks)        │
│       │                                                                     │
│       ▼ (if healthy)                                                        │
│  FULL DEPLOY 100% (rolling update) → canary deleted                         │
│                                                                             │
│  External: https://chunking.yourdomain.com ← accessible from ANYWHERE      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### External Access — Localhost vs Production:

```
❌ WRONG (localhost — sirf dev machine pe):
   curl http://localhost:8000/health
   → Cluster ke bahar se kaam NAHI karega

✅ RIGHT (Ingress + LoadBalancer + DNS — production):
   curl https://chunking.yourdomain.com/health
   → Duniya mein kahin se bhi accessible
   → TLS encrypted
   → Rate limited
   → Auto-scaled
```

**Kaise kaam karta hai:**
```
Client (Internet) → DNS → Cloud LoadBalancer → Ingress Controller → K8s Service → Pods
```
