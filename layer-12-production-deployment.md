## Layer 12: Production-Grade CI/CD Pipeline + External Access (Real Dev Workflow)

> **Yeh section existing Layer 11 ke baad add hoga.** Layer 11 mein basic GitLab CI tha — yahan full production pipeline hai jaise real companies deploy karti hain.

### Pipeline Flow (Exact Production Sequence):

```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Updated Project Structure:

```
chunking-pipeline/
├── src/
│   ├── chunker.py
│   ├── pipeline.py
│   ├── health.py              # Health/readiness endpoints
│   ├── requirements.txt
│   └── tests/
│       ├── test_chunker.py
│       └── test_pipeline.py
├── Dockerfile
├── k8s/
│   ├── base/
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml      # Long-running service (API + CronJob trigger)
│   │   ├── service.yaml
│   │   ├── ingress.yaml         # External access (cluster ke bahar se)
│   │   ├── cronjob.yaml
│   │   ├── configmap.yaml
│   │   └── hpa.yaml
│   ├── overlays/
│   │   ├── staging/
│   │   │   ├── kustomization.yaml
│   │   │   └── patches/
│   │   │       └── replicas.yaml
│   │   └── production/
│   │       ├── kustomization.yaml
│   │       └── patches/
│   │           ├── replicas.yaml
│   │           └── resources.yaml
│   └── canary/
│       └── canary-deployment.yaml
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── scripts/
│   ├── smoke-test.sh
│   └── canary-check.sh
├── .gitlab-ci.yml              # Full production pipeline
├── .trivyignore
├── sonar-project.properties
└── README.md
```

---

### Step 1: Application Code with Health Endpoints (`src/health.py`)

Cluster ke bahar se access ke liye ek lightweight **FastAPI service** banaenge jo:
- Health check expose kare (K8s probes ke liye)
- Manual trigger API de (CronJob ke alawa bhi run kar sako)
- Status/metrics endpoint de

```python
# src/health.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
import time
import os

app = FastAPI(title="Chunking Pipeline API", version="1.0.0")

# Metrics
CHUNKS_PROCESSED = Counter("chunks_processed_total", "Total chunks processed")
PIPELINE_DURATION = Histogram("pipeline_duration_seconds", "Pipeline execution time")
PIPELINE_STATUS = {"status": "idle", "last_run": None, "docs_processed": 0}


@app.get("/health")
def health():
    """Liveness probe — service alive hai?"""
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    """Readiness probe — traffic accept karne ke liye ready hai?"""
    # Check dependencies
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
    """Pipeline ka current status"""
    return PIPELINE_STATUS


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


async def run_pipeline_async():
    """Background task mein pipeline run karo"""
    from pipeline import run
    PIPELINE_STATUS["status"] = "running"
    start = time.time()
    try:
        docs_count = run()
        PIPELINE_STATUS["status"] = "completed"
        PIPELINE_STATUS["docs_processed"] = docs_count
    except Exception as e:
        PIPELINE_STATUS["status"] = f"failed: {str(e)}"
    finally:
        PIPELINE_STATUS["last_run"] = time.time()
        PIPELINE_DURATION.observe(time.time() - start)
```

### Step 2: Dockerfile (Production-Grade Multi-Stage)

```dockerfile
# ===== Stage 1: Builder =====
FROM python:3.11-slim AS builder

WORKDIR /build
COPY src/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== Stage 2: Production =====
FROM python:3.11-slim AS production

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ .

# No root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Gunicorn for production (not uvicorn dev server)
CMD ["gunicorn", "health:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "300", "--access-logfile", "-"]
```

### Step 3: Requirements (`src/requirements.txt`)

```txt
fastapi==0.115.0
gunicorn==22.0.0
uvicorn==0.30.0
boto3==1.35.0
qdrant-client==1.12.0
langchain==0.3.0
langchain-text-splitters==0.3.0
tiktoken==0.8.0
requests==2.32.0
prometheus-client==0.21.0
nltk==3.9.0
```

---

### Step 4: Full Production `.gitlab-ci.yml`

```yaml
# =============================================================================
# PRODUCTION CI/CD PIPELINE
# Flow: test → lint → security scan → build → push → staging deploy →
#       smoke test → approval gate → prod deploy (canary 10%) → monitor →
#       prod deploy (100%)
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
  KUBE_CONTEXT_STAGING: "yourgroup/project:staging-agent"
  KUBE_CONTEXT_PROD: "yourgroup/project:prod-agent"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: TEST
# ─────────────────────────────────────────────────────────────────────────────
unit_tests:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install -r src/requirements.txt pytest pytest-cov pytest-asyncio httpx
  script:
    - pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
    - echo "Coverage report generated"
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    when: always
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

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: LINT
# ─────────────────────────────────────────────────────────────────────────────
lint_python:
  stage: lint
  image: python:3.11-slim
  before_script:
    - pip install ruff mypy
  script:
    - ruff check src/ --output-format=gitlab
    - ruff format src/ --check
    - mypy src/ --ignore-missing-imports --no-error-summary || true
  allow_failure: false

lint_dockerfile:
  stage: lint
  image: hadolint/hadolint:latest-debian
  script:
    - hadolint Dockerfile --failure-threshold warning

lint_kubernetes:
  stage: lint
  image: garethr/kubeval:latest
  script:
    - kubeval k8s/base/*.yaml --strict

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: SECURITY SCAN
# ─────────────────────────────────────────────────────────────────────────────
sast_scan:
  stage: security_scan
  image: python:3.11-slim
  before_script:
    - pip install bandit safety
  script:
    # Static Application Security Testing
    - bandit -r src/ -f json -o bandit-report.json || true
    - bandit -r src/ -ll  # Fail only on HIGH/CRITICAL
    # Dependency vulnerability check
    - pip install -r src/requirements.txt
    - safety check --full-report
  artifacts:
    paths:
      - bandit-report.json
    when: always

container_scan:
  stage: security_scan
  image: docker:24.0
  services: [docker:24.0-dind]
  before_script:
    - apk add --no-cache curl
    - curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
  script:
    - docker build -t ${DOCKER_IMAGE}:scan .
    # Scan for CRITICAL and HIGH vulnerabilities
    - trivy image --exit-code 1 --severity CRITICAL,HIGH --ignore-unfixed ${DOCKER_IMAGE}:scan
    # Full report (informational)
    - trivy image --format json -o trivy-report.json ${DOCKER_IMAGE}:scan
  artifacts:
    paths:
      - trivy-report.json
    when: always

secrets_scan:
  stage: security_scan
  image: zricethezav/gitleaks:latest
  script:
    - gitleaks detect --source . --report-format json --report-path gitleaks-report.json
  artifacts:
    paths:
      - gitleaks-report.json
    when: always

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: BUILD
# ─────────────────────────────────────────────────────────────────────────────
build_image:
  stage: build
  image: docker:24.0
  services: [docker:24.0-dind]
  script:
    - docker build
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        --build-arg VCS_REF=${CI_COMMIT_SHA}
        --build-arg VERSION=${DOCKER_TAG}
        --label "org.opencontainers.image.created=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        --label "org.opencontainers.image.revision=${CI_COMMIT_SHA}"
        -t ${DOCKER_IMAGE}:${DOCKER_TAG}
        -t ${DOCKER_IMAGE}:latest
        .
    - docker save ${DOCKER_IMAGE}:${DOCKER_TAG} > image.tar
  artifacts:
    paths:
      - image.tar
    expire_in: 1 hour

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: PUSH
# ─────────────────────────────────────────────────────────────────────────────
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
    # Tag with semver if tagged commit
    - |
      if [ -n "$CI_COMMIT_TAG" ]; then
        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:${CI_COMMIT_TAG}
        docker push ${DOCKER_IMAGE}:${CI_COMMIT_TAG}
      fi
  only:
    - main
    - tags

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: STAGING DEPLOY
# ─────────────────────────────────────────────────────────────────────────────
deploy_staging:
  stage: staging_deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: ${STAGING_URL}
  before_script:
    - kubectl config use-context ${KUBE_CONTEXT_STAGING}
  script:
    # Update image tag in staging
    - kubectl set image deployment/chunking-pipeline
        chunking-pipeline=${DOCKER_IMAGE}:${DOCKER_TAG}
        -n rag-staging
    # Wait for rollout
    - kubectl rollout status deployment/chunking-pipeline -n rag-staging --timeout=300s
    # Verify pods are healthy
    - kubectl get pods -n rag-staging -l app=chunking-pipeline
  only:
    - main

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7: SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
smoke_test:
  stage: smoke_test
  image: curlimages/curl:latest
  variables:
    TARGET_URL: ${STAGING_URL}
  script:
    # Health check
    - |
      echo "=== Health Check ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${TARGET_URL}/health)
      if [ "$HTTP_CODE" != "200" ]; then
        echo "FAILED: Health endpoint returned $HTTP_CODE"
        exit 1
      fi
      echo "PASSED: Health endpoint OK"

    # Readiness check
    - |
      echo "=== Readiness Check ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${TARGET_URL}/ready)
      if [ "$HTTP_CODE" != "200" ]; then
        echo "FAILED: Ready endpoint returned $HTTP_CODE"
        exit 1
      fi
      echo "PASSED: Readiness OK"

    # Trigger pipeline (functional test)
    - |
      echo "=== Functional Test: Trigger Pipeline ==="
      RESPONSE=$(curl -s -X POST ${TARGET_URL}/trigger)
      echo "Response: $RESPONSE"
      if echo "$RESPONSE" | grep -q "accepted"; then
        echo "PASSED: Pipeline trigger accepted"
      else
        echo "FAILED: Pipeline trigger rejected"
        exit 1
      fi

    # Wait and check status
    - |
      echo "=== Status Check (waiting 30s) ==="
      sleep 30
      STATUS=$(curl -s ${TARGET_URL}/status)
      echo "Status: $STATUS"

    # Metrics endpoint
    - |
      echo "=== Metrics Check ==="
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${TARGET_URL}/metrics)
      if [ "$HTTP_CODE" != "200" ]; then
        echo "FAILED: Metrics endpoint returned $HTTP_CODE"
        exit 1
      fi
      echo "PASSED: Metrics OK"

    - echo "=== ALL SMOKE TESTS PASSED ==="
  only:
    - main

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8: APPROVAL GATE (Manual — human approves prod deploy)
# ─────────────────────────────────────────────────────────────────────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging tests passed. Awaiting manual approval for production deployment."
    - echo "Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    - echo "Commit: ${CI_COMMIT_SHA}"
    - echo "Author: ${CI_COMMIT_AUTHOR}"
  when: manual  # <-- MANUAL APPROVAL REQUIRED
  allow_failure: false
  only:
    - main

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9: PROD DEPLOY — CANARY (10%)
# ─────────────────────────────────────────────────────────────────────────────
deploy_prod_canary:
  stage: prod_deploy_canary
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: ${PROD_URL}
  before_script:
    - kubectl config use-context ${KUBE_CONTEXT_PROD}
  script:
    # Deploy canary (10% traffic)
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
        replicas: 1    # 1 out of 10 total = 10% traffic
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
              readinessProbe:
                httpGet:
                  path: /ready
                  port: 8000
                initialDelaySeconds: 10
                periodSeconds: 5
      EOF
    - kubectl rollout status deployment/chunking-pipeline-canary -n rag-production --timeout=300s
    - echo "Canary deployed (10% traffic). Monitoring..."
  only:
    - main

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10: MONITOR (Canary health check — automated)
# ─────────────────────────────────────────────────────────────────────────────
monitor_canary:
  stage: monitor
  image: curlimages/curl:latest
  script:
    - |
      echo "=== Monitoring canary for 5 minutes ==="
      FAILURES=0
      CHECKS=10
      INTERVAL=30

      for i in $(seq 1 $CHECKS); do
        echo "--- Check $i/$CHECKS ---"
        
        # Health check
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${PROD_URL}/health)
        if [ "$HTTP_CODE" != "200" ]; then
          FAILURES=$((FAILURES + 1))
          echo "WARN: Health check failed (HTTP $HTTP_CODE)"
        else
          echo "OK: Health check passed"
        fi

        # Error rate check (from metrics)
        METRICS=$(curl -s ${PROD_URL}/metrics)
        echo "Metrics snapshot collected"

        if [ $FAILURES -ge 3 ]; then
          echo "=== CANARY FAILED: Too many failures ($FAILURES) ==="
          echo "Rolling back canary..."
          # Note: rollback happens in next step or manually
          exit 1
        fi

        [ $i -lt $CHECKS ] && sleep $INTERVAL
      done

      echo "=== CANARY MONITORING PASSED: $FAILURES failures out of $CHECKS checks ==="
  only:
    - main
  # If monitoring fails, auto-rollback canary
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        echo "Auto-rolling back canary deployment..."
        kubectl delete deployment chunking-pipeline-canary -n rag-production --ignore-not-found
      fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 11: PROD DEPLOY — FULL (100%)
# ─────────────────────────────────────────────────────────────────────────────
deploy_prod_full:
  stage: prod_deploy_full
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: ${PROD_URL}
  before_script:
    - kubectl config use-context ${KUBE_CONTEXT_PROD}
  script:
    # Update stable deployment to new image
    - kubectl set image deployment/chunking-pipeline
        chunking-pipeline=${DOCKER_IMAGE}:${DOCKER_TAG}
        -n rag-production
    - kubectl rollout status deployment/chunking-pipeline -n rag-production --timeout=600s
    
    # Remove canary (traffic now 100% on stable)
    - kubectl delete deployment chunking-pipeline-canary -n rag-production --ignore-not-found
    
    # Verify all pods healthy
    - kubectl get pods -n rag-production -l app=chunking-pipeline
    - echo "=== PRODUCTION DEPLOYMENT COMPLETE ==="
    - echo "Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    - echo "URL: ${PROD_URL}"
  only:
    - main
```

---

### Step 5: Kubernetes Manifests — External Access

#### `k8s/base/namespace.yaml`
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-production
  labels:
    app.kubernetes.io/part-of: rag-system
```

#### `k8s/base/deployment.yaml`
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
      maxUnavailable: 0    # Zero-downtime deployment
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
      serviceAccountName: chunking-pipeline-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: chunking-pipeline
        image: registry.gitlab.com/yourgroup/chunking-pipeline:latest
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
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
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 12   # 60 seconds max startup
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: chunking-pipeline
```

#### `k8s/base/service.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: chunking-pipeline
  namespace: rag-production
  labels:
    app: chunking-pipeline
spec:
  type: ClusterIP    # Internal — Ingress handles external
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: chunking-pipeline    # Both stable + canary get traffic
```

#### `k8s/base/ingress.yaml` — **EXTERNAL ACCESS (Cluster ke bahar se)**

```yaml
# =============================================================================
# INGRESS — Yeh cluster ke bahar se access enable karta hai
# Localhost NAHI — proper domain + TLS + load balancing
# =============================================================================
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chunking-pipeline-ingress
  namespace: rag-production
  annotations:
    # NGINX Ingress Controller
    kubernetes.io/ingress.class: nginx
    # SSL/TLS — Let's Encrypt auto certificate
    cert-manager.io/cluster-issuer: letsencrypt-prod
    # Rate limiting (DDoS protection)
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    # Request size limit (large docs upload protection)
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    # Timeouts (pipeline can take time)
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    # Security headers
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-XSS-Protection: 1; mode=block";
      more_set_headers "Strict-Transport-Security: max-age=31536000; includeSubDomains";
spec:
  tls:
  - hosts:
    - chunking.yourdomain.com
    secretName: chunking-tls-cert    # Auto-managed by cert-manager
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
---
# Alternative: If using AWS EKS — ALB Ingress (AWS Load Balancer Controller)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chunking-pipeline-alb
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing    # <-- EXTERNAL
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-south-1:123456789:certificate/xxx
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "15"
    alb.ingress.kubernetes.io/success-codes: "200"
    # WAF integration (optional)
    alb.ingress.kubernetes.io/wafv2-acl-arn: arn:aws:wafv2:ap-south-1:123456789:regional/webacl/xxx
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

#### `k8s/base/configmap.yaml`
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
  APP_VERSION: "latest"
```

#### `k8s/base/hpa.yaml` (Auto-scaling)
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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

#### `k8s/base/cronjob.yaml` (Scheduled chunking)
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: chunking-scheduled
  namespace: rag-production
spec:
  schedule: "0 2 * * *"           # Daily 2 AM IST
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
            - secretRef:
                name: chunking-secrets
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
          restartPolicy: OnFailure
```

#### `k8s/base/kustomization.yaml`
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: rag-production

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - ingress.yaml
  - configmap.yaml
  - hpa.yaml
  - cronjob.yaml

commonLabels:
  app.kubernetes.io/name: chunking-pipeline
  app.kubernetes.io/managed-by: argocd
```

---

### Step 6: ArgoCD Applications (Staging + Production)

#### `argocd/staging-app.yaml`
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: chunking-pipeline-staging
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/chunking-pipeline.git
    targetRevision: main
    path: k8s/overlays/staging
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-staging
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - ApplyOutOfSyncOnly=true
```

#### `argocd/production-app.yaml`
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: chunking-pipeline-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/chunking-pipeline.git
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-production
  syncPolicy:
    # Production: NO auto-sync — manual only (safety)
    syncOptions:
      - CreateNamespace=true
      - ApplyOutOfSyncOnly=true
      - PrunePropagationPolicy=foreground
```

---

### Step 7: External Access — Summary (Cluster ke bahar se kaise access karein)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL ACCESS FLOW                         │
│                                                                     │
│  User/Client (Internet)                                             │
│       │                                                             │
│       ▼                                                             │
│  DNS (chunking.yourdomain.com)                                      │
│       │                                                             │
│       ▼                                                             │
│  Cloud Load Balancer (ALB / NLB / Cloud LB)                         │
│       │  ← TLS termination (HTTPS)                                  │
│       ▼                                                             │
│  Ingress Controller (nginx-ingress / ALB controller)                │
│       │  ← Rate limiting, security headers                          │
│       ▼                                                             │
│  K8s Service (ClusterIP)                                            │
│       │  ← Internal load balancing                                  │
│       ▼                                                             │
│  Pods (chunking-pipeline)  [3 replicas + canary]                    │
│       │                                                             │
│       ▼                                                             │
│  :8000 → FastAPI (health, trigger, status, metrics)                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Key difference from localhost:**
- `localhost:8000` → sirf development mein kaam karta hai, cluster mein nahi
- Ingress + LoadBalancer → proper DNS, TLS, external IP assign hota hai
- Client kisi bhi jagah se `https://chunking.yourdomain.com/health` call kar sakta hai

**Access karne ke commands:**
```bash
# Check external IP/hostname
kubectl get ingress -n rag-production

# Output:
# NAME                         CLASS   HOSTS                      ADDRESS              PORTS     AGE
# chunking-pipeline-ingress    nginx   chunking.yourdomain.com    a1b2c3.elb.amazonaws.com   80, 443   5m

# Test from anywhere (not just cluster)
curl https://chunking.yourdomain.com/health
curl https://chunking.yourdomain.com/status
curl -X POST https://chunking.yourdomain.com/trigger
```

---

### Step 8: Quick Setup (Agar pehli baar deploy kar rahe ho)

```bash
# 1. Ingress Controller install karo (one-time)
# NGINX:
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.service.type=LoadBalancer

# OR AWS ALB Controller (EKS):
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName=your-cluster-name

# 2. Cert-Manager (auto TLS certificates)
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true

# 3. ClusterIssuer for Let's Encrypt
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

# 4. DNS Point karo
# chunking.yourdomain.com → Ingress ka external IP/hostname
# (Route53 / Cloudflare / your DNS provider)

# 5. Deploy!
kubectl apply -k k8s/base/
```

---

### Complete Pipeline Visualization:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          FULL DEPLOYMENT PIPELINE                                 │
│                                                                                  │
│  Developer pushes code to main                                                   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────┐  ┌──────┐  ┌───────────────┐  ┌───────┐  ┌──────┐                │
│  │  TEST   │→│ LINT  │→│ SECURITY SCAN │→│ BUILD │→│ PUSH │                    │
│  │(pytest) │  │(ruff) │  │(trivy+bandit) │  │(docker)│ │(registry)│              │
│  └─────────┘  └──────┘  └───────────────┘  └───────┘  └──────┘                │
│       │                                                                          │
│       ▼                                                                          │
│  ┌────────────────┐  ┌────────────┐  ┌─────────────────┐                       │
│  │ STAGING DEPLOY │→│ SMOKE TEST │→│ APPROVAL GATE   │  ← Human clicks         │
│  │ (kubectl)      │  │ (curl)     │  │ (manual trigger) │                       │
│  └────────────────┘  └────────────┘  └─────────────────┘                       │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐                  │
│  │ CANARY (10%)     │→│ MONITOR      │→│ FULL DEPLOY (100%)│                   │
│  │ (1/10 replicas)  │  │ (5 min watch)│  │ (rolling update)  │                  │
│  └──────────────────┘  └──────────────┘  └──────────────────┘                  │
│                                                                                  │
│  External Access: https://chunking.yourdomain.com                                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### Interview Mein Kaise Explain Karoge:

> "Humara chunking pipeline GitLab CI se build hota hai — full pipeline hai: unit tests, linting (ruff), security scanning (Trivy for container, Bandit for SAST, Gitleaks for secrets), Docker multi-stage build, registry push. Phir staging mein deploy hota hai ArgoCD ke through, automated smoke tests run hote hain (health, readiness, functional). Sab pass hone pe manual approval gate hai — koi senior engineer approve karega. Approve ke baad canary deployment hoti hai — production mein 10% traffic new version pe jaati hai (1 canary pod out of 10 total). 5 minutes monitoring hoti hai — health checks, error rates. Agar canary pass hota hai, toh rolling update se 100% traffic new version pe shift hoti hai. Canary fail hua toh auto-rollback. External access Ingress + LoadBalancer se hai with TLS, rate limiting, aur security headers."
