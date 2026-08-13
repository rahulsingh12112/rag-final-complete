# Topic 1 (DevOps Part): Embedding Service — GitLab CI/CD + ArgoCD Deployment

> **Goal:** Embedding model ko production-ready deploy karna — Docker → GitLab CI/CD → ArgoCD → Kubernetes
> **Real environment simulation:** Jaise tum actual company mein karte

---

## 🗺️ Full Picture — Kya Banayenge?

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FLOW                                          │
│                                                                      │
│  Code Push (GitLab) → CI Pipeline → Docker Build → Push to Registry  │
│                                          ↓                            │
│                              ArgoCD detects change                    │
│                                          ↓                            │
│                         Deploy to Kubernetes (EKS)                    │
│                                          ↓                            │
│                    Embedding Service running on K8s                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Project Structure

```
embedding-service/
├── app/
│   ├── main.py              ← FastAPI embedding server
│   ├── embedding_model.py   ← Model loading & encoding
│   └── requirements.txt     ← Python dependencies
├── Dockerfile               ← Container image
├── .gitlab-ci.yml           ← CI/CD pipeline
├── k8s/
│   ├── deployment.yaml      ← Kubernetes Deployment
│   ├── service.yaml         ← Kubernetes Service
│   ├── hpa.yaml             ← Horizontal Pod Autoscaler
│   └── configmap.yaml       ← Configuration
└── argocd/
    └── application.yaml     ← ArgoCD Application manifest
```

---

## Step 2: Application Code

### `app/main.py` — FastAPI Embedding Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from embedding_model import EmbeddingService
import time

app = FastAPI(title="Embedding Service", version="1.0.0")
model = EmbeddingService()


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False  # True = search query, False = document


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    processing_time_ms: float


@app.get("/health")
def health():
    """K8s readiness/liveness probe ke liye"""
    return {"status": "healthy", "model_loaded": model.is_loaded}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """Text ko embedding vector mein convert karo"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    
    if len(request.texts) > 64:
        raise HTTPException(status_code=400, detail="Max 64 texts per request")

    start = time.time()

    if request.is_query:
        embeddings = model.encode_queries(request.texts)
    else:
        embeddings = model.encode_documents(request.texts)

    elapsed = (time.time() - start) * 1000

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimensions=model.dimensions,
        processing_time_ms=round(elapsed, 2)
    )


@app.get("/info")
def info():
    """Model info return karo"""
    return {
        "model_name": model.model_name,
        "dimensions": model.dimensions,
        "max_tokens": 512,
        "device": str(model.device)
    }
```

### `app/embedding_model.py` — Model Logic

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os


class EmbeddingService:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))

        print(f"Loading model: {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        self.is_loaded = True
        print(f"Model loaded! Dimensions: {self.dimensions}")

        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def encode_documents(self, texts: list) -> np.ndarray:
        """Documents encode karo (no prefix)"""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def encode_queries(self, queries: list) -> np.ndarray:
        """Queries encode karo (with BGE prefix)"""
        prefixed = [f"{self.query_prefix}{q}" for q in queries]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
```

### `app/requirements.txt`

```
fastapi==0.109.0
uvicorn==0.27.0
sentence-transformers==2.3.1
torch==2.1.2
numpy==1.26.3
pydantic==2.5.3
```

---

## Step 3: Dockerfile

```dockerfile
# Multi-stage build — production image chhoti hogi
FROM python:3.11-slim AS builder

WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Production Stage ---
FROM python:3.11-slim

WORKDIR /app

# Dependencies copy karo builder se
COPY --from=builder /install /usr/local

# Application code copy karo
COPY app/ .

# Non-root user (security best practice)
RUN useradd -m appuser
USER appuser

# Model download at build time (optional — ya runtime pe hoga)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

### Local test karo:

```bash
# Build karo
docker build -t embedding-service:local .

# Run karo
docker run -p 8080:8080 embedding-service:local

# Test karo (dusre terminal mein)
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is Kubernetes?"], "is_query": true}'
```

---

## Step 4: GitLab CI/CD Pipeline

### `.gitlab-ci.yml`

```yaml
# ====================================
# Embedding Service — CI/CD Pipeline
# ====================================

stages:
  - test          # Code quality + unit tests
  - build         # Docker image build + push
  - security      # Container security scan
  - deploy        # ArgoCD sync trigger

variables:
  IMAGE_NAME: registry.gitlab.com/$CI_PROJECT_PATH/embedding-service
  IMAGE_TAG: $CI_COMMIT_SHORT_SHA
  # ArgoCD
  ARGOCD_SERVER: argocd.your-company.com
  APP_NAME: embedding-service

# -----------------------------------
# Stage 1: TEST
# -----------------------------------
unit-test:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install -r app/requirements.txt
    - pip install pytest httpx
  script:
    - echo "Running unit tests..."
    - pytest tests/ -v --tb=short
  rules:
    - if: $CI_MERGE_REQUEST_ID    # MR pe run karo
    - if: $CI_COMMIT_BRANCH == "main"  # main branch pe bhi

lint:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install ruff
  script:
    - echo "Running linter..."
    - ruff check app/
  allow_failure: true  # Lint fail hone pe pipeline rok nahi

# -----------------------------------
# Stage 2: BUILD
# -----------------------------------
docker-build:
  stage: build
  image: docker:24.0
  services:
    - docker:24.0-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - echo "Building Docker image..."
    - docker build -t $IMAGE_NAME:$IMAGE_TAG .
    - docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:latest
    - echo "Pushing to registry..."
    - docker push $IMAGE_NAME:$IMAGE_TAG
    - docker push $IMAGE_NAME:latest
    - echo "✅ Image pushed → $IMAGE_NAME:$IMAGE_TAG"
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

# -----------------------------------
# Stage 3: SECURITY SCAN
# -----------------------------------
container-scan:
  stage: security
  image: aquasec/trivy:latest
  script:
    - echo "Scanning image for vulnerabilities..."
    - trivy image --severity HIGH,CRITICAL --exit-code 1 $IMAGE_NAME:$IMAGE_TAG
  allow_failure: false  # Critical vulnerability = pipeline fail
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

# -----------------------------------
# Stage 4: DEPLOY (ArgoCD Sync)
# -----------------------------------
deploy-staging:
  stage: deploy
  image: argoproj/argocd:v2.9.3
  script:
    - echo "Updating image tag in ArgoCD..."
    - argocd login $ARGOCD_SERVER --username $ARGOCD_USER --password $ARGOCD_PASS --insecure
    - argocd app set $APP_NAME --parameter image.tag=$IMAGE_TAG
    - argocd app sync $APP_NAME --prune
    - argocd app wait $APP_NAME --timeout 300
    - echo "✅ Deployed to staging!"
  environment:
    name: staging
    url: https://embedding-staging.your-company.com
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

deploy-production:
  stage: deploy
  image: argoproj/argocd:v2.9.3
  script:
    - argocd login $ARGOCD_SERVER --username $ARGOCD_USER --password $ARGOCD_PASS --insecure
    - argocd app set ${APP_NAME}-prod --parameter image.tag=$IMAGE_TAG
    - argocd app sync ${APP_NAME}-prod --prune
    - argocd app wait ${APP_NAME}-prod --timeout 300
    - echo "✅ Deployed to production!"
  environment:
    name: production
    url: https://embedding.your-company.com
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  when: manual  # Production = manual approval required ⚠️
```

### Pipeline visually:

```
┌──────┐    ┌───────┐    ┌──────────┐    ┌──────────────────┐
│ TEST │ →  │ BUILD │ →  │ SECURITY │ →  │ DEPLOY (ArgoCD)  │
│      │    │       │    │  SCAN    │    │                  │
│• unit│    │• docker│   │• trivy   │    │• staging (auto)  │
│• lint│    │• push  │   │  scan    │    │• prod (manual)   │
└──────┘    └───────┘    └──────────┘    └──────────────────┘
```

---

## Step 5: Kubernetes Manifests

### `k8s/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: embedding-config
  namespace: ml-services
data:
  MODEL_NAME: "BAAI/bge-base-en-v1.5"
  BATCH_SIZE: "32"
  LOG_LEVEL: "info"
```

### `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
  namespace: ml-services
  labels:
    app: embedding-service
    version: v1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
        version: v1
    spec:
      containers:
      - name: embedding
        image: registry.gitlab.com/your-project/embedding-service:IMAGE_TAG_PLACEHOLDER
        ports:
        - containerPort: 8080
          name: http
        envFrom:
        - configMapRef:
            name: embedding-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 45   # Model load hone mein time lagta hai
          periodSeconds: 10
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
      # Model cache persist karo (restart pe dubara download na ho)
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 5Gi
```

### `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
  namespace: ml-services
spec:
  selector:
    app: embedding-service
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  type: ClusterIP   # Internal only — bahar expose nahi
```

### `k8s/hpa.yaml` — Auto Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedding-service-hpa
  namespace: ml-services
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70   # 70% CPU pe scale up
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60    # 1 min wait before scaling up
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300   # 5 min wait before scaling down
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

---

## Step 6: ArgoCD Application

### `argocd/application.yaml`

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: embedding-service
  namespace: argocd
  # Auto-delete cleanup
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: ml-platform   # ArgoCD project (RBAC grouping)

  source:
    repoURL: https://gitlab.your-company.com/ml-team/embedding-service.git
    targetRevision: main
    path: k8s              # K8s manifests ka folder

  destination:
    server: https://kubernetes.default.svc   # In-cluster
    namespace: ml-services

  syncPolicy:
    automated:
      prune: true          # Old resources delete karo
      selfHeal: true       # Manual changes revert karo (GitOps!)
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
    retry:
      limit: 3
      backoff:
        duration: 30s
        factor: 2
        maxDuration: 3m
```

### ArgoCD kya karega:

```
1. GitLab repo watch karega (k8s/ folder)
2. Koi change aaye (new image tag, config change)
3. Automatically K8s mein deploy karega
4. Agar koi manually K8s mein change kare → revert karega (self-heal)
5. Failed deployment → auto retry (3 attempts)
```

---

## Step 7: Complete Flow — End to End

### Jab tum code push karte ho:

```
YOU: git push origin main
         │
         ▼
┌─────────────────────────────┐
│     GitLab CI Pipeline       │
│                              │
│  1. pytest (tests pass?)     │
│  2. ruff (code clean?)       │
│  3. docker build + push      │
│  4. trivy scan (secure?)     │
│  5. argocd app set (new tag) │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│         ArgoCD               │
│                              │
│  1. Detects new image tag    │
│  2. Syncs K8s manifests      │
│  3. Rolling update starts    │
│  4. Old pods → New pods      │
│  5. Health check passes ✅    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Kubernetes (EKS)          │
│                              │
│  Pod 1: embedding-service ✅  │
│  Pod 2: embedding-service ✅  │
│  Service: ClusterIP:80       │
│  HPA: auto-scale 2→10       │
└─────────────────────────────┘
```

---

## Step 8: Setup Commands — Apne Machine Pe

### 8.1: GitLab Repo Setup

```bash
# Naya repo banao ya existing use karo
cd ~/projects
mkdir embedding-service && cd embedding-service
git init

# Folders banao
mkdir -p app k8s argocd tests

# Files banao (upar wala code daalo)
# ... (main.py, embedding_model.py, Dockerfile, etc.)

# GitLab pe push karo
git remote add origin https://gitlab.your-company.com/ml-team/embedding-service.git
git add .
git commit -m "Initial: Embedding service with CI/CD + ArgoCD"
git push -u origin main
```

### 8.2: GitLab Variables Set Karo (Settings → CI/CD → Variables)

```
CI_REGISTRY_USER    = your-gitlab-username
CI_REGISTRY_PASSWORD = your-access-token
ARGOCD_SERVER       = argocd.your-company.com
ARGOCD_USER         = admin
ARGOCD_PASS         = your-argocd-password (masked, protected)
```

### 8.3: ArgoCD Mein App Register Karo

```bash
# ArgoCD CLI login
argocd login argocd.your-company.com --username admin --password <pass>

# App create karo
argocd app create embedding-service \
  --repo https://gitlab.your-company.com/ml-team/embedding-service.git \
  --path k8s \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace ml-services \
  --sync-policy automated \
  --auto-prune \
  --self-heal

# Status check
argocd app get embedding-service
```

### 8.4: Verify Deployment

```bash
# Pods check
kubectl get pods -n ml-services

# Service check
kubectl get svc -n ml-services

# Logs check
kubectl logs -l app=embedding-service -n ml-services --tail=50

# Test karo (port-forward)
kubectl port-forward svc/embedding-service 8080:80 -n ml-services

# Dusre terminal mein:
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is Kubernetes?"], "is_query": true}'
```

---

## Step 9: Monitoring & Troubleshooting

### Pipeline fail hone pe:

```bash
# GitLab UI → CI/CD → Pipelines → Click failed job → Read logs

# Common issues:
# 1. Docker build fail → Check Dockerfile, dependencies
# 2. Trivy scan fail → Fix vulnerabilities in base image
# 3. ArgoCD sync fail → Check K8s manifests syntax
# 4. Tests fail → Fix code, run locally first
```

### ArgoCD issues:

```bash
# App status
argocd app get embedding-service

# Sync history
argocd app history embedding-service

# Force sync (stuck hone pe)
argocd app sync embedding-service --force

# Rollback (problem hone pe)
argocd app rollback embedding-service <revision-number>
```

### Kubernetes issues:

```bash
# Pod crash?
kubectl describe pod <pod-name> -n ml-services

# OOMKilled? → Memory increase karo deployment.yaml mein
# CrashLoopBackOff? → Model load fail ho raha, logs dekho
# ImagePullBackOff? → Registry credentials check karo
```

---

## Step 10: Key Takeaways (Interview Mein Bolo)

| Question | Answer |
|----------|--------|
| "CI/CD pipeline describe karo" | "Test → Build → Security Scan → Deploy via ArgoCD. Pipeline GitLab mein, deployment GitOps style ArgoCD se" |
| "GitOps kya hai?" | "Git repo = single source of truth. ArgoCD continuously reconciles cluster state with git state. Manual changes auto-revert" |
| "Zero-downtime deployment kaise?" | "Rolling update — new pods healthy hone ke baad old pods terminate. readinessProbe ensure karta hai traffic sirf healthy pods ko jaaye" |
| "Rollback kaise?" | "ArgoCD rollback command ya git revert + push. ArgoCD automatically purana state deploy kar dega" |
| "Security kaise handle karte ho?" | "Trivy scan in CI, non-root container user, ClusterIP (no public exposure), GitLab masked variables for secrets" |

---

## Summary — Kya Seekha Is Section Mein:

```
✅ Embedding service as Docker container (FastAPI)
✅ GitLab CI/CD pipeline (test → build → scan → deploy)
✅ ArgoCD GitOps deployment (auto-sync, self-heal)
✅ Kubernetes manifests (Deployment, Service, HPA, ConfigMap)
✅ Monitoring & troubleshooting commands
✅ Real-world flow: code push → production deployment
```

**Next Section:** Chunking Strategies + GitLab CI (batch chunking job) + ArgoCD (CronJob deployment)
