# Topic 9: End-to-End RAG Deployment & Scaling — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Topics 1-8 complete
> **Source:** Engineer Repo → mod-110-llm-infrastructure + Production RAG Architecture Patterns 2026

---

## 🎯 One-Liner (Interview):

> "Production RAG deployment is a distributed microservices architecture — embedding service, vector DB cluster, retrieval API, LLM gateway, guardrails — each independently scalable, with load testing to identify bottlenecks, auto-scaling policies, cost optimization through tiered routing, and blue-green deployments for zero-downtime updates."

---

## Layer 1: Kya Hai Aur Kyun Complex Hai?

Topics 1-8 mein tumne har component individually seekha. Ab sab ko **together** deploy karna hai as one cohesive system. Challenges:

1. **6+ microservices** coordinate karte hain per request
2. **Latency budget** — user ko 2-3 seconds mein answer chahiye, total budget distribute karna hai
3. **Scaling** — har service ka bottleneck alag hai (CPU, GPU, RAM, network)
4. **Cost** — LLM calls expensive, vector DB RAM-heavy, embedding GPU-intensive
5. **Reliability** — any single service down = whole RAG down
6. **Updates** — model change = re-indexing millions of documents

---

## Layer 2: Complete Architecture (All Services Together)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RAG SYSTEM ARCHITECTURE                             │
│                                                                                 │
│  ┌──────────┐                                                                   │
│  │  Client  │ ─── HTTPS ──→ ┌─────────────────────────┐                        │
│  │ (Browser/│               │    API Gateway /         │                        │
│  │  Mobile) │               │    Load Balancer         │                        │
│  └──────────┘               └────────────┬────────────┘                        │
│                                          │                                     │
│                                          ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐     │
│  │                     RAG ORCHESTRATOR SERVICE                           │     │
│  │  (receives query, coordinates all services, returns answer)           │     │
│  └───┬──────────┬──────────────┬───────────────┬────────────┬───────────┘     │
│      │          │              │               │            │                  │
│      ▼          ▼              ▼               ▼            ▼                  │
│  ┌────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐           │
│  │Guard-  │ │ Embedding  │ │Retrieval │ │   LLM    │ │   Eval    │           │
│  │rails   │ │ Service    │ │ Service  │ │ Gateway  │ │  Service  │           │
│  │(Topic8)│ │ (Topic 1)  │ │(Topic 4) │ │(Topic 6) │ │ (Topic 7) │           │
│  └────────┘ └─────┬──────┘ └────┬─────┘ └────┬─────┘ └───────────┘           │
│                    │             │             │                                │
│              ┌─────┘       ┌────┴────┐        │                                │
│              ▼             ▼         ▼        ▼                                │
│         [GPU Nodes]   ┌────────┐ ┌──────┐  ┌────────────┐                     │
│                       │ Qdrant │ │Redis │  │ OpenAI /   │                     │
│                       │Cluster │ │Cache │  │ Bedrock /  │                     │
│                       │(3 node)│ │      │  │ Self-hosted│                     │
│                       └────────┘ └──────┘  └────────────┘                     │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │ INFRASTRUCTURE: Prometheus + Grafana + Jaeger + ArgoCD + GitLab CI  │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Communication Matrix:

| From → To | Protocol | Latency Target | Why |
|-----------|----------|---------------|-----|
| Client → API Gateway | HTTPS | N/A | External |
| API GW → Orchestrator | HTTP/gRPC | <5ms | Internal LB |
| Orchestrator → Guardrails | HTTP | <50ms | Input check |
| Orchestrator → Embedding | gRPC | <30ms | Vector encode |
| Orchestrator → Retrieval | HTTP | <100ms | Hybrid search |
| Retrieval → Qdrant | gRPC | <20ms | Vector search |
| Retrieval → Elasticsearch | HTTP | <15ms | BM25 |
| Retrieval → Redis | TCP | <5ms | Cache |
| Orchestrator → LLM GW | HTTP/SSE | <2000ms | Generation |
| LLM GW → Provider API | HTTPS | <1500ms | External |

---

## Layer 3: Latency Budget Breakdown

**Target: < 3 seconds end-to-end (p95)**

```
┌─────────────────────────────────────────────────────────────────────┐
│ LATENCY BUDGET (3000ms total)                                        │
│                                                                     │
│  Input Guardrails:     50ms  ████                                   │
│  Query Embedding:      30ms  ███                                    │
│  Hybrid Retrieval:    100ms  ████████                               │
│  Reranking:            80ms  ███████                                │
│  Context Assembly:      5ms  █                                      │
│  LLM Generation:     2000ms  ████████████████████████████████████   │
│  Output Guardrails:    50ms  ████                                   │
│  Network overhead:    100ms  ████████                               │
│  ─────────────────────────────                                      │
│  TOTAL:              2415ms  ✅ Under 3s budget                     │
│                                                                     │
│  BOTTLENECK: LLM Generation (67% of total time)                     │
│  Optimization: Streaming reduces perceived latency to ~200ms TTFT   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 4: Scaling Each Service

### Scaling Matrix:

| Service | Bottleneck | Scale Strategy | Min Replicas | HPA Metric |
|---------|-----------|---------------|-------------|------------|
| Guardrails | CPU | Horizontal (stateless) | 3 | CPU 65% |
| Embedding | GPU/CPU | Horizontal + GPU nodes | 2 | CPU 70% |
| Retrieval API | CPU + Network | Horizontal (stateless) | 3 | CPU 65% |
| Qdrant | RAM + Disk | Vertical + Sharding | 3 (StatefulSet) | N/A (manual) |
| Elasticsearch | RAM + Disk | Sharding + Replicas | 3 (StatefulSet) | N/A (manual) |
| Redis | RAM | Vertical (or cluster) | 1 (+ replica) | Memory 80% |
| LLM Gateway | Network I/O | Horizontal (stateless) | 3 | CPU 60% |
| Orchestrator | CPU + Network | Horizontal (stateless) | 3 | CPU 65% |
| Eval Service | CPU | Horizontal | 2 | N/A (batch) |

### Load Testing (Pre-Production):

```python
# locustfile.py — Load testing RAG system
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task(10)
    def ask_question(self):
        """Primary user action — ask a question"""
        self.client.post("/ask", json={
            "query": "How does EKS autoscaling work?",
            "top_k": 5
        })

    @task(3)
    def ask_with_history(self):
        """Multi-turn conversation"""
        # First question
        resp = self.client.post("/ask", json={"query": "What is EKS pricing?"})
        # Follow-up
        self.client.post("/ask", json={
            "query": "Is it cheaper than GKE?",
            "conversation_history": [{"user": "What is EKS pricing?", "assistant": resp.json().get("answer", "")}]
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

```bash
# Run load test
locust -f locustfile.py --host=https://rag.yourdomain.com \
  --users 100 --spawn-rate 10 --run-time 10m

# Key metrics to watch:
# - p95 latency < 3s
# - Error rate < 1%
# - Throughput: queries/second
# - At what concurrency does system degrade?
```

### Auto-Scaling Configuration:

```yaml
# Orchestrator HPA (main entry point — most critical)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-orchestrator-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-orchestrator
  minReplicas: 3
  maxReplicas: 30
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"    # Scale if > 50 req/s per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 5
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 120
```

---

## Layer 5: Cost Optimization

### Monthly Cost Breakdown (10K queries/day):

| Service | Instance/Resource | Monthly Cost |
|---------|-----------------|-------------|
| EKS Control Plane | 1 cluster | $73 |
| Worker Nodes (3× m5.xlarge) | General workloads | $420 |
| GPU Node (1× g4dn.xlarge) | Embedding service | $384 |
| Qdrant (3× r6g.large) | Vector storage (16GB each) | $240 |
| Elasticsearch (3× m5.large) | BM25 index | $210 |
| Redis (r6g.large) | Query cache | $80 |
| LLM API (GPT-4o-mini) | ~300K queries × 4K tokens | $180 |
| S3 (document storage) | 100GB | $2 |
| CloudWatch/Monitoring | Logs + metrics | $50 |
| **TOTAL** | | **~$1,640/month** |

### Cost Optimization Strategies:

```python
class CostOptimizer:
    """Strategies to reduce RAG system cost"""

    def optimize_llm_cost(self):
        """LLM is usually biggest cost — optimize here first"""
        strategies = {
            "smart_routing": "70% queries → cheap model, 30% → expensive",
            "response_caching": "Cache identical queries (30-40% hit rate)",
            "shorter_prompts": "Less context = fewer tokens = cheaper",
            "batch_non_urgent": "Batch low-priority queries for off-peak pricing",
        }
        return strategies

    def optimize_compute_cost(self):
        """Instance optimization"""
        strategies = {
            "spot_instances": "Embedding batch jobs on spot (70% savings)",
            "right_sizing": "Monitor actual CPU/RAM usage, downsize over-provisioned",
            "scale_to_zero": "Eval service → 0 replicas at night",
            "arm_instances": "Graviton (r6g) = 20% cheaper than Intel (r5)",
        }
        return strategies

    def optimize_storage_cost(self):
        """Vector DB + search optimization"""
        strategies = {
            "quantization": "INT8 on Qdrant = 4x less RAM",
            "tiered_storage": "Old vectors → mmap (disk), recent → RAM",
            "index_lifecycle": "Delete vectors older than 1 year",
            "compression": "Enable ES compression for BM25 index",
        }
        return strategies
```

### Cost at Scale:

| Scale | Queries/Day | Monthly Cost | Cost/Query |
|-------|------------|-------------|-----------|
| Small | 1K | ~$500 | $0.017 |
| Medium | 10K | ~$1,640 | $0.0055 |
| Large | 100K | ~$5,000 | $0.0017 |
| Enterprise | 1M | ~$15,000 | $0.0005 |

**Economy of scale:** Cost per query drops 97% from 1K to 1M queries/day.

---

## Layer 6: Deployment Strategies

### Blue-Green Deployment (For Model Changes):

```
Current (Blue):                    New (Green):
┌─────────────────────┐           ┌─────────────────────┐
│ Embedding: BGE-base │           │ Embedding: BGE-large │
│ Collection: docs_v1 │           │ Collection: docs_v2  │
│ Traffic: 100% ██████│           │ Traffic: 0%          │
└─────────────────────┘           └─────────────────────┘

Step 1: Build green (new model + re-index)  [background, hours/days]
Step 2: Validate green (eval tests pass)
Step 3: Switch traffic: blue 0% ← → green 100%
Step 4: Delete blue after 24h (rollback window)
```

```yaml
# Blue-green via ArgoCD Rollouts
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: rag-orchestrator
  namespace: rag-production
spec:
  replicas: 5
  strategy:
    blueGreen:
      activeService: rag-orchestrator-active
      previewService: rag-orchestrator-preview
      autoPromotionEnabled: false     # Manual promotion after validation
      prePromotionAnalysis:
        templates:
        - templateName: rag-quality-check
  template:
    spec:
      containers:
      - name: rag-orchestrator
        image: registry.gitlab.com/yourgroup/rag-orchestrator:latest
```

### Canary with Progressive Traffic:

```
Time 0:    Canary 5%  → Stable 95%
Time 5m:   Canary 10% → Stable 90%   (if metrics OK)
Time 15m:  Canary 25% → Stable 75%   (if metrics OK)
Time 30m:  Canary 50% → Stable 50%   (if metrics OK)
Time 60m:  Canary 100%               (promotion)
```

---

## Layer 7: Disaster Recovery & High Availability

### Multi-AZ Deployment:

```yaml
# Topology spread — pods across AZs
topologySpreadConstraints:
- maxSkew: 1
  topologyKey: topology.kubernetes.io/zone
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchLabels:
      app: rag-orchestrator
```

### Backup Strategy:

| Component | Backup Method | Frequency | RTO | RPO |
|-----------|-------------|-----------|-----|-----|
| Qdrant vectors | Snapshot → S3 | Daily | 1 hour | 24 hours |
| Elasticsearch | Snapshot → S3 | Daily | 1 hour | 24 hours |
| Redis cache | No backup (ephemeral) | N/A | Instant (empty) | N/A |
| Config/Secrets | Git + K8s etcd backup | Per change | 5 min | 0 |
| Documents (S3) | Cross-region replication | Real-time | Minutes | ~0 |

### Circuit Breaker at Orchestrator Level:

```python
class RAGOrchestrator:
    """Orchestrates all services with fallback behavior"""

    def query(self, user_query: str) -> dict:
        # Guardrails — if down, proceed without (log warning)
        try:
            guard_result = self.guardrails.check_input(user_query)
            if guard_result["blocked"]:
                return {"answer": guard_result["reason"], "fallback": False}
        except Exception:
            pass  # Guardrails down → proceed (graceful degradation)

        # Retrieval — if down, use cached results or generic answer
        try:
            chunks = self.retrieval.search(user_query, top_k=5)
        except Exception:
            chunks = self.cache.get_fallback(user_query)  # Try cache
            if not chunks:
                return {"answer": "Service temporarily unavailable. Please try again.", "fallback": True}

        # LLM — if primary down, fallback chain
        answer = self.llm_gateway.generate(prompt, fallback=True)

        return {"answer": answer, "fallback": False}
```

---

## Layer 8: Document Ingestion Pipeline (Background)

```
┌──────────────────────────────────────────────────────────────────┐
│              DOCUMENT INGESTION PIPELINE (Async)                   │
│                                                                  │
│  S3 Upload → EventBridge → Step Functions:                       │
│                                                                  │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────┐ │
│  │ Parse  │──▶│ Chunk  │──▶│ Embed  │──▶│ Index  │──▶│ Verify│ │
│  │(PDF/MD)│   │(Topic2)│   │(Topic1)│   │(Qdrant │   │(count)│ │
│  │        │   │        │   │        │   │ + ES)  │   │       │ │
│  └────────┘   └────────┘   └────────┘   └────────┘   └──────┘ │
│                                                                  │
│  Triggers: S3 upload, Scheduled (daily), Manual API              │
│  Processing: K8s Job (batch) or Lambda (event-driven)            │
│  Monitoring: Document count, chunk count, failed docs            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Layer 9: Production Pitfalls

### Pitfall 1: Tightly Coupled Services

One service change breaks others. Deployment requires coordinating all services.

**Fix:** Versioned APIs. Each service independently deployable. Contract testing.

### Pitfall 2: No Graceful Degradation

Elasticsearch down → entire RAG system returns 500.

**Fix:** Fallback to dense-only search if BM25 unavailable. Cached responses if vector DB down.

### Pitfall 3: Cold Start After Deployment

New pods start, model not loaded yet, requests fail for 30-60 seconds.

**Fix:** Readiness probes with appropriate `initialDelaySeconds`. Pre-warm models. Rolling updates (maxUnavailable: 0).

### Pitfall 4: Re-indexing Blocks Production

Model change requires re-indexing 10M docs. During re-index, search returns partial results.

**Fix:** Blue-green collections. Index into new collection in background. Switch atomically when complete.

### Pitfall 5: No Cost Visibility

Month-end bill surprise. Don't know which component costs most.

**Fix:** Cost allocation tags on all resources. Per-service cost dashboards. Budget alerts.

---

## Layer 10: Interview Ready

### 2-Line Answer (Screening):

> "Production RAG is a distributed microservices system with 6+ services (embedding, retrieval, LLM gateway, guardrails, eval) each independently scaled — the key challenges are latency budget distribution across services, cost optimization through smart routing and caching, and zero-downtime deployments with blue-green for model updates."

### 5-Min Answer (Technical Round):

> Above + architecture diagram (all services + communication), latency budget (LLM = 67%), scaling strategies per service (stateless horizontal vs StatefulSet), cost breakdown ($1.6K/month for 10K queries/day), load testing approach, graceful degradation patterns.

### 10-Min Deep Dive (System Design):

> Above + blue-green for model changes, progressive canary rollout, disaster recovery (multi-AZ, backups, RTO/RPO), document ingestion pipeline (async), cost optimization at scale (spot instances, quantization, tiered routing), observability across services (distributed tracing), capacity planning formula.

### Follow-up Questions:

**Q: "10K queries/day se 100K pe scale karna hai — kya change hoga?"**
A: (1) HPA limits increase (max 30 → 100 for orchestrator). (2) Qdrant: add sharding (6 shards across 3 nodes). (3) LLM: increase rate limits, add caching (Redis). (4) Embedding: add GPU node or second replica. (5) Cost: optimize with smart routing (save 60% on LLM). Total cost: ~$5K/month.

**Q: "Embedding model change karna hai production mein — process kya hoga?"**
A: Blue-green. (1) Create new Qdrant collection. (2) Re-index all docs with new model (background job, hours/days). (3) Run eval on new index — confirm quality >= old. (4) Switch retrieval service to new collection (config update). (5) Keep old collection 24h for rollback. (6) Delete old.

**Q: "Single region mein latency 3s acceptable hai — multi-region kaise karoge?"**
A: (1) User-facing services: deploy in each region. (2) Vector DB: replicate to each region (Qdrant supports cross-region replication) OR use managed (Pinecone global). (3) LLM: API providers are global. (4) CDN for static assets. Cost: ~2.5x single region.

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| Complete architecture (all services together) | ✅ |
| Service communication matrix | ✅ |
| Latency budget breakdown | ✅ |
| Scaling strategy per service | ✅ |
| Load testing | ✅ |
| Auto-scaling configuration | ✅ |
| Cost breakdown + optimization | ✅ |
| Deployment strategies (blue-green, canary) | ✅ |
| Disaster recovery + HA | ✅ |
| Document ingestion pipeline | ✅ |
| Graceful degradation | ✅ |
| Production pitfalls (5) | ✅ |
| Interview answers | ✅ |

**Topic 9: End-to-End RAG Deployment & Scaling — DONE.**

---
## Layer 12: GitLab CI/CD + ArgoCD — RAG Orchestrator (Master Service) Production Deployment

RAG Orchestrator — the master service jo saari services ko coordinate karta hai (guardrails → embedding → retrieval → LLM → output) — production mein deploy with full pipeline + external access.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
rag-orchestrator/
├── src/
│   ├── app.py                  # FastAPI — main entry point
│   ├── orchestrator.py         # Service coordination logic
│   ├── circuit_breaker.py      # Resilience patterns
│   ├── config.py
│   ├── requirements.txt
│   └── tests/
│       ├── test_orchestrator.py
│       └── test_app.py
├── loadtest/
│   └── locustfile.py           # Load testing
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml            # External access — MAIN ENTRY POINT
│   ├── configmap.yaml
│   ├── hpa.yaml
│   └── pdb.yaml
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: RAG Orchestrator (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from orchestrator import RAGOrchestrator
from config import settings
import json
import os
import time

app = FastAPI(title="RAG Orchestrator", version="1.0.0",
              description="Main entry point for RAG system")

# Metrics
QUERIES_TOTAL = Counter("rag_queries_total", "Total queries", ["status"])
QUERY_LATENCY = Histogram("rag_query_latency_seconds", "End-to-end latency",
                          buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
FALLBACK_USED = Counter("rag_fallback_total", "Fallback activations", ["service"])

orchestrator = RAGOrchestrator(settings)


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True
    stream: bool = False
    conversation_history: list[dict] | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    metadata: dict


@app.get("/health")
def health():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    """Check all downstream services"""
    status = orchestrator.check_all_services()
    all_ok = all(status.values())
    if not all_ok:
        # Still ready if non-critical services down (graceful degradation)
        critical = ["retrieval", "llm_gateway"]
        critical_ok = all(status.get(s, False) for s in critical)
        if not critical_ok:
            raise HTTPException(503, f"Critical services down: {status}")
    return {"status": "ready", "services": status}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Main RAG endpoint — full pipeline"""
    start = time.time()
    try:
        result = orchestrator.process(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            history=request.conversation_history
        )
        QUERIES_TOTAL.labels(status="success").inc()
        QUERY_LATENCY.observe(time.time() - start)

        if result.get("fallback_used"):
            for svc in result.get("fallback_services", []):
                FALLBACK_USED.labels(service=svc).inc()

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            metadata={
                "latency_ms": round((time.time() - start) * 1000, 2),
                "model_used": result.get("model_used", "unknown"),
                "chunks_used": result.get("chunks_used", 0),
                "cached": result.get("cached", False),
                "fallback": result.get("fallback_used", False),
                "grounded": result.get("grounded", True),
            }
        )
    except Exception as e:
        QUERIES_TOTAL.labels(status="error").inc()
        raise HTTPException(500, str(e))


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    """Streaming RAG response"""
    async def stream():
        async for event in orchestrator.process_stream(
            query=request.query, top_k=request.top_k,
            history=request.conversation_history
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/services/status")
def services_status():
    """All downstream service health"""
    return orchestrator.check_all_services()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
```

### Step 2: Orchestrator Logic (`src/orchestrator.py`)

```python
import requests
import time
from circuit_breaker import CircuitBreaker
from config import Settings


class RAGOrchestrator:
    """Coordinates all RAG services with resilience"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.breakers = {
            "guardrails": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
            "retrieval": CircuitBreaker(failure_threshold=3, recovery_timeout=60),
            "llm": CircuitBreaker(failure_threshold=3, recovery_timeout=60),
        }

    def process(self, query: str, top_k: int = 5,
                use_reranking: bool = True, history: list = None) -> dict:
        fallback_services = []

        # 1. Input Guardrails (non-critical — proceed if down)
        try:
            guard = self.breakers["guardrails"].call(
                self._call_guardrails_input, query
            )
            if guard.get("blocked"):
                return {"answer": guard["reason"], "sources": [], "fallback_used": False}
            query = guard.get("processed_query", query)
        except Exception:
            fallback_services.append("guardrails")

        # 2. Retrieval (critical)
        try:
            retrieval_result = self.breakers["retrieval"].call(
                self._call_retrieval, query, top_k, use_reranking
            )
            chunks = retrieval_result["results"]
        except Exception as e:
            return {"answer": "Search temporarily unavailable. Please try again shortly.",
                    "sources": [], "fallback_used": True, "fallback_services": ["retrieval"]}

        # 3. LLM Generation (critical)
        try:
            llm_result = self.breakers["llm"].call(
                self._call_llm, query, chunks, history
            )
            answer = llm_result["text"]
            model_used = llm_result.get("model_used", "unknown")
        except Exception:
            return {"answer": "Generation service temporarily unavailable.",
                    "sources": chunks[:3], "fallback_used": True, "fallback_services": ["llm"]}

        # 4. Output Guardrails (non-critical)
        grounded = True
        try:
            context = "\n".join([c.get("text", "") for c in chunks[:5]])
            output_guard = self._call_guardrails_output(answer, context)
            if output_guard.get("blocked"):
                answer = "I cannot provide a safe response for this query."
            grounded = output_guard.get("grounded", True)
        except Exception:
            fallback_services.append("guardrails_output")

        return {
            "answer": answer,
            "sources": [{"text": c.get("text", "")[:200], "source": c.get("metadata", {}).get("source", "")}
                       for c in chunks[:top_k]],
            "model_used": model_used,
            "chunks_used": len(chunks),
            "cached": llm_result.get("cached", False),
            "fallback_used": len(fallback_services) > 0,
            "fallback_services": fallback_services,
            "grounded": grounded,
        }

    def _call_guardrails_input(self, query):
        r = requests.post(f"{self.settings.GUARDRAILS_URL}/check/input",
                         json={"query": query}, timeout=5)
        return r.json()

    def _call_guardrails_output(self, answer, context):
        r = requests.post(f"{self.settings.GUARDRAILS_URL}/check/output",
                         json={"answer": answer, "context": context}, timeout=10)
        return r.json()

    def _call_retrieval(self, query, top_k, use_reranking):
        r = requests.post(f"{self.settings.RETRIEVAL_URL}/retrieve",
                         json={"query": query, "top_k": top_k, "use_reranking": use_reranking},
                         timeout=15)
        return r.json()

    def _call_llm(self, query, chunks, history):
        # Build prompt (simplified — full version in Topic 5)
        context = "\n\n".join([f"[Source {i+1}]\n{c['text']}" for i, c in enumerate(chunks[:5])])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        r = requests.post(f"{self.settings.LLM_URL}/generate",
                         json={"prompt": prompt, "max_tokens": 1024, "temperature": 0.3},
                         timeout=60)
        return r.json()

    def check_all_services(self) -> dict:
        services = {
            "guardrails": self.settings.GUARDRAILS_URL,
            "retrieval": self.settings.RETRIEVAL_URL,
            "llm_gateway": self.settings.LLM_URL,
        }
        status = {}
        for name, url in services.items():
            try:
                r = requests.get(f"{url}/health", timeout=5)
                status[name] = r.status_code == 200
            except:
                status[name] = False
        return status
```

### Step 3: Dockerfile

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY src/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ .
USER appuser
EXPOSE 8000
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "--access-logfile", "-"]
```

### Step 4: Full Production `.gitlab-ci.yml`

```yaml
stages:
  - test
  - lint
  - security_scan
  - build
  - push
  - staging_deploy
  - smoke_test
  - load_test
  - approval_gate
  - prod_deploy_canary
  - monitor
  - prod_deploy_full

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-orchestrator
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://rag-staging.yourdomain.com"
  PROD_URL: "https://rag.yourdomain.com"

unit_tests:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install -r src/requirements.txt pytest pytest-cov httpx
  script:
    - pytest src/tests/ -v --cov=src --cov-report=xml

lint:
  stage: lint
  image: python:3.11-slim
  before_script:
    - pip install ruff
  script:
    - ruff check src/
    - ruff format src/ --check

lint_k8s:
  stage: lint
  image: garethr/kubeval:latest
  script:
    - kubeval k8s/*.yaml --strict

sast:
  stage: security_scan
  image: python:3.11-slim
  before_script:
    - pip install bandit safety
  script:
    - bandit -r src/ -ll
    - pip install -r src/requirements.txt && safety check

container_scan:
  stage: security_scan
  image: docker:24.0
  services: [docker:24.0-dind]
  before_script:
    - apk add --no-cache curl
    - curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
  script:
    - docker build -t ${DOCKER_IMAGE}:scan .
    - trivy image --exit-code 1 --severity CRITICAL,HIGH ${DOCKER_IMAGE}:scan

secrets_scan:
  stage: security_scan
  image: zricethezav/gitleaks:latest
  script:
    - gitleaks detect --source .

build_image:
  stage: build
  image: docker:24.0
  services: [docker:24.0-dind]
  script:
    - docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} -t ${DOCKER_IMAGE}:latest .
    - docker save ${DOCKER_IMAGE}:${DOCKER_TAG} > image.tar
  artifacts:
    paths: [image.tar]
    expire_in: 1 hour

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

deploy_staging:
  stage: staging_deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/rag-orchestrator
        rag-orchestrator=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/rag-orchestrator -n rag-staging --timeout=300s
  only: [main]

smoke_test:
  stage: smoke_test
  image: python:3.11-slim
  before_script:
    - pip install requests
  script:
    - |
      python3 -c "
      import requests, time
      URL = '${STAGING_URL}'

      # Health + Ready
      assert requests.get(f'{URL}/health').status_code == 200
      print('✓ Health')
      r = requests.get(f'{URL}/ready')
      assert r.status_code == 200
      print(f'✓ Ready: {r.json()[\"services\"]}')

      # Full RAG query
      start = time.time()
      r = requests.post(f'{URL}/ask', json={
          'query': 'How does EKS autoscaling work?',
          'top_k': 5
      }, timeout=30)
      latency = (time.time() - start) * 1000
      assert r.status_code == 200
      data = r.json()
      assert len(data['answer']) > 10
      assert len(data['sources']) > 0
      print(f'✓ Full RAG: {latency:.0f}ms, {len(data[\"answer\"])} chars')

      # Service status
      r = requests.get(f'{URL}/services/status')
      assert r.status_code == 200
      print(f'✓ Services: {r.json()}')

      print('=== ALL SMOKE TESTS PASSED ===')
      "
  only: [main]

# ─── LOAD TEST (short burst — validates no regression under load) ───
load_test:
  stage: load_test
  image: python:3.11-slim
  before_script:
    - pip install requests
  script:
    - |
      python3 -c "
      import requests, time, concurrent.futures
      URL = '${STAGING_URL}'
      CONCURRENT = 20
      TOTAL = 50

      def send_query(i):
          start = time.time()
          r = requests.post(f'{URL}/ask', json={'query': f'test query {i}', 'top_k': 3}, timeout=30)
          return time.time() - start, r.status_code

      print(f'Load test: {TOTAL} queries, {CONCURRENT} concurrent')
      with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT) as executor:
          results = list(executor.map(send_query, range(TOTAL)))

      latencies = [r[0] for r in results]
      errors = [r for r in results if r[1] != 200]
      p50 = sorted(latencies)[len(latencies)//2]
      p95 = sorted(latencies)[int(len(latencies)*0.95)]

      print(f'p50: {p50*1000:.0f}ms | p95: {p95*1000:.0f}ms | errors: {len(errors)}/{TOTAL}')
      assert p95 < 10, f'p95 too high: {p95:.1f}s'
      assert len(errors) < TOTAL * 0.05, f'Error rate too high: {len(errors)/TOTAL*100:.0f}%'
      print('✓ Load test PASSED')
      "
  only: [main]
  allow_failure: true   # Don't block deploy, but alert

approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging + smoke + load passed. Awaiting approval."
  when: manual
  allow_failure: false
  only: [main]

deploy_prod_canary:
  stage: prod_deploy_canary
  image: bitnami/kubectl:latest
  script:
    - |
      cat <<EOF | kubectl apply -f -
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: rag-orchestrator-canary
        namespace: rag-production
        labels: { app: rag-orchestrator, track: canary }
      spec:
        replicas: 1
        selector:
          matchLabels: { app: rag-orchestrator, track: canary }
        template:
          metadata:
            labels: { app: rag-orchestrator, track: canary }
          spec:
            containers:
            - name: rag-orchestrator
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports: [{ containerPort: 8000 }]
              envFrom: [{ configMapRef: { name: rag-orchestrator-config } }]
              resources:
                requests: { memory: "512Mi", cpu: "500m" }
                limits: { memory: "1Gi", cpu: "1" }
              livenessProbe: { httpGet: { path: /health, port: 8000 }, initialDelaySeconds: 10 }
              readinessProbe: { httpGet: { path: /ready, port: 8000 }, initialDelaySeconds: 5 }
      EOF
    - kubectl rollout status deployment/rag-orchestrator-canary -n rag-production --timeout=300s
  only: [main]

monitor_canary:
  stage: monitor
  image: python:3.11-slim
  before_script:
    - pip install requests
  script:
    - |
      python3 -c "
      import requests, time, sys
      FAILURES = 0
      for i in range(1, 11):
          try:
              r = requests.get('${PROD_URL}/health', timeout=10)
              if r.status_code != 200: FAILURES += 1
              else: print(f'Check {i}/10: OK')
              r2 = requests.post('${PROD_URL}/ask', json={'query': 'test', 'top_k': 2}, timeout=15)
              if r2.status_code != 200: FAILURES += 1
          except: FAILURES += 1
          if FAILURES >= 3: print('CANARY FAILED'); sys.exit(1)
          time.sleep(30)
      print('=== CANARY HEALTHY ===')
      "
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        kubectl delete deployment rag-orchestrator-canary -n rag-production --ignore-not-found
      fi
  only: [main]

deploy_prod_full:
  stage: prod_deploy_full
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/rag-orchestrator
        rag-orchestrator=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/rag-orchestrator -n rag-production --timeout=600s
    - kubectl delete deployment rag-orchestrator-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100%: ${PROD_URL} ==="
  only: [main]
```

### Step 5: Kubernetes + Ingress (MAIN EXTERNAL ENTRY POINT)

```yaml
# k8s/ingress.yaml — THIS IS THE USER-FACING URL
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-orchestrator-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"    # Streaming support
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "Strict-Transport-Security: max-age=31536000";
spec:
  tls:
  - hosts:
    - rag.yourdomain.com
    secretName: rag-orchestrator-tls
  rules:
  - host: rag.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-orchestrator
            port:
              number: 80
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-orchestrator-config
  namespace: rag-production
data:
  GUARDRAILS_URL: "http://rag-guardrails.rag-production.svc.cluster.local"
  RETRIEVAL_URL: "http://rag-retrieval-service.rag-production.svc.cluster.local"
  LLM_URL: "http://llm-gateway.rag-production.svc.cluster.local"
  EVAL_URL: "http://rag-eval-service.rag-production.svc.cluster.local"
  APP_VERSION: "latest"
```

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-orchestrator-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-orchestrator
  minReplicas: 3
  maxReplicas: 30
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
```

### Step 6: Operations

```bash
# Main user-facing URL (from ANYWHERE)
curl https://rag.yourdomain.com/health
curl https://rag.yourdomain.com/ready
curl https://rag.yourdomain.com/services/status

# Ask a question (full RAG pipeline)
curl -X POST https://rag.yourdomain.com/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does EKS autoscaling work?", "top_k": 5}'

# Streaming
curl -X POST https://rag.yourdomain.com/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain VPC networking", "stream": true}'

# Rollback
kubectl rollout undo deployment/rag-orchestrator -n rag-production
```

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Additional Sources:** AWS Well-Architected Framework, Kubernetes production best practices, Locust load testing
- **Extra added:** Complete architecture diagram, latency budget, cost breakdown, scaling matrix, blue-green deployment, graceful degradation, load testing in CI, orchestrator with circuit breakers — not in original curriculum
