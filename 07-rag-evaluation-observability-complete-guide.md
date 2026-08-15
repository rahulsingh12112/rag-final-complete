# Topic 7: RAG Evaluation & Observability — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Embedding (1), Chunking (2), Vector DB (3), Retrieval (4), Prompt (5), LLM (6)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/03-rag-systems.md + RAGAS Framework + 2026 LLM Observability

---

## 🎯 One-Liner (Interview):

> "RAG Evaluation measure karta hai ki retrieval kitna accurate hai (Recall, MRR) aur generation kitna faithful hai (groundedness, relevance) — production mein RAGAS framework + LLM-as-judge + distributed tracing se continuous monitoring karte hain taaki quality regression detect ho pipeline change ke baad."

---

## Layer 1: Kya Hai Aur Kyun Zaroori Hai?

Tumne RAG pipeline bana li — retrieval, chunking, prompting, LLM sab kaam kar raha hai. Lekin:

- **Kaise pata chalega ki answers correct hain?**
- **Kaise pata chalega ki retrieval sahi docs la raha hai?**
- **Kaise detect karoge jab quality drop hogi?**
- **Kaise debug karoge jab user complain kare "galat answer aaya"?**

Without evaluation + observability = **flying blind**. Tumhe pata nahi chalega ki system better ho raha hai ya worse.

**Key insight:** RAG ke do independently measurable components hain:
1. **Retrieval quality** — sahi documents mile? (Recall, MRR, NDCG)
2. **Generation quality** — answer faithful hai? Relevant hai? Hallucinated toh nahi? (Groundedness, Relevance, Faithfulness)

Dono independently measure karo. Agar retrieval achha hai but answer galat → prompt problem. Agar retrieval galat hai → chunking/embedding problem.

---

## Layer 2: Retrieval Metrics (Information Retrieval)

### Metric 1: Recall@K

"Top-K retrieved documents mein se kitne actually relevant the?"

```python
def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Of all relevant docs, how many did we find in top-K?"""
    top_k = set(retrieved_ids[:k])
    found = top_k & relevant_ids
    return len(found) / len(relevant_ids) if relevant_ids else 0.0

# Example:
# Retrieved top-5: [doc1, doc3, doc7, doc9, doc2]
# Relevant: {doc1, doc2, doc5}
# Recall@5 = 2/3 = 0.67 (found doc1 and doc2, missed doc5)
```

**Target:** Recall@5 >= 0.80

### Metric 2: MRR (Mean Reciprocal Rank)

"First relevant document kitne rank pe mila?"

```python
def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    """1/position of first relevant document"""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

# Example:
# Retrieved: [doc7, doc1, doc3, ...]
# Relevant: {doc1, doc5}
# MRR = 1/2 = 0.5 (first relevant at position 2)
```

**Target:** MRR >= 0.70

### Metric 3: NDCG@K (Normalized Discounted Cumulative Gain)

"Relevant docs top pe hain ya neeche?"

```python
import numpy as np

def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Position-weighted relevance — top positions matter more"""
    dcg = sum(
        (1.0 if retrieved_ids[i] in relevant_ids else 0.0) / np.log2(i + 2)
        for i in range(min(k, len(retrieved_ids)))
    )
    # Ideal: all relevant docs at top
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

**Target:** NDCG@10 >= 0.70

### Metric 4: Hit Rate

"Top-K mein koi bhi relevant document mila ya nahi?" (binary)

```python
def hit_rate(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """1 if any relevant doc in top-K, else 0"""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0
```

---

## Layer 3: Generation Metrics (Answer Quality)

### RAGAS Framework (Industry Standard):

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["How does EKS networking work?"],
    "answer": ["EKS uses VPC-CNI plugin for pod networking [Source 1]..."],
    "contexts": [["EKS integrates with VPC through CNI plugin...", "Pod IPs come from VPC subnet..."]],
    "ground_truth": ["EKS uses the VPC-CNI plugin which assigns VPC IP addresses directly to pods..."]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results)
```

### RAGAS Metrics Explained:

| Metric | Kya Measure Karta Hai | Formula (Simplified) | Target |
|--------|----------------------|---------------------|--------|
| **Faithfulness** | Answer context se supported hai? | Claims in answer that are in context / total claims | >= 0.85 |
| **Answer Relevancy** | Answer question ka response hai? | Semantic similarity(answer, question) | >= 0.80 |
| **Context Precision** | Retrieved context relevant hai? | Relevant chunks / total retrieved chunks | >= 0.75 |
| **Context Recall** | Ground truth context mein hai? | Ground truth claims found in context / total GT claims | >= 0.80 |

### Custom LLM-as-Judge (No Ground Truth Needed):

```python
class LLMJudge:
    """Use LLM to evaluate answer quality — no labeled data needed"""

    def __init__(self, llm):
        self.llm = llm

    def judge_groundedness(self, question: str, answer: str, context: str) -> dict:
        """Is the answer supported by the context?"""
        prompt = f"""You are an evaluation judge. Score the following on a scale of 1-5.

Question: {question}
Context provided: {context}
Answer given: {answer}

Score the answer on:
1. Groundedness (1-5): Is every claim in the answer supported by the context?
   1 = Completely hallucinated, 5 = Fully supported
2. Relevance (1-5): Does the answer address the question?
   1 = Irrelevant, 5 = Perfectly relevant
3. Completeness (1-5): Does the answer cover all important points from context?
   1 = Very incomplete, 5 = Comprehensive

Respond in JSON: {{"groundedness": N, "relevance": N, "completeness": N, "reasoning": "..."}}"""

        result = self.llm.generate(prompt, temperature=0)
        return json.loads(result)

    def judge_hallucination(self, answer: str, context: str) -> dict:
        """Detect specific hallucinated claims"""
        prompt = f"""List any claims in the Answer that are NOT supported by the Context.

Context: {context}
Answer: {answer}

For each unsupported claim, explain why it's not in the context.
If all claims are supported, respond with "NO HALLUCINATION".

Response:"""
        result = self.llm.generate(prompt, temperature=0)
        has_hallucination = "NO HALLUCINATION" not in result.upper()
        return {"has_hallucination": has_hallucination, "details": result}
```

---

## Layer 4: Observability & Tracing

### Distributed Tracing for RAG:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import time

# Setup
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag-service")


class ObservableRAG:
    """RAG pipeline with full tracing"""

    def query(self, user_query: str) -> dict:
        with tracer.start_as_current_span("rag.query") as span:
            span.set_attribute("query", user_query)

            # Step 1: Embedding
            with tracer.start_as_current_span("rag.embed_query") as embed_span:
                start = time.time()
                query_vector = self.embedding_service.encode(user_query)
                embed_span.set_attribute("latency_ms", (time.time() - start) * 1000)
                embed_span.set_attribute("dimensions", len(query_vector))

            # Step 2: Retrieval
            with tracer.start_as_current_span("rag.retrieve") as retrieve_span:
                start = time.time()
                chunks = self.retriever.search(query_vector, top_k=5)
                retrieve_span.set_attribute("latency_ms", (time.time() - start) * 1000)
                retrieve_span.set_attribute("chunks_retrieved", len(chunks))
                retrieve_span.set_attribute("top_score", chunks[0]["score"] if chunks else 0)

            # Step 3: Context Assembly
            with tracer.start_as_current_span("rag.assemble") as assemble_span:
                prompt = self.assembler.assemble(user_query, chunks)
                assemble_span.set_attribute("total_tokens", prompt["total_tokens"])
                assemble_span.set_attribute("chunks_used", prompt["chunks_used"])

            # Step 4: LLM Generation
            with tracer.start_as_current_span("rag.generate") as gen_span:
                start = time.time()
                answer = self.llm.generate(prompt["prompt"])
                gen_span.set_attribute("latency_ms", (time.time() - start) * 1000)
                gen_span.set_attribute("model", self.llm.model_id)
                gen_span.set_attribute("output_tokens", len(answer.split()) * 1.3)

            # Set final span attributes
            span.set_attribute("total_latency_ms", span.end_time - span.start_time)
            span.set_attribute("answer_length", len(answer))

            return {"answer": answer, "sources": chunks}
```

### Trace Visualization:

```
rag.query (total: 1850ms)
├── rag.embed_query (30ms)
│   └── dimensions: 1024
├── rag.retrieve (45ms)
│   ├── chunks_retrieved: 5
│   └── top_score: 0.89
├── rag.assemble (2ms)
│   ├── total_tokens: 4500
│   └── chunks_used: 5
└── rag.generate (1770ms)       ← Bottleneck! LLM is slowest
    ├── model: gpt-4o-mini
    └── output_tokens: 350
```

### Prometheus Metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Latency breakdown
EMBED_LATENCY = Histogram("rag_embed_latency_seconds", "Embedding latency")
RETRIEVE_LATENCY = Histogram("rag_retrieve_latency_seconds", "Retrieval latency")
GENERATE_LATENCY = Histogram("rag_generate_latency_seconds", "LLM generation latency")
TOTAL_LATENCY = Histogram("rag_total_latency_seconds", "Total RAG latency")

# Quality metrics (updated periodically by eval job)
RETRIEVAL_RECALL = Gauge("rag_retrieval_recall_at_5", "Current Recall@5")
GROUNDEDNESS_SCORE = Gauge("rag_groundedness_score", "Current groundedness")
HALLUCINATION_RATE = Gauge("rag_hallucination_rate", "Hallucination detection rate")

# Operational
QUERIES_TOTAL = Counter("rag_queries_total", "Total queries", ["status"])
CACHE_HIT_RATE = Gauge("rag_cache_hit_rate", "LLM cache hit rate")
COST_PER_QUERY = Histogram("rag_cost_per_query_dollars", "Cost per query")
```

### Grafana Dashboard (Key Panels):

```
┌─────────────────────────────────────────────────────────────┐
│ RAG SYSTEM DASHBOARD                                         │
├──────────────────┬──────────────────┬───────────────────────┤
│ Total Latency    │ Retrieval Recall │ Groundedness Score    │
│ p50: 1.2s       │ Current: 0.83    │ Current: 0.91         │
│ p99: 3.5s       │ Target: 0.80 ✅  │ Target: 0.85 ✅       │
├──────────────────┼──────────────────┼───────────────────────┤
│ Queries/min      │ Cache Hit Rate   │ Hallucination Rate    │
│ 45 q/min        │ 34%              │ 3.2%                  │
├──────────────────┼──────────────────┼───────────────────────┤
│ Latency Breakdown│ Cost (today)     │ Error Rate            │
│ Embed: 30ms     │ $12.50           │ 0.5%                  │
│ Retrieve: 45ms  │ Budget: $50/day  │                       │
│ LLM: 1100ms    │                  │                       │
└──────────────────┴──────────────────┴───────────────────────┘
```

---

## Layer 5: Evaluation Pipeline (Automated)

### Continuous Evaluation Architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                 EVALUATION PIPELINE                            │
│                                                              │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Test Set │───▶│ RAG API  │───▶│ Evaluator│───▶│ Alert  │ │
│  │ (100+   │    │ (staging)│    │ (RAGAS + │    │ (Slack/│ │
│  │ queries)│    │          │    │ LLM-judge)│   │ PD)    │ │
│  └─────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │                              │                       │
│       │              ┌───────────────┘                       │
│       ▼              ▼                                       │
│  ┌──────────────────────────────┐                           │
│  │ Metrics Store (Prometheus)    │                           │
│  │ - Recall@5 trend             │                           │
│  │ - Groundedness trend         │                           │
│  │ - Regression detection       │                           │
│  └──────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

### Evaluation Script:

```python
#!/usr/bin/env python3
"""
Automated RAG evaluation — runs daily or on deploy.
Detects quality regressions.
"""
import json
import requests
import numpy as np
from datetime import datetime


class RAGEvaluator:
    def __init__(self, api_url: str, llm_judge_url: str):
        self.api_url = api_url
        self.judge_url = llm_judge_url

    def evaluate(self, test_set_path: str) -> dict:
        with open(test_set_path) as f:
            test_set = json.load(f)

        results = {
            "retrieval": {"recall_at_5": [], "mrr": [], "hit_rate": []},
            "generation": {"groundedness": [], "relevance": [], "hallucination": []},
            "latency": {"total_ms": [], "ttft_ms": []},
            "timestamp": datetime.now().isoformat()
        }

        for item in test_set:
            # Call RAG API
            resp = requests.post(f"{self.api_url}/ask", json={
                "query": item["query"], "top_k": 5
            }, timeout=60)

            if resp.status_code != 200:
                continue

            data = resp.json()
            answer = data["answer"]
            sources = data["sources"]
            metadata = data["metadata"]

            # Retrieval metrics (if ground truth available)
            if "relevant_doc_ids" in item:
                retrieved_ids = [s.get("id", "") for s in sources]
                relevant = set(item["relevant_doc_ids"])
                results["retrieval"]["recall_at_5"].append(
                    len(set(retrieved_ids[:5]) & relevant) / len(relevant) if relevant else 0
                )
                results["retrieval"]["hit_rate"].append(
                    1.0 if set(retrieved_ids[:5]) & relevant else 0.0
                )

            # Generation metrics (LLM-as-judge)
            context = "\n".join([s.get("text", "") for s in sources])
            judge_resp = requests.post(f"{self.judge_url}/judge", json={
                "question": item["query"],
                "answer": answer,
                "context": context
            }, timeout=30)

            if judge_resp.status_code == 200:
                scores = judge_resp.json()
                results["generation"]["groundedness"].append(scores.get("groundedness", 0) / 5.0)
                results["generation"]["relevance"].append(scores.get("relevance", 0) / 5.0)
                results["generation"]["hallucination"].append(1 if scores.get("has_hallucination") else 0)

            # Latency
            results["latency"]["total_ms"].append(metadata.get("latency_ms", 0))

        # Aggregate
        summary = {
            "recall_at_5": np.mean(results["retrieval"]["recall_at_5"]) if results["retrieval"]["recall_at_5"] else 0,
            "hit_rate": np.mean(results["retrieval"]["hit_rate"]) if results["retrieval"]["hit_rate"] else 0,
            "groundedness": np.mean(results["generation"]["groundedness"]) if results["generation"]["groundedness"] else 0,
            "relevance": np.mean(results["generation"]["relevance"]) if results["generation"]["relevance"] else 0,
            "hallucination_rate": np.mean(results["generation"]["hallucination"]) if results["generation"]["hallucination"] else 0,
            "latency_p50_ms": np.median(results["latency"]["total_ms"]) if results["latency"]["total_ms"] else 0,
            "latency_p99_ms": np.percentile(results["latency"]["total_ms"], 99) if results["latency"]["total_ms"] else 0,
            "queries_evaluated": len(test_set),
            "timestamp": results["timestamp"]
        }

        return summary

    def check_regression(self, current: dict, baseline_path: str) -> list:
        """Compare current metrics against baseline — detect regressions"""
        with open(baseline_path) as f:
            baseline = json.load(f)

        regressions = []
        thresholds = {
            "recall_at_5": 0.05,       # Alert if drops > 5%
            "groundedness": 0.05,
            "relevance": 0.05,
            "hallucination_rate": -0.03  # Alert if increases > 3%
        }

        for metric, threshold in thresholds.items():
            current_val = current.get(metric, 0)
            baseline_val = baseline.get(metric, 0)
            diff = current_val - baseline_val

            if metric == "hallucination_rate":
                if diff > abs(threshold):  # Hallucination increased
                    regressions.append(f"{metric}: {baseline_val:.3f} → {current_val:.3f} (+{diff:.3f})")
            else:
                if diff < -threshold:  # Quality decreased
                    regressions.append(f"{metric}: {baseline_val:.3f} → {current_val:.3f} ({diff:.3f})")

        return regressions
```

---

## Layer 6: Debugging Failed Queries

### Query Debugging Pipeline:

```python
class RAGDebugger:
    """Debug why a specific query gave bad results"""

    def debug_query(self, query: str) -> dict:
        """Full diagnostic for a single query"""
        report = {"query": query, "issues": []}

        # Step 1: Check embedding
        embedding = self.embed(query)
        report["embedding_norm"] = np.linalg.norm(embedding)
        if report["embedding_norm"] < 0.9 or report["embedding_norm"] > 1.1:
            report["issues"].append("Embedding not normalized")

        # Step 2: Check retrieval
        chunks = self.retrieve(query, top_k=10)
        report["top_score"] = chunks[0]["score"] if chunks else 0
        report["chunks_count"] = len(chunks)

        if report["top_score"] < 0.5:
            report["issues"].append(f"Low retrieval score ({report['top_score']:.3f}) — query may not match any docs")

        # Step 3: Check context relevance
        for i, chunk in enumerate(chunks[:5]):
            relevance = self.judge_relevance(query, chunk["text"])
            if relevance < 0.3:
                report["issues"].append(f"Chunk {i+1} irrelevant (score {relevance:.2f}): {chunk['text'][:50]}...")

        # Step 4: Check generation
        answer = self.generate(query, chunks)
        groundedness = self.check_groundedness(answer, chunks)
        if groundedness < 0.7:
            report["issues"].append(f"Answer poorly grounded ({groundedness:.2f}) — possible hallucination")

        # Step 5: Diagnosis
        if not report["issues"]:
            report["diagnosis"] = "No issues detected"
        elif "Low retrieval score" in str(report["issues"]):
            report["diagnosis"] = "RETRIEVAL PROBLEM: Query doesn't match indexed content. Check chunking/embedding."
        elif "irrelevant" in str(report["issues"]):
            report["diagnosis"] = "RANKING PROBLEM: Wrong chunks ranked high. Consider reranking."
        elif "hallucination" in str(report["issues"]):
            report["diagnosis"] = "GENERATION PROBLEM: LLM ignoring context. Check prompt/system instruction."

        return report
```

---

## Layer 7: Production Pitfalls

### Pitfall 1: No Baseline Metrics

Changes kiye (new model, new chunking) but pehle ki quality measure nahi ki thi. Improvement ya regression — pata nahi.

**Fix:** Day 1 se evaluation run karo. Store baseline. Compare after every change.

### Pitfall 2: Evaluating on Wrong Data

Test set mein easy queries hain. Production mein hard queries aati hain. Metrics look good, users complain.

**Fix:** Continuously add real user queries (that failed) to test set. Sample from production logs.

### Pitfall 3: Not Separating Retrieval vs Generation Issues

"Answer galat hai" → kyun galat? Retrieval ne wrong docs laye? Ya LLM ne hallucinate kiya?

**Fix:** Measure both independently. Retrieval metrics + generation metrics alag rakhko.

### Pitfall 4: Expensive Evaluation

RAGAS uses LLM calls for judging. 1000 queries × 4 metrics = 4000 LLM calls = expensive.

**Fix:** Evaluate on sample (100-200 queries). Use cheap model for judging (GPT-4o-mini). Run weekly, not per-query.

### Pitfall 5: No Alerting on Drift

Metrics slowly degrade over weeks. Nobody notices until users start complaining.

**Fix:** Prometheus alert rules: "If recall@5 < 0.75 for 24h → alert. If hallucination_rate > 5% → critical alert."

---

## Layer 8: Trade-offs & Decisions

### Evaluation Approaches:

| Approach | Accuracy | Cost | Speed | When |
|----------|----------|------|-------|------|
| Human evaluation | Highest | Very expensive | Days | Gold standard, quarterly |
| RAGAS (automated) | High | Medium (LLM calls) | Hours | Weekly, per-deploy |
| LLM-as-judge | Good | Medium | Minutes | Per-deploy, staging |
| Heuristic checks (citations, length) | Low | Free | Instant | Every request, real-time |

### Observability Stack:

| Tool | Purpose | Cost |
|------|---------|------|
| OpenTelemetry + Jaeger | Distributed tracing | Free (self-hosted) |
| Prometheus + Grafana | Metrics + dashboards | Free (self-hosted) |
| LangSmith (LangChain) | LLM-specific tracing | $$ (managed) |
| Arize Phoenix | LLM observability | Free (open-source) |
| Weights & Biases | Experiment tracking | $ (managed) |

---

## Layer 9: Interview Ready

### 2-Line Answer (Screening):

> "RAG evaluation has two dimensions: retrieval quality (Recall@K, MRR, NDCG) and generation quality (faithfulness, groundedness, hallucination rate). In production, we use automated RAGAS evaluation + LLM-as-judge + distributed tracing for continuous monitoring and regression detection."

### 5-Min Answer (Technical Round):

> Above + RAGAS metrics explained, LLM-as-judge pattern (no labeled data needed), tracing architecture (OpenTelemetry spans per pipeline stage), Prometheus metrics + Grafana dashboards, alerting on drift, test set management (continuously updated from production failures).

### 10-Min Deep Dive (System Design):

> Above + full evaluation pipeline (daily CronJob, baseline comparison, regression alerts), debugging workflow (isolate retrieval vs generation issues), cost of evaluation (sampling strategies), human-in-the-loop feedback loop, A/B testing different pipeline configs, observability stack selection, capacity planning using trace data.

### Follow-up Questions:

**Q: "User complain karta hai 'galat answer' — debug kaise karoge?"**
A: (1) Pull trace for that request. (2) Check retrieval: did correct docs get retrieved? If no → retrieval problem. (3) If yes, check prompt: was context properly assembled? (4) If yes → LLM hallucinated despite good context. Fix: stronger grounding prompt, better model, reranking.

**Q: "Evaluation automate karna hai — kaise set up karoge?"**
A: CronJob runs daily. Sends 100 test queries to staging. RAGAS + LLM-judge scores each. Compares against baseline. If regression > 5% → Slack alert + blocks next deploy. Results stored in Prometheus for trend analysis.

**Q: "Ground truth data nahi hai — evaluation kaise karoge?"**
A: LLM-as-judge (no ground truth needed). Also: human annotators label 50-100 queries initially. Then use LLM-judge for continuous monitoring. Periodically validate LLM-judge agrees with human labels (meta-evaluation).

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| Retrieval metrics (Recall, MRR, NDCG, Hit Rate) | ✅ |
| Generation metrics (Faithfulness, Groundedness, Relevance) | ✅ |
| RAGAS framework | ✅ |
| LLM-as-judge pattern | ✅ |
| Distributed tracing (OpenTelemetry) | ✅ |
| Prometheus metrics + Grafana | ✅ |
| Automated evaluation pipeline | ✅ |
| Regression detection | ✅ |
| Query debugging workflow | ✅ |
| Production pitfalls (5) | ✅ |
| Trade-offs (evaluation approaches) | ✅ |
| Interview answers | ✅ |

**Topic 7: RAG Evaluation & Observability — DONE.**

---
## Layer 12: GitLab CI/CD + ArgoCD — RAG Evaluation & Observability Service Deployment

Evaluation service — jo RAG quality monitor karta hai (retrieval accuracy, groundedness, hallucination) + observability stack — production mein deploy with full pipeline.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
rag-eval-service/
├── src/
│   ├── app.py                  # FastAPI — Evaluation + Judge API
│   ├── evaluator.py            # RAGAS + custom metrics
│   ├── judge.py                # LLM-as-judge
│   ├── debugger.py             # Query debugging
│   ├── config.py
│   ├── requirements.txt
│   └── tests/
│       └── test_evaluator.py
├── eval/
│   ├── test_set.json
│   ├── baseline_metrics.json
│   └── run_eval.py             # CronJob script
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml            # External access
│   ├── configmap.yaml
│   ├── eval-cronjob.yaml       # Daily evaluation
│   ├── hpa.yaml
│   └── observability/
│       ├── prometheus-rules.yaml
│       └── grafana-dashboard.json
├── Dockerfile
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: Evaluation API (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, generate_latest
from starlette.responses import PlainTextResponse
from evaluator import RAGEvaluator
from judge import LLMJudge
from debugger import RAGDebugger
from config import settings
import os

app = FastAPI(title="RAG Evaluation Service", version="1.0.0")

# Exposed metrics (scraped by Prometheus)
RECALL_GAUGE = Gauge("rag_eval_recall_at_5", "Current Recall@5")
GROUNDEDNESS_GAUGE = Gauge("rag_eval_groundedness", "Current groundedness score")
HALLUCINATION_GAUGE = Gauge("rag_eval_hallucination_rate", "Current hallucination rate")
EVAL_RUNS = Counter("rag_eval_runs_total", "Total evaluation runs", ["status"])

evaluator = RAGEvaluator(rag_api_url=settings.RAG_API_URL, llm_url=settings.LLM_SERVICE_URL)
judge = LLMJudge(llm_url=settings.LLM_SERVICE_URL)
debugger = RAGDebugger(rag_api_url=settings.RAG_API_URL)


class JudgeRequest(BaseModel):
    question: str
    answer: str
    context: str


class DebugRequest(BaseModel):
    query: str


class EvalTriggerRequest(BaseModel):
    test_set_path: str = "eval/test_set.json"
    sample_size: int | None = None


@app.get("/health")
def health():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    return {"status": "ready"}


@app.post("/judge")
def judge_answer(request: JudgeRequest):
    """Judge a single answer for groundedness/relevance"""
    return judge.judge_groundedness(request.question, request.answer, request.context)


@app.post("/debug")
def debug_query(request: DebugRequest):
    """Full diagnostic for a failing query"""
    return debugger.debug_query(request.query)


@app.post("/evaluate")
def run_evaluation(request: EvalTriggerRequest):
    """Trigger full evaluation run"""
    try:
        results = evaluator.evaluate(request.test_set_path, sample_size=request.sample_size)
        # Update Prometheus gauges
        RECALL_GAUGE.set(results.get("recall_at_5", 0))
        GROUNDEDNESS_GAUGE.set(results.get("groundedness", 0))
        HALLUCINATION_GAUGE.set(results.get("hallucination_rate", 0))
        EVAL_RUNS.labels(status="success").inc()
        return results
    except Exception as e:
        EVAL_RUNS.labels(status="failed").inc()
        raise HTTPException(500, str(e))


@app.get("/metrics/latest")
def latest_metrics():
    """Get latest evaluation metrics"""
    return {
        "recall_at_5": RECALL_GAUGE._value.get(),
        "groundedness": GROUNDEDNESS_GAUGE._value.get(),
        "hallucination_rate": HALLUCINATION_GAUGE._value.get(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
```

### Step 2: Dockerfile

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
COPY eval/ eval/
USER appuser
EXPOSE 8000
CMD ["gunicorn", "app:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "300", "--access-logfile", "-"]
```

### Step 3: Full Production `.gitlab-ci.yml`

```yaml
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
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-eval-service
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://rag-eval-staging.yourdomain.com"
  PROD_URL: "https://rag-eval.yourdomain.com"

# ─────────────── TEST ───────────────
unit_tests:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install -r src/requirements.txt pytest pytest-cov httpx
  script:
    - pytest src/tests/ -v --cov=src --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'

# ─────────────── LINT ───────────────
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

# ─────────────── SECURITY SCAN ───────────────
sast:
  stage: security_scan
  image: python:3.11-slim
  before_script:
    - pip install bandit safety
  script:
    - bandit -r src/ -ll
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
    - gitleaks detect --source .

# ─────────────── BUILD ───────────────
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

# ─────────────── PUSH ───────────────
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

# ─────────────── STAGING DEPLOY ───────────────
deploy_staging:
  stage: staging_deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: ${STAGING_URL}
  script:
    - kubectl set image deployment/rag-eval-service
        rag-eval-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/rag-eval-service -n rag-staging --timeout=300s
  only: [main]

# ─────────────── SMOKE TEST ───────────────
smoke_test:
  stage: smoke_test
  image: python:3.11-slim
  before_script:
    - pip install requests
  script:
    - |
      python3 -c "
      import requests
      URL = '${STAGING_URL}'
      assert requests.get(f'{URL}/health').status_code == 200
      print('✓ Health OK')
      assert requests.get(f'{URL}/ready').status_code == 200
      print('✓ Ready OK')

      # Judge endpoint
      r = requests.post(f'{URL}/judge', json={
          'question': 'What is EKS?',
          'answer': 'EKS is a managed Kubernetes service.',
          'context': 'Amazon EKS is a managed Kubernetes service on AWS.'
      })
      assert r.status_code == 200
      print(f'✓ Judge OK: {r.json()}')

      # Metrics
      assert requests.get(f'{URL}/metrics').status_code == 200
      print('✓ Metrics OK')
      print('=== ALL SMOKE TESTS PASSED ===')
      "
  only: [main]

# ─────────────── APPROVAL GATE ───────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging passed. Awaiting approval."
  when: manual
  allow_failure: false
  only: [main]

# ─────────────── CANARY 10% ───────────────
deploy_prod_canary:
  stage: prod_deploy_canary
  image: bitnami/kubectl:latest
  script:
    - |
      cat <<EOF | kubectl apply -f -
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: rag-eval-service-canary
        namespace: rag-production
        labels: { app: rag-eval-service, track: canary }
      spec:
        replicas: 1
        selector:
          matchLabels: { app: rag-eval-service, track: canary }
        template:
          metadata:
            labels: { app: rag-eval-service, track: canary }
          spec:
            containers:
            - name: rag-eval-service
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports: [{ containerPort: 8000 }]
              envFrom: [{ configMapRef: { name: rag-eval-config } }]
              resources:
                requests: { memory: "512Mi", cpu: "250m" }
                limits: { memory: "1Gi", cpu: "500m" }
              livenessProbe: { httpGet: { path: /health, port: 8000 }, initialDelaySeconds: 10 }
              readinessProbe: { httpGet: { path: /ready, port: 8000 }, initialDelaySeconds: 5 }
      EOF
    - kubectl rollout status deployment/rag-eval-service-canary -n rag-production --timeout=300s
  only: [main]

# ─────────────── MONITOR ───────────────
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
          except: FAILURES += 1
          if FAILURES >= 3: print('CANARY FAILED'); sys.exit(1)
          time.sleep(30)
      print('=== CANARY HEALTHY ===')
      "
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        kubectl delete deployment rag-eval-service-canary -n rag-production --ignore-not-found
      fi
  only: [main]

# ─────────────── FULL 100% ───────────────
deploy_prod_full:
  stage: prod_deploy_full
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/rag-eval-service
        rag-eval-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/rag-eval-service -n rag-production --timeout=600s
    - kubectl delete deployment rag-eval-service-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100%: ${PROD_URL} ==="
  only: [main]
```

### Step 4: Evaluation CronJob (`k8s/eval-cronjob.yaml`)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rag-daily-evaluation
  namespace: rag-production
spec:
  schedule: "0 6 * * *"            # Daily 6 AM
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 1800   # Max 30 min
      template:
        spec:
          containers:
          - name: evaluator
            image: registry.gitlab.com/yourgroup/rag-eval-service:latest
            command: ["python", "eval/run_eval.py"]
            env:
            - name: RAG_API_URL
              value: "http://rag-context-service.rag-production.svc.cluster.local"
            - name: EVAL_SERVICE_URL
              value: "http://rag-eval-service.rag-production.svc.cluster.local"
            - name: SLACK_WEBHOOK
              valueFrom:
                secretKeyRef:
                  name: rag-eval-secrets
                  key: slack-webhook
          restartPolicy: OnFailure
```

### Step 5: Kubernetes + Ingress

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-eval-service
  namespace: rag-production
spec:
  replicas: 2
  selector:
    matchLabels: { app: rag-eval-service }
  template:
    metadata:
      labels: { app: rag-eval-service }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: rag-eval-service
        image: registry.gitlab.com/yourgroup/rag-eval-service:latest
        ports: [{ containerPort: 8000 }]
        envFrom: [{ configMapRef: { name: rag-eval-config } }]
        resources:
          requests: { memory: "512Mi", cpu: "250m" }
          limits: { memory: "1Gi", cpu: "500m" }
```

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-eval-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "50"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts: [rag-eval.yourdomain.com]
    secretName: rag-eval-tls
  rules:
  - host: rag-eval.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service: { name: rag-eval-service, port: { number: 80 } }
```

```yaml
# k8s/observability/prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: rag-quality-alerts
  namespace: rag-production
spec:
  groups:
  - name: rag-quality
    rules:
    - alert: RAGRetrievalQualityDrop
      expr: rag_eval_recall_at_5 < 0.75
      for: 24h
      labels: { severity: warning }
      annotations:
        summary: "RAG Recall@5 dropped below 0.75"

    - alert: RAGHallucinationHigh
      expr: rag_eval_hallucination_rate > 0.08
      for: 6h
      labels: { severity: critical }
      annotations:
        summary: "Hallucination rate > 8% — immediate attention needed"

    - alert: RAGGroundednessLow
      expr: rag_eval_groundedness < 0.80
      for: 12h
      labels: { severity: warning }
      annotations:
        summary: "Groundedness score below 0.80"
```

### Step 6: Operations

```bash
# External access
curl https://rag-eval.yourdomain.com/health
curl https://rag-eval.yourdomain.com/metrics/latest

# Judge a response
curl -X POST https://rag-eval.yourdomain.com/judge \
  -H "Content-Type: application/json" \
  -d '{"question": "What is EKS?", "answer": "EKS is...", "context": "Amazon EKS..."}'

# Debug a failing query
curl -X POST https://rag-eval.yourdomain.com/debug \
  -d '{"query": "Error EKS-AUTH-403"}'

# Trigger full evaluation
curl -X POST https://rag-eval.yourdomain.com/evaluate \
  -d '{"test_set_path": "eval/test_set.json"}'

# Check CronJob
kubectl get cronjobs -n rag-production
kubectl logs job/rag-daily-evaluation-xxxxx -n rag-production
```

### Architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│              EVALUATION & OBSERVABILITY ARCHITECTURE              │
│                                                                  │
│  External: https://rag-eval.yourdomain.com                       │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────┐     ┌────────────────────┐            │
│  │ Eval Service (API)   │     │ Daily CronJob      │            │
│  │ - /judge             │     │ - Runs 100 queries │            │
│  │ - /debug             │     │ - Updates gauges   │            │
│  │ - /evaluate          │     │ - Alerts on regress│            │
│  │ - /metrics           │     └────────┬───────────┘            │
│  └──────────┬───────────┘              │                        │
│             │                          ▼                        │
│             ▼                  ┌───────────────┐                │
│  ┌──────────────────┐         │ Prometheus     │                │
│  │ RAG Context API  │         │ (scrape /metrics)│               │
│  │ (runs queries)   │         └───────┬───────┘                │
│  └──────────────────┘                 │                        │
│                                       ▼                        │
│                               ┌───────────────┐                │
│                               │ Grafana       │                │
│                               │ (dashboards)  │                │
│                               └───────┬───────┘                │
│                                       │                        │
│                                       ▼                        │
│                               ┌───────────────┐                │
│                               │ Alertmanager  │                │
│                               │ → Slack/PD    │                │
│                               └───────────────┘                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Additional Sources:** RAGAS framework docs, OpenTelemetry documentation, Stanford "Lost in the Middle" paper, Arize Phoenix docs
- **Extra added:** LLM-as-judge, query debugger, regression detection, Prometheus alerting rules, evaluation CronJob, observability stack, production pipeline — not in original curriculum
