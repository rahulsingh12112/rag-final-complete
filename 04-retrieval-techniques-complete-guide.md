# Topic 4: Retrieval Techniques — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Embedding (Topic 1), Chunking (Topic 2), Vector DB (Topic 3)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/03-rag-systems.md + 2026 Hybrid Search Reference (DigitalApplied, markaicode)

---

## 🎯 One-Liner (Interview):

> "Production RAG retrieval is a 3-stage pipeline: dual first-stage retrieval (dense vector + sparse BM25), rank fusion via RRF, and second-stage cross-encoder reranking — neither dense nor sparse alone wins across all query types."

---

## Layer 1: Retrieval Kya Hai Aur Kyun Complex Hai?

Embedding topic mein tumne seekha ki text ko vectors mein convert karte hain. Vector DB mein tumne seekha ki kaise store aur search karte hain. Ab retrieval = **"query aane pe sahi documents kaise dhundhein?"**

Simple case: Query embed karo → Vector DB mein cosine search → Top-5 return. **Yeh kaam karta hai 70% cases mein.** But 30% cases mein fail hota hai:

```
Query: "Error code EKS-AUTH-403 resolution"
Dense retrieval result: Generic EKS authentication troubleshooting doc ❌
Expected: Specific doc mentioning that exact error code ✅

Query: "Infosys Q3 2024 revenue"
Dense retrieval result: Generic Indian IT earnings doc ❌
Expected: Exact Infosys Q3 2024 document ✅
```

**Kyun fail hota hai?** Dense vectors **meaning** capture karte hain, exact **keywords** nahi. Rare terms (error codes, product names, IDs) ko embedding model properly represent nahi karta kyunki training mein kam dekha hai.

Isi liye production RAG mein **multiple retrieval techniques** combine karte hain.

---

## Layer 2: Retrieval Techniques (All Types)

### Technique 1: Basic Dense Similarity Search

Sabse simple — query embed karo, nearest vectors dhundho.

```python
def dense_search(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(query)
    results = vector_db.search(
        collection_name="documents",
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    return results
```

**Strengths:** Paraphrases samajhta hai ("car repair" ↔ "automobile maintenance" match hoga)
**Weakness:** Exact rare terms miss karta hai (error codes, IDs, proper nouns)
**Use when:** General semantic Q&A, well-trained domain

---

### Technique 2: Filtered Search (Metadata-Based)

Vector search + metadata filter combine karo. Pehle filter laga ke scope narrow karo, phir usme vector search.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Example: Only search networking docs from 2024+
results = vector_db.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="networking")),
            FieldCondition(key="year", range=Range(gte=2024))
        ]
    ),
    limit=5
)
```

**Strengths:** Dramatically reduces search space, improves relevance, faster
**Weakness:** User ko pata hona chahiye filter values (ya auto-detect karo)
**Use when:** Multi-tenant systems, categorized docs, time-sensitive content

**Production pattern — Auto-detect filters from query:**
```python
def extract_filters(query: str) -> dict:
    """Use LLM or rules to extract filter intent from query"""
    # "Show me EKS networking docs from last month"
    # → category: "networking", service: "EKS", date: last 30 days
    # Can use a small LLM call or regex patterns
    pass
```

---

### Technique 3: Sparse Retrieval (BM25 / Keyword Search)

Traditional keyword matching. Token frequency based scoring — exact term match matters.

```python
# BM25 scoring (conceptual)
# score(doc, query) = Σ IDF(term) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × docLen/avgDocLen))

# In practice, use Elasticsearch or built-in sparse search:
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

results = es.search(
    index="documents",
    query={
        "match": {
            "text": "EKS-AUTH-403 error resolution"
        }
    },
    size=50
)
```

**BM25 Parameters:**
- **k1 (1.2-2.0):** Term frequency saturation. Higher = more weight to repeated terms.
- **b (0.75):** Document length normalization. 1.0 = full normalize, 0 = no normalize.

**Strengths:** Exact keyword match (error codes, IDs, names), very fast (inverted index), no GPU needed
**Weakness:** No semantic understanding ("car" ≠ "automobile")
**Use when:** Exact-match critical queries, product codes, technical identifiers

---

### Technique 4: Hybrid Search (Dense + Sparse) — PRODUCTION STANDARD

**The most important retrieval technique for production RAG.**

Run BOTH dense and sparse retrieval in parallel, then combine results.

```python
class HybridRetriever:
    """Production hybrid retrieval: Dense + BM25 + RRF Fusion"""

    def __init__(self, vector_db, bm25_index, embedding_model):
        self.vector_db = vector_db
        self.bm25 = bm25_index
        self.embedder = embedding_model

    def search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> list:
        # Stage 1: Parallel retrieval
        # Dense search (semantic)
        query_emb = self.embedder.encode(query)
        dense_results = self.vector_db.search(query_vector=query_emb, limit=50)

        # Sparse search (keyword)
        sparse_results = self.bm25.search(query, top_k=50)

        # Stage 2: RRF Fusion
        fused = self.reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=rrf_k
        )

        return fused[:top_k]

    def reciprocal_rank_fusion(self, result_lists: list, k: int = 60) -> list:
        """
        RRF formula: score(d) = Σ 1/(k + rank)
        - Operates on RANKS only, not scores
        - Solves score incompatibility (BM25 unbounded vs cosine [-1,1])
        - k=60 is production default (Elasticsearch standard)
        """
        scores = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc["id"]
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0, "doc": doc}
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        # Sort by fused score
        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked]
```

**Why RRF and not simple score averaging?**

BM25 scores are unbounded integers (0 to 50+). Cosine similarity is bounded [-1, 1]. If you do `0.5 * bm25_score + 0.5 * cosine_score`, BM25 will ALWAYS dominate because its numbers are bigger. RRF fixes this by ignoring scores entirely — only rank positions matter.

```
RRF example (k=60):
Doc A: Dense rank=1, Sparse rank=3
  RRF score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

Doc B: Dense rank=5, Sparse rank=1
  RRF score = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

Doc C: Dense rank=2, Sparse rank=50
  RRF score = 1/(60+2) + 1/(60+50) = 0.0161 + 0.0091 = 0.0252

Result: A > B > C
(Doc appearing high in BOTH lists wins)
```

**Benchmark Results (WANDS e-commerce dataset, 2025):**

| Method | NDCG Score |
|--------|-----------|
| BM25 alone | 0.698 |
| Dense (KNN) alone | 0.695 |
| Hybrid (RRF basic) | 0.707 (+1.3%) |
| Hybrid + field boosting | 0.750 (+7.4%) |

**Hybrid adds ~30% relevance improvement in production RAG systems.**

---

### Technique 5: Re-ranking (Cross-Encoder Second Stage)

First-stage retrieval (dense/sparse/hybrid) is **fast but rough**. Re-ranking is **slow but precise**.

**Key difference:** Bi-encoder (embedding model) encodes query and document separately, then compares vectors. Cross-encoder sees query AND document together — much more accurate but can't scan millions of docs.

```
Bi-encoder (first stage):     Encode("query") → vec1
                               Encode("doc") → vec2
                               Compare(vec1, vec2) → score
                               Speed: 10,000 docs/sec

Cross-encoder (second stage):  Encode("query [SEP] doc") → single score
                               Sees both together, understands relationships
                               Speed: 100 docs/sec (10x slower)
```

**Architecture:**
```
Query → [First stage: Top-100 candidates via hybrid search] → [Second stage: Rerank top-100 with cross-encoder] → Top-5 to LLM
```

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Cross-encoder reranking for precision improvement"""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Models (speed vs quality):
        - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good (production default)
        - cross-encoder/ms-marco-MiniLM-L-12-v2: Slower, better
        - BAAI/bge-reranker-v2-m3: Multilingual, excellent

        Managed APIs:
        - Cohere Rerank 3.5: Production standard, 100+ languages
        - Voyage rerank-2.5: 32K context, instruction-following (2025 SOTA)
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """Rerank documents for a query"""
        # Create (query, doc) pairs
        pairs = [[query, doc["text"]] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [{"doc": doc, "rerank_score": float(score)}
                for doc, score in scored_docs[:top_k]]


# Usage in RAG pipeline:
class RAGWithReranking:
    def retrieve(self, query: str, top_k: int = 5):
        # Stage 1: Get 50-100 candidates (fast, rough)
        candidates = self.hybrid_retriever.search(query, top_k=100)

        # Stage 2: Rerank to top-5 (slow, precise)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return reranked
```

**Impact:** Cross-encoder reranking cuts retrieval failures by ~67% in production systems.

**Managed Reranking APIs (Production):**

| Provider | Model | Context | Latency | Strength |
|----------|-------|---------|---------|----------|
| Cohere | Rerank 3.5 | 4K tokens | ~50ms/100 docs | Multi-language, tables, code |
| Voyage AI | rerank-2.5 | 32K tokens | ~60ms/100 docs | Instruction-following (2025 SOTA) |
| AWS Bedrock | Cohere Rerank | 4K tokens | ~70ms/100 docs | AWS-native integration |

**Voyage rerank-2.5 instruction-following (NEW 2025):**
```python
# You can STEER relevance judgment with instructions!
voyage_client.rerank(
    query="EKS pricing",
    documents=candidates,
    instruction="Prefer results about enterprise pricing, not free tier. Prioritize official AWS documentation over blog posts."
)
```

---

### Technique 6: Query Expansion / Transformation

Sometimes user query too short ya vague hoti hai. Transform karo before retrieval:

```python
class QueryExpander:
    """Expand/transform query for better retrieval"""

    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query: str) -> list:
        """Generate multiple query variants"""
        prompt = f"""Generate 3 different versions of this search query to improve retrieval.
Original: {query}
Variants:"""
        variants = self.llm.generate(prompt)
        # Returns: ["original query", "variant 1", "variant 2", "variant 3"]
        return [query] + variants

    def hypothetical_answer(self, query: str) -> str:
        """HyDE: Generate hypothetical answer, use IT for retrieval"""
        prompt = f"Write a short, detailed answer to: {query}"
        hypothesis = self.llm.generate(prompt)
        # Embed this hypothesis (closer to documents than short query)
        return hypothesis
```

**HyDE (Hypothetical Document Embeddings):**
```
Normal:  "What is EKS pricing?" → embed this short query → search
HyDE:    "What is EKS pricing?" → LLM generates hypothetical answer →
         "EKS pricing includes $0.10/hr for control plane..." →
         embed THIS (closer to actual docs) → search (better results)
```

**When to use:** Short/vague queries, complex multi-part questions. Adds latency (extra LLM call).

---

### Technique 7: Multi-Hop Retrieval

Complex questions need multiple rounds of retrieval:

```python
class MultiHopRetriever:
    """For questions that need info from multiple documents"""

    def retrieve(self, query: str, max_hops: int = 3) -> list:
        all_docs = []
        current_query = query

        for hop in range(max_hops):
            # Retrieve for current query
            docs = self.hybrid_search(current_query, top_k=3)
            all_docs.extend(docs)

            # Generate follow-up query based on what we found
            follow_up = self.generate_follow_up(query, docs)
            if not follow_up or follow_up == current_query:
                break  # No more info needed
            current_query = follow_up

        return self.deduplicate(all_docs)

    def generate_follow_up(self, original_query, retrieved_docs):
        """LLM decides: do we need more info? What to search next?"""
        prompt = f"""Original question: {original_query}
Information found so far: {retrieved_docs}
Is more information needed? If yes, what should we search for next?
If no, respond with 'DONE'."""
        return self.llm.generate(prompt)
```

**Example:**
```
Q: "What's the total cost of running EKS with ALB and RDS in us-east-1?"

Hop 1: Search "EKS pricing us-east-1" → Gets control plane cost
Hop 2: Search "ALB pricing us-east-1" → Gets load balancer cost
Hop 3: Search "RDS pricing us-east-1" → Gets database cost
Final: LLM combines all three → Complete cost breakdown
```

---

## Layer 3: The Complete Production Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCTION RETRIEVAL PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌──────────────────┐
│ Query Processing │  ← Query expansion, filter extraction
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Dense  │ │ Sparse │  ← Stage 1: Parallel retrieval
│ (ANN)  │ │ (BM25) │     Dense: semantic matching
│ Top-50 │ │ Top-50 │     Sparse: keyword matching
└───┬────┘ └───┬────┘
    │           │
    ▼           ▼
┌─────────────────────┐
│   RRF Fusion (k=60) │  ← Stage 2: Combine ranked lists
│   Top-100 merged    │     Rank-based, score-agnostic
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Metadata Filtering  │  ← Optional: scope reduction
│ (category, date,    │     After fusion, before reranking
│  tenant_id)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Cross-Encoder       │  ← Stage 3: Precision reranking
│ Reranking           │     Top-100 → Top-5
│ (ms-marco / Cohere) │     67% fewer retrieval failures
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Top-K Results       │  → To LLM for answer generation
│ (with scores +      │
│  metadata)          │
└─────────────────────┘
```

---

## Layer 4: Vendor Implementation (How to Set Up Hybrid)

### Qdrant (Server-side RRF, v1.10+):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery, Fusion, Prefetch,
    SparseVector, NamedSparseVector
)

client = QdrantClient(url="http://localhost:6333")

# Native hybrid search with RRF
results = client.query_points(
    collection_name="documents",
    prefetch=[
        # Dense retrieval
        Prefetch(query=dense_query_vector, limit=50),
        # Sparse retrieval
        Prefetch(
            query=SparseVector(indices=[1, 5, 100], values=[0.5, 0.3, 0.8]),
            using="sparse",
            limit=50
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # Server-side RRF
    limit=20
)
```

### Weaviate (Built-in Hybrid):

```python
result = client.query.get(
    "Document", ["content", "category"]
).with_hybrid(
    query="EKS networking",
    alpha=0.7,  # 0.7 = 70% dense, 30% sparse
    fusion_type="rankedFusion"  # Explicitly set RRF (v1.24+ changed default)
).with_limit(20).do()
```

### Pinecone (Alpha-weighted):

```python
# Pinecone stores dense + sparse in same index
results = index.query(
    vector=dense_embedding,        # Dense component
    sparse_vector={                 # Sparse component
        "indices": [1, 5, 100],
        "values": [0.5, 0.3, 0.8]
    },
    top_k=20,
    include_metadata=True
)
# Alpha weighting: start with alpha=0.75 (majority dense)
```

### Elasticsearch (RRF — Enterprise plan, or client-side):

```python
# Native RRF (Enterprise plan required)
response = es.search(
    index="documents",
    retriever={
        "rrf": {
            "retrievers": [
                {"standard": {"query": {"match": {"text": query}}}},     # BM25
                {"knn": {"field": "embedding", "query_vector": emb, "num_candidates": 50}}  # Dense
            ],
            "rank_constant": 60,
            "rank_window_size": 100
        }
    }
)

# Client-side RRF (free tier — use ranx library):
from ranx import fuse, Run
fused = fuse(runs=[bm25_run, dense_run], method="rrf", params={"k": 60})
```

---

## Layer 5: Infra & Architecture

### Deployment Pattern (Hybrid Retrieval):

```yaml
# You need TWO indexes:
# 1. Vector DB (dense) — Qdrant/Pinecone
# 2. Text search (sparse) — Elasticsearch/OpenSearch

# Option A: Separate services
services:
  qdrant:        # Dense vectors
    image: qdrant/qdrant
    ports: ["6333:6333"]

  elasticsearch:  # BM25 sparse
    image: elasticsearch:8.12.0
    ports: ["9200:9200"]

  rag-api:       # Orchestrates both
    image: your-rag-service
    environment:
      VECTOR_DB_URL: "http://qdrant:6333"
      BM25_URL: "http://elasticsearch:9200"

# Option B: Single DB with hybrid support (simpler)
# Qdrant v1.10+ or Weaviate — both support hybrid natively
# No separate Elasticsearch needed
```

### Latency Budget:

```
Total retrieval target: < 200ms

Breakdown:
├─ Query embedding:           30ms  (Bedrock API call)
├─ Dense search (Qdrant):     15ms  (p99)
├─ Sparse search (BM25):      10ms  (p99)
├─ RRF fusion:                 1ms  (in-memory rank merge)
├─ Cross-encoder reranking:   80ms  (100 docs × Cohere API)
├─ Network overhead:          20ms
└─ Total:                    ~156ms ✅ (within budget)
```

### Scaling Retrieval:

| Component | Scaling Strategy |
|-----------|-----------------|
| Dense search | Vector DB replicas (horizontal) |
| Sparse search | Elasticsearch shards |
| RRF fusion | Stateless, runs in RAG service |
| Reranking | API call (managed) or GPU pod |
| Query expansion | LLM call (adds 200-500ms, use async) |

---

## Layer 6: Trade-offs & Decisions

### When to Use What:

| Query Type | Best Technique | Why |
|-----------|---------------|-----|
| Semantic / paraphrase | Dense only | "How to fix pods" ↔ "Kubernetes pod troubleshooting" |
| Exact keyword / code | Sparse (BM25) only | "Error EKS-AUTH-403" |
| Mixed intent (most real queries) | Hybrid (Dense + Sparse + RRF) | Best of both |
| High-precision needed | Hybrid + Reranking | 67% fewer failures |
| Complex multi-part question | Multi-hop | Needs info from multiple docs |
| Short/vague query | Query expansion + Hybrid | "EKS cost?" → expanded |

### Reranker: Self-hosted vs API?

| Factor | Self-hosted (ms-marco) | Managed (Cohere/Voyage) |
|--------|----------------------|------------------------|
| Latency | ~20ms/100 docs (GPU) | ~50-80ms/100 docs (API call) |
| Cost | GPU instance fixed cost | Per-query pricing |
| Quality | Good | Better (Voyage > Cohere > ms-marco) |
| Context window | 512 tokens | 4K-32K tokens |
| Maintenance | You manage | Zero ops |
| Production recommendation | High-traffic (>10K queries/day) | Low-medium traffic |

### Do You Even Need Hybrid?

```
"My RAG is working fine with just dense search" → Don't add complexity
"Users searching for specific error codes/IDs and not finding them" → ADD hybrid
"Retrieval quality good but precision could be better" → ADD reranking
"Complex questions getting partial answers" → ADD multi-hop
```

**Rule of thumb:** Start simple (dense only) → Add hybrid when you find keyword-failure cases → Add reranking when precision matters → Add multi-hop for complex questions.

---

## Layer 7: Production Pitfalls

### Pitfall 1: BM25 Score Dominating in Naive Hybrid

BM25 scores are unbounded (0-50+), cosine is bounded [-1,1]. Simple weighted average = BM25 always wins.

**Fix:** Use RRF (rank-based, ignores scores). Or normalize scores before combining.

### Pitfall 2: Reranking Too Many Documents

Cross-encoder on 1000 documents = 5-10 seconds latency. Unusable.

**Fix:** First-stage retrieval returns top-100 max. Rerank only those. Never apply cross-encoder to full index.

### Pitfall 3: Weaviate Default Changed (v1.24)

Weaviate v1.24 silently changed default fusion from RRF to "Relative Score Fusion". Existing systems upgrading will get different results without code changes.

**Fix:** Always explicitly set `fusion_type="rankedFusion"` if you depend on RRF behavior.

### Pitfall 4: Sparse Index Not Updated

Documents re-indexed in vector DB but forgot to update Elasticsearch/BM25 index. Hybrid search returns stale sparse results.

**Fix:** Single pipeline that updates BOTH indexes atomically. Same trigger, same documents.

### Pitfall 5: Query Expansion Adding Noise

LLM-generated query variants sometimes drift from original intent. Bad variants → irrelevant retrieval.

**Fix:** Always include original query in retrieval set. Limit to 2-3 variants. Validate variants are on-topic before using.

### Pitfall 6: No Evaluation Baseline

Added hybrid + reranking but never measured if it actually improved results.

**Fix:** Create 50+ test queries with known-relevant documents. Measure Recall@5, MRR, NDCG before and after each change.

---

## Layer 8: Evaluation Metrics

```python
import numpy as np

class RetrievalEvaluator:
    """Measure retrieval quality"""

    def precision_at_k(self, retrieved_ids: list, relevant_ids: set, k: int) -> float:
        """Of top-K retrieved, how many are relevant?"""
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len(set(top_k) & relevant_ids)
        return relevant_in_top_k / k

    def recall_at_k(self, retrieved_ids: list, relevant_ids: set, k: int) -> float:
        """Of all relevant docs, how many did we find in top-K?"""
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len(set(top_k) & relevant_ids)
        return relevant_in_top_k / len(relevant_ids) if relevant_ids else 0

    def mrr(self, retrieved_ids: list, relevant_ids: set) -> float:
        """Mean Reciprocal Rank — where does first relevant doc appear?"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def ndcg_at_k(self, retrieved_ids: list, relevant_ids: set, k: int) -> float:
        """Normalized Discounted Cumulative Gain — position-weighted relevance"""
        dcg = sum(
            (1.0 if retrieved_ids[i] in relevant_ids else 0.0) / np.log2(i + 2)
            for i in range(min(k, len(retrieved_ids)))
        )
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

**Production targets:**
- Recall@5 > 0.8 (80% of relevant docs found in top-5)
- MRR > 0.7 (first relevant doc usually in top 1-2)
- NDCG@10 > 0.7

---

## Layer 9: Interview Ready

### 2-Line Answer (Screening):

> "Production RAG uses a 3-stage retrieval pipeline: dual first-stage retrieval combining dense vectors and BM25 for complementary coverage, RRF fusion to merge ranked lists without score incompatibility, and cross-encoder reranking for precision — reducing retrieval failures by 67%."

### 5-Min Answer (Technical Round):

> Above + why dense alone fails (exact terms, rare keywords), RRF formula and why ranks > scores, cross-encoder vs bi-encoder difference, latency budget breakdown, vendor implementations (Qdrant native RRF, Weaviate hybrid, Elasticsearch), evaluation metrics (Recall@K, MRR, NDCG).

### 10-Min Deep Dive (System Design):

> Above + complete pipeline architecture with latency numbers, scaling each component independently, HyDE and query expansion patterns, multi-hop for complex queries, instruction-following rerankers (Voyage 2.5), production pitfalls (score incompatibility, Weaviate default change, dual-index sync), cost-performance tradeoffs (self-hosted reranker vs API), evaluation harness setup, A/B testing retrieval changes.

### Expected Follow-up Questions:

**Q: "Dense search accha results de raha hai, hybrid add karne ka kya fayda?"**
A: Measure first. Create test set with exact-keyword queries (error codes, product IDs). If dense retrieval misses those → hybrid will fix it (30% relevance improvement per benchmarks). If all queries are semantic → maybe hybrid not needed.

**Q: "Reranker add karna hai but latency budget tight hai (100ms total)"**
A: Self-hosted ms-marco model on GPU = ~20ms for 50 docs. Or reduce first-stage to top-30, rerank those. Or async reranking with cached results for repeated queries.

**Q: "RRF ka k parameter tune karna ho toh kaise?"**
A: k=60 is default, works for most cases. Lower k (30-40) = more weight to top-ranked docs (better for top-1 precision). Higher k (80-100) = more weight to consistent appearance across lists (better for recall). Test on your query set with NDCG.

**Q: "100 languages support karna hai — retrieval kaise karoge?"**
A: Multilingual embedding model (bge-m3 or multilingual-e5-large) for dense. Language-specific BM25 analyzers in Elasticsearch for sparse. Cross-encoder: Cohere Rerank 3.5 (100+ languages) or bge-reranker-v2-m3 (multilingual).

**Q: "Retrieval evaluation automate kaise karoge?"**
A: LLM-as-judge pattern. For each query, ask LLM: "Is this retrieved doc relevant to this query? Score 0-2." Build evaluation set of 100+ queries. Run weekly. Alert if metrics drop >5%.

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| Dense similarity search | ✅ |
| Filtered (metadata) search | ✅ |
| Sparse/BM25 retrieval | ✅ |
| Hybrid search (dense + sparse) | ✅ Deep |
| RRF fusion (formula, why, tuning) | ✅ Deep |
| Cross-encoder reranking | ✅ Deep |
| Managed rerankers (Cohere, Voyage) | ✅ |
| Query expansion / HyDE | ✅ |
| Multi-hop retrieval | ✅ |
| Complete pipeline architecture | ✅ |
| Vendor implementations (Qdrant, Weaviate, Pinecone, ES) | ✅ |
| Infra deployment (latency budget, scaling) | ✅ |
| Trade-offs & decision matrix | ✅ |
| Evaluation metrics (Precision, Recall, MRR, NDCG) | ✅ |
| 6 production pitfalls | ✅ |
| Interview answers (all levels) | ✅ |
| Follow-up Q&A | ✅ |

**Topic 4: Retrieval Techniques — DONE.**

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **2026 Reference:** DigitalApplied Hybrid Search BM25+Vector+Reranking Reference 2026, markaicode.com production benchmarks
- **Extra added:** RRF implementation code, Voyage rerank-2.5 instruction-following, vendor-specific hybrid implementations, latency budget breakdown, evaluation code, production pitfalls (Weaviate default change, score incompatibility), multi-hop retrieval, query expansion patterns — not in original curriculum


---

## Layer 12: GitLab CI/CD + ArgoCD — RAG API Deployment + Retrieval Test Automation

RAG API (jo retrieval karta hai) ko deploy karna + GitLab mein retrieval accuracy tests automate karna.

### Project Structure:

```
rag-api/
├── src/
│   ├── app.py               # FastAPI RAG service
│   ├── retriever.py         # Hybrid retrieval logic
│   ├── reranker.py          # Cross-encoder reranking
│   ├── requirements.txt
├── tests/
│   ├── test_retrieval.py    # Retrieval accuracy tests
│   └── test_set.json        # 50+ test queries with expected docs
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
├── argocd/
│   └── application.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: Retrieval Test Automation (`tests/test_retrieval.py`)

```python
import json
import requests
import pytest

RAG_API_URL = "http://localhost:8000"  # or staging URL in CI

with open("tests/test_set.json") as f:
    TEST_SET = json.load(f)


class TestRetrievalQuality:
    """Automated retrieval accuracy tests — runs in GitLab CI"""

    def test_recall_at_5(self):
        """At least 80% of relevant docs should be in top-5"""
        recalls = []
        for item in TEST_SET:
            response = requests.post(f"{RAG_API_URL}/retrieve", json={
                "query": item["query"], "top_k": 5
            })
            retrieved_ids = [r["id"] for r in response.json()["results"]]
            relevant_ids = set(item["relevant_doc_ids"])
            recall = len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)
            recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls)
        assert avg_recall >= 0.8, f"Recall@5 = {avg_recall:.3f}, expected >= 0.8"

    def test_mrr(self):
        """First relevant doc should be in top-2 on average"""
        mrrs = []
        for item in TEST_SET:
            response = requests.post(f"{RAG_API_URL}/retrieve", json={
                "query": item["query"], "top_k": 10
            })
            retrieved_ids = [r["id"] for r in response.json()["results"]]
            relevant_ids = set(item["relevant_doc_ids"])

            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    mrrs.append(1.0 / (i + 1))
                    break
            else:
                mrrs.append(0.0)

        avg_mrr = sum(mrrs) / len(mrrs)
        assert avg_mrr >= 0.7, f"MRR = {avg_mrr:.3f}, expected >= 0.7"

    def test_latency_p99(self):
        """p99 latency should be under 200ms"""
        import time
        latencies = []
        for item in TEST_SET[:20]:  # Sample 20 queries
            start = time.time()
            requests.post(f"{RAG_API_URL}/retrieve", json={"query": item["query"], "top_k": 5})
            latencies.append((time.time() - start) * 1000)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        assert p99 < 200, f"p99 latency = {p99:.0f}ms, expected < 200ms"
```

**`tests/test_set.json` (sample):**
```json
[
  {
    "query": "How does VPC-CNI work in EKS?",
    "relevant_doc_ids": ["doc-1", "doc-7"]
  },
  {
    "query": "S3 durability percentage",
    "relevant_doc_ids": ["doc-6"]
  }
]
```

### Step 2: GitLab CI (Build + Retrieval Tests)

```yaml
stages:
  - test-unit
  - build
  - push
  - deploy
  - test-retrieval   # Post-deploy quality gate

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-api
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}

test-unit:
  stage: test-unit
  image: python:3.11-slim
  script:
    - pip install -r src/requirements.txt pytest
    - pytest tests/unit/ -v

build:
  stage: build
  image: docker:24.0
  services: [docker:24.0-dind]
  script:
    - docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .

push:
  stage: push
  image: docker:24.0
  services: [docker:24.0-dind]
  script:
    - echo ${CI_REGISTRY_PASSWORD} | docker login -u ${CI_REGISTRY_USER} --password-stdin ${CI_REGISTRY}
    - docker push ${DOCKER_IMAGE}:${DOCKER_TAG}
    - docker push ${DOCKER_IMAGE}:latest
  only: [main]

deploy:
  stage: deploy
  image: alpine:3.18
  script:
    - apk add --no-cache git sed
    - sed -i "s|image:.*rag-api:.*|image: ${DOCKER_IMAGE}:${DOCKER_TAG}|" k8s/deployment.yaml
    - git config user.email "ci@gitlab.com"
    - git config user.name "GitLab CI"
    - git add k8s/deployment.yaml
    - git commit -m "deploy: rag-api ${DOCKER_TAG}" || true
    - git push origin main
  only: [main]

# ━━━ POST-DEPLOY: Retrieval Quality Gate ━━━
test-retrieval:
  stage: test-retrieval
  image: python:3.11-slim
  script:
    - pip install requests pytest
    - sleep 30  # Wait for rollout
    - pytest tests/test_retrieval.py -v --tb=short
  only: [main]
  allow_failure: false  # BLOCKS if retrieval quality drops
```

### Step 3: K8s Deployment + ArgoCD

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    spec:
      containers:
      - name: rag-api
        image: registry.gitlab.com/yourgroup/rag-api:latest
        ports: [{containerPort: 8000}]
        envFrom: [{configMapRef: {name: rag-api-config}}]
        resources:
          requests: {memory: "2Gi", cpu: "1"}
          limits: {memory: "4Gi", cpu: "2"}
---
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rag-api
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/rag-api.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-system
  syncPolicy:
    automated: {prune: true, selfHeal: true}
```

### Flow:

```
Code push → GitLab CI:
  1. Unit tests
  2. Build Docker image
  3. Push to registry
  4. Update k8s manifest → ArgoCD deploys
  5. POST-DEPLOY: Run retrieval accuracy tests
     - If Recall@5 < 0.8 → Pipeline FAILS → Alert → Rollback
     - If pass → Deployment confirmed ✅
```

### Key Insight:

**Retrieval tests as quality gates** — agar naya code deploy hone ke baad retrieval quality drop karti hai, pipeline fail hogi. Yeh ensures ki production mein kabhi degraded retrieval nahi jayega.

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **2026 Reference:** DigitalApplied Hybrid Search Reference, markaicode.com benchmarks
- **Extra added:** RRF code, Voyage rerank-2.5, vendor implementations, latency budget, evaluation code, production pitfalls, GitLab CI/CD + ArgoCD + retrieval quality gates — not in original curriculum
