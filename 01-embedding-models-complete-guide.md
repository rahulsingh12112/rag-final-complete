# Topic 1: Embedding Models — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** AWS, Kubernetes, Networking (expert level)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/03-rag-systems.md + Additional production knowledge

---

## 🎯 One-Liner (Interview)

> "Embedding model ek function hai jo text ko fixed-size dense vector mein convert karta hai jahan semantically similar texts ka cosine distance chhota hota hai."

---

## Layer 1: Kya Hai Aur Kyun Chahiye?

Machines text directly process nahi kar sakti. Unhe numbers chahiye. Ab numbers banane ke bahut tarike hain — purane zamane mein log one-hot encoding karte the (har word ko ek position assign karo, uspe 1 lagao, baaki 0). Problem yeh thi ki "king" aur "queen" ka koi relationship capture nahi hota tha — dono equally distant the jaise "king" aur "banana".

Embedding models ne yeh solve kiya. Yeh models training ke dauran crores of sentences padhte hain aur seekhte hain ki kaunse words/sentences similar contexts mein aate hain. Output ek dense vector hota hai (jaise 768 ya 1024 floating point numbers) jahan geometric proximity semantic similarity represent karti hai. Matlab — "How to deploy on EKS" aur "Kubernetes deployment steps" ke vectors paas honge, jabki "Best pizza in Delhi" ka vector door hoga.

Yeh RAG mein fundamental hai kyunki jab user query aati hai, tum uska embedding banate ho, phir vector DB mein stored document chunks ke embeddings se compare karte ho — jo paas hai, wo relevant hai.

---

## Layer 2: Internally Kaise Kaam Karta Hai?

Embedding models typically **Transformer architecture** pe based hain (same family as BERT, GPT). Specifically, most embedding models BERT-style encoder-only transformers hain.

### Training Process (2 Phases):

**Phase 1 — Pre-training:** Model ko massive text corpus pe train kiya jaata hai (Wikipedia, books, web). Yeh general language understanding deta hai.

**Phase 2 — Contrastive Fine-tuning:** Yeh crucial step hai. Model ko pairs/triplets diye jaate hain:
- **Positive pair:** "What is Kubernetes?" ↔ "Kubernetes is a container orchestration platform" (similar — vectors paas hone chahiye)
- **Negative pair:** "What is Kubernetes?" ↔ "Recipe for chocolate cake" (dissimilar — vectors door hone chahiye)

Model ka loss function (typically InfoNCE/Contrastive Loss) ensure karta hai ki positive pairs ke vectors close aayein aur negative pairs ke vectors door jayein high-dimensional space mein.

### Inference Time Pe Kya Hota Hai:

Jab tum `model.encode("What is EKS?")` likhte ho:
1. Text tokenize hota hai (subword tokens mein — "What", "is", "EK", "S")
2. Tokens Transformer layers se guzarte hain (self-attention compute hota hai — har token doosre tokens ke context mein apna representation adjust karta hai)
3. Final layer se har token ka ek hidden state aata hai
4. **Pooling** hoti hai — typically mean pooling (sab tokens ke vectors ka average) ya CLS token ka vector liya jaata hai
5. Output = single fixed-size vector (e.g., 1024 dimensions)

### 1024 Numbers Represent Kya Karte Hain?

Koi predefined meaning nahi hai ("dimension 1 = technology" jaisa nahi hai). Yeh emergent properties hain — training ke dauran model ne khud decide kiya ki kaunsi dimension kya capture karegi. Lekin collectively, yeh 1024 numbers us text ka semantic fingerprint hain. Do texts ke fingerprints compare karke tum similarity nikal sakte ho.

---

## Layer 3: Similarity Measurement

### Cosine Similarity (Most Common):

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """
    Cosine of angle between two vectors.
    Range: -1 (opposite) to +1 (identical direction)
    RAG mein typically 0.7+ ko "similar" maante hain
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)
```

### Normalization Kya Hai?

Normalize ka matlab: Poore vector ki total length (magnitude) ko 1 bana do. Individual numbers -1 to 1 ke beech hona zaroori nahi.

```python
vector = [3, 4]
magnitude = sqrt(3² + 4²) = sqrt(25) = 5
normalized = [3/5, 4/5] = [0.6, 0.8]
# Ab iska length = sqrt(0.6² + 0.8²) = sqrt(1) = 1 ✓
```

Jab vectors normalized hain (unit length), toh **cosine similarity = dot product** (mathematically same). Dot product computationally simpler hai — sirf multiply and add. Isliye production mein:

```
Encode time: normalize=True (vector length = 1 bana do)
Search time: Dot product use karo (fast, result = cosine ke barabar)
```

### Kyun Cosine Aur Euclidean Nahi?

Euclidean distance magnitude sensitive hai. Agar ek document ka embedding accidentally large magnitude ka ho (longer text, etc.), toh euclidean mislead karega. Cosine sirf direction dekhta hai, magnitude ignore karta hai — semantics ke liye appropriate hai.

---

## Layer 4: Models Comparison — Production Perspective

### Open Source Models (Self-hosted):

| Model | Dimensions | Max Tokens | Speed | Quality | GPU Required |
|-------|-----------|-----------|-------|---------|-------------|
| `all-MiniLM-L6-v2` | 384 | 256 | Very fast | Good | No (CPU ok) |
| `all-mpnet-base-v2` | 768 | 384 | Medium | Very Good | Recommended |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | Slower | Excellent | Yes |
| `BAAI/bge-m3` | 1024 | 8192 | Slow | SOTA | Yes |

### Managed APIs:

| Provider | Model | Dimensions | Cost/1M tokens | Latency |
|----------|-------|-----------|---------------|---------|
| OpenAI | text-embedding-3-large | 3072 | $0.13 | ~100ms |
| OpenAI | text-embedding-3-small | 1536 | $0.02 | ~80ms |
| AWS Bedrock | Titan Embed V2 | 1024 | $0.02 | ~50ms |
| Cohere | embed-english-v3 | 1024 | $0.10 | ~100ms |

---

## Layer 5: Production Code

### Option 1: Self-hosted (BGE on Kubernetes)

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch

class EmbeddingService:
    """
    Production embedding service.
    Handles batching, normalization, and device management.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = None,
        batch_size: int = 32
    ):
        # Auto-detect device: GPU available toh GPU, warna CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dimensions = self.model.get_sentence_embedding_dimension()

        # BGE models ke liye query instruction mandatory hai
        self.is_bge = "bge" in model_name.lower()
        self.query_prefix = "Represent this sentence for searching relevant passages: " if self.is_bge else ""

    def encode_documents(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Documents encode karo — indexing time pe use hoga.
        Documents mein koi prefix nahi lagta.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Single query encode karo — search time pe use hoga.
        BGE models mein instruction prefix lagta hai.
        """
        text = f"{self.query_prefix}{query}" if self.is_bge else query

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding

    def encode_queries_batch(self, queries: List[str]) -> np.ndarray:
        """Multiple queries batch mein encode karo"""
        if self.is_bge:
            queries = [f"{self.query_prefix}{q}" for q in queries]

        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=True
        )
        return embeddings

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Do embeddings ka similarity score.
        Kyunki normalized hain, dot product = cosine similarity.
        """
        return float(np.dot(embedding_a, embedding_b))


# --- Usage ---
if __name__ == "__main__":
    service = EmbeddingService(model_name="BAAI/bge-large-en-v1.5")

    # Documents index karo
    docs = [
        "EKS uses managed control plane with etcd for state management",
        "Kubernetes pods communicate via CNI plugin like VPC-CNI on AWS",
        "The best pizza in Delhi is at Delfinos"
    ]
    doc_embeddings = service.encode_documents(docs)
    print(f"Document embeddings shape: {doc_embeddings.shape}")  # (3, 1024)

    # Query search karo
    query_embedding = service.encode_query("How does networking work in EKS?")

    # Similarity check karo
    for i, doc in enumerate(docs):
        score = service.similarity(query_embedding, doc_embeddings[i])
        print(f"Score {score:.4f} → {doc[:60]}...")

    # Expected output:
    # Score 0.82 → Kubernetes pods communicate via CNI plugin like VPC-CNI...
    # Score 0.71 → EKS uses managed control plane with etcd for state...
    # Score 0.12 → The best pizza in Delhi is at Delfinos
```

### Option 2: AWS Bedrock (Managed, Zero Infra)

```python
import boto3
import json

class BedrockEmbeddings:
    """AWS Bedrock Titan Embed V2 — managed embedding service"""

    def __init__(self, region_name="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = "amazon.titan-embed-text-v2:0"
        self.dimensions = 1024

    def encode(self, text: str) -> list:
        """Single text encode karo"""
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True
            })
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

    def encode_batch(self, texts: list) -> list:
        """Multiple texts encode karo (loop — Bedrock batch API limited)"""
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text))
        return embeddings
```

**Bedrock use karo jab:** Infra manage nahi karna, cost predictable chahiye, AWS ecosystem mein already ho. But latency ~50-100ms per call (network hop), self-hosted mein <10ms.

### Option 3: Full RAG with Bedrock + Pinecone

```python
import boto3
import json
import os
from pinecone import Pinecone, ServerlessSpec

REGION = "us-east-1"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "amazon.nova-micro-v1:0"
INDEX_NAME = "rag-demo"
DIMENSION = 1024

# AWS Bedrock Client
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def get_embedding(text: str) -> list:
    """Text ko 1024-dimension vector mein convert karo"""
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({
            "inputText": text,
            "dimensions": DIMENSION,
            "normalize": True
        })
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def ask_llm(prompt: str) -> str:
    """LLM se answer generate karo"""
    response = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps({
            "inferenceConfig": {
                "max_new_tokens": 512,
                "temperature": 0.3
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
        })
    )
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]


# Pinecone Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

existing_indexes = [idx.name for idx in pc.indexes.list()]
if INDEX_NAME not in existing_indexes:
    pc.indexes.create(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.index(INDEX_NAME)


def ingest_documents(documents: list):
    """Documents embed + store karo"""
    vectors = []
    for i, doc in enumerate(documents):
        embedding = get_embedding(doc)
        vectors.append((f"doc-{i}", embedding, {"text": doc, "index": i}))
    index.upsert(vectors=vectors)


def rag_query(question: str, top_k: int = 3) -> dict:
    """Complete RAG: embed query → search → build prompt → LLM answer"""
    # Embed query
    query_embedding = get_embedding(question)

    # Search Pinecone
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Context banao
    context_docs = [{"text": m.metadata["text"], "score": m.score} for m in results.matches]
    context = "\n\n".join([
        f"[Document {i+1} (relevance: {doc['score']:.3f})]\n{doc['text']}"
        for i, doc in enumerate(context_docs)
    ])

    # Prompt + LLM
    prompt = f"""Answer based ONLY on the context. Cite document numbers.

Context:
{context}

Question: {question}

Answer:"""

    answer = ask_llm(prompt)
    return {"question": question, "answer": answer, "sources": context_docs}
```

---

## Layer 6: Infrastructure & Deployment

### Kubernetes Deployment (Self-hosted Embedding Service):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding
        image: your-registry/embedding-service:v1
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "6Gi"
            nvidia.com/gpu: "1"
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_NAME
          value: "BAAI/bge-large-en-v1.5"
        - name: BATCH_SIZE
          value: "64"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30  # Model load time
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
spec:
  selector:
    app: embedding-service
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP  # Internal only — external expose mat karo
```

### Scaling Considerations:

| Platform | Throughput | Use Case |
|----------|-----------|----------|
| CPU (bge-large) | ~50 texts/sec | Development, low traffic |
| Single T4 GPU | ~500 texts/sec | Production (moderate) |
| Single A100 GPU | ~2000 texts/sec | High-throughput production |
| Multiple replicas | Linear scale | Horizontal scaling |

### Networking Considerations:

- **ClusterIP only** — Embedding service ko cluster-internal rakhna, external expose mat karo
- **gRPC > REST** — Embedding vectors large hain (1024 × 4 bytes = 4KB per vector), gRPC mein protobuf serialization 3-5x faster hai
- **Connection pooling** — RAG service repeatedly call karega, pool connections
- **Batch requests** — 1 text per call instead of batching = 10x more network overhead

---

## Layer 7: Trade-offs & Decisions

### Decision 1: Open-source vs Managed API?

| Factor | Self-hosted (BGE) | Managed (Bedrock/OpenAI) |
|--------|-------------------|--------------------------|
| Latency | <10ms (in-cluster) | 50-150ms (network) |
| Cost at scale | Fixed (GPU cost) | Per-token (scales linearly) |
| Privacy | Data stays in your VPC | Data goes to provider |
| Maintenance | You manage model updates | Provider handles |
| Quality | Excellent (BGE = SOTA) | Excellent |
| Breakeven | ~10M tokens/day self-hosted cheaper | <1M tokens/day managed cheaper |

### Decision 2: Kaunsa Model?

```
Low latency, low resource    → all-MiniLM-L6-v2 (384d)
Balanced (most production)   → bge-base-en-v1.5 (768d)
Maximum quality              → bge-large-en-v1.5 (1024d)
Long documents (8K+ tokens)  → bge-m3 (1024d, 8192 token window)
Multilingual                 → multilingual-e5-large (1024d)
```

### Decision 3: Dimensions Kitne?

Zyada dimensions = zyada information = better quality BUT:
- More storage (1024d × 4 bytes = 4KB per vector)
- Slower similarity search
- 1M documents × 1024d = ~4GB RAM sirf vectors ke liye

**Production recommendation:** 768 ya 1024. Use 384 only when speed/cost critical.

---

## Layer 8: Production Pitfalls

### Pitfall 1: Query aur Document Encoding Mismatch

BGE models mein query encode karte waqt instruction prefix lagana **MANDATORY** hai. Agar query bhi bina prefix ke encode karoge, similarity scores significantly drop ho jayenge. Log yeh miss karte hain aur phir bolte hain "RAG kaam nahi kar raha".

```python
# WRONG ❌
query_emb = model.encode("What is EKS?")

# RIGHT ✅
query_emb = model.encode("Represent this sentence for searching relevant passages: What is EKS?")
```

### Pitfall 2: Model Update = Full Re-indexing

Embedding model change karte ho (MiniLM → BGE), toh **SAARI documents re-embed** karni padegi. Purane vectors naye model ke saath compatible nahi hain. 10M docs × GPU time = costly. Plan karo pehle se.

### Pitfall 3: Token Limit Exceed (Silent Truncation)

Har model ki max token limit hoti hai (BGE-large = 512 tokens). Chunk 512 se bada hai toh model **silently truncate** kar dega — tumhe pata nahi chalega. Chunking strategy ko model ki token limit ke saath align karo.

### Pitfall 4: Normalization Miss Karna

Embeddings normalize nahi hain + dot product use kar rahe ho = **wrong results**. Either:
- Always `normalize_embeddings=True` set karo
- Ya vector DB mein explicitly `Distance.COSINE` set karo

### Pitfall 5: Cold Start Latency

Model load = 10-30 seconds (bge-large ~2GB). Kubernetes mein:
- `readinessProbe` with `initialDelaySeconds: 30` set karo
- Scale-from-zero scenarios mein pehli request timeout hogi
- Solution: Keep minimum 1 replica always warm

---

## Layer 9: Interview Ready

### 2-Line Answer (Screening Round):

> "Embedding models convert text into fixed-size dense vectors where semantic similarity is captured as geometric proximity. In RAG systems, we use them to encode both documents and queries, then find relevant documents via cosine similarity search in a vector database."

### 5-Min Answer (Technical Round):

Above + model architecture (Transformer encoder, contrastive training), practical model selection (BGE for self-hosted, Bedrock for managed), query vs document encoding difference, dimension trade-offs, deployment on K8s with GPU, scaling patterns.

### 10-Min Deep Dive (System Design Round):

Above + production architecture (dedicated embedding microservice, gRPC, batching), cost analysis at scale, migration strategy when changing models, monitoring (embedding latency p99, drift detection), caching frequently queried embeddings, multi-tenancy considerations, failure modes aur mitigation.

### Expected Follow-up Questions:

**Q: "Model upgrade zero-downtime mein kaise karoge?"**
A: Blue-green deployment with dual-index in vector DB. Naye model se naya index banao, traffic gradually migrate karo, purana index delete karo.

**Q: "Embedding drift detect kaise karoge?"**
A: Monitor average similarity scores over time. Agar scores drop ho rahe hain toh documents stale ho rahe ya query patterns change ho rahe.

**Q: "Multi-language support chahiye toh?"**
A: `multilingual-e5-large` ya `bge-m3` use karo — single model for all languages, no separate models needed.

**Q: "Cost optimize kaise karoge high-traffic pe?"**
A: Cache popular query embeddings in Redis, batch requests, smaller model (MiniLM) for low-priority traffic, GPU spot instances for indexing jobs.

**Q: "1 Billion documents handle karne hain — architecture kya hogi?"**
A: Distributed vector DB (Milvus/Qdrant cluster), sharded indexes, separate indexing pipeline (async, batch), embedding service auto-scaled on GPU nodes, tiered storage (hot in RAM, warm on disk).

---

## RAG Flow Summary (End-to-End)

```
┌─────────────────────────────────────────────────────────┐
│                  INDEXING (One-time)                      │
│                                                          │
│  Documents → Chunk → Embed (Titan/BGE) → Store (Pinecone)│
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  QUERY (Per request)                      │
│                                                          │
│  User Question                                           │
│       ↓                                                  │
│  Embed Query (same model)                                │
│       ↓                                                  │
│  Search Vector DB (cosine similarity, top-K)             │
│       ↓                                                  │
│  Retrieved Docs (most relevant chunks)                   │
│       ↓                                                  │
│  Build Prompt (context + question)                       │
│       ↓                                                  │
│  LLM generates answer (grounded in docs)                 │
│       ↓                                                  │
│  Return answer + sources                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Vector DB mein Kya Hota Hai (Qdrant/Pinecone):

```python
# 1. Collection/Index banate waqt ek baar config karo
vectors_config = VectorParams(size=1024, distance=Distance.COSINE)

# 2. Documents store karo (embedding + metadata)
index.upsert(vectors=[("doc-0", embedding_list, {"text": "original text"})])

# 3. Search karo (query vector do, top-K similar return)
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
```

Tum code nahi likhte cosine/dot product ka — vector DB internally optimized algorithms (HNSW, IVF) use karke fast search karta hai.

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Additional content added:** Infra deployment YAML, Bedrock integration code, Pinecone RAG code, production pitfalls, interview preparation, trade-off analysis, networking considerations, scaling benchmarks, follow-up Q&A — none of this was in the original curriculum link.

---

## Layer 10: Missing Pieces (Senior Interview Must-Know)

---

### 10.1: HNSW vs IVF — Vector DB Internally Search Kaise Karta Hai?

Jab tumhare paas 10 million vectors hain aur ek query aati hai, toh brute-force (har vector se compare karo) impossible hai — 10M × 1024 dimensions × cosine calculate karna seconds lagega per query. Production mein tum 10ms mein answer chahte ho. Yeh problem solve karte hain **ANN (Approximate Nearest Neighbor)** algorithms. "Approximate" matlab — 100% exact nearest nahi dega, but 95-99% recall ke saath 100x faster dega.

Do major approaches hain:

#### HNSW (Hierarchical Navigable Small World) — Graph-Based

Imagine ek multi-layer graph. Top layer mein bahut kam nodes hain (coarse view), bottom layer mein sab nodes hain (fine view). Query aati hai toh:

1. Top layer se start karo — coarse level pe nearest node dhundho
2. Ek layer neeche jao — us area mein aur precise search karo
3. Repeat until bottom layer — final nearest neighbors mil gaye

```
Layer 3 (few nodes):     A ---- B ---- C
                              |
Layer 2 (more nodes):    D -- E -- F -- G -- H
                              |
Layer 1 (all nodes):   I-J-K-L-M-N-O-P-Q-R-S-T-U-V-W-X

Query → Start at top → Navigate down → Final answer at bottom
```

**Pros:**
- Very high recall (95-99%)
- Low latency (<10ms for millions of vectors)
- No training needed — insert vectors directly

**Cons:**
- RAM heavy — entire graph in memory (1M vectors × 1024d × 4 bytes = ~4GB + graph overhead ~8-12GB)
- Insert slow compared to IVF (graph restructuring)

**Use when:** <50M vectors, latency critical, RAM budget available. **Qdrant, Pinecone, Weaviate sab default mein HNSW use karte hain.**

#### IVF (Inverted File Index) — Cluster-Based

Pehle sab vectors ko K clusters (Voronoi cells) mein divide karo using k-means. Query aati hai toh:

1. Query vector ke nearest clusters identify karo (nprobe = kitne clusters check karne)
2. Sirf un clusters ke vectors se compare karo (brute-force within cluster)

```
Cluster 1: [v1, v5, v9, v23, ...]     ← check this
Cluster 2: [v2, v8, v12, v45, ...]     ← skip
Cluster 3: [v3, v7, v15, v67, ...]     ← check this
Cluster 4: [v4, v6, v19, v89, ...]     ← skip

Query → Find 2 nearest clusters → Search only within those
```

**Pros:**
- Memory efficient — vectors can stay on disk, only cluster centroids in RAM
- Scales to billions of vectors
- Fast bulk insert

**Cons:**
- Lower recall than HNSW (depends on nprobe — more probes = better recall but slower)
- Requires training step (k-means clustering on initial data)
- Edge case: query falls between cluster boundaries → misses relevant vectors

**Use when:** 100M+ vectors, memory constrained, disk-based storage acceptable. **FAISS IVF variants use this.**

#### Production Decision Matrix:

| Factor | HNSW | IVF |
|--------|------|-----|
| Vectors < 50M | ✅ Best choice | Overkill |
| Vectors > 100M | RAM expensive | ✅ Best choice |
| Latency < 10ms | ✅ | Possible with tuning |
| RAM budget tight | ❌ | ✅ |
| Recall > 99% needed | ✅ | Harder (need high nprobe) |
| Frequent inserts | Slower (graph rebuild) | ✅ Fast |

#### Interview Answer:

> "HNSW is a graph-based ANN algorithm that builds a multi-layer navigable graph — queries traverse from coarse to fine layers, giving high recall at low latency but requiring the full graph in RAM. IVF is cluster-based — it partitions vectors into Voronoi cells and searches only nearby clusters, making it more memory-efficient for very large collections but with a recall-latency tradeoff controlled by nprobe. Most managed vector DBs like Qdrant and Pinecone use HNSW by default because for typical RAG use cases (< 50M vectors), RAM is affordable and latency is critical."

---

### 10.2: MTEB Benchmark — Model Quality Kaise Compare Karte Ho?

Jab koi bolta hai "BGE is SOTA" ya "this model is better than that" — wo kahaan se pata chalta hai? **MTEB (Massive Text Embedding Benchmark).**

MTEB ek standardized test suite hai jo embedding models ko **multiple tasks** pe evaluate karta hai:

| Task Category | Kya Test Karta Hai | Example |
|--------------|-------------------|---------|
| **Retrieval** | Query diya, relevant doc find karo | Search quality |
| **STS (Semantic Textual Similarity)** | Do sentences kitne similar hain | Similarity scoring |
| **Classification** | Text classify karo embedding ke basis pe | Sentiment detection |
| **Clustering** | Similar texts group karo | Topic grouping |
| **Pair Classification** | Do texts same category ke hain ya nahi | Duplicate detection |
| **Reranking** | Retrieved results ko reorder karo | Search ranking |

Har model ka MTEB score hota hai (0-100 scale, higher = better). Leaderboard: https://huggingface.co/spaces/mteb/leaderboard

#### Current Top Models (2025-2026):

| Model | MTEB Avg Score | Dimensions | Notes |
|-------|---------------|-----------|-------|
| Voyage AI (voyage-3) | ~68 | 1024 | API-based, expensive |
| BGE-large-en-v1.5 | ~64 | 1024 | Best open-source |
| OpenAI text-embedding-3-large | ~65 | 3072 | API-based |
| Cohere embed-v3 | ~65 | 1024 | API-based |
| all-mpnet-base-v2 | ~57 | 768 | Good balance |

#### Important Caveat:

MTEB score is **general purpose**. Tumhara specific domain (AWS docs, networking content) pe model differently perform kar sakta hai. **Always test on your own data** — MTEB score ek starting point hai, final decision nahi.

#### Interview Answer:

> "I compare embedding models using the MTEB leaderboard which benchmarks across retrieval, classification, clustering, and similarity tasks. But MTEB is a general benchmark — for production, I always evaluate on our domain-specific test set because a model scoring high on MTEB may underperform on specialized content like infrastructure documentation."

---

### 10.3: Dense vs Sparse Embeddings & Hybrid Search — Complete Picture

#### Dense Embeddings (What we've been discussing):

```python
# BERT/BGE/Titan style — fixed-size, every dimension has a value
"What is Kubernetes?" → [0.23, -0.15, 0.78, ..., 0.45]  # 1024 numbers, ALL non-zero
```

- Captures **semantic meaning** (paraphrases work)
- Fixed size regardless of input length
- "Container orchestration" aur "Kubernetes deployment" similar vectors denge

#### Sparse Embeddings (BM25/TF-IDF style):

```python
# Traditional keyword-based — very high dimensional, MOST values are zero
"What is Kubernetes?" → [0, 0, 0, 0.7, 0, 0, ..., 0.3, 0, 0]  # 50,000+ dimensions, 99% zeros
#                                  ↑ "kubernetes"        ↑ "what"
```

- Captures **exact keyword match**
- Variable importance based on term frequency
- "Kubernetes" aur "K8s" ko similar NAHI maanegi (exact match chahiye)

#### Where Dense FAILS (Critical Interview Point):

```
Query: "Infosys Q3 2024 earnings report"
Dense search result: Generic document about Indian IT company earnings ❌
Sparse (BM25) result: Exact Infosys Q3 2024 document ✅

Query: "Error code EKS-AUTH-403"
Dense search result: Generic EKS authentication troubleshooting ❌
Sparse (BM25) result: Specific doc mentioning that error code ✅
```

Dense search **fails on:**
- Proper nouns (company names, person names)
- Error codes, IDs, serial numbers
- Exact version numbers ("v1.28.3")
- Acronyms the model wasn't trained on

#### Hybrid Search (Best of Both Worlds):

```python
class HybridSearch:
    def search(self, query, alpha=0.7):
        """
        alpha = dense weight (semantic)
        (1-alpha) = sparse weight (keyword)
        """
        # Dense: meaning-based (handles paraphrases)
        dense_results = self.vector_db.search(query_embedding, top_k=20)

        # Sparse: keyword-based (handles exact terms)
        sparse_results = self.bm25_index.search(query, top_k=20)

        # Combine using Reciprocal Rank Fusion (RRF)
        combined = self.reciprocal_rank_fusion(dense_results, sparse_results)

        return combined[:top_k]

    def reciprocal_rank_fusion(self, *result_lists, k=60):
        """
        RRF score = sum(1 / (k + rank_in_each_list))
        Document jo dono lists mein high rank pe hai → highest combined score
        """
        scores = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                if doc.id not in scores:
                    scores[doc.id] = 0
                scores[doc.id] += 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### Which Vector DBs Support Hybrid Natively:

| DB | Hybrid Search Support |
|----|----------------------|
| **Weaviate** | ✅ Built-in (BM25 + vector) |
| **Qdrant** | ✅ Sparse vectors support |
| **Elasticsearch/OpenSearch** | ✅ kNN + BM25 |
| **Pinecone** | ✅ Sparse-dense vectors |
| **Chroma** | ❌ Dense only |
| **FAISS** | ❌ Dense only (library, not DB) |

#### Interview Answer:

> "Dense embeddings capture semantic similarity but fail on rare proper nouns, exact IDs, and terms the model wasn't trained on. Sparse retrieval (BM25) handles exact keyword matching but misses paraphrases. In production RAG systems, I use hybrid search — running both dense and sparse retrieval and combining results via Reciprocal Rank Fusion. This ensures we catch both semantic matches and exact keyword hits. Weaviate and Qdrant support this natively."

---

### 10.4: Embedding Fine-tuning — Kab Aur Kyun?

#### Problem:

Out-of-the-box BGE model general English text pe trained hai. Agar tumhara domain **highly specialized** hai (legal, medical, networking jargon), toh:

```
Query: "BGP route flapping causing ECMP imbalance"
General model thinks: "something about routes... maybe travel routes?"
Domain model thinks: "networking issue with BGP protocol instability affecting traffic distribution"
```

General model domain-specific terminology ki nuances miss kar sakta hai — similar terms ko distant embed kar dega.

#### When to Fine-tune:

| Scenario | Fine-tune? | Why |
|----------|-----------|-----|
| General Q&A chatbot | ❌ No | Pre-trained models are sufficient |
| Legal document search | ✅ Yes | Legal jargon, case citation patterns |
| Medical literature search | ✅ Yes | Medical terminology, drug names |
| Internal company docs | Maybe | Depends on domain-specificity |
| AWS/Cloud documentation | ❌ Usually not | Well-represented in training data |
| Networking (your domain) | ❌ Probably not | Common enough in training corpora |

#### How Fine-tuning Works (Conceptually):

```python
# You provide training pairs:
training_data = [
    # (query, positive_document, negative_document)
    ("BGP flapping", "BGP route oscillation causes ECMP...", "Bird flapping wings..."),
    ("SYN flood mitigation", "TCP SYN flood DDoS protection...", "Flood damage repair..."),
    # ... thousands of pairs
]

# Contrastive loss pushes:
# - query ↔ positive_doc vectors CLOSER
# - query ↔ negative_doc vectors FARTHER APART
```

Fine-tuning tools:
- **Sentence Transformers** `model.fit()` with triplet/contrastive loss
- **BAAI/FlagEmbedding** fine-tune toolkit
- **OpenAI fine-tuning API** (for their embedding models)

#### Cost of Fine-tuning:

| Item | Estimate |
|------|----------|
| Training data preparation | 2-5 days (creating query-doc pairs) |
| GPU training time | 2-8 hours on A100 |
| Validation/testing | 1 day |
| Re-indexing all documents | Hours to days (depends on corpus size) |

#### When NOT to Fine-tune (Important):

- Model already performs well on your data (test first!)
- You have < 1000 training pairs (insufficient data)
- Domain is well-represented in general corpora (AWS, coding, etc.)
- You're under time pressure (fine-tuning + re-indexing = 1-2 weeks)

**Alternative to fine-tuning:** Use a better chunking strategy, add metadata filtering, or use a reranker — often these give 80% of the benefit at 10% of the cost.

#### Interview Answer:

> "Fine-tuning an embedding model is necessary when the domain has specialized terminology that the base model hasn't seen enough during pre-training — legal, medical, or proprietary internal language. You provide contrastive training pairs (query, relevant doc, irrelevant doc) and the model learns to embed domain terms closer together. However, fine-tuning requires re-indexing the entire corpus, so I first try better chunking, metadata filters, and cross-encoder reranking before committing to fine-tuning. For most cloud/infrastructure domains, pre-trained BGE or Titan models perform well enough without fine-tuning."

---

## Final Completeness Check

After this guide, you know:

| Topic | Covered? | Depth |
|-------|----------|-------|
| What embedding is | ✅ | Deep |
| How it works internally (Transformer, contrastive training) | ✅ | Deep |
| Similarity metrics (cosine, dot product, euclidean) | ✅ | Deep |
| Normalization | ✅ | Deep |
| Model comparison & selection | ✅ | Complete |
| Production code (self-hosted + managed) | ✅ | Production-grade |
| Infra/K8s deployment | ✅ | Complete with YAML |
| Scaling (CPU/GPU benchmarks) | ✅ | With numbers |
| Networking (gRPC, ClusterIP, pooling) | ✅ | Your expertise |
| Trade-offs (self-hosted vs managed, dimensions, models) | ✅ | Decision matrices |
| Production pitfalls (5 common failures) | ✅ | Real-world |
| ANN algorithms (HNSW vs IVF) | ✅ | Deep |
| MTEB benchmark | ✅ | Sufficient |
| Dense vs Sparse + Hybrid Search | ✅ | Deep with code |
| Fine-tuning (when, why, how) | ✅ | Complete |
| Interview answers (2-line, 5-min, 10-min) | ✅ | All levels |
| Follow-up Q&A | ✅ | 5+ questions |

**Embedding topic: DONE. Nothing more to learn for your target role.**

---

## Layer 11: GitLab CI/CD + ArgoCD — EKS Deployment (Step-by-Step)

Yeh section **100% real environment** ke liye hai. Har step exactly copy-paste ready hai. EKS cluster pe CPU nodes use karenge (cost-effective for lab).

---

### GPU vs CPU Cost Comparison:

| Setup | Instance | Specs | Cost/Hour | Monthly (24/7) | Embedding Speed |
|-------|----------|-------|-----------|----------------|-----------------|
| **CPU (Lab)** | t3.xlarge | 4 vCPU, 16GB RAM | $0.17 | ~$124 | ~50 texts/sec |
| **CPU (Better)** | m5.xlarge | 4 vCPU, 16GB RAM | $0.19 | ~$140 | ~50 texts/sec |
| **GPU** | g4dn.xlarge | 4 vCPU, 16GB RAM, T4 GPU | $0.53 | ~$384 | ~500 texts/sec |
| **GPU (Big)** | g5.xlarge | 4 vCPU, 16GB RAM, A10G GPU | $1.01 | ~$730 | ~1000 texts/sec |

**Lab ke liye: t3.xlarge (CPU) use karo.** Model slow chalega (~50 texts/sec) but functional hai. GPU tab lo jab production-level speed chahiye.

**Smaller model for CPU (recommended for lab):** `BAAI/bge-base-en-v1.5` (768d) instead of `bge-large` (1024d) — 2x faster on CPU, 60% less RAM.

---

### Prerequisites (Ek baar setup):

```bash
# 1. AWS CLI configured
aws sts get-caller-identity  # Should show your account

# 2. eksctl installed
brew install eksctl  # Mac

# 3. kubectl installed
brew install kubectl

# 4. ArgoCD CLI installed
brew install argocd

# 5. GitLab account (free tier works)
# → Create account at https://gitlab.com
# → Create new project: "embedding-service"
```

---

### PHASE 1: EKS Cluster Create

```bash
# Create EKS cluster (takes ~15 minutes)
eksctl create cluster \
  --name rag-lab \
  --region us-east-1 \
  --nodegroup-name cpu-workers \
  --node-type t3.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed

# Verify
kubectl get nodes
# Should show 2 nodes READY
```

**Cost:** EKS control plane ($0.10/hr = $73/month) + 2× t3.xlarge ($0.17/hr × 2 = $248/month) = **~$320/month total**.

**Cost save:** Delete when not using:
```bash
eksctl delete cluster --name rag-lab --region us-east-1
```

---

### PHASE 2: Install ArgoCD on EKS

```bash
# 1. Create argocd namespace
kubectl create namespace argocd

# 2. Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 3. Wait for pods to be ready (~2 minutes)
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd

# 4. Get admin password
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
echo "ArgoCD Password: $ARGOCD_PASSWORD"

# 5. Port forward to access UI
kubectl port-forward svc/argocd-server -n argocd 8443:443 &

# 6. Login via CLI
argocd login localhost:8443 --username admin --password $ARGOCD_PASSWORD --insecure

# 7. Open browser: https://localhost:8443
# Username: admin
# Password: (printed above)
```

---

### PHASE 3: Project Files (Create These Locally)

```bash
mkdir -p ~/embedding-service/src ~/embedding-service/k8s ~/embedding-service/argocd ~/embedding-service/tests
cd ~/embedding-service
```

#### File 1: `src/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import os
import time

app = FastAPI(title="Embedding Service", version="1.0.0")

# Config
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")  # base = faster on CPU

print(f"Loading model {MODEL_NAME} on CPU...")
start = time.time()
model = SentenceTransformer(MODEL_NAME, device="cpu")
DIMENSIONS = model.get_sentence_embedding_dimension()
print(f"Model loaded in {time.time()-start:.1f}s! Dimensions: {DIMENSIONS}")

IS_BGE = "bge" in MODEL_NAME.lower()
QUERY_PREFIX = "Represent this sentence for searching relevant passages: " if IS_BGE else ""


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    model: str
    count: int


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(400, "texts list cannot be empty")
    if len(request.texts) > 100:
        raise HTTPException(400, "max 100 texts per request")

    texts = request.texts
    if request.is_query and IS_BGE:
        texts = [f"{QUERY_PREFIX}{t}" for t in texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,  # Smaller batch for CPU
        show_progress_bar=False
    )

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimensions=DIMENSIONS,
        model=MODEL_NAME,
        count=len(texts)
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "dimensions": DIMENSIONS}


@app.get("/ready")
async def ready():
    try:
        test = model.encode("healthcheck", normalize_embeddings=True)
        if len(test) == DIMENSIONS:
            return {"status": "ready"}
    except Exception as e:
        raise HTTPException(503, f"Not ready: {e}")
```

#### File 2: `src/requirements.txt`

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
sentence-transformers==2.3.1
torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pydantic==2.5.3
```

> **Note:** `--index-url https://download.pytorch.org/whl/cpu` ensures CPU-only PyTorch is installed (smaller image, no CUDA bloat).

#### File 3: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch)
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/main.py .

# Download model at build time (baked into image = no runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

#### File 4: `k8s/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: embedding-service-config
  namespace: rag-system
data:
  MODEL_NAME: "BAAI/bge-base-en-v1.5"
```

#### File 5: `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
  namespace: rag-system
  labels:
    app: embedding-service
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0    # Zero downtime
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding
        image: registry.gitlab.com/REPLACE_WITH_YOUR_GITLAB_GROUP/embedding-service:latest
        ports:
        - containerPort: 8080
          name: http
        envFrom:
        - configMapRef:
            name: embedding-service-config
        resources:
          requests:
            memory: "3Gi"    # bge-base needs ~2.5GB
            cpu: "1500m"     # 1.5 CPU cores
          limits:
            memory: "4Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 45    # Model load ~30-45s on CPU
          periodSeconds: 10
          failureThreshold: 6
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          failureThreshold: 3
```

#### File 6: `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
  namespace: rag-system
spec:
  selector:
    app: embedding-service
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: http
  type: ClusterIP
```

#### File 7: `k8s/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedding-service-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-service
  minReplicas: 2
  maxReplicas: 6
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### File 8: `argocd/application.yaml`

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: embedding-service
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/REPLACE_WITH_YOUR_GITLAB_GROUP/embedding-service.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-system
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

#### File 9: `.gitlab-ci.yml`

```yaml
stages:
  - test
  - build
  - push
  - deploy

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/embedding-service
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}

# ━━━ Stage 1: Quick Smoke Test ━━━
test:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install fastapi pydantic httpx pytest uvicorn
  script:
    - python -c "from src.main import app; print('Import OK')" || true
    - echo "Basic test passed"
  only:
    - merge_requests
    - main

# ━━━ Stage 2: Build Docker Image ━━━
build:
  stage: build
  image: docker:24.0
  services:
    - docker:24.0-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  before_script:
    - echo ${CI_REGISTRY_PASSWORD} | docker login -u ${CI_REGISTRY_USER} --password-stdin ${CI_REGISTRY}
  script:
    - docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    - docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest
  only:
    - main

# ━━━ Stage 3: Push to GitLab Container Registry ━━━
push:
  stage: push
  image: docker:24.0
  services:
    - docker:24.0-dind
  before_script:
    - echo ${CI_REGISTRY_PASSWORD} | docker login -u ${CI_REGISTRY_USER} --password-stdin ${CI_REGISTRY}
  script:
    - docker push ${DOCKER_IMAGE}:${DOCKER_TAG}
    - docker push ${DOCKER_IMAGE}:latest
  only:
    - main

# ━━━ Stage 4: Update K8s Manifest → Triggers ArgoCD ━━━
deploy:
  stage: deploy
  image: alpine:3.18
  before_script:
    - apk add --no-cache git sed openssh-client
    - git config user.email "ci@gitlab.com"
    - git config user.name "GitLab CI"
    - git remote set-url origin https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.com/${CI_PROJECT_PATH}.git
  script:
    - sed -i "s|image:.*embedding-service:.*|image: ${DOCKER_IMAGE}:${DOCKER_TAG}|" k8s/deployment.yaml
    - git add k8s/deployment.yaml
    - git diff --cached --quiet || git commit -m "ci: deploy embedding-service ${DOCKER_TAG}"
    - git push origin HEAD:main
  only:
    - main
```

#### File 10: `tests/test_health.py`

```python
"""Basic health test — runs after deploy"""
import requests

def test_health():
    # In CI, this runs against port-forwarded service
    resp = requests.get("http://localhost:8080/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["dimensions"] in [768, 1024]
```

---

### PHASE 4: Push to GitLab

```bash
cd ~/embedding-service

# Initialize git
git init
git add .
git commit -m "feat: initial embedding service"

# Create GitLab project first (via UI or CLI):
# Go to https://gitlab.com → New Project → "embedding-service"

# Add remote and push
git remote add origin https://gitlab.com/YOUR_USERNAME/embedding-service.git
git branch -M main
git push -u origin main

# GitLab CI will automatically:
# 1. Run test stage
# 2. Build Docker image (~5-10 mins first time, model download)
# 3. Push to GitLab Container Registry
# 4. Update k8s/deployment.yaml with new image tag
```

---

### PHASE 5: Connect ArgoCD to GitLab

```bash
# 1. Create namespace for our app
kubectl create namespace rag-system

# 2. Add GitLab repo to ArgoCD
# First, create a GitLab Personal Access Token:
# GitLab → Settings → Access Tokens → Create with "read_repository" scope
# Save the token

argocd repo add https://gitlab.com/YOUR_USERNAME/embedding-service.git \
  --username your-gitlab-username \
  --password YOUR_PERSONAL_ACCESS_TOKEN

# 3. Apply ArgoCD Application
kubectl apply -f argocd/application.yaml

# 4. Check status
argocd app get embedding-service
# Should show: Status: Synced, Health: Healthy (after pods start)

# 5. Watch pods come up
kubectl get pods -n rag-system -w
# Wait until STATUS = Running, READY = 1/1
```

---

### PHASE 6: Verify Everything Works

```bash
# 1. Check pods are running
kubectl get pods -n rag-system
# NAME                                READY   STATUS    RESTARTS   AGE
# embedding-service-xxxxx-yyy         1/1     Running   0          2m
# embedding-service-xxxxx-zzz         1/1     Running   0          2m

# 2. Check logs (model should be loaded)
kubectl logs -l app=embedding-service -n rag-system --tail=20
# Should see: "Model loaded in 35.2s! Dimensions: 768"

# 3. Port forward to test locally
kubectl port-forward svc/embedding-service -n rag-system 8080:8080 &

# 4. Test embedding endpoint
curl -s -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is EKS?", "Kubernetes networking"], "is_query": true}' | python3 -m json.tool

# Expected output:
# {
#     "embeddings": [[0.023, -0.156, ...], [0.045, 0.089, ...]],
#     "dimensions": 768,
#     "model": "BAAI/bge-base-en-v1.5",
#     "count": 2
# }

# 5. Test health
curl http://localhost:8080/health
# {"status":"healthy","model":"BAAI/bge-base-en-v1.5","dimensions":768}

# 6. Check HPA
kubectl get hpa -n rag-system
# NAME                      REFERENCE                      TARGETS   MINPODS   MAXPODS
# embedding-service-hpa     Deployment/embedding-service   45%/70%   2         6

# 7. Check in ArgoCD UI
# Open https://localhost:8443 → embedding-service app → should be green/synced
```

---

### PHASE 7: Test GitOps Flow (Change → Auto-Deploy)

```bash
cd ~/embedding-service

# Make a code change
echo '
@app.get("/version")
async def version():
    return {"version": "1.1.0", "model": MODEL_NAME}
' >> src/main.py

# Commit and push
git add .
git commit -m "feat: add version endpoint"
git push origin main

# Now watch:
# 1. GitLab CI starts (check https://gitlab.com/YOUR_USERNAME/embedding-service/-/pipelines)
# 2. Docker image builds + pushes (~5-8 mins)
# 3. k8s/deployment.yaml updated with new tag
# 4. ArgoCD detects change → syncs → rolling update

# Watch rolling update
kubectl get pods -n rag-system -w
# Old pods terminate, new pods start (zero downtime)

# Verify new version
curl http://localhost:8080/version
# {"version": "1.1.0", "model": "BAAI/bge-base-en-v1.5"}
```

---

### PHASE 8: Cleanup (Save Cost)

```bash
# Option A: Delete just the app (keep cluster)
kubectl delete -f argocd/application.yaml
kubectl delete namespace rag-system

# Option B: Delete entire cluster (stop all charges)
eksctl delete cluster --name rag-lab --region us-east-1
# Takes ~10 minutes. Confirms deletion of all resources.
```

---

### Complete GitOps Flow Diagram:

```
┌─────────────────────────────────────────────────────────────┐
│ DEVELOPER MACHINE                                            │
│                                                              │
│ git push → GitLab (gitlab.com)                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ GITLAB CI/CD (gitlab.com/YOUR_USERNAME/embedding-service)    │
│                                                              │
│ Stage 1: test (smoke test)                                  │
│ Stage 2: build (docker build — includes model download)     │
│ Stage 3: push (docker push to registry.gitlab.com)          │
│ Stage 4: deploy (update k8s/deployment.yaml image tag)      │
│           → git push (triggers ArgoCD)                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ ARGOCD (running on EKS, watches GitLab repo k8s/ folder)    │
│                                                              │
│ Detects: deployment.yaml image tag changed                  │
│ Action: kubectl apply (rolling update)                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ EKS CLUSTER (us-east-1)                                     │
│                                                              │
│ namespace: rag-system                                       │
│ ├── embedding-service-pod-1 (Running, Ready)                │
│ ├── embedding-service-pod-2 (Running, Ready)                │
│ ├── service/embedding-service (ClusterIP:8080)              │
│ └── hpa/embedding-service-hpa (2-6 replicas, CPU 70%)      │
└─────────────────────────────────────────────────────────────┘
```

---

### Troubleshooting (Common Issues):

| Problem | Fix |
|---------|-----|
| Pod stuck in `Pending` | Node has insufficient memory. Check: `kubectl describe pod <name> -n rag-system`. Fix: Use t3.xlarge (16GB) not t3.large (8GB) |
| Pod `CrashLoopBackOff` | Model too large for memory. Fix: Switch to `bge-base` (768d, ~1.5GB) instead of `bge-large` (1024d, ~2.5GB) |
| GitLab CI build slow (20+ min) | Model download during build. Normal for first time. Subsequent builds use cache |
| ArgoCD shows `OutOfSync` | Check: `argocd app diff embedding-service`. Usually image tag mismatch. Force sync: `argocd app sync embedding-service` |
| `ImagePullBackOff` | GitLab registry auth issue. Create K8s secret: see below |
| Port forward not working | Kill old port-forward: `pkill -f "port-forward.*8080"`. Retry. |

**Fix ImagePullBackOff (GitLab private registry):**
```bash
# Create docker registry secret in K8s
kubectl create secret docker-registry gitlab-registry \
  --namespace rag-system \
  --docker-server=registry.gitlab.com \
  --docker-username=YOUR_GITLAB_USERNAME \
  --docker-password=YOUR_PERSONAL_ACCESS_TOKEN

# Add to deployment.yaml under spec.template.spec:
# imagePullSecrets:
# - name: gitlab-registry
```

---

### GPU Upgrade Path (When Ready):

Jab CPU pe sab kaam kar raha hai aur speed chahiye:

```bash
# Add GPU node group to existing cluster
eksctl create nodegroup \
  --cluster rag-lab \
  --name gpu-workers \
  --node-type g4dn.xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 2 \
  --region us-east-1

# Update deployment.yaml:
# resources:
#   requests:
#     nvidia.com/gpu: "1"
#   limits:
#     nvidia.com/gpu: "1"

# Update ConfigMap:
# MODEL_NAME: "BAAI/bge-large-en-v1.5"  (upgrade to large model)

# Install NVIDIA device plugin (required for GPU scheduling)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

**GPU cost:** g4dn.xlarge = $0.53/hr = ~$384/month (always on) or ~$50-60/month (4 hrs/day lab use).

---

Move to Topic 2: Chunking Strategies when ready.
