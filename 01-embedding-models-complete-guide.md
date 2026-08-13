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
