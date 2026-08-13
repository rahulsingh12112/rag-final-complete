# Topic 3: Vector Databases — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Embedding Models (Topic 1), Chunking (Topic 2)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/04-vector-databases.md + 2026 Production Benchmarks

---

## 🎯 One-Liner (Interview):

> "Vector database ek specialized database hai jo high-dimensional embeddings store karta hai aur ANN (Approximate Nearest Neighbor) algorithms use karke milliseconds mein similarity search karta hai — yeh RAG systems ka backbone hai."

---

## Layer 1: Kya Hai Aur Kyun Regular DB Nahi Chalega?

SQL database mein embeddings store kar sakte ho (as arrays). Lekin jab user query aati hai aur tumhe 10 million vectors mein se top-5 similar dhundhne hain — SQL database full scan karega. 10M × 1024 dimensions × cosine calculation = **seconds per query**. Production mein yeh unusable hai.

Vector databases solve this with **ANN indexing**. Brute-force ke bajaye, yeh space ko intelligently structure karte hain taaki sirf ek fraction vectors check karne padein. Result: **milliseconds mein answer**, with 95-99% accuracy (approximate, not exact — but practically identical).

**Core difference:**

| | SQL Database | Vector Database |
|--|---|---|
| Query type | Exact match, range, join | Nearest neighbor, similarity |
| Data type | Structured rows | High-dimensional vectors |
| Index type | B-tree, hash | HNSW, IVF, PQ |
| Search result | Exact | Approximate (configurable recall) |
| Typical use | Transactions, analytics | Semantic search, RAG, recommendations |

**Key operations a vector DB does:**
1. **Store** vectors with metadata (payload)
2. **Index** vectors for fast search (HNSW/IVF)
3. **Search** by similarity (cosine/dot product)
4. **Filter** by metadata during search
5. **Update/Delete** vectors in real-time
6. **Scale** horizontally for billions of vectors

---

## Layer 2: Major Options — Honest Comparison (2026 Benchmarks)

### Feature Matrix:

| Feature | Qdrant | Weaviate | Pinecone | Chroma | Milvus | pgvector |
|---------|--------|----------|----------|--------|--------|----------|
| **Open Source** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ (extension) |
| **Language** | Rust | Go | Proprietary | Python | C++/Go | C (PostgreSQL) |
| **Managed Cloud** | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Hybrid Search** | ✅ | ✅ (best) | ✅ | ❌ | ✅ | ❌ |
| **Filtering** | Excellent | Excellent | Good | Basic | Excellent | SQL (native) |
| **Multi-tenancy** | ✅ | ✅ (native) | ✅ (namespaces) | ❌ | ✅ | Schema-based |
| **Max vectors tested** | 100M+ | 50M+ | Billions | <1M | Billions | 10M+ |
| **gRPC support** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Quantization** | ✅ (scalar, binary) | ✅ | ✅ | ❌ | ✅ (PQ, SQ) | ❌ |

### Performance Benchmarks (2026, 1M vectors, 768d, concurrent queries):

| Metric | Qdrant | Weaviate | Pinecone | Milvus |
|--------|--------|----------|----------|--------|
| **p99 Latency** | 15ms | 22ms | 38ms | 18ms |
| **QPS (queries/sec)** | 9,500 | 6,200 | 4,800 | 9,000 |
| **Recall@10** | 0.98 | 0.97 | 0.99 | 0.98 |
| **Filtered search p99** | 18ms | 25ms | 45ms | 22ms |

### Cost Comparison (Monthly, production workload):

| Setup | Qdrant (self-hosted) | Qdrant Cloud | Pinecone | Weaviate Cloud |
|-------|---------------------|-------------|----------|----------------|
| 1M vectors | ~$30 (t3.large) | ~$100 | ~$70 | ~$100 |
| 10M vectors | ~$100 (r6g.xlarge) | ~$300 | ~$350 | ~$300 |
| 100M vectors | ~$500 (r6g.4xlarge) | ~$1,200 | ~$2,000 | ~$1,500 |

---

## Layer 3: When to Use What — Decision Tree

```
START
│
├─ "I want ZERO infra management"
│   └─ Budget okay? → Pinecone ✅
│   └─ Budget tight? → Qdrant Cloud (cheaper) ✅
│
├─ "I need hybrid search (vector + keyword)"
│   └─ Weaviate ✅ (best native hybrid)
│   └─ Qdrant (sparse vectors support) ✅
│
├─ "I want best performance, self-hosted"
│   └─ < 100M vectors → Qdrant ✅
│   └─ > 100M vectors → Milvus ✅
│
├─ "I already use PostgreSQL, small scale"
│   └─ pgvector ✅ (no new infra, < 5M vectors)
│
├─ "I'm just prototyping/learning"
│   └─ Chroma ✅ (embedded, zero setup)
│
└─ "Enterprise, billions of vectors, distributed"
    └─ Milvus ✅ or Pinecone Enterprise ✅
```

---

## Layer 4: Qdrant Deep Dive (Recommended for Most Production RAG)

### Why Qdrant?

- Written in Rust = memory safe, fast, no GC pauses
- Best price-performance ratio for self-hosted
- Excellent filtering (pre-filter, not post-filter = fast)
- gRPC support (3-5x faster than REST for large vectors)
- Quantization built-in (reduce RAM by 4-8x)
- Simple deployment (single binary, Docker, K8s)

### Production Code:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    OptimizersConfigDiff, HnswConfigDiff,
    ScalarQuantizationConfig, ScalarType
)
import uuid
import numpy as np

class ProductionVectorDB:
    """Production-ready Qdrant wrapper"""

    def __init__(self, url="http://localhost:6333", grpc_port=6334):
        self.client = QdrantClient(
            url=url,
            grpc_port=grpc_port,
            prefer_grpc=True,  # Always use gRPC in production
            timeout=60
        )

    def create_collection(self, name: str, dimension: int = 1024):
        """Create optimized production collection"""
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            ),
            # HNSW tuning for production
            hnsw_config=HnswConfigDiff(
                m=16,              # Connections per node (16 = good default)
                ef_construct=128,  # Index build quality (higher = better, slower build)
                full_scan_threshold=10000  # Below this, brute-force is faster
            ),
            # Optimizer settings
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20000,  # Use mmap for large segments
                indexing_threshold=10000  # Start indexing after this many vectors
            ),
            # Quantization — reduce RAM by 4x with minimal quality loss
            quantization_config=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True  # Keep quantized vectors in RAM
            )
        )

    def upsert_batch(self, collection: str, texts: list, embeddings: list, metadata: list):
        """Batch upsert with metadata"""
        points = []
        for text, embedding, meta in zip(texts, embeddings, metadata):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding if isinstance(embedding, list) else embedding.tolist(),
                payload={"text": text, **meta}
            ))

        # Batch upsert (Qdrant handles batching internally)
        self.client.upsert(
            collection_name=collection,
            points=points,
            wait=True  # Wait for indexing confirmation
        )

    def search(self, collection: str, query_vector, top_k: int = 5, filters: dict = None):
        """Search with optional metadata filtering"""
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector if isinstance(query_vector, list) else query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )

        return [
            {"text": r.payload["text"], "score": r.score, "metadata": r.payload}
            for r in results
        ]

    def delete_by_filter(self, collection: str, filter_key: str, filter_value: str):
        """Delete vectors by metadata filter (useful for document updates)"""
        self.client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key=filter_key, match=MatchValue(value=filter_value))]
            )
        )

    def get_collection_info(self, collection: str) -> dict:
        """Get collection stats for monitoring"""
        info = self.client.get_collection(collection)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors": info.indexed_vectors_count,
            "status": info.status.value,
            "segments": len(info.segments) if info.segments else 0
        }
```

### HNSW Parameter Tuning Guide:

| Parameter | Default | Effect of Increasing | Production Recommendation |
|-----------|---------|---------------------|--------------------------|
| **m** (connections) | 16 | Better recall, more RAM | 16 (default is optimal for most) |
| **ef_construct** | 128 | Better index quality, slower build | 128-200 |
| **ef** (search-time) | 128 | Better recall, slower search | 64-256 (tune based on latency budget) |

```
ef=64  → p99 ~8ms,  recall 0.95 (fast, acceptable)
ef=128 → p99 ~15ms, recall 0.98 (balanced, production default)
ef=256 → p99 ~30ms, recall 0.99 (high accuracy, slower)
```

### Quantization — Reduce RAM by 4-8x:

| Type | RAM Reduction | Quality Loss | Use When |
|------|--------------|-------------|----------|
| **Scalar (INT8)** | 4x | < 1% | Default for production |
| **Binary** | 32x | 5-10% | Budget very tight, quality acceptable |
| **Product Quantization** | 16-64x | 3-5% | Milvus/FAISS (not Qdrant native) |

1M vectors × 1024d × float32 = **4GB RAM**
With INT8 quantization = **1GB RAM** (4x saving, <1% quality loss)

---

## Layer 5: Kubernetes Deployment (Production)

### Qdrant StatefulSet (HA with Replication):

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: vector-db
spec:
  serviceName: qdrant
  replicas: 3  # 3 nodes for HA
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
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
          name: internal  # Cluster communication
        env:
        - name: QDRANT__CLUSTER__ENABLED
          value: "true"
        - name: QDRANT__SERVICE__GRPC_PORT
          value: "6334"
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
      storageClassName: gp3  # AWS EBS gp3
      resources:
        requests:
          storage: 100Gi
---
# Headless service for StatefulSet internal DNS
apiVersion: v1
kind: Service
metadata:
  name: qdrant-headless
  namespace: vector-db
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
# Regular service for client access
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: vector-db
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    name: http
    targetPort: 6333
  - port: 6334
    name: grpc
    targetPort: 6334
  type: ClusterIP
```

### Networking Considerations (Tumhara Forte):

- **ClusterIP** for internal access (RAG service → Qdrant)
- **gRPC (port 6334)** for client connections — 3-5x faster than REST for vector payloads
- **Internal port (6335)** for cluster node communication — keep in same AZ to reduce latency
- **Network policy:** Only allow RAG service pods to reach Qdrant — deny all else
- **No external exposure** — vector DB should NEVER be publicly accessible
- **Inter-AZ traffic cost:** Qdrant replicas across AZs = cross-AZ data transfer charges. For cost optimization, keep primary reads in-AZ using topology-aware routing

```yaml
# Network Policy — only RAG service can access Qdrant
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qdrant-access
  namespace: vector-db
spec:
  podSelector:
    matchLabels:
      app: qdrant
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: rag-service
    - podSelector:
        matchLabels:
          app: rag-api
    ports:
    - port: 6333
    - port: 6334
```

---

## Layer 6: Scaling Strategies

### Vertical Scaling (Single Node):

| Vectors | RAM Needed (1024d, float32) | With INT8 Quantization | Instance |
|---------|----------------------------|----------------------|----------|
| 1M | 4GB | 1GB | t3.large |
| 5M | 20GB | 5GB | r6g.xlarge |
| 10M | 40GB | 10GB | r6g.2xlarge |
| 50M | 200GB | 50GB | r6g.8xlarge |
| 100M+ | Need horizontal scaling | | → Sharding |

### Horizontal Scaling (Cluster):

```python
# Qdrant cluster with sharding + replication
# Create collection with sharding
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    shard_number=6,          # Split data across 6 shards
    replication_factor=2,    # Each shard has 2 copies
    write_consistency_factor=1  # 1 node ack = write confirmed
)
```

**Sharding strategies:**
- **Auto-sharding** (Qdrant default) — distributes evenly
- **By tenant** — each customer gets own shard (multi-tenancy)
- **By time** — recent data on fast nodes, old data on cheaper storage

### Replication for HA:

```
3-node cluster, replication_factor=2:

Node 1: [Shard A primary] [Shard B replica]
Node 2: [Shard A replica]  [Shard C primary]
Node 3: [Shard B primary]  [Shard C replica]

If Node 2 dies → Shard A still served by Node 1, Shard C by Node 3
Zero downtime.
```

---

## Layer 7: Monitoring & Observability

### Key Metrics to Track:

```python
# Prometheus metrics for vector DB monitoring
metrics = {
    # Latency
    "vector_db_search_latency_p50": "< 10ms target",
    "vector_db_search_latency_p99": "< 50ms target",

    # Throughput
    "vector_db_queries_per_second": "Track for capacity planning",
    "vector_db_upserts_per_second": "Track indexing throughput",

    # Health
    "vector_db_collection_size": "Total vectors stored",
    "vector_db_indexed_percentage": "Should be ~100%",
    "vector_db_segment_count": "Too many = needs optimization",

    # Resources
    "vector_db_memory_usage_bytes": "Alert if > 80% capacity",
    "vector_db_disk_usage_bytes": "Alert if > 70% capacity",

    # Quality
    "vector_db_average_score": "Track retrieval quality drift"
}
```

### Qdrant Built-in Metrics (Prometheus endpoint):

```yaml
# Qdrant exposes /metrics endpoint
# Scrape config for Prometheus
- job_name: 'qdrant'
  metrics_path: '/metrics'
  static_configs:
  - targets: ['qdrant:6333']
```

### Alerting Rules:

```yaml
groups:
- name: vector-db-alerts
  rules:
  - alert: VectorDBHighLatency
    expr: histogram_quantile(0.99, vector_db_search_duration_seconds) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Vector DB p99 latency > 100ms"

  - alert: VectorDBMemoryHigh
    expr: vector_db_memory_usage_bytes / vector_db_memory_limit_bytes > 0.85
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Vector DB memory > 85% - risk of OOM"

  - alert: VectorDBUnindexedVectors
    expr: (vector_db_vectors_total - vector_db_indexed_vectors) > 10000
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Unindexed vectors growing - indexing may be stuck"
```

---

## Layer 8: Cost Optimization

### Strategy 1: Quantization (Biggest Win)

```python
# Before: 10M vectors × 1024d × 4 bytes = 40GB RAM
# After (INT8): 10M × 1024 × 1 byte = 10GB RAM
# Savings: 75% RAM reduction, <1% quality loss

client.create_collection(
    collection_name="cost_optimized",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    quantization_config=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True  # Keep quantized in RAM, full vectors on disk
    )
)
```

### Strategy 2: Memory-Mapped Files (mmap)

Vectors disk pe rakho, sirf index RAM mein. Slower searches but drastically less RAM:

```python
optimizers_config=OptimizersConfigDiff(
    memmap_threshold=10000  # Segments > 10K vectors → mmap from disk
)
```

**Trade-off:** Latency ~2-3x increase, but RAM usage drops 80%.

### Strategy 3: Tiered Storage

```
HOT tier (RAM):     Recent/frequent vectors     → Fast search, expensive
WARM tier (SSD):    Older vectors with mmap     → Moderate speed, cheaper
COLD tier (S3):     Archive, rarely queried     → Cheap, high latency
```

### Strategy 4: Query Caching

```python
import redis
import json
import hashlib

class CachedVectorSearch:
    def __init__(self, vector_db, redis_client):
        self.db = vector_db
        self.cache = redis_client
        self.ttl = 3600  # 1 hour cache

    def search(self, query_vector, top_k=5):
        # Cache key = hash of query vector
        cache_key = hashlib.md5(json.dumps(query_vector[:10]).encode()).hexdigest()

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)

        # Cache miss — search DB
        results = self.db.search(query_vector, top_k)

        # Store in cache
        self.cache.setex(cache_key, self.ttl, json.dumps(results))
        return results
```

### Strategy 5: Right-sizing Instances

| Workload | Instance | Monthly Cost | When |
|----------|----------|-------------|------|
| Dev/Test | t3.medium (4GB) | ~$30 | < 500K vectors |
| Small Prod | r6g.large (16GB) | ~$80 | 1-5M vectors |
| Medium Prod | r6g.xlarge (32GB) | ~$150 | 5-20M vectors |
| Large Prod | r6g.4xlarge (128GB) | ~$500 | 20-100M vectors |
| Cluster | 3× r6g.2xlarge | ~$900 | 50M+ with HA |

---

## Layer 9: Production Pitfalls

### Pitfall 1: Not Using gRPC

REST API mein vector (1024 floats) JSON serialize hota hai = ~10KB per vector. gRPC + protobuf = ~4KB. **60% bandwidth saving, 3-5x faster.**

**Fix:** Always set `prefer_grpc=True` in production.

### Pitfall 2: Collection Not Indexed

Vectors insert kiye but indexing incomplete — search returns wrong results ya very slow.

**Fix:** Check `indexed_vectors_count == vectors_count`. Wait for indexing after bulk inserts:
```python
client.upsert(points=points, wait=True)
# Or check:
info = client.get_collection("my_collection")
assert info.indexed_vectors_count == info.vectors_count
```

### Pitfall 3: No Backup Strategy

Vector DB crash ho gaya = sab embeddings gone. Re-embedding expensive (GPU time).

**Fix:**
- Qdrant: Snapshots API → S3 backup daily
- Pinecone: Managed (automatic)
- Kubernetes: PVC snapshots via Velero

```python
# Qdrant snapshot
client.create_snapshot(collection_name="production")
# Then copy to S3
```

### Pitfall 4: Single Node in Production

No replication = single point of failure. Node down = service down.

**Fix:** Minimum 2 replicas. Ideally 3 nodes with `replication_factor=2`.

### Pitfall 5: Not Monitoring Index Quality

Index degrade hota hai over time (too many deletes, segments fragmentation).

**Fix:** Monitor segment count. If growing without vector count growing → run optimization:
```python
client.update_collection(
    collection_name="production",
    optimizer_config=OptimizersConfigDiff(
        max_segment_size=500000  # Force segment merging
    )
)
```

### Pitfall 6: Cross-AZ Latency & Cost

Qdrant cluster across AZs = inter-AZ data transfer ($0.01/GB). High-throughput = expensive.

**Fix:** Keep primary shard in same AZ as RAG service. Use topology-aware routing in K8s.

---

## Layer 10: Trade-offs & Decisions

### Self-Hosted vs Managed:

| Factor | Self-Hosted (Qdrant on K8s) | Managed (Pinecone/Qdrant Cloud) |
|--------|----------------------------|-------------------------------|
| Cost (10M vectors) | ~$100-150/month | ~$300-350/month |
| Ops effort | You manage (backups, upgrades, scaling) | Zero ops |
| Latency | <15ms (in-cluster) | 30-50ms (network hop) |
| Control | Full (tuning, networking, security) | Limited |
| HA setup | You configure replication | Built-in |
| Breakeven | >5M vectors = self-hosted cheaper | <5M vectors = managed easier |

### Qdrant vs Weaviate vs Pinecone:

| "I need..." | Choose |
|-------------|--------|
| Best raw performance | Qdrant |
| Hybrid search (vector + BM25) | Weaviate |
| Zero infrastructure management | Pinecone |
| Complex metadata filtering | Qdrant |
| GraphQL API | Weaviate |
| Multi-tenancy (thousands of tenants) | Weaviate (native) or Pinecone (namespaces) |
| Cheapest at scale (self-hosted) | Qdrant |
| Already using PostgreSQL, small scale | pgvector |

---

## Layer 11: Interview Ready

### 2-Line Answer (Screening):

> "A vector database stores embeddings and uses ANN algorithms like HNSW to perform similarity search in milliseconds instead of brute-force scanning. In RAG systems, it's the retrieval engine that finds relevant document chunks for a given query."

### 5-Min Answer (Technical Round):

> Above + comparison (Qdrant for performance, Weaviate for hybrid, Pinecone for managed), HNSW internals (multi-layer graph, ef/m parameters), deployment patterns (StatefulSet on K8s, gRPC, replication), quantization for cost optimization, filtering capabilities.

### 10-Min Deep Dive (System Design):

> Above + scaling architecture (sharding strategies, replication factor, cross-AZ considerations), cost modeling (RAM calculation: vectors × dimensions × bytes), monitoring (latency p99, indexed percentage, segment health), backup strategy, incremental indexing, network policies, query caching with Redis, capacity planning formula.

### Expected Follow-up Questions:

**Q: "10M documents, each 5 chunks average = 50M vectors. Architecture kya hogi?"**
A: Qdrant cluster, 3 nodes r6g.4xlarge (128GB each). Shard_number=6, replication_factor=2. INT8 quantization (50M × 1024 × 1 byte = 50GB, fits in RAM across 3 nodes). gRPC for client access. Daily S3 snapshots. Prometheus monitoring.

**Q: "Latency requirement 20ms p99 — kaise guarantee karoge?"**
A: gRPC (not REST), quantized vectors in RAM (`always_ram=True`), ef search parameter tuned (64-128), same-AZ deployment (no cross-AZ hop), connection pooling, pre-warm index after restart.

**Q: "Multi-tenant SaaS — har customer ka data isolate kaise karoge?"**
A: Option 1: Pinecone namespaces (easiest). Option 2: Qdrant collections per tenant (< 1000 tenants). Option 3: Weaviate native multi-tenancy (thousands of tenants, recommended). Option 4: Metadata filtering with tenant_id field (simplest, no isolation).

**Q: "Vector DB migrate karna hai Chroma se Qdrant — kaise?"**
A: Export all vectors + metadata from Chroma. Create Qdrant collection with same config. Batch upsert (10K per batch). Verify vector count matches. Switch application config. No re-embedding needed (vectors are portable between DBs).

**Q: "Cost optimize karo — currently $2000/month on Pinecone"**
A: Migrate to self-hosted Qdrant on K8s. 3× r6g.2xlarge = ~$900/month. Add INT8 quantization = can use smaller instances (~$500/month). Add Redis cache for frequent queries = reduce QPS to vector DB. Total: ~60-75% cost reduction.

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| What vector DB is & why needed | ✅ |
| All major options (6 databases compared) | ✅ |
| Performance benchmarks (2026) | ✅ |
| Cost comparison | ✅ |
| Decision tree (when to use what) | ✅ |
| Qdrant deep dive (production code) | ✅ |
| HNSW parameter tuning | ✅ |
| Quantization (INT8, binary, PQ) | ✅ |
| Kubernetes deployment (StatefulSet, HA) | ✅ |
| Networking (gRPC, NetworkPolicy, cross-AZ) | ✅ |
| Scaling (vertical + horizontal, sharding, replication) | ✅ |
| Monitoring (Prometheus metrics, alerts) | ✅ |
| Cost optimization (5 strategies) | ✅ |
| Production pitfalls (6 common issues) | ✅ |
| Trade-offs (self-hosted vs managed, DB comparison) | ✅ |
| Interview answers (all levels) | ✅ |
| Follow-up Q&A (5 questions) | ✅ |

**Topic 3: Vector Databases — DONE.**

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/04-vector-databases.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Benchmark Data:** 2026 production benchmarks from markaicode.com, letsdatascience.com, HuggingFace blog comparisons
- **Extra added:** 2026 benchmarks, cost modeling, K8s StatefulSet with HA, NetworkPolicy, gRPC optimization, Redis caching, cross-AZ considerations, monitoring alerts, capacity planning formula, migration strategy — not in original curriculum


---

## Layer 12: GitLab CI/CD + ArgoCD — Vector DB Deployment

Vector DB (Qdrant) ko Kubernetes pe deploy karna with GitLab for Helm chart management aur ArgoCD for GitOps sync.

### Project Structure:

```
vector-db-infra/
├── helm/
│   └── qdrant/
│       ├── Chart.yaml
│       ├── values.yaml          # Default values
│       ├── values-prod.yaml     # Production overrides
│       └── templates/
│           ├── statefulset.yaml
│           ├── service.yaml
│           ├── pvc.yaml
│           └── networkpolicy.yaml
├── argocd/
│   └── application.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: Helm Values (`helm/qdrant/values.yaml`)

```yaml
replicaCount: 3

image:
  repository: qdrant/qdrant
  tag: "v1.12.0"

resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"

persistence:
  enabled: true
  size: 100Gi
  storageClass: gp3

service:
  type: ClusterIP
  httpPort: 6333
  grpcPort: 6334

config:
  cluster:
    enabled: true
  storage:
    performance:
      memmap_threshold_kb: 20000
  quantization:
    scalar:
      type: int8
      quantile: 0.99
      always_ram: true

networkPolicy:
  enabled: true
  allowedNamespaces:
    - rag-system
```

### Step 2: GitLab CI (Lint + Package Helm Chart)

```yaml
stages:
  - lint
  - package

lint:
  stage: lint
  image: alpine/helm:3.14
  script:
    - helm lint helm/qdrant/
    - helm template qdrant helm/qdrant/ -f helm/qdrant/values-prod.yaml

package:
  stage: package
  image: alpine/helm:3.14
  script:
    - helm package helm/qdrant/
  artifacts:
    paths:
      - "*.tgz"
  only: [main]
```

### Step 3: ArgoCD Application (Helm-based)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: qdrant-cluster
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/vector-db-infra.git
    targetRevision: main
    path: helm/qdrant
    helm:
      valueFiles:
        - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: vector-db
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

### Step 4: Commands

```bash
# ArgoCD deploys automatically, but manual commands:
kubectl get statefulsets -n vector-db
kubectl get pods -n vector-db
# Verify cluster health
curl http://qdrant-0.qdrant-headless.vector-db:6333/cluster

# Scale (update values-prod.yaml, push to git → ArgoCD syncs)
# replicaCount: 3 → replicaCount: 5 → git push → auto-deployed

# Backup (snapshot)
curl -X POST http://qdrant:6333/collections/documents/snapshots
```

### Flow:

```
Helm values change → git push → GitLab CI (lint) → ArgoCD detects → Helm upgrade on K8s
                                                                         │
                                                              StatefulSet rolling update
                                                              (one pod at a time, zero downtime)
```
