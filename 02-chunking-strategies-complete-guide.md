# Topic 2: Chunking Strategies — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Embedding Models (Topic 1 complete)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/03-rag-systems.md + LangCopilot 2026 Guide + Anthropic Contextual Retrieval + Jina AI Late Chunking

---

## 🎯 One-Liner (Interview):

> "Chunking wo process hai jisme documents ko chhote, meaningful pieces mein todte hain taaki embedding model har piece ka focused representation bana sake aur retrieval precise ho — galat chunking se best embedding model bhi fail hoga."

---

## Layer 1: Kya Hai Aur Kyun Zaroori Hai?

Embedding model ki ek fundamental limitation hai — uski **max token window**. BGE-large max 512 tokens process kar sakta hai, Titan Embed V2 max 8192 tokens. Agar tumhara document 50 pages ka hai (50,000+ tokens), toh tum seedha poora document embed nahi kar sakte.

Lekin sirf size limit nahi hai — agar tum ek 5000 word document ka ek hi embedding banaoge, toh wo embedding bahut **vague** hogi. Usme har topic thoda thoda mix hoga — "networking section" aur "pricing section" dono ka influence hoga us single vector mein. Jab koi specific question aayega ("What's the pricing for t3.large?"), toh yeh vague vector match nahi karega kyunki uska signal dilute ho gaya hai noise mein.

Chunking ek **art hai, science nahi**. Same document ko 10 different tarike se chunk kar sakte ho — aur har tarika alag retrieval quality dega. Yeh RAG pipeline ka **single highest-leverage decision** hai. Baaki sab (model, vector DB, reranker) secondary hai agar chunking galat hai.

**Key insight:** Chunking ka goal hai ki har chunk:
1. Ek **coherent concept** capture kare (ek topic, ek idea)
2. **Self-contained** ho (context ke bina bhi samajh aa jaye)
3. Embedding model ki **token limit** ke andar fit ho
4. Retrieval ke waqt **precise match** de (noise kam ho)

---

## Layer 2: Core Strategies (5 Essential + 4 Advanced)

### Strategy 1: Fixed-Size Chunking

Sabse simple — har N characters/words pe kaat do.

```python
def fixed_size_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

**Kab use karo:** Sirf prototyping. Production mein KABHI nahi — kyunki yeh sentences beech mein kaat deta hai.

**Problem example:**
```
Original: "VPC peering does not support transitive routing. Each VPC..."
Chunk 1: "...VPC peering does not support"
Chunk 2: "transitive routing. Each VPC..."
```
Meaning destroyed. Embedding useless.

---

### Strategy 2: Recursive Character Splitting (LangChain Default — MOST USED)

Yeh intelligent hai. Pehle `\n\n` (paragraphs) pe split try karta hai. Agar chunk still too big hai, toh `\n` (lines) pe try karta hai. Phir ` ` (spaces) pe. Last resort `""` (characters) pe.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]  # Priority order
)
chunks = splitter.split_text(document_text)
```

**Kab use karo:** Default choice for 80% cases. General-purpose documents, blogs, articles, technical docs.

**Kyun best balance hai:**
- Paragraphs preserve karta hai (logical units)
- Agar paragraph bahut bada hai toh sentences pe fallback karta hai
- Predictable chunk sizes deta hai
- Simple to configure

---

### Strategy 3: Token-Based Splitting

Characters ki jagah **tokens** count karo — kyunki embedding models tokens mein sochte hain, characters mein nahi. "Kubernetes" = 1 word but 3 tokens. Agar character-based splitting use karo toh tumhara chunk embedding model ki limit cross kar sakta hai bina tumhe pata lage.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def chunk_by_tokens(text: str, chunk_size: int = 512, overlap: int = 64) -> list:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks
```

**Kab use karo:** Jab strict token budget hai (embedding model ki hard limit), cost optimization critical hai, ya multilingual content hai (non-English text mein character count misleading hota hai).

---

### Strategy 4: Sentence-Based Chunking

Sentences ko atomic unit maano. Group karo N sentences per chunk.

```python
from nltk.tokenize import sent_tokenize

def chunk_by_sentences(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list:
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks
```

**Kab use karo:** Legal documents, contracts, news articles — jahan har sentence independently meaningful hai.

**Guarantee:** Koi bhi sentence beech mein nahi kategi.

---

### Strategy 5: Structure-Aware (Markdown/HTML Headers)

Document ki existing structure use karo — headings, sections, chapters.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
# Each chunk has metadata: {"Header 1": "Chapter Name", "Header 2": "Section Name"}
```

**Kab use karo:** Well-formatted docs (Markdown, HTML, DOCX with proper headings). **Easiest biggest win** — agar docs structured hain toh pehle yeh try karo.

**Bonus:** Metadata automatically mil jaata hai (heading path) — filtering mein kaam aata hai.

---

### Strategy 6: Semantic Chunking (Advanced — Meaning-Based)

Topic change detect karo automatically. Consecutive sentences embed karo, jab similarity DROP ho, wahan split karo.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70  # Lower = more chunks, Higher = fewer chunks
)
chunks = splitter.create_documents([text])
```

**Internally kya hota hai:**
```
Sentence 1: "EKS runs managed control plane..."     ─┐
Sentence 2: "etcd stores cluster state..."           ─┤ similarity=0.85 → SAME CHUNK
Sentence 3: "Control plane scales automatically..."  ─┘

Sentence 4: "For pricing, see t3.large costs..."     ─┐ similarity=0.3 → NEW CHUNK!
Sentence 5: "Reserved instances save 40%..."         ─┘
```

**Kab use karo:** Knowledge bases, research papers, mixed-topic documents. Best accuracy (up to 70% retrieval improvement over naive baselines per benchmarks).

**Downside:** Computationally expensive (har sentence embed karna padta hai splitting ke liye). Variable chunk sizes (some chunks very small, some very large).

---

### Strategy 7: Parent-Child (Small-to-Large) Chunking

**Best of both worlds:** Small chunks for precise retrieval, large chunks for context.

```
Document
└── Parent chunk (1000 tokens) ← LLM ko yeh milta hai (context-rich)
    ├── Child chunk 1 (200 tokens) ← Retrieval isse match karti hai (precise)
    ├── Child chunk 2 (200 tokens)
    └── Child chunk 3 (200 tokens)
```

Flow:
1. Query embedding → child chunks se match hota hai (precise)
2. Match mila → parent chunk return karo (context-rich)
3. LLM ko parent chunk milta hai (enough context for good answer)

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**Kab use karo:** Complex Q&A jahan precise matching AUR rich context dono chahiye. E.g., technical documentation, troubleshooting guides.

---

### Strategy 8: Late Chunking (2024-2025 Innovation)

Normal chunking mein har chunk **independently** embed hota hai — context lost. Late chunking flips the process:

1. Pehle poora section/document ek **long-context embedding model** se encode karo (token-level embeddings milte hain)
2. Phir tokens ko pool karo chunks mein

```python
# Pseudocode
token_embeddings = long_context_model.encode(full_section, return_token_embeddings=True)
# token_embeddings shape: (seq_len, 1024)

# Define chunk boundaries
spans = [(0, 128), (96, 256), (224, 384)]  # overlapping spans

# Pool token embeddings per chunk
chunk_embeddings = [token_embeddings[s:e].mean(axis=0) for (s, e) in spans]
```

**Kyun powerful hai:** "This approach" jaise references ab resolve ho jaate hain kyunki embedding bante waqt surrounding context available tha.

**Kab use karo:** Long technical docs with cross-references, pronouns, header-dependent content.

**Requirement:** Long-context embedding model (e.g., Jina AI models with 8K+ context).

---

### Strategy 9: Contextual Retrieval (Anthropic, 2024)

Problem: Chunk akela padhne pe meaningless lagta hai.

```
Chunk: "It supports up to 5000 connections per second."
→ KYA support karta hai? Context lost!
```

Solution: Har chunk ke aage ek short context prepend karo **before embedding**:

```python
def contextualize_chunk(doc_title, heading_path, chunk_text):
    context = f"Title: {doc_title}\nSection: {heading_path}"
    return f"{context}\n\n{chunk_text}"

# Embed this contextualized version
contextualized = contextualize_chunk(
    "AWS EKS Guide",
    "Chapter 3 > Networking > Load Balancing",
    "It supports up to 5000 connections per second."
)
embedding = model.encode(contextualized)

# But STORE the raw chunk as retrieval payload (user sees clean text)
store_in_db(embedding=embedding, payload=original_chunk)
```

**Result:** Embedding mein context encoded hai, but user ko clean answer milta hai.

**Reference:** Anthropic Contextual Retrieval research (2024).

---

## Layer 3: Parameters — Chunk Size & Overlap

### Chunk Size Decision:

| Embedding Model Token Limit | Recommended Chunk Size | Reasoning |
|-----------------------------|----------------------|-----------|
| 256 tokens (MiniLM) | 200-240 tokens | Leave headroom |
| 512 tokens (BGE-large) | 400-480 tokens | Sweet spot |
| 8192 tokens (Titan V2, BGE-M3) | 512-1024 tokens | Don't use full window, precision drops |

**Rule:** Embedding model ki limit ke 80-90% tak jaao, 100% NAHI. Last tokens pe model ka attention weak hota hai.

**Larger chunks (1000+):**
- More context per chunk ✅
- But embedding becomes vague (too many topics mixed) ❌
- Retrieval precision drops ❌

**Smaller chunks (100-200):**
- Very precise retrieval ✅
- But may lack context (single sentence meaningless) ❌
- More chunks = more storage, slower search ❌

**Sweet spot for most production RAG:** **256-512 tokens**

### Overlap Decision:

```
chunk_overlap = 10-20% of chunk_size
```

| Chunk Size | Recommended Overlap |
|-----------|-------------------|
| 256 tokens | 25-50 tokens |
| 512 tokens | 50-100 tokens |
| 1024 tokens | 100-200 tokens |

**Kyun overlap chahiye:**
```
Without overlap:
Chunk 1: "...VPC peering does not support transitive"
Chunk 2: "routing. You need Transit Gateway instead."
→ Neither chunk has the complete answer!

With overlap (50 tokens):
Chunk 1: "...VPC peering does not support transitive routing. You need Transit Gateway..."
Chunk 2: "...does not support transitive routing. You need Transit Gateway instead. For..."
→ Both chunks have the complete thought!
```

---

## Layer 4: Metadata — The Hidden Multiplier

Chunks ke saath metadata store karo. Yeh retrieval mein filtering enable karta hai:

```python
point = {
    "id": "uuid",
    "vector": embedding,
    "payload": {
        "text": chunk_text,
        "source": "eks-user-guide.pdf",
        "page": 23,
        "section": "Networking > VPC-CNI",
        "chunk_index": 7,
        "total_chunks": 45,
        "timestamp": "2025-01-15",
        "doc_type": "official_docs"
    }
}
```

**Query time pe filter:**
```python
# Only search in networking section
results = vector_db.search(
    query_vector=query_emb,
    filter={"section": {"$contains": "Networking"}},
    top_k=5
)
```

**Kyun important hai:** Agar 10 lakh chunks hain aur user networking ke baare mein puch raha hai — filter lagao, sirf networking chunks mein search karo. Speed bhi badhti hai, relevance bhi.

---

## Layer 5: Content-Type Specific Chunking

| Content Type | Best Strategy | Why |
|-------------|--------------|-----|
| **Markdown docs** | Structure-aware (headers) | Logical sections already defined |
| **PDFs (unstructured)** | Recursive + sentence-based | No reliable structure, fallback to text |
| **Code** | Function/class-based | Each function = atomic unit |
| **Chat logs/transcripts** | Speaker turn-based | Preserve conversation flow |
| **Tables** | Keep whole table as 1 chunk | Splitting table = destroying data |
| **Legal contracts** | Sentence-based with clause detection | Each clause independently meaningful |
| **API documentation** | Endpoint-based | Each endpoint = self-contained unit |

**Code chunking example:**
```python
import ast

def chunk_python_code(code: str) -> list:
    """Split code by functions and classes"""
    tree = ast.parse(code)
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            chunk = ast.get_source_segment(code, node)
            if chunk:
                chunks.append(chunk)
    return chunks
```

**Table handling:**
```python
def is_table(text_block: str) -> bool:
    """Detect if block is a table (Markdown/ASCII)"""
    lines = text_block.strip().split("\n")
    pipe_lines = [l for l in lines if "|" in l]
    return len(pipe_lines) > 2  # Likely a table

def chunk_with_table_preservation(text: str, chunk_size: int = 512) -> list:
    blocks = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for block in blocks:
        if is_table(block):
            # Save current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Table = atomic chunk (never split)
            chunks.append(block)
        elif len(current_chunk) + len(block) < chunk_size:
            current_chunk += block + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = block + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
```

---

## Layer 6: Infra & Production Considerations

### Chunking Pipeline Architecture:

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Document   │───▶│  Parser    │───▶│  Chunker   │───▶│  Embedder  │───▶ Vector DB
│ Source     │    │ (PDF/MD/   │    │ (Recursive │    │ (Bedrock/  │
│ (S3/Local) │    │  HTML)     │    │  + Overlap)│    │  BGE)      │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

### Kubernetes Deployment (Chunking as Async Job):

Chunking is **batch processing** — not real-time. Deploy as a Kubernetes Job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: document-chunking-job
spec:
  template:
    spec:
      containers:
      - name: chunker
        image: your-registry/chunking-service:v1
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: S3_BUCKET
          value: "raw-documents"
        - name: CHUNK_SIZE
          value: "512"
        - name: CHUNK_OVERLAP
          value: "50"
        - name: VECTOR_DB_URL
          value: "http://qdrant:6333"
      restartPolicy: OnFailure
```

### Scaling Considerations:

| Documents | Strategy | Infra |
|-----------|----------|-------|
| < 1000 docs | Inline (same process as RAG API) | No separate infra needed |
| 1K - 100K docs | Async job (K8s Job / Lambda) | Separate from serving |
| 100K+ docs | Distributed pipeline (Airflow/Prefect + workers) | Dedicated workers, parallelized |

### Incremental Indexing (Production Critical):

Documents update hote hain. Poora re-index mat karo — sirf changed documents process karo:

```python
import hashlib

def incremental_ingest(doc_id, new_content, old_hash):
    new_hash = hashlib.md5(new_content.encode()).hexdigest()
    if new_hash == old_hash:
        return  # No change, skip

    # Delete old chunks for this doc
    vector_db.delete(filter={"doc_id": doc_id})

    # Re-chunk and re-embed
    chunks = chunker.split(new_content)
    embeddings = embedder.encode(chunks)
    vector_db.upsert(chunks, embeddings, metadata={"doc_id": doc_id})

    # Update hash in metadata store
    metadata_store.update(doc_id, hash=new_hash)
```

---

## Layer 7: Trade-offs & Decisions

### Decision Matrix:

| Scenario | Strategy | Chunk Size | Overlap |
|----------|----------|-----------|---------|
| Quick prototype | Recursive | 512 | 50 |
| Production (general docs) | Recursive + metadata | 512 | 50-100 |
| Structured docs (MD/HTML) | Structure-aware → Recursive (for large sections) | Per section | 50 |
| Maximum accuracy needed | Semantic | Variable | N/A |
| Complex Q&A (precision + context) | Parent-Child | Child: 200, Parent: 1000 | 20 |
| Documents with cross-references | Late chunking / Contextual | 512 | 50 |
| Mixed format (code + text + tables) | Hybrid | Variable | Variable |

### Common Mistake: "Bigger Chunks = Better Context" — WRONG

```
Big chunk (2000 tokens):
"EKS overview... networking... pricing... monitoring... troubleshooting..."

Query: "What's the pricing for EKS?"
Similarity score: 0.6 (diluted by all other topics)

Small chunk (300 tokens):
"EKS pricing: $0.10 per hour for control plane. Worker nodes charged separately..."

Query: "What's the pricing for EKS?"
Similarity score: 0.92 (precise match!)
```

---

## Layer 8: Production Pitfalls

### Pitfall 1: Token Limit Silent Truncation

Chunk 600 tokens ka hai, model 512 max accept karta hai → model silently last 88 tokens IGNORE karta hai. Tumhe pata nahi chalta. Important info last mein thi → embedding incomplete.

**Fix:** Always ensure `chunk_size < model_max_tokens`. Use token-based counting, not character counting.

### Pitfall 2: Tables/Lists Splitting

```
Original:
| Instance | vCPU | RAM |
| t3.small | 2    | 2GB |
| t3.medium| 2    | 4GB |

After naive chunking:
Chunk 1: "| Instance | vCPU | RAM |\n| t3.small"
Chunk 2: "| 2 | 2GB |\n| t3.medium | 2 | 4GB |"
```

Table completely destroyed. **Fix:** Detect tables, keep them as single atomic chunks.

### Pitfall 3: Chunk Too Small = No Context

```
Chunk: "It is 99.999999999%."
→ KYA 99.999999999% hai? S3 durability? Something else?
```

Without context, embedding is useless. **Fix:** Minimum chunk size enforce karo (at least 50-100 tokens). Or use contextual retrieval (prepend heading).

### Pitfall 4: No Overlap = Boundary Blindness

Important sentence exactly boundary pe hai → neither chunk captures it fully.

**Fix:** Always use 10-20% overlap. Non-negotiable in production.

### Pitfall 5: Re-chunking Without Re-indexing

Chunking strategy change kiya but purane chunks vector DB mein rehne diye. Naye aur purane chunks ka mix = inconsistent retrieval.

**Fix:** Strategy change = full re-index. No shortcuts.

---

## Layer 9: Interview Ready

### 2-Line Answer (Screening):

> "Chunking splits documents into smaller, meaningful pieces before embedding. The chunk size, overlap, and splitting strategy directly determine retrieval precision — it's the single highest-leverage decision in a RAG pipeline."

### 5-Min Answer (Technical Round):

> Above + strategies (recursive as default, semantic for accuracy, parent-child for complex Q&A), practical parameters (256-512 tokens, 10-20% overlap), metadata importance, token-based vs character-based counting, content-type specific approaches.

### 10-Min Deep Dive (System Design):

> Above + production architecture (async chunking pipeline, incremental indexing, hash-based change detection), scaling strategies, hybrid approaches for mixed-format docs, contextual retrieval, late chunking, evaluation (how to measure chunking quality via retrieval metrics), trade-off matrices.

### Expected Follow-up Questions:

**Q: "Document mein tables hain — kaise handle karoge?"**
A: Detect tables using parser, keep as single atomic chunk. Don't split. Add structured metadata (column headers as payload). For very large tables, split by rows but keep header row in every chunk.

**Q: "Chunk size kaise decide karoge naye project mein?"**
A: Start with 512 tokens (recursive splitting). Run evaluation with 50 test queries. Measure Recall@5. Then experiment: try 256 and 1024. Pick what gives best Recall@5. Usually 256-512 wins.

**Q: "1 million documents — chunking pipeline kaise design karoge?"**
A: Async distributed pipeline. Documents in S3 → SQS/EventBridge trigger → Lambda/ECS tasks parallel processing → chunks + embeddings → batch upsert to vector DB. Track document hashes for incremental updates. Monitor: chunk count growth, average chunk size, failed documents.

**Q: "Chunking strategy change karna ho production mein — kaise?"**
A: Blue-green approach. Create new collection with new strategy. Backfill in background. Once complete, switch traffic to new collection. Delete old. Zero downtime.

**Q: "Multilingual documents mein chunking kaise alag hogi?"**
A: Character count unreliable (Chinese: 1 char = 1 token, English: 1 word = 1-3 tokens). Must use token-based splitting with the correct tokenizer. Sentence detection needs language-specific models (not NLTK default). Use spaCy with language-specific model.

---

## Layer 10: 2026 Cutting Edge (Nice to Know)

| Technique | What It Does | Maturity |
|-----------|-------------|----------|
| **Contextual Retrieval** (Anthropic) | Prepend context before embedding each chunk | Production-ready |
| **Late Chunking** (Jina AI) | Encode full doc, then pool into chunk embeddings | Early production |
| **Cross-Granularity** | Index sentences, assemble chunks at query time | Research/experimental |
| **Agentic Chunking** | Use LLM to intelligently decide chunk boundaries | Expensive, experimental |

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| Why chunking matters | ✅ |
| 5 core strategies (fixed, recursive, token, sentence, structure) | ✅ |
| 4 advanced strategies (semantic, parent-child, late, contextual) | ✅ |
| Chunk size & overlap parameters with reasoning | ✅ |
| Metadata for filtering | ✅ |
| Content-type specific strategies (code, tables, chat, legal) | ✅ |
| Production code (all strategies) | ✅ |
| Infra/K8s deployment | ✅ |
| Scaling (small to million docs) | ✅ |
| Incremental indexing | ✅ |
| Trade-offs & decision matrix | ✅ |
| 5 production pitfalls | ✅ |
| Interview answers (all levels) | ✅ |
| 2026 cutting edge techniques | ✅ |

**Topic 2: Chunking Strategies — DONE. Nothing more to learn for your target role.**

---

## Layer 11: GitLab CI/CD + ArgoCD — Chunking Pipeline Deployment

Chunking ek **batch job** hai — real-time nahi. Documents S3 se aate hain, chunk hote hain, embed hote hain, vector DB mein store hote hain. Yeh Kubernetes CronJob ke through daily run hota hai, GitLab CI se image build hota hai, ArgoCD se deploy.

### Project Structure:

```
chunking-pipeline/
├── src/
│   ├── chunker.py           # Chunking logic
│   ├── pipeline.py          # S3 → Chunk → Embed → Vector DB
│   ├── requirements.txt
├── Dockerfile
├── k8s/
│   ├── cronjob.yaml         # Scheduled daily processing
│   ├── job.yaml             # One-time bulk reindex
│   ├── configmap.yaml
├── argocd/
│   └── application.yaml
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


def run():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="docs/"):
        for obj in page.get("Contents", []):
            content = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read().decode()
            process_document(obj["Key"], content, {"category": "general"})

if __name__ == "__main__":
    run()
```

### Step 2: Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .
CMD ["python", "pipeline.py"]
```

### Step 3: GitLab CI/CD (`.gitlab-ci.yml`)

```yaml
stages:
  - test
  - build
  - push

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/chunking-pipeline
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}

test:
  stage: test
  image: python:3.11-slim
  script:
    - pip install -r src/requirements.txt pytest
    - pytest tests/ -v

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
```

### Step 4: Kubernetes CronJob (`k8s/cronjob.yaml`)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: chunking-pipeline
  namespace: rag-system
spec:
  schedule: "0 2 * * *"        # Daily at 2 AM
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: chunker
            image: registry.gitlab.com/yourgroup/chunking-pipeline:latest
            envFrom:
            - configMapRef:
                name: chunking-config
            resources:
              requests: { memory: "2Gi", cpu: "1" }
              limits: { memory: "4Gi", cpu: "2" }
          restartPolicy: OnFailure
```

### Step 5: ArgoCD Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: chunking-pipeline
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/chunking-pipeline.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-system
  syncPolicy:
    automated: { prune: true, selfHeal: true }
```

### Step 6: Commands

```bash
# Manual trigger (test)
kubectl create job --from=cronjob/chunking-pipeline manual-run -n rag-system
kubectl logs -f job/manual-run -n rag-system

# Bulk reindex
kubectl apply -f k8s/job.yaml

# Check schedule
kubectl get cronjobs -n rag-system
```

### Flow:

```
S3 (raw docs) → CronJob (2AM daily) → Chunk → Embedding Service → Qdrant
                     ↑
    GitLab CI builds image on code push
    ArgoCD deploys updated CronJob
```

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Additional Sources:** LangCopilot 2026 Chunking Guide, Anthropic Contextual Retrieval Research, Jina AI Late Chunking (arXiv:2409.04701), FreeChunker Cross-Granularity (arXiv:2510.20356)
- **Extra added:** Production pitfalls, infra deployment, incremental indexing, content-type specific strategies, table/code handling, interview Q&A, 2026 techniques, GitLab CI/CD + ArgoCD deployment — not in original curriculum
