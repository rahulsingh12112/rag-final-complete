# 📘 RAG Fundamentals

## Overview
RAG ka basic flow samjho:

```
User Query → Embedding → Vector Search → Retrieve Chunks → LLM + Context → Answer
```

## Topics

### 01 - Embeddings
Text ko numerical vectors mein convert karna taaki machine samajh sake.

**Models to learn:**
- OpenAI `text-embedding-3-small` / `text-embedding-3-large`
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Cohere `embed-english-v3.0`

### 02 - Vector Databases
Embeddings store karna aur efficiently search karna.

**Databases to learn:**
- Pinecone (managed, easy to start)
- ChromaDB (local, open-source, great for prototyping)
- pgvector (PostgreSQL extension, good for existing Postgres users)
- Weaviate (feature-rich, hybrid search built-in)

### 03 - Chunking
Documents ko small, meaningful pieces mein todna.

**Strategies:**
- Fixed-size chunking (simple, 500-1000 tokens)
- Recursive character splitting (LangChain default)
- Semantic chunking (meaning-based splits)
- Document-based (headings, paragraphs respect karna)

### 04 - Retrieval
User query ke liye relevant chunks find karna.

**Methods:**
- Similarity Search (cosine similarity)
- MMR - Maximum Marginal Relevance (diversity + relevance balance)
- Hybrid (keyword BM25 + vector search)

### 05 - Generation
Retrieved context ko LLM ko deke answer generate karna.

**Key concepts:**
- Context injection (system prompt mein context dalna)
- Prompt engineering (instructions for accuracy)
- Citation/source tracking
- Handling "I don't know" cases
