# 🔬 Advanced RAG

## Overview
Basic RAG mein problems aati hain production mein. Advanced techniques se accuracy 44% → 63%+ badhti hai.

```
User Query → Query Transform → Hybrid Retrieval → Rerank → Filter → LLM → Evaluate → Answer
```

## Topics

### 01 - Reranking
Retrieved chunks ko re-order karna — best results top pe laana.

**Tools:**
- Cohere Rerank API
- Cross-encoder models (HuggingFace)
- ColBERT reranking

**Why needed:** Vector search sometimes irrelevant results top pe laata hai. Reranker fix karta hai.

### 02 - Query Transformation
User ki query ko better form mein convert karna for improved retrieval.

**Techniques:**
- HyDE (Hypothetical Document Embeddings) - pehle hypothetical answer generate karo, fir usse search karo
- Multi-query - ek question ke multiple versions banao
- Step-back prompting - broader question puchho pehle
- Query decomposition - complex query ko parts mein todo

### 03 - Hybrid Search
Keyword search (BM25) + Vector search combine karna.

**Why:** 
- Vector search: meaning samajhta hai but exact keywords miss karta hai
- BM25: exact words match karta hai but meaning nahi samajhta
- Dono combine = best results

### 04 - Self-Correcting RAG
Answer galat ya insufficient ho toh automatically retry karna.

**Flow:**
1. Retrieve → Generate answer
2. Check: Kya answer context se supported hai?
3. Nahi → Different retrieval strategy try karo
4. Retry with better context

### 05 - Evaluation
RAG system ki quality measure karna with metrics.

**Frameworks:**
- RAGAS (Retrieval Augmented Generation Assessment)
- DeepEval

**Key Metrics:**
- Faithfulness (answer context se match karta hai?)
- Answer Relevancy (answer question ka jawab hai?)
- Context Precision (retrieved chunks relevant hain?)
- Context Recall (saari zaroori info retrieve hui?)

### 06 - Metadata Filtering
Retrieval ke time specific filters apply karna.

**Example:** 
- Sirf 2024 ke baad ke documents se search karo
- Sirf "finance" category ke documents mein dekho
- Sirf specific author ke documents

### 07 - Parent-Child Document Retrieval
Chhote chunks retrieve karo (precision), but full context return karo (completeness).

**How:**
- Documents ko small chunks mein todo (child)
- Har child ka parent (bigger chunk) store karo
- Search small chunks mein karo
- Return parent chunk (more context)
