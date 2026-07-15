# 🚀 RAG - Retrieval Augmented Generation (Complete Learning Path)

A comprehensive, hands-on repository to master RAG from fundamentals to production-grade agentic systems.

## 📂 Repository Structure

```
RAG/
├── 01-RAG-Fundamentals/
│   ├── 01-Embeddings/          (OpenAI, HuggingFace, Cohere)
│   ├── 02-Vector-Databases/    (Pinecone, ChromaDB, pgvector, Weaviate)
│   ├── 03-Chunking/            (fixed, recursive, semantic, document-based)
│   ├── 04-Retrieval/           (similarity search, MMR, hybrid)
│   └── 05-Generation/          (context injection, prompt engineering)
│
├── 02-Advanced-RAG/
│   ├── 01-Reranking/           (Cohere Rerank, cross-encoder)
│   ├── 02-Query-Transformation/ (HyDE, multi-query)
│   ├── 03-Hybrid-Search/       (BM25 + vector)
│   ├── 04-Self-Correcting-RAG/ (check → retry → refine)
│   ├── 05-Evaluation/          (RAGAS, faithfulness, relevancy scores)
│   ├── 06-Metadata-Filtering/
│   └── 07-Parent-Child-Retrieval/
│
├── 03-Agentic-RAG/
│   ├── 01-Query-Planning/      (complex questions todna)
│   ├── 02-Multi-Step-Retrieval/
│   ├── 03-Retrieval-Grading/   (retrieved content accha hai ya nahi?)
│   ├── 04-Self-Correction-Loops/
│   └── 05-Multiple-Knowledge-Sources/
│
└── README.md
```

## 📚 Learning Path

### Phase 1: RAG Fundamentals (Week 1-3)
| Topic | What You'll Learn |
|-------|------------------|
| Embeddings | Text ko numbers mein convert karna, different embedding models |
| Vector Databases | Embeddings store karna aur search karna |
| Chunking | Documents ko chhote pieces mein todna |
| Retrieval | Relevant chunks find karna |
| Generation | Retrieved context se answer generate karna |

### Phase 2: Advanced RAG (Week 4-5)
| Topic | What You'll Learn |
|-------|------------------|
| Reranking | Retrieved results ko re-order karna for better accuracy |
| Query Transformation | User query ko better form mein convert karna |
| Hybrid Search | Keyword + Semantic search combine karna |
| Self-Correcting RAG | Answer galat ho toh retry karna |
| Evaluation | RAG system ki quality measure karna |
| Metadata Filtering | Specific filters lagake search narrow karna |
| Parent-Child Retrieval | Small chunks retrieve, full context return |

### Phase 3: Agentic RAG (Week 6-7)
| Topic | What You'll Learn |
|-------|------------------|
| Query Planning | Complex questions ko sub-questions mein todna |
| Multi-Step Retrieval | Multiple rounds mein information gather karna |
| Retrieval Grading | Retrieved content useful hai ya nahi — judge karna |
| Self-Correction Loops | Automatic retry with different strategies |
| Multiple Knowledge Sources | Multiple databases/APIs se data orchestrate karna |

## 🛠️ Tech Stack

- **Language:** Python
- **Frameworks:** LangChain, LlamaIndex
- **Embedding Models:** OpenAI, HuggingFace, Cohere
- **Vector DBs:** Pinecone, ChromaDB, pgvector, Weaviate
- **LLMs:** OpenAI GPT, Claude, Open-source (Llama, Mistral)
- **Evaluation:** RAGAS, DeepEval

## 🎯 Projects

1. **Basic Q&A Chatbot** - PDF documents pe simple RAG
2. **Multi-document RAG** - Multiple sources se answer
3. **Production RAG Pipeline** - With evaluation & monitoring
4. **Agentic RAG System** - Self-correcting, multi-step retrieval

## 📖 Recommended Resources

### Free Courses
- [Building and Evaluating Advanced RAG - DeepLearning.AI](https://www.deeplearning.ai/short-courses/)
- [LangChain: Chat with Your Data - DeepLearning.AI](https://www.deeplearning.ai/short-courses/)
- [Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/)
- [Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/)

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [ChromaDB Docs](https://docs.trychroma.com/)

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/snghptm/RAG.git
cd RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## 📝 Progress Tracker

- [ ] 01-RAG-Fundamentals
  - [ ] Embeddings
  - [ ] Vector Databases
  - [ ] Chunking
  - [ ] Retrieval
  - [ ] Generation
- [ ] 02-Advanced-RAG
  - [ ] Reranking
  - [ ] Query Transformation
  - [ ] Hybrid Search
  - [ ] Self-Correcting RAG
  - [ ] Evaluation
  - [ ] Metadata Filtering
  - [ ] Parent-Child Retrieval
- [ ] 03-Agentic-RAG
  - [ ] Query Planning
  - [ ] Multi-Step Retrieval
  - [ ] Retrieval Grading
  - [ ] Self-Correction Loops
  - [ ] Multiple Knowledge Sources

---

⭐ Star this repo if you find it helpful!
