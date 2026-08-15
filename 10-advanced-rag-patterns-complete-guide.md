# Topic 10: Advanced RAG Patterns — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Topics 1-9 complete (full RAG pipeline mastery)
> **Source:** Engineer Repo → mod-110-llm-infrastructure + 2026 Research Papers + Production Case Studies

---

## 🎯 One-Liner (Interview):

> "Advanced RAG patterns solve limitations of naive RAG — Agentic RAG uses LLM to decide retrieval strategy dynamically, Graph RAG leverages entity relationships for multi-hop reasoning, Corrective RAG self-heals bad retrievals, and Multi-Modal RAG handles images/tables alongside text — each pattern adds complexity but solves specific failure modes."

---

## Layer 1: Kyun Advanced Patterns Chahiye?

Naive RAG (retrieve → prompt → answer) fails in these scenarios:

| Failure Mode | Example | Why Naive RAG Fails |
|-------------|---------|-------------------|
| Multi-hop reasoning | "Compare EKS vs GKE pricing AND networking" | Needs info from multiple independent docs |
| Ambiguous queries | "Tell me about it" | Needs context resolution |
| No relevant docs | "What's the weather?" | Retrieves irrelevant, hallucinates |
| Relationship queries | "Which services depend on VPC?" | Needs graph/relationship understanding |
| Multi-modal | "Explain this architecture diagram" | Text-only retrieval misses images |
| Self-contradicting sources | Doc A says X, Doc B says Y | Naive RAG picks one randomly |

**Advanced patterns = specialized solutions for each failure mode.**

---

## Layer 2: Agentic RAG (LLM Decides Strategy)

### Concept:

Instead of fixed pipeline (always retrieve → always answer), LLM **decides** what to do:
- "Do I need to search? Or can I answer directly?"
- "Should I search once or multiple times?"
- "Should I search documents or call an API?"
- "Is the retrieved info sufficient or should I search again?"

```python
from enum import Enum
from typing import Optional

class Action(Enum):
    SEARCH = "search"
    ANSWER = "answer"
    CLARIFY = "clarify"
    SEARCH_AGAIN = "search_again"
    CALL_API = "call_api"

class AgenticRAG:
    """LLM decides retrieval strategy dynamically"""

    def __init__(self, retriever, llm, tools):
        self.retriever = retriever
        self.llm = llm
        self.tools = tools
        self.max_iterations = 5

    def process(self, query: str) -> dict:
        context = []
        actions_taken = []

        for iteration in range(self.max_iterations):
            # LLM decides next action
            action = self._decide_action(query, context, actions_taken)
            actions_taken.append(action)

            if action["type"] == Action.ANSWER:
                # Sufficient info — generate answer
                answer = self._generate_answer(query, context)
                return {"answer": answer, "iterations": iteration + 1,
                        "actions": actions_taken, "sources": context}

            elif action["type"] == Action.SEARCH:
                # Need more info — search
                search_query = action.get("search_query", query)
                results = self.retriever.search(search_query, top_k=5)
                context.extend(results)

            elif action["type"] == Action.SEARCH_AGAIN:
                # Previous search insufficient — refined search
                refined_query = action.get("refined_query")
                results = self.retriever.search(refined_query, top_k=5)
                context.extend(results)

            elif action["type"] == Action.CALL_API:
                # Need real-time data
                tool_name = action.get("tool")
                tool_args = action.get("args", {})
                result = self.tools[tool_name](**tool_args)
                context.append({"text": str(result), "source": f"API: {tool_name}"})

            elif action["type"] == Action.CLARIFY:
                return {"answer": action.get("clarification_question"),
                        "needs_clarification": True}

        # Max iterations reached
        return {"answer": self._generate_answer(query, context),
                "iterations": self.max_iterations, "warning": "max_iterations_reached"}

    def _decide_action(self, query: str, context: list, history: list) -> dict:
        """LLM decides what to do next"""
        context_summary = "\n".join([c.get("text", "")[:200] for c in context[-5:]])
        history_str = ", ".join([a["type"].value for a in history])

        prompt = f"""You are a search assistant. Decide the next action.

Query: {query}
Actions taken so far: {history_str}
Context gathered: {context_summary[:1000] if context else "None yet"}

Choose ONE action:
1. SEARCH - Need to search documents (provide search_query)
2. SEARCH_AGAIN - Previous search insufficient (provide refined_query)
3. ANSWER - Have enough info to answer
4. CALL_API - Need real-time data (provide tool name and args)
5. CLARIFY - Query is ambiguous (provide clarification question)

Respond in JSON: {{"type": "...", ...}}"""

        result = self.llm.generate(prompt, temperature=0)
        return json.loads(result)

    def _generate_answer(self, query: str, context: list) -> str:
        ctx = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(context[:8])])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer (cite sources):"
        return self.llm.generate(prompt)
```

### When to Use Agentic RAG:

| Scenario | Use Agentic? | Why |
|----------|-------------|-----|
| Simple factual Q&A | ❌ No | Overhead not justified |
| Complex multi-part questions | ✅ Yes | Needs multiple searches |
| Mixed (docs + real-time data) | ✅ Yes | Needs tool calling |
| Ambiguous queries | ✅ Yes | Can ask for clarification |
| High-volume, low-latency | ❌ No | Extra LLM calls = slow + expensive |

---

## Layer 3: Graph RAG (Relationship-Aware)

### Concept:

Documents contain entities with relationships. Graph RAG extracts these and uses them for reasoning.

```
Traditional RAG:
  "EKS" → finds docs mentioning "EKS"

Graph RAG:
  "EKS" → knows: EKS → uses → VPC-CNI
                  EKS → requires → IAM Roles
                  VPC-CNI → creates → Pod IPs
                  Pod IPs → belong to → Subnet
  → Can answer: "What networking components does EKS need?"
    by traversing the graph, not just keyword matching
```

### Implementation:

```python
from neo4j import GraphDatabase
import networkx as nx

class GraphRAG:
    """RAG with knowledge graph for relationship queries"""

    def __init__(self, vector_db, graph_db_uri, llm):
        self.vector_db = vector_db
        self.graph = GraphDatabase.driver(graph_db_uri)
        self.llm = llm

    def query(self, question: str) -> dict:
        # Step 1: Extract entities from question
        entities = self._extract_entities(question)

        # Step 2: Get graph context (relationships)
        graph_context = self._get_graph_context(entities)

        # Step 3: Also do vector search (traditional)
        vector_results = self.vector_db.search(question, top_k=5)

        # Step 4: Combine graph + vector context
        combined_context = self._merge_contexts(graph_context, vector_results)

        # Step 5: Generate answer with enriched context
        answer = self._generate(question, combined_context)

        return {"answer": answer, "entities": entities,
                "graph_paths": graph_context, "vector_sources": vector_results}

    def _extract_entities(self, text: str) -> list:
        """Use LLM to extract key entities"""
        prompt = f"""Extract key technical entities from this question.
Return as JSON array of strings.

Question: {text}
Entities:"""
        result = self.llm.generate(prompt, temperature=0)
        return json.loads(result)

    def _get_graph_context(self, entities: list) -> list:
        """Query knowledge graph for relationships"""
        paths = []
        with self.graph.session() as session:
            for entity in entities:
                # Find entity and its neighbors (2 hops)
                result = session.run("""
                    MATCH (e {name: $entity})-[r*1..2]-(connected)
                    RETURN e.name, type(r[0]), connected.name, 
                           connected.description
                    LIMIT 20
                """, entity=entity)

                for record in result:
                    paths.append({
                        "from": record["e.name"],
                        "relation": record["type(r[0])"],
                        "to": record["connected.name"],
                        "description": record.get("connected.description", "")
                    })
        return paths

    def _merge_contexts(self, graph_ctx, vector_ctx):
        """Combine graph relationships with vector search results"""
        context_parts = []

        # Graph context (relationships)
        if graph_ctx:
            graph_text = "Known relationships:\n"
            for path in graph_ctx:
                graph_text += f"- {path['from']} --[{path['relation']}]--> {path['to']}\n"
            context_parts.append(graph_text)

        # Vector context (documents)
        for doc in vector_ctx:
            context_parts.append(doc["text"])

        return "\n\n---\n\n".join(context_parts)
```

### Knowledge Graph Construction (Offline Pipeline):

```python
class KnowledgeGraphBuilder:
    """Extract entities + relationships from documents → build graph"""

    def __init__(self, llm, graph_db):
        self.llm = llm
        self.graph = graph_db

    def process_document(self, doc_text: str, doc_id: str):
        """Extract and store entity-relationship triples"""
        prompt = f"""Extract entity-relationship triples from this text.
Format: (entity1, relationship, entity2)

Text: {doc_text[:3000]}

Triples (JSON array of [entity1, relation, entity2]):"""

        result = self.llm.generate(prompt, temperature=0)
        triples = json.loads(result)

        # Store in graph DB
        with self.graph.session() as session:
            for e1, rel, e2 in triples:
                session.run("""
                    MERGE (a:Entity {name: $e1})
                    MERGE (b:Entity {name: $e2})
                    MERGE (a)-[r:RELATES {type: $rel, source: $doc}]->(b)
                """, e1=e1, e2=e2, rel=rel, doc=doc_id)
```

---

## Layer 4: Corrective RAG (Self-Healing)

### Concept:

After retrieval, check if results are actually relevant. If not → correct the retrieval before generating.

```
Normal RAG:     Query → Retrieve → Generate (even if retrieval was bad)
Corrective RAG: Query → Retrieve → EVALUATE → if bad → Re-retrieve/Web search → Generate
```

```python
class CorrectiveRAG:
    """Self-correcting RAG — detects and fixes bad retrievals"""

    def __init__(self, retriever, llm, web_search=None):
        self.retriever = retriever
        self.llm = llm
        self.web_search = web_search  # Fallback to web

    def query(self, question: str) -> dict:
        # Step 1: Initial retrieval
        docs = self.retriever.search(question, top_k=5)

        # Step 2: Grade each document (is it relevant?)
        graded = self._grade_documents(question, docs)
        relevant_docs = [d for d in graded if d["grade"] == "relevant"]
        irrelevant_docs = [d for d in graded if d["grade"] == "irrelevant"]

        # Step 3: Decide correction strategy
        if len(relevant_docs) >= 3:
            # Enough relevant docs — proceed normally
            strategy = "normal"
            final_docs = relevant_docs

        elif len(relevant_docs) >= 1:
            # Some relevant — supplement with refined search
            strategy = "supplement"
            refined_query = self._refine_query(question, irrelevant_docs)
            extra_docs = self.retriever.search(refined_query, top_k=3)
            final_docs = relevant_docs + extra_docs

        else:
            # Nothing relevant — web search fallback
            strategy = "web_fallback"
            if self.web_search:
                web_results = self.web_search.search(question, top_k=5)
                final_docs = web_results
            else:
                return {"answer": "I couldn't find relevant information in our documents.",
                        "strategy": strategy, "sources": []}

        # Step 4: Generate with corrected context
        answer = self._generate(question, final_docs)

        return {
            "answer": answer,
            "strategy": strategy,
            "relevant_count": len(relevant_docs),
            "total_retrieved": len(docs),
            "sources": final_docs
        }

    def _grade_documents(self, question: str, docs: list) -> list:
        """LLM grades each doc as relevant or irrelevant"""
        graded = []
        for doc in docs:
            prompt = f"""Is this document relevant to the question?
Question: {question}
Document: {doc['text'][:500]}

Answer ONLY "relevant" or "irrelevant":"""
            grade = self.llm.generate(prompt, temperature=0, max_tokens=10).strip().lower()
            graded.append({**doc, "grade": grade if grade in ["relevant", "irrelevant"] else "relevant"})
        return graded

    def _refine_query(self, question: str, irrelevant_docs: list) -> str:
        """Generate better search query based on what didn't work"""
        prompt = f"""The original search for "{question}" returned irrelevant results.
Irrelevant results were about: {', '.join([d['text'][:50] for d in irrelevant_docs[:3]])}

Generate a more specific search query that would find the right answer:"""
        return self.llm.generate(prompt, temperature=0, max_tokens=50).strip()
```

---

## Layer 5: Multi-Modal RAG (Images + Tables + Text)

### Concept:

Documents contain images, diagrams, tables — not just text. Multi-modal RAG handles all.

```python
class MultiModalRAG:
    """RAG that handles text, images, and tables"""

    def __init__(self, text_retriever, image_retriever, vision_llm, text_llm):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.vision_llm = vision_llm  # GPT-4o, Claude 3.5 (vision capable)
        self.text_llm = text_llm

    def query(self, question: str, image: bytes = None) -> dict:
        # Step 1: Text retrieval (standard)
        text_docs = self.text_retriever.search(question, top_k=5)

        # Step 2: Image retrieval (CLIP-based similarity)
        relevant_images = self.image_retriever.search(question, top_k=3)

        # Step 3: If user provided image, analyze it
        image_description = ""
        if image:
            image_description = self.vision_llm.describe_image(image)

        # Step 4: Combine all modalities
        context_parts = []

        # Text context
        for doc in text_docs:
            context_parts.append(f"[Text] {doc['text']}")

        # Image descriptions
        for img in relevant_images:
            desc = img.get("description", "")
            context_parts.append(f"[Image: {img['source']}] {desc}")

        if image_description:
            context_parts.append(f"[User's image] {image_description}")

        # Step 5: Generate with multi-modal context
        context = "\n\n".join(context_parts)
        answer = self.text_llm.generate(
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        return {"answer": answer, "text_sources": text_docs,
                "image_sources": relevant_images}
```

### Image Indexing Pipeline:

```python
class ImageIndexer:
    """Index images for retrieval using CLIP + VLM descriptions"""

    def __init__(self, clip_model, vision_llm, vector_db):
        self.clip = clip_model          # CLIP for image embeddings
        self.vlm = vision_llm           # Vision LLM for descriptions
        self.db = vector_db

    def index_image(self, image_path: str, metadata: dict):
        # 1. Generate CLIP embedding (for similarity search)
        image_embedding = self.clip.encode_image(image_path)

        # 2. Generate text description (for text-based retrieval)
        description = self.vlm.describe_image(
            image_path,
            prompt="Describe this technical diagram in detail. Include all components, connections, and labels."
        )

        # 3. Store both
        self.db.upsert(
            collection="images",
            id=image_path,
            vector=image_embedding,
            payload={"source": image_path, "description": description, **metadata}
        )
```

---

## Layer 6: Other Advanced Patterns

### Self-RAG (2024):

```python
# LLM decides: should I retrieve? Is retrieval helpful?
class SelfRAG:
    def query(self, question):
        # Token 1: [Retrieve] or [No Retrieve]
        need_retrieval = self.llm.decide_retrieval(question)
        
        if need_retrieval:
            docs = self.retriever.search(question)
            # Token 2: [Relevant] or [Irrelevant] per doc
            # Token 3: [Supported] or [Not Supported] for generated answer
            answer = self.generate_with_reflection(question, docs)
        else:
            answer = self.llm.generate(question)  # Direct answer
        
        return answer
```

### Adaptive RAG:

```python
# Route to different strategies based on query complexity
class AdaptiveRAG:
    def query(self, question):
        complexity = self.classify_complexity(question)
        
        if complexity == "simple":
            return self.naive_rag(question)        # Fast, 1 retrieval
        elif complexity == "moderate":
            return self.hybrid_rag(question)       # Dense + sparse + rerank
        elif complexity == "complex":
            return self.agentic_rag(question)      # Multi-step, tools
        elif complexity == "multi_hop":
            return self.graph_rag(question)        # Relationship traversal
```

### RAPTOR (Recursive Abstractive Processing):

```
Documents → Chunk → Cluster similar chunks → Summarize each cluster → 
Cluster summaries → Summarize again → ... → Tree of abstractions

Query → Search at MULTIPLE levels (detailed chunks + high-level summaries)
→ Better for broad questions that need overview + detail
```

---

## Layer 7: Production Pitfalls

### Pitfall 1: Agentic RAG = Unpredictable Latency

LLM decides to search 5 times → 5 retrieval calls + 5 LLM decisions = 15+ seconds.

**Fix:** Max iteration limit (3-5). Timeout per iteration. Fallback to simple RAG if too slow.

### Pitfall 2: Graph RAG = Expensive to Build & Maintain

Entity extraction requires LLM calls per document. Graph gets stale as docs update.

**Fix:** Incremental graph updates. Extract entities only from new/changed docs. Batch processing overnight.

### Pitfall 3: Corrective RAG = Extra LLM Costs

Grading 5 docs = 5 extra LLM calls before even generating answer.

**Fix:** Use cheap model for grading (GPT-4o-mini). Only grade when top retrieval score < threshold. Skip grading for high-confidence retrievals.

### Pitfall 4: Multi-Modal = Storage & Compute Heavy

Images need CLIP encoding (GPU), VLM descriptions (expensive LLM calls), separate index.

**Fix:** Lazy description generation (only when image retrieved). Image embedding cache. Thumbnail storage for previews.

### Pitfall 5: Over-Engineering

Using Agentic + Graph + Corrective + Multi-Modal when simple RAG works fine.

**Fix:** Start simple. Add complexity only when you can measure the failure mode it solves. Each pattern should improve a specific metric by a measurable amount.

---

## Layer 8: Decision Matrix — When to Use What

| Pattern | Use When | Complexity | Latency Impact | Cost Impact |
|---------|----------|-----------|---------------|-------------|
| **Naive RAG** | 80% of cases, simple Q&A | Low | Baseline | Baseline |
| **Hybrid (Dense+Sparse)** | Exact keywords matter | Low-Medium | +10ms | Same |
| **Corrective RAG** | Retrieval often irrelevant | Medium | +500ms-1s | +30% LLM |
| **Agentic RAG** | Complex multi-step queries | High | +2-10s | +200-500% LLM |
| **Graph RAG** | Relationship/dependency queries | High | +100ms search | Graph DB cost |
| **Multi-Modal** | Images/diagrams important | High | +200ms | VLM + storage |
| **Self-RAG** | Mix of knowledge + retrieval queries | Medium | Variable | +50% LLM |
| **Adaptive** | Mixed query types | Medium | Varies | Optimal routing |

---

## Layer 9: Interview Ready

### 2-Line Answer (Screening):

> "Advanced RAG patterns solve specific failure modes of naive RAG — Agentic RAG lets the LLM decide retrieval strategy dynamically, Graph RAG leverages entity relationships for multi-hop reasoning, Corrective RAG self-heals bad retrievals by grading and re-searching, and Multi-Modal handles images/tables alongside text."

### 5-Min Answer (Technical Round):

> Above + implementation details (Agentic with iteration loop, Graph with Neo4j entity extraction, Corrective with document grading), when to use what (decision matrix), practical trade-offs (latency, cost, complexity), Adaptive RAG for routing to appropriate pattern.

### 10-Min Deep Dive (System Design):

> Above + RAPTOR for hierarchical retrieval, Self-RAG reflection tokens, knowledge graph construction pipeline, multi-modal indexing (CLIP + VLM descriptions), production considerations (max iterations, timeouts, cost caps), evaluation per pattern, migration path (naive → hybrid → corrective → agentic as needs grow).

### Follow-up Questions:

**Q: "Graph RAG implement karna hai — knowledge graph kaise build karoge?"**
A: (1) NER on all documents (extract entities: services, concepts, people). (2) LLM extracts relationships between entities (triples). (3) Store in Neo4j/Neptune. (4) Incremental updates when docs change. (5) Query: extract entities from question → traverse graph 2 hops → combine with vector search results. Cost: ~$0.01/doc for extraction.

**Q: "Corrective RAG vs better retrieval — kab kya?"**
A: Better retrieval first (hybrid search, reranking). If recall@5 already > 0.9 but some queries still fail → Corrective RAG for those edge cases. Don't use Corrective RAG as substitute for bad retrieval — fix the root cause first.

**Q: "Agentic RAG mein LLM infinite loop mein fas gaya — handle kaise?"**
A: (1) Hard max iterations (5). (2) Timeout per iteration (10s). (3) If same action repeated 2x → force different action. (4) Budget cap (max 10 LLM calls per query). (5) Fallback to naive RAG on timeout.

**Q: "Multi-modal mein architecture diagrams samajhne hain — approach kya?"**
A: (1) During indexing: VLM (GPT-4o) describes each diagram in text. Store description + CLIP embedding. (2) During query: If query seems visual ("show architecture", "diagram"), search image index. (3) Return image reference + text description in response. (4) For user-uploaded images: VLM analyzes real-time, adds to context.

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| Agentic RAG (LLM-driven strategy) | ✅ |
| Graph RAG (entity relationships) | ✅ |
| Corrective RAG (self-healing retrieval) | ✅ |
| Multi-Modal RAG (images + tables) | ✅ |
| Self-RAG (reflection tokens) | ✅ |
| Adaptive RAG (complexity routing) | ✅ |
| RAPTOR (hierarchical) | ✅ |
| Knowledge graph construction | ✅ |
| Decision matrix (when to use what) | ✅ |
| Production pitfalls (5) | ✅ |
| Interview answers | ✅ |

**Topic 10: Advanced RAG Patterns — DONE.**

---
## Layer 12: GitLab CI/CD + ArgoCD — Advanced RAG Patterns Service Deployment

Advanced patterns service — supports Agentic, Graph, Corrective, Multi-Modal RAG — deployed as an extension to the base orchestrator. External access included.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
rag-advanced-patterns/
├── src/
│   ├── app.py                  # FastAPI — Advanced patterns API
│   ├── agentic.py              # Agentic RAG (multi-step)
│   ├── graph_rag.py            # Graph RAG (Neo4j)
│   ├── corrective.py           # Corrective RAG (self-healing)
│   ├── multimodal.py           # Multi-modal RAG
│   ├── adaptive_router.py      # Routes to appropriate pattern
│   ├── config.py
│   ├── requirements.txt
│   └── tests/
│       ├── test_agentic.py
│       ├── test_corrective.py
│       └── test_router.py
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml
│   └── neo4j-statefulset.yaml  # Graph DB
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: Advanced RAG API (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from adaptive_router import AdaptiveRouter
from agentic import AgenticRAG
from corrective import CorrectiveRAG
from graph_rag import GraphRAG
from config import settings
import os
import time

app = FastAPI(title="Advanced RAG Patterns Service", version="1.0.0")

# Metrics
PATTERN_USED = Counter("rag_advanced_pattern_total", "Pattern usage", ["pattern"])
PATTERN_LATENCY = Histogram("rag_advanced_latency_seconds", "Pattern latency", ["pattern"])
ITERATIONS_USED = Histogram("rag_advanced_iterations", "Agentic iterations",
                            buckets=[1, 2, 3, 4, 5])

router = AdaptiveRouter(settings)


class AdvancedQueryRequest(BaseModel):
    query: str
    pattern: str | None = None    # Force specific pattern (or auto-route)
    top_k: int = 5
    max_iterations: int = 5       # For agentic
    include_graph: bool = False   # Force graph lookup


class AdvancedQueryResponse(BaseModel):
    answer: str
    pattern_used: str
    sources: list[dict]
    metadata: dict


@app.get("/health")
def health():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    checks = router.check_dependencies()
    if not checks.get("retrieval"):
        raise HTTPException(503, f"Dependencies: {checks}")
    return {"status": "ready", "dependencies": checks}


@app.post("/query", response_model=AdvancedQueryResponse)
def advanced_query(request: AdvancedQueryRequest):
    """Advanced RAG — auto-routes or uses specified pattern"""
    start = time.time()

    # Determine pattern
    pattern = request.pattern or router.classify(request.query)
    PATTERN_USED.labels(pattern=pattern).inc()

    try:
        result = router.execute(
            pattern=pattern,
            query=request.query,
            top_k=request.top_k,
            max_iterations=request.max_iterations,
            include_graph=request.include_graph
        )

        latency = time.time() - start
        PATTERN_LATENCY.labels(pattern=pattern).observe(latency)

        if "iterations" in result:
            ITERATIONS_USED.observe(result["iterations"])

        return AdvancedQueryResponse(
            answer=result["answer"],
            pattern_used=pattern,
            sources=result.get("sources", []),
            metadata={
                "latency_ms": round(latency * 1000, 2),
                "iterations": result.get("iterations", 1),
                "strategy": result.get("strategy", "direct"),
                "graph_entities": result.get("entities", []),
                "corrective_action": result.get("corrective_action"),
            }
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query/agentic")
def agentic_query(request: AdvancedQueryRequest):
    """Force agentic pattern"""
    request.pattern = "agentic"
    return advanced_query(request)


@app.post("/query/corrective")
def corrective_query(request: AdvancedQueryRequest):
    """Force corrective pattern"""
    request.pattern = "corrective"
    return advanced_query(request)


@app.post("/query/graph")
def graph_query(request: AdvancedQueryRequest):
    """Force graph RAG pattern"""
    request.pattern = "graph"
    return advanced_query(request)


@app.get("/patterns")
def list_patterns():
    """Available patterns and their descriptions"""
    return {
        "naive": "Simple retrieve + generate",
        "corrective": "Grade retrieval, re-search if bad",
        "agentic": "LLM decides strategy dynamically (multi-step)",
        "graph": "Entity relationship traversal + vector search",
        "adaptive": "Auto-routes to best pattern based on query"
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
```

### Step 2: Adaptive Router (`src/adaptive_router.py`)

```python
class AdaptiveRouter:
    """Routes queries to the optimal RAG pattern"""

    def __init__(self, settings):
        self.settings = settings
        self.agentic = AgenticRAG(settings)
        self.corrective = CorrectiveRAG(settings)
        self.graph = GraphRAG(settings)

    def classify(self, query: str) -> str:
        """Classify query complexity → pick pattern"""
        query_lower = query.lower()

        # Graph indicators
        graph_signals = ["depends on", "related to", "connected to",
                        "what uses", "what requires", "relationship between"]
        if any(s in query_lower for s in graph_signals):
            return "graph"

        # Agentic indicators (complex, multi-part)
        agentic_signals = ["compare", "step by step", "first...then",
                          "multiple", "and also", "additionally"]
        if any(s in query_lower for s in agentic_signals) or len(query.split()) > 25:
            return "agentic"

        # Default: corrective (self-healing, safe default)
        return "corrective"

    def execute(self, pattern: str, query: str, **kwargs) -> dict:
        if pattern == "agentic":
            return self.agentic.process(query, max_iterations=kwargs.get("max_iterations", 5))
        elif pattern == "graph":
            return self.graph.query(query)
        elif pattern == "corrective":
            return self.corrective.query(query)
        else:
            # Fallback: corrective (safest)
            return self.corrective.query(query)

    def check_dependencies(self) -> dict:
        return {
            "retrieval": self._check_url(f"{self.settings.RETRIEVAL_URL}/health"),
            "llm": self._check_url(f"{self.settings.LLM_URL}/health"),
            "graph_db": self._check_url(f"{self.settings.GRAPH_DB_URL}"),
        }

    def _check_url(self, url):
        try:
            import requests
            return requests.get(url, timeout=5).status_code == 200
        except:
            return False
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
     "--bind", "0.0.0.0:8000", "--timeout", "180", "--access-logfile", "-"]
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
  - approval_gate
  - prod_deploy_canary
  - monitor
  - prod_deploy_full

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-advanced
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://rag-advanced-staging.yourdomain.com"
  PROD_URL: "https://rag-advanced.yourdomain.com"

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
    - kubectl set image deployment/rag-advanced
        rag-advanced=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/rag-advanced -n rag-staging --timeout=300s
  only: [main]

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
      print('✓ Health')

      # Patterns list
      r = requests.get(f'{URL}/patterns')
      assert r.status_code == 200
      print(f'✓ Patterns: {list(r.json().keys())}')

      # Auto-routed query
      r = requests.post(f'{URL}/query', json={'query': 'How does EKS work?', 'top_k': 3})
      assert r.status_code == 200
      data = r.json()
      assert len(data['answer']) > 10
      print(f'✓ Query OK: pattern={data[\"pattern_used\"]}, {data[\"metadata\"][\"latency_ms\"]}ms')

      # Corrective
      r = requests.post(f'{URL}/query/corrective', json={'query': 'EKS pricing details', 'top_k': 3})
      assert r.status_code == 200
      print(f'✓ Corrective: strategy={r.json()[\"metadata\"].get(\"strategy\")}')

      print('=== ALL SMOKE TESTS PASSED ===')
      "
  only: [main]

approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging passed."
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
        name: rag-advanced-canary
        namespace: rag-production
        labels: { app: rag-advanced, track: canary }
      spec:
        replicas: 1
        selector:
          matchLabels: { app: rag-advanced, track: canary }
        template:
          metadata:
            labels: { app: rag-advanced, track: canary }
          spec:
            containers:
            - name: rag-advanced
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports: [{ containerPort: 8000 }]
              envFrom: [{ configMapRef: { name: rag-advanced-config } }]
              resources:
                requests: { memory: "1Gi", cpu: "500m" }
                limits: { memory: "2Gi", cpu: "1" }
              livenessProbe: { httpGet: { path: /health, port: 8000 }, initialDelaySeconds: 10 }
              readinessProbe: { httpGet: { path: /ready, port: 8000 }, initialDelaySeconds: 5 }
      EOF
    - kubectl rollout status deployment/rag-advanced-canary -n rag-production --timeout=300s
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
          except: FAILURES += 1
          if FAILURES >= 3: print('CANARY FAILED'); sys.exit(1)
          time.sleep(30)
      print('=== CANARY HEALTHY ===')
      "
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        kubectl delete deployment rag-advanced-canary -n rag-production --ignore-not-found
      fi
  only: [main]

deploy_prod_full:
  stage: prod_deploy_full
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/rag-advanced
        rag-advanced=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/rag-advanced -n rag-production --timeout=600s
    - kubectl delete deployment rag-advanced-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100%: ${PROD_URL} ==="
  only: [main]
```

### Step 5: Kubernetes + Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-advanced-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "50"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "180"   # Agentic can be slow
spec:
  tls:
  - hosts: [rag-advanced.yourdomain.com]
    secretName: rag-advanced-tls
  rules:
  - host: rag-advanced.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service: { name: rag-advanced, port: { number: 80 } }
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-advanced-config
  namespace: rag-production
data:
  RETRIEVAL_URL: "http://rag-retrieval-service.rag-production.svc.cluster.local"
  LLM_URL: "http://llm-gateway.rag-production.svc.cluster.local"
  GRAPH_DB_URL: "bolt://neo4j.rag-production.svc.cluster.local:7687"
  MAX_ITERATIONS: "5"
  AGENTIC_TIMEOUT: "30"
```

### Step 6: Operations

```bash
# External access
curl https://rag-advanced.yourdomain.com/health
curl https://rag-advanced.yourdomain.com/patterns

# Auto-routed (system picks best pattern)
curl -X POST https://rag-advanced.yourdomain.com/query \
  -d '{"query": "Compare EKS and GKE pricing and networking", "top_k": 5}'

# Force agentic (multi-step)
curl -X POST https://rag-advanced.yourdomain.com/query/agentic \
  -d '{"query": "What is the total cost of EKS + ALB + RDS in us-east-1?", "max_iterations": 5}'

# Force graph (relationship)
curl -X POST https://rag-advanced.yourdomain.com/query/graph \
  -d '{"query": "What services depend on VPC in our infrastructure?"}'

# Force corrective (self-healing)
curl -X POST https://rag-advanced.yourdomain.com/query/corrective \
  -d '{"query": "Error EKS-AUTH-403 resolution steps"}'
```

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Research Papers:** Self-RAG (2024), RAPTOR (2024), Corrective RAG (CRAG 2024), Graph RAG (Microsoft 2024)
- **Extra added:** Agentic RAG with iteration control, Graph RAG with Neo4j, Corrective RAG with grading, Multi-Modal with CLIP, Adaptive routing, production deployment — not in original curriculum

---

## 🎉 RAG CURRICULUM COMPLETE — ALL 10 TOPICS DONE

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPLETE RAG MASTERY ROADMAP                    │
│                                                                 │
│  ✅ Topic 01: Embedding Models                                   │
│  ✅ Topic 02: Chunking Strategies                                │
│  ✅ Topic 03: Vector Databases                                   │
│  ✅ Topic 04: Retrieval Techniques                               │
│  ✅ Topic 05: Prompt Engineering & Context Assembly               │
│  ✅ Topic 06: LLM Integration & Orchestration                    │
│  ✅ Topic 07: RAG Evaluation & Observability                     │
│  ✅ Topic 08: Guardrails & Safety                                │
│  ✅ Topic 09: End-to-End Deployment & Scaling                    │
│  ✅ Topic 10: Advanced RAG Patterns                              │
│                                                                 │
│  Each topic includes:                                           │
│  • Deep conceptual understanding (Layers 1-9)                   │
│  • Production code                                              │
│  • Interview answers (2-line, 5-min, 10-min)                    │
│  • Full CI/CD pipeline (Layer 12):                              │
│    test → lint → security → build → push → staging →            │
│    smoke test → approval → canary 10% → monitor → full 100%    │
│  • External access (Ingress + TLS)                              │
│  • Kubernetes manifests                                         │
│  • ArgoCD GitOps                                                │
└─────────────────────────────────────────────────────────────────┘
```
