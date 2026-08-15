# Topic 5: Prompt Engineering & Context Assembly — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Embedding (Topic 1), Chunking (Topic 2), Vector DB (Topic 3), Retrieval (Topic 4)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/03-rag-systems.md + 2026 Prompt Engineering Best Practices

---

## 🎯 One-Liner (Interview):

> "Context Assembly wo process hai jisme retrieved chunks ko LLM ke liye optimal prompt mein arrange karte hain — chunk ordering, token budget management, system instructions, aur output formatting sab milke determine karta hai ki LLM kitna accurate answer dega."

---

## Layer 1: Kya Hai Aur Kyun Zaroori Hai?

Tumne embedding se query encode kiya, vector DB se relevant chunks retrieve kiye, reranker se top-5 select kiye. **Ab kya?**

Ab yeh 5 chunks LLM ko dene hain taaki wo answer generate kare. Lekin LLM ko seedha chunks dump kar doge toh:
- Answer vague hoga
- Hallucination hoga (apne se bana lega)
- Source cite nahi karega
- Format inconsistent hoga

**Context Assembly = Art of arranging information for the LLM.**

Yeh RAG pipeline ka **last mile** hai — agar yeh galat hai toh baaki sab (embedding, chunking, retrieval) bekar hai. Best retrieved docs bhi useless hain agar prompt galat hai.

**Key insight:** LLM ek function hai — `f(prompt) → output`. Prompt ki quality directly output quality determine karti hai. Same 5 chunks ko 10 different ways arrange karo → 10 different quality answers milenge.

---

## Layer 2: Prompt Architecture (Production RAG)

### The 4-Part Prompt Structure:

```
┌─────────────────────────────────────────────┐
│ 1. SYSTEM INSTRUCTION                       │ ← Who you are, how to behave
├─────────────────────────────────────────────┤
│ 2. CONTEXT (Retrieved Chunks)               │ ← Grounding information
├─────────────────────────────────────────────┤
│ 3. USER QUERY                               │ ← What user wants to know
├─────────────────────────────────────────────┤
│ 4. OUTPUT FORMAT INSTRUCTION                │ ← How to structure the answer
└─────────────────────────────────────────────┘
```

### Production Prompt Template:

```python
def build_rag_prompt(query: str, retrieved_chunks: list, config: dict) -> str:
    """Production RAG prompt builder"""

    # Part 1: System instruction
    system = """You are a technical documentation assistant. Answer questions 
ONLY based on the provided context. Follow these rules strictly:
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- NEVER make up information not present in the context.
- Cite sources using [Source N] format.
- Be concise but complete.
- If multiple sources conflict, mention the discrepancy."""

    # Part 2: Context assembly
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_info = chunk.get("metadata", {}).get("source", "Unknown")
        context_parts.append(f"[Source {i}] (from: {source_info})\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)

    # Part 3: Query
    # Part 4: Output format
    output_instruction = """Provide a clear, structured answer. 
Use bullet points for lists. Cite sources with [Source N]."""

    # Assemble full prompt
    prompt = f"""{system}

## Context:
{context}

## Question:
{query}

## Instructions:
{output_instruction}

## Answer:"""

    return prompt
```

---

## Layer 3: Context Window Management (Token Budget)

### The Problem:

LLM ki context window limited hai:
- GPT-4o: 128K tokens
- Claude 3.5 Sonnet: 200K tokens
- Llama 3.1 70B: 128K tokens
- AWS Nova Pro: 300K tokens

Lekin **zyada context = worse performance**. Research (2024-2025) shows:

```
┌────────────────────────────────────────────────────────┐
│ "Lost in the Middle" Problem (Stanford/Google, 2024)   │
│                                                        │
│ LLMs attend most to:                                   │
│   1. Beginning of context (primacy bias)               │
│   2. End of context (recency bias)                     │
│   3. Middle gets IGNORED                               │
│                                                        │
│ Relevance score by position:                           │
│ ████████░░░░░░░░░░░░░░░░░░████████                    │
│ ^Beginning    Middle(weak)    End^                     │
└────────────────────────────────────────────────────────┘
```

### Token Budget Strategy:

```python
import tiktoken

class TokenBudgetManager:
    """Manage token allocation for RAG prompts"""
    
    def __init__(self, model: str = "gpt-4o", max_output_tokens: int = 1024):
        self.encoder = tiktoken.encoding_for_model(model)
        self.model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "claude-3.5-sonnet": 200000,
            "nova-pro": 300000,
            "llama-3.1-70b": 128000,
        }
        self.max_context = self.model_limits.get(model, 128000)
        self.max_output = max_output_tokens
        
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def allocate_budget(self, system_prompt: str, query: str, 
                        chunks: list, max_chunks: int = 10) -> list:
        """
        Token budget allocation:
        - System prompt: fixed
        - Query: fixed  
        - Output reserve: fixed (max_output_tokens)
        - Context: whatever remains
        """
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        reserved = system_tokens + query_tokens + self.max_output + 200  # 200 buffer
        
        available_for_context = self.max_context - reserved
        
        # Greedily add chunks until budget exhausted
        selected_chunks = []
        used_tokens = 0
        
        for chunk in chunks[:max_chunks]:
            chunk_tokens = self.count_tokens(chunk["text"])
            if used_tokens + chunk_tokens > available_for_context:
                break
            selected_chunks.append(chunk)
            used_tokens += chunk_tokens
        
        return selected_chunks
```

### Production Token Budget:

| Component | Tokens | % of Budget |
|-----------|--------|-------------|
| System instruction | 200-500 | 1-2% |
| Context (retrieved chunks) | 2000-8000 | 60-80% |
| User query | 50-200 | 1-2% |
| Output reserve | 512-2048 | 10-20% |
| Buffer/formatting | 200 | ~1% |

**Rule:** Context ko **60-80% budget** do. Zyada nahi — "Lost in the Middle" effect badhega.

**Sweet spot for most RAG:** 3000-5000 tokens of context (5-10 chunks of 300-500 tokens each).

---

## Layer 4: Chunk Ordering Strategies

### Strategy 1: Relevance-First (Default)

Most relevant chunk sabse pehle. Simple, works 80% of time.

```python
def order_by_relevance(chunks: list) -> list:
    """Highest relevance score first"""
    return sorted(chunks, key=lambda x: x["score"], reverse=True)
```

### Strategy 2: Reverse Relevance (Lost-in-Middle Fix)

Most relevant chunks at START and END. Least relevant in middle.

```python
def order_lost_in_middle_fix(chunks: list) -> list:
    """
    Place most relevant at start AND end.
    Least relevant in middle (where LLM pays least attention).
    """
    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    
    if len(sorted_chunks) <= 2:
        return sorted_chunks
    
    # Split: odd-indexed go to end (reversed), even-indexed stay at start
    result = []
    end_chunks = []
    for i, chunk in enumerate(sorted_chunks):
        if i % 2 == 0:
            result.append(chunk)
        else:
            end_chunks.append(chunk)
    
    result.extend(reversed(end_chunks))
    return result
```

### Strategy 3: Chronological (Time-Sensitive)

For docs with timestamps — arrange by date.

```python
def order_chronological(chunks: list) -> list:
    """For time-sensitive content — latest first"""
    return sorted(chunks, key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
```

### Strategy 4: Document-Grouped

Same document ke chunks saath raho (context continuity).

```python
def order_grouped_by_source(chunks: list) -> list:
    """Group chunks from same document together"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for chunk in chunks:
        source = chunk["metadata"].get("source", "unknown")
        groups[source].append(chunk)
    
    # Sort groups by best score in group, then chunks by position
    ordered = []
    sorted_groups = sorted(groups.items(), 
                          key=lambda x: max(c["score"] for c in x[1]), 
                          reverse=True)
    for source, group_chunks in sorted_groups:
        group_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        ordered.extend(group_chunks)
    
    return ordered
```

### When to Use What:

| Scenario | Ordering | Why |
|----------|----------|-----|
| General Q&A | Relevance-first | Simple, effective |
| Long context (8+ chunks) | Lost-in-middle fix | Prevents middle-blindness |
| "What happened recently?" | Chronological | Time matters |
| Technical docs (multi-section) | Document-grouped | Maintains coherence |

---

## Layer 5: Advanced Prompt Patterns

### Pattern 1: Citation-Enforced Prompt

```python
CITATION_PROMPT = """Answer the question based ONLY on the provided sources.
Every factual claim MUST have a citation in [Source N] format.
If you cannot find the answer in the sources, respond with:
"Based on the available documentation, I cannot find information about this topic."

Sources:
{context}

Question: {query}

Answer (with citations):"""
```

### Pattern 2: Step-by-Step Reasoning (Chain-of-Thought)

```python
COT_PROMPT = """You are an expert assistant. Use the following process:

1. First, identify which sources are relevant to the question.
2. Extract key facts from relevant sources.
3. Reason through the answer step by step.
4. Provide a final concise answer with citations.

Sources:
{context}

Question: {query}

Let me analyze the sources:
Step 1 - Relevant sources:"""
```

### Pattern 3: Structured Output (JSON)

```python
JSON_PROMPT = """Based on the context, answer in the following JSON format:
{{
  "answer": "concise answer here",
  "confidence": "high/medium/low",
  "sources_used": [1, 3],
  "key_facts": ["fact 1", "fact 2"],
  "limitations": "what info is missing, if any"
}}

Context:
{context}

Question: {query}

JSON Response:"""
```

### Pattern 4: Multi-Query Synthesis

```python
SYNTHESIS_PROMPT = """You have information from multiple sources about: {query}

Some sources may contain different or complementary information.
Your job is to:
1. Synthesize a complete answer from all sources
2. Note any contradictions between sources
3. Indicate confidence level based on source agreement

Sources:
{context}

Synthesized Answer:"""
```

### Pattern 5: Guard Against Hallucination

```python
ANTI_HALLUCINATION_PROMPT = """CRITICAL RULES:
1. You can ONLY use information explicitly stated in the sources below.
2. Do NOT infer, assume, or add information from your training data.
3. If a question asks about something not in the sources, say so clearly.
4. Prefixing with "Based on the provided context..." helps stay grounded.

Test: If I ask "What is the capital of France?" and sources only discuss 
AWS EKS, the correct answer is "The provided sources do not contain 
information about this topic."

Sources:
{context}

Question: {query}

Based on the provided context:"""
```

---

## Layer 6: Context Assembly Pipeline (Production Code)

```python
from dataclasses import dataclass
from typing import Optional
import tiktoken


@dataclass
class PromptConfig:
    model: str = "gpt-4o"
    max_context_tokens: int = 6000
    max_output_tokens: int = 1024
    chunk_ordering: str = "relevance"  # relevance, lost_in_middle, chronological
    include_metadata: bool = True
    citation_style: str = "numbered"   # numbered, inline, none
    prompt_style: str = "standard"     # standard, cot, json, synthesis


class ContextAssembler:
    """Production context assembly pipeline"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.encoder = tiktoken.encoding_for_model(config.model)

    def assemble(self, query: str, chunks: list, 
                 conversation_history: list = None) -> dict:
        """
        Full assembly pipeline:
        1. Filter low-quality chunks
        2. Deduplicate
        3. Budget allocation
        4. Ordering
        5. Format
        6. Build final prompt
        """
        # Step 1: Filter (remove low relevance)
        chunks = self._filter_low_quality(chunks, min_score=0.3)

        # Step 2: Deduplicate (near-identical chunks)
        chunks = self._deduplicate(chunks, threshold=0.9)

        # Step 3: Token budget — select chunks that fit
        chunks = self._apply_token_budget(chunks)

        # Step 4: Order chunks
        chunks = self._order_chunks(chunks)

        # Step 5: Build prompt
        prompt = self._build_prompt(query, chunks, conversation_history)

        return {
            "prompt": prompt,
            "chunks_used": len(chunks),
            "total_tokens": self._count_tokens(prompt),
            "sources": [c["metadata"].get("source") for c in chunks]
        }

    def _filter_low_quality(self, chunks, min_score=0.3):
        return [c for c in chunks if c.get("score", 1.0) >= min_score]

    def _deduplicate(self, chunks, threshold=0.9):
        """Remove near-duplicate chunks (same text slightly different)"""
        seen_texts = []
        unique = []
        for chunk in chunks:
            text = chunk["text"].strip().lower()
            is_dup = False
            for seen in seen_texts:
                # Simple overlap check
                overlap = len(set(text.split()) & set(seen.split())) / max(len(text.split()), 1)
                if overlap > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(chunk)
                seen_texts.append(text)
        return unique

    def _apply_token_budget(self, chunks):
        budget = self.config.max_context_tokens
        selected = []
        used = 0
        for chunk in chunks:
            tokens = self._count_tokens(chunk["text"])
            if used + tokens > budget:
                break
            selected.append(chunk)
            used += tokens
        return selected

    def _order_chunks(self, chunks):
        if self.config.chunk_ordering == "relevance":
            return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        elif self.config.chunk_ordering == "lost_in_middle":
            sorted_c = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
            result, end = [], []
            for i, c in enumerate(sorted_c):
                (result if i % 2 == 0 else end).append(c)
            return result + list(reversed(end))
        elif self.config.chunk_ordering == "chronological":
            return sorted(chunks, key=lambda x: x.get("metadata", {}).get("timestamp", ""), reverse=True)
        return chunks

    def _build_prompt(self, query, chunks, history=None):
        system = self._get_system_prompt()
        context = self._format_context(chunks)
        
        # Include conversation history if multi-turn
        history_str = ""
        if history:
            history_str = "\n## Previous Conversation:\n"
            for turn in history[-3:]:  # Last 3 turns only
                history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt = f"""{system}
{history_str}
## Context:
{context}

## Question:
{query}

## Answer:"""
        return prompt

    def _get_system_prompt(self):
        prompts = {
            "standard": "You are a helpful technical assistant. Answer based ONLY on the provided context. Cite sources as [Source N]. If unsure, say so.",
            "cot": "You are an expert analyst. Think step by step. First identify relevant sources, then reason through the answer, then provide a final concise response with citations.",
            "json": 'Respond in JSON format: {"answer": "...", "confidence": "high/medium/low", "sources": [1,2]}',
            "synthesis": "Synthesize information from multiple sources into a coherent answer. Note any contradictions."
        }
        return prompts.get(self.config.prompt_style, prompts["standard"])

    def _format_context(self, chunks):
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}]"
            if self.config.include_metadata:
                meta = chunk.get("metadata", {})
                source = meta.get("source", "")
                section = meta.get("section", "")
                if source:
                    header += f" (from: {source}"
                    if section:
                        header += f" > {section}"
                    header += ")"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _count_tokens(self, text):
        return len(self.encoder.encode(text))
```

---

## Layer 7: Multi-Turn Conversation Context

RAG sirf single question nahi hota — users follow-up karte hain:

```
User: "What's EKS pricing?"
Assistant: "EKS control plane costs $0.10/hour..."
User: "And what about worker nodes?"  ← "what about" = refers to EKS
User: "Is it cheaper than GKE?"       ← "it" = EKS
```

### Conversation-Aware Retrieval:

```python
class ConversationalRAG:
    """Handle multi-turn with context carryover"""
    
    def __init__(self, retriever, assembler, llm):
        self.retriever = retriever
        self.assembler = assembler
        self.llm = llm
        self.history = []
    
    def query(self, user_message: str) -> str:
        # Step 1: Resolve references using history
        resolved_query = self._resolve_references(user_message)
        
        # Step 2: Retrieve with resolved query
        chunks = self.retriever.search(resolved_query, top_k=5)
        
        # Step 3: Assemble with history
        result = self.assembler.assemble(
            query=user_message,  # Original for display
            chunks=chunks,
            conversation_history=self.history[-3:]
        )
        
        # Step 4: Generate answer
        answer = self.llm.generate(result["prompt"])
        
        # Step 5: Store in history
        self.history.append({"user": user_message, "assistant": answer})
        
        return answer
    
    def _resolve_references(self, query: str) -> str:
        """Use LLM to resolve pronouns/references"""
        if not self.history:
            return query
        
        recent = self.history[-2:]
        history_str = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent])
        
        resolution_prompt = f"""Given this conversation:
{history_str}

Rewrite this follow-up question as a standalone question:
"{query}"

Standalone question:"""
        
        resolved = self.llm.generate(resolution_prompt, max_tokens=100)
        return resolved.strip()
```

---

## Layer 8: Guardrails & Safety in Prompts

### Preventing Hallucination:

```python
GROUNDING_RULES = """
GROUNDING RULES (MANDATORY):
1. ONLY use facts explicitly stated in the Context section.
2. If context doesn't contain the answer → respond "Information not available in the provided sources."
3. NEVER use your training knowledge to supplement the context.
4. If you're uncertain → say "Based on limited information in the sources..."
5. Quote exact text from sources when possible.
"""
```

### Preventing Prompt Injection:

```python
def sanitize_user_input(query: str) -> str:
    """Basic prompt injection prevention"""
    # Remove potential injection patterns
    dangerous_patterns = [
        "ignore previous instructions",
        "ignore above instructions", 
        "disregard all prior",
        "you are now",
        "new instructions:",
        "system prompt:",
    ]
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            return "Invalid query detected."
    
    # Limit length
    if len(query) > 2000:
        query = query[:2000]
    
    return query
```

### Output Validation:

```python
def validate_response(response: str, chunks: list) -> dict:
    """Check if response is grounded in provided chunks"""
    # Check for citation presence
    has_citations = "[Source" in response
    
    # Check for hallucination signals
    hallucination_signals = [
        "as an AI",
        "I don't have access to",
        "in my training data",
        "as of my knowledge cutoff",
    ]
    may_be_hallucinating = any(s in response.lower() for s in hallucination_signals)
    
    return {
        "has_citations": has_citations,
        "possible_hallucination": may_be_hallucinating,
        "response_length": len(response),
        "grounded": has_citations and not may_be_hallucinating
    }
```

---

## Layer 9: Production Pitfalls

### Pitfall 1: Context Too Long = Worse Answers

Tumne 20 chunks daal diye (15K tokens). LLM ne middle ke 10 chunks ignore kar diye. Answer incomplete.

**Fix:** Maximum 5-8 chunks. Quality > Quantity. Better retrieval + reranking > more context.

### Pitfall 2: No System Prompt = Hallucination

Bina system instruction ke LLM apni training knowledge use karega, context ignore karega.

**Fix:** Always include explicit grounding instruction: "ONLY use the provided context."

### Pitfall 3: Query Not in Prompt

Retrieved chunks daal diye but original query include karna bhool gaye. LLM ko pata nahi kya answer karna hai.

**Fix:** Always include the exact user query clearly marked in the prompt.

### Pitfall 4: No Citation Instruction = Unverifiable Answers

LLM answer deta hai but source cite nahi karta. User trust nahi kar sakta.

**Fix:** Explicitly instruct: "Cite every claim using [Source N] format."

### Pitfall 5: Conversation History Bloat

10 turns ka history prompt mein daal diya = token budget exhaust, context ke liye jagah nahi bachi.

**Fix:** Only last 2-3 turns. Resolve references with LLM, then discard old turns.

### Pitfall 6: Chunk Metadata Not Included

Chunks daal diye but source information nahi. LLM ko pata nahi kaunsa chunk kahaan se aaya.

**Fix:** Always include source metadata with each chunk: filename, section, page number.

---

## Layer 10: Trade-offs & Decisions

### Context Length vs Quality:

| Chunks in Context | Answer Quality | Latency | Cost |
|-------------------|---------------|---------|------|
| 1-2 chunks | May miss info | Low | Low |
| 3-5 chunks | **Sweet spot** | Medium | Medium |
| 5-8 chunks | Good coverage | Higher | Higher |
| 10+ chunks | Diminishing returns + "Lost in Middle" | High | High |

### Prompt Style Decision:

| Use Case | Prompt Style | Why |
|----------|-------------|-----|
| Simple factual Q&A | Standard | Fast, clear |
| Complex analysis | Chain-of-Thought | Step-by-step reasoning |
| API/programmatic use | JSON output | Parseable |
| Research/comparison | Synthesis | Combines multiple views |
| High-stakes (medical/legal) | Citation-enforced + CoT | Verifiable + reasoned |

### Model Selection for RAG:

| Model | Context Window | Best For | Cost |
|-------|---------------|----------|------|
| GPT-4o-mini | 128K | Cost-effective RAG | $0.15/1M input |
| GPT-4o | 128K | Complex reasoning | $2.50/1M input |
| Claude 3.5 Sonnet | 200K | Long context, coding | $3.00/1M input |
| AWS Nova Pro | 300K | AWS-native, long docs | ~$0.80/1M input |
| Llama 3.1 70B | 128K | Self-hosted, privacy | GPU cost only |

---

## Layer 11: Interview Ready

### 2-Line Answer (Screening):

> "Context Assembly is the process of arranging retrieved chunks into an optimal prompt for the LLM — including system instructions for grounding, ordered context with metadata, the user query, and output format. The key challenge is balancing context coverage against the 'Lost in the Middle' effect where LLMs ignore information in the center of long prompts."

### 5-Min Answer (Technical Round):

> Above + 4-part prompt structure, token budget management (60-80% for context), chunk ordering strategies (relevance-first vs lost-in-middle fix), citation enforcement, hallucination prevention, multi-turn conversation handling with reference resolution.

### 10-Min Deep Dive (System Design):

> Above + production code (ContextAssembler class), guardrails (prompt injection prevention, output validation), prompt patterns (CoT, JSON, synthesis), model-specific optimizations, evaluation (groundedness metrics, citation accuracy), A/B testing prompt variants, streaming considerations.

### Expected Follow-up Questions:

**Q: "LLM context window 128K hai — saara content daal do, kya problem hai?"**
A: "Lost in the Middle" — LLM beginning aur end pe attend karta hai, middle ignore karta hai. Plus: more tokens = more cost, more latency. Sweet spot 3000-5000 tokens context. Quality retrieval > quantity.

**Q: "Hallucination kaise detect karoge production mein?"**
A: 3 approaches: (1) Citation check — har claim ke saath [Source N] hona chahiye, (2) Entailment model — check if answer is logically supported by context, (3) LLM-as-judge — second LLM verify kare ki answer context se consistent hai.

**Q: "Multi-turn mein context kaise manage karoge?"**
A: Reference resolution — LLM se follow-up query ko standalone query mein convert karao. Only last 2-3 turns keep karo. Token budget mein history ka fixed allocation (500-1000 tokens max).

**Q: "Different prompt templates A/B test kaise karoge?"**
A: Same queries, different prompts, measure: (1) Groundedness score (answer supported by context?), (2) Citation accuracy, (3) User satisfaction, (4) Latency/cost. Run for 1000+ queries, statistical significance check.

**Q: "Streaming response mein context assembly kaise alag hogi?"**
A: Prompt same hota hai. Difference: output incrementally aata hai. Citation validation post-hoc karna padta hai (stream complete hone ke baad). Guardrails real-time mein check karo (toxic content filter on partial output).

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| What context assembly is & why it matters | ✅ |
| 4-part prompt structure | ✅ |
| Token budget management (with code) | ✅ |
| "Lost in the Middle" problem + fix | ✅ |
| Chunk ordering strategies (4 approaches) | ✅ |
| Advanced prompt patterns (5 patterns) | ✅ |
| Production ContextAssembler class | ✅ |
| Multi-turn conversation handling | ✅ |
| Guardrails (hallucination, injection, validation) | ✅ |
| Production pitfalls (6 issues) | ✅ |
| Trade-offs & decision matrices | ✅ |
| Interview answers (all levels) | ✅ |
| Follow-up Q&A | ✅ |

**Topic 5: Prompt Engineering & Context Assembly — DONE.**

---
## Layer 12: GitLab CI/CD + ArgoCD — Production Context Assembly Service Deployment

Context Assembly service — jo retrieved chunks ko optimal prompt mein assemble karta hai, LLM call karta hai, aur grounded answer return karta hai — isko production mein deploy karna with full pipeline + external access.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
rag-context-service/
├── src/
│   ├── app.py                  # FastAPI — RAG answer generation API
│   ├── assembler.py            # ContextAssembler (prompt building)
│   ├── guardrails.py           # Input sanitization + output validation
│   ├── config.py
│   ├── requirements.txt
│   └── tests/
│       ├── test_assembler.py
│       ├── test_guardrails.py
│       └── test_app.py
├── eval/
│   ├── test_set.json           # Groundedness + citation test queries
│   ├── eval_quality.py         # Groundedness, citation accuracy, hallucination check
│   └── baseline_metrics.json
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml            # External access
│   ├── configmap.yaml
│   ├── hpa.yaml
│   └── pdb.yaml
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: RAG Context Service (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from assembler import ContextAssembler, PromptConfig
from guardrails import sanitize_input, validate_response
from config import settings
import requests
import time
import os

app = FastAPI(title="RAG Context Assembly Service", version="1.0.0")

# Metrics
RAG_REQUESTS = Counter("rag_requests_total", "Total RAG requests", ["status"])
RAG_LATENCY = Histogram("rag_latency_seconds", "End-to-end RAG latency",
                        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
HALLUCINATION_DETECTED = Counter("rag_hallucination_detected_total", "Hallucination signals")
TOKENS_USED = Counter("rag_tokens_used_total", "Total tokens consumed", ["type"])

assembler = ContextAssembler(PromptConfig(
    model=settings.LLM_MODEL,
    max_context_tokens=int(settings.MAX_CONTEXT_TOKENS),
    chunk_ordering=settings.CHUNK_ORDERING,
    prompt_style=settings.PROMPT_STYLE
))


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True
    conversation_history: list[dict] | None = None
    output_format: str = "standard"  # standard, json, cot


class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]
    metadata: dict


@app.get("/health")
def health():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    try:
        # Check retrieval service
        r = requests.get(f"{settings.RETRIEVAL_SERVICE_URL}/health", timeout=5)
        retrieval_ok = r.status_code == 200
        # Check LLM service
        r2 = requests.get(f"{settings.LLM_SERVICE_URL}/health", timeout=5)
        llm_ok = r2.status_code == 200
        if not (retrieval_ok and llm_ok):
            raise HTTPException(503, f"Dependencies: retrieval={retrieval_ok}, llm={llm_ok}")
        return {"status": "ready", "retrieval": retrieval_ok, "llm": llm_ok}
    except Exception as e:
        raise HTTPException(503, str(e))


@app.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    """Main RAG endpoint: query → retrieve → assemble → LLM → answer"""
    start = time.time()
    try:
        # Step 1: Sanitize input
        clean_query = sanitize_input(request.query)
        if not clean_query:
            raise HTTPException(400, "Invalid query")

        # Step 2: Retrieve relevant chunks
        retrieval_resp = requests.post(f"{settings.RETRIEVAL_SERVICE_URL}/retrieve", json={
            "query": clean_query,
            "top_k": request.top_k,
            "use_reranking": request.use_reranking
        }, timeout=30)
        chunks = retrieval_resp.json()["results"]

        # Step 3: Assemble context
        assembled = assembler.assemble(
            query=clean_query,
            chunks=chunks,
            conversation_history=request.conversation_history
        )

        # Step 4: Call LLM
        llm_resp = requests.post(f"{settings.LLM_SERVICE_URL}/generate", json={
            "prompt": assembled["prompt"],
            "max_tokens": 1024,
            "temperature": 0.3
        }, timeout=60)
        answer = llm_resp.json()["text"]

        # Step 5: Validate response
        validation = validate_response(answer, chunks)
        if validation["possible_hallucination"]:
            HALLUCINATION_DETECTED.inc()

        # Metrics
        RAG_REQUESTS.labels(status="success").inc()
        RAG_LATENCY.observe(time.time() - start)
        TOKENS_USED.labels(type="input").inc(assembled["total_tokens"])

        return RAGResponse(
            answer=answer,
            sources=[{"text": c["text"][:200], "source": c.get("metadata", {}).get("source", "")}
                     for c in chunks[:request.top_k]],
            metadata={
                "latency_ms": round((time.time() - start) * 1000, 2),
                "chunks_used": assembled["chunks_used"],
                "tokens_used": assembled["total_tokens"],
                "grounded": validation["grounded"],
                "has_citations": validation["has_citations"]
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        RAG_REQUESTS.labels(status="error").inc()
        raise HTTPException(500, str(e))


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
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
EXPOSE 8000
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "--access-logfile", "-"]
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
  - quality_gate
  - approval_gate
  - prod_deploy_canary
  - monitor
  - prod_deploy_full

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/rag-context-service
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://rag-context-staging.yourdomain.com"
  PROD_URL: "https://rag-context.yourdomain.com"

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
    - pip install ruff mypy
  script:
    - ruff check src/ --output-format=gitlab
    - ruff format src/ --check

lint_dockerfile:
  stage: lint
  image: hadolint/hadolint:latest-debian
  script:
    - hadolint Dockerfile --failure-threshold warning

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
    - gitleaks detect --source . --report-format json --report-path gitleaks-report.json

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
    - kubectl set image deployment/rag-context-service
        rag-context-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/rag-context-service -n rag-staging --timeout=300s
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

      # Health
      r = requests.get(f'{URL}/health')
      assert r.status_code == 200, f'Health failed: {r.status_code}'
      print('✓ Health OK')

      # Ready
      r = requests.get(f'{URL}/ready')
      assert r.status_code == 200, f'Ready failed: {r.status_code}'
      print('✓ Ready OK')

      # Ask endpoint
      r = requests.post(f'{URL}/ask', json={
          'query': 'What is EKS?',
          'top_k': 3
      })
      assert r.status_code == 200, f'Ask failed: {r.status_code}'
      data = r.json()
      assert 'answer' in data
      assert 'sources' in data
      assert data['metadata']['latency_ms'] < 10000
      print(f'✓ Ask OK: {len(data[\"answer\"])} chars, {data[\"metadata\"][\"latency_ms\"]}ms')

      # Metrics
      r = requests.get(f'{URL}/metrics')
      assert r.status_code == 200
      assert 'rag_requests_total' in r.text
      print('✓ Metrics OK')

      print('=== ALL SMOKE TESTS PASSED ===')
      "
  only: [main]

# ─────────────── QUALITY GATE (Groundedness) ───────────────
quality_gate:
  stage: quality_gate
  image: python:3.11-slim
  before_script:
    - pip install requests numpy
  script:
    - python3 eval/eval_quality.py --api-url ${STAGING_URL} --test-set eval/test_set.json --min-groundedness 0.80 --min-citation-accuracy 0.70
  artifacts:
    paths: [eval/results.json]
    when: always
  only: [main]
  allow_failure: false

# ─────────────── APPROVAL GATE ───────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging + Quality gate passed. Awaiting approval."
    - echo "Image → ${DOCKER_IMAGE}:${DOCKER_TAG}"
  when: manual
  allow_failure: false
  only: [main]

# ─────────────── CANARY 10% ───────────────
deploy_prod_canary:
  stage: prod_deploy_canary
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: ${PROD_URL}
  script:
    - |
      cat <<EOF | kubectl apply -f -
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: rag-context-service-canary
        namespace: rag-production
        labels:
          app: rag-context-service
          track: canary
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: rag-context-service
            track: canary
        template:
          metadata:
            labels:
              app: rag-context-service
              track: canary
          spec:
            containers:
            - name: rag-context-service
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports:
              - containerPort: 8000
              envFrom:
              - configMapRef:
                  name: rag-context-config
              resources:
                requests: { memory: "1Gi", cpu: "500m" }
                limits: { memory: "2Gi", cpu: "1" }
              livenessProbe:
                httpGet: { path: /health, port: 8000 }
                initialDelaySeconds: 10
              readinessProbe:
                httpGet: { path: /ready, port: 8000 }
                initialDelaySeconds: 5
      EOF
    - kubectl rollout status deployment/rag-context-service-canary -n rag-production --timeout=300s
    - echo "Canary deployed — 10% traffic"
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
          print(f'--- Check {i}/10 ---')
          try:
              r = requests.get('${PROD_URL}/health', timeout=10)
              if r.status_code != 200:
                  FAILURES += 1
                  print(f'WARN: Health failed')
              else:
                  print('OK')

              # Functional check
              r2 = requests.post('${PROD_URL}/ask', json={'query': 'test', 'top_k': 2}, timeout=30)
              if r2.status_code != 200:
                  FAILURES += 1
                  print(f'WARN: Ask failed')
          except Exception as e:
              FAILURES += 1
              print(f'WARN: {e}')

          if FAILURES >= 3:
              print('=== CANARY FAILED ===')
              sys.exit(1)
          time.sleep(30)
      print('=== CANARY HEALTHY ===')
      "
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        kubectl delete deployment rag-context-service-canary -n rag-production --ignore-not-found
      fi
  only: [main]

# ─────────────── FULL 100% ───────────────
deploy_prod_full:
  stage: prod_deploy_full
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: ${PROD_URL}
  script:
    - kubectl set image deployment/rag-context-service
        rag-context-service=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/rag-context-service -n rag-production --timeout=600s
    - kubectl delete deployment rag-context-service-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100% DEPLOYED ==="
    - echo "URL: ${PROD_URL}"
  only: [main]
```

### Step 4: Kubernetes — Deployment + External Access

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-context-service
  namespace: rag-production
  labels:
    app: rag-context-service
    track: stable
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: rag-context-service
      track: stable
  template:
    metadata:
      labels:
        app: rag-context-service
        track: stable
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: rag-context-service
        image: registry.gitlab.com/yourgroup/rag-context-service:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: rag-context-config
        resources:
          requests: { memory: "1Gi", cpu: "500m" }
          limits: { memory: "2Gi", cpu: "1" }
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet: { path: /ready, port: 8000 }
          initialDelaySeconds: 5
          periodSeconds: 5
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-context-service
  namespace: rag-production
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: rag-context-service
```

```yaml
# k8s/ingress.yaml — EXTERNAL ACCESS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-context-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "Strict-Transport-Security: max-age=31536000";
spec:
  tls:
  - hosts:
    - rag-context.yourdomain.com
    secretName: rag-context-tls
  rules:
  - host: rag-context.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-context-service
            port:
              number: 80
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-context-config
  namespace: rag-production
data:
  RETRIEVAL_SERVICE_URL: "http://rag-retrieval-service.rag-production.svc.cluster.local"
  LLM_SERVICE_URL: "http://llm-gateway.rag-production.svc.cluster.local:8080"
  LLM_MODEL: "gpt-4o-mini"
  MAX_CONTEXT_TOKENS: "6000"
  CHUNK_ORDERING: "relevance"
  PROMPT_STYLE: "standard"
  APP_VERSION: "latest"
```

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-context-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-context-service
  minReplicas: 3
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
```

### Step 5: ArgoCD

```yaml
# argocd/production-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rag-context-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://gitlab.com/yourgroup/rag-context-service.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: rag-production
  syncPolicy:
    syncOptions: [CreateNamespace=true]
```

### Step 6: Operations

```bash
# External access (from ANYWHERE)
curl https://rag-context.yourdomain.com/health
curl https://rag-context.yourdomain.com/ready

# Ask a question (full RAG pipeline)
curl -X POST https://rag-context.yourdomain.com/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does EKS networking work?", "top_k": 5}'

# Response:
# {
#   "answer": "EKS networking uses VPC-CNI plugin... [Source 1]",
#   "sources": [...],
#   "metadata": {"latency_ms": 1500, "chunks_used": 5, "grounded": true}
# }

# Metrics
curl https://rag-context.yourdomain.com/metrics

# Rollback
kubectl rollout undo deployment/rag-context-service -n rag-production
```

### Architecture (Complete RAG Pipeline):

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FULL RAG ARCHITECTURE                             │
│                                                                      │
│  External: https://rag-context.yourdomain.com                        │
│       │                                                              │
│       ▼                                                              │
│  Ingress (TLS + rate limit)                                          │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────┐                                            │
│  │ RAG Context Service  │ (this service)                             │
│  │ - Assemble prompt    │                                            │
│  │ - Call LLM           │                                            │
│  │ - Validate output    │                                            │
│  └──────┬───────┬───────┘                                            │
│         │       │                                                    │
│         ▼       ▼                                                    │
│  ┌────────────┐  ┌─────────────┐                                    │
│  │ Retrieval  │  │ LLM Gateway │                                    │
│  │ Service    │  │ (GPT/Claude/│                                    │
│  │ (Topic 4)  │  │  Nova)      │                                    │
│  └──────┬─────┘  └─────────────┘                                    │
│         │                                                            │
│    ┌────┴────┐                                                       │
│    ▼         ▼                                                       │
│ Qdrant   Elasticsearch                                               │
│ (Dense)  (Sparse/BM25)                                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Additional Sources:** Stanford "Lost in the Middle" paper (2024), Anthropic prompt engineering guide, LangChain documentation
- **Extra added:** ContextAssembler production class, token budget management, chunk ordering strategies, guardrails, multi-turn handling, quality gate CI stage, production deployment pipeline, external access — not in original curriculum
