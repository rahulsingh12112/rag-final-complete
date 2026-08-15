# Topic 6: LLM Integration & Orchestration — Complete Deep Dive

> **Target Role:** AI Infrastructure Architect / Senior ML Platform Engineer
> **Prerequisites:** Embedding (1), Chunking (2), Vector DB (3), Retrieval (4), Prompt Engineering (5)
> **Source:** Engineer Repo → mod-110-llm-infrastructure/03-rag-systems.md + 2026 LLM Orchestration Best Practices

---

## 🎯 One-Liner (Interview):

> "LLM Integration wo layer hai jahan tum model ko call karte ho — streaming, retries, fallback chains, token management, cost control, aur multi-model routing sab yahan handle hota hai. Orchestration frameworks (LangChain/LlamaIndex) isko abstract karte hain but production mein tumhe internals samajhne padenge."

---

## Layer 1: Kya Hai Aur Kyun Complex Hai?

Topic 5 mein tumne prompt banaya. Ab wo prompt LLM ko dena hai. Simple lagta hai — ek API call. But production mein:

1. **Which model?** GPT-4o, Claude 3.5, Nova Pro, Llama 3.1 — kaunsa, kab?
2. **Streaming?** User ko 5 seconds wait karana ya token-by-token dikhana?
3. **Failures?** API down, rate limited, timeout — kya karna?
4. **Cost?** GPT-4o = $2.50/1M tokens. 10K queries/day = $750/month sirf LLM cost.
5. **Latency?** Cold start, network hop, token generation speed.
6. **Orchestration?** Simple call ya multi-step chain (retrieve → reason → answer)?

**Key insight:** LLM call ek network I/O operation hai — unreliable, expensive, slow. Production mein isko treat karo jaise external API dependency treat karte ho: retries, timeouts, circuit breakers, fallbacks, caching.

---

## Layer 2: Direct LLM Integration (No Framework)

### AWS Bedrock (Recommended for AWS shops):

```python
import boto3
import json
from typing import Generator

class BedrockLLM:
    """Production Bedrock LLM client with streaming"""

    def __init__(self, region="us-east-1", model_id="amazon.nova-pro-v1:0"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.model_configs = {
            "amazon.nova-pro-v1:0": {"max_tokens": 5000, "cost_per_1k_input": 0.0008, "cost_per_1k_output": 0.0032},
            "amazon.nova-micro-v1:0": {"max_tokens": 5000, "cost_per_1k_input": 0.000035, "cost_per_1k_output": 0.00014},
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {"max_tokens": 8192, "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
            "meta.llama3-1-70b-instruct-v1:0": {"max_tokens": 2048, "cost_per_1k_input": 0.00099, "cost_per_1k_output": 0.00099},
        }

    def generate(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.3, system: str = None) -> str:
        """Synchronous generation"""
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        body = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }
        if system:
            body["system"] = [{"text": system}]

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        return result["output"]["message"]["content"][0]["text"]

    def stream(self, prompt: str, max_tokens: int = 1024,
               temperature: float = 0.3) -> Generator[str, None, None]:
        """Streaming generation — token by token"""
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps({
                "messages": messages,
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}
            })
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk["delta"].get("text", "")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request"""
        config = self.model_configs.get(self.model_id, {})
        input_cost = (input_tokens / 1000) * config.get("cost_per_1k_input", 0)
        output_cost = (output_tokens / 1000) * config.get("cost_per_1k_output", 0)
        return round(input_cost + output_cost, 6)
```

### OpenAI (Most Common):

```python
from openai import OpenAI
from typing import Generator

class OpenAILLM:
    """Production OpenAI client"""

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model

    def generate(self, prompt: str, system: str = None,
                 max_tokens: int = 1024, temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, system: str = None,
               max_tokens: int = 1024) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

---

## Layer 3: Resilience Patterns (Production Must-Have)

### Pattern 1: Retry with Exponential Backoff

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0):
    """Decorator: retry on transient failures"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, Exception) as e:
                    if attempt == max_retries:
                        raise
                    # Rate limit (429) → longer wait
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        delay = min(base_delay * (2 ** attempt) * 2, max_delay)
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    # Jitter to avoid thundering herd
                    delay += random.uniform(0, delay * 0.1)
                    print(f"Retry {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator
```

### Pattern 2: Fallback Chain (Multi-Model)

```python
class LLMFallbackChain:
    """Try primary model → fallback to secondary → tertiary"""

    def __init__(self):
        self.models = [
            {"name": "gpt-4o-mini", "client": OpenAILLM("gpt-4o-mini"), "priority": 1},
            {"name": "nova-pro", "client": BedrockLLM(model_id="amazon.nova-pro-v1:0"), "priority": 2},
            {"name": "nova-micro", "client": BedrockLLM(model_id="amazon.nova-micro-v1:0"), "priority": 3},
        ]

    def generate(self, prompt: str, **kwargs) -> dict:
        """Try each model in order until one succeeds"""
        errors = []
        for model in self.models:
            try:
                result = model["client"].generate(prompt, **kwargs)
                return {
                    "text": result,
                    "model_used": model["name"],
                    "fallback": model["priority"] > 1
                }
            except Exception as e:
                errors.append(f"{model['name']}: {str(e)}")
                continue

        raise Exception(f"All models failed: {errors}")
```

### Pattern 3: Circuit Breaker

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal — requests go through
    OPEN = "open"          # Broken — fail fast, don't call
    HALF_OPEN = "half_open"  # Testing — allow one request

class CircuitBreaker:
    """Prevent cascading failures when LLM API is down"""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker OPEN — failing fast")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### Pattern 4: Response Caching

```python
import hashlib
import json
import redis

class LLMCache:
    """Cache LLM responses — same prompt = same answer (deterministic)"""

    def __init__(self, redis_host="redis", ttl=3600):
        self.cache = redis.Redis(host=redis_host, decode_responses=True)
        self.ttl = ttl

    def get_or_generate(self, llm, prompt: str, **kwargs) -> dict:
        # Only cache if temperature = 0 (deterministic)
        if kwargs.get("temperature", 0.3) > 0:
            return {"text": llm.generate(prompt, **kwargs), "cached": False}

        cache_key = self._key(prompt, kwargs)
        cached = self.cache.get(cache_key)
        if cached:
            return {"text": cached, "cached": True}

        result = llm.generate(prompt, **kwargs)
        self.cache.setex(cache_key, self.ttl, result)
        return {"text": result, "cached": False}

    def _key(self, prompt, kwargs):
        raw = f"{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return f"llm:cache:{hashlib.md5(raw.encode()).hexdigest()}"
```

---

## Layer 4: Streaming (User Experience Critical)

### Why Streaming Matters:

```
Without streaming:
User asks → waits 3-5 seconds (blank screen) → full answer appears

With streaming:
User asks → first token in 200ms → tokens flow in real-time → feels instant
```

**Time-to-First-Token (TTFT)** is the key metric. Users perceive <500ms TTFT as "instant".

### FastAPI Streaming Endpoint:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/ask/stream")
async def ask_stream(request: RAGRequest):
    """Streaming RAG response — token by token"""

    async def generate_stream():
        # Step 1: Retrieve (non-streaming, fast)
        chunks = retriever.search(request.query, top_k=5)
        
        # Step 2: Assemble prompt
        assembled = assembler.assemble(query=request.query, chunks=chunks)

        # Step 3: Stream LLM response
        # First, send sources metadata
        yield json.dumps({"type": "sources", "data": [c["metadata"]["source"] for c in chunks]}) + "\n"

        # Then stream answer tokens
        for token in llm.stream(assembled["prompt"]):
            yield json.dumps({"type": "token", "data": token}) + "\n"

        # Finally, send completion signal
        yield json.dumps({"type": "done", "metadata": {"chunks_used": len(chunks)}}) + "\n"

    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )
```

### Client-Side Streaming (JavaScript):

```javascript
async function streamRAG(query) {
    const response = await fetch('/ask/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query, top_k: 5})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\n');
        for (const line of lines) {
            if (!line) continue;
            const event = JSON.parse(line);
            
            if (event.type === 'sources') {
                displaySources(event.data);
            } else if (event.type === 'token') {
                appendToken(event.data);  // Real-time display
            } else if (event.type === 'done') {
                onComplete(event.metadata);
            }
        }
    }
}
```

---

## Layer 5: LLM Router (Cost + Quality Optimization)

### Smart Model Routing:

```python
class LLMRouter:
    """Route queries to appropriate model based on complexity/cost"""

    def __init__(self):
        self.models = {
            "simple": BedrockLLM(model_id="amazon.nova-micro-v1:0"),    # $0.035/1M — fast, cheap
            "standard": OpenAILLM("gpt-4o-mini"),                       # $0.15/1M — balanced
            "complex": OpenAILLM("gpt-4o"),                             # $2.50/1M — best quality
        }

    def route(self, query: str, context_tokens: int) -> str:
        """Decide which model to use"""
        complexity = self._assess_complexity(query, context_tokens)
        return complexity

    def _assess_complexity(self, query: str, context_tokens: int) -> str:
        """
        Simple heuristics for routing:
        - Short factual questions → simple model
        - Standard Q&A → standard model
        - Multi-step reasoning, comparison, analysis → complex model
        """
        query_lower = query.lower()

        # Complex indicators
        complex_signals = ["compare", "analyze", "why", "explain the difference",
                          "step by step", "pros and cons", "trade-off", "design"]
        if any(signal in query_lower for signal in complex_signals):
            return "complex"

        # Simple indicators
        if len(query.split()) < 10 and context_tokens < 2000:
            return "simple"

        return "standard"

    def generate(self, query: str, prompt: str, context_tokens: int) -> dict:
        tier = self.route(query, context_tokens)
        model = self.models[tier]
        result = model.generate(prompt)
        return {"text": result, "model_tier": tier, "model": model.model_id if hasattr(model, 'model_id') else model.model}
```

### Cost Impact:

| Routing Strategy | Monthly Cost (10K queries/day) | Quality |
|-----------------|-------------------------------|---------|
| Always GPT-4o | ~$2,250 | Best |
| Always GPT-4o-mini | ~$135 | Good |
| Smart routing (70% simple, 25% standard, 5% complex) | ~$180 | Good+ (complex queries get best model) |

---

## Layer 6: Orchestration Frameworks

### LangChain (Most Popular):

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA

# Simple RAG chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on context. Cite sources."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# Usage
answer = chain.invoke("How does EKS networking work?")
```

### LlamaIndex (RAG-Focused):

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
Settings.embed_model = OpenAIEmbedding()

# Build index and query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("How does EKS networking work?")
```

### When to Use What:

| Factor | LangChain | LlamaIndex | Custom (No Framework) |
|--------|-----------|------------|----------------------|
| Best for | Complex chains, agents | RAG-specific workflows | Full control, performance |
| Learning curve | Medium | Low | High |
| Flexibility | Very high | Medium | Maximum |
| Debugging | Hard (abstractions) | Medium | Easy |
| Production readiness | Good (with care) | Good | You decide |
| Overhead | Some | Minimal | None |

**Production recommendation:** Start with custom (Topic 1-5 code). Use LangChain/LlamaIndex only when you need their specific features (agents, complex chains, tool-calling). Don't add framework overhead for simple RAG.

---

## Layer 7: Function Calling / Tool Use

### LLM as Decision Maker:

```python
import json

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search internal documentation for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "enum": ["networking", "security", "pricing", "general"]}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_pricing",
            "description": "Get real-time AWS pricing for a service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "region": {"type": "string", "default": "us-east-1"}
                },
                "required": ["service"]
            }
        }
    }
]

class ToolCallingRAG:
    """LLM decides which tools to call"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.tool_handlers = {
            "search_documents": self._search_docs,
            "get_current_pricing": self._get_pricing,
        }

    def query(self, user_message: str) -> str:
        # Step 1: LLM decides if tools needed
        response = self.llm.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_message}],
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Step 2: If tool call requested, execute it
        if message.tool_calls:
            tool_results = []
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                result = self.tool_handlers[func_name](**func_args)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })

            # Step 3: LLM generates final answer with tool results
            messages = [
                {"role": "user", "content": user_message},
                message,
                *tool_results
            ]
            final = self.llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return final.choices[0].message.content

        # No tools needed — direct answer
        return message.content

    def _search_docs(self, query, category=None):
        # Call retrieval service
        return {"results": ["doc1", "doc2"]}

    def _get_pricing(self, service, region="us-east-1"):
        # Call pricing API
        return {"service": service, "price": "$0.10/hour"}
```

---

## Layer 8: Cost Management

### Token Tracking:

```python
import tiktoken
from prometheus_client import Counter, Histogram

TOKENS_INPUT = Counter("llm_tokens_input_total", "Input tokens", ["model"])
TOKENS_OUTPUT = Counter("llm_tokens_output_total", "Output tokens", ["model"])
COST_TOTAL = Counter("llm_cost_dollars_total", "LLM cost in dollars", ["model"])

class CostTracker:
    """Track and alert on LLM costs"""

    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},       # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "nova-pro": {"input": 0.80, "output": 3.20},
        "nova-micro": {"input": 0.035, "output": 0.14},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    }

    def track(self, model: str, input_tokens: int, output_tokens: int):
        TOKENS_INPUT.labels(model=model).inc(input_tokens)
        TOKENS_OUTPUT.labels(model=model).inc(output_tokens)

        pricing = self.PRICING.get(model, {"input": 1.0, "output": 1.0})
        cost = (input_tokens / 1_000_000) * pricing["input"] + \
               (output_tokens / 1_000_000) * pricing["output"]
        COST_TOTAL.labels(model=model).inc(cost)
        return cost

    def estimate_monthly(self, queries_per_day: int, avg_input: int = 3000,
                         avg_output: int = 500, model: str = "gpt-4o-mini") -> float:
        daily_cost = self.track(model, avg_input * queries_per_day,
                               avg_output * queries_per_day)
        return daily_cost * 30
```

### Cost Optimization Strategies:

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Use gpt-4o-mini instead of gpt-4o | 90%+ | Slightly lower quality |
| Response caching (Redis) | 30-60% | Stale answers possible |
| Smart routing (simple→cheap model) | 50-70% | Needs routing logic |
| Shorter prompts (concise context) | 20-40% | Less context |
| Prompt caching (Anthropic/OpenAI) | 50% on repeated prefixes | Only for same system prompt |

---

## Layer 9: Production Pitfalls

### Pitfall 1: No Timeout on LLM Calls

LLM API hung → your service hangs → cascading failure.

**Fix:** Always set timeout (30-60s). Use async with timeout wrapper.

### Pitfall 2: No Retry on Rate Limits (429)

Rate limited → single failure → user gets error.

**Fix:** Exponential backoff with jitter. Max 3 retries. Alert if persistent.

### Pitfall 3: Not Tracking Costs

Month-end bill $5000 surprise.

**Fix:** Track tokens per request. Set daily budget alerts. Use Prometheus + Grafana dashboard.

### Pitfall 4: Synchronous Calls for Streaming Use Case

User waiting 5s for full response when streaming gives 200ms TTFT.

**Fix:** Always implement streaming for user-facing endpoints. Sync only for batch/internal.

### Pitfall 5: Single Model Dependency

OpenAI API down → your entire RAG system down.

**Fix:** Fallback chain. Primary = OpenAI, Secondary = Bedrock, Tertiary = self-hosted Llama.

### Pitfall 6: Not Handling Partial Streaming Failures

Stream starts, 50% response generated, then connection drops. User sees incomplete answer.

**Fix:** Buffer partial response. On disconnect, retry with same prompt or show "Response interrupted. Retry?"

---

## Layer 10: Trade-offs & Decisions

### Model Selection:

| Need | Model | Why |
|------|-------|-----|
| Cheapest production RAG | Nova Micro / GPT-4o-mini | Lowest cost per token |
| Best quality | GPT-4o / Claude 3.5 Sonnet | Highest reasoning |
| Longest context | Claude 3.5 (200K) / Nova Pro (300K) | Large document QA |
| Self-hosted (privacy) | Llama 3.1 70B | Data stays on-prem |
| Fastest TTFT | GPT-4o-mini / Nova Micro | Streaming UX |

### Framework vs Custom:

| Scenario | Use |
|----------|-----|
| Simple RAG (retrieve → prompt → answer) | Custom code (Topics 1-5) |
| Multi-step agents with tool calling | LangChain |
| Complex document processing pipeline | LlamaIndex |
| Need maximum performance/control | Custom code |
| Prototyping quickly | LangChain/LlamaIndex |

---

## Layer 11: Interview Ready

### 2-Line Answer (Screening):

> "LLM Integration in production requires resilience patterns — retry with backoff, fallback chains across providers, circuit breakers, streaming for UX, response caching for cost, and smart routing to balance quality vs cost. It's not just an API call — it's treating the LLM as an unreliable external dependency."

### 5-Min Answer (Technical Round):

> Above + direct provider integration (Bedrock/OpenAI), streaming architecture (SSE/NDJSON), token budget tracking, cost optimization (routing 70% to cheap models), function calling for agentic RAG, LangChain vs LlamaIndex vs custom trade-offs.

### 10-Min Deep Dive (System Design):

> Above + complete LLM gateway architecture (routing, caching, fallback, metrics), circuit breaker implementation, cost tracking with Prometheus, model selection matrix, streaming client implementation, multi-model A/B testing, latency optimization (TTFT, prompt caching), capacity planning (tokens/second limits per provider).

### Expected Follow-up Questions:

**Q: "LLM API rate limited ho gaya production mein — kya karoge?"**
A: Immediate: fallback to secondary model (Bedrock if OpenAI limited). Short-term: implement token bucket rate limiter on our side. Long-term: request higher rate limits, add response caching, smart routing to distribute load.

**Q: "Cost $3000/month hai LLM ka — optimize kaise karoge?"**
A: (1) Audit: which queries consume most tokens? (2) Route simple queries to nova-micro ($0.035/1M). (3) Cache frequent queries in Redis. (4) Reduce context size (better retrieval = fewer chunks needed). (5) Prompt compression. Expected saving: 60-80%.

**Q: "Self-hosted Llama vs API-based — kab kya?"**
A: Self-hosted when: data privacy mandatory, high volume (>50K queries/day makes GPU cost < API cost), need fine-tuning, low latency required. API when: variable load, no GPU expertise, need latest models, budget predictability.

**Q: "Streaming implement karna hai — architecture kya hogi?"**
A: FastAPI with StreamingResponse (NDJSON). Backend: LLM provider streaming API. Frontend: fetch + ReadableStream. Protocol: Server-Sent Events (SSE) or NDJSON over HTTP. Buffer handling: collect tokens, flush every 50ms for smooth UX.

---

## Completeness Check:

| Topic | Covered? |
|-------|----------|
| Direct LLM integration (Bedrock, OpenAI) | ✅ |
| Streaming (server + client) | ✅ |
| Resilience (retry, fallback, circuit breaker) | ✅ |
| Response caching | ✅ |
| Smart model routing | ✅ |
| Cost tracking & optimization | ✅ |
| Orchestration frameworks (LangChain, LlamaIndex) | ✅ |
| Function calling / Tool use | ✅ |
| Production pitfalls (6) | ✅ |
| Trade-offs & decisions | ✅ |
| Interview answers | ✅ |

**Topic 6: LLM Integration & Orchestration — DONE.**

---
## Layer 12: GitLab CI/CD + ArgoCD — Production LLM Gateway Service Deployment

LLM Gateway — jo model routing, streaming, caching, fallbacks, cost tracking handle karta hai — isko production mein deploy karna with full pipeline + external access.

**Production Pipeline Flow:**
```
test → lint → security scan → build → push → staging deploy → smoke test → approval gate → prod deploy (canary 10%) → monitor → prod deploy (100%)
```

### Project Structure:

```
llm-gateway-service/
├── src/
│   ├── app.py                  # FastAPI — LLM Gateway API
│   ├── router.py               # Smart model routing
│   ├── providers/
│   │   ├── bedrock.py          # AWS Bedrock client
│   │   ├── openai_client.py    # OpenAI client
│   │   └── base.py             # Base LLM interface
│   ├── resilience.py           # Retry, circuit breaker, fallback
│   ├── cache.py                # Response caching (Redis)
│   ├── cost_tracker.py         # Token & cost tracking
│   ├── config.py
│   ├── requirements.txt
│   └── tests/
│       ├── test_router.py
│       ├── test_resilience.py
│       └── test_app.py
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml            # External access
│   ├── configmap.yaml
│   ├── hpa.yaml
│   └── redis.yaml              # Cache layer
├── argocd/
│   ├── staging-app.yaml
│   └── production-app.yaml
├── .gitlab-ci.yml
└── README.md
```

### Step 1: LLM Gateway API (`src/app.py`)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from router import LLMRouter
from cache import LLMCache
from cost_tracker import CostTracker
from config import settings
import json
import time
import os

app = FastAPI(title="LLM Gateway Service", version="1.0.0")

# Metrics
LLM_REQUESTS = Counter("llm_requests_total", "Total LLM requests", ["model", "status"])
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM call latency", ["model"])
LLM_TOKENS = Counter("llm_tokens_total", "Tokens used", ["model", "type"])

router = LLMRouter()
cache = LLMCache(redis_host=settings.REDIS_HOST)
cost_tracker = CostTracker()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.3
    model: str | None = None       # Override routing
    system: str | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    model_used: str
    cached: bool
    latency_ms: float
    tokens: dict
    cost_usd: float


@app.get("/health")
def health():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "unknown")}


@app.get("/ready")
def ready():
    checks = {"redis": cache.check_health(), "providers": router.check_providers()}
    if not all(checks.values()):
        raise HTTPException(503, f"Not ready: {checks}")
    return {"status": "ready", **checks}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Synchronous LLM generation with routing + caching + cost tracking"""
    start = time.time()

    # Check cache first
    if request.temperature == 0:
        cached_result = cache.get(request.prompt, request.model)
        if cached_result:
            LLM_REQUESTS.labels(model="cache", status="hit").inc()
            return GenerateResponse(
                text=cached_result, model_used="cache", cached=True,
                latency_ms=round((time.time() - start) * 1000, 2),
                tokens={"input": 0, "output": 0}, cost_usd=0.0
            )

    # Route to appropriate model
    model_name = request.model or router.route(request.prompt, len(request.prompt.split()))
    
    try:
        result = router.generate(
            model_name=model_name,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system
        )

        latency = (time.time() - start) * 1000
        
        # Track metrics
        LLM_REQUESTS.labels(model=model_name, status="success").inc()
        LLM_LATENCY.labels(model=model_name).observe(time.time() - start)
        
        input_tokens = len(request.prompt.split()) * 1.3  # Rough estimate
        output_tokens = len(result.split()) * 1.3
        cost = cost_tracker.track(model_name, int(input_tokens), int(output_tokens))

        # Cache result
        if request.temperature == 0:
            cache.set(request.prompt, request.model, result)

        return GenerateResponse(
            text=result, model_used=model_name, cached=False,
            latency_ms=round(latency, 2),
            tokens={"input": int(input_tokens), "output": int(output_tokens)},
            cost_usd=cost
        )
    except Exception as e:
        LLM_REQUESTS.labels(model=model_name, status="error").inc()
        raise HTTPException(500, str(e))


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest):
    """Streaming LLM generation"""
    model_name = request.model or router.route(request.prompt, len(request.prompt.split()))

    async def stream_tokens():
        try:
            for token in router.stream(model_name, request.prompt,
                                       request.max_tokens, request.temperature):
                yield json.dumps({"type": "token", "data": token}) + "\n"
            yield json.dumps({"type": "done", "model": model_name}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_tokens(), media_type="application/x-ndjson")


@app.get("/models")
def list_models():
    """Available models and their status"""
    return router.list_models()


@app.get("/cost/summary")
def cost_summary():
    """Cost summary for monitoring"""
    return cost_tracker.get_summary()


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
  - approval_gate
  - prod_deploy_canary
  - monitor
  - prod_deploy_full

variables:
  DOCKER_IMAGE: ${CI_REGISTRY_IMAGE}/llm-gateway
  DOCKER_TAG: ${CI_COMMIT_SHORT_SHA}
  STAGING_URL: "https://llm-staging.yourdomain.com"
  PROD_URL: "https://llm.yourdomain.com"

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
    - ruff check src/
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
    - gitleaks detect --source .

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
    - kubectl set image deployment/llm-gateway
        llm-gateway=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-staging
    - kubectl rollout status deployment/llm-gateway -n rag-staging --timeout=300s
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
      import requests, json

      URL = '${STAGING_URL}'

      # Health + Ready
      assert requests.get(f'{URL}/health').status_code == 200
      print('✓ Health OK')
      assert requests.get(f'{URL}/ready').status_code == 200
      print('✓ Ready OK')

      # Generate
      r = requests.post(f'{URL}/generate', json={
          'prompt': 'Say hello in one word.',
          'max_tokens': 10,
          'temperature': 0.0
      })
      assert r.status_code == 200
      data = r.json()
      assert len(data['text']) > 0
      print(f'✓ Generate OK: model={data[\"model_used\"]}, {data[\"latency_ms\"]}ms')

      # Models list
      r = requests.get(f'{URL}/models')
      assert r.status_code == 200
      print(f'✓ Models: {len(r.json())} available')

      # Metrics
      assert requests.get(f'{URL}/metrics').status_code == 200
      print('✓ Metrics OK')

      print('=== ALL SMOKE TESTS PASSED ===')
      "
  only: [main]

# ─────────────── APPROVAL GATE ───────────────
approval_for_production:
  stage: approval_gate
  script:
    - echo "Staging passed. Awaiting approval."
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
        name: llm-gateway-canary
        namespace: rag-production
        labels:
          app: llm-gateway
          track: canary
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: llm-gateway
            track: canary
        template:
          metadata:
            labels:
              app: llm-gateway
              track: canary
          spec:
            containers:
            - name: llm-gateway
              image: ${DOCKER_IMAGE}:${DOCKER_TAG}
              ports:
              - containerPort: 8000
              envFrom:
              - configMapRef:
                  name: llm-gateway-config
              - secretRef:
                  name: llm-gateway-secrets
              resources:
                requests: { memory: "512Mi", cpu: "250m" }
                limits: { memory: "1Gi", cpu: "500m" }
              livenessProbe:
                httpGet: { path: /health, port: 8000 }
                initialDelaySeconds: 10
              readinessProbe:
                httpGet: { path: /ready, port: 8000 }
                initialDelaySeconds: 5
      EOF
    - kubectl rollout status deployment/llm-gateway-canary -n rag-production --timeout=300s
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
              if r.status_code != 200: FAILURES += 1; print('WARN: Health')
              else: print('OK')
              # Functional
              r2 = requests.post('${PROD_URL}/generate', json={
                  'prompt': 'test', 'max_tokens': 5, 'temperature': 0
              }, timeout=30)
              if r2.status_code != 200: FAILURES += 1
          except Exception as e:
              FAILURES += 1; print(f'WARN: {e}')
          if FAILURES >= 3: print('CANARY FAILED'); sys.exit(1)
          time.sleep(30)
      print('=== CANARY HEALTHY ===')
      "
  after_script:
    - |
      if [ "$CI_JOB_STATUS" = "failed" ]; then
        kubectl delete deployment llm-gateway-canary -n rag-production --ignore-not-found
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
    - kubectl set image deployment/llm-gateway
        llm-gateway=${DOCKER_IMAGE}:${DOCKER_TAG} -n rag-production
    - kubectl rollout status deployment/llm-gateway -n rag-production --timeout=600s
    - kubectl delete deployment llm-gateway-canary -n rag-production --ignore-not-found
    - echo "=== PRODUCTION 100% DEPLOYED: ${PROD_URL} ==="
  only: [main]
```

### Step 4: Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-gateway
  namespace: rag-production
  labels:
    app: llm-gateway
    track: stable
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
  selector:
    matchLabels:
      app: llm-gateway
      track: stable
  template:
    metadata:
      labels:
        app: llm-gateway
        track: stable
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: llm-gateway
        image: registry.gitlab.com/yourgroup/llm-gateway:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: llm-gateway-config
        - secretRef:
            name: llm-gateway-secrets
        resources:
          requests: { memory: "512Mi", cpu: "250m" }
          limits: { memory: "1Gi", cpu: "500m" }
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
  name: llm-gateway
  namespace: rag-production
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: llm-gateway
```

```yaml
# k8s/ingress.yaml — EXTERNAL ACCESS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-gateway-ingress
  namespace: rag-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "200"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"   # Required for streaming
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
spec:
  tls:
  - hosts:
    - llm.yourdomain.com
    secretName: llm-gateway-tls
  rules:
  - host: llm.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-gateway
            port:
              number: 80
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-gateway-config
  namespace: rag-production
data:
  REDIS_HOST: "redis.rag-production.svc.cluster.local"
  DEFAULT_MODEL: "gpt-4o-mini"
  FALLBACK_MODEL: "amazon.nova-pro-v1:0"
  MAX_RETRIES: "3"
  CACHE_TTL: "3600"
  BEDROCK_REGION: "us-east-1"
```

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-gateway-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

### Step 5: Operations

```bash
# External access
curl https://llm.yourdomain.com/health
curl https://llm.yourdomain.com/models
curl https://llm.yourdomain.com/cost/summary

# Generate (sync)
curl -X POST https://llm.yourdomain.com/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain EKS in 2 sentences", "max_tokens": 100}'

# Stream
curl -X POST https://llm.yourdomain.com/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain EKS networking", "stream": true}'

# Rollback
kubectl rollout undo deployment/llm-gateway -n rag-production
```

### Architecture:

```
┌────────────────────────────────────────────────────────────────┐
│                    LLM GATEWAY ARCHITECTURE                      │
│                                                                │
│  External: https://llm.yourdomain.com                          │
│       │                                                        │
│       ▼                                                        │
│  Ingress (TLS + rate limit + proxy-buffering=off for stream)   │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────┐                          │
│  │ LLM Gateway Service (3-20 pods)  │                          │
│  │ - Smart routing                  │                          │
│  │ - Response caching (Redis)       │                          │
│  │ - Fallback chain                 │                          │
│  │ - Circuit breaker                │                          │
│  │ - Cost tracking                  │                          │
│  │ - Streaming support              │                          │
│  └────────┬──────────┬─────────────┘                          │
│           │          │                                         │
│     ┌─────┴───┐ ┌───┴─────┐                                   │
│     │  Redis  │ │ Secrets  │ (API keys)                        │
│     │ (cache) │ │ (K8s)   │                                    │
│     └─────────┘ └─────────┘                                    │
│           │                                                    │
│     ┌─────┴──────────────────────────────┐                     │
│     │         External LLM Providers      │                    │
│     │                                     │                    │
│     │  ┌─────────┐ ┌────────┐ ┌───────┐  │                    │
│     │  │ OpenAI  │ │Bedrock │ │ Self- │  │                    │
│     │  │ GPT-4o  │ │ Nova   │ │hosted │  │                    │
│     │  │ (primary)│ │(fallback)│ │(Llama)│ │                    │
│     │  └─────────┘ └────────┘ └───────┘  │                    │
│     └─────────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Source & Attribution

- **Primary Source:** [ai-infra-engineer-learning/mod-110-llm-infrastructure/03-rag-systems.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/tree/main/lessons/mod-110-llm-infrastructure)
- **Additional Sources:** OpenAI API docs, AWS Bedrock docs, LangChain documentation, Circuit breaker pattern (Martin Fowler)
- **Extra added:** LLM Gateway architecture, smart routing, cost tracking, streaming implementation, fallback chains, circuit breaker, response caching, production pipeline — not in original curriculum
