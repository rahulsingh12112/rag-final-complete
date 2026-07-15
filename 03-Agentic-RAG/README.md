# 🤖 Agentic RAG

## Overview
Standard RAG ek one-shot pipeline hai — ek baar retrieve, ek baar generate. Agentic RAG mein ek AI agent **decide** karta hai ki kab retrieve karna hai, kaise karna hai, aur kab stop karna hai.

```
User Query → Agent Plans → Retrieve → Grade → Sufficient? 
                                                    ↓ No
                                         Re-plan → Retrieve again
                                                    ↓ Yes
                                              Generate Answer → Verify → Return
```

## Topics

### 01 - Query Planning
Complex questions ko sub-questions mein todna.

**Example:**
- User: "Compare Tesla and BYD's revenue growth and market strategy in 2024"
- Agent plans:
  1. Tesla revenue 2024 search karo
  2. BYD revenue 2024 search karo
  3. Tesla market strategy search karo
  4. BYD market strategy search karo
  5. Compare and synthesize

**Tools:** LangGraph, LlamaIndex SubQuestionQueryEngine

### 02 - Multi-Step Retrieval
Ek baar mein sab nahi milta — multiple rounds mein information gather karna.

**Flow:**
1. First retrieval → partial information mili
2. Agent analyze karta hai → kya missing hai?
3. Second retrieval with refined query
4. Combine all information
5. Generate comprehensive answer

### 03 - Retrieval Grading
Retrieved content actually useful hai ya nahi — LLM se judge karwana.

**Process:**
1. Retrieve top-k chunks
2. LLM evaluates each chunk: "Is this relevant to the question?"
3. Irrelevant chunks discard karo
4. Agar sab irrelevant → different search strategy try karo
5. Sirf relevant chunks se answer generate karo

**Benefit:** Hallucination drastically reduce hota hai

### 04 - Self-Correction Loops
Generated answer ko verify karna aur galat ho toh fix karna.

**Loop:**
1. Generate answer from context
2. Check: "Is this answer grounded in the retrieved context?"
3. Check: "Does this actually answer the user's question?"
4. If No → identify what's wrong → retry with better approach
5. If Yes → return to user

**Max retries set karo** (2-3) taaki infinite loop na ho.

### 05 - Multiple Knowledge Sources
Different databases, APIs, aur documents se simultaneously data fetch karna.

**Example Architecture:**
```
Agent
├── Source 1: Company internal docs (Pinecone)
├── Source 2: Web search (Tavily/Serper API)
├── Source 3: SQL Database (structured data)
├── Source 4: API calls (real-time data)
└── Source 5: Knowledge Graph (Neo4j)
```

Agent decides: "Is question ke liye kaunsa source best hai?"

**Tools:** LangGraph, CrewAI, LlamaIndex Agents
