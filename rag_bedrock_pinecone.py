import boto3
import json
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG System: AWS Bedrock (Embedding + LLM) + Pinecone (Vector DB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGION = "us-east-1"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "amazon.nova-micro-v1:0"
INDEX_NAME = "rag-demo"
DIMENSION = 1024  # Titan Embed V2 output dimension

# ━━━ STEP A: AWS Bedrock Client Setup ━━━
print("Setting up AWS Bedrock client...")
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def get_embedding(text: str) -> list:
    """Text ko 1024-dimension vector mein convert karo using Titan Embed V2"""
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
    """LLM se answer generate karo using Nova Micro"""
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


# ━━━ STEP B: Pinecone Setup ━━━
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index create karo (agar nahi hai)
existing_indexes = [idx.name for idx in pc.indexes.list()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.indexes.create(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{INDEX_NAME}' created! Waiting for it to be ready...")
    import time
    time.sleep(10)  # Pinecone ko ready hone mein kuch seconds lagte hain
else:
    print(f"Index '{INDEX_NAME}' already exists!")

index = pc.index(INDEX_NAME)

# ━━━ STEP C: Documents Define + Embed + Store ━━━
documents = [
    "EKS uses a managed Kubernetes control plane with etcd for cluster state management. AWS manages the control plane nodes across multiple AZs.",
    "Kubernetes pods communicate via CNI plugin. On AWS EKS, VPC-CNI assigns real VPC IP addresses to pods enabling native VPC networking.",
    "Application Load Balancer operates at Layer 7 and routes HTTP/HTTPS traffic to target groups based on path, host, or header rules.",
    "Security groups act as virtual firewalls for EC2 instances, controlling inbound and outbound traffic at the instance level.",
    "Route 53 is AWS managed DNS service providing domain registration, DNS routing with health checks, failover, and latency-based routing.",
    "CloudWatch collects metrics, logs, and events from AWS resources. It supports custom metrics, alarms, dashboards, and log insights queries.",
    "S3 provides object storage with 99.999999999% (11 nines) durability. It supports versioning, lifecycle policies, and cross-region replication.",
    "AWS IAM provides identity and access management with users, roles, policies, and federation. It follows the principle of least privilege.",
    "VPC peering connects two VPCs privately using AWS backbone network. Traffic never traverses the public internet. No transitive peering.",
    "ECS Fargate runs containers without managing EC2 instances. You define task CPU/memory and AWS handles the underlying infrastructure."
]

print(f"\nEmbedding {len(documents)} documents using Titan Embed V2...")
vectors = []
for i, doc in enumerate(documents):
    embedding = get_embedding(doc)
    vectors.append((
        f"doc-{i}",
        embedding,
        {"text": doc, "index": i}
    ))
    print(f"  Embedded doc-{i}: {doc[:50]}...")

print("\nUpserting vectors to Pinecone...")
index.upsert(vectors=vectors)
print(f"Stored {len(vectors)} documents in Pinecone!")

# ━━━ STEP D: RAG Query Function ━━━
def rag_query(question: str, top_k: int = 3) -> dict:
    """
    Complete RAG pipeline:
    1. Question embed karo
    2. Pinecone se similar docs dhundho
    3. Docs + question ko LLM ko do
    4. Answer return karo
    """
    # Step 1: Query embed
    query_embedding = get_embedding(question)

    # Step 2: Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Step 3: Context banao retrieved docs se
    context_docs = []
    for match in results.matches:
        context_docs.append({
            "text": match.metadata["text"],
            "score": match.score
        })

    context = "\n\n".join([
        f"[Document {i+1} (relevance: {doc['score']:.3f})]\n{doc['text']}"
        for i, doc in enumerate(context_docs)
    ])

    # Step 4: LLM prompt banao
    prompt = f"""You are a helpful AWS infrastructure expert. Answer the question based ONLY on the provided context documents. If the context doesn't contain enough information, say so. Always cite which document number you're referencing.

Context:
{context}

Question: {question}

Answer:"""

    # Step 5: LLM se answer lo
    answer = ask_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": context_docs
    }


# ━━━ STEP E: Test Queries ━━━
print(f"\n{'='*70}")
print("RAG SYSTEM READY - Testing queries...")
print(f"{'='*70}")

test_questions = [
    "How does networking work in EKS pods?",
    "What is the durability of S3?",
    "How do security groups work?",
    "What is VPC peering and does it support transitive routing?",
]

for question in test_questions:
    print(f"\n{'─'*70}")
    print(f"Q: {question}")
    print(f"{'─'*70}")

    result = rag_query(question)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for i, src in enumerate(result["sources"]):
        print(f"   {i+1}. [Score: {src['score']:.4f}] {src['text'][:80]}...")

print(f"\n{'='*70}")
print("Done! RAG system working with Bedrock + Pinecone")
print(f"{'='*70}")


