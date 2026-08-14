import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Energy Intelligence API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporary. Restrict to Vercel later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# GLOBAL MODELS / DATA
# ============================================================

embedding_model = None
groq_client = None

chunks = []
chunk_embeddings = None


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document():

    with open(
        "documents/energy.txt",
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    return text


# ============================================================
# SIMPLE CHUNKING
# ============================================================

def create_chunks(text, chunk_size=120, overlap=30):

    words = text.split()

    result = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        result.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return result


# ============================================================
# BUILD RETRIEVAL INDEX
# ============================================================

def build_index():

    global embedding_model
    global chunks
    global chunk_embeddings

    print("Loading document...")

    text = load_document()

    print("Creating chunks...")

    chunks = create_chunks(text)

    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    chunk_embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    print("RAG index ready!")


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        chunk_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index],
            "score": float(scores[index])
        })

    return results


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        [
            f"[SOURCE {i+1}]\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    system_prompt = """
You are an Energy Information Assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not invent information.
2. If the context does not contain enough information, say so.
3. Give a concise and useful answer.
4. Mention the relevant source number when making claims.
"""

    user_prompt = f"""
CONTEXT:

{context}

USER QUESTION:

{question}

Answer using only the context above.
"""

    response = groq_client.chat.completions.create(

        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=500
    )

    return response.choices[0].message.content


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global groq_client

    print("Starting Energy Intelligence API...")

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY environment variable is missing."
        )

    groq_client = Groq(api_key=api_key)

    build_index()

    print("Startup complete!")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Energy Intelligence API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "rag_loaded": embedding_model is not None,
        "chunks": len(chunks)
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):

    retrieved = retrieve(
        request.question,
        top_k=3
    )

    answer = generate_answer(
        request.question,
        retrieved
    )

    return {

        "status": "success",

        "answer": answer,

        "sources": [
            {
                "text": item["text"],
                "score": item["score"]
            }
            for item in retrieved
        ]
    }
