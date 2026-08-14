import os
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from fastembed import TextEmbedding


app = FastAPI(
    title="Energy Embedding Service"
)


# Load the lightweight ONNX embedding model once
print("Loading embedding model...")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded!")


class EmbedRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Embedding service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/embed")
def embed(request: EmbedRequest):

    embeddings = list(
        embedding_model.embed(
            [request.text]
        )
    )

    vector = embeddings[0]

    return {
        "status": "success",
        "dimension": len(vector),
        "embedding_preview": vector[:10].tolist()
    }
