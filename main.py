from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Energy Intelligence API")

# Temporary CORS configuration for testing.
# We will restrict this to the Vercel domain later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Energy Intelligence API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/ask")
def ask(data: dict):
    question = data.get("question", "")

    return {
        "status": "success",
        "answer": f"Backend received: {question}"
    }
