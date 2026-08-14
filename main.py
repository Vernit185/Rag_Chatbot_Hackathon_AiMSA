from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Energy Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


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
def ask(request: AskRequest):

    return {
        "status": "success",
        "answer": f"Backend received: {request.question}"
    }
