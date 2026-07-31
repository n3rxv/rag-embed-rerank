from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
import os

app = FastAPI()

# Use ONNX backend to keep memory under 512MB
embed_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    backend='onnx',
)
rerank_model = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    backend='onnx',
)

@app.on_event("startup")
async def warmup():
    embed_model.encode(["warmup"], normalize_embeddings=True)
    rerank_model.predict([["warmup query", "warmup document"]])
    print("Models warmed up!")

class EmbedRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 6

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/warmup")
def warmup_get():
    embed_model.encode(["warmup query"], normalize_embeddings=True)
    rerank_model.predict([["warmup query", "warmup document"]])
    return {"status": "warmed up"}

@app.post("/embed")
def embed(req: EmbedRequest):
    embeddings = embed_model.encode(req.texts, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}

@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, doc] for doc in req.documents]
    scores = rerank_model.predict(pairs)
    results = sorted(
        [{"index": i, "score": float(scores[i])} for i in range(len(scores))],
        key=lambda x: x["score"],
        reverse=True,
    )[:req.top_n]
    return {"results": results}
