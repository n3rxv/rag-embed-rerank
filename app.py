from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List

app = FastAPI()

# Loads once at startup, stays in memory
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class EmbedRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 6

@app.get("/")
def health():
    return {"status": "ok"}

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
