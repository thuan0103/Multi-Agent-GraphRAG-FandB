"""
B1.2 — Reranker Service (FastAPI)
Folder: services/reranker/main.py
"""
import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="Reranker Service", version="1.0")

model: CrossEncoder = None


@app.on_event("startup")
async def load_model():
    global model
    logger.info(f"Loading reranker: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, device=DEVICE, max_length=512)
    logger.info("Reranker loaded.")


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    threshold: float = 0.7


class ScoredDoc(BaseModel):
    index: int
    text: str
    score: float


class RerankResponse(BaseModel):
    results: List[ScoredDoc]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not req.documents:
        return RerankResponse(results=[])

    pairs = [[req.query, doc] for doc in req.documents]
    with torch.no_grad():
        scores = model.predict(pairs, show_progress_bar=False)

    scored = [
        ScoredDoc(index=i, text=doc, score=float(scores[i]))
        for i, doc in enumerate(req.documents)
        if float(scores[i]) >= req.threshold
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    return RerankResponse(results=scored)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": model is not None}
