"""
B1.2 — Embedding Service (FastAPI)
Folder: services/embedding/main.py
"""
import os
import logging
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="Embedding Service", version="1.0")

model: SentenceTransformer = None


@app.on_event("startup")
async def load_model():
    global model
    logger.info(f"Loading embedding model: {MODEL_NAME} on {DEVICE}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info("Embedding model loaded.")


class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    texts = [req.texts] if isinstance(req.texts, str) else req.texts
    with torch.no_grad():
        vecs = model.encode(
            texts,
            normalize_embeddings=req.normalize,
            batch_size=32,
            show_progress_bar=False,
        )
    return EmbedResponse(
        embeddings=vecs.tolist(),
        model=MODEL_NAME,
        dim=vecs.shape[1],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": model is not None}
