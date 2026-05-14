"""
Folder: services/ingestion/embedder.py
HTTP client gọi Embedding Service
"""
import os
import httpx
from typing import List

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8001")


async def embed_texts(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{EMBEDDING_URL}/embed", json={"texts": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


async def embed_one(text: str) -> List[float]:
    vecs = await embed_texts([text])
    return vecs[0]
