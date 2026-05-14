"""
B2.3 — Semantic Cache (Layer 4 / bonus)
Folder: services/graph_rag/semantic_cache.py

- Lưu embedding + response vào Redis
- Tìm kiếm cosine similarity ≥ 0.95 → trả cache ngay
"""
import json
import logging
import os
import math
from typing import Optional, List
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
CACHE_TTL = 3600  # 1 hour
CACHE_PREFIX = "sem_cache:"
MAX_CACHE_ENTRIES = 1000


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x ** 2 for x in a))
    nb = math.sqrt(sum(x ** 2 for x in b))
    return dot / (na * nb + 1e-9)


class SemanticCache:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get(self, query_embedding: List[float]) -> Optional[dict]:
        """Return cached result if similarity ≥ threshold."""
        keys = await self.redis.keys(f"{CACHE_PREFIX}*")
        for key in keys[:MAX_CACHE_ENTRIES]:
            try:
                raw = await self.redis.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                sim = cosine_sim(query_embedding, entry["embedding"])
                if sim >= THRESHOLD:
                    logger.info(f"Semantic cache HIT (sim={sim:.3f})")
                    return entry["response"]
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None

    async def set(self, query_embedding: List[float], response: dict, query: str):
        import hashlib
        key = CACHE_PREFIX + hashlib.md5(query.encode()).hexdigest()
        entry = {"embedding": query_embedding, "response": response, "query": query}
        await self.redis.setex(key, CACHE_TTL, json.dumps(entry))
        logger.info(f"Semantic cache SET for: {query[:60]}")

    async def flush(self):
        keys = await self.redis.keys(f"{CACHE_PREFIX}*")
        if keys:
            await self.redis.delete(*keys)
