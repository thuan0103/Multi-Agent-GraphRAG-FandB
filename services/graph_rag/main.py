"""
B1.1 / B1.2 — Graph RAG API
Folder: services/graph_rag/main.py
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase

from search import hybrid_search, get_embedding
from semantic_cache import SemanticCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secret")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

driver = None
sem_cache: SemanticCache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global driver, sem_cache
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    sem_cache = SemanticCache(REDIS_URL)
    yield
    await driver.close()


app = FastAPI(title="Graph RAG API", version="1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_cache: bool = True


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    from_cache: bool = False


@app.get("/menu/all")
async def get_all_menu():
    """Return every MenuItem from Neo4j — used by OrderAgent for full menu context."""
    query = """
    MATCH (m:MenuItem)
    RETURN m.id AS id, m.name AS name, m.price AS price,
           m.size AS size, m.category AS category,
           m.ingredients AS ingredients, m.description AS description
    ORDER BY m.category, m.name
    """
    async with driver.session() as session:
        result = await session.run(query)
        items = [dict(r) async for r in result]
    return {"items": items, "count": len(items)}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    # ── Semantic cache check ──────────────────────────────────
    query_emb = await get_embedding(req.query)
    if req.use_cache:
        cached = await sem_cache.get(query_emb)
        if cached:
            return SearchResponse(query=req.query, results=cached, from_cache=True)

    # ── Full hybrid search pipeline ───────────────────────────
    results = await hybrid_search(req.query, driver)

    # Store in semantic cache
    if req.use_cache:
        await sem_cache.set(query_emb, results, req.query)

    return SearchResponse(query=req.query, results=results, from_cache=False)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.delete("/cache")
async def clear_cache():
    await sem_cache.flush()
    return {"message": "Semantic cache cleared"}
