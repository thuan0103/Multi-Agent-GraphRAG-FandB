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


class ExpandRequest(BaseModel):
    chunk_ids: List[str]


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


@app.post("/expand")
async def expand_chunks(req: ExpandRequest):
    """Return NEXT/PREV neighbors and MENTIONS entities for given Chunk IDs."""
    cypher = """
    MATCH (c:Chunk)
    WHERE c.id IN $ids
    OPTIONAL MATCH (c)-[:NEXT|PREV]-(n:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e)
    WITH
        collect(DISTINCT CASE WHEN n IS NOT NULL
            THEN {id: n.id, text: n.text,
                  question: coalesce(n.question, ''), doc_type: coalesce(n.doc_type, 'faq')}
            END) AS neighbors,
        collect(DISTINCT e.name) AS entity_names
    RETURN neighbors, entity_names
    """
    async with driver.session() as session:
        result = await session.run(cypher, ids=req.chunk_ids)
        record = await result.single()
    if not record:
        return {"neighbors": [], "entities": []}
    neighbors = [n for n in (record["neighbors"] or []) if n]
    entities = [e for e in (record["entity_names"] or []) if e]
    return {"neighbors": neighbors, "entities": entities}


@app.get("/menu/fulltext")
async def fulltext_menu_search(q: str, limit: int = 5):
    """Fulltext search on MenuItem by name/description/ingredients."""
    cypher = """
    CALL db.index.fulltext.queryNodes('menu_fulltext', $q)
    YIELD node, score
    RETURN node.id AS id, node.name AS name, node.price AS price,
           node.size AS size, node.category AS category,
           node.ingredients AS ingredients, node.description AS description,
           score
    ORDER BY score DESC LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(cypher, q=q, limit=limit)
        items = [dict(r) async for r in result]
    return {"items": items, "count": len(items)}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query_emb = await get_embedding(req.query)
    if req.use_cache:
        cached = await sem_cache.get(query_emb)
        if cached:
            return SearchResponse(query=req.query, results=cached, from_cache=True)

    results = await hybrid_search(req.query, driver)

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
