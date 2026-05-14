"""
B1.2 — Hybrid Search Pipeline
Folder: services/graph_rag/search.py

Pipeline:
  1. Dual-Domain vector search (MenuItem + Chunk)
  2. Graph Expansion via NEXT/PREV + MENTIONS
  3. Late Reranking (threshold ≥ 0.7)
"""
import os
import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8001")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8002")
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.7"))
TOP_K = 5


async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{EMBEDDING_URL}/embed", json={"texts": text})
        r.raise_for_status()
        return r.json()["embeddings"][0]


async def rerank(query: str, docs: List[str], threshold: float) -> List[Dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "documents": docs, "threshold": threshold},
        )
        r.raise_for_status()
        return r.json()["results"]


async def hybrid_search(query: str, driver) -> List[Dict[str, Any]]:
    """Full hybrid search pipeline."""
    query_emb = await get_embedding(query)

    # ── Step 1: Dual-Domain vector search ──────────────────────
    menu_results = await _vector_search_menu(query_emb, driver, top_k=TOP_K)
    chunk_results = await _vector_search_chunks(query_emb, driver, top_k=TOP_K)

    # ── Step 2: Graph Expansion ─────────────────────────────────
    chunk_ids = [r["id"] for r in chunk_results]
    expanded = await _graph_expand(chunk_ids, driver)

    # Merge all results, deduplicate by id
    all_results: Dict[str, Dict] = {}
    for r in menu_results + chunk_results + expanded:
        rid = r.get("id", r.get("text", "")[:50])
        if rid not in all_results:
            all_results[rid] = r

    candidates = list(all_results.values())

    # ── Step 3: Late Reranking ──────────────────────────────────
    texts = [_result_to_text(r) for r in candidates]
    if not texts:
        return []

    ranked = await rerank(query, texts, RERANK_THRESHOLD)

    final = []
    for item in ranked:
        original = candidates[item["index"]]
        original["rerank_score"] = item["score"]
        final.append(original)

    return final[:TOP_K]


def _result_to_text(r: Dict) -> str:
    if r.get("type") == "menu":
        return f"{r.get('name','')} {r.get('description','')} {r.get('ingredients','')} price:{r.get('price','')}"
    return r.get("text", "")


async def _vector_search_menu(embedding: List[float], driver, top_k: int = 5) -> List[Dict]:
    query = """
    CALL db.index.vector.queryNodes('menu_embedding', $top_k, $embedding)
    YIELD node, score
    RETURN node.id AS id,
           node.name AS name,
           node.price AS price,
           node.size AS size,
           node.category AS category,
           node.ingredients AS ingredients,
           node.description AS description,
           score,
           'menu' AS type
    ORDER BY score DESC
    """
    async with driver.session() as session:
        result = await session.run(query, embedding=embedding, top_k=top_k)
        return [dict(r) async for r in result]


async def _vector_search_chunks(embedding: List[float], driver, top_k: int = 5) -> List[Dict]:
    query = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
    YIELD node, score
    RETURN node.id AS id,
           node.text AS text,
           node.source AS source,
           node.doc_type AS doc_type,
           score,
           'chunk' AS type
    ORDER BY score DESC
    """
    async with driver.session() as session:
        result = await session.run(query, embedding=embedding, top_k=top_k)
        return [dict(r) async for r in result]


async def _graph_expand(chunk_ids: List[str], driver) -> List[Dict]:
    """Traverse NEXT/PREV neighbors and MENTIONS entities."""
    if not chunk_ids:
        return []
    query = """
    UNWIND $ids AS cid
    MATCH (c:Chunk {id: cid})
    OPTIONAL MATCH (c)-[:NEXT|PREV]-(neighbor:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    WITH collect(DISTINCT neighbor) AS neighbors,
         collect(DISTINCT e.name) AS entities
    UNWIND neighbors AS n
    RETURN n.id AS id,
           n.text AS text,
           n.source AS source,
           n.doc_type AS doc_type,
           0.0 AS score,
           'chunk' AS type
    """
    async with driver.session() as session:
        result = await session.run(query, ids=chunk_ids)
        return [dict(r) async for r in result]
