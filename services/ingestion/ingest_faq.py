"""
B1.3 — Ingest FAQ CSV
Folder: services/ingestion/ingest_faq.py

Expected CSV columns: question, answer  (+ optional: id, category)
"""
import csv
import uuid
import logging
from neo4j import AsyncGraphDatabase
from embedder import embed_texts

logger = logging.getLogger(__name__)

UPSERT_CHUNK = """
MERGE (c:Chunk {id: $id})
SET c.text      = $text,
    c.source    = $source,
    c.doc_type  = 'faq',
    c.question  = $question,
    c.embedding = $embedding
"""

LINK_NEXT = """
MATCH (a:Chunk {id: $id_a}), (b:Chunk {id: $id_b})
MERGE (a)-[:NEXT]->(b)
MERGE (b)-[:PREV]->(a)
"""


async def ingest_faq_csv(path: str, driver):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    texts = [f"Q: {r.get('question','')} A: {r.get('answer','')}" for r in rows]
    embeddings = await embed_texts(texts)

    ids = []
    async with driver.session() as session:
        for i, (row, emb) in enumerate(zip(rows, embeddings)):
            cid = row.get("id") or str(uuid.uuid4())
            ids.append(cid)
            await session.run(
                UPSERT_CHUNK,
                id=cid,
                text=texts[i],
                question=row.get("question", ""),
                source="faq.csv",
                embedding=emb,
            )
        # Link consecutive chunks
        for i in range(len(ids) - 1):
            await session.run(LINK_NEXT, id_a=ids[i], id_b=ids[i + 1])

    logger.info(f"Ingested {len(rows)} FAQ chunks from {path}")
    return len(rows)
