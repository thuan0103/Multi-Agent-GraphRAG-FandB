"""
B1.3 — Ingest Menu CSV
Folder: services/ingestion/ingest_menu.py

Expected CSV columns (tối thiểu):
  id, name, price, size, category, ingredients, description
"""
import csv
import uuid
import logging
from neo4j import AsyncGraphDatabase
from embedder import embed_texts

logger = logging.getLogger(__name__)

UPSERT_MENU = """
MERGE (cat:Category {name: $category})
MERGE (m:MenuItem {id: $id})
SET m.name        = $name,
    m.price       = $price,
    m.size        = $size,
    m.category    = $category,
    m.ingredients = $ingredients,
    m.description = $description,
    m.embedding   = $embedding
MERGE (m)-[:BELONGS_TO]->(cat)
"""


async def ingest_menu_csv(path: str, driver):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    texts = [
        f"{r.get('name','')} {r.get('description','')} {r.get('ingredients','')}"
        for r in rows
    ]
    embeddings = await embed_texts(texts)

    async with driver.session() as session:
        for row, emb in zip(rows, embeddings):
            await session.run(
                UPSERT_MENU,
                id=row.get("id") or str(uuid.uuid4()),
                name=row.get("name", ""),
                price=float(row.get("price", 0)),
                size=row.get("size", ""),
                category=row.get("category", "other"),
                ingredients=row.get("ingredients", ""),
                description=row.get("description", ""),
                embedding=emb,
            )
    logger.info(f"Ingested {len(rows)} MenuItems from {path}")
    return len(rows)
