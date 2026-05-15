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


DELETE_MENU = "MATCH (m:MenuItem) DETACH DELETE m"
DELETE_CATEGORY = "MATCH (c:Category) WHERE NOT (c)<-[:BELONGS_TO]-() DELETE c"


async def ingest_menu_csv(path: str, driver):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)         
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _fix_row(row: dict) -> dict:
        """Fix CSV parsing artifact: ingredients like '["a","b"]' get split on inner comma."""
        ingr = row.get("ingredients", "")
        desc = row.get("description", "")
        if ingr.startswith('"') and not ingr.endswith('"'):
            row["ingredients"] = (ingr[1:] + ", " + desc.rstrip('"')).strip(", ")
            row["description"] = ""
        else:
            row["ingredients"] = ingr.strip('"').strip()
            row["description"] = desc.strip('"').strip()
        return row

    rows = [_fix_row(r) for r in rows]

    texts = [
        f"{r.get('name','')} {r.get('description','')} {r.get('ingredients','')}".strip()
        for r in rows
    ]
    embeddings = await embed_texts(texts)

    async with driver.session() as session:
        await session.run(DELETE_MENU)
        await session.run(DELETE_CATEGORY)
        logger.info("Cleared existing MenuItems and orphaned Categories")

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
