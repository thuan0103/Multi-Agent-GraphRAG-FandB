"""
B1.1 — Graph Database Schema
Folder: services/ingestion/schema.py
Node types: MenuItem, Chunk, Entity, Category
Relationships: BELONGS_TO, NEXT, MENTIONS
"""
from neo4j import AsyncGraphDatabase
import logging

logger = logging.getLogger(__name__)

SCHEMA_QUERIES = [
    # ── Constraints ────────────────────────────────────────────
    "CREATE CONSTRAINT menu_item_id IF NOT EXISTS FOR (n:MenuItem) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE",

    # ── Vector Indexes (Neo4j 5.x) ─────────────────────────────
    """
    CREATE VECTOR INDEX menu_embedding IF NOT EXISTS
    FOR (n:MenuItem) ON (n.embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
      }
    }
    """,
    """
    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
    FOR (n:Chunk) ON (n.embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
      }
    }
    """,

    # ── Full-text indexes (hybrid search) ──────────────────────
    "CREATE FULLTEXT INDEX menu_fulltext IF NOT EXISTS FOR (n:MenuItem) ON EACH [n.name, n.description, n.ingredients]",
    "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
]


async def init_schema(uri: str, user: str, password: str):
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    async with driver.session() as session:
        for q in SCHEMA_QUERIES:
            try:
                await session.run(q.strip())
                logger.info(f"Schema OK: {q.strip()[:60]}...")
            except Exception as e:
                logger.warning(f"Schema skip (may exist): {e}")
    await driver.close()
    logger.info("Graph schema initialized.")
