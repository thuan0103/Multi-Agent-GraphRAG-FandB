"""
B1.3 — Ingestion Service Entrypoint
Folder: services/ingestion/main.py
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from neo4j import AsyncGraphDatabase

from schema import init_schema
from ingest_menu import ingest_menu_csv
from ingest_faq import ingest_faq_csv
from ingest_doc import ingest_document
from watcher import start_watcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secret")
GRAPH_RAG_URL = os.getenv("GRAPH_RAG_URL", "http://graph_rag:8004")
WATCH_DIR = os.getenv("WATCH_DIR", "/data/watch")
DATA_DIR = "/data"

driver = None


async def _clear_rag_cache():
    """Xóa semantic cache của graph_rag sau khi data thay đổi."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(f"{GRAPH_RAG_URL}/cache")
            logger.info(f"Cleared graph_rag cache: {resp.json()}")
    except Exception as e:
        logger.warning(f"Could not clear graph_rag cache: {e}")


async def auto_ingest(path: str):
    p = Path(path)
    ingested = False
    if p.name.lower().startswith("menu") and p.suffix == ".csv":
        await ingest_menu_csv(str(p), driver)
        ingested = True
    elif p.name.lower().startswith("faq") and p.suffix == ".csv":
        await ingest_faq_csv(str(p), driver)
        ingested = True
    elif p.suffix in (".pdf", ".docx", ".txt"):
        await ingest_document(str(p), driver)
        ingested = True
    else:
        logger.warning(f"Unknown file type: {path}")
    if ingested:
        await _clear_rag_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global driver
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    await init_schema(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Auto-ingest existing data files
    for fname in ["menu.csv", "faq.csv"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            logger.info(f"Auto-ingesting {fpath}")
            await auto_ingest(fpath)

    # Start watch mode
    os.makedirs(WATCH_DIR, exist_ok=True)
    asyncio.create_task(start_watcher(WATCH_DIR, auto_ingest))
    yield
    await driver.close()


app = FastAPI(title="Ingestion Service", version="1.0", lifespan=lifespan)


@app.post("/ingest/menu")
async def ingest_menu(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {path}")
    count = await ingest_menu_csv(path, driver)
    await _clear_rag_cache()
    return {"ingested": count, "type": "menu"}


@app.post("/ingest/faq")
async def ingest_faq(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {path}")
    count = await ingest_faq_csv(path, driver)
    await _clear_rag_cache()
    return {"ingested": count, "type": "faq"}


@app.post("/ingest/doc")
async def ingest_doc(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {path}")
    count = await ingest_document(path, driver)
    await _clear_rag_cache()
    return {"ingested": count, "type": "document"}


@app.post("/ingest/upload")
async def ingest_upload(bg: BackgroundTasks, file: UploadFile = File(...)):
    dest = os.path.join(WATCH_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    bg.add_task(auto_ingest, dest)
    return {"message": f"Queued: {file.filename}"}


@app.get("/health")
async def health():
    return {"status": "ok"}
