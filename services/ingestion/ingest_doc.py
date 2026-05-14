"""
B1.3 — Ingest PDF / DOCX
Folder: services/ingestion/ingest_doc.py

- Semantic chunking (gradient breakpoint)
- Entity extraction via LLM
- Entity deduplication (Jaccard ≥ 0.85)
- NEXT/PREV relationships between consecutive chunks
- MENTIONS relationships chunk → entity
"""
import os
import uuid
import json
import logging
import httpx
from typing import List, Tuple
from rapidfuzz import fuzz
from embedder import embed_texts

logger = logging.getLogger(__name__)

LLM_URL = os.getenv("LLM_URL", "http://llm_serving:8000")
JACCARD_THRESHOLD = 0.85


# ── Text extraction ─────────────────────────────────────────────

def extract_text_pdf(path: str) -> str:
    import PyPDF2
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_text_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── Semantic chunking (gradient breakpoint) ─────────────────────

async def semantic_chunk(text: str, max_tokens: int = 300) -> List[str]:
    """Split text into sentences, embed, find gradient breakpoints."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= 1:
        return sentences

    embeddings = await embed_texts(sentences)

    # Cosine similarity between consecutive sentences
    import math

    def cos_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x ** 2 for x in a))
        nb = math.sqrt(sum(x ** 2 for x in b))
        return dot / (na * nb + 1e-9)

    sims = [cos_sim(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    mean_sim = sum(sims) / len(sims)
    std_sim = math.sqrt(sum((s - mean_sim) ** 2 for s in sims) / len(sims))
    threshold = mean_sim - std_sim  # breakpoint below mean-1std

    chunks, current = [], [sentences[0]]
    token_count = len(sentences[0].split())
    for i, sim in enumerate(sims):
        next_sent = sentences[i + 1]
        next_tokens = len(next_sent.split())
        if sim < threshold or token_count + next_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [next_sent]
            token_count = next_tokens
        else:
            current.append(next_sent)
            token_count += next_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Entity extraction via LLM ────────────────────────────────────

async def extract_entities(chunk_text: str) -> List[str]:
    prompt = (
        "Extract named entities from the text below. "
        "Return ONLY a JSON array of strings. No explanation.\n\n"
        f"Text: {chunk_text[:800]}"
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": "router",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0,
                },
            )
            content = r.json()["choices"][0]["message"]["content"]
            entities = json.loads(content)
            return [str(e).strip() for e in entities if str(e).strip()]
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return []


# ── Entity deduplication (Jaccard ≥ 0.85) ───────────────────────

def deduplicate_entities(existing: List[str], new_entities: List[str]) -> Tuple[List[str], dict]:
    """Returns (final_list, mapping: new_name → canonical_name)"""
    canonical = list(existing)
    mapping = {}
    for entity in new_entities:
        matched = False
        for canon in canonical:
            score = fuzz.token_set_ratio(entity.lower(), canon.lower()) / 100.0
            if score >= JACCARD_THRESHOLD:
                mapping[entity] = canon
                matched = True
                break
        if not matched:
            canonical.append(entity)
            mapping[entity] = entity
    return canonical, mapping


# ── Neo4j writers ────────────────────────────────────────────────

UPSERT_CHUNK = """
MERGE (c:Chunk {id: $id})
SET c.text      = $text,
    c.source    = $source,
    c.doc_type  = 'document',
    c.embedding = $embedding
"""

UPSERT_ENTITY = """
MERGE (e:Entity {id: $id})
SET e.name = $name
"""

LINK_NEXT = """
MATCH (a:Chunk {id: $id_a}), (b:Chunk {id: $id_b})
MERGE (a)-[:NEXT]->(b)
MERGE (b)-[:PREV]->(a)
"""

LINK_MENTIONS = """
MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
MERGE (c)-[:MENTIONS]->(e)
"""


async def ingest_document(path: str, driver):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw = extract_text_pdf(path)
    elif ext in (".docx", ".doc"):
        raw = extract_text_docx(path)
    else:
        with open(path, encoding="utf-8") as f:
            raw = f.read()

    chunks = await semantic_chunk(raw)
    embeddings = await embed_texts(chunks)

    # Track existing entity names for deduplication
    existing_entities: List[str] = []

    chunk_ids = []
    async with driver.session() as session:
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            cid = str(uuid.uuid4())
            chunk_ids.append(cid)
            await session.run(
                UPSERT_CHUNK,
                id=cid,
                text=chunk_text,
                source=os.path.basename(path),
                embedding=emb,
            )
            # Entity extraction
            raw_entities = await extract_entities(chunk_text)
            existing_entities, mapping = deduplicate_entities(existing_entities, raw_entities)
            for entity_name in set(mapping.values()):
                eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, entity_name.lower()))
                await session.run(UPSERT_ENTITY, id=eid, name=entity_name)
                await session.run(LINK_MENTIONS, chunk_id=cid, entity_id=eid)

        # NEXT/PREV links
        for i in range(len(chunk_ids) - 1):
            await session.run(LINK_NEXT, id_a=chunk_ids[i], id_b=chunk_ids[i + 1])

    logger.info(f"Ingested {len(chunks)} chunks from {path}")
    return len(chunks)
