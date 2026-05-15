import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from session_cache import SessionCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_RAG_URL = os.getenv("GRAPH_RAG_URL", "http://graph_rag:8004")
GENERATOR_URL = os.getenv("GENERATOR_URL", "http://localhost:8080/v1")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ROUTER_URL = os.getenv("ROUTER_URL", GENERATOR_URL)  # Same or different

session_cache: SessionCache = None

SYSTEM_PROMPT = """Bạn là trợ lý nhà hàng/quán cà phê. 
Chỉ trả lời dựa trên context được cung cấp bên dưới.
Nếu thông tin không có trong context, hãy nói "Tôi không có thông tin về điều này".
TUYỆT ĐỐI không bịa đặt thông tin về giá, menu, hay chính sách."""

CLAUSE_SPLIT_RE = re.compile(r'(?<=[.?!;,])\s+')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_cache
    session_cache = SessionCache(REDIS_URL)
    yield


app = FastAPI(title="LLM Serving", version="1.0", lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: bool = True
    use_rag: bool = True


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    context_used: List[Dict[str, Any]]
    latency_ms: float

async def retrieve_context(query: str) -> List[Dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{GRAPH_RAG_URL}/search",
            json={"query": query, "top_k": 5, "use_cache": True},
        )
        r.raise_for_status()
        return r.json()["results"]


def format_context(results: List[Dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        if r.get("type") == "menu":
            parts.append(
                f"[{i}] Menu item: {r.get('name')} | "
                f"Giá: {r.get('price')} | "
                f"Mô tả: {r.get('description')} | "
                f"Thành phần: {r.get('ingredients')}"
            )
        else:
            parts.append(f"[{i}] {r.get('text', '')}")
    return "\n".join(parts)

async def call_generator_stream(messages: List[Dict]) -> AsyncGenerator[str, None]:
    """Stream tokens from vLLM/SGLang OpenAI-compatible API."""
    payload = {
        "model": "generator",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{GENERATOR_URL}/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":  
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue


async def call_generator_full(messages: List[Dict]) -> str:
    """Non-streaming fallback."""
    payload = {
        "model": "generator",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{GENERATOR_URL}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def stream_response(query: str, session_id: str) -> AsyncGenerator[str, None]:
    context_docs = await retrieve_context(query)
    context_str = format_context(context_docs)

    history = await session_cache.get_history(session_id)
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_str}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    full_response = []
    clause_buffer = []

    async for token in call_generator_stream(messages):
        full_response.append(token)
        clause_buffer.append(token)

        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        buffered = "".join(clause_buffer)
        parts = CLAUSE_SPLIT_RE.split(buffered)
        if len(parts) > 1:
            for clause in parts[:-1]:
                clause = clause.strip()
                if clause:
                    yield f"data: {json.dumps({'type': 'clause', 'text': clause})}\n\n"
            clause_buffer = [parts[-1]]

    if clause_buffer:
        remaining = "".join(clause_buffer).strip()
        if remaining:
            yield f"data: {json.dumps({'type': 'clause', 'text': remaining})}\n\n"

    complete = "".join(full_response)
    await session_cache.append_message(session_id, "user", query)
    await session_cache.append_message(session_id, "assistant", complete)

    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """B2.2 — True token streaming via SSE."""
    session_id = req.session_id or str(uuid.uuid4())
    return EventSourceResponse(
        stream_response(req.query, session_id),
        media_type="text/event-stream",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming endpoint."""
    t0 = time.time()
    session_id = req.session_id or str(uuid.uuid4())

    context_docs = await retrieve_context(req.query)
    context_str = format_context(context_docs)

    history = await session_cache.get_history(session_id)
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_str}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": req.query})

    try:
        answer = await call_generator_full(messages)
    except Exception as e:
        logger.error(f"Generator error: {e}")
        answer = "Xin lỗi, hệ thống đang gặp sự cố."

    await session_cache.append_message(session_id, "user", req.query)
    await session_cache.append_message(session_id, "assistant", answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        context_used=context_docs,
        latency_ms=(time.time() - t0) * 1000,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    await session_cache.clear(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/cache/info")
async def cache_info():
    """
    B2.3 Cache layer summary:
    Layer 1 - Model Cache:   Disk (~4.8GB generator AWQ + ~1.1GB router AWQ)
    Layer 2 - Graph Cache:   Neo4j persistent (RAM: ~512MB heap + 512MB pagecache)
    Layer 3 - KV Cache:      SGLang/vLLM VRAM (allocated by mem-fraction-static)
    Layer 4 - Session Cache: Redis in-memory (max 512MB, TTL=3600s)
    Semantic Cache:          Redis in-memory (cosine sim ≥ 0.95, TTL=3600s)
    """
    return {
        "layers": [
            {
                "layer": 1,
                "name": "Model Cache (Disk)",
                "storage": "Disk",
                "size_estimate": "~6GB total (router ~1.1GB AWQ + generator ~4.8GB AWQ)",
                "ttl": "permanent",
            },
            {
                "layer": 2,
                "name": "Graph Cache (Neo4j)",
                "storage": "Persistent DB",
                "size_estimate": "heap 1GB + pagecache 512MB RAM",
                "ttl": "permanent until re-ingestion",
            },
            {
                "layer": 3,
                "name": "KV Cache (VRAM)",
                "storage": "GPU VRAM",
                "size_estimate": "~6GB VRAM (remaining after models)",
                "ttl": "per request / prefix-sharing via lpm policy",
            },
            {
                "layer": 4,
                "name": "Session Cache (Redis)",
                "storage": "In-memory",
                "size_estimate": "≤512MB (maxmemory)",
                "ttl": "3600s per session",
            },
            {
                "layer": "5 (bonus)",
                "name": "Semantic Cache (Redis)",
                "storage": "In-memory",
                "size_estimate": "≤512MB shared with session cache",
                "ttl": "3600s, cosine similarity threshold 0.95",
            },
        ]
    }
