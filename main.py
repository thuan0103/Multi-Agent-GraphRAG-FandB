import asyncio
import logging
import uuid
import uvicorn
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.router import IntentClassifier
from src.agents import OrderAgent, ConsultantAgent, FAQAgent
from src.session import SessionStore, ConversationSummarizer, SessionCleanup
from src.queue import RequestQueue, retry_with_backoff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

with open('config.yaml', 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

router_classifier = IntentClassifier()

summarizer = ConversationSummarizer()
session_store = SessionStore(
    history_window=CONFIG["session"]["history_window"],
    ttl_minutes=CONFIG["session"]["ttl_minutes"],
    context_window_threshold=CONFIG["session"]["context_window_threshold"],
)
session_store.set_summarizer(summarizer)
cleanup = SessionCleanup(
    session_store,
    interval_seconds=CONFIG["session"]["cleanup_interval_seconds"],
)

request_queue = RequestQueue(
    max_concurrent=CONFIG["queue"]["max_concurrent_llm"],
    request_timeout=CONFIG["queue"]["request_timeout_seconds"],
)

agents = {}    

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    router_classifier.load()

    agents["order"] = OrderAgent(CONFIG)
    agents["consultant"] = ConsultantAgent(CONFIG)
    agents["faq"] = FAQAgent(CONFIG)

    await cleanup.start()
    logger.info("All components ready")

    yield

    await cleanup.stop()
    logger.info("Shutdown complete")


app = FastAPI(
    title="MAS-LLM Coffee Shop",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = None     


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    latency_ms: float
    agent_latency_ms: float


@retry_with_backoff(max_attempts=3, base_delay=1.0)
async def _call_agent(agent, query, history, session_id):
    return await agent.handle(query, history, session_id)


async def _process_chat(message: str, session_id: str) -> ChatResponse:
    router_result = router_classifier.classify(message)
    intent = router_result["action"]
    router_latency = router_result["latency_ms"]

    if intent == "ignore":
        return ChatResponse(
            reply="Xin chào! Tôi có thể giúp gì cho bạn? 😊" if True else "Hello! How can I help you?",
            session_id=session_id,
            intent=intent,
            latency_ms=router_latency,
            agent_latency_ms=0.0,
        )

    history = await session_store.get_history(session_id)

    agent = agents.get(intent)
    if not agent:
        raise HTTPException(status_code=500, detail=f"No agent for intent: {intent}")

    agent_response = await request_queue.submit(
        _call_agent, agent, message, history, session_id
    )

    await session_store.add_turn(session_id, "user", message)
    await session_store.add_turn(
        session_id, "assistant", agent_response.text,
        metadata=agent_response.metadata,
    )

    return ChatResponse(
        reply=agent_response.text,
        session_id=session_id,
        intent=intent,
        latency_ms=router_latency + agent_response.latency_ms,
        agent_latency_ms=agent_response.latency_ms,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    try:
        return await _process_chat(req.message, session_id)
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health():
    queue_stats = await request_queue.stats()
    session_stats = await session_store.stats()
    return {
        "status": "ok",
        "router_loaded": router_classifier.model.is_loaded(),
        "queue": queue_stats,
        "sessions": session_stats,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await session_store.delete(session_id)
    return {"deleted": session_id}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=18000, reload=True)