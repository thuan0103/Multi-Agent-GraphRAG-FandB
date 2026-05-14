# src/agents/faq_agent.py
"""
FAQ Agent: hỏi đáp chung với Hybrid RAG.
Pipeline ưu tiên: graph_rag:8004 (Neo4j + bge-m3 + reranker)
Fallback: in-memory keyword/semantic search từ data/faq.json
"""

import logging
import os
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .base import BaseAgent, AgentResponse
from .prompts.faq_prompt import FAQ_SYSTEM_VI, FAQ_SYSTEM_EN

load_dotenv()

logger = logging.getLogger(__name__)

GRAPH_RAG_URL = os.getenv("GRAPH_RAG_URL", "http://localhost:8004")

DEFAULT_FAQ = [
    {"id": "wifi", "question": "Wifi quán tên gì? Mật khẩu là gì?",
     "answer": "Wifi: CafeXYZ_Guest | Mật khẩu: cafe2024", "tags": ["wifi", "internet"]},
    {"id": "hours", "question": "Quán mở cửa mấy giờ? Đóng cửa mấy giờ?",
     "answer": "Quán mở 7:00 - 22:00 tất cả các ngày trong tuần.", "tags": ["hours", "time"]},
    {"id": "address", "question": "Quán ở đâu? Địa chỉ?",
     "answer": "123 Nguyễn Huệ, Q.1, TP.HCM. Gần công viên 23/9.", "tags": ["address", "location"]},
    {"id": "payment", "question": "Thanh toán bằng gì?",
     "answer": "Nhận tiền mặt, chuyển khoản, QR MoMo/ZaloPay.", "tags": ["payment"]},
]


class FAQAgent(BaseAgent):

    agent_type = "faq"

    def __init__(self, config: dict, faq_path: str = "data/faq.json"):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=os.getenv("API_OPENAI"))
        self.faq_docs = self._load_faq(faq_path)   # dùng khi fallback

    def _load_faq(self, path: str) -> list[dict]:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
        logger.warning(f"FAQ file not found at {path}, using defaults")
        return DEFAULT_FAQ

    # ── RAG: gọi graph_rag:8004 ──────────────────────────────────────

    async def _search_graph_rag(self, query: str) -> tuple[list[dict], bool]:
        """
        Gọi graph_rag:8004/search.
        Trả về (docs, from_cache).
        Raise exception nếu service không available.
        """
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{GRAPH_RAG_URL}/search",
                json={"query": query, "top_k": 5},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        docs = []
        for r in results:
            # FAQ chunk: có doc_type='faq', text="Q: ... A: ...", question
            if r.get("type") == "chunk":
                docs.append({
                    "id":       r.get("id", ""),
                    "question": r.get("question", ""),
                    "answer":   r.get("text", ""),
                    "tags":     [r.get("doc_type", "faq")],
                    "score":    r.get("rerank_score", 0.0),
                    "source":   "neo4j",
                })
        return docs, data.get("from_cache", False)

    # ── Fallback: in-memory search ────────────────────────────────────

    def _keyword_search(self, query: str, top_k: int = 3) -> list[dict]:
        query_lower = query.lower()
        results = []
        for doc in self.faq_docs:
            score = sum(3.0 for tag in doc.get("tags", []) if tag in query_lower)
            score += sum(1.0 for w in doc["question"].lower().split()
                         if len(w) > 2 and w in query_lower)
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in results[:top_k]]

    def _semantic_search(self, query: str, top_k: int = 3) -> list[dict]:
        query_chars = set(query.lower())
        results = []
        for doc in self.faq_docs:
            doc_chars = set(doc["question"].lower() + " " + doc["answer"].lower())
            overlap = len(query_chars & doc_chars) / max(len(query_chars), 1)
            results.append((overlap, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in results[:top_k]]

    def _hybrid_retrieve_local(self, query: str) -> list[dict]:
        kw  = self._keyword_search(query, top_k=3)
        sem = self._semantic_search(query, top_k=3)
        seen, merged = set(), []
        for doc in kw + sem:
            if doc["id"] not in seen:
                merged.append(doc)
                seen.add(doc["id"])
        return merged[:4]

    # ── Format context ────────────────────────────────────────────────

    def _format_faq_context(self, docs: list[dict]) -> str:
        lines = []
        for doc in docs:
            q = doc.get("question", "")
            a = doc.get("answer", "")
            # graph_rag trả về "Q: ... A: ..." trong field answer/text
            if a.startswith("Q:"):
                lines.append(a)
            else:
                lines.append(f"Q: {q}\nA: {a}" if q else a)
        return "\n\n".join(lines)

    # ── Main process ──────────────────────────────────────────────────

    async def _process(
        self,
        query: str,
        history: list[dict],
        language: str,
        session_id: str,
    ) -> AgentResponse:
        # Ưu tiên graph_rag thật, fallback về in-memory
        from_graph_rag = False
        from_cache     = False
        try:
            docs, from_cache = await self._search_graph_rag(query)
            if docs:
                from_graph_rag = True
                logger.info(f"[FAQAgent] graph_rag returned {len(docs)} chunks "
                            f"(cache={from_cache})")
            else:
                raise ValueError("empty result from graph_rag")
        except Exception as e:
            logger.warning(f"[FAQAgent] graph_rag unavailable: {e} → fallback")
            docs = self._hybrid_retrieve_local(query)

        faq_context    = self._format_faq_context(docs)
        history_context = self._build_history_context(history)

        system_prompt = (FAQ_SYSTEM_VI if language == "vi" else FAQ_SYSTEM_EN).format(
            faq_context=faq_context,
            history=history_context or "Chưa có / None",
        )

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.2,
            max_tokens=300,
        )

        answer = response.choices[0].message.content.strip()

        return AgentResponse(
            text=answer,
            agent_type=self.agent_type,
            session_id=session_id,
            latency_ms=0.0,
            metadata={
                "retrieved_faq_ids": [d.get("id", "") for d in docs],
                "rag_source":        "neo4j" if from_graph_rag else "fallback",
                "from_cache":        from_cache,
                "rag_pipeline":      "graph_rag+reranker" if from_graph_rag else "keyword+ngram",
            },
            language=language,
        )
