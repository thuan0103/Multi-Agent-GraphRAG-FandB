# src/agents/consultant_agent.py
"""
Consultant Agent: tư vấn món uống.
Pipeline ưu tiên: graph_rag:8004 (Neo4j + bge-m3 + reranker)
Fallback: in-memory keyword search từ data/menu.json
"""

import os
import json
import logging
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .base import BaseAgent, AgentResponse
from .prompts.consultant_prompt import CONSULTANT_SYSTEM_VI, CONSULTANT_SYSTEM_EN

logger = logging.getLogger(__name__)

load_dotenv()

GRAPH_RAG_URL = os.getenv("GRAPH_RAG_URL", "http://localhost:8004")


class ConsultantAgent(BaseAgent):

    agent_type = "consultant"

    def __init__(self, config: dict, menu_path: str = "data/menu.json"):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=os.getenv("API_OPENAI"))
        self.menu_docs = self._load_menu_docs(menu_path)   # dùng khi fallback

    def _load_menu_docs(self, path: str) -> list[dict]:
        if not Path(path).exists():
            return self._default_menu_docs()
        menu = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = []
        for category, items in menu.items():
            for item in items:
                docs.append({
                    "name":        item.get("name", ""),
                    "name_en":     item.get("name_en", ""),
                    "price":       item.get("price", 0),
                    "category":    category,
                    "description": item.get("description", ""),
                    "tags":        item.get("tags", []),
                })
        return docs

    def _default_menu_docs(self) -> list[dict]:
        return [
            {"name": "Cà phê sữa đá", "name_en": "Iced Milk Coffee", "price": 35000,
             "category": "drinks", "description": "Cà phê đậm đà pha sữa đặc, uống lạnh",
             "tags": ["cold", "sweet", "popular", "coffee"]},
            {"name": "Sinh tố bơ", "name_en": "Avocado Smoothie", "price": 55000,
             "category": "drinks", "description": "Bơ tươi xay mịn, béo ngậy bổ dưỡng",
             "tags": ["cold", "creamy", "healthy", "fruity"]},
        ]

    # ── RAG: gọi graph_rag:8004 ──────────────────────────────────────

    async def _search_graph_rag(self, query: str) -> tuple[list[dict], bool]:
        """
        Gọi graph_rag:8004/search, lọc lấy MenuItem nodes.
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
            if r.get("type") == "menu":
                docs.append({
                    "name":        r.get("name", ""),
                    "name_en":     "",
                    "price":       int(r.get("price", 0) or 0),
                    "category":    r.get("category", ""),
                    "description": r.get("description", ""),
                    "ingredients": r.get("ingredients", ""),
                    "tags":        [r.get("category", "")],
                    "score":       r.get("rerank_score", 0.0),
                    "source":      "neo4j",
                })
        return docs, data.get("from_cache", False)

    # ── Fallback: in-memory keyword search ───────────────────────────

    def _retrieve_relevant_local(self, query: str, top_k: int = 4) -> list[dict]:
        query_lower = query.lower()
        scored = []
        for doc in self.menu_docs:
            searchable = (
                doc["name"].lower() + " " +
                doc["name_en"].lower() + " " +
                doc.get("description", "").lower() + " " +
                " ".join(doc.get("tags", []))
            )
            score = sum(2 for kw in query_lower.split() if kw in searchable)
            if "popular" in doc.get("tags", []):
                score += 1
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    # ── Format context ────────────────────────────────────────────────

    def _format_menu_context(self, docs: list[dict]) -> str:
        lines = []
        for doc in docs:
            price_str = f"{doc['price']:,}đ"
            extra = doc.get("ingredients", "") or doc.get("description", "")
            lines.append(
                f"- {doc['name']} ({doc.get('name_en','')}) [{doc.get('category','')}]: {price_str}\n"
                f"  {extra}"
            )
        return "\n".join(lines)

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
                logger.info(f"[ConsultantAgent] graph_rag returned {len(docs)} items "
                            f"(cache={from_cache})")
            else:
                raise ValueError("no MenuItem results from graph_rag")
        except Exception as e:
            logger.warning(f"[ConsultantAgent] graph_rag unavailable: {e} → fallback")
            docs = self._retrieve_relevant_local(query)

        menu_context    = self._format_menu_context(docs)
        history_context = self._build_history_context(history)

        system_prompt = (CONSULTANT_SYSTEM_VI if language == "vi" else CONSULTANT_SYSTEM_EN).format(
            menu_context=menu_context,
            history=history_context or "Chưa có / None",
        )

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.7,
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()

        return AgentResponse(
            text=answer,
            agent_type=self.agent_type,
            session_id=session_id,
            latency_ms=0.0,
            metadata={
                "retrieved_items": [d["name"] for d in docs],
                "rag_source":      "neo4j" if from_graph_rag else "fallback",
                "from_cache":      from_cache,
                "rag_docs_count":  len(docs),
            },
            language=language,
        )
