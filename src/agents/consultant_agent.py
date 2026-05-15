# src/agents/consultant_agent.py
"""
Consultant Agent: tư vấn món uống.
Pipeline ưu tiên: fulltext entity search → vector search (graph_rag:8004)
Fallback: in-memory keyword search từ data/menu.json
"""

import os
import re
import json
import logging
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .base import BaseAgent, AgentResponse, _make_openai_client
from .prompts.consultant_prompt import CONSULTANT_SYSTEM_VI, CONSULTANT_SYSTEM_EN

logger = logging.getLogger(__name__)

load_dotenv()

GRAPH_RAG_URL = os.getenv("GRAPH_RAG_URL", "http://localhost:8004")

_FULLTEXT_STOPWORDS = frozenset({
    'tôi', 'muốn', 'cho', 'một', 'cái', 'ly', 'cốc', 'ơi', 'bạn',
    'em', 'ạ', 'à', 'nhé', 'đi', 'thôi', 'mình', 'có', 'không', 'gì',
    'là', 'và', 'với', 'thêm', 'bớt', 'xin', 'hỏi', 'về', 'thì', 'được',
    'của', 'hay', 'hoặc', 'nào', 'đó', 'này', 'kia', 'đây', 'lấy', 'đặt',
    'order', 'gọi', 'đơn', 'giúp', 'i', 'me', 'want', 'please', 'a', 'an', 'the',
    'gợi', 'ý', 'tư', 'vấn', 'món', 'nên', 'uống', 'ăn', 'recommend',
})


def _clean_for_fulltext(text: str) -> str:
    """Strip Lucene special chars and common stop words for Neo4j fulltext query."""
    text = re.sub(r'[+\-&|!(){}[\]^"~*?:\\/]', ' ', text)
    words = [w for w in text.split() if w.lower() not in _FULLTEXT_STOPWORDS and len(w) > 1]
    return ' '.join(words)


class ConsultantAgent(BaseAgent):

    agent_type = "consultant"

    def __init__(self, config: dict, menu_path: str = "data/menu.json"):
        super().__init__(config)
        self.client, self.llm_model = _make_openai_client()
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

    # ── Primary: fulltext entity search ──────────────────────────────

    async def _search_menu_fulltext(self, query: str) -> list[dict]:
        """Call /menu/fulltext with cleaned query terms."""
        clean_q = _clean_for_fulltext(query)
        if not clean_q:
            return []
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{GRAPH_RAG_URL}/menu/fulltext",
                params={"q": clean_q, "limit": 5},
            )
            resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{
            "name":        it.get("name", ""),
            "name_en":     "",
            "price":       int(it.get("price") or 0),
            "category":    it.get("category", ""),
            "description": it.get("description", ""),
            "ingredients": it.get("ingredients", ""),
            "tags":        [it.get("category", "")],
            "score":       it.get("score", 0.0),
            "source":      "neo4j_fulltext",
        } for it in items]

    # ── Secondary: vector search via /search ──────────────────────────

    async def _search_graph_rag(self, query: str) -> tuple[list[dict], bool]:
        """Call graph_rag:8004/search, filter MenuItem nodes."""
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
                    "source":      "neo4j_vector",
                })
        return docs, data.get("from_cache", False)

    async def _search_menu_for_context(self, query: str) -> tuple[list[dict], str]:
        """
        fulltext search → vector search → in-memory fallback.
        Returns (docs, source_label).
        """
        # 1. Fulltext entity search
        try:
            docs = await self._search_menu_fulltext(query)
            if docs:
                return docs, "neo4j_fulltext"
        except Exception as e:
            logger.warning(f"[ConsultantAgent] fulltext search failed: {e}")

        # 2. Vector search
        try:
            docs, _ = await self._search_graph_rag(query)
            if docs:
                return docs, "neo4j_vector"
        except Exception as e:
            logger.warning(f"[ConsultantAgent] vector search failed: {e}")

        # 3. In-memory fallback
        docs = self._retrieve_relevant_local(query)
        return docs, "fallback"

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
        docs, rag_source = await self._search_menu_for_context(query)

        menu_context    = self._format_menu_context(docs)
        history_context = self._build_history_context(history)

        system_prompt = (CONSULTANT_SYSTEM_VI if language == "vi" else CONSULTANT_SYSTEM_EN).format(
            menu_context=menu_context,
            history=history_context or "Chưa có / None",
        )

        response = await self.client.chat.completions.create(
            model=self.llm_model,
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
                "rag_source":      rag_source,
                "rag_docs_count":  len(docs),
            },
            language=language,
        )
