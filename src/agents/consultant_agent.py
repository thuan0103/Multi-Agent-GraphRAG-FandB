import os
import json
import logging
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import load_dotenv
from .base import BaseAgent, AgentResponse
from .prompts.consultant_prompt import CONSULTANT_SYSTEM_VI, CONSULTANT_SYSTEM_EN

logger = logging.getLogger(__name__)

load_dotenv()

class ConsultantAgent(BaseAgent):

    agent_type = "consultant"

    def __init__(self, config: dict, menu_path: str = "data/menu.json"):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=os.getenv("API_OPENAI"))
        self.menu_docs = self._load_menu_docs(menu_path)

    def _load_menu_docs(self, path: str) -> list[dict]:
        if not Path(path).exists():
            return self._default_menu_docs()

        menu = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = []
        for category, items in menu.items():
            for item in items:
                doc = {
                    "name": item.get("name", ""),
                    "name_en": item.get("name_en", ""),
                    "price": item.get("price", 0),
                    "category": category,
                    "description": item.get("description", ""),
                    "tags": item.get("tags", []),  # ["cold", "sweet", "popular"]
                }
                docs.append(doc)
        return docs

    def _default_menu_docs(self) -> list[dict]:
        return [
            {"name": "Cà phê sữa đá", "name_en": "Iced Milk Coffee", "price": 35000,
             "category": "drinks", "description": "Cà phê đậm đà pha sữa đặc, uống lạnh",
             "tags": ["cold", "sweet", "popular", "coffee"]},
            {"name": "Bạc xỉu", "name_en": "Bac Xiu", "price": 35000,
             "category": "drinks", "description": "Nhiều sữa hơn cà phê, vị ngọt nhẹ",
             "tags": ["cold", "sweet", "mild", "coffee"]},
            {"name": "Sinh tố bơ", "name_en": "Avocado Smoothie", "price": 55000,
             "category": "drinks", "description": "Bơ tươi xay mịn, béo ngậy bổ dưỡng",
             "tags": ["cold", "creamy", "healthy", "fruity"]},
            {"name": "Trà sữa", "name_en": "Milk Tea", "price": 45000,
             "category": "drinks", "description": "Trà oolong pha sữa, có topping trân châu",
             "tags": ["cold", "sweet", "popular", "tea"]},
        ]

    def _retrieve_relevant(self, query: str, top_k: int = 4) -> list[dict]:
        query_lower = query.lower()

        scored = []
        for doc in self.menu_docs:
            score = 0
            searchable = (
                doc["name"].lower() + " " +
                doc["name_en"].lower() + " " +
                doc.get("description", "").lower() + " " +
                " ".join(doc.get("tags", []))
            )

            keywords = query_lower.split()
            for kw in keywords:
                if kw in searchable:
                    score += 2

            if "popular" in doc.get("tags", []):
                score += 1
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _format_menu_context(self, docs: list[dict]) -> str:
        lines = []
        for doc in docs:
            price_str = f"{doc['price']:,}đ"
            tags_str = ", ".join(doc.get("tags", []))
            lines.append(
                f"- {doc['name']} ({doc['name_en']}): {price_str}\n"
                f"  Mô tả: {doc.get('description', 'N/A')}\n"
                f"  Tags: {tags_str}"
            )
        return "\n".join(lines)

    async def _process(
        self,
        query: str,
        history: list[dict],
        language: str,
        session_id: str,
    ) -> AgentResponse:
        relevant_docs = self._retrieve_relevant(query)
        menu_context = self._format_menu_context(relevant_docs)
        history_context = self._build_history_context(history)

        system_prompt = (CONSULTANT_SYSTEM_VI if language == "vi" else CONSULTANT_SYSTEM_EN).format(
            menu_context=menu_context,
            history=history_context or "Chưa có / None",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
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
                "retrieved_items": [d["name"] for d in relevant_docs],
                "rag_docs_count": len(relevant_docs),
            },
            language=language,
        )