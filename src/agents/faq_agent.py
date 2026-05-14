# src/agents/faq_agent.py
"""
FAQ Agent: hỏi đáp chung với Hybrid RAG.
Pipeline: Dual-Domain Search → keyword expand → Late Reranking.
"""

import logging
from pathlib import Path
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .base import BaseAgent, AgentResponse
from .prompts.faq_prompt import FAQ_SYSTEM_VI, FAQ_SYSTEM_EN

load_dotenv()

logger = logging.getLogger(__name__)

# FAQ knowledge base mẫu — production: load từ DB hoặc file
DEFAULT_FAQ = [
    {"id": "wifi", "question": "Wifi quán tên gì? Mật khẩu là gì?",
     "answer": "Wifi: CafeXYZ_Guest | Mật khẩu: cafe2024", "tags": ["wifi", "internet"]},
    {"id": "hours", "question": "Quán mở cửa mấy giờ? Đóng cửa mấy giờ?",
     "answer": "Quán mở 7:00 - 22:00 tất cả các ngày trong tuần.", "tags": ["hours", "time", "open", "close"]},
    {"id": "parking", "question": "Có chỗ để xe không?",
     "answer": "Quán có bãi giữ xe miễn phí phía sau, sức chứa 30 xe máy và 5 ô tô.", "tags": ["parking", "xe"]},
    {"id": "payment", "question": "Thanh toán bằng gì? Có nhận thẻ không?",
     "answer": "Nhận tiền mặt, thẻ ngân hàng, và chuyển khoản (QR). Không nhận thẻ tín dụng.", "tags": ["payment", "card", "cash"]},
    {"id": "reservation", "question": "Có đặt bàn trước được không?",
     "answer": "Có, đặt bàn qua số điện thoại 0901234567 hoặc Facebook @CafeXYZ.", "tags": ["reservation", "book", "table"]},
    {"id": "address", "question": "Quán ở đâu? Địa chỉ?",
     "answer": "123 Nguyễn Huệ, Q.1, TP.HCM. Gần công viên 23/9.", "tags": ["address", "location", "where"]},
    {"id": "pet", "question": "Có cho mang thú cưng vào không?",
     "answer": "Quán có khu vực ngoài trời thân thiện với thú cưng. Khu trong nhà không cho phép.", "tags": ["pet", "dog", "cat"]},
    {"id": "smoking", "question": "Có khu vực hút thuốc không?",
     "answer": "Có khu vực hút thuốc riêng ngoài trời, cách khu ngồi chính 5m.", "tags": ["smoking", "cigarette"]},
]


class FAQAgent(BaseAgent):

    agent_type = "faq"

    def __init__(self, config: dict, faq_path: str = "data/faq.json"):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=os.getenv("API_OPENAI"))
        self.faq_docs = self._load_faq(faq_path)

    def _load_faq(self, path: str) -> list[dict]:
        if Path(path).exists():
            return json.loads(Path(path).read_text())
        logger.warning(f"FAQ file not found at {path}, using defaults")
        return DEFAULT_FAQ

    # ──────────────────────────────────────
    # Hybrid RAG Pipeline
    # ──────────────────────────────────────

    def _keyword_search(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """Domain 1: keyword/tag matching."""
        query_lower = query.lower()
        results = []
        for doc in self.faq_docs:
            score = 0.0
            # Match tags
            for tag in doc.get("tags", []):
                if tag in query_lower:
                    score += 3.0
            # Match question words
            for word in doc["question"].lower().split():
                if len(word) > 2 and word in query_lower:
                    score += 1.0
            if score > 0:
                results.append((score, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _semantic_search(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """
        Domain 2: semantic similarity (simplified).
        Production: dùng sentence-transformers + FAISS.
        Hiện tại: character n-gram overlap làm proxy.
        """
        query_chars = set(query.lower())
        results = []
        for doc in self.faq_docs:
            doc_chars = set(doc["question"].lower() + " " + doc["answer"].lower())
            overlap = len(query_chars & doc_chars) / max(len(query_chars), 1)
            results.append((overlap, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _graph_expand(self, docs: list[dict]) -> list[dict]:
        """
        Graph expansion: từ docs đã tìm, mở rộng sang docs liên quan cùng tag.
        Đơn giản: tìm docs có tag overlap.
        """
        found_ids = {d["id"] for d in docs}
        found_tags = set()
        for doc in docs:
            found_tags.update(doc.get("tags", []))

        expanded = list(docs)
        for candidate in self.faq_docs:
            if candidate["id"] in found_ids:
                continue
            candidate_tags = set(candidate.get("tags", []))
            if candidate_tags & found_tags:   # có tag chung
                expanded.append(candidate)
                found_ids.add(candidate["id"])

        return expanded

    def _late_rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Late reranking: score lại toàn bộ candidates sau khi gộp.
        Kết hợp keyword score + position bias (ưu tiên docs tìm trước).
        """
        query_lower = query.lower()
        scored = []
        for i, doc in enumerate(candidates):
            score = 0.0
            # Keyword score
            for tag in doc.get("tags", []):
                if tag in query_lower:
                    score += 3.0
            for word in doc["question"].lower().split():
                if len(word) > 2 and word in query_lower:
                    score += 1.0
            # Position bias — docs xuất hiện sớm hơn được ưu tiên nhẹ
            score += max(0, (len(candidates) - i)) * 0.1
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _hybrid_retrieve(self, query: str) -> list[dict]:
        """
        Full pipeline: Dual-Domain → Graph Expand → Late Rerank.
        """
        # Domain 1: keyword search
        kw_results = [doc for _, doc in self._keyword_search(query, top_k=3)]

        # Domain 2: semantic search
        sem_results = [doc for _, doc in self._semantic_search(query, top_k=3)]

        # Merge, dedup
        seen_ids = set()
        candidates = []
        for doc in kw_results + sem_results:
            if doc["id"] not in seen_ids:
                candidates.append(doc)
                seen_ids.add(doc["id"])

        # Graph expansion
        candidates = self._graph_expand(candidates)

        # Late reranking
        return self._late_rerank(query, candidates, top_k=3)

    def _format_faq_context(self, docs: list[dict]) -> str:
        lines = []
        for doc in docs:
            lines.append(f"Q: {doc['question']}\nA: {doc['answer']}")
        return "\n\n".join(lines)

    async def _process(
        self,
        query: str,
        history: list[dict],
        language: str,
        session_id: str,
    ) -> AgentResponse:
        # Hybrid RAG retrieve
        relevant_docs = self._hybrid_retrieve(query)
        faq_context = self._format_faq_context(relevant_docs)
        history_context = self._build_history_context(history)

        system_prompt = (FAQ_SYSTEM_VI if language == "vi" else FAQ_SYSTEM_EN).format(
            faq_context=faq_context,
            history=history_context or "Chưa có / None",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,        # thấp — FAQ cần chính xác, không sáng tạo
            max_tokens=300,
        )

        answer = response.choices[0].message.content.strip()

        return AgentResponse(
            text=answer,
            agent_type=self.agent_type,
            session_id=session_id,
            latency_ms=0.0,
            metadata={
                "retrieved_faq_ids": [d["id"] for d in relevant_docs],
                "rag_pipeline": "dual_domain_graph_rerank",
            },
            language=language,
        )