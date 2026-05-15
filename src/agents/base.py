import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


def _make_openai_client() -> tuple[AsyncOpenAI, str]:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if base_url:
        return AsyncOpenAI(api_key="local", base_url=base_url), model
    return AsyncOpenAI(api_key=os.getenv("API_OPENAI")), "gpt-4o-mini"

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    text: str                        # response trả cho user
    agent_type: str                  # "order" | "consultant" | "faq"
    session_id: str
    latency_ms: float
    metadata: dict                   # context thêm: items ordered, sources, v.v.
    language: str = "vi"             # ngôn ngữ detect từ input
    error: Optional[str] = None      # None nếu thành công


class BaseAgent(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def agent_type(self) -> str:
        ...

    @abstractmethod
    async def _process(
        self,
        query: str,
        history: list[dict],
        language: str,
        session_id: str,
    ) -> AgentResponse:
        """
        Core logic của từng agent.
        Subclass implement method này.
        """
        ...

    async def handle(
        self,
        query: str,
        history: list[dict],
        session_id: str,
    ) -> AgentResponse:
        """
        Public interface — gọi từ Router.
        Wrap _process với timing + error handling + language detect.
        """
        start = time.perf_counter()
        language = self._detect_language(query)

        try:
            response = await self._process(query, history, language, session_id)
            response.latency_ms = (time.perf_counter() - start) * 1000
            self.logger.info(
                f"[{self.agent_type}] session={session_id} "
                f"lang={language} latency={response.latency_ms:.0f}ms"
            )
            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self.logger.error(f"[{self.agent_type}] Error: {e}", exc_info=True)
            return AgentResponse(
                text=self._error_message(language),
                agent_type=self.agent_type,
                session_id=session_id,
                latency_ms=latency_ms,
                metadata={},
                language=language,
                error=str(e),
            )

    def _detect_language(self, text: str) -> str:
        vietnamese_chars = set("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
        text_lower = text.lower()
        vi_count = sum(1 for c in text_lower if c in vietnamese_chars)
        return "vi" if vi_count > 0 else "en"

    def _error_message(self, language: str) -> str:
        if language == "vi":
            return "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        return "Sorry, I encountered an issue processing your request. Please try again."

    def _build_history_context(self, history: list[dict]) -> str:
        if not history:
            return ""
        lines = ["[Conversation history]"]
        for turn in history:
            role = "Customer" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)