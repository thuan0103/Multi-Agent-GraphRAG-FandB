# src/session/store.py
"""
SessionStore: quản lý lịch sử hội thoại theo session_id.
In-memory với TTL tự động.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    history: list[dict] = field(default_factory=list)
    summary: Optional[str] = None      # tóm tắt phần history cũ
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # cart, preferences, v.v.

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.last_active) > ttl_seconds


class SessionStore:
    """
    Thread-safe session store với:
    - History window (giới hạn N turns gần nhất)
    - Auto-summarization khi vượt token threshold
    - TTL-based expiry
    """

    def __init__(
        self,
        history_window: int = 5,
        ttl_minutes: int = 30,
        context_window_threshold: float = 0.70,
        max_tokens_estimate: int = 4000,    # ước tính context window của LLM
    ):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.history_window = history_window
        self.ttl_seconds = ttl_minutes * 60
        self.context_window_threshold = context_window_threshold
        self.max_tokens_estimate = max_tokens_estimate
        self._summarizer = None           # inject từ ngoài để tránh circular import

    def set_summarizer(self, summarizer) -> None:
        self._summarizer = summarizer

    # ──────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────

    async def get_or_create(self, session_id: str) -> Session:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
                logger.info(f"New session created: {session_id}")
            session = self._sessions[session_id]
            session.touch()
            return session

    async def add_turn(
        self,
        session_id: str,
        role: str,          # "user" | "assistant"
        content: str,
        metadata: dict = None,
    ) -> None:
        """Thêm 1 turn vào history, auto-summarize nếu cần."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = Session(session_id=session_id)
                self._sessions[session_id] = session

            session.history.append({
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })
            session.touch()

        # Kiểm tra và summarize ngoài lock để không block
        await self._maybe_summarize(session_id)

    async def get_history(self, session_id: str) -> list[dict]:
        """
        Trả history window + summary nếu có.
        Format: [summary_turn (nếu có)] + [N turns gần nhất]
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []

            recent = session.history[-self.history_window:]

            # Prepend summary nếu có phần history bị truncate
            if session.summary and len(session.history) > self.history_window:
                summary_turn = {
                    "role": "system",
                    "content": f"[Conversation summary so far]: {session.summary}",
                    "metadata": {},
                }
                return [summary_turn] + recent

            return list(recent)

    async def update_metadata(self, session_id: str, key: str, value) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.metadata[key] = value
                session.touch()

    async def get_metadata(self, session_id: str) -> dict:
        async with self._lock:
            session = self._sessions.get(session_id)
            return dict(session.metadata) if session else {}

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            logger.info(f"Session deleted: {session_id}")

    # ──────────────────────────────────────
    # Auto-summarization
    # ──────────────────────────────────────

    def _estimate_tokens(self, session: Session) -> int:
        """Ước tính token count đơn giản: ~4 chars/token."""
        total_chars = sum(len(t["content"]) for t in session.history)
        if session.summary:
            total_chars += len(session.summary)
        return total_chars // 4

    async def _maybe_summarize(self, session_id: str) -> None:
        """Summarize nếu token count vượt threshold."""
        if not self._summarizer:
            return

        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            estimated_tokens = self._estimate_tokens(session)
            threshold = self.max_tokens_estimate * self.context_window_threshold

            if estimated_tokens < threshold:
                return

            # Lấy phần cũ để summarize (giữ lại history_window turns mới nhất)
            old_turns = session.history[:-self.history_window]
            if not old_turns:
                return

        # Gọi summarizer ngoài lock
        logger.info(f"Auto-summarizing session {session_id} ({estimated_tokens} tokens)")
        summary = await self._summarizer.summarize(old_turns, existing_summary=session.summary)

        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.summary = summary
                # Giữ lại chỉ history_window turns gần nhất
                session.history = session.history[-self.history_window:]

    # ──────────────────────────────────────
    # Stats
    # ──────────────────────────────────────

    async def stats(self) -> dict:
        async with self._lock:
            now = time.time()
            active = sum(
                1 for s in self._sessions.values()
                if not s.is_expired(self.ttl_seconds)
            )
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": active,
                "expired_sessions": len(self._sessions) - active,
            }