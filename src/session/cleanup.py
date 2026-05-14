# src/session/cleanup.py
"""
Background cleanup task: xóa expired sessions định kỳ.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class SessionCleanup:
    """
    Chạy background task dọn dẹp sessions hết TTL.
    Tự động start/stop cùng app lifecycle.
    """

    def __init__(self, session_store, interval_seconds: int = 120):
        self.store = session_store
        self.interval = interval_seconds
        self._task: asyncio.Task = None
        self._running = False

    async def start(self) -> None:
        """Gọi lúc app startup."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Session cleanup started (interval={self.interval}s)")

    async def stop(self) -> None:
        """Gọi lúc app shutdown."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Session cleanup stopped")

    async def _run(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)

    async def _cleanup(self) -> None:
        """Xóa tất cả sessions đã expired."""
        async with self.store._lock:
            expired_ids = [
                sid for sid, session in self.store._sessions.items()
                if session.is_expired(self.store.ttl_seconds)
            ]
            for sid in expired_ids:
                del self.store._sessions[sid]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions: {expired_ids}")

        stats = await self.store.stats()
        logger.debug(f"Session stats after cleanup: {stats}")