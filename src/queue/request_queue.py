import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

@dataclass
class QueuedRequest:
    request_id: str
    func: Callable
    args: tuple
    kwargs: dict
    enqueued_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    def wait_time(self) -> float:
        return time.time() - self.enqueued_at


class RequestQueue:
    def __init__(
        self,
        max_concurrent: int = 3,
        request_timeout: float = 60.0,
    ):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_timeout = request_timeout
        self.max_concurrent = max_concurrent
        self._active_count = 0
        self._total_processed = 0
        self._total_timeouts = 0
        self._total_errors = 0

    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        request_id = str(uuid.uuid4())[:8]
        enqueued_at = time.time()
        acquired = False

        logger.debug(f"Request {request_id} queued (active={self._active_count}/{self.max_concurrent})")

        try:
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=self.request_timeout,
                )
                acquired = True
            except asyncio.TimeoutError:
                self._total_timeouts += 1
                wait_time = time.time() - enqueued_at
                logger.warning(f"Request {request_id} timed out after {wait_time:.1f}s in queue")
                raise TimeoutError(
                    f"Request waited {wait_time:.1f}s, exceeded timeout of {self.request_timeout}s. "
                    f"System is overloaded, please try again."
                )

            self._active_count += 1
            wait_time = time.time() - enqueued_at
            logger.debug(f"Request {request_id} started (waited {wait_time:.1f}s)")

            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.request_timeout,
                )
                self._total_processed += 1
                return result

            except asyncio.TimeoutError:
                self._total_timeouts += 1
                logger.warning(f"Request {request_id} execution timed out")
                raise TimeoutError(f"Request execution exceeded {self.request_timeout}s timeout")

            except Exception as e:
                self._total_errors += 1
                logger.error(f"Request {request_id} failed: {e}")
                raise

        finally:
            if acquired:
                self.semaphore.release()
                self._active_count = max(0, self._active_count - 1)

    async def stats(self) -> dict:
        return {
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "total_processed": self._total_processed,
            "total_timeouts": self._total_timeouts,
            "total_errors": self._total_errors,
            "available_slots": self.max_concurrent - self._active_count,
        }