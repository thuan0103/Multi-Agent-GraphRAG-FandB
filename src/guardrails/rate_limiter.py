"""
C3: Rate Limiting cho FastAPI endpoints.
Dùng sliding window counter trong memory (hoặc Redis nếu có).
Config trong config.yaml.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default limits nếu không đọc từ config.yaml
DEFAULT_LIMITS: Dict[str, Tuple[int, int]] = {
    # endpoint_prefix → (max_requests, window_seconds)
    "/api/chat":      (30, 60),     # 30 req/min per IP
    "/api/order":     (20, 60),
    "/api/consultant":(20, 60),
    "/api/faq":       (40, 60),
    "/health":        (120, 60),    # health check thoải mái hơn
    "default":        (60, 60),
}


# ---------------------------------------------------------------------------
# Sliding Window Rate Limiter
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """
    In-memory sliding window rate limiter.
    Key = (client_ip, endpoint_prefix).
    Thread-safe với asyncio.Lock.
    """

    def __init__(self, limits: Optional[Dict[str, Tuple[int, int]]] = None):
        self._limits = limits or DEFAULT_LIMITS
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    def _get_limit(self, endpoint: str) -> Tuple[int, int]:
        for prefix, limit in self._limits.items():
            if prefix != "default" and endpoint.startswith(prefix):
                return limit
        return self._limits.get("default", (60, 60))

    async def check(self, client_ip: str, endpoint: str) -> bool:
        """
        True = allowed, False = rate limited.
        """
        max_req, window_sec = self._get_limit(endpoint)
        key = f"{client_ip}:{endpoint}"
        now = time.time()
        cutoff = now - window_sec

        async with self._lock:
            window = self._windows[key]
            # Xóa timestamps cũ
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) >= max_req:
                return False

            window.append(now)
            return True

    async def get_remaining(self, client_ip: str, endpoint: str) -> dict:
        max_req, window_sec = self._get_limit(endpoint)
        key = f"{client_ip}:{endpoint}"
        now = time.time()
        cutoff = now - window_sec

        async with self._lock:
            window = self._windows[key]
            while window and window[0] < cutoff:
                window.popleft()
            used = len(window)

        oldest = window[0] if window else now
        reset_at = oldest + window_sec

        return {
            "limit": max_req,
            "remaining": max(0, max_req - used),
            "window_seconds": window_sec,
            "reset_at": reset_at,
        }


# ---------------------------------------------------------------------------
# FastAPI dependency / middleware
# ---------------------------------------------------------------------------

_limiter = SlidingWindowRateLimiter()


def get_client_ip(request: Request) -> str:
    """Lấy IP thực từ request (qua proxy nếu có)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_dependency(request: Request):
    """
    FastAPI Depends() — dùng trong router:
    @app.post("/api/chat", dependencies=[Depends(rate_limit_dependency)])
    """
    client_ip = get_client_ip(request)
    endpoint = request.url.path

    allowed = await _limiter.check(client_ip, endpoint)
    if not allowed:
        info = await _limiter.get_remaining(client_ip, endpoint)
        logger.warning(f"[RateLimit] BLOCKED {client_ip} → {endpoint}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Quá nhiều yêu cầu. Vui lòng thử lại sau.",
                "retry_after_seconds": int(info["reset_at"] - time.time()) + 1,
            },
            headers={"Retry-After": str(int(info["reset_at"] - time.time()) + 1)},
        )

    # Thêm rate limit headers vào response
    info = await _limiter.get_remaining(client_ip, endpoint)
    request.state.rate_limit_info = info