"""
C3: Graceful Degradation — fallback khi Generator quá tải.
Router vẫn classify được intent, dùng template cứng thay vì gọi LLM.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback templates theo intent
# ---------------------------------------------------------------------------

FALLBACK_TEMPLATES = {
    "order": (
        "Dạ quán đang xử lý nhiều đơn hàng, {subject} vui lòng chờ một chút "
        "hoặc liên hệ lại sau ít phút ạ. Quán xin lỗi vì sự bất tiện này!"
    ),
    "consultant": (
        "Dạ {subject} ơi, hệ thống tư vấn đang bận, vui lòng thử lại sau vài giây ạ. "
        "Quán có menu đầy đủ tại quầy để {subject} tham khảo nhé!"
    ),
    "faq": (
        "Dạ {subject} ơi, hệ thống đang tải, câu hỏi của {subject} sẽ được "
        "trả lời ngay khi hệ thống sẵn sàng ạ. Xin lỗi vì sự bất tiện!"
    ),
    "chitchat": (
        "Dạ {subject} ơi, hệ thống đang bận một chút, "
        "{subject} vui lòng thử lại sau nhé ạ!"
    ),
    "unknown": (
        "Dạ hệ thống đang bận, vui lòng thử lại sau ít phút ạ. Xin cảm ơn!"
    ),
}

# Thông báo ngắn hơn cho TTS
FALLBACK_SHORT = {
    "order": "Dạ hệ thống đang bận, quán sẽ xử lý đơn ngay khi có thể ạ.",
    "consultant": "Dạ hệ thống tư vấn đang tải, vui lòng thử lại sau ạ.",
    "faq": "Dạ hệ thống đang bận, xin thử lại sau ít phút ạ.",
    "chitchat": "Dạ hệ thống đang bận, xin lỗi quý khách ạ.",
    "unknown": "Dạ hệ thống đang bận, vui lòng thử lại ạ.",
}


class GracefulDegradation:
    """
    Fallback handler khi LLM Generator quá tải.
    Cung cấp response template ngay lập tức (< 5ms).
    """

    def __init__(self):
        self._degraded = False
        self._degraded_since: Optional[float] = None
        self._degraded_count = 0

    def enter_degraded_mode(self):
        if not self._degraded:
            self._degraded = True
            self._degraded_since = time.time()
            logger.warning("[Fallback] Entering degraded mode — Generator overloaded")

    def exit_degraded_mode(self):
        if self._degraded:
            duration = time.time() - (self._degraded_since or time.time())
            logger.info(f"[Fallback] Exiting degraded mode (was degraded {duration:.0f}s)")
            self._degraded = False
            self._degraded_since = None

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    def get_fallback_response(
        self,
        intent: str = "unknown",
        subject: str = "anh/chị",
        short: bool = False,
    ) -> str:
        """
        Trả về response template ngay lập tức.
        short=True cho TTS (câu ngắn hơn).
        """
        self._degraded_count += 1
        templates = FALLBACK_SHORT if short else FALLBACK_TEMPLATES
        template = templates.get(intent, templates["unknown"])
        response = template.format(subject=subject)
        logger.debug(f"[Fallback] intent={intent} subject={subject} → {response[:50]}...")
        return response

    def stats(self) -> dict:
        return {
            "is_degraded": self._degraded,
            "degraded_since": self._degraded_since,
            "total_fallback_responses": self._degraded_count,
        }


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Circuit Breaker cho LLM Generator calls.
    States: CLOSED (normal) → OPEN (fallback) → HALF_OPEN (testing)

    Chuyển OPEN khi error_rate vượt threshold trong window.
    Thử recover sau recovery_timeout giây.
    """

    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        error_threshold: float = 0.5,    # 50% error rate → OPEN
        window_size: int = 10,            # last N calls
        recovery_timeout: float = 30.0,  # giây trước khi thử lại
    ):
        self._threshold = error_threshold
        self._window = window_size
        self._recovery_timeout = recovery_timeout

        self._state = self.CLOSED
        self._results: list = []          # True=success, False=error
        self._opened_at: Optional[float] = None
        self._lock = __import__("threading").Lock()
        self._fallback = GracefulDegradation()

    @property
    def state(self) -> str:
        return self._state

    @property
    def fallback(self) -> GracefulDegradation:
        return self._fallback

    def record_success(self):
        with self._lock:
            self._results.append(True)
            if len(self._results) > self._window:
                self._results.pop(0)
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                self._fallback.exit_degraded_mode()
                logger.info("[CircuitBreaker] HALF_OPEN → CLOSED (recovered)")

    def record_error(self):
        with self._lock:
            self._results.append(False)
            if len(self._results) > self._window:
                self._results.pop(0)
            self._check_trip()

    def _check_trip(self):
        if self._state == self.OPEN:
            return
        if len(self._results) < 3:
            return
        error_rate = self._results.count(False) / len(self._results)
        if error_rate >= self._threshold:
            self._state = self.OPEN
            self._opened_at = time.time()
            self._fallback.enter_degraded_mode()
            logger.error(
                f"[CircuitBreaker] CLOSED → OPEN (error_rate={error_rate:.0%})"
            )

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN:
                elapsed = time.time() - (self._opened_at or 0)
                if elapsed >= self._recovery_timeout:
                    self._state = self.HALF_OPEN
                    logger.info("[CircuitBreaker] OPEN → HALF_OPEN (testing recovery)")
                    return True
                return False
            # HALF_OPEN: allow one request
            return True

    def stats(self) -> dict:
        with self._lock:
            recent = self._results[-self._window:]
            err_rate = recent.count(False) / len(recent) if recent else 0.0
            return {
                "state": self._state,
                "error_rate_recent": round(err_rate, 3),
                "window_size": self._window,
                **self._fallback.stats(),
            }


# Global instances
_circuit_breaker = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    return _circuit_breaker