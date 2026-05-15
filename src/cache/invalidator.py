import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CacheInvalidator:
    def __init__(self, semantic_cache=None, session_cache=None):
        self._semantic: Optional[object] = semantic_cache
        self._session: Optional[object] = session_cache
        self._history: list = []    # log các lần invalidate

    def on_menu_changed(self) -> dict:
        return self._invalidate("menu", tags=["menu", "order", "consultant"])

    def on_faq_changed(self) -> dict:
        return self._invalidate("faq", tags=["faq", "general"])

    def on_doc_changed(self) -> dict:
        return self._invalidate("doc", tags=["doc", "general"])

    def on_full_reset(self) -> dict:
        removed = 0
        if self._semantic:
            n = len(self._semantic)
            self._semantic.clear()
            removed += n
            logger.info(f"[Invalidator] Full reset semantic cache ({n} entries)")

        event = {
            "event": "full_reset",
            "removed": removed,
            "timestamp": time.time(),
        }
        self._history.append(event)
        return event

    def _invalidate(self, reason: str, tags: list) -> dict:
        removed_semantic = 0
        if self._semantic:
            for tag in tags:
                removed_semantic += self._semantic.invalidate_by_tag(tag)

        event = {
            "event": f"invalidate_{reason}",
            "tags": tags,
            "removed_semantic": removed_semantic,
            "timestamp": time.time(),
        }
        self._history.append(event)
        logger.info(f"[Invalidator] {reason} changed → removed {removed_semantic} cache entries")
        return event

    def get_history(self) -> list:
        return list(self._history)