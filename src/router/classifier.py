import json
import logging
import re
from typing import Optional
import sys
sys.path.append('.')
from model import RouterModel

logger = logging.getLogger(__name__)

VALID_INTENTS = {"order", "consultant", "faq", "ignore"}
FALLBACK_INTENT = "ignore"


class IntentClassifier:
    def __init__(self, config_path: str = "config.yaml"):
        self.model = RouterModel(config_path)
        self.config_path = config_path

    def load(self) -> None:
        self.model.load()

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return self._make_response(FALLBACK_INTENT, 0.0, "", fallback=True)

        raw_output, latency_ms = self.model.generate(text.strip())
        intent, fallback = self._parse_intent(raw_output)

        if fallback:
            logger.warning(f"Parse fallback for text='{text[:50]}' | raw='{raw_output}'")

        self._log_latency(latency_ms)

        return self._make_response(intent, latency_ms, raw_output, fallback)

    def _parse_intent(self, raw: str) -> tuple[str, bool]:
        intent = self._try_parse_json(raw)
        if intent:
            return intent, False

        intent = self._try_extract_json(raw)
        if intent:
            return intent, False

        intent = self._try_keyword_match(raw)
        if intent:
            return intent, True 

        return FALLBACK_INTENT, True

    def _try_parse_json(self, raw: str) -> Optional[str]:
        try:
            data = json.loads(raw.strip())
            intent = data.get("action", "").lower()
            return intent if intent in VALID_INTENTS else None
        except (json.JSONDecodeError, AttributeError):
            return None

    def _try_extract_json(self, raw: str) -> Optional[str]:
        pattern = r'\{"action"\s*:\s*"(\w+)"\}'
        match = re.search(pattern, raw)
        if match:
            intent = match.group(1).lower()
            return intent if intent in VALID_INTENTS else None
        return None

    def _try_keyword_match(self, raw: str) -> Optional[str]:
        raw_lower = raw.lower()
        for intent in ["order", "consultant", "faq", "ignore"]:
            if intent in raw_lower:
                return intent
        return None

    def _make_response(
        self,
        intent: str,
        latency_ms: float,
        raw_output: str,
        fallback: bool,
    ) -> dict:
        return {
            "action": intent,
            "latency_ms": round(latency_ms, 2),
            "raw_output": raw_output,
            "fallback": fallback,
        }

    def _log_latency(self, latency_ms: float) -> None:
        import yaml

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["router"]

        excellent = cfg.get("latency_excellent_ms", 100)
        target = cfg.get("latency_target_ms", 200)
        slow = cfg.get("latency_slow_ms", 1000)

        if latency_ms <= excellent:
            logger.debug(f"Latency: {latency_ms:.1f}ms ✓ Excellent")
        elif latency_ms <= target:
            logger.debug(f"Latency: {latency_ms:.1f}ms ✓ OK")
        else:
            logger.warning(f"Latency: {latency_ms:.1f}ms ✗ Exceeded target")

if __name__ == "__main__":
    intent = IntentClassifier()
    intent.load()
    clas = intent.classify("cho tôi xem menu")
    # print(clas)