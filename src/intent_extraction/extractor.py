"""
C2.1 + C2.2: Parse output của SLM thành 3 thành phần cấu trúc.
Output chuẩn: {"subject": str, "action": str, "context": str}
Phần "action" dùng làm cache key.
Phần "context" gửi kèm sang Agent.
"""

import json
import re
import time
import logging
from typing import Optional
from dataclasses import dataclass

from src.intent_extraction.model import IntentExtractionModel
from src.intent_extraction.prompts import build_few_shot_prompt

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    subject: str
    action: str
    context: str
    raw_input: str
    latency_ms: float
    from_finetuned: bool


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_json_output(text: str) -> Optional[dict]:
    """Tìm và parse JSON từ output của model."""
    text = text.strip()
    # Thử parse trực tiếp
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tìm JSON object trong text
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Fallback: regex từng field
    result = {}
    for field in ("subject", "action", "context"):
        m = re.search(
            rf'"{field}"\s*:\s*"([^"]*)"', text
        )
        if m:
            result[field] = m.group(1)

    if "action" in result:
        result.setdefault("subject", "khách")
        result.setdefault("context", "")
        return result

    return None


def _rule_based_fallback(text: str) -> dict:
    """
    Rule-based fallback khi model fail.
    Cố gắng tách action và context bằng regex.
    """
    text = text.strip()

    # Subject patterns
    subject = "khách"
    for pat, sub in [
        (r"^(anh|em|chị|tôi|mình|bạn)\b", None),
        (r"^cho\s+(anh|em|chị|tôi|mình)\b", None),
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            subject = m.group(1) if m.lastindex else m.group(0)
            break

    # Context patterns (time, number of people, occasion)
    ctx_patterns = [
        r"(vào\s+(?:ngày mai|hôm nay|tối nay|sáng mai|chiều nay)[^,\.]*)",
        r"(lúc\s+\d+[gh]\d*(?:\s+\w+)?)",
        r"(cho\s+\d+\s+người)",
        r"(dịp\s+\w+)",
        r"(ngày\s+\d+[^,\.]*)",
    ]
    contexts = []
    clean_text = text
    for pat in ctx_patterns:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            contexts.append(m.group(1).strip())
            clean_text = clean_text.replace(m.group(1), "")

    context = ", ".join(contexts)

    # Action: phần còn lại sau khi bỏ subject/context/filler words
    action = re.sub(
        r"^(cho\s+)?(anh|em|chị|tôi|mình|bạn|khách)\s*", "", clean_text,
        flags=re.IGNORECASE
    ).strip()
    action = re.sub(r"\s+(nhé|em|nhé em|ạ|à)\.?\s*$", "", action, flags=re.IGNORECASE).strip()

    return {"subject": subject, "action": action or text, "context": context}


# ---------------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------------

class IntentExtractor:
    def __init__(self, model_path: str = "training/checkpoints/intent_merged"):
        self._model = IntentExtractionModel.get_instance(model_path)

    def extract(self, user_input: str) -> IntentResult:
        """
        Tách câu thành 3 thành phần.
        Luôn trả về kết quả (không raise exception).
        """
        t0 = time.perf_counter()
        from_finetuned = self._model.is_loaded()

        try:
            if from_finetuned:
                raw = self._model.infer_raw(user_input)
                parsed = _parse_json_output(raw)
            else:
                # Few-shot mode: gọi local LLM hoặc fallback
                parsed = None

            if parsed is None:
                logger.warning(f"[extractor] Parse fail, dùng fallback: {user_input!r}")
                parsed = _rule_based_fallback(user_input)
                from_finetuned = False

        except Exception as e:
            logger.error(f"[extractor] Error: {e}")
            parsed = _rule_based_fallback(user_input)
            from_finetuned = False

        latency_ms = (time.perf_counter() - t0) * 1000

        return IntentResult(
            subject=parsed.get("subject", "khách").strip(),
            action=parsed.get("action", user_input).strip(),
            context=parsed.get("context", "").strip(),
            raw_input=user_input,
            latency_ms=latency_ms,
            from_finetuned=from_finetuned,
        )