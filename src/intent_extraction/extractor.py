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


def _detect_lang(text: str) -> str:
    """Nhận diện ngôn ngữ nhanh bằng Unicode range."""
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(
        r"[àáảãạăắặẳẵâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
        text, re.IGNORECASE,
    ):
        return "vi"
    return "en"


def _rule_based_fallback(text: str) -> dict:
    """
    Rule-based fallback khi model fail — hỗ trợ vi / en / ko.
    """
    text = text.strip()
    lang = _detect_lang(text)

    if lang == "ko":
        return _fallback_ko(text)
    if lang == "en":
        return _fallback_en(text)
    return _fallback_vi(text)


def _fallback_vi(text: str) -> dict:
    subject = "khách"
    for pat in [
        r"^(anh|em|chị|tôi|mình|bạn)\b",
        r"^cho\s+(anh|em|chị|tôi|mình)\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            subject = m.group(1)
            break

    ctx_patterns = [
        r"(vào\s+(?:ngày mai|hôm nay|tối nay|sáng mai|chiều nay)[^,\.]*)",
        r"(lúc\s+\d+[gh]\d*(?:\s+\w+)?)",
        r"(cho\s+\d+\s+người)",
        r"(dịp\s+\w+)",
        r"(ngày\s+\d+[^,\.]*)",
    ]
    contexts = []
    clean = text
    for pat in ctx_patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            contexts.append(m.group(1).strip())
            clean = clean.replace(m.group(1), "")

    action = re.sub(r"^(cho\s+)?(anh|em|chị|tôi|mình|bạn|khách)\s*", "", clean, flags=re.IGNORECASE).strip()
    action = re.sub(r"\s+(nhé|em|nhé em|ạ|à)\.?\s*$", "", action, flags=re.IGNORECASE).strip()
    return {"subject": subject, "action": action or text, "context": ", ".join(contexts)}


def _fallback_en(text: str) -> dict:
    subject = "customer"
    for pat, sub in [
        (r"\b(I|we|my\s+\w+)\b", None),
        (r"^can\s+i\b", "I"),
        (r"^i(?:'d|'m)?\b", "I"),
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            subject = sub or m.group(1)
            break

    ctx_patterns = [
        r"(for\s+\d+\s+people)",
        r"(at\s+\d+\s*(?:am|pm))",
        r"(this\s+(?:weekend|morning|afternoon|evening))",
        r"(tomorrow\s+\w+)",
        r"(on\s+\w+\s+\d+(?:st|nd|rd|th)?)",
        r"(for\s+(?:a|an|our)\s+\w+(?:\s+\w+)?)",
        r"(right\s+now|today|tonight)",
    ]
    contexts = []
    clean = text
    for pat in ctx_patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            contexts.append(m.group(1).strip())
            clean = clean.replace(m.group(1), "")

    action = re.sub(r"^(can\s+i|i(?:'d|'m)?|could\s+i|please)\s*", "", clean, flags=re.IGNORECASE).strip()
    action = re.sub(r"\s+please\.?\s*$", "", action, flags=re.IGNORECASE).strip()
    return {"subject": subject, "action": action or text, "context": ", ".join(contexts)}


def _fallback_ko(text: str) -> dict:
    subject = "손님"
    for pat, sub in [
        (r"^(저희|저|제\s+\w+)\b", None),
    ]:
        m = re.search(pat, text)
        if m:
            subject = sub or m.group(1)
            break

    ctx_patterns = [
        r"(\d+명이서)",
        r"(내일\s+\w+에?)",
        r"(이번\s+\w+에?)",
        r"(오늘\s+\w+에?)",
        r"(\d+월\s+\d+일에?)",
        r"(지금\s+바로|오늘|점심에|저녁에|아침에)",
    ]
    contexts = []
    clean = text
    for pat in ctx_patterns:
        m = re.search(pat, clean)
        if m:
            contexts.append(m.group(1).strip())
            clean = clean.replace(m.group(1), "")

    action = re.sub(r"^(저희|저|제\s+\w+)\s*", "", clean).strip()
    # Bỏ đuôi lịch sự
    action = re.sub(r"\s*(?:주세요|부탁드려요|드려요|할게요|싶어요|되나요|있나요|인가요)\.?\s*$", "", action).strip()
    return {"subject": subject, "action": action or text, "context": ", ".join(contexts)}


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