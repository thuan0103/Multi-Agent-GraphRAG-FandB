import random
import re
from typing import Optional


PARAPHRASE_TEMPLATES = {
    "đặt bàn": [
        "Dạ {subject} ơi, {response}. {context_phrase} Quán sẽ sắp xếp ngay ạ!",
        "Vâng ạ, {response} cho {subject}. {context_phrase} Quán xác nhận nhé!",
        "Dạ được ạ! {response}. {context_phrase} Quán sẽ chuẩn bị chu đáo cho {subject} ạ.",
    ],
    "order": [
        "Dạ {subject} ơi, {response}. {context_phrase} Quán nhận đơn ngay ạ!",
        "Vâng, {response} đã được ghi nhận rồi ạ. {context_phrase}",
        "Dạ xong rồi ạ! {response}. {context_phrase} Quán sẽ phục vụ {subject} ngay nhé.",
    ],
    "hủy order": [
        "Dạ {subject} ơi, {response}. Quán đã hủy đơn cho {subject} rồi ạ.",
        "Vâng ạ, {response}. Quán đã xử lý xong, {subject} yên tâm nhé!",
    ],
    "xem menu": [
        "Dạ {subject} ơi, {response}. {context_phrase} Quán có thể tư vấn thêm nếu {subject} cần nhé!",
        "Vâng ạ! {response}. {context_phrase} {subject} cứ thoải mái lựa chọn ạ.",
    ],
    "hỏi": [
        "Dạ {subject} ơi, {response}. {context_phrase} Quán xin được hỗ trợ thêm nếu cần ạ!",
        "Vâng ạ, {response}. {context_phrase}",
        "Dạ, {response}. {context_phrase} {subject} có cần hỗ trợ gì thêm không ạ?",
    ],
    "tư vấn": [
        "Dạ {subject} ơi, {response}. {context_phrase} Quán hy vọng {subject} sẽ thích ạ!",
        "Vâng ạ, {response}. {context_phrase} {subject} cứ hỏi thêm nếu muốn nhé!",
    ],
    "default": [
        "Dạ {subject} ơi, {response}. {context_phrase} Quán xin phục vụ {subject} ạ!",
        "Vâng ạ! {response}. {context_phrase}",
        "Dạ được ạ, {response}. {context_phrase} Quán cảm ơn {subject} đã ghé thăm!",
    ],
}

CONTEXT_FILLERS = [
    "{context}",
    "Thông tin thêm: {context}.",
    "Lưu ý: {context}.",
    "Quán ghi nhận thêm: {context}.",
]


def _get_category(action: str) -> str:
    """Map action string về template category."""
    action_lower = action.lower()
    for key in PARAPHRASE_TEMPLATES:
        if key != "default" and key in action_lower:
            return key
    return "default"


def _build_context_phrase(context: str) -> str:
    """Tạo câu ngữ cảnh tự nhiên từ context string."""
    if not context or not context.strip():
        return ""
    ctx = context.strip()
    filler = random.choice(CONTEXT_FILLERS)
    phrase = filler.format(context=ctx)
    # Đảm bảo có dấu câu cuối
    if not phrase.endswith((".", "!", "?")):
        phrase += "."
    return phrase


def _normalize_subject_display(subject: str) -> str:
    """Chuẩn hóa subject để hiển thị thân thiện."""
    mapping = {
        "tôi": "anh/chị",
        "mình": "anh/chị",
        "bạn": "quý khách",
        "khách": "quý khách",
    }
    return mapping.get(subject.lower(), subject)


# ---------------------------------------------------------------------------
# Main Paraphraser
# ---------------------------------------------------------------------------

class Paraphraser:
    """
    Cache Hit Paraphraser.
    Nhận: cache_response (template từ DB) + intent_result (từ extractor)
    Trả: câu hoàn chỉnh sẵn sàng phát TTS.
    """

    def paraphrase(
        self,
        cache_response: str,
        action: str,
        context: str,
        subject: str,
        seed: Optional[int] = None,
    ) -> str:
        """
        Tạo câu paraphrase tự nhiên.

        Args:
            cache_response: Nội dung response gốc từ cache (template).
            action: Phần hành động từ extractor (dùng để chọn template).
            context: Phần ngữ cảnh từ extractor.
            subject: Chủ ngữ từ extractor.
            seed: Random seed để reproducible (test).
        """
        if seed is not None:
            random.seed(seed)

        category = _get_category(action)
        templates = PARAPHRASE_TEMPLATES[category]
        template = random.choice(templates)

        subject_display = _normalize_subject_display(subject)
        context_phrase = _build_context_phrase(context)

        result = template.format(
            subject=subject_display,
            response=cache_response.strip(),
            context_phrase=context_phrase,
        )

        # Post-process: bỏ khoảng trắng thừa
        result = re.sub(r"\s{2,}", " ", result).strip()
        # Bỏ dấu câu trùng
        result = re.sub(r"([.!?])\s*\1+", r"\1", result)

        return result

    def paraphrase_from_intent(
        self,
        cache_response: str,
        intent_result,  # IntentResult
        seed: Optional[int] = None,
    ) -> str:
        """Shorthand nhận IntentResult trực tiếp."""
        return self.paraphrase(
            cache_response=cache_response,
            action=intent_result.action,
            context=intent_result.context,
            subject=intent_result.subject,
            seed=seed,
        )