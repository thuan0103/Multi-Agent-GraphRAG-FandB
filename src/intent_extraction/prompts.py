"""
C2.1: System prompt và few-shot examples cho Intent Extraction SLM.
Hỗ trợ 3 ngôn ngữ: Vietnamese (vi), English (en), Korean (ko).
"""

SYSTEM_PROMPT = """Bạn là AI phân tích câu hỏi của khách hàng quán F&B. Hỗ trợ tiếng Việt, English và 한국어.
Tách câu đầu vào thành 3 thành phần và trả về JSON hợp lệ. Trả lời bằng cùng ngôn ngữ với câu đầu vào.

Format bắt buộc:
{"subject": "<chủ ngữ>", "action": "<hành động/yêu cầu>", "context": "<ngữ cảnh>"}

Quy tắc:
- subject: Ai đang nói/yêu cầu — trích nguyên từ câu gốc.
- action: Động từ chính + nội dung cốt lõi — KHÔNG chứa thời gian hay số lượng người.
- context: Thời gian, số lượng người, dịp đặc biệt... (để "" nếu không có).
- Chỉ trả về JSON, không giải thích thêm."""

FEW_SHOT_EXAMPLES = [
    # ── Vietnamese ──────────────────────────────────────────────────────────
    {
        "input": "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé",
        "output": {"subject": "anh", "action": "đặt bàn tiệc sinh nhật", "context": "ngày mai lúc 7h"},
    },
    {
        "input": "Em muốn order 2 ly cà phê sữa đá",
        "output": {"subject": "em", "action": "order cà phê sữa đá", "context": ""},
    },
    {
        "input": "Anh muốn đặt bàn cho 10 người tối nay lúc 7 giờ dịp kỷ niệm công ty",
        "output": {"subject": "anh", "action": "đặt bàn", "context": "10 người, tối nay lúc 7 giờ, dịp kỷ niệm công ty"},
    },
    {
        "input": "Chị hỏi quán có wifi không",
        "output": {"subject": "chị", "action": "hỏi wifi", "context": ""},
    },
    {
        "input": "Hủy order vừa đặt nhé em",
        "output": {"subject": "khách", "action": "hủy order", "context": ""},
    },
    # ── English ─────────────────────────────────────────────────────────────
    {
        "input": "I'd like to order an iced latte for tomorrow morning at 7am",
        "output": {"subject": "I", "action": "order iced latte", "context": "tomorrow morning at 7am"},
    },
    {
        "input": "Can I book a table for 4 people this weekend",
        "output": {"subject": "I", "action": "book a table", "context": "4 people, this weekend"},
    },
    {
        "input": "Do you have WiFi",
        "output": {"subject": "customer", "action": "ask WiFi", "context": ""},
    },
    {
        "input": "Can I cancel my order please",
        "output": {"subject": "I", "action": "cancel order", "context": ""},
    },
    # ── Korean ──────────────────────────────────────────────────────────────
    {
        "input": "아이스 아메리카노 주문하고 싶어요 내일 아침에",
        "output": {"subject": "저", "action": "아이스 아메리카노 주문", "context": "내일 아침에"},
    },
    {
        "input": "테이블 예약하고 싶어요 4명이서 이번 주말에",
        "output": {"subject": "저", "action": "테이블 예약", "context": "4명이서, 이번 주말에"},
    },
    {
        "input": "와이파이 있나요",
        "output": {"subject": "손님", "action": "와이파이 문의", "context": ""},
    },
    {
        "input": "주문 취소할게요",
        "output": {"subject": "저", "action": "주문 취소", "context": ""},
    },
]


def build_few_shot_prompt(user_input: str) -> list:
    """
    Trả về messages list (ChatML format) với few-shot examples đa ngôn ngữ.
    Dùng cho inference khi không có fine-tuned model.
    """
    import json
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["output"], ensure_ascii=False),
        })

    messages.append({"role": "user", "content": user_input})
    return messages
