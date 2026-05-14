"""
C2.1: System prompt và few-shot examples cho Intent Extraction SLM.
"""

SYSTEM_PROMPT = """Bạn là AI phân tích câu hỏi của khách hàng F&B.
Hãy tách câu thành 3 thành phần và trả về JSON hợp lệ.

Format bắt buộc:
{"subject": "<chủ ngữ>", "action": "<hành động/yêu cầu>", "context": "<ngữ cảnh>"}

Quy tắc:
- subject: Ai đang nói / yêu cầu (anh, em, tôi, khách, ...)
- action: Động từ chính + nội dung cốt lõi — KHÔNG chứa thời gian hay số lượng người
- context: Thời gian, số lượng người, dịp đặc biệt, địa điểm, ... (có thể rỗng "")
- Nếu không có context, để "".
- Chỉ trả về JSON, không giải thích thêm.
"""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé",
        "output": {
            "subject": "anh",
            "action": "đặt bàn tiệc sinh nhật",
            "context": "ngày mai lúc 7h",
        },
    },
    {
        "input": "Cho tôi xem menu",
        "output": {
            "subject": "tôi",
            "action": "xem menu",
            "context": "",
        },
    },
    {
        "input": "Em muốn order 2 ly cà phê sữa đá và 1 bánh croissant",
        "output": {
            "subject": "em",
            "action": "order cà phê sữa đá và bánh croissant",
            "context": "",
        },
    },
    {
        "input": "Chị hỏi quán có wifi không",
        "output": {
            "subject": "chị",
            "action": "hỏi wifi",
            "context": "",
        },
    },
    {
        "input": "Anh muốn đặt bàn cho 10 người tối nay lúc 7 giờ dịp kỷ niệm công ty",
        "output": {
            "subject": "anh",
            "action": "đặt bàn",
            "context": "10 người, tối nay lúc 7 giờ, dịp kỷ niệm công ty",
        },
    },
    {
        "input": "Mình muốn tư vấn món uống phù hợp buổi chiều",
        "output": {
            "subject": "mình",
            "action": "tư vấn món uống phù hợp buổi chiều",
            "context": "",
        },
    },
    {
        "input": "Thanh toán bằng thẻ được không em",
        "output": {
            "subject": "khách",
            "action": "hỏi thanh toán bằng thẻ",
            "context": "",
        },
    },
    {
        "input": "Thêm 1 ly trà sữa trân châu vào đơn giúp anh",
        "output": {
            "subject": "anh",
            "action": "thêm trà sữa trân châu",
            "context": "",
        },
    },
    {
        "input": "Hủy order vừa đặt nhé em",
        "output": {
            "subject": "khách",
            "action": "hủy order",
            "context": "",
        },
    },
    {
        "input": "Quán mở cửa đến mấy giờ vậy em",
        "output": {
            "subject": "khách",
            "action": "hỏi giờ đóng cửa",
            "context": "",
        },
    },
]


def build_few_shot_prompt(user_input: str) -> list:
    """
    Trả về messages list (ChatML format) với few-shot examples.
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