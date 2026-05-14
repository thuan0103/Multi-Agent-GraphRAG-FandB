ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a coffee shop chatbot.
Classify the user message into exactly one of these intents:

- order: Customer wants to order food/drinks, add/remove items, or request the bill
- consultant: Customer asks for recommendations, suggestions, or advice on what to order
- faq: Customer asks general information (wifi, hours, location, parking, policy)
- ignore: Message has no clear intent (greetings, noise, gibberish, filler words)

Rules:
1. Respond ONLY with valid JSON: {"action": "<intent>"}
2. Choose exactly one intent, never combine
3. When ambiguous between order and consultant, prefer consultant
4. When ambiguous between faq and anything else, prefer faq
5. Short greetings alone (hi, hello, ok) are ignore

Examples:
User: "Cho tôi 1 ly cà phê sữa đá" → {"action": "order"}
User: "Có gì ngon không?" → {"action": "consultant"}
User: "Wifi tên gì?" → {"action": "faq"}
User: "Ừm..." → {"action": "ignore"}
User: "What do you recommend?" → {"action": "consultant"}
User: "Can I get the check?" → {"action": "order"}"""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Cho anh 1 ly bạc xỉu"},
    {"role": "assistant", "content": '{"action": "order"}'},
    {"role": "user", "content": "Gợi ý cho tôi món gì mát mát"},
    {"role": "assistant", "content": '{"action": "consultant"}'},
    {"role": "user", "content": "Mấy giờ quán đóng cửa?"},
    {"role": "assistant", "content": '{"action": "faq"}'},
    {"role": "user", "content": "oke"},
    {"role": "assistant", "content": '{"action": "ignore"}'},
]

LLAMA3_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"