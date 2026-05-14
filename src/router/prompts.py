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

def prompt_for_llama3(question: str, contexts=None):

    if isinstance(contexts, str):
        contexts = [contexts]

    prompt = "<|begin_of_text|>"

    prompt += "<|start_header_id|>system<|end_header_id|>\n"
    prompt += (
        "You are an intent classifier for a coffee shop chatbot.\n"
        "Classify the user message into exactly one of these intents:\n\n"
        "- order: Customer wants to order food/drinks, add/remove items, or request the bill\n"
        "- consultant: Customer asks for recommendations, suggestions, or advice on what to order\n"
        "- faq: Customer asks general information (wifi, hours, location, parking, policy)\n"
        "- ignore: Message has no clear intent (greetings, noise, gibberish, filler words)\n\n"
        "Rules:\n"
        "1. Respond ONLY with valid JSON: {\"action\": \"<intent>\"}\n"
        "2. Choose exactly one intent, never combine\n"
        "3. When ambiguous between order and consultant, prefer consultant\n"
        "4. When ambiguous between faq and anything else, prefer faq\n"
        "5. Short greetings alone (hi, hello, ok) are ignore\n"
    )
    prompt += "<|eot_id|>"

    prompt += "<|start_header_id|>user<|end_header_id|>\n"

    if contexts:
        prompt += "Context:\n"
        for i, c in enumerate(contexts, 1):
            prompt += f"[{i}] {c}\n"
        prompt += "\n"

    prompt += f"User: {question}\n"
    prompt += "Return JSON only.\n"
    prompt += "<|eot_id|>"

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    return prompt