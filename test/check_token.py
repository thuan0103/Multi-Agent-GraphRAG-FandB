from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model = AutoModelForCausalLM.from_pretrained("models/router-merged", dtype=torch.bfloat16)
# base  = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-1.5B", dtype=torch.bfloat16)

# layer = "model.layers.0.self_attn.q_proj.weight"
# same = torch.allclose(
#     model.state_dict()[layer],
#     base.state_dict()[layer]
# )
# print("Merge thất bại (weights giống base):", same)

# tok = AutoTokenizer.from_pretrained("models/router-merged")

# print("\n=== Chat template ===")
# print(tok.chat_template)

# print("\n=== Tokenization 4 intents ===")
# for intent in ["order", "consultant", "faq", "ignore"]:
#     ids = tok.encode(f'{{"action": "{intent}"}}', add_special_tokens=False)
#     print(f"{intent:12s}: {ids} ({len(ids)} tokens)")

import time

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a coffee shop chatbot.
Classify the user message into exactly one of these intents:

- order: Customer wants to order a SPECIFIC named item, add/remove a specific item, or request the bill
- consultant: Customer asks for recommendations or advice, OR wants "something" without naming a specific item
- faq: Customer asks general information (wifi, hours, location, parking, payment, delivery, membership, discount)
- ignore: Message has no clear intent (greetings, noise, gibberish, filler words)

Examples:
User: "Cho tôi 1 ly cà phê sữa đá" → {"action": "order"}
User: "Thêm 1 bánh croissant" → {"action": "order"}
User: "Hủy đơn hiện tại" → {"action": "order"}
User: "Tính tiền đi" → {"action": "order"}
User: "I'll take 2 cappuccinos" → {"action": "order"}

User: "Có gì ngon không?" → {"action": "consultant"}
User: "What do you recommend?" → {"action": "consultant"}
User: "Cho mình ly gì đó không caffeine" → {"action": "consultant"}
User: "Cho mình cái gì đó mát mẻ" → {"action": "consultant"}
User: "Mình muốn gọi cái gì đó phù hợp thời tiết này" → {"action": "consultant"}
User: "Gợi ý cho mình uống gì mùa hè" → {"action": "consultant"}
User: "Something light and refreshing please" → {"action": "consultant"}
User: "Something without caffeine" → {"action": "consultant"}
User: "I want something cold" → {"action": "consultant"}

User: "Wifi tên gì?" → {"action": "faq"}
User: "Thanh toán bằng gì?" → {"action": "faq"}
User: "Có giao hàng không?" → {"action": "faq"}
User: "Do you accept credit cards?" → {"action": "faq"}
User: "Có giảm giá sinh viên không?" → {"action": "faq"}
User: "Remove the sandwich from my cart" → {"action": "order"}

User: "Ừm..." → {"action": "ignore"}
User: "ok" → {"action": "ignore"}
User: "hello" → {"action": "ignore"}"""

tok = AutoTokenizer.from_pretrained("models/router-merged")
import time
start = time.perf_counter()
ids = tok.encode(ROUTER_SYSTEM_PROMPT, return_tensors="pt")
print(f"Prompt tokens: {ids.shape[1]}")
print(f"Tokenize time: {(time.perf_counter()-start)*1000:.1f}ms")