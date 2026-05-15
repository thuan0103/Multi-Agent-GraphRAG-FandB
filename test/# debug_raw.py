# # debug_raw.py
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch, sys
# sys.path.append(".")
# from src.router.prompts import ROUTER_SYSTEM_PROMPT

# model = AutoModelForCausalLM.from_pretrained("models/router-merged", dtype=torch.bfloat16, device_map="auto")
# tok = AutoTokenizer.from_pretrained("models/router-merged")
# model.eval()

# tests = [
#     "Có gì ngon không em?",       # expected: consultant
#     "Wifi tên gì?",               # expected: faq
#     "Cho tôi 1 ly cà phê",        # expected: order
#     "ok",                         # expected: ignore
# ]

# INTENTS = ["order", "consultant", "faq", "ignore"]

# for text in tests:
#     messages = [
#         {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
#         {"role": "user",   "content": text},
#     ]
#     prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     ids = tok.encode(prompt, return_tensors="pt").to(model.device)

#     # 1. Greedy generate — xem model tự output gì
#     with torch.no_grad():
#         out = model.generate(ids, max_new_tokens=20, do_sample=False)
#     generated = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
#     print(f"\nInput : {text}")
#     print(f"Output: {generated}")

#     # 2. In score từng intent
#     with torch.no_grad():
#         prompt_out = model(input_ids=ids, use_cache=True)
    
#     scores = {}
#     for intent in INTENTS:
#         response = f'{{"action": "{intent}"}}'
#         r_ids = tok.encode(response, add_special_tokens=False, return_tensors="pt").to(model.device)
#         n = r_ids.shape[1]
#         with torch.no_grad():
#             out2 = model(input_ids=r_ids, past_key_values=prompt_out.past_key_values, use_cache=False)
#         all_logits = torch.cat([prompt_out.logits[:, -1:, :], out2.logits[:, :-1, :]], dim=1)
#         lp = torch.nn.functional.log_softmax(all_logits, dim=-1)
#         score = lp[0, torch.arange(n), r_ids[0]].mean().item()
#         scores[intent] = round(score, 4)
#     print(f"Scores: {scores}")

# Chạy file này riêng để xem structure
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("models/router-merged", dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained("models/router-merged")

ids = tok.encode("test", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)

pkv = out.past_key_values
print("type:", type(pkv))
print("dir:", [x for x in dir(pkv) if not x.startswith("__")])

# Thử iterate
for i, item in enumerate(pkv):
    print(f"layer {i}: type={type(item)}, ", end="")
    if hasattr(item, '__len__'):
        print(f"len={len(item)}")
        for j, x in enumerate(item):
            print(f"  [{j}]: type={type(x)}, shape={x.shape if hasattr(x, 'shape') else 'N/A'}")
    else:
        print(item)
    if i >= 1:
        print("...")
        break