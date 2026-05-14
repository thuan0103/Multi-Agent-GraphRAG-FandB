"""
C2.1: Đánh giá SLM Intent Extraction trên test set.
Metrics:
  - Subject accuracy
  - Action accuracy  ← quan trọng nhất (dùng làm cache key)
  - Context accuracy
  - Combined (cả 3 đúng)
Target: combined accuracy ≥ 90%
"""

import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là AI phân tích câu hỏi của khách hàng F&B. \
Hãy tách câu thành 3 thành phần và trả về JSON hợp lệ.
Format bắt buộc:
{"subject": "<chủ ngữ>", "action": "<hành động/yêu cầu>", "context": "<ngữ cảnh>"}
Nếu không có context, để chuỗi rỗng "".
Chỉ trả về JSON, không giải thích."""


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def infer(model, tokenizer, user_input: str, max_new_tokens: int = 128) -> Optional[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    try:
        # Tìm JSON trong output
        m = re.search(r"\{.*\}", generated, re.DOTALL)
        if m:
            return json.loads(m.group())
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Chuẩn hóa để so sánh: lowercase, strip, bỏ dấu câu thừa."""
    return re.sub(r"\s+", " ", s.strip().lower())


def field_match(pred: str, gt: str, threshold: float = 0.85) -> bool:
    """
    So sánh flexible:
    - Exact match sau normalize
    - Hoặc gt là substring của pred (hoặc ngược lại) — xử lý trường hợp
      model thêm từ đệm nhưng ý nghĩa đúng
    """
    p = normalize(pred)
    g = normalize(gt)
    if p == g:
        return True
    # Substring check
    if g in p or p in g:
        return True
    # Jaccard token overlap
    p_tokens = set(p.split())
    g_tokens = set(g.split())
    if not g_tokens:
        return not p_tokens
    overlap = len(p_tokens & g_tokens) / len(p_tokens | g_tokens)
    return overlap >= threshold


def evaluate(model_path: str, test_file: str, verbose: bool = False) -> dict:
    print(f"[eval] Model: {model_path}")
    print(f"[eval] Test file: {test_file}")

    model, tokenizer = load_model(model_path)

    samples = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"[eval] Số samples: {len(samples)}")

    sub_correct = 0
    act_correct = 0
    ctx_correct = 0
    combined_correct = 0
    parse_fail = 0
    latencies = []

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        pred = infer(model, tokenizer, sample["input"])
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        if pred is None:
            parse_fail += 1
            if verbose:
                print(f"[{i}] PARSE FAIL: {sample['input']}")
            continue

        s_ok = field_match(pred.get("subject", ""), sample["subject"])
        a_ok = field_match(pred.get("action", ""), sample["action"])
        # Context: nếu GT rỗng, pred cũng nên rỗng
        gt_ctx = sample.get("context", "")
        pred_ctx = pred.get("context", "")
        if gt_ctx == "":
            c_ok = normalize(pred_ctx) in ("", "không có", "none", "n/a")
        else:
            c_ok = field_match(pred_ctx, gt_ctx)

        if s_ok:
            sub_correct += 1
        if a_ok:
            act_correct += 1
        if c_ok:
            ctx_correct += 1
        if s_ok and a_ok and c_ok:
            combined_correct += 1

        if verbose and not (s_ok and a_ok and c_ok):
            print(f"\n[{i}] INPUT   : {sample['input']}")
            print(f"     GT     : sub={sample['subject']} | act={sample['action']} | ctx={gt_ctx}")
            print(f"     PRED   : sub={pred.get('subject')} | act={pred.get('action')} | ctx={pred_ctx}")
            print(f"     Match  : sub={s_ok} | act={a_ok} | ctx={c_ok}")

    n = len(samples)
    import statistics
    results = {
        "model_path": model_path,
        "n_samples": n,
        "parse_fail": parse_fail,
        "subject_accuracy": sub_correct / n,
        "action_accuracy": act_correct / n,
        "context_accuracy": ctx_correct / n,
        "combined_accuracy": combined_correct / n,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "meets_target": (combined_correct / n) >= 0.90,
    }

    # Print report
    print("\n" + "=" * 60)
    print("C2.1 INTENT EXTRACTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"Samples        : {n}")
    print(f"Parse failures : {parse_fail} ({parse_fail/n*100:.1f}%)")
    print(f"Subject acc    : {results['subject_accuracy']:.3f}")
    print(f"Action acc     : {results['action_accuracy']:.3f}  ← cache key")
    print(f"Context acc    : {results['context_accuracy']:.3f}")
    print(f"Combined acc   : {results['combined_accuracy']:.3f}  (target ≥ 0.90)")
    print(f"Avg latency    : {results['avg_latency_ms']:.1f}ms")
    print(f"P95 latency    : {results['p95_latency_ms']:.1f}ms")
    print(f"Meets target   : {'✅ YES' if results['meets_target'] else '❌ NO'}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="training/checkpoints/intent_merged")
    parser.add_argument("--test-file", default="data/intent/intent_test.jsonl")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out-json", default="reports/intent_eval.json")
    args = parser.parse_args()

    results = evaluate(args.model_path, args.test_file, args.verbose)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[eval] Lưu: {args.out_json}")


if __name__ == "__main__":
    main()