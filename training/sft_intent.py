"""
C2.1 SFT — Fine-tune SLM < 3B cho tác vụ Intent Extraction
Tách câu thành {subject, action, context} → JSON output

Model mặc định: Qwen/Qwen2.5-0.5B-Instruct
Target: combined accuracy ≥ 90% (subject + action + context đều đúng)

Usage (Google Colab H100 / cloud GPU):
    python training/sft_intent.py
    python training/sft_intent.py --base_model Qwen/Qwen3-0.6B --epochs 3
    python training/sft_intent.py --base_model models/Qwen2.5-0.5B-Instruct --epochs 5

Sau khi train:
    - LoRA adapter: training/checkpoints/intent_sft/
    - Full merged model: training/checkpoints/intent_merged/  ← src/intent_extraction/model.py load từ đây
    - Metrics: training/checkpoints/intent_sft/test_metrics.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Prompt (phải khớp với src/intent_extraction/model.py để inference đúng)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Bạn là AI phân tích câu hỏi của khách hàng quán F&B. Hỗ trợ tiếng Việt, English và 한국어.\n"
    "Tách câu đầu vào thành 3 thành phần và trả về JSON hợp lệ. Trả lời bằng cùng ngôn ngữ với câu đầu vào.\n\n"
    'Format bắt buộc:\n{"subject": "<chủ ngữ>", "action": "<hành động/yêu cầu>", "context": "<ngữ cảnh>"}\n\n'
    "Quy tắc:\n"
    "- subject: Ai đang nói/yêu cầu — trích nguyên từ câu gốc.\n"
    "- action: Động từ chính + nội dung cốt lõi — KHÔNG chứa thời gian hay số lượng người.\n"
    '- context: Thời gian, số lượng người, dịp đặc biệt... (để "" nếu không có).\n'
    "- Chỉ trả về JSON, không giải thích thêm."
)

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_sample(sample: dict) -> str:
    """Build full ChatML conversation string — loss only on assistant turn."""
    target = json.dumps(
        {
            "subject": sample["subject"],
            "action":  sample["action"],
            "context": sample["context"],
        },
        ensure_ascii=False,
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )


def make_dataset(samples: list) -> Dataset:
    return Dataset.from_dict({"text": [format_sample(s) for s in samples]})


# ─────────────────────────────────────────────────────────────────────────────
# Post-training evaluation (exact match + Jaccard)
# ─────────────────────────────────────────────────────────────────────────────

import re

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _field_match(pred: str, gt: str, jaccard_threshold: float = 0.85) -> bool:
    p, g = _normalize(pred), _normalize(gt)
    if p == g:
        return True
    if g in p or p in g:
        return True
    p_tok = set(p.split())
    g_tok = set(g.split())
    if not g_tok:
        return not p_tok
    return len(p_tok & g_tok) / len(p_tok | g_tok) >= jaccard_threshold


def evaluate_on_testset(model, tokenizer, test_file: str) -> dict:
    """Generate predictions on test set, compute per-field accuracy."""
    samples = load_jsonl(test_file)
    model.eval()
    device = next(model.parameters()).device

    hits = {"subject": 0, "action": 0, "context": 0, "all": 0}
    parse_fail = 0
    latencies = []

    for sample in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": sample["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencies.append((time.perf_counter() - t0) * 1000)

        gen = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

        # Extract JSON from output
        try:
            m = re.search(r"\{.*\}", gen, re.DOTALL)
            pred = json.loads(m.group()) if m else None
        except (json.JSONDecodeError, AttributeError):
            pred = None

        if pred is None:
            parse_fail += 1
            continue

        gt_ctx = sample.get("context", "")
        pred_ctx = pred.get("context", "")
        ctx_empty_ok = gt_ctx == "" and _normalize(pred_ctx) in ("", "không có", "none", "n/a")

        s_ok = _field_match(pred.get("subject", ""), sample["subject"])
        a_ok = _field_match(pred.get("action",  ""), sample["action"])
        c_ok = ctx_empty_ok if gt_ctx == "" else _field_match(pred_ctx, gt_ctx)

        if s_ok: hits["subject"] += 1
        if a_ok: hits["action"]  += 1
        if c_ok: hits["context"] += 1
        if s_ok and a_ok and c_ok: hits["all"] += 1

    n = len(samples)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    results = {
        "n_samples":          n,
        "parse_fail":         parse_fail,
        "subject_accuracy":   hits["subject"] / n,
        "action_accuracy":    hits["action"]  / n,
        "context_accuracy":   hits["context"] / n,
        "combined_accuracy":  hits["all"]     / n,
        "avg_latency_ms":     round(avg_lat, 2),
        "p95_latency_ms":     round(p95_lat, 2),
        "meets_target":       (hits["all"] / n) >= 0.90,
    }

    print("\n" + "=" * 58)
    print("  C2.1 TEST RESULTS")
    print("=" * 58)
    print(f"  Samples      : {n}")
    print(f"  Parse fail   : {parse_fail} ({parse_fail/n*100:.1f}%)")
    print(f"  Subject acc  : {results['subject_accuracy']:.1%}")
    print(f"  Action acc   : {results['action_accuracy']:.1%}  ← cache key")
    print(f"  Context acc  : {results['context_accuracy']:.1%}")
    print(f"  Combined acc : {results['combined_accuracy']:.1%}  (target ≥ 90%)")
    print(f"  Avg latency  : {avg_lat:.1f}ms")
    print(f"  P95 latency  : {p95_lat:.1f}ms")
    status = "✅ ĐẠT" if results["meets_target"] else "❌ CHƯA ĐẠT"
    print(f"  Status       : {status}")
    print("=" * 58)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace):
    # ── Precision detection ───────────────────────────────────────────────────
    has_gpu  = torch.cuda.is_available()
    use_bf16 = has_gpu and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else (torch.float16 if has_gpu else torch.float32)
    print(f"[sft_intent] GPU={has_gpu}, bf16={use_bf16}, dtype={dtype}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"[sft_intent] Loading tokenizer: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[sft_intent] Loading model: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False      # required for gradient checkpointing
    model.enable_input_require_grads()  # required for PEFT + grad checkpointing

    # ── LoRA ──────────────────────────────────────────────────────────────────
    # Cover both attention (q/k/v/o) and MLP (gate/up/down) — better for small models
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_samples = load_jsonl(cfg.train_file)
    val_samples   = load_jsonl(cfg.val_file)
    print(f"[sft_intent] Train={len(train_samples)}, Val={len(val_samples)}")

    train_ds = make_dataset(train_samples)
    val_ds   = make_dataset(val_samples)

    # Response-only collator: compute loss ONLY on <|im_start|>assistant\n ... <|im_end|>
    # Encode as token IDs (more robust than string matching in newer trl versions)
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        bf16=use_bf16,
        fp16=(not use_bf16 and has_gpu),
        eval_strategy="steps",          # không dùng evaluation_strategy (deprecated)
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,       # 0 cho Windows compatibility
        remove_unused_columns=False,    # cần thiết với custom collator
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_len,
        packing=False,                  # tắt packing — collator cần biết ranh giới sample
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[sft_intent] Bắt đầu training...")
    trainer.train()

    # Lưu LoRA adapter (best checkpoint được load lại nhờ load_best_model_at_end=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[sft_intent] LoRA adapter đã lưu: {cfg.output_dir}")

    # ── Merge LoRA → full model ───────────────────────────────────────────────
    print("\n[sft_intent] Merging LoRA vào base weights...")
    merged = model.merge_and_unload()
    Path(cfg.merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(cfg.merged_dir)
    tokenizer.save_pretrained(cfg.merged_dir)
    print(f"[sft_intent] Merged model đã lưu: {cfg.merged_dir}")
    print("  → src/intent_extraction/model.py sẽ load từ thư mục này")

    # ── Test evaluation ───────────────────────────────────────────────────────
    if Path(cfg.test_file).exists():
        print(f"\n[sft_intent] Đánh giá trên test set: {cfg.test_file}")
        metrics = evaluate_on_testset(merged, tokenizer, cfg.test_file)

        out_path = Path(cfg.output_dir) / "test_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[sft_intent] Metrics lưu: {out_path}")
    else:
        print(f"[sft_intent] Test file không tồn tại, bỏ qua: {cfg.test_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2.1 SFT for Intent Extraction")

    # Model paths
    p.add_argument("--base_model",  default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID hoặc local path")
    p.add_argument("--output_dir",  default="training/checkpoints/intent_sft",
                   help="Thư mục lưu LoRA adapter checkpoints")
    p.add_argument("--merged_dir",  default="training/checkpoints/intent_merged",
                   help="Thư mục lưu full merged model (dùng trong production)")

    # Data
    p.add_argument("--train_file",  default="data/intent/intent_train.jsonl")
    p.add_argument("--val_file",    default="data/intent/intent_val.jsonl")
    p.add_argument("--test_file",   default="data/intent/intent_test.jsonl")

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=8,
                   help="Per-device batch size (H100: 8–16, T4: 4)")
    p.add_argument("--grad_accum",   type=int,   default=4,
                   help="Effective batch = batch_size × grad_accum × n_gpu")
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_seq_len",  type=int,   default=256,
                   help="Cắt sequence dài hơn giá trị này")

    # Logging / saving
    p.add_argument("--save_steps",    type=int, default=100)
    p.add_argument("--eval_steps",    type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=20)

    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
