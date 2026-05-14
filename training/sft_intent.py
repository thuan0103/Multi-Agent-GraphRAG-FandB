"""
C2.1: Fine-tune SLM < 3B cho tác vụ Intent Extraction.
Model: Qwen2.5-0.5B-Instruct hoặc Qwen3-0.6B
Target: accuracy ≥ 90% trên 3 thành phần (subject/action/context)

Dùng SFTTrainer (trl) + LoRA (peft) để fine-tune hiệu quả trên H100.
Sau khi train: merge LoRA, save full model → dùng trong src/intent_extraction/model.py
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IntentSFTConfig:
    base_model: str   = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str   = "training/checkpoints/intent_sft"
    merged_dir: str   = "training/checkpoints/intent_merged"
    train_file: str   = "data/intent/intent_train.jsonl"
    val_file: str     = "data/intent/intent_val.jsonl"

    # LoRA
    lora_r: int       = 16
    lora_alpha: int   = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")

    # Training
    num_epochs: int   = 5
    batch_size: int   = 8
    grad_accum: int   = 4
    lr: float         = 2e-4
    max_seq_len: int  = 256
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    save_steps: int   = 100
    eval_steps: int   = 100
    logging_steps: int = 20
    fp16: bool        = True


CFG = IntentSFTConfig()

# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là AI phân tích câu hỏi của khách hàng F&B. \
Hãy tách câu thành 3 thành phần và trả về JSON hợp lệ.
Format bắt buộc:
{"subject": "<chủ ngữ>", "action": "<hành động/yêu cầu>", "context": "<ngữ cảnh>"}
Nếu không có context, để chuỗi rỗng "".
Chỉ trả về JSON, không giải thích."""


def format_sample(sample: dict) -> str:
    """Tạo chuỗi conversation hoàn chỉnh theo ChatML (Qwen format)."""
    output_json = json.dumps({
        "subject": sample["subject"],
        "action": sample["action"],
        "context": sample["context"],
    }, ensure_ascii=False)

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{output_json}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def make_hf_dataset(samples: list) -> Dataset:
    return Dataset.from_dict({"text": [format_sample(s) for s in samples]})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"[sft_intent] Loading base model: {CFG.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        torch_dtype=torch.float16 if CFG.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    lora_config = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        target_modules=list(CFG.target_modules),
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_data = load_jsonl(CFG.train_file)
    val_data = load_jsonl(CFG.val_file)
    print(f"[sft_intent] Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = make_hf_dataset(train_data)
    val_dataset   = make_hf_dataset(val_data)

    # Response-only training (chỉ tính loss trên phần assistant)
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.num_epochs,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.grad_accum,
        learning_rate=CFG.lr,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        fp16=CFG.fp16,
        evaluation_strategy="steps",
        eval_steps=CFG.eval_steps,
        save_strategy="steps",
        save_steps=CFG.save_steps,
        logging_steps=CFG.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=CFG.max_seq_len,
        args=training_args,
    )

    print("[sft_intent] Bắt đầu fine-tune...")
    trainer.train()
    trainer.save_model(CFG.output_dir)
    print(f"[sft_intent] Lưu LoRA checkpoint: {CFG.output_dir}")

    # Merge LoRA → full model
    print("[sft_intent] Merging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(CFG.merged_dir)
    tokenizer.save_pretrained(CFG.merged_dir)
    print(f"[sft_intent] Merged model: {CFG.merged_dir}")


if __name__ == "__main__":
    train()