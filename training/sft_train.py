import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INTENT_NAMES = ["order", "consultant", "faq", "ignore"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_sample(sample: dict, tokenizer) -> str:
    """
    Format sample thành conversation string cho SFT.
    Router chỉ cần predict JSON output ngắn.
    """
    from src.router.prompts import ROUTER_SYSTEM_PROMPT

    intent = INTENT_NAMES[sample["label"]]
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": sample["text"]},
        {"role": "assistant", "content": f'{{"action": "{intent}"}}'},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_dataset_split(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def train(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    router_cfg = cfg["router"]

    model_id = train_cfg["base_model"]
    output_dir = train_cfg["output_dir"]

    logger.info(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_samples = load_dataset_split("data/processed/train.json")
    val_samples = load_dataset_split("data/processed/val.json")

    train_texts = [format_sample(s, tokenizer) for s in train_samples]
    val_texts = [format_sample(s, tokenizer) for s in val_samples]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        report_to="none",          
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # dataset_text_field="text",
        # max_seq_length=512,   
        # packing=True,               
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_path = Path(output_dir) / "merged"
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"Merged model saved to {merged_path}")