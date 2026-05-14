import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTENT_NAMES = ["order", "consultant", "faq", "ignore"]


# ======================
# CONFIG
# ======================
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ======================
# PREPROCESS (FAST VERSION)
# ======================
def build_messages(example, tokenizer):
    from src.router.prompts import ROUTER_SYSTEM_PROMPT

    intent = INTENT_NAMES[example["label"]]

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": example["text"]},
        {"role": "assistant", "content": f'{{"action": "{intent}"}}'},
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


# ======================
# LOAD DATA
# ======================
def load_json(path):
    return json.loads(Path(path).read_text())


# ======================
# TRAIN
# ======================
def train(config_path: str = "config.yaml"):

    # -------- config --------
    cfg = load_config(config_path)
    train_cfg = cfg["training"]

    model_id = train_cfg["base_model"]
    output_dir = train_cfg["output_dir"]

    # -------- speed tweaks --------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True, # 🔥 IMPORTANT
    )

    # optional speed boost
    try:
        model = torch.compile(model)
        logger.info("torch.compile enabled")
    except Exception as e:
        logger.warning(f"torch.compile skipped: {e}")

    # -------- LoRA (faster config) --------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # 🔥 smaller = faster
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------- dataset --------
    train_samples = load_json("data/processed/train.json")
    val_samples = load_json("data/processed/val.json")

    train_ds = Dataset.from_list(train_samples)
    val_ds = Dataset.from_list(val_samples)

    # FAST MAP (parallel preprocessing)
    def preprocess(ex):
        return build_messages(ex, tokenizer)

    logger.info("Tokenizing dataset (fast map)...")

    train_ds = train_ds.map(
        preprocess,
        num_proc=4,
        remove_columns=train_ds.column_names,
    )

    val_ds = val_ds.map(
        preprocess,
        num_proc=4,
        remove_columns=val_ds.column_names,
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # -------- training args --------
    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=train_cfg["num_epochs"],
        learning_rate=train_cfg["learning_rate"],

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=2,

        warmup_steps=100,
        lr_scheduler_type="cosine",

        bf16=True,
        fp16=False,

        logging_steps=20,

        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=500,

        save_total_limit=2,
        load_best_model_at_end=True,

        report_to="none",

        dataloader_num_workers=2,
        pin_memory=True,
    )

    # -------- trainer --------
    trainer = SFTTrainer(
        model=model,
        args=training_args,

        train_dataset=train_ds,
        eval_dataset=val_ds,

        dataset_text_field="text",

        max_seq_length=512,
        packing=True,   # 🔥 HUGE SPEED BOOST
    )

    logger.info("Starting training...")
    trainer.train()

    # -------- merge LoRA --------
    logger.info("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()

    merged_path = Path(output_dir) / "merged"
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    logger.info(f"Saved to {merged_path}")