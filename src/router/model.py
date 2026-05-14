import logging
import time
from pathlib import Path
from typing import Optional
from peft import PeftModel
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class RouterModel:
    _instance: Optional["RouterModel"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if self._initialized:
            return

        with open('config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg["router"]

        self.model_id = "models/llama3-1b" # "models/Qwen2.5-1.5B-Instruct" # self.cfg["model_id"]
        self.max_new_tokens = self.cfg["max_new_tokens"]
        self.temperature = self.cfg["temperature"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self._initialized = True

    def load(self) -> None:
        # logger.info(f"Loading router model: {self.model_id} on {self.device}")
        # start = time.perf_counter()

        # base_model_id = self.model_id

        # adapter_path = Path("models/router-sft/checkpoint-500").resolve().as_posix()

        # # 1. base model
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     base_model_id,
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #     device_map="auto",
        #     trust_remote_code=True,
        # )

        # # 2. LoRA adapter (FIX WINDOWS PATH ISSUE)
        # self.model = PeftModel.from_pretrained(
        #     base_model,
        #     adapter_path,
        #     local_files_only=True
        # )

        # # 3. tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     base_model_id,
        #     trust_remote_code=True,
        # )

        # self.model.eval()

        # logger.info(f"Loaded in {(time.perf_counter()-start)*1000:.0f}ms")

        # self._warmup()

        # logger.info(f"Loading router model: {self.model_id} on {self.device}")
        # start = time.perf_counter()

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_id,
        #     trust_remote_code=True,
        # )

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id,
        #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        #     device_map="auto",
        #     trust_remote_code=True,
        # )
        # self.model.eval()

        # elapsed = (time.perf_counter() - start) * 1000
        # logger.info(f"Model loaded in {elapsed:.0f}ms")
        # self._warmup()
        logger.info(f"Loading router model: {self.model_id} on {self.device}")
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Model loaded in {elapsed:.0f}ms")
        self._warmup()


    def _warmup(self) -> None:
        logger.info("Warming up router model...")
        warmup_texts = ["hello", "cho tôi cà phê", "wifi gì?"]
        for text in warmup_texts:
            self._raw_generate(text)
        logger.info("Warmup complete")

    def _raw_generate(self, text: str) -> str:
        from src.router.prompts import ROUTER_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, prompt_for_llama3

        # messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        # messages.extend(FEW_SHOT_EXAMPLES)
        # messages.append({"role": "user", "content": text})

        # prompt = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )

        # inputs = self.tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     truncation=True
        # )

        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        query = prompt_for_llama3(question=text)
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        # return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        if "<|start_header_id|>assistant<|end_header_id|>" in text:
            text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

        text = text.split("<|start_header_id|>")[0]
        return text.strip()

    def generate(self, text: str) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.perf_counter()
        output = self._raw_generate(text)
        print(output)
        latency_ms = (time.perf_counter() - start) * 1000

        return output, latency_ms

    def is_loaded(self) -> bool:
        return self.model is not None
    
if __name__ == "__main__":
    r = RouterModel()
    r.load()