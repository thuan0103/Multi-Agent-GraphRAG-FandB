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

        self.base_model_id = "models/Qwen2.5-1.5B"
        self.adapter_path = str(Path("models/checkpoint-400").resolve().as_posix())
        self.max_new_tokens = self.cfg["max_new_tokens"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self._initialized = True

    def load(self) -> None:
        logger.info(f"Loading router: {self.base_model_id} + LoRA {self.adapter_path} on {self.device}")
        start = time.perf_counter()

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
            local_files_only=True,
        )

        # Dùng tokenizer từ checkpoint để giữ đúng chat_template lúc train
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            trust_remote_code=True,
        )

        self.model.eval()
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Model loaded in {elapsed:.0f}ms")
        self._warmup()

    def _warmup(self) -> None:
        logger.info("Warming up router model...")
        for text in ["hello", "cho tôi cà phê", "wifi gì?"]:
            self._raw_generate(text)
        logger.info("Warmup complete")

    def _raw_generate(self, text: str) -> str:
        # import sys
        # sys.path.append(".")
        from src.router.prompts import ROUTER_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

        messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": text})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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
    logging.basicConfig(level=logging.INFO)
    r = RouterModel()
    r.load()
    tests = [
        "Cho tôi 1 ly cà phê sữa đá",
        "Wifi tên gì?",
        "Có gì ngon không?",
        "Hello",
        "Tính tiền đi anh",
        "Gợi ý món mùa hè đi",
    ]
    print("\n" + "="*50)
    for t in tests:
        out, ms = r.generate(t)
        print(f"[{ms:.0f}ms] {t!r:45s} → {out}")