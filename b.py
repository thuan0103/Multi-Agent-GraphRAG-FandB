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

        self.model_id = "models/llama3-1b"
        self.max_new_tokens = self.cfg["max_new_tokens"]
        self.temperature = self.cfg["temperature"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self._initialized = True

    def load(self) -> None:
        logger.info(f"Loading router model: {self.model_id} on {self.device}")
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
           self.model_id,
            torch_dtype=torch.float16, 
            device_map="auto"
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
        from src.router.prompts import LLAMA3_CHAT_TEMPLATE

        self.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

        messages = [
            {"role": "system", "content": """
                Bạn là trợ lý phân tích ý định người dùng.
                Từ câu hỏi người dùng háy phán tích xem nó là order, consultant, faq, ignore.
                Yêu cầu đâu ra bắt buộc phải là json kiểu {"action": intent}
                Ví dụ: 
                - User: "Cho tôi 1 ly cà phê sữa đá" → {"action": "order"}
                - User: "Có gì ngon không?" → {"action": "consultant"}
                - User: "Wifi tên gì?" → {"action": "faq"}
                - User: "Ừm..." → {"action": "ignore"}
            """}
        ]

        messages.append({"role": "user", "content": text})

        enc = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )

        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        else:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return result.strip()

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