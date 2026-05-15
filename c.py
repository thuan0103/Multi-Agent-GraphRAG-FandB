import logging
import time
from pathlib import Path
from typing import Optional
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

        # Dùng merged model (LoRA đã bake in) — nhanh hơn PeftModel
        self.model_id = "models/router-merged"
        self.max_new_tokens = self.cfg["max_new_tokens"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self._initialized = True

    def load(self) -> None:
        logger.info(f"Loading router: {self.model_id} on {self.device}")
        start = time.perf_counter()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
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
        """
        KV-cache log-prob scoring:
        - Prompt chạy 1 lần → cache KV
        - 4 intents dùng lại KV → tổng ~5 forward pass thay vì 4×full
        - Bỏ few-shot (model đã fine-tune, không cần)
        """
        import sys
        sys.path.append(".")
        from src.router.prompts import ROUTER_SYSTEM_PROMPT
        INTENTS = ["order", "consultant", "faq", "ignore"]

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        best_intent = None
        best_score  = float("-inf")

        with torch.no_grad():
            # 1 forward pass cho toàn bộ prompt → cache KV
            prompt_out  = self.model(input_ids=prompt_ids, use_cache=True)
            past_kv     = prompt_out.past_key_values
            prev_logits = prompt_out.logits[:, -1:, :]   # logits tại vị trí cuối prompt

            for intent in INTENTS:
                response     = f'{{"action": "{intent}"}}'
                response_ids = self.tokenizer.encode(
                    response, add_special_tokens=False, return_tensors="pt"
                ).to(self.device)
                n_tokens     = response_ids.shape[1]

                score        = 0.0
                cur_past     = past_kv
                cur_logits   = prev_logits

                for i, token_id in enumerate(response_ids[0]):
                    lp     = torch.nn.functional.log_softmax(cur_logits[:, -1, :], dim=-1)
                    score += lp[0, token_id].item()

                    if i < n_tokens - 1:   # không cần forward pass sau token cuối
                        out        = self.model(
                            input_ids=response_ids[:, i : i + 1],
                            past_key_values=cur_past,
                            use_cache=True,
                        )
                        cur_logits = out.logits
                        cur_past   = out.past_key_values

                if score / n_tokens > best_score:
                    best_score  = score / n_tokens
                    best_intent = intent

        return f'{{"action": "{best_intent}"}}'

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