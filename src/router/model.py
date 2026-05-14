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
        INTENTS = ["order", "consultant", "faq", "ignore"]

        prompt = f"""You are classifying café chatbot messages. Output MUST be one word only.

INTENT DEFINITIONS:
- order: customer places an order or requests bill
- consultant: customer asks for recommendations or what to try
- faq: customer asks about wifi, hours, address, price, parking
- ignore: greeting or meaningless message

KEYWORD TRIGGERS:
- consultant: "nên thử", "có món nào", "gợi ý", "món gì ngon", "nên uống", "nên ăn", "thử món", "recommend", "추천", "뭐가 맛있", "어떤 게 좋", "뭐 시킬"
- order: "cho tôi", "tôi muốn", "đặt", "thêm", "tính tiền", "주세요", "시켜", "주문", "I want", "give me", "can I get"
- faq: "wifi", "giờ", "mấy giờ", "địa chỉ", "giá", "bao nhiêu", "chỗ đậu xe", "영업시간", "와이파이", "주소", "얼마", "주차"
- ignore: "hi", "hello", "ok", "bye", "xin chào", "cảm ơn", "ừm", "안녕", "감사", "응"

PRIORITY RULES (when ambiguous):
1. Any keyword from consultant list → consultant (even if sentence is long)
2. Any keyword from order list → order
3. Any keyword from faq list → faq
4. Nothing matches → ignore

HARD EXAMPLES (tricky cases):
"Có món nào nên thử khi ghé quán cà phê hôm nay không?" → consultant
"Hôm nay có món mới không?" → consultant
"Cho tôi xem menu" → consultant
"Quán có gì đặc biệt không?" → consultant
"추천해줄 만한 메뉴가 뭔가요?" → consultant
"오늘 뭐가 맛있어요?" → consultant
"뭐 시킬까요?" → consultant
"Cho tôi 1 ly cà phê sữa đá" → order
"Thêm 1 cái bánh nữa đi" → order
"Tính tiền đi anh" → order
"아메리카노 한 잔 주세요" → order
"Wifi tên gì?" → faq
"Mấy giờ quán đóng cửa?" → faq
"영업시간이 어떻게 되나요?" → faq
"Xin chào" → ignore
"Ok" → ignore
"안녕하세요" → ignore

User: {text}
Intent:"""

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.shape[1]

        best_intent = None
        best_score = float("-inf")

        with torch.no_grad():
            for intent in INTENTS:
                intent_ids = self.tokenizer.encode(
                    intent, add_special_tokens=False, return_tensors="pt"
                ).to(self.device)

                full_ids = torch.cat([prompt_ids, intent_ids], dim=1)
                outputs = self.model(input_ids=full_ids)
                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

                score = 0.0
                for i, token_id in enumerate(intent_ids[0]):
                    pos = prompt_len - 1 + i
                    score += log_probs[0, pos, token_id].item()

                if score > best_score:
                    best_score = score
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
    r = RouterModel()
    r.load()