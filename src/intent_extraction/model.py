
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class IntentExtractionModel:

    _instance: Optional["IntentExtractionModel"] = None
    _instance_path: Optional[str] = None

    @classmethod
    def get_instance(cls, model_path: str = "training/checkpoints/intent_merged") -> "IntentExtractionModel":
        if cls._instance is None or cls._instance_path != model_path:
            cls._instance = cls(model_path)
            cls._instance_path = model_path
        return cls._instance

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self._try_load()

    def _try_load(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            logger.warning(
                f"[IntentExtractionModel] Model path not found: '{self._model_path}'. "
                "Chạy ở chế độ rule-based fallback."
            )
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"[IntentExtractionModel] Loading from {self._model_path} ...")
            t0 = time.perf_counter()

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            if self._device == "cpu":
                self._model = self._model.to("cpu")

            self._model.eval()
            self._loaded = True
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[IntentExtractionModel] Loaded in {elapsed:.0f}ms on {self._device}")

        except Exception as e:
            logger.error(f"[IntentExtractionModel] Load failed: {e}. Dùng fallback.")
            self._model = None
            self._tokenizer = None
            self._loaded = False

    def infer_raw(self, user_input: str, max_new_tokens: int = 128) -> str:
        """
        Chạy inference, trả về raw string từ model.
        extractor.py sẽ parse JSON từ string này.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Gọi is_loaded() trước.")

        from src.intent_extraction.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_input},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def is_loaded(self) -> bool:
        return self._loaded

    def model_path(self) -> str:
        return self._model_path
