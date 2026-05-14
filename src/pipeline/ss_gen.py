import os
import json
import yaml
import httpx
import hashlib
import asyncio
import logging
import re
from typing import Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

INTENT_TO_LABEL = {"order": 0, "consultant": 1, "faq": 2, "ignore": 3}

GENERATION_PROMPTS = {
    "order": {
        "vi": """Sinh {n} câu khách hàng nói tại quán cà phê với ý định ĐẶT HÀNG hoặc TÍNH TIỀN.
Chỉ trả JSON array.
Format: ["câu 1", "câu 2"]""",
        "en": """Generate {n} customer utterances with ORDER intent.
Return JSON array only.
Format: ["text1", "text2"]""",
        "ko": """주문 또는 계산 의도를 가진 문장 {n}개 생성.
JSON 배열만 반환.
형식: ["문장1", "문장2"]"""
    },
    "consultant": {
        "vi": """Sinh {n} câu hỏi tư vấn món.
Chỉ trả JSON array.
Format: ["câu 1", "câu 2"]""",
        "en": """Generate {n} recommendation queries.
Return JSON array only.
Format: ["text1", "text2"]""",
        "ko": """추천 요청 문장 {n}개 생성.
JSON 배열만 반환."""
    },
    "faq": {
        "vi": """Sinh {n} câu hỏi thông tin chung.
JSON array only.""",
        "en": """Generate {n} FAQ queries.
JSON array only.""",
        "ko": """일반 정보 질문 {n}개 생성.
JSON 배열만 반환."""
    },
    "ignore": {
        "vi": """Sinh {n} câu vô nghĩa.
JSON array only.""",
        "en": """Generate {n} noise utterances.
JSON array only.""",
        "ko": """의미 없는 발화 {n}개 생성.
JSON 배열만 반환."""
    },
}

HARD_SAMPLE_PROMPTS = {
    "vi": """Sinh {n} câu ambiguous + label + reason.
JSON array only.
Format: [{"text": "...", "label": "...", "reason": "..."}]""",
    "en": """Generate {n} ambiguous utterances + label + reason.
JSON array only.""",
    "ko": """모호한 문장 {n}개 + 라벨 + 이유 생성.
JSON 배열만 반환."""
}


class DataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["pipeline"]
        self.semaphore = asyncio.Semaphore(self.cfg["max_concurrent"])
        self.seen_hashes: set[str] = set()

        self.client = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )

        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.output_dir = Path(self.cfg["output_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def close(self):
        await self.client.aclose()

    async def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    res = await self.client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {os.getenv('API_GROQ_CLOUD')}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "openai/gpt-oss-120b",
                            "messages": [{"role": "user", "content": prompt}]
                        }
                    )
                    return res.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def _dedup(self, texts: list[str]) -> list[str]:
        unique = []
        for t in texts:
            h = self._hash(t)
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                unique.append(t)
        return unique

    def _parse_json_list(self, raw: str):
        raw = raw.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError("Invalid JSON")

    async def _generate_batch(
        self,
        intent: str,
        language: str,
        n: int,
        is_hard: bool = False,
    ) -> list[dict[str, Any]]:

        prompt = (
            HARD_SAMPLE_PROMPTS[language].format(n=n)
            if is_hard
            else GENERATION_PROMPTS[intent][language].format(n=n)
        )

        raw = await self._call_llm(prompt)

        try:
            parsed = self._parse_json_list(raw)
        except Exception:
            return []

        samples = []

        if is_hard:
            for item in parsed:
                if not isinstance(item, dict):
                    continue

                text = item.get("text", "").strip()
                label_name = item.get("label", intent)

                if not text or label_name not in INTENT_TO_LABEL:
                    continue

                h = self._hash(text)
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                samples.append({
                    "text": text,
                    "label": INTENT_TO_LABEL[label_name],
                    "intent": label_name,
                    "is_hard": True,
                    "language": language,
                    "reason": item.get("reason", "")
                })
        else:
            texts = self._dedup([
                t.strip() for t in parsed
                if isinstance(t, str) and t.strip()
            ])

            for text in texts:
                samples.append({
                    "text": text,
                    "label": INTENT_TO_LABEL[intent],
                    "intent": intent,
                    "is_hard": False,
                    "language": language,
                })

        return samples

    def _checkpoint_path(self, intent: str) -> Path:
        return self.checkpoint_dir / f"{intent}.json"

    def _load_checkpoint(self, intent: str) -> list[dict]:
        path = self._checkpoint_path(intent)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []

            for sample in data:
                self.seen_hashes.add(self._hash(sample["text"]))

            return data
        return []

    def _save_checkpoint(self, intent: str, samples: list[dict]) -> None:
        path = self._checkpoint_path(intent)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    async def generate_intent(self, intent: str) -> list[dict]:
        target = self.cfg["samples_per_intent"]
        hard_target = int(target * self.cfg["hard_sample_ratio"])
        normal_target = target - hard_target

        samples = self._load_checkpoint(intent)

        batch_size = self.cfg["batch_size"]
        languages = self.cfg["languages"]

        while sum(1 for s in samples if not s["is_hard"]) < normal_target:
            remaining = normal_target - sum(1 for s in samples if not s["is_hard"])
            n_per_lang = min(batch_size, remaining) // len(languages) + 1

            tasks = [
                self._generate_batch(intent, lang, n_per_lang)
                for lang in languages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                samples.extend(result)

            samples = samples[:target]
            self._save_checkpoint(intent, samples)

        while sum(1 for s in samples if s["is_hard"]) < hard_target:
            remaining = hard_target - sum(1 for s in samples if s["is_hard"])
            n_per_lang = min(batch_size // 2, remaining) // len(languages) + 1

            tasks = [
                self._generate_batch(intent, lang, n_per_lang, is_hard=True)
                for lang in languages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                samples.extend(result)

            samples = samples[:target]
            self._save_checkpoint(intent, samples)

        return samples

    async def generate_all(self) -> list[dict]:
        intents = [i["name"] for i in self.cfg["intents"]]

        results = await asyncio.gather(
            *[self.generate_intent(intent) for intent in intents]
        )

        all_samples = []
        for r in results:
            all_samples.extend(r)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"dataset_{timestamp}.json"
        path.write_text(json.dumps(all_samples, ensure_ascii=False, indent=2), encoding="utf-8")

        return all_samples


async def main():
    d = DataGenerator()
    await d.generate_all()
    await d.close()


if __name__ == "__main__":
    asyncio.run(main())