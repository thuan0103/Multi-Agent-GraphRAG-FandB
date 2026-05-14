import os
import re
import json
import yaml
import httpx
import hashlib
import asyncio
import logging
import unicodedata
from groq import Groq
from typing import Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

INTENT_TO_LABEL = {"order": 0, "consultant": 1, "faq": 2, "ignore": 3}

GENERATION_PROMPTS = {
    "order": {
        "vi": """Sinh {n} câu khách hàng nói tại quán cà phê với ý định ĐẶT HÀNG hoặc TÍNH TIỀN.
Ví dụ: "Cho tôi 1 ly cà phê sữa đá", "Tính tiền đi", "Gọi thêm bánh mì".
Đa dạng cách nói, độ dài, phương ngữ. Chỉ trả JSON array, không giải thích.
Format: ["câu 1", "câu 2", ...]""",

        "en": """Generate {n} customer utterances at a coffee shop with ORDER or CHECKOUT intent.
Examples: "I'd like a black coffee", "Can I get the bill?", "Add a croissant please".
Vary phrasing, length, formality. Return JSON array only, no explanation.
Format: ["utterance 1", "utterance 2", ...]""",

        "ko": """카페에서 주문 또는 계산 의도를 가진 고객 발화를 {n}개 생성하세요.
예시: "아메리카노 한 잔 주세요", "계산해주세요", "크루아상 하나 추가해주세요".
표현, 길이, 말투를 다양하게 하세요. 설명 없이 JSON 배열만 반환하세요.
형식: ["문장 1", "문장 2", ...]"""
    },

    "consultant": {
        "vi": """Sinh {n} câu khách hàng hỏi GỢI Ý hoặc TƯ VẤN món tại quán cà phê.
Ví dụ: "Có gì ngon không?", "Gợi ý cho tôi món ít ngọt", "Hôm nay uống gì mát?".
Đa dạng cách hỏi. Chỉ trả JSON array, không giải thích.
Format: ["câu 1", "câu 2", ...]""",

        "en": """Generate {n} customer utterances asking for RECOMMENDATIONS at a coffee shop.
Examples: "What do you recommend?", "Something not too sweet?", "What's popular today?".
Vary phrasing. Return JSON array only, no explanation.
Format: ["utterance 1", "utterance 2", ...]""",

        "ko": """카페에서 메뉴 추천이나 상담을 요청하는 고객 발화를 {n}개 생성하세요.
예시: "추천 메뉴 뭐예요?", "덜 단 음료 있을까요?", "오늘 인기 메뉴 뭐예요?".
다양한 표현으로 작성하세요. 설명 없이 JSON 배열만 반환하세요.
형식: ["문장 1", "문장 2", ...]"""
    },

    "faq": {
        "vi": """Sinh {n} câu khách hàng hỏi THÔNG TIN CHUNG về quán cà phê (không liên quan đặt món).
Ví dụ: "Wifi mật khẩu gì?", "Mấy giờ đóng cửa?", "Có chỗ để xe không?".
Đa dạng chủ đề. Chỉ trả JSON array, không giải thích.
Format: ["câu 1", "câu 2", ...]""",

        "en": """Generate {n} customer utterances asking GENERAL INFO about the coffee shop (not ordering).
Examples: "What's the wifi password?", "What time do you close?", "Is there parking?".
Vary topics. Return JSON array only, no explanation.
Format: ["utterance 1", "utterance 2", ...]""",

        "ko": """카페의 일반 정보(주문과 무관)를 묻는 고객 발화를 {n}개 생성하세요.
예시: "와이파이 비밀번호 뭐예요?", "몇 시에 닫아요?", "주차장 있나요?".
다양한 주제로 작성하세요. 설명 없이 JSON 배열만 반환하세요.
형식: ["문장 1", "문장 2", ...]"""
    },

    "ignore": {
        "vi": """Sinh {n} câu KHÔNG CÓ Ý NGHĨA rõ ràng: tiếng ồn, chào hỏi đơn giản, độc thoại.
Ví dụ: "Ừm...", "Hello", "haha", "oke bạn ơi", "...", "À quên".
Đa dạng. Chỉ trả JSON array, không giải thích.
Format: ["câu 1", "câu 2", ...]""",

        "en": """Generate {n} utterances that are NOISE or have no clear intent: greetings, gibberish, filler.
Examples: "Um...", "Hello", "haha", "ok", "...", "oh wait".
Vary. Return JSON array only, no explanation.
Format: ["utterance 1", "utterance 2", ...]""",

        "ko": """명확한 의도가 없는 발화(잡음, 인사, 혼잣말 등)를 {n}개 생성하세요.
예시: "음...", "안녕하세요", "ㅋㅋ", "오케이", "...", "아 맞다".
다양하게 작성하세요. 설명 없이 JSON 배열만 반환하세요.
형식: ["문장 1", "문장 2", ...]"""
    },
}

HARD_SAMPLE_PROMPTS = {
    "vi": """Sinh {n} câu AMBIGUOUS tại quán cà phê.

PRIMARY INTENT (ground truth): {intent}

Yêu cầu:
- Mỗi câu PHẢI chủ yếu thuộc intent: {intent}
- Nhưng phải mơ hồ với 1 intent khác (order/consultant/faq/ignore)
- Câu phải khiến người đọc phân vân giữa 2 intents

Mỗi câu phải có:
- text
- label (PHẢI là {intent})
- reason (giải thích vì sao mơ hồ)

QUY TẮC:
- label CHỈ được là một trong: "order", "consultant", "faq", "ignore"
- KHÔNG tạo label mới
- label LUÔN = {intent}

Chỉ trả JSON array.

Format:
[{{"text": "câu", "label": "{intent}", "reason": "lý do"}}]
""",

    "en": """Generate {n} AMBIGUOUS utterances at a coffee shop.

PRIMARY INTENT (ground truth): {intent}

Requirements:
- Each utterance MUST primarily belong to: {intent}
- But be ambiguous with another intent (order/consultant/faq/ignore)
- The sentence should create confusion between intents

Each item MUST include:
- text
- label (MUST be {intent})
- reason (why it is ambiguous)

STRICT RULES:
- Label MUST be one of: "order", "consultant", "faq", "ignore"
- DO NOT create new labels
- Label MUST ALWAYS be {intent}

Return ONLY JSON array.

Format:
[{{"text": "utterance", "label": "{intent}", "reason": "short reason"}}]
""",

    "ko": """카페에서 {n}개의 모호한(AMBIGUOUS) 발화를 생성하세요.

PRIMARY INTENT (정답): {intent}

요구사항:
- 각 문장은 반드시 {intent} 의도에 주로 해당해야 합니다
- 그러나 다른 의도(order/consultant/faq/ignore)와 혼동 가능해야 합니다
- 문장은 의도 간 혼동을 유발해야 합니다

각 문장은 다음을 포함해야 합니다:
- text
- label (반드시 {intent})
- reason (왜 모호한지 설명)

규칙:
- label은 반드시 다음 중 하나여야 합니다: "order", "consultant", "faq", "ignore"
- 새로운 label 생성 금지
- label은 항상 {intent}

JSON 배열만 반환하세요.

Format:
[{{"text": "문장", "label": "{intent}", "reason": "이유"}}]
"""
}
    
class DataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["pipeline"]
        self.semaphore = asyncio.Semaphore(self.cfg["max_concurrent"])
        self.seen_hashes: set[str] = set()

        self.client = AsyncOpenAI(api_key=os.getenv('API_OPENAI'))
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.output_dir = Path(self.cfg["output_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Gọi LLM với semaphore + exponential backoff retry."""
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt-5-nano",  
                        messages=[{"role": "user", "content": prompt}],
                        # temperature=0.9,      
                        # max_tokens=2000,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retry in {delay}s")
                    await asyncio.sleep(delay)

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
        
    def _hash(self, text: str) -> str:
        normalized = self._normalize(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _dedup(self, texts: list[str]) -> list[str]:
        """Lọc duplicate dựa trên MD5 hash của text đã normalize."""
        unique = []
        for t in texts:
            h = self._hash(t)
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                unique.append(t)
        return unique
    
    def _parse_json_list(self, raw: str) -> list[str]:
        """Parse JSON array từ LLM output, tolerant với markdown fences."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    
    async def _generate_batch(
        self,
        intent: str,
        language: str,
        n: int,
        is_noise: bool = False,
    ) -> list[dict[str, Any]]:
        """Sinh 1 batch samples cho 1 intent + ngôn ngữ."""
        if is_noise:
            prompt = HARD_SAMPLE_PROMPTS[language].format(n=n, intent=intent)
        else:
            prompt = GENERATION_PROMPTS[intent][language].format(n=n)

        raw = await self._call_llm(prompt)

        try:
            parsed = self._parse_json_list(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Parse error for {intent}/{language}: {e}\nRaw: {raw[:200]}")
            return []
        
        samples = []
        if is_noise:
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "").strip()
                label_name = item.get("label", intent)
                if not text:
                    continue
                
                h = self._hash(text)
                if h in self.seen_hashes:
                    continue

                self.seen_hashes.add(h)

                samples.append({
                    "text": text,
                    "label": INTENT_TO_LABEL.get(label_name, INTENT_TO_LABEL[intent]),
                    "intent": label_name,
                    "is_noise": True,
                    "language": language,
                    "reason": item.get("reason", ""),
                })
        else:
            texts = self._dedup([t.strip() for t in parsed if isinstance(t, str) and t.strip()])
            for text in texts:
                samples.append({
                    "text": text,
                    "label": INTENT_TO_LABEL[intent],
                    "intent": intent,
                    "is_noise": False,
                    "language": language,
                })

        return samples
    
    def _checkpoint_path(self, intent: str) -> Path:
        return self.checkpoint_dir / f"{intent}.json"
    
    def _load_checkpoint(self, intent: str) -> list[dict]:
        path = self._checkpoint_path(intent)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for sample in data:
                self.seen_hashes.add(self._hash(sample["text"]))
            logger.info(f"Resumed {intent}: {len(data)} samples loaded")
            return data
        return []

    def _save_checkpoint(self, intent: str, samples: list[dict]) -> None:
        path = self._checkpoint_path(intent)
        path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')

    async def generate_intent(self, intent: str) -> list[dict]:
        target = self.cfg["samples_per_intent"]
        hard_target = int(target * self.cfg["hard_sample_ratio"])
        normal_target = target - hard_target

        samples = self._load_checkpoint(intent)
        current_normal = sum(1 for s in samples if not s["is_noise"])
        current_hard = sum(1 for s in samples if s["is_noise"])

        logger.info(f"[{intent}] Normal: {current_normal}/{normal_target}, Hard: {current_hard}/{hard_target}")

        batch_size = self.cfg["batch_size"]
        languages = self.cfg["languages"]

        while current_normal < normal_target:
            remaining = normal_target - current_normal
            n_per_lang = min(batch_size, remaining) // len(languages) + 1

            tasks = [
                self._generate_batch(intent, lang, n_per_lang)
                for lang in languages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch failed: {result}")
                    continue
                samples.extend(result)
                current_normal += len([s for s in result if not s["is_noise"]])

            self._save_checkpoint(intent, samples)
            logger.info(f"[{intent}] Normal progress: {current_normal}/{normal_target}")

        while current_hard < hard_target:
            remaining = hard_target - current_hard
            n_per_lang = min(batch_size // 2, remaining) // len(languages) + 1

            tasks = [
                self._generate_batch(intent, lang, n_per_lang, is_noise=True)
                for lang in languages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                samples.extend(result)
                current_hard += len([s for s in result if s["is_noise"]])

            self._save_checkpoint(intent, samples)
            logger.info(f"[{intent}] Hard progress: {current_hard}/{hard_target}")

        return samples

    async def generate_all(self) -> list[dict]:
        """Chạy song song tất cả 4 intents."""
        intents = [i["name"] for i in self.cfg["intents"]]

        tasks = [self.generate_intent(intent) for intent in intents]
        results = await asyncio.gather(*tasks)

        all_samples = []
        for samples in results:
            all_samples.extend(samples)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"dataset_{timestamp}.json"
        output_path.write_text(
            json.dumps(all_samples, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"Saved {len(all_samples)} samples to {output_path}")

        return all_samples
