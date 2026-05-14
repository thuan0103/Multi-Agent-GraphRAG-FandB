# src/session/summarizer.py
"""
Auto-summarizer: tóm tắt phần history cũ khi vượt context window.
"""

import logging
from typing import Optional
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT_VI = """Tóm tắt ngắn gọn cuộc hội thoại sau thành 2-3 câu.
Giữ lại: món đã order, thông tin quan trọng, ngữ cảnh cần thiết.
Bỏ qua: chào hỏi, câu xã giao.

{existing_summary}

Hội thoại cần tóm tắt:
{conversation}

Tóm tắt:"""

SUMMARIZE_PROMPT_EN = """Summarize the following conversation in 2-3 sentences.
Keep: items ordered, important info, necessary context.
Skip: greetings, small talk.

{existing_summary}

Conversation to summarize:
{conversation}

Summary:"""


class ConversationSummarizer:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("API_OPENAI"))

    async def summarize(
        self,
        turns: list[dict],
        existing_summary: Optional[str] = None,
        language: str = "vi",
    ) -> str:
        """
        Tóm tắt list turns thành 1 đoạn ngắn.
        Nếu đã có summary cũ, tích hợp vào.
        """
        conversation = "\n".join(
            f"{'Khách' if t['role'] == 'user' else 'Nhân viên'}: {t['content']}"
            for t in turns
        )

        existing_block = ""
        if existing_summary:
            existing_block = f"Tóm tắt trước đó: {existing_summary}\n\n"

        prompt_template = SUMMARIZE_PROMPT_VI if language == "vi" else SUMMARIZE_PROMPT_EN
        prompt = prompt_template.format(
            existing_summary=existing_block,
            conversation=conversation,
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated summary: {summary[:80]}...")
            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: dùng N turns đầu làm summary thô
            return conversation[:300] + "..."