import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base import BaseAgent, AgentResponse
from .prompts.order_prompt import ORDER_SYSTEM_VI, ORDER_SYSTEM_EN

logger = logging.getLogger(__name__)

load_dotenv()

class OrderAgent(BaseAgent):
    agent_type = "order"
    def __init__(self, config: dict, menu_path: str = "data/menu.json"):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=os.getenv("API_OPENAI"))
        self.menu = self._load_menu(menu_path)
        self.menu_text = self._format_menu()

    def _load_menu(self, path: str) -> dict:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))

        return {
            "drinks": [
                {"name": "Cà phê sữa đá", "price": 35000, "name_en": "Iced Milk Coffee"},
                {"name": "Bạc xỉu", "price": 35000, "name_en": "Bac Xiu"},
                {"name": "Cà phê đen đá", "price": 25000, "name_en": "Iced Black Coffee"},
                {"name": "Trà sữa", "price": 45000, "name_en": "Milk Tea"},
                {"name": "Sinh tố bơ", "price": 55000, "name_en": "Avocado Smoothie"},
                {"name": "Nước ép cam", "price": 40000, "name_en": "Orange Juice"},
            ],
            "food": [
                {"name": "Bánh mì", "price": 25000, "name_en": "Banh Mi"},
                {"name": "Bánh croissant", "price": 35000, "name_en": "Croissant"},
                {"name": "Bánh tiramisu", "price": 45000, "name_en": "Tiramisu"},
            ]
        }

    def _format_menu(self) -> str:
        lines = ["=== DRINKS ==="]
        for item in self.menu.get("drinks", []):
            lines.append(f"- {item['name']} ({item['name_en']}): {item['price']:,}đ")
        lines.append("=== FOOD ===")
        for item in self.menu.get("food", []):
            lines.append(f"- {item['name']} ({item['name_en']}): {item['price']:,}đ")
        return "\n".join(lines)

    def _format_cart(self, cart: list[dict]) -> str:
        if not cart:
            return "Chưa có món nào / No items yet"
        lines = []
        total = 0
        for item in cart:
            subtotal = item["price"] * item["quantity"]
            total += subtotal
            lines.append(f"- {item['name']} x{item['quantity']}: {subtotal:,}đ")
        lines.append(f"TOTAL: {total:,}đ")
        return "\n".join(lines)

    async def _process(
        self,
        query: str,
        history: list[dict],
        language: str,
        session_id: str,
    ) -> AgentResponse:
        cart = self._extract_cart_from_history(history)
        history_context = self._build_history_context(history)

        system_prompt = (ORDER_SYSTEM_VI if language == "vi" else ORDER_SYSTEM_EN).format(
            menu=self.menu_text,
            history=history_context or "Chưa có / None",
            cart=self._format_cart(cart),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,    
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()
        updated_cart = self._parse_cart_update(answer, cart)

        return AgentResponse(
            text=answer,
            agent_type=self.agent_type,
            session_id=session_id,
            latency_ms=0.0,
            metadata={
                "cart": updated_cart,
                "menu_items": len(self.menu.get("drinks", [])) + len(self.menu.get("food", [])),
            },
            language=language,
        )

    def _extract_cart_from_history(self, history: list[dict]) -> list[dict]:
        for turn in reversed(history):
            if turn.get("role") == "assistant" and "cart" in turn.get("metadata", {}):
                return turn["metadata"]["cart"]
        return []

    def _parse_cart_update(self, response_text: str, current_cart: list[dict]) -> list[dict]:
        return current_cart