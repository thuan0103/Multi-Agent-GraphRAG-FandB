import os
import re
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx

from .base import BaseAgent, AgentResponse, _make_openai_client
from .prompts.order_prompt import ORDER_SYSTEM_VI, ORDER_SYSTEM_EN

logger = logging.getLogger(__name__)

load_dotenv()

GRAPH_RAG_URL = os.getenv("GRAPH_RAG_URL", "http://localhost:8004")
MENU_TTL = 300 

_FULLTEXT_STOPWORDS = frozenset({
    'tôi', 'muốn', 'cho', 'một', 'cái', 'ly', 'cốc', 'ơi', 'bạn',
    'em', 'ạ', 'à', 'nhé', 'đi', 'thôi', 'mình', 'có', 'không', 'gì',
    'là', 'và', 'với', 'thêm', 'bớt', 'xin', 'hỏi', 'về', 'thì', 'được',
    'của', 'hay', 'hoặc', 'nào', 'đó', 'này', 'kia', 'đây', 'lấy', 'đặt',
    'order', 'gọi', 'đơn', 'giúp', 'i', 'me', 'want', 'please', 'a', 'an', 'the',
})


def _clean_for_fulltext(text: str) -> str:
    """Strip Lucene special chars and common stop words for Neo4j fulltext query."""
    text = re.sub(r'[+\-&|!(){}[\]^"~*?:\\/]', ' ', text)
    words = [w for w in text.split() if w.lower() not in _FULLTEXT_STOPWORDS and len(w) > 1]
    return ' '.join(words)


class OrderAgent(BaseAgent):
    agent_type = "order"

    def __init__(self, config: dict, menu_path: str = "data/menu.json"):
        super().__init__(config)
        self.client, self.llm_model = _make_openai_client()
        self._menu_path = menu_path
        self._menu_items: list[dict] = self._bootstrap_from_json()
        self._menu_loaded_at: float = 0

    def _bootstrap_from_json(self) -> list[dict]:
        path = Path(self._menu_path)
        if not path.exists():
            return []
        try:
            menu = json.loads(path.read_text(encoding="utf-8"))
            items = []
            for cat, lst in menu.items():
                if isinstance(lst, list):
                    for item in lst:
                        items.append({
                            "id":          item.get("id", ""),
                            "name":        item.get("name", ""),
                            "price":       item.get("price", 0),
                            "size":        item.get("size", ""),
                            "category":    cat,
                            "ingredients": item.get("ingredients", ""),
                            "description": item.get("description", ""),
                        })
            return items
        except Exception as e:
            logger.warning(f"[OrderAgent] bootstrap JSON failed: {e}")
            return []
        
    async def _fetch_menu_from_graph_rag(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{GRAPH_RAG_URL}/menu/all")
            resp.raise_for_status()
            data = resp.json()
        items = data.get("items", [])
        for item in items:
            item["price"] = int(item.get("price") or 0)
        return items

    async def _ensure_fresh_menu(self):
        now = time.monotonic()
        if now - self._menu_loaded_at < MENU_TTL:
            return
        try:
            items = await self._fetch_menu_from_graph_rag()
            if items:
                self._menu_items = items
                self._menu_loaded_at = now
                logger.info(f"[OrderAgent] Refreshed {len(items)} menu items from Neo4j")
        except Exception as e:
            logger.warning(f"[OrderAgent] Cannot refresh menu: {e} — using cached")

    async def _search_menu_for_context(self, query: str) -> tuple[list[dict], str]:
        clean_q = _clean_for_fulltext(query)
        if clean_q:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(
                        f"{GRAPH_RAG_URL}/menu/fulltext",
                        params={"q": clean_q, "limit": 8},
                    )
                    resp.raise_for_status()
                items = resp.json().get("items", [])
                if items:
                    for it in items:
                        it["price"] = int(it.get("price") or 0)
                    return items, "neo4j_fulltext"
            except Exception as e:
                logger.warning(f"[OrderAgent] fulltext search failed: {e}")

        await self._ensure_fresh_menu()
        return self._menu_items, "cached_full"

    def _format_items(self, items: list[dict]) -> str:
        if not items:
            return "Chưa có dữ liệu menu."
        by_cat: dict[str, list] = {}
        for item in items:
            cat = item.get("category", "other")
            by_cat.setdefault(cat, []).append(item)

        lines = []
        for cat in sorted(by_cat):
            lines.append(f"=== {cat.upper()} ===")
            for item in by_cat[cat]:
                size = f" (size {item['size']})" if item.get("size") else ""
                price = item.get("price", 0)
                ingr = item.get("ingredients", "") or item.get("description", "")
                ingr_str = f" — {ingr}" if ingr else ""
                lines.append(f"- {item['name']}{size}: {price:,}đ{ingr_str}")
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
        menu_items, rag_source = await self._search_menu_for_context(query)

        cart = self._extract_cart_from_history(history)
        history_context = self._build_history_context(history)

        system_prompt = (ORDER_SYSTEM_VI if language == "vi" else ORDER_SYSTEM_EN).format(
            menu=self._format_items(menu_items),
            history=history_context or "Chưa có / None",
            cart=self._format_cart(cart),
        )

        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        updated_cart = self._parse_cart_update(raw, cart)
        answer = self._strip_cart_json(raw)

        return AgentResponse(
            text=answer,
            agent_type=self.agent_type,
            session_id=session_id,
            latency_ms=0.0,
            metadata={
                "cart":          updated_cart,
                "menu_items":    len(menu_items),
                "rag_source":    rag_source,
            },
            language=language,
        )

    def _extract_cart_from_history(self, history: list[dict]) -> list[dict]:
        for turn in reversed(history):
            if turn.get("role") == "assistant" and "cart" in turn.get("metadata", {}):
                return turn["metadata"]["cart"]
        return []

    def _strip_cart_json(self, text: str) -> str:
        return re.sub(r'\s*\[CART_JSON\].*?\[/CART_JSON\]', '', text, flags=re.DOTALL).strip()

    def _parse_cart_update(self, response_text: str, current_cart: list[dict]) -> list[dict]:
        m = re.search(r'\[CART_JSON\](.*?)\[/CART_JSON\]', response_text, re.DOTALL)
        if not m:
            return current_cart
        try:
            data = json.loads(m.group(1).strip())
            items = data.get("items", [])
            return items if isinstance(items, list) else current_cart
        except Exception:
            return current_cart
