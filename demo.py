import asyncio
import json
import logging
import uuid
from pathlib import Path

import httpx

import yaml
from dotenv import load_dotenv

import gradio as gr

from src.router import IntentClassifier
from src.agents import OrderAgent, ConsultantAgent, FAQAgent
from src.session import SessionStore, ConversationSummarizer
from src.queue import RequestQueue, retry_with_backoff
from src.cache import SemanticCache, EmbeddingProvider
from src.intent_extraction import IntentExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

router_classifier = IntentClassifier()
summarizer        = ConversationSummarizer()
session_store     = SessionStore(
    history_window=CONFIG["session"]["history_window"],
    ttl_minutes=CONFIG["session"]["ttl_minutes"],
    context_window_threshold=CONFIG["session"]["context_window_threshold"],
)
session_store.set_summarizer(summarizer)
request_queue = RequestQueue(
    max_concurrent=CONFIG["queue"]["max_concurrent_llm"],
    request_timeout=CONFIG["queue"]["request_timeout_seconds"],
)

_emb_provider  = EmbeddingProvider()        
semantic_cache = SemanticCache(embedding_provider=_emb_provider, threshold=0.92)
intent_extractor = IntentExtractor()

agents: dict = {}

_FOOD_CATS = {"food"}

def _sync_data_from_csv() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas không có — bỏ qua CSV sync, dùng JSON cũ")
        return

    menu_csv = Path("data/watch/menu.csv")
    if menu_csv.exists():
        try:
            df = pd.read_csv(menu_csv, skiprows=1)  
            df.columns = df.columns.str.strip()
            first_col = df.columns[0]
            if first_col.isdigit() or first_col == "1":
                df = df.drop(columns=[first_col])

            drinks, food = [], []
            seen = set()
            for _, row in df.iterrows():
                try:
                    cat  = str(row.get("category", "")).strip().lower()
                    name = str(row["name"]).strip()
                    price = int(float(str(row["price"]).replace(",", "")))

                    raw_desc = str(row.get("description", "") or "").strip().strip('"')
                    desc = raw_desc if len(raw_desc) > 8 else ""

                    key = (name.lower(), price)
                    if key in seen:  
                        continue
                    seen.add(key)

                    item = {
                        "name":        name,
                        "name_en":     "",
                        "price":       price,
                        "description": desc,
                        "tags":        [cat],
                    }
                    (food if cat in _FOOD_CATS else drinks).append(item)
                except Exception:
                    continue

            menu_json = {"drinks": drinks, "food": food}
            Path("data/menu.json").write_text(
                json.dumps(menu_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"✓ Sync menu: {len(drinks)} đồ uống, {len(food)} đồ ăn")
        except Exception as e:
            print(f"⚠️  Lỗi đọc menu.csv: {e}")

    faq_csv = next(
        (p for p in Path("data/watch").glob("*.csv")
         if p.stem.lower() == "faq"),
        None,
    )
    if faq_csv:
        try:
            df = pd.read_csv(faq_csv, skiprows=1)
            df.columns = df.columns.str.strip()
            first_col = df.columns[0]
            if str(first_col).isdigit() or str(first_col) == "1":
                df = df.drop(columns=[first_col])

            faq_items = []
            for _, row in df.iterrows():
                try:
                    cat = str(row.get("category", "") or "").strip()
                    faq_items.append({
                        "id":       str(row["id"]),
                        "question": str(row["question"]).strip(),
                        "answer":   str(row["answer"]).strip(),
                        "tags":     [cat] if cat else [],
                    })
                except Exception:
                    continue

            Path("data/faq.json").write_text(
                json.dumps(faq_items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"✓ Sync FAQ: {len(faq_items)} mục")
        except Exception as e:
            print(f"⚠️  Lỗi đọc Faq.csv: {e}")


def startup() -> None:
    banner = """
╔══════════════════════════════════════════════════╗
║   MAS-LLM Coffee Shop — Khởi động hệ thống...   ║
║   MAS-LLM-2026-FINAL | FI-AI                     ║
╚══════════════════════════════════════════════════╝"""
    print(banner)
    print("► Sync dữ liệu từ CSV...")
    _sync_data_from_csv()

    print("► Đang load Router model (Llama3-1B, log-prob scoring)...")
    print("  Có thể mất 1-2 phút lần đầu...\n")

    router_classifier.load()
    agents["order"]      = OrderAgent(CONFIG)
    agents["consultant"] = ConsultantAgent(CONFIG)
    agents["faq"]        = FAQAgent(CONFIG)

    print("\n✓ Router loaded")
    print("✓ Order Agent ready")
    print("✓ Consultant Agent ready")
    print("✓ FAQ Agent ready")
    print("✓ Semantic Cache ready")
    print("✓ Intent Extractor ready")
    print("\n Hệ thống sẵn sàng! Gradio UI đang khởi động...\n")

def _paraphrase_cache_hit(template: str, context: str, language: str) -> str:
    ctx = (context or "").strip()
    if not ctx or ctx.lower() in ("", "none", "không có", "n/a"):
        return template

    if language == "vi":
        prefixes = [
            f"Dạ, về yêu cầu {ctx} — {template}",
            f"Với {ctx}: {template}",
            f"Lưu ý {ctx}, {template[0].lower() + template[1:]}",
        ]
    else:
        prefixes = [
            f"Regarding {ctx} — {template}",
            f"For {ctx}: {template}",
        ]

    import hashlib
    idx = int(hashlib.md5(ctx.encode()).hexdigest(), 16) % len(prefixes)
    return prefixes[idx]


INTENT_LABELS = {
    "order":      "🛒 Order",
    "consultant": "💡 Consultant",
    "faq":        "❓ FAQ",
    "ignore":     "💬 Ignore",
}


@retry_with_backoff(max_attempts=3, base_delay=1.0)
async def _call_agent(agent, query, history, session_id):
    return await agent.handle(query, history, session_id)


def _format_retrieved(intent: str, metadata: dict) -> str:
    """Tạo chuỗi hiển thị retrieved context cho demo."""
    src = metadata.get("rag_source", "fallback")
    tag = "🔵 Neo4j" if src == "neo4j" else "🟡 fallback"

    if intent == "faq":
        ids = metadata.get("retrieved_faq_ids", [])
        nodes = ", ".join(ids) if ids else "—"
        return f"{tag} | FAQ chunks: {nodes}"
    if intent == "consultant":
        items = metadata.get("retrieved_items", [])
        nodes = ", ".join(items) if items else "—"
        return f"{tag} | MenuItem: {nodes}"
    if intent == "order":
        cart = metadata.get("cart", [])
        if cart:
            lines = [f"{i['name']} x{i['quantity']} ({i['price']:,}đ)" for i in cart]
            return "Cart: " + " | ".join(lines)
        return "Cart: (trống)"
    return "—"


async def process_message(message: str, session_id: str) -> dict:
    """
    Pipeline đầy đủ:
    1. Router classify → intent
    2. ignore → trả lời ngay
    3. IntentExtractor → action key
    4. SemanticCache.query(action) → cache hit/miss
    5. Cache miss → gọi agent qua queue → lưu cache
    6. Lưu session
    """
    router_result  = router_classifier.classify(message)
    intent         = router_result["action"]
    router_latency = router_result["latency_ms"]
    fallback       = router_result.get("fallback", False)

    if intent == "ignore":
        return {
            "reply":             "Xin chào! Tôi có thể giúp gì cho bạn? 😊\n"
                                 "Thử hỏi về menu, đặt món, hoặc thông tin quán nhé!",
            "intent":            "ignore",
            "total_latency":     router_latency,
            "router_latency":    router_latency,
            "agent_latency":     0.0,
            "agent_type":        "—",
            "fallback":          fallback,
            "cache_hit":         False,
            "cache_score":       0.0,
            "retrieved_context": "—",
        }

    extracted  = intent_extractor.extract(message)
    cache_key  = extracted.action   
    extr_ms    = extracted.latency_ms

    cache_result = semantic_cache.query(cache_key)
    if cache_result:
        entry, score = cache_result
        language = "vi" if any(c in message for c in "àáâãèéêìíòóôõùúýăđơư") else "en"
        reply = _paraphrase_cache_hit(entry.response_template, extracted.context, language)
        logger.info(f"[Cache] HIT score={score:.3f} key={cache_key!r} ctx={extracted.context!r}")
        return {
            "reply":             reply,
            "intent":            intent,
            "total_latency":     router_latency + extr_ms,
            "router_latency":    router_latency,
            "agent_latency":     0.0,
            "agent_type":        intent,
            "fallback":          fallback,
            "cache_hit":         True,
            "cache_score":       score,
            "retrieved_context": f"key: \"{entry.key_text}\"",
        }

    history = await session_store.get_history(session_id)
    agent   = agents.get(intent)
    if not agent:
        return {
            "reply":             f"⚠️ Lỗi: Không tìm thấy agent cho intent '{intent}'.",
            "intent":            intent,
            "total_latency":     router_latency,
            "router_latency":    router_latency,
            "agent_latency":     0.0,
            "agent_type":        "—",
            "fallback":          fallback,
            "cache_hit":         False,
            "cache_score":       0.0,
            "retrieved_context": "—",
        }

    agent_response = await request_queue.submit(
        _call_agent, agent, message, history, session_id
    )

    semantic_cache.put(
        key_text=cache_key,
        response_template=agent_response.text,
        domain=intent,
        tags=[intent],
    )
    logger.info(f"[Cache] PUT key={cache_key!r} domain={intent}")

    await session_store.add_turn(session_id, "user", message)
    await session_store.add_turn(
        session_id, "assistant", agent_response.text,
        metadata=agent_response.metadata,
    )

    return {
        "reply":             agent_response.text,
        "intent":            intent,
        "total_latency":     router_latency + agent_response.latency_ms,
        "router_latency":    router_latency,
        "agent_latency":     agent_response.latency_ms,
        "agent_type":        intent,
        "fallback":          fallback,
        "cache_hit":         False,
        "cache_score":       0.0,
        "retrieved_context": _format_retrieved(intent, agent_response.metadata),
    }

async def _sglang_respond(message: str, h: list, session_id: str, api_url: str):
    """
    Gọi POST {api_url}/chat/stream, đọc SSE token-by-token.
    Yield từng bước giống respond() để Gradio streaming hoạt động.
    Events: {"type": "meta"|"token"|"done"|"error", ...}
    """
    url = api_url.rstrip("/") + "/chat/stream"
    partial = ""
    intent = "—"
    total_latency = 0.0
    agent_latency = 0.0

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url,
                json={"message": message, "session_id": session_id},
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code != 200:
                    h[-1] = {"role": "assistant", "content": f"⚠️ API lỗi {resp.status_code}"}
                    yield (h, "", *["❌"] * _N_METRICS, session_id, session_id[:8] + "...")
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "meta":
                        intent = event.get("intent", "—")
                        intent_label = INTENT_LABELS.get(intent, intent)
                        yield (h, "", intent_label, *["⏳"] * (_N_METRICS - 1),
                               session_id, session_id[:8] + "...")

                    elif etype == "token":
                        partial += event.get("text", "")
                        h[-1] = {"role": "assistant", "content": partial}
                        yield (h, "", INTENT_LABELS.get(intent, intent),
                               *["⏳"] * (_N_METRICS - 1),
                               session_id, session_id[:8] + "...")

                    elif etype == "done":
                        total_latency = event.get("latency_ms", 0.0)
                        agent_latency = event.get("agent_latency_ms", 0.0)

                    elif etype == "error":
                        err = event.get("detail", "unknown error")
                        h[-1] = {"role": "assistant", "content": f"⚠️ {err}"}
                        yield (h, "", *["❌"] * _N_METRICS, session_id, session_id[:8] + "...")
                        return

    except httpx.ConnectError:
        h[-1] = {"role": "assistant",
                  "content": f"⚠️ Không kết nối được tới {url}\n→ Hãy chạy: `python main.py`"}
        yield (h, "", *["❌"] * _N_METRICS, session_id, session_id[:8] + "...")
        return
    except Exception as e:
        h[-1] = {"role": "assistant", "content": f"⚠️ Lỗi: {e}"}
        yield (h, "", *["❌"] * _N_METRICS, session_id, session_id[:8] + "...")
        return

    h[-1] = {"role": "assistant", "content": partial.strip()}
    intent_label = INTENT_LABELS.get(intent, intent)
    yield (
        h, "",
        intent_label,
        f"🖥️ SGLang",
        f"{total_latency:.0f} ms",
        "—",
        f"{agent_latency:.0f} ms",
        "SGLang API",
        "❌ N/A (via API)",
        "—",
        session_id,
        session_id[:8] + "...",
    )

QUICK_PROMPTS = [
    ("🛒 Đặt cà phê",      "Cho anh 1 ly cà phê sữa đá"),
    ("💡 Tư vấn mùa hè",   "Có gì mát mát uống mùa hè nhỉ?"),
    ("❓ Wifi",             "Wifi quán tên gì? Mật khẩu là gì?"),
    ("❓ Giờ mở cửa",      "Quán mở cửa lúc mấy giờ?"),
    ("🛒 Thêm món",        "Thêm 1 cái bánh croissant nữa nhé"),
    ("💡 Không uống cafe", "Tôi không uống được cà phê, gợi ý món khác đi"),
    ("❓ Địa chỉ",         "Quán ở đâu vậy?"),
    ("🛒 Tính tiền",       "Tính tiền đi em"),
]

_N_METRICS = 8


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="☕ MAS-LLM Coffee Bot") as app:

        gr.Markdown("""
        # ☕ MAS-LLM Coffee Shop — Multi-Agent Demo
        **MAS-LLM-2026-FINAL | FI-AI** · Router (Llama3-1B) + 3 Agents + Semantic Cache + Session
        """)

        session_id_state = gr.State(value=str(uuid.uuid4()))

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Hội thoại đa lượt (Multi-turn)",
                    height=480,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Nhập tiếng Việt hoặc English... (Enter để gửi)",
                        label="",
                        show_label=False,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Gửi ▶", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ New session", size="sm", variant="secondary")
                    session_display = gr.Textbox(
                        label="Session ID",
                        interactive=False,
                        scale=4,
                        max_lines=1,
                    )

                with gr.Accordion("⚙️ Backend", open=False):
                    backend_radio = gr.Radio(
                        choices=["OpenAI (GPT-4o-mini)", "SGLang (local)"],
                        value="OpenAI (GPT-4o-mini)",
                        label="LLM Backend",
                        info="SGLang yêu cầu python main.py đang chạy",
                    )
                    api_url_input = gr.Textbox(
                        value="http://localhost:18000",
                        label="API URL",
                        visible=False,
                        placeholder="http://localhost:18000",
                    )
                    backend_radio.change(
                        fn=lambda b: gr.update(visible="SGLang" in b),
                        inputs=backend_radio,
                        outputs=api_url_input,
                    )

                gr.Markdown("**Gợi ý nhanh:**")
                with gr.Row():
                    for label, prompt in QUICK_PROMPTS[:4]:
                        gr.Button(label, size="sm").click(
                            fn=lambda p=prompt: p, outputs=msg_input
                        )
                with gr.Row():
                    for label, prompt in QUICK_PROMPTS[4:]:
                        gr.Button(label, size="sm").click(
                            fn=lambda p=prompt: p, outputs=msg_input
                        )

            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 📊 Metrics")

                intent_box     = gr.Textbox(label="🎯 Intent",          interactive=False, value="—")
                agent_box      = gr.Textbox(label="🤖 Agent",           interactive=False, value="—")
                total_lat_box  = gr.Textbox(label="⏱ Total latency",    interactive=False, value="—")
                router_lat_box = gr.Textbox(label="⚡ Router latency",   interactive=False, value="—")
                agent_lat_box  = gr.Textbox(label="🔄 Agent latency",    interactive=False, value="—")
                parse_box      = gr.Textbox(label="⚙️ Parse method",     interactive=False, value="—")
                cache_box      = gr.Textbox(label="💾 Semantic Cache",   interactive=False, value="—")
                context_box    = gr.Textbox(label="📚 Retrieved context",interactive=False, value="—", max_lines=3)

                gr.Markdown("---")
                gr.Markdown("""
### 🏗️ Architecture
**Router** · Llama3-1B
Log-prob scoring · 4 intents

**Agents** (GPT-4o-mini)
🛒 Order → Menu + Cart
💡 Consultant → Tag RAG
❓ FAQ → Hybrid RAG

**Semantic Cache** (C2.2)
Threshold: 0.92 · paraphrase-MiniLM
Cache hit → skip LLM

**Session**
Window: 5 turns · TTL: 30 min
Auto-summarize at 70% ctx
                """)

        ALL_METRIC_OUTPUTS = [
            intent_box, agent_box,
            total_lat_box, router_lat_box, agent_lat_box,
            parse_box, cache_box, context_box,
        ]

        RESPOND_OUTPUTS = [
            chatbot, msg_input,
            *ALL_METRIC_OUTPUTS,
            session_id_state, session_display,
        ]

        async def respond(message, history, session_id, backend, api_url):
            if not message or not message.strip():
                yield (history or [], "", *["—"] * _N_METRICS,
                       session_id, session_id[:8] + "...")
                return

            msg = message.strip()
            h = list(history or [])
            h.append({"role": "user",      "content": msg})
            h.append({"role": "assistant", "content": "⏳ đang xử lý..."})

            yield (h, "", "⏳ Routing...", *["—"] * (_N_METRICS - 1),
                   session_id, session_id[:8] + "...")

            if "SGLang" in backend:
                async for chunk in _sglang_respond(msg, h, session_id, api_url):
                    yield chunk
                return

            result = await process_message(msg, session_id)
            reply  = result["reply"]

            words   = reply.split(" ")
            partial = ""
            for i, word in enumerate(words):
                partial += word + (" " if i < len(words) - 1 else "")
                h[-1] = {"role": "assistant", "content": partial}
                yield (h, "", "⏳", *["—"] * (_N_METRICS - 1),
                       session_id, session_id[:8] + "...")
                await asyncio.sleep(0.018)

            h[-1] = {"role": "assistant", "content": reply}

            intent_label = INTENT_LABELS.get(result["intent"], result["intent"])
            agent_label  = INTENT_LABELS.get(result["agent_type"], result["agent_type"])
            parse_method = "✅ JSON" if not result["fallback"] else "⚠️ Fallback"

            if result["cache_hit"]:
                cache_status = f"✅ HIT  score={result['cache_score']:.2f}  (<100ms)"
            else:
                cache_status = "❌ MISS → cached for next time"

            yield (
                h,
                "",
                intent_label,
                agent_label,
                f"{result['total_latency']:.0f} ms",
                f"{result['router_latency']:.0f} ms",
                f"{result['agent_latency']:.0f} ms",
                parse_method,
                cache_status,
                result.get("retrieved_context", "—"),
                session_id,
                session_id[:8] + "...",
            )

        def new_session():
            new_id = str(uuid.uuid4())
            return ([], "", *["—"] * _N_METRICS, new_id, new_id[:8] + "...")

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, session_id_state, backend_radio, api_url_input],
            outputs=RESPOND_OUTPUTS,
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, session_id_state, backend_radio, api_url_input],
            outputs=RESPOND_OUTPUTS,
        )
        clear_btn.click(
            fn=new_session,
            outputs=RESPOND_OUTPUTS,
        )

        app.load(
            fn=lambda s: s[:8] + "...",
            inputs=session_id_state,
            outputs=session_display,
        )

    return app


if __name__ == "__main__":
    startup()
    ui = create_ui()
    ui.queue()  
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="orange"),
        css="footer { display: none !important; }",
    )
