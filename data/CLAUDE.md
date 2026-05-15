# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PHAN_A** is a multi-agent LLM system for a Vietnamese coffee shop chatbot (MAS-LLM-2026-FINAL, FI-AI). It uses intent routing to dispatch user queries to specialized agents (Order, Consultant, FAQ) with session management, request queueing, semantic caching, and conversation summarization.

### Architecture Layers

1. **Data Layer** (`src/pipeline/`, `data/`, `training/`) — Synthetic data generation and model fine-tuning (one-time activities)
2. **Runtime Layer** (`src/agents/`, `src/session/`, `src/queue/`, `src/router/`, `src/cache/`, `src/guardrails/`, `src/intent_extraction/`, `main.py`) — Production serving with FastAPI
3. **Infrastructure** (`config.yaml`, `docker-compose.yml`, `services/`) — Configuration, deployment, and testing

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Development server on http://0.0.0.0:18000
python main.py

# Gradio demo UI on http://0.0.0.0:7860  (requires: pip install gradio pandas)
python demo.py

# Full infrastructure stack (Neo4j, Redis, embedding, reranker, ingestion, graph_rag, llm_serving)
docker-compose up -d
```

### Testing

```bash
pytest test/
pytest test/test_router.py -v
pytest test/ --cov=src/
```

### Health Check

```bash
curl http://localhost:18000/health
```

### Run Router Standalone

```bash
# Test Qwen2.5-1.5B + PEFT log-prob router (active, loads models/router-merged)
python src/router/model.py

# Test Qwen2.5-1.5B-Instruct + PEFT adapter router (staging alternative)
python a.py
```

## Codebase Structure

### Core Components

**Intent Routing (`src/router/`)**
- `classifier.py` — `IntentClassifier` classifies messages into `order`, `consultant`, `faq`, `ignore` with three fallback parse strategies: JSON strict → regex extract → keyword match
- `model.py` — Active router: loads **Qwen2.5-1.5B + PEFT** (`models/router-merged` — merged from base `models/Qwen2.5-1.5B` + adapter `models/checkpoint-400`) and uses **token log-probability scoring** — batch=4 forward pass scores all 4 intent tokens simultaneously, returns argmax mean log-prob (no auto-regressive generation). dtype=bfloat16.
- `a.py` (root) — Alternative router: **Qwen2.5-1.5B-Instruct** + **PEFT adapter** from `models/router-sft/checkpoint-1500/`. Uses auto-regressive generation. Includes Windows path fix (`.resolve().as_posix()`) for PEFT on Windows.
- `b.py` (root) — Matches the current `src/router/model.py`; used for A/B comparison

**Agents (`src/agents/`)**
- All agents inherit from `BaseAgent` with `async handle(query, history, session_id)`, language auto-detection (Vietnamese Unicode), timing, and error handling
- **OrderAgent** — GPT-4o-mini with menu context and cart history. GPT appends `[CART_JSON]{"items":[...]}[/CART_JSON]` at the end of each response; `_parse_cart_update()` extracts it, `_strip_cart_json()` removes it before displaying. Cart accumulates correctly across multi-turn.
- **ConsultantAgent** — Keyword scoring + tag matching against `data/menu.json` (no embeddings)
- **FAQAgent** — Hybrid RAG: keyword search → semantic n-gram → graph tag expansion → late rerank, all in-memory from `data/faq.json`. Read with `encoding="utf-8"` (Windows cp1252 default causes crash).

**Session Management (`src/session/`)**
- `SessionStore` — In-memory, fixed history window (5 turns), auto-summarization when token count > 70% of context window (via GPT-4o-mini), 30-minute TTL
- `SessionCleanup` — Background async task (every 120s) deletes expired sessions
- Summarized history prepended as `[Conversation summary so far]: ...` system turn on retrieval
- **Lưu ý**: `src/session/summarizer.py` hardcode `model="gpt-4o-mini"` — không đọc từ config.yaml

**Request Queue (`src/queue/`)**
- `RequestQueue` — Asyncio semaphore (max 3 concurrent LLM calls), 60s per-request timeout
- `retry_with_backoff` — Exponential backoff decorator (1s → 2s → 4s, max 30s, 3 attempts)
- **Bug fixed**: `finally` block only releases semaphore when `acquired = True`, prevents over-release on timeout before acquire

**Semantic Cache (`src/cache/`)**
- `SemanticCache` — FAISS IndexFlatIP với fallback numpy cosine similarity (FAISS chưa được install trong môi trường hiện tại → đang dùng numpy brute-force)
  - Intent mode (C2.2): threshold 0.92, key = `action` text từ `IntentExtractor`
  - Full-query mode (B2.3): threshold 0.95, key = full query
- **Benchmark thực tế**: hit_rate=7.3% (36/490) — thấp vì C2.1 SLM chưa được train, rule-based fallback nhét context vào action key → cache miss. Upper-bound hit_rate=100% (khi action key đúng). Latency cache hit: avg=15.7ms, p90=20.5ms.
- Tag-based invalidation (`invalidate_by_tag`) khi menu/FAQ thay đổi
- `EmbeddingProvider` — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` locally, hoặc TEI HTTP API
- `src/cache/__init__.py` exports `SemanticCache`, `EmbeddingProvider`, `CacheEntry`

**Guardrails (`src/guardrails/`)**
- `SlidingWindowRateLimiter` — In-memory per-IP rate limits:
  - `/api/chat`: 30 req/min
  - `/api/order`: 20 req/min
  - `/api/consultant`: 20 req/min
  - `/api/faq`: 40 req/min
  - `/health`: 120 req/min
  - default: 60 req/min
- `rate_limit_dependency` — FastAPI `Depends()` cho route-level enforcement
- `health.py` — Health checks cho router, generator, graph DB, cache. Graph check dùng `neo4j.GraphDatabase.driver()` trực tiếp (không qua `src.graph.client` — module này không tồn tại)
- `fallback.py`, `tts_preprocessor.py` — Graceful degradation, TTS text preprocessing

**Intent Extraction (`src/intent_extraction/`)**
- Module C2.1/C2.2: tách câu user thành 3 phần cấu trúc cho Semantic Cache pipeline
- `model.py` — `IntentExtractionModel` singleton: load fine-tuned SLM (`Qwen2.5-0.5B-Instruct` base) từ `training/checkpoints/intent_merged`. **Checkpoint chưa tồn tại** → `is_loaded() = False`, extractor tự fallback rule-based. SFT script: `training/sft_intent.py`.
- `extractor.py` — `IntentExtractor`: chạy model → parse JSON → fallback rule-based nếu parse fail. **Root cause của cache hit rate thấp**: rule-based fallback nhét context (thời gian, số người, dịp) vào action key → "order cà phê sữa đá cuối tuần này" thay vì "order cà phê sữa đá" → không match warmup entry.
- Output: `{"subject": str, "action": str, "context": str}` — `action` dùng làm cache key
- `src/intent_extraction/__init__.py` exports `IntentExtractor`, `IntentResult`

### Configuration

**`config.yaml`** — All hyperparameters:
- `router` — Model path, max tokens, temperature, latency thresholds (excellent: 100ms, target: 200ms, slow: 1000ms)
- `session` — Window size (5), context threshold (0.70), TTL (30 min), cleanup interval (120s)
- `queue` — Max concurrent LLM calls (3), timeout (60s), retry settings

**`.env`** — API keys và biến môi trường:
- `API_OPENAI` — GPT-4o-mini cho agents và summarizer
- `API_GROQ_CLOUD` — Groq fallback (chưa active)
- `DEVICE` — **Chú ý**: nếu set `DEVICE=cuda` trong `.env`, docker-compose sẽ truyền vào embedding/reranker container và gây crash (container không có GPU). Embedding/reranker đã hardcode `DEVICE: cpu` trong docker-compose.yml để tránh vấn đề này.

### Docker Infrastructure

The `docker-compose.yml` runs 7 services:
| Service | Port | Purpose |
|---|---|---|
| neo4j | 7474/7687 | Graph DB (APOC enabled), credentials từ `NEO4J_PASSWORD` env |
| redis | 6379 | Session cache + semantic cache persistence |
| embedding | 8001 | BAAI/bge-m3 embedding service — luôn chạy CPU |
| reranker | 8002 | BAAI/bge-reranker-v2-m3 reranking service — luôn chạy CPU |
| ingestion | 8003 | Watches `data/watch/`, ingests into Neo4j. Requires `python-multipart` |
| graph_rag | 8004 | Graph RAG API (semantic cache threshold 0.95) |
| llm_serving | 8000 | vLLM-compatible LLM serving endpoint |

**Lưu ý Dockerfile** (`services/embedding/`, `services/reranker/`): phải install **CPU torch trước**, rồi mới `sentence-transformers`. Nếu đảo thứ tự, pip sẽ kéo CUDA torch về làm dependency → container crash vì không có NVIDIA driver.

**Healthcheck**: `python:3.10-slim` không có `curl`. Healthcheck dùng `python -c "import urllib.request; urllib.request.urlopen(...)"` thay vì `curl -f`. Áp dụng cho embedding, reranker, graph_rag.

**RERANK_THRESHOLD**: `bge-reranker-v2-m3` (cross-encoder) cho score rất thấp khi query không dấu vs. document có dấu tiếng Việt (ví dụ "wifi mat khau" vs "Mật khẩu WiFi": score=0.19, nhưng "trà đào" vs "tra nao mat" chỉ ~0.004). Threshold phải là `0.001`, KHÔNG phải 0.7. Đặt trong `docker-compose.yml`: `RERANK_THRESHOLD: "0.001"`.

**CSV Ingestion (services/ingestion/ingest_menu.py)**: Excel export CSV có ingredients là JSON array, ví dụ `"""cà phê robusta","đường"""`. Python `csv.DictReader` split nhầm trên dấu phẩy bên trong, khiến `ingredients='"cà phê robusta'` và `description='đường"'`. Hàm `_fix_row()` trong `ingest_menu.py` detect và merge lại: nếu `ingredients` bắt đầu bằng `"` nhưng không kết thúc bằng `"` thì field bị split → merge `ingredients` + `description` thành một.

### Data & Models Directory

**Nguồn dữ liệu chính:**
- `data/watch/menu.csv` — 25 món (20 đồ uống + 5 đồ ăn), nguồn thật để cập nhật
- `data/watch/Faq.csv` — 30 FAQ entries, nguồn thật để cập nhật

**Generated on startup (demo.py):**
- `data/menu.json` — Generated từ `menu.csv` khi `demo.py` khởi động
- `data/faq.json` — Generated từ `Faq.csv` khi `demo.py` khởi động

**Cách update dữ liệu:** sửa file CSV trong `data/watch/`, restart `python demo.py`. Hàm `_sync_data_from_csv()` trong `demo.py` tự đọc lại và ghi đè JSON.

**Format CSV:** exported từ Excel với row 0 là chữ cái cột (A,B,C...), row 1 là header thực. Dùng `pd.read_csv(path, skiprows=1)` để parse đúng. **Cột `ingredients` chứa JSON array** (ví dụ `["cà phê robusta","đường"]`) → Python csv DictReader split nhầm → `ingest_menu.py` có hàm `_fix_row()` để merge lại. Khi thay đổi format CSV cần kiểm tra lại hàm này.

**Models:**
- `models/Qwen2.5-1.5B/` — Base model nguồn cho merge (router active)
- `models/checkpoint-400/` — PEFT/LoRA adapter trained trên Qwen2.5-1.5B (r=16, alpha=32)
- `models/router-merged/` — **Active router**: Qwen2.5-1.5B + PEFT merged, đây là model `src/router/model.py` load
- `models/Qwen2.5-1.5B-Instruct/` — Base model cho `a.py` (staging alternative)
- `models/router-sft/checkpoint-1500/` — PEFT adapter cho `a.py` variant (staging)
- `training/checkpoints/intent_merged/` — Fine-tuned SLM cho intent extraction (C2.1) — **chưa có**, extractor dùng rule-based fallback

## Key Design Patterns

### Intent Dispatch Pipeline (demo.py + main.py)

1. `IntentClassifier.classify(message)` → intent + latency
2. `ignore` intent → return greeting, không gọi agent
3. `IntentExtractor.extract(message)` → `action` (cache key)
4. `SemanticCache.query(action)` threshold 0.92 → **cache hit** → return ngay (≤100ms)
5. **Cache miss**: fetch history từ `SessionStore` → submit qua `RequestQueue` → agent
6. `SemanticCache.put(action, response)` — lưu cho lần sau
7. Lưu user + assistant turns vào session

### Cart Tracking (OrderAgent)

GPT-4o-mini được yêu cầu append `[CART_JSON]{"items":[{"name":..,"price":..,"quantity":..}]}[/CART_JSON]` cuối mỗi response. `_parse_cart_update()` extract JSON này; `_strip_cart_json()` xóa nó khỏi text hiển thị. History metadata lưu cart, lần sau `_extract_cart_from_history()` đọc lại.

### Semantic Cache Pipeline (C2.2)

1. `IntentExtractor.extract(message)` → tách `action` + `context`
2. `SemanticCache.query(action)` với threshold 0.92
3. **Cache hit** → trả về ngay (≤100ms, không qua LLM)
4. **Cache miss** → gọi Agent, lưu kết quả vào cache

### Part A vs Part B (Tách biệt hoàn toàn)

| | Part A (`demo.py`) | Part B (Docker) |
|---|---|---|
| Embedding | paraphrase-MiniLM (local, cache only) | BAAI/bge-m3 (port 8001) |
| Knowledge base | `data/*.json` (in-memory) | Neo4j graph DB (port 7687) |
| Reranker | Keyword score heuristic | BGE-reranker-v2-m3 (port 8002) |
| LLM | GPT-4o-mini (API) | vLLM/SGLang (port 8000) |
| Trạng thái | **Hoạt động ngay** | Cần LLM serving riêng |

**Cả 3 agents đều lấy dữ liệu từ Neo4j qua graph_rag:8004**, không đọc JSON trực tiếp:
- **OrderAgent** — Gọi `GET /menu/all` lấy toàn bộ MenuItem, TTL cache 5 phút, fallback JSON nếu Neo4j chưa sẵn sàng
- **ConsultantAgent** — Gọi `POST /search` per-request, lấy MenuItem nodes
- **FAQAgent** — Gọi `POST /search` per-request, lấy Chunk nodes

**Data update flow** (khi thay đổi CSV):
1. Admin sửa `data/watch/menu.csv` hoặc `data/watch/Faq.csv`
2. Watcher service tự detect → gọi `ingest_menu_csv()` / `ingest_faq_csv()`
3. Ingest xóa data cũ (`DETACH DELETE`) rồi insert mới → đảm bảo không có stale data
4. Sau ingest thành công → tự gọi `DELETE http://graph_rag:8004/cache` → xóa semantic cache
5. OrderAgent sẽ refresh menu tự động trong ≤5 phút (TTL), hoặc ngay khi restart demo.py

### Router: Two Competing Implementations

| | `src/router/model.py` (active) | `a.py` (staging) |
|---|---|---|
| Base model | `models/Qwen2.5-1.5B` | `models/Qwen2.5-1.5B-Instruct` |
| Loaded from | `models/router-merged` (PEFT merged) | `models/Qwen2.5-1.5B-Instruct` + adapter `models/router-sft/checkpoint-1500/` |
| Method | Log-prob scoring, batch=4 forward pass | Auto-regressive generation + PEFT |
| dtype | bfloat16 | bfloat16 |
| PEFT | Merged into weights (không cần adapter riêng) | Loaded via PEFT (adapter riêng) |
| Windows fix | N/A | `.resolve().as_posix()` cho PEFT path |

### Agent Response Format

```python
AgentResponse(
    text: str,               # user-facing reply (đã strip [CART_JSON] block)
    agent_type: str,         # "order" | "consultant" | "faq"
    session_id: str,
    latency_ms: float,
    metadata: dict,          # cart items, retrieved docs, etc.
    language: str,           # auto-detected "vi" or "en"
    error: Optional[str],
)
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Main: `{message, session_id?}` → `{reply, session_id, intent, latency_ms, agent_latency_ms}` |
| GET | `/health` | Router loaded status, queue stats, session stats |
| DELETE | `/session/{session_id}` | Manual session cleanup |

## Demo

`demo.py` — Gradio 6.x web UI chạy toàn bộ hệ thống trong 1 process (không cần FastAPI riêng):

```bash
pip install gradio pandas
python demo.py   # mở browser tự động tại http://localhost:7860
```

**Startup sequence:**
1. `_sync_data_from_csv()` — đọc `data/watch/*.csv` → ghi `data/menu.json` + `data/faq.json`
2. `router_classifier.load()` — load Qwen2.5-1.5B + PEFT (`models/router-merged`) trên GPU (~25s lần đầu)
3. Khởi tạo 3 agents + SemanticCache + IntentExtractor

**UI gồm:**
- Chat panel với streaming (yield từng từ, delay 18ms)
- 8 metrics: Intent, Agent, Total/Router/Agent latency, Parse method, Cache status, Retrieved context
- 8 quick prompts cover cả 3 agents
- Nút New session (reset history + cache display)
- `ui.queue()` bắt buộc để streaming hoạt động

**5 luồng demo được implement:**
1. **Router** — Intent + latency hiện ngay trên panel
2. **RAG context** — Panel "Retrieved context" hiện FAQ nodes / MenuItem / Cart items
3. **Semantic Cache** — Panel "Semantic Cache" hiện HIT/MISS + score; HIT bỏ qua LLM hoàn toàn
4. **Multi-turn Cart** — `[CART_JSON]` mechanism đảm bảo cart cộng dồn đúng qua các turn
5. **Streaming** — Response xuất hiện từng từ trong Gradio

## Important Notes

- **Production target**: RTX 3060, max 3 concurrent LLM requests phòng OOM
- **No persistent sessions**: All in-memory; mất khi restart. Redis trong docker-compose là path để persist.
- **Multilingual**: Vietnamese detect qua Unicode range; dual-language system prompts trong `src/agents/prompts/`
- **Router singleton**: `RouterModel` dùng `__new__` singleton — chỉ load GPU model 1 lần
- **Gradio 6.x breaking changes**: `theme` và `css` chuyển từ `Blocks()` sang `launch()`. `Chatbot` không còn `type=`, `bubble_full_width`, `show_copy_button` — mặc định đã là messages format. Streaming dùng async generator + `ui.queue()`.
- **Windows encoding**: luôn dùng `encoding="utf-8"` khi đọc/ghi file JSON chứa tiếng Việt. Windows default là cp1252 → crash.
