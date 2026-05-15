# PHAN_A — Multi-Agent LLM Coffee Shop Chatbot

**MAS-LLM-2026-FINAL | FI-AI**

Hệ thống chatbot đa tác nhân phục vụ quán cà phê, xử lý đặt món, tư vấn thực đơn và hỏi đáp bằng tiếng Việt, tiếng Anh và tiếng Hàn.

---

## Kiến Trúc

```
User Query
    │
    ▼
Router Agent (Qwen2.5-1.5B + PEFT, log-prob scoring)
    │
    ├── order      → OrderAgent      (GPT-4o-mini + menu context)
    ├── consultant → ConsultantAgent (keyword scoring + menu tags)
    ├── faq        → FAQAgent        (Hybrid RAG: keyword + semantic + rerank)
    └── ignore     → Greeting template (no LLM call)
          │
          ↕  SemanticCache (threshold 0.92) — cache hit ≤ 50ms
          ↕  SessionStore (history 5 turns, TTL 30 phút)
          ↕  RequestQueue (semaphore max=3, timeout=60s)
```

**Part A** (`demo.py`) — chạy độc lập, GPT-4o-mini API, dữ liệu JSON in-memory

**Part B** (`docker-compose.yml`) — full stack: Neo4j + SGLang + BGE embedding/reranker

---

## Yêu Cầu

- Python 3.10 
- NVIDIA GPU với CUDA (RTX 2050 4GB VRAM hoặc tốt hơn)
- OpenAI API key (cho Part A)
- Docker Desktop (cho Part B)

---

## Cài Đặt

```bash
# 1. Clone repo
git clone <repo-url>
cd 

# 2. Tạo conda env
conda create -n name_env python=3.11
conda activate name_env

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Tạo .env
cp .env.example .env
# Điền API_OPENAI vào .env
```

---

## Tải Models (Google Drive)

Các model file không được lưu trong Git do dung lượng lớn. Tải về từ Google Drive và đặt vào đúng thư mục:

| Model | Thư mục đích | Kích thước |
|---|---|---|
| Qwen2.5-1.5B (base) | `models/Qwen2.5-1.5B/` | ~3 GB |
| router-merged (active router) | `models/router-merged/` | ~3 GB |
| checkpoint-400 (LoRA adapter) | `models/checkpoint-400/` | ~30 MB |
| router-awq (AWQ int4, SGLang) | `models/router-awq/` | 1.1 GB |
| GGUF variants | `models/gguf/` | 861 MB – 2.9 GB mỗi file |

> **Link Google Drive**: *[điền link Drive vào đây trước khi nộp bài]*

Sau khi tải về, cấu trúc thư mục phải như sau:

```
models/
├── Qwen2.5-1.5B/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── router-merged/
│   ├── config.json
│   └── model.safetensors
├── checkpoint-400/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── router-awq/
│   ├── config.json
│   └── model.safetensors
└── gguf/
    ├── router-f16.gguf      (2.9 GB)
    ├── router-q4km.gguf     (941 MB)
    ├── router-q4_0.gguf     (892 MB)
    ├── router-q4ks.gguf     (897 MB)
    └── router-iq4xs.gguf    (861 MB)
```

---

## Chạy Part A (Demo Gradio)

```bash
# Cài thêm nếu chưa có
pip install gradio pandas

# Chạy demo (tự động load model, sync CSV → JSON, mở browser)
python demo.py
# → http://localhost:7860
```

Startup sequence tự động:
1. Đọc `data/watch/menu.csv` + `data/watch/Faq.csv` → ghi `data/menu.json` + `data/faq.json`
2. Load Qwen2.5-1.5B + PEFT từ `models/router-merged/` (~25s lần đầu)
3. Khởi tạo 3 agents + SemanticCache + IntentExtractor

---

## Chạy FastAPI Server

```bash
python main.py
# → http://localhost:18000

# Health check
curl http://localhost:18000/health

# Chat API
curl -X POST http://localhost:18000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "cho tôi 1 ly cà phê đen", "session_id": "test-001"}'
```

---

## Chạy Part B (Full Docker Stack)

```bash
# Khởi động tất cả services
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps

# Xem log
docker-compose logs -f graph_rag
```

**Services và ports:**

| Service | Port | Mô tả |
|---|---|---|
| neo4j | 7474 / 7687 | Graph DB (Neo4j Browser: http://localhost:7474) |
| redis | 6379 | Session cache |
| embedding | 8001 | BAAI/bge-m3 embedding service |
| reranker | 8002 | BAAI/bge-reranker-v2-m3 |
| ingestion | 8003 | Auto-watch `data/watch/`, ingest vào Neo4j |
| graph_rag | 8004 | Graph RAG API |
| llm_serving | 8000 | SGLang LLM serving (AWQ) |

---

## Router Standalone

```bash
# Test active router (Qwen2.5-1.5B + PEFT merged, log-prob scoring)
python src/router/model.py

```

---

## Benchmark

```bash
# Router accuracy + latency (140 samples)
python scripts/benchmark_router.py
# → data/benchmark_results.json

# Semantic cache hit rate (500 samples)
python scripts/benchmark_cache.py
# → data/benchmark_cache_results.json
```

**Kết quả thực tế (RTX 2050, CPU inference):**

| Metric | Kết quả |
|---|---|
| Router accuracy | 95.71% (134/140) |
| Hard sample accuracy | 100% |
| Router latency p90 (CPU) | 390.5ms |
| Cache hit rate | 7.3% (rule-based extractor) |
| Cache latency p90 | 21.0ms |

---

## Cập Nhật Dữ Liệu Menu / FAQ

```bash
# Sửa file CSV
# data/watch/menu.csv  — 25 món
# data/watch/Faq.csv   — 30 FAQ entries

# Part A: restart demo.py (tự đọc lại CSV khi khởi động)
python demo.py

# Part B: watcher service tự detect và ingest lại vào Neo4j
# Không cần restart docker
```

---

## Testing

```bash
pytest test/
pytest test/test_router.py -v
pytest test/ --cov=src/
```

---

## Cấu Trúc Thư Mục

```
PHAN_A/
├── src/
│   ├── agents/            # OrderAgent, ConsultantAgent, FAQAgent, BaseAgent
│   ├── router/            # RouterModel (log-prob scoring), IntentClassifier
│   ├── session/           # SessionStore, SessionCleanup, Summarizer
│   ├── queue/             # RequestQueue, retry_with_backoff
│   ├── cache/             # SemanticCache, EmbeddingProvider
│   ├── intent_extraction/ # IntentExtractor (rule-based)
│   └── guardrails/        # RateLimiter, CircuitBreaker, TTSPreprocessor
├── services/
│   ├── embedding/         # BAAI/bge-m3 Docker service
│   ├── reranker/          # BAAI/bge-reranker-v2-m3 Docker service
│   ├── ingestion/         # CSV/PDF ingest vào Neo4j
│   ├── graph_rag/         # Graph RAG API (Neo4j + hybrid search)
│   └── llm_serving/       # SGLang serving endpoint
├── data/
│   ├── watch/             # menu.csv, Faq.csv (nguồn chỉnh sửa)
│   ├── menu.json          # Generated từ menu.csv (auto)
│   ├── faq.json           # Generated từ Faq.csv (auto)
│   ├── synthetic/         # Training data router SFT (8,800 samples)
│   ├── cache_test_set/    # Test set semantic cache (500 samples)
│   └── docs/              # Documents ingest vào Neo4j
├── models/                # Tải từ Google Drive (xem hướng dẫn trên)
├── training/
│   ├── merge_and_export.py    # Merge LoRA adapter vào base model
│   └── quantize_edge.py       # Export GGUF quantization
├── scripts/
│   ├── benchmark_router.py    # Router accuracy + latency benchmark
│   └── benchmark_cache.py     # Semantic cache hit rate benchmark
├── demo.py                # Gradio UI (Part A standalone)
├── main.py                # FastAPI server (port 18000)
├── docker-compose.yml     # Full Part B stack
├── config.yaml            # Hyperparameters
└── .env                   # API keys (không commit)
```

---

## Môi Trường

- OS: Windows 11
- Python: 3.11 (conda env `name_env`)
- GPU: NVIDIA RTX 2050 (4GB VRAM)
- CUDA: tự động detect qua `device_map="auto"`
- Docker: Docker Desktop for Windows
