import os
import json
import time
import argparse
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROUTER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"          # base model gốc
FINETUNED_CKPT  = "training/checkpoints/router_sft"       # checkpoint sau SFT
GGUF_OUT_DIR    = "models/gguf"
LLAMA_CPP_DIR   = os.environ.get("LLAMA_CPP_DIR", "./llama.cpp")  # clone llama.cpp vào đây

QUANT_LEVELS = ["Q4_0", "Q4_K_M", "Q4_K_S", "IQ4_XS"]

# Prompt mẫu dùng để benchmark latency
BENCHMARK_PROMPTS = [
    "Cho tôi xem menu",
    "Đặt 2 ly cà phê sữa đá",
    "Giờ mở cửa của quán là mấy giờ",
    "Tôi muốn tư vấn món uống phù hợp buổi chiều",
    "Hủy order vừa đặt",
    "Quán có wifi không",
    "Cho tôi hỏi về chương trình khuyến mãi",
    "Đặt bàn cho 4 người tối nay",
    "Thêm một bánh croissant vào đơn",
    "Thanh toán bằng thẻ được không",
]


@dataclass
class QuantBenchmarkResult:
    quant_level: str
    model_size_mb: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    accuracy_score: float          # so với FP16 baseline, 1.0 = identical intent
    memory_rss_mb: float
    tokens_per_second: float
    meets_edge_target: bool        # ≤ 500ms


# ---------------------------------------------------------------------------
# Step 1: Convert sang GGUF F16 (bước đầu)
# ---------------------------------------------------------------------------

def convert_to_gguf_f16(model_path: str, out_dir: str) -> str:
    """Dùng llama.cpp convert script để tạo GGUF F16 từ HuggingFace checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f16_path = out_dir / "router-f16.gguf"
    if f16_path.exists():
        print(f"[convert] F16 GGUF đã tồn tại: {f16_path}")
        return str(f16_path)

    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Fallback: dùng convert.py (llama.cpp cũ hơn)
        convert_script = Path(LLAMA_CPP_DIR) / "convert.py"

    cmd = [
        "python", str(convert_script),
        model_path,
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ]
    print(f"[convert] Chạy: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Convert thất bại:\n{result.stderr}")

    print(f"[convert] Xong: {f16_path} ({f16_path.stat().st_size / 1e6:.1f} MB)")
    return str(f16_path)


# ---------------------------------------------------------------------------
# Step 2: Quantize sang từng level
# ---------------------------------------------------------------------------

def quantize_model(f16_gguf: str, quant_level: str, out_dir: str) -> str:
    """Gọi llama-quantize để tạo model với quant level tương ứng."""
    out_dir = Path(out_dir)
    out_path = out_dir / f"router-{quant_level.lower()}.gguf"

    if out_path.exists():
        print(f"[quantize] {quant_level} đã tồn tại: {out_path}")
        return str(out_path)

    quantize_bin = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = Path(LLAMA_CPP_DIR) / "quantize"   # tên cũ

    cmd = [str(quantize_bin), f16_gguf, str(out_path), quant_level]
    print(f"[quantize] {quant_level} → {out_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantize {quant_level} thất bại:\n{result.stderr}")

    size_mb = out_path.stat().st_size / 1e6
    print(f"[quantize] Xong {quant_level}: {size_mb:.1f} MB")
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 3: Benchmark latency trên thiết bị hiện tại
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "Bạn là Router Agent. Phân loại câu hỏi thành một trong 4 intent: "
    "order | consultant | faq | chitchat. "
    "Chỉ trả về JSON: {\"intent\": \"<label>\"}. Không giải thích thêm."
)


def _build_llama_prompt(user_text: str) -> str:
    """Tạo prompt theo format ChatML cho Qwen."""
    return (
        "<|im_start|>system\n"
        f"{ROUTER_SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def benchmark_gguf_model(
    gguf_path: str,
    quant_level: str,
    prompts: List[str],
    n_threads: Optional[int] = None,
) -> QuantBenchmarkResult:
    """
    Chạy llama-cli (llama-main) với từng prompt, đo latency.
    Trả QuantBenchmarkResult.
    """
    llama_cli = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-cli"
    if not llama_cli.exists():
        llama_cli = Path(LLAMA_CPP_DIR) / "main"

    n_threads = n_threads or os.cpu_count() or 4
    latencies = []
    tps_list = []
    correct = 0

    # Ground-truth intent đơn giản dựa trên keyword (dùng cho accuracy proxy)
    def _gt_intent(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["đặt", "thêm", "hủy", "thanh toán", "order"]):
            return "order"
        if any(k in t for k in ["tư vấn", "phù hợp", "gợi ý"]):
            return "consultant"
        if any(k in t for k in ["giờ", "wifi", "menu", "khuyến mãi", "bàn", "thẻ"]):
            return "faq"
        return "chitchat"

    for user_text in prompts:
        prompt = _build_llama_prompt(user_text)

        cmd = [
            str(llama_cli),
            "-m", gguf_path,
            "--prompt", prompt,
            "-n", "32",          # max tokens sinh ra
            "--temp", "0",
            "-t", str(n_threads),
            "--no-display-prompt",
            "-ngl", "0",         # CPU only (ARM)
        ]

        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        output = result.stdout.strip()

        # Parse tokens/s từ stderr của llama.cpp
        for line in result.stderr.splitlines():
            if "eval time" in line and "ms per token" in line:
                # format: "llama_print_timings: eval time = X ms / Y tokens (Z ms per token, W tokens/s)"
                try:
                    tps = float(line.split("tokens/s")[0].split(",")[-1].strip())
                    tps_list.append(tps)
                except Exception:
                    pass

        # Accuracy check (proxy)
        gt = _gt_intent(user_text)
        try:
            import re
            m = re.search(r'"intent"\s*:\s*"(\w+)"', output)
            pred = m.group(1) if m else "unknown"
            if pred == gt:
                correct += 1
        except Exception:
            pass

    import statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    gguf_size_mb = Path(gguf_path).stat().st_size / 1e6

    # RSS memory: đọc /proc/self/status nếu Linux ARM
    rss_mb = 0.0
    if platform.system() == "Linux":
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        rss_mb = int(line.split()[1]) / 1024
                        break
        except Exception:
            pass

    avg_lat = statistics.mean(latencies)
    p50 = latencies_sorted[int(n * 0.50)]
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]
    p99 = latencies_sorted[min(int(n * 0.99), n - 1)]
    accuracy = correct / len(prompts)
    avg_tps = statistics.mean(tps_list) if tps_list else 0.0

    return QuantBenchmarkResult(
        quant_level=quant_level,
        model_size_mb=gguf_size_mb,
        avg_latency_ms=avg_lat,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        accuracy_score=accuracy,
        memory_rss_mb=rss_mb,
        tokens_per_second=avg_tps,
        meets_edge_target=p95 <= 500.0,
    )


# ---------------------------------------------------------------------------
# Step 4: Report
# ---------------------------------------------------------------------------

def print_report(results: List[QuantBenchmarkResult]) -> None:
    print("\n" + "=" * 90)
    print("C1 EDGE BENCHMARK REPORT — Router trên ARM CPU (CIX P1 target ≤ 500ms)")
    print("=" * 90)
    header = (
        f"{'Quant':<12} {'Size(MB)':>10} {'Avg(ms)':>10} {'P50(ms)':>10} "
        f"{'P95(ms)':>10} {'P99(ms)':>10} {'Acc':>6} {'TPS':>8} {'≤500ms':>8}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        ok = "✅" if r.meets_edge_target else "❌"
        print(
            f"{r.quant_level:<12} {r.model_size_mb:>10.1f} {r.avg_latency_ms:>10.1f} "
            f"{r.p50_latency_ms:>10.1f} {r.p95_latency_ms:>10.1f} {r.p99_latency_ms:>10.1f} "
            f"{r.accuracy_score:>6.2f} {r.tokens_per_second:>8.1f} {ok:>8}"
        )
    print("=" * 90)

    # Recommendation
    passing = [r for r in results if r.meets_edge_target]
    if passing:
        best = max(passing, key=lambda r: r.accuracy_score)
        print(f"\n✅ Khuyến nghị: {best.quant_level} — accuracy {best.accuracy_score:.2f}, "
              f"P95 latency {best.p95_latency_ms:.0f}ms, size {best.model_size_mb:.0f}MB")
    else:
        fastest = min(results, key=lambda r: r.p95_latency_ms)
        print(f"\n⚠️  Không quant level nào đạt ≤500ms trên thiết bị này. "
              f"Nhanh nhất: {fastest.quant_level} ({fastest.p95_latency_ms:.0f}ms P95)")


def save_report_json(results: List[QuantBenchmarkResult], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"[report] Lưu JSON: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C1: Export GGUF + Edge Benchmark")
    parser.add_argument("--model-path", default=FINETUNED_CKPT,
                        help="Đường dẫn HuggingFace checkpoint (sau SFT hoặc base)")
    parser.add_argument("--out-dir", default=GGUF_OUT_DIR)
    parser.add_argument("--quant-levels", nargs="+", default=QUANT_LEVELS,
                        choices=["Q4_0", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q8_0", "Q5_K_M"])
    parser.add_argument("--skip-convert", action="store_true",
                        help="Bỏ qua convert nếu đã có F16 GGUF")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Chỉ chạy benchmark (đã có GGUF sẵn)")
    parser.add_argument("--n-threads", type=int, default=None,
                        help="Số CPU threads (mặc định: all cores)")
    parser.add_argument("--report-json", default="reports/edge_benchmark.json")
    args = parser.parse_args()

    results: List[QuantBenchmarkResult] = []

    if not args.benchmark_only:
        # 1. Convert sang F16 GGUF
        f16_path = convert_to_gguf_f16(args.model_path, args.out_dir)

        # 2. Quantize từng level
        for ql in args.quant_levels:
            try:
                quantize_model(f16_path, ql, args.out_dir)
            except RuntimeError as e:
                print(f"[ERROR] {ql}: {e}")

    # 3. Benchmark
    for ql in args.quant_levels:
        gguf_path = Path(args.out_dir) / f"router-{ql.lower()}.gguf"
        if not gguf_path.exists():
            print(f"[skip] {gguf_path} không tồn tại, bỏ qua.")
            continue
        print(f"\n[benchmark] Đang đo {ql} ...")
        try:
            result = benchmark_gguf_model(
                str(gguf_path), ql, BENCHMARK_PROMPTS, args.n_threads
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Benchmark {ql}: {e}")

    if results:
        print_report(results)
        save_report_json(results, args.report_json)


if __name__ == "__main__":
    main()