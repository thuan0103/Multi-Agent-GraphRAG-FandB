"""
A1.3 — Merge LoRA adapter vào base model, sau đó export GGUF (Q4_K_M).

Bước 1: Merge → models/router-merged/
Bước 2: Convert sang GGUF F16 → models/gguf/router-f16.gguf
Bước 3: Quantize Q4_K_M → models/gguf/router-q4km.gguf

Chạy:
    conda run -n partA python training/merge_and_export.py
    conda run -n partA python training/merge_and_export.py --skip-merge   # nếu đã merge rồi
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL  = "models/Qwen2.5-1.5B"
ADAPTER     = "models/checkpoint-400"
MERGED_OUT  = "models/router-merged"
GGUF_DIR    = "models/gguf"


# ---------------------------------------------------------------------------
# Step 1: Merge LoRA vào base
# ---------------------------------------------------------------------------

def merge_lora():
    print("=" * 55)
    print("Step 1: Merge LoRA → base model")
    print("=" * 55)

    if Path(MERGED_OUT, "config.json").exists():
        print(f"[skip] Merged model đã tồn tại tại {MERGED_OUT}")
        return

    print(f"  Loading base model: {BASE_MODEL}")
    t0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    adapter_path = str(Path(ADAPTER).resolve().as_posix())
    print(f"  Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)

    print("  Merging weights (merge_and_unload)...")
    model = model.merge_and_unload()

    Path(MERGED_OUT).mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged model → {MERGED_OUT}")
    model.save_pretrained(MERGED_OUT)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_OUT)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s\n")


# ---------------------------------------------------------------------------
# Step 2 & 3: Convert + Quantize dùng llama.cpp Python package
# ---------------------------------------------------------------------------

def convert_and_quantize():
    print("=" * 55)
    print("Step 2: Convert HF → GGUF + Quantize Q4_K_M")
    print("=" * 55)

    Path(GGUF_DIR).mkdir(parents=True, exist_ok=True)
    f16_path  = Path(GGUF_DIR) / "router-f16.gguf"
    q4km_path = Path(GGUF_DIR) / "router-q4km.gguf"

    # Tìm convert_hf_to_gguf.py từ llama-cpp-python hoặc llama.cpp clone
    convert_script = _find_convert_script()

    if convert_script is None:
        print("  [INFO] Không tìm thấy convert_hf_to_gguf.py")
        print("  Để tạo GGUF, chạy một trong hai cách:")
        print()
        print("  Cách 1 — dùng llama.cpp clone:")
        print("    git clone https://github.com/ggml-org/llama.cpp llama.cpp")
        print(f"    python llama.cpp/convert_hf_to_gguf.py {MERGED_OUT} \\")
        print(f"           --outfile {f16_path} --outtype f16")
        print(f"    llama.cpp/build/bin/llama-quantize {f16_path} {q4km_path} Q4_K_M")
        print()
        print("  Cách 2 — pip install llama-cpp-python (chỉ Windows CPU):")
        print("    pip install llama-cpp-python")
        print("    python -c \"from llama_cpp import Llama\"  # verify")
        return

    # Bước 2: Convert F16
    if not f16_path.exists():
        print(f"  Converting {MERGED_OUT} → {f16_path}")
        result = subprocess.run(
            [sys.executable, str(convert_script),
             MERGED_OUT, "--outfile", str(f16_path), "--outtype", "f16"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [ERROR] Convert thất bại:\n{result.stderr[-500:]}")
            return
        size_mb = f16_path.stat().st_size / 1e6
        print(f"  F16 GGUF: {size_mb:.0f} MB\n")
    else:
        print(f"  [skip] F16 đã tồn tại: {f16_path}")

    # Bước 3: Quantize Q4_K_M
    if not q4km_path.exists():
        quantize_bin = _find_quantize_bin()
        if quantize_bin:
            print(f"  Quantizing → Q4_K_M")
            result = subprocess.run(
                [str(quantize_bin), str(f16_path), str(q4km_path), "Q4_K_M"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                size_mb = q4km_path.stat().st_size / 1e6
                print(f"  Q4_K_M GGUF: {size_mb:.0f} MB")
            else:
                print(f"  [ERROR] Quantize thất bại:\n{result.stderr[-300:]}")
        else:
            print("  [INFO] llama-quantize không tìm thấy.")
            print(f"  Chạy thủ công: llama-quantize {f16_path} {q4km_path} Q4_K_M")
    else:
        print(f"  [skip] Q4_K_M đã tồn tại: {q4km_path}")
        print(f"  Size: {q4km_path.stat().st_size / 1e6:.0f} MB")


def _find_convert_script() -> "Path | None":
    candidates = [
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path("llama.cpp/convert.py"),
    ]
    # Tìm trong llama-cpp-python package
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        for name in ["convert_hf_to_gguf.py", "convert.py"]:
            p = pkg_dir / name
            if p.exists():
                candidates.insert(0, p)
    except ImportError:
        pass

    for p in candidates:
        if p.exists():
            print(f"  Found convert script: {p}")
            return p
    return None


def _find_quantize_bin() -> "Path | None":
    candidates = [
        Path("llama.cpp/build/bin/llama-quantize"),
        Path("llama.cpp/build/bin/llama-quantize.exe"),
        Path("llama.cpp/quantize"),
        Path("llama.cpp/quantize.exe"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n" + "=" * 55)
    print("A1.3 Export Summary")
    print("=" * 55)
    items = [
        (MERGED_OUT + "/config.json",        "Merged model (HF format)"),
        (GGUF_DIR + "/router-f16.gguf",      "GGUF F16"),
        (GGUF_DIR + "/router-q4km.gguf",     "GGUF Q4_K_M (edge)"),
    ]
    for path, label in items:
        p = Path(path)
        if p.exists():
            size = p.stat().st_size / 1e6
            print(f"  ✅ {label:<30} {size:.0f} MB  → {p}")
        else:
            print(f"  ❌ {label:<30} không có")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf",  action="store_true")
    args = parser.parse_args()

    if not args.skip_merge:
        merge_lora()
    if not args.skip_gguf:
        convert_and_quantize()
    print_summary()
