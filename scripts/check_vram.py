"""
Đo VRAM thực tế khi load từng model.
Chạy: python scripts/check_vram.py
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def vram_info(label: str):
    if not torch.cuda.is_available():
        print(f"  [{label}] Không có CUDA — đang chạy CPU")
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  [{label}] allocated={alloc:.2f} GB | reserved={reserv:.2f} GB | total={total:.2f} GB")


def load_and_measure(model_path: str, dtype=torch.bfloat16):
    print(f"\n{'='*55}")
    print(f"  Model: {model_path}")
    print(f"  dtype: {dtype}")
    print(f"{'='*55}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    vram_info("trước load")
    t0 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Load time: {load_ms:.0f} ms")
    vram_info("sau load")

    # Chạy 5 lần inference để đo latency
    test_prompts = [
        "Cho tôi 1 ly cà phê sữa đá",
        "Wifi tên gì?",
        "Có gì ngon không?",
        "Hello",
        "카드 결제 되나요?",
    ]
    latencies = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        t1 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        latencies.append((time.perf_counter() - t1) * 1000)

    print(f"\n  Latency (forward pass, 5 samples):")
    for p, ms in zip(test_prompts, latencies):
        print(f"    {ms:6.1f} ms  |  {p}")
    print(f"  avg={sum(latencies)/len(latencies):.1f} ms | p90={sorted(latencies)[int(len(latencies)*0.9)]:.1f} ms")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak VRAM (kể cả forward pass): {peak:.2f} GB")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vram_info("sau del")


if __name__ == "__main__":
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory/1024**3:.1f} GB")

    # 1. router-merged (bfloat16) — active router
    load_and_measure("models/router-merged", dtype=torch.bfloat16)

    # 2. router-awq (AWQ int4) — SGLang Part B
    load_and_measure("models/router-awq", dtype=torch.float16)
