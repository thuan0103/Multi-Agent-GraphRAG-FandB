"""
B2.4 — Benchmark TTFT & Throughput
Folder: services/llm_serving/benchmark.py

Chạy: python benchmark.py --url http://localhost:8000 --concurrency 5
"""
import asyncio
import argparse
import json
import statistics
import time
import httpx

TEST_QUERIES = [
    "Giờ mở cửa?",
    "Có wifi không?",
    "Menu có những gì ngon?",
    "Giá cà phê sữa đá?",
    "Cho tôi xem danh sách món ăn",
    "Có chỗ đỗ xe không?",
    "Nhận đặt bàn online không?",
    "Có món chay không?",
    "Giá trà sữa?",
    "Thứ 7 mở cửa không?",
]


async def measure_ttft(client: httpx.AsyncClient, url: str, query: str) -> float:
    """Measure Time-To-First-Token via SSE."""
    t0 = time.perf_counter()
    first_token_time = None
    async with client.stream(
        "POST",
        f"{url}/chat/stream",
        json={"query": query, "stream": True},
        timeout=30,
    ) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    if payload.get("type") == "token" and first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                        break
                except Exception:
                    pass
    return first_token_time or -1


async def measure_total(client: httpx.AsyncClient, url: str, query: str) -> float:
    t0 = time.perf_counter()
    await client.post(f"{url}/chat", json={"query": query, "stream": False}, timeout=30)
    return time.perf_counter() - t0


async def run_benchmark(url: str, concurrency: int, n_requests: int):
    print(f"\n=== Benchmark: url={url}, concurrency={concurrency}, n={n_requests} ===\n")
    async with httpx.AsyncClient() as client:
        # TTFT
        ttft_results = []
        for i in range(min(10, n_requests)):
            q = TEST_QUERIES[i % len(TEST_QUERIES)]
            t = await measure_ttft(client, url, q)
            ttft_results.append(t)
            print(f"  TTFT [{i+1}]: {t*1000:.1f}ms  query={q[:30]}")

        # Total latency (concurrent)
        total_results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_call(q):
            async with semaphore:
                return await measure_total(client, url, q)

        queries = [TEST_QUERIES[i % len(TEST_QUERIES)] for i in range(n_requests)]
        t_start = time.perf_counter()
        total_results = await asyncio.gather(*[bounded_call(q) for q in queries])
        wall_time = time.perf_counter() - t_start

    print("\n── TTFT ──────────────────────────────────")
    print(f"  Mean:   {statistics.mean(ttft_results)*1000:.1f} ms")
    print(f"  Median: {statistics.median(ttft_results)*1000:.1f} ms")
    print(f"  P95:    {sorted(ttft_results)[int(len(ttft_results)*0.95)]*1000:.1f} ms")
    print(f"  {'✅ PASS (≤200ms)' if statistics.mean(ttft_results) <= 0.2 else '❌ FAIL'}")

    print("\n── Total Latency ──────────────────────────")
    print(f"  Mean:   {statistics.mean(total_results)*1000:.1f} ms")
    print(f"  Median: {statistics.median(total_results)*1000:.1f} ms")
    rating = "✅ Xuất sắc (≤800ms)" if statistics.mean(total_results) <= 0.8 else \
             "✅ Đạt (≤1500ms)" if statistics.mean(total_results) <= 1.5 else "❌ Chưa đạt"
    print(f"  {rating}")

    print("\n── Throughput ─────────────────────────────")
    print(f"  {n_requests} requests in {wall_time:.2f}s")
    print(f"  RPS: {n_requests / wall_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url, args.concurrency, args.n))
