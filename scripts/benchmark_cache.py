import sys, os, json, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import SemanticCache, EmbeddingProvider, CacheEntry
from src.intent_extraction import IntentExtractor

def load_test_set(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

_TEMPLATES = {
    "đặt bàn":                  "Dạ để mình đặt bàn cho bạn nhé! Bạn muốn đặt cho mấy người?",
    "đặt bàn tiệc sinh nhật":   "Tuyệt! Quán có thể tổ chức tiệc sinh nhật. Bạn dự kiến bao nhiêu khách?",
    "xem menu":                 "Menu quán gồm: cà phê, trà, sinh tố, bánh ngọt. Bạn muốn xem phần nào?",
    "order cà phê sữa đá":      "Okay! Bạn muốn size S/M/L và ít đường hay không đường?",
    "hủy order":                "Mình đã hủy đơn cho bạn rồi ạ. Bạn có muốn đặt lại không?",
    "hỏi giờ mở cửa":           "Quán mở cửa từ 7:00 đến 22:00 tất cả các ngày trong tuần ạ.",
    "hỏi wifi":                 "Wifi: CafeXYZ_Guest | Mật khẩu: cafe2024",
    "tư vấn món uống":          "Thời tiết này mình gợi ý trà đào size M hoặc cold brew ạ.",
    "hỏi thanh toán bằng thẻ":  "Quán nhận thanh toán tiền mặt, QR MoMo, ZaloPay và thẻ ngân hàng ạ.",
    "hỏi khuyến mãi":           "Hiện quán có Happy Hour 14:00-16:00 giảm 20% tất cả đồ uống.",
}


def run_benchmark():
    print("=" * 60)
    print("  SEMANTIC CACHE HIT RATE BENCHMARK — C2.2")
    print("=" * 60)

    test_path = "data/cache_test_set/cache_test_500.jsonl"
    if not Path(test_path).exists():
        print(f"[ERROR] Test set not found: {test_path}")
        print("  Chạy: python data/cache_test_set/generate_cache_testset.py")
        sys.exit(1)

    samples = load_test_set(test_path)
    print(f"\n► Loaded {len(samples)} test samples")

    print("► Initializing embedding + cache...")
    embedder = EmbeddingProvider()
    cache = SemanticCache(embedder, threshold=0.92)
    extractor = IntentExtractor()
    extractor_mode = "rule-based fallback"
    if extractor._model.is_loaded():
        extractor_mode = "fine-tuned SLM"
    print(f"  Extractor: {extractor_mode}")
    print(f"  Cache threshold: 0.92")

    print("\n► Warming up cache (1 sample/group)...")
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group_id"]].append(s)

    warmup_count = 0
    for gid, group_samples in groups.items():
        warmup_sample = group_samples[0]
        extracted = extractor.extract(warmup_sample["input"])
        action_key = extracted.action
        template = _TEMPLATES.get(warmup_sample["action_gt"], warmup_sample["action_gt"])
        cache.put(action_key, template)
        warmup_count += 1

    print(f"  Warmed up {warmup_count} groups")

    print(f"\n► Querying {len(samples) - warmup_count} non-warmup samples...")

    total = 0
    hits = 0
    misses = 0
    latencies = []
    per_group_hits = defaultdict(int)
    per_group_total = defaultdict(int)
    misclassified = []

    for gid, group_samples in groups.items():
        for sample in group_samples[1:]:  
            t0 = time.perf_counter()
            extracted = extractor.extract(sample["input"])
            action_key = extracted.action
            entry = cache.query(action_key)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            total += 1
            per_group_total[gid] += 1

            if entry is not None:
                hits += 1
                per_group_hits[gid] += 1
            else:
                misses += 1
                misclassified.append({
                    "input": sample["input"],
                    "action_gt": sample["action_gt"],
                    "action_extracted": action_key,
                    "group_id": gid,
                })

    hit_rate = hits / total if total > 0 else 0
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
    p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0
    avg = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'─'*60}")
    print(f"  Overall Hit Rate: {hit_rate:.1%}  ({hits}/{total})")
    status = "XUẤT SẮC" if hit_rate >= 0.70 else ("⚠️ ĐẠT" if hit_rate >= 0.60 else "❌ CHƯA ĐẠT")
    print(f"  Status: {status}  (target ≥ 60%)")
    print(f"{'─'*60}")

    print(f"\n  Per-group Hit Rate:")
    for gid in sorted(per_group_total.keys()):
        gh = per_group_hits[gid]
        gt = per_group_total[gid]
        rate = gh / gt if gt > 0 else 0
        bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"    {gid:30s}  {bar}  {rate:.0%}  ({gh}/{gt})")

    print(f"\n  Cache Lookup Latency (ms):")
    print(f"    avg={avg:.1f}  p50={p50:.1f}  p90={p90:.1f}")
    lat_status = "✅" if p90 <= 50 else ("⚠️" if p90 <= 100 else "❌")
    print(f"    Status: {lat_status}  (target p90 ≤ 100ms)")

    if misclassified:
        print(f"\n  Cache Miss samples ({len(misclassified)}, first 8):")
        for m in misclassified[:8]:
            print(f"    [MISS] \"{m['input'][:50]}\"")
            print(f"           action_gt={m['action_gt']!r}")
            print(f"           extracted={m['action_extracted']!r}")

    print(f"\n► Upper-bound test: dùng action_gt trực tiếp làm cache key...")
    ub_hits = 0
    ub_total = 0
    cache2 = SemanticCache(embedder, threshold=0.92)
    for gid, group_samples in groups.items():
        template = _TEMPLATES.get(group_samples[0]["action_gt"], group_samples[0]["action_gt"])
        cache2.put(group_samples[0]["action_gt"], template)
    for gid, group_samples in groups.items():
        for sample in group_samples[1:]:
            entry = cache2.query(sample["action_gt"])
            ub_total += 1
            if entry is not None:
                ub_hits += 1
    ub_rate = ub_hits / ub_total if ub_total > 0 else 0
    print(f"  Upper-bound hit rate (action_gt): {ub_rate:.1%}  ({ub_hits}/{ub_total})")

    results = {
        "hit_rate": hit_rate,
        "hits": hits,
        "total": total,
        "warmup_groups": warmup_count,
        "latency": {"avg_ms": avg, "p50_ms": p50, "p90_ms": p90},
        "upper_bound_hit_rate": ub_rate,
        "extractor_mode": extractor_mode,
        "per_group": {
            gid: {"hits": per_group_hits[gid], "total": per_group_total[gid]}
            for gid in per_group_total
        },
    }
    out = Path("data/benchmark_cache_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Results saved -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
