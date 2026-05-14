"""
A1.1 + B2.4 — Router Accuracy & Latency Benchmark
Chạy: python scripts/benchmark_router.py

Output:
  - Accuracy tổng, per-class F1, confusion matrix
  - Latency p50/p90/p99 (ms)
  - Kết quả lưu vào data/benchmark_results.json
"""

import sys, os, time, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import IntentClassifier

# ── Test set (100 samples, 25/intent) ─────────────────────────────────
TEST_SAMPLES = [
    # order (25)
    ("Cho anh 1 ly cà phê sữa đá", "order"),
    ("Em ơi tính tiền đi", "order"),
    ("Thêm 1 cái bánh croissant nữa nhé", "order"),
    ("Gọi 2 ly trà đào size M", "order"),
    ("Tôi muốn đặt 1 sinh tố bơ", "order"),
    ("Bỏ ly matcha latte ra khỏi đơn", "order"),
    ("Cho tôi 1 americano không đường", "order"),
    ("Order thêm 1 phần khoai tây chiên", "order"),
    ("Đổi ly latte sang size L được không?", "order"),
    ("Cho mình 2 cà phê muối size M", "order"),
    ("Give me an iced latte please", "order"),
    ("I'll take 2 cappuccinos", "order"),
    ("Can I get a cold brew to go?", "order"),
    ("Add one cheesecake to my order", "order"),
    ("Remove the sandwich from my cart", "order"),
    ("Tôi muốn 3 ly trà sữa taro cho cả nhóm", "order"),
    ("Một phần bánh tiramisu cho tôi", "order"),
    ("Hủy đơn hiện tại đi em", "order"),
    ("Lấy thêm 1 ly nước ép cam", "order"),
    ("Cho anh 1 flat white và 1 mocha", "order"),
    ("Em tính xem tổng bao nhiêu tiền?", "order"),
    ("I want to order one espresso", "order"),
    ("Please add a brownie to my order", "order"),
    ("Đặt 1 bạc xỉu và 1 bánh mì thịt", "order"),
    ("Mình lấy combo sáng có không?", "order"),

    # consultant (25)
    ("Có gì ngon không em?", "consultant"),
    ("Gợi ý cho mình uống gì mùa hè đi", "consultant"),
    ("Tôi không uống được cà phê thì uống gì?", "consultant"),
    ("Món nào phù hợp người ăn kiêng?", "consultant"),
    ("Đang buồn ngủ nên uống gì cho tỉnh?", "consultant"),
    ("Bạn bè rủ nhau đến quán gọi gì hợp?", "consultant"),
    ("Tôi thích vị ngọt béo nên dùng gì?", "consultant"),
    ("What do you recommend for a hot day?", "consultant"),
    ("Tư vấn cho mình ly nào ngon nhất quán?", "consultant"),
    ("Trời lạnh uống gì ấm bụng?", "consultant"),
    ("Mình đang bầu có uống gì được không?", "consultant"),
    ("Something light and refreshing please", "consultant"),
    ("Gợi ý đồ ăn kèm với cà phê đi", "consultant"),
    ("Món nào phù hợp với buổi chiều làm việc?", "consultant"),
    ("Tôi chưa uống cà phê bao giờ nên bắt đầu từ đâu?", "consultant"),
    ("Which coffee is the least bitter?", "consultant"),
    ("Có gì phù hợp cho người không dùng sữa bò không?", "consultant"),
    ("Muốn thử cái gì đó mới lạ đi", "consultant"),
    ("Ngân sách 50k thì uống được gì?", "consultant"),
    ("Hai người bạn muốn uống gì khác nhau thì sao?", "consultant"),
    ("Em tư vấn cho mình loại trà ngon nhất đi", "consultant"),
    ("What's popular here?", "consultant"),
    ("Có gì vừa ngon vừa bổ không?", "consultant"),
    ("Tôi dị ứng sữa nên uống gì?", "consultant"),
    ("Recommend something with no caffeine", "consultant"),

    # faq (25)
    ("Wifi tên gì mật khẩu là gì?", "faq"),
    ("Quán mở cửa lúc mấy giờ?", "faq"),
    ("Quán ở địa chỉ nào?", "faq"),
    ("Thanh toán bằng gì?", "faq"),
    ("Có chỗ đậu xe không?", "faq"),
    ("What time does the shop close?", "faq"),
    ("Do you have free WiFi?", "faq"),
    ("Có giao hàng không?", "faq"),
    ("Phí ship là bao nhiêu?", "faq"),
    ("Có chương trình thành viên không?", "faq"),
    ("Có nhận thẻ ngân hàng không?", "faq"),
    ("Ngồi tối đa bao lâu?", "faq"),
    ("Có cho mang thú cưng vào không?", "faq"),
    ("Số điện thoại quán là bao nhiêu?", "faq"),
    ("Happy Hour mấy giờ?", "faq"),
    ("Quán có mở ngày Tết không?", "faq"),
    ("Có phòng riêng không?", "faq"),
    ("Where are you located?", "faq"),
    ("Do you accept credit cards?", "faq"),
    ("Is there parking available?", "faq"),
    ("Có ổ cắm điện không?", "faq"),
    ("Ly size L bao nhiêu ml?", "faq"),
    ("Có giảm giá sinh viên không?", "faq"),
    ("Làm thẻ thành viên ở đâu?", "faq"),
    ("Quán có cho hút thuốc không?", "faq"),

    # ignore (25)
    ("Ừm...", "ignore"),
    ("Hello", "ignore"),
    ("haha", "ignore"),
    ("ok", "ignore"),
    ("thanks", "ignore"),
    ("bye", "ignore"),
    ("...", "ignore"),
    ("xin chào", "ignore"),
    ("hi there", "ignore"),
    ("good morning", "ignore"),
    ("cảm ơn bạn", "ignore"),
    ("tạm biệt", "ignore"),
    ("👍", "ignore"),
    ("🙂", "ignore"),
    ("test test", "ignore"),
    ("asdfgh", "ignore"),
    ("1234", "ignore"),
    ("ok cảm ơn", "ignore"),
    ("yeah sure", "ignore"),
    ("hmm", "ignore"),
    ("noted", "ignore"),
    ("alright", "ignore"),
    ("oke bạn", "ignore"),
    ("k", "ignore"),
    ("thks", "ignore"),
]

HARD_SAMPLES = [
    ("Có gì vừa ngon vừa rẻ không?", "consultant"),   # ambiguous order vs consultant
    ("Cho mình ly gì đó không caffeine", "consultant"),
    ("Mình muốn gọi cái gì đó phù hợp thời tiết này", "consultant"),
    ("Quán có gì ăn sáng không?", "consultant"),
    ("Tính tiền hết bao nhiêu?", "order"),
    ("Mình đặt 1 ly cà phê nhé?", "order"),
]


def run_benchmark():
    print("=" * 60)
    print("  ROUTER BENCHMARK — MAS-LLM-2026-FINAL")
    print("=" * 60)

    clf = IntentClassifier()
    print("\n► Loading router model...")
    clf.load()
    print("✓ Model loaded\n")

    intents = ["order", "consultant", "faq", "ignore"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    conf_matrix = defaultdict(lambda: defaultdict(int))
    latencies = []
    errors = []

    print(f"► Running {len(TEST_SAMPLES)} test samples...")
    for i, (text, expected) in enumerate(TEST_SAMPLES):
        result = clf.classify(text)
        predicted = result["action"]
        lat = result["latency_ms"]
        latencies.append(lat)
        conf_matrix[expected][predicted] += 1
        if predicted == expected:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1
            errors.append({"text": text, "expected": expected, "predicted": predicted, "latency_ms": lat})

    # ── Metrics ──────────────────────────────────────────────────────
    total = len(TEST_SAMPLES)
    correct = sum(tp.values())
    accuracy = correct / total

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p90 = latencies[int(len(latencies) * 0.90)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = sum(latencies) / len(latencies)

    print(f"\n{'─'*60}")
    print(f"  Overall Accuracy: {accuracy:.1%}  ({correct}/{total})")
    status = "✅ XUẤT SẮC" if accuracy >= 0.92 else ("⚠️ ĐẠT" if accuracy >= 0.85 else "❌ CHƯA ĐẠT")
    print(f"  Status: {status}  (target ≥ 92%)")
    print(f"{'─'*60}")

    print("\n  Per-class F1 Score:")
    for intent in intents:
        p = tp[intent] / (tp[intent] + fp[intent]) if (tp[intent] + fp[intent]) > 0 else 0
        r = tp[intent] / (tp[intent] + fn[intent]) if (tp[intent] + fn[intent]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"    {intent:12s}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}  (TP={tp[intent]})")

    print(f"\n  Confusion Matrix (row=expected, col=predicted):")
    header = f"{'':12s}" + "".join(f"{i:12s}" for i in intents)
    print(f"    {header}")
    for expected in intents:
        row = f"    {expected:12s}" + "".join(f"{conf_matrix[expected][p]:12d}" for p in intents)
        print(row)

    print(f"\n  Latency (ms):")
    print(f"    avg={avg:.1f}  p50={p50:.1f}  p90={p90:.1f}  p99={p99:.1f}")
    lat_status = "✅ XUẤT SẮC" if p90 <= 100 else ("⚠️ ĐẠT" if p90 <= 200 else "❌ CHẬM")
    print(f"    Status: {lat_status}  (target p90 ≤ 200ms)")

    if errors:
        print(f"\n  Misclassified ({len(errors)} cases):")
        for e in errors[:10]:
            print(f"    [{e['expected']} → {e['predicted']}] \"{e['text'][:50]}\"")
        if len(errors) > 10:
            print(f"    ... và {len(errors)-10} cases khác")

    # ── Hard samples ──────────────────────────────────────────────────
    print(f"\n► Hard samples ({len(HARD_SAMPLES)} cases):")
    hard_correct = 0
    for text, expected in HARD_SAMPLES:
        result = clf.classify(text)
        predicted = result["action"]
        ok = predicted == expected
        hard_correct += ok
        mark = "✓" if ok else "✗"
        print(f"    [{mark}] \"{text[:45]}\" → {predicted} (expected {expected})")
    hard_acc = hard_correct / len(HARD_SAMPLES)
    print(f"  Hard accuracy: {hard_acc:.1%}  (target ≥ 75%)")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "latency": {"avg_ms": avg, "p50_ms": p50, "p90_ms": p90, "p99_ms": p99},
        "hard_accuracy": hard_acc,
        "per_class": {
            intent: {
                "tp": tp[intent], "fp": fp[intent], "fn": fn[intent],
            } for intent in intents
        },
        "errors": errors,
    }
    out = Path("data/benchmark_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Results saved → {out}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
