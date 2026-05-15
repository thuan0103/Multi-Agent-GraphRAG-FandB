import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

import sys, os, time, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import IntentClassifier

TEST_SAMPLES = [
    # order (25 VI + 10 EN + 10 KO = 45)
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
    # Korean - order
    ("아이스 아메리카노 한 잔 주세요", "order"),
    ("카페라떼 두 잔이랑 크루아상 하나 추가해 주세요", "order"),
    ("주문 취소해 주세요", "order"),
    ("라지 사이즈로 바꿔 주실 수 있어요?", "order"),
    ("치즈케이크 하나 더 주문할게요", "order"),
    ("그린티 라떼 두 개 포장이요", "order"),
    ("샌드위치 빼 주세요", "order"),
    ("지금까지 주문한 거 계산해 주세요", "order"),
    ("에스프레소 샷 추가해 주세요", "order"),
    ("카라멜 마키아토 한 잔 주세요", "order"),

    # consultant (25 VI + 10 EN + 10 KO = 45)
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
    # Korean - consultant
    ("더운 날씨에 뭐가 좋을까요?", "consultant"),
    ("커피 못 마시는데 뭐 마시면 좋을까요?", "consultant"),
    ("달콤한 거 추천해 주세요", "consultant"),
    ("카페인 없는 음료 뭐가 있어요?", "consultant"),
    ("졸린데 뭐 마시면 좋을까요?", "consultant"),
    ("유제품 알레르기가 있는데 뭐 마실 수 있어요?", "consultant"),
    ("여기서 제일 인기 있는 메뉴가 뭐예요?", "consultant"),
    ("다이어트 중인데 칼로리 낮은 거 추천해 주세요", "consultant"),
    ("친구랑 같이 왔는데 둘 다 좋아할 만한 거 추천해 주세요", "consultant"),
    ("처음 왔는데 뭐부터 시작하면 좋을까요?", "consultant"),

    # faq (25 VI + 10 EN + 10 KO = 45)
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
    # Korean - faq
    ("와이파이 비밀번호가 뭐예요?", "faq"),
    ("몇 시에 문 닫아요?", "faq"),
    ("여기 주소가 어떻게 돼요?", "faq"),
    ("카드 결제 되나요?", "faq"),
    ("주차 공간 있나요?", "faq"),
    ("배달도 되나요?", "faq"),
    ("콘센트 사용할 수 있나요?", "faq"),
    ("반려동물 입장 가능한가요?", "faq"),
    ("멤버십 혜택이 어떻게 되나요?", "faq"),
    ("학생 할인 되나요?", "faq"),

    # ignore (25 VI + 10 EN + 10 KO = 45)
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
    # Korean - ignore
    ("안녕하세요", "ignore"),
    ("감사합니다", "ignore"),
    ("ㅋㅋㅋ", "ignore"),
    ("네", "ignore"),
    ("알겠어요", "ignore"),
    ("잠깐만요", "ignore"),
    ("어...", "ignore"),
    ("ㅇㅇ", "ignore"),
    ("잘 있어요", "ignore"),
    ("좋아요 감사해요", "ignore"),
]

HARD_SAMPLES = [
    ("Có gì vừa ngon vừa rẻ không?", "consultant"),   
    ("Cho mình ly gì đó không caffeine", "consultant"),
    ("Mình muốn gọi cái gì đó phù hợp thời tiết này", "consultant"),
    ("Quán có gì ăn sáng không?", "consultant"),
    ("Tính tiền hết bao nhiêu?", "order"),
    ("Mình đặt 1 ly cà phê nhé?", "order"),
    ("뭔가 달달한 거 시켜도 될까요?", "consultant"),    
    ("오늘 날씨에 맞는 거 한 잔 주문할게요", "order"),   
    ("아무거나 맛있는 걸로 주세요", "consultant"),
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