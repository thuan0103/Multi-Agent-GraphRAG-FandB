import json
import random
from pathlib import Path

SEEDS_VI = [
    ("Cho {sub} đặt bàn {ctx}", "{sub}", "đặt bàn", "{ctx}"),
    ("Cho {sub} đặt bàn tiệc sinh nhật {ctx}", "{sub}", "đặt bàn tiệc sinh nhật", "{ctx}"),
    ("Cho {sub} đặt bàn tiệc công ty {ctx}", "{sub}", "đặt bàn tiệc công ty", "{ctx}"),
    ("{sub} muốn đặt {item} {ctx}", "{sub}", "đặt {item}", "{ctx}"),
    ("{sub} order {item} {ctx}", "{sub}", "order {item}", "{ctx}"),
    ("Thêm {item} vào đơn {ctx}", "khách", "thêm {item}", "{ctx}"),
    ("Hủy order {ctx}", "khách", "hủy order", "{ctx}"),
    ("Cho {sub} hỏi giờ mở cửa {ctx}", "{sub}", "hỏi giờ mở cửa", "{ctx}"),
    ("{sub} hỏi có wifi không {ctx}", "{sub}", "hỏi wifi", "{ctx}"),
    ("{sub} muốn tư vấn món uống {ctx}", "{sub}", "tư vấn món uống", "{ctx}"),
    ("{sub} muốn tư vấn món ăn phù hợp {ctx}", "{sub}", "tư vấn món ăn", "{ctx}"),
    ("Thanh toán bằng thẻ được không {ctx}", "khách", "hỏi thanh toán bằng thẻ", "{ctx}"),
    ("Quán có nhận thẻ không {ctx}", "khách", "hỏi nhận thẻ", "{ctx}"),
    ("{sub} hỏi về khuyến mãi {ctx}", "{sub}", "hỏi khuyến mãi", "{ctx}"),
    ("Cho {sub} xem menu {ctx}", "{sub}", "xem menu", "{ctx}"),
    ("{sub} muốn đổi món {ctx}", "{sub}", "đổi món", "{ctx}"),
    ("Tính tiền giúp {sub} {ctx}", "{sub}", "tính tiền", "{ctx}"),
    ("{sub} muốn gọi thêm {item} {ctx}", "{sub}", "gọi thêm {item}", "{ctx}"),
    ("{sub} đặt chỗ ngồi view đẹp {ctx}", "{sub}", "đặt chỗ ngồi view đẹp", "{ctx}"),
    ("Cho {sub} 1 ly {item} không đường {ctx}", "{sub}", "gọi {item} không đường", "{ctx}"),
]

SUBJECTS_VI = [
    "anh", "em", "tôi", "mình", "chị", "bạn",
    "anh ấy", "chị ấy", "nhóm tụi anh", "gia đình tôi",
]

ITEMS_VI = [
    "cà phê sữa đá", "bạc xỉu", "trà sữa", "matcha latte",
    "bánh croissant", "bánh mì sandwich", "nước ép cam",
    "cold brew", "cappuccino", "americano",
]

CONTEXTS_VI = [
    "vào ngày mai lúc 7h", "tối nay lúc 8 giờ", "lúc 3 giờ chiều nay",
    "cuối tuần này", "cho 4 người", "cho 10 người",
    "cho buổi họp sáng mai", "nhé em", "giúp mình với",
    "", "ngày 20 tháng 12", "dịp kỷ niệm", "hôm nay",
    "ngay bây giờ", "lúc 12h trưa",
]

SEEDS_EN = [
    ("I'd like to order {item} {ctx}", "I", "order {item}", "{ctx}"),
    ("Can I book a table {ctx}", "I", "book a table", "{ctx}"),
    ("I want to order {item} {ctx}", "I", "order {item}", "{ctx}"),
    ("Please add {item} to my order {ctx}", "I", "add {item}", "{ctx}"),
    ("Can I cancel my order {ctx}", "I", "cancel order", "{ctx}"),
    ("What are your opening hours {ctx}", "customer", "ask opening hours", "{ctx}"),
    ("Do you have WiFi {ctx}", "customer", "ask WiFi", "{ctx}"),
    ("Can you recommend a drink {ctx}", "I", "ask for drink recommendation", "{ctx}"),
    ("Can I pay by card {ctx}", "I", "ask payment by card", "{ctx}"),
    ("Can I see the menu {ctx}", "I", "view menu", "{ctx}"),
    ("I'd like to change my order {ctx}", "I", "change order", "{ctx}"),
    ("How much is the total {ctx}", "I", "ask total price", "{ctx}"),
    ("I'd like to add {item} {ctx}", "I", "add {item}", "{ctx}"),
    ("Can I reserve a table {ctx}", "I", "reserve a table", "{ctx}"),
    ("Can I get {item} without sugar {ctx}", "I", "order {item} without sugar", "{ctx}"),
    ("Do you have any promotions {ctx}", "customer", "ask promotions", "{ctx}"),
    ("I want to cancel my reservation {ctx}", "I", "cancel reservation", "{ctx}"),
    ("{sub} would like to order {item} {ctx}", "{sub}", "order {item}", "{ctx}"),
    ("Could I have {item} {ctx}", "I", "order {item}", "{ctx}"),
    ("I need help choosing a drink {ctx}", "I", "ask for drink recommendation", "{ctx}"),
]

SUBJECTS_EN = ["I", "we", "my friend", "my family", "my group"]

ITEMS_EN = [
    "iced coffee", "americano", "matcha latte", "milk tea", "croissant",
    "sandwich", "orange juice", "cold brew", "cappuccino", "iced latte",
]

CONTEXTS_EN = [
    "for tomorrow morning at 7am", "tonight at 8pm", "this afternoon",
    "this weekend", "for 4 people", "for 10 people",
    "for a morning meeting tomorrow", "please", "",
    "on December 20th", "for our anniversary", "today",
    "right now", "at noon", "for a birthday party",
]

SEEDS_KO = [
    ("{item} 주문하고 싶어요 {ctx}", "저", "{item} 주문", "{ctx}"),
    ("테이블 예약하고 싶어요 {ctx}", "저", "테이블 예약", "{ctx}"),
    ("{item} 추가해 주세요 {ctx}", "저", "{item} 추가", "{ctx}"),
    ("주문 취소할게요 {ctx}", "저", "주문 취소", "{ctx}"),
    ("영업 시간이 어떻게 되나요 {ctx}", "손님", "영업 시간 문의", "{ctx}"),
    ("와이파이 있나요 {ctx}", "손님", "와이파이 문의", "{ctx}"),
    ("음료 추천해 주세요 {ctx}", "저", "음료 추천 요청", "{ctx}"),
    ("카드 결제 되나요 {ctx}", "손님", "카드 결제 문의", "{ctx}"),
    ("메뉴 보여 주세요 {ctx}", "저", "메뉴 확인", "{ctx}"),
    ("주문 변경하고 싶어요 {ctx}", "저", "주문 변경", "{ctx}"),
    ("계산서 부탁드려요 {ctx}", "저", "계산 요청", "{ctx}"),
    ("{item} 한 잔 주세요 {ctx}", "저", "{item} 주문", "{ctx}"),
    ("생일 파티 예약하고 싶어요 {ctx}", "저", "생일 파티 예약", "{ctx}"),
    ("단체 예약 가능한가요 {ctx}", "손님", "단체 예약 문의", "{ctx}"),
    ("{item} 설탕 없이 주세요 {ctx}", "저", "{item} 설탕 없이 주문", "{ctx}"),
    ("할인 행사 있나요 {ctx}", "손님", "할인 행사 문의", "{ctx}"),
    ("{sub} {item} 주문하려고 하는데요 {ctx}", "{sub}", "{item} 주문", "{ctx}"),
    ("예약 취소하고 싶어요 {ctx}", "저", "예약 취소", "{ctx}"),
    ("{item} 주세요 {ctx}", "저", "{item} 주문", "{ctx}"),
    ("자리 예약 부탁드려요 {ctx}", "저", "자리 예약", "{ctx}"),
]

SUBJECTS_KO = ["저", "저희", "제 친구", "우리 가족", "저희 팀"]

ITEMS_KO = [
    "아이스 아메리카노", "카페라테", "녹차라테", "밀크티", "크루아상",
    "샌드위치", "오렌지 주스", "콜드브루", "카푸치노", "바닐라 라테",
]

CONTEXTS_KO = [
    "내일 아침 7시에", "오늘 저녁 8시에", "오늘 오후에",
    "이번 주말에", "4명이서", "10명이서",
    "내일 아침 회의 때", "부탁드려요", "",
    "12월 20일에", "기념일에", "오늘",
    "지금 바로", "점심에", "생일 파티로",
]

def _gen_samples(seeds, subjects, items, contexts, n: int, rng: random.Random) -> list:
    samples = []
    seen = set()
    attempts = 0

    while len(samples) < n and attempts < n * 30:
        attempts += 1
        tmpl_inp, tmpl_sub, tmpl_act, tmpl_ctx = rng.choice(seeds)
        sub  = rng.choice(subjects)
        item = rng.choice(items)
        ctx  = rng.choice(contexts)

        inp     = tmpl_inp.format(sub=sub, item=item, ctx=ctx).strip()
        action  = tmpl_act.format(sub=sub, item=item, ctx=ctx).strip()
        subject = tmpl_sub.format(sub=sub, item=item, ctx=ctx).strip()

        if inp in seen:
            continue
        seen.add(inp)

        samples.append({
            "input":   inp,
            "subject": subject,
            "action":  action,
            "context": ctx,
            "lang":    None, 
        })

    return samples


def generate_samples(n_per_lang: int = 1000, seed: int = 42) -> list:
    rng = random.Random(seed)

    vi = _gen_samples(SEEDS_VI, SUBJECTS_VI, ITEMS_VI, CONTEXTS_VI, n_per_lang, rng)
    en = _gen_samples(SEEDS_EN, SUBJECTS_EN, ITEMS_EN, CONTEXTS_EN, n_per_lang, rng)
    ko = _gen_samples(SEEDS_KO, SUBJECTS_KO, ITEMS_KO, CONTEXTS_KO, n_per_lang, rng)

    for s in vi: s["lang"] = "vi"
    for s in en: s["lang"] = "en"
    for s in ko: s["lang"] = "ko"

    all_samples = vi + en + ko
    rng.shuffle(all_samples)
    return all_samples


def split_and_save(samples: list, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    splits = {
        "intent_train.jsonl": samples[:train_end],
        "intent_val.jsonl":   samples[train_end:val_end],
        "intent_test.jsonl":  samples[val_end:],
    }

    for fname, data in splits.items():
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                row = {k: v for k, v in item.items() if k != "lang"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        lang_counts = {}
        for s in data:
            lang_counts[s.get("lang", "?")] = lang_counts.get(s.get("lang", "?"), 0) + 1
        print(f"[data] {fname}: {len(data)} samples  {lang_counts}")


if __name__ == "__main__":
    samples = generate_samples(n_per_lang=1000)
    split_and_save(samples, "data/intent")
    print(f"\n[data] Total: {len(samples)} samples (vi+en+ko)")
