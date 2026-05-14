import json
import random
from pathlib import Path


SEEDS = [
    # (input_template, subject, action, context)
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

SUBJECTS = [
    "anh", "em", "tôi", "mình", "chị", "bạn",
    "anh ấy", "chị ấy", "nhóm tụi anh", "gia đình tôi",
]

ITEMS = [
    "cà phê sữa đá", "bạc xỉu", "trà sữa", "matcha latte",
    "bánh croissant", "bánh mì sandwich", "nước ép cam",
    "cold brew", "cappuccino", "americano",
]

CONTEXTS = [
    "vào ngày mai lúc 7h",
    "tối nay lúc 8 giờ",
    "lúc 3 giờ chiều nay",
    "cuối tuần này",
    "cho 4 người",
    "cho 10 người",
    "cho buổi họp sáng mai",
    "nhé em",
    "giúp mình với",
    "",  # không có context
    "ngày 20 tháng 12",
    "dịp kỷ niệm",
    "hôm nay",
    "ngay bây giờ",
    "lúc 12h trưa",
]


def generate_samples(n: int = 3000) -> list:
    random.seed(42)
    samples = []
    seen = set()

    attempts = 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        tmpl_input, tmpl_sub, tmpl_action, tmpl_ctx = random.choice(SEEDS)
        sub = random.choice(SUBJECTS)
        item = random.choice(ITEMS)
        ctx = random.choice(CONTEXTS)

        inp = tmpl_input.format(sub=sub, item=item, ctx=ctx).strip()
        action = tmpl_action.format(sub=sub, item=item, ctx=ctx).strip()
        subject = tmpl_sub.format(sub=sub, item=item, ctx=ctx).strip()
        context = ctx

        key = inp
        if key in seen:
            continue
        seen.add(key)

        samples.append({
            "input": inp,
            "subject": subject,
            "action": action,
            "context": context,
        })

    return samples


def split_and_save(samples: list, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "intent_train.jsonl": samples[:train_end],
        "intent_val.jsonl": samples[train_end:val_end],
        "intent_test.jsonl": samples[val_end:],
    }

    for fname, data in splits.items():
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[data] {fname}: {len(data)} samples → {path}")


if __name__ == "__main__":
    samples = generate_samples(n=3000)
    split_and_save(samples, "data/intent")
    print(f"[data] Tổng: {len(samples)} samples")