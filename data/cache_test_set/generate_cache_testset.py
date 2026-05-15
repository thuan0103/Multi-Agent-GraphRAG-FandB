import json
import random
from pathlib import Path

random.seed(2026)

ACTION_GROUPS = [
    {
        "group_id": "g_dat_ban",
        "action_gt": "đặt bàn",
        "domain": "order",
        "templates": [
            "Cho {sub} đặt bàn {ctx}",
            "{sub} muốn đặt bàn {ctx}",
            "Đặt bàn cho {sub} {ctx} nhé",
            "Book bàn giúp {sub} {ctx}",
            "Mình muốn đặt chỗ {ctx}",
        ],
    },
    {
        "group_id": "g_dat_ban_sinh_nhat",
        "action_gt": "đặt bàn tiệc sinh nhật",
        "domain": "order",
        "templates": [
            "Cho {sub} đặt bàn tiệc sinh nhật {ctx}",
            "{sub} muốn book bàn tiệc sinh nhật {ctx}",
            "Đặt chỗ tổ chức sinh nhật {ctx}",
            "Mình cần đặt bàn sinh nhật {ctx}",
        ],
    },
    {
        "group_id": "g_xem_menu",
        "action_gt": "xem menu",
        "domain": "faq",
        "templates": [
            "Cho {sub} xem menu {ctx}",
            "Menu quán có gì {ctx}",
            "{sub} muốn xem thực đơn {ctx}",
            "Quán có những món gì {ctx}",
            "Show menu cho {sub} {ctx}",
        ],
    },
    {
        "group_id": "g_order_cafe",
        "action_gt": "order cà phê sữa đá",
        "domain": "order",
        "templates": [
            "{sub} order {item_cafe} {ctx}",
            "Cho {sub} 1 ly {item_cafe} {ctx}",
            "Gọi {item_cafe} cho {sub} {ctx}",
            "{sub} muốn uống {item_cafe} {ctx}",
        ],
    },
    {
        "group_id": "g_huy_order",
        "action_gt": "hủy order",
        "domain": "order",
        "templates": [
            "Hủy order {ctx}",
            "{sub} muốn hủy đơn {ctx}",
            "Cancel order giúp {sub} {ctx}",
            "Bỏ đơn vừa đặt {ctx}",
            "{sub} không muốn order nữa {ctx}",
        ],
    },
    {
        "group_id": "g_hoi_gio",
        "action_gt": "hỏi giờ mở cửa",
        "domain": "faq",
        "templates": [
            "Quán mở cửa mấy giờ {ctx}",
            "Giờ mở cửa của quán {ctx}",
            "{sub} hỏi quán đóng cửa lúc mấy giờ {ctx}",
            "Quán hoạt động đến bao giờ {ctx}",
        ],
    },
    {
        "group_id": "g_hoi_wifi",
        "action_gt": "hỏi wifi",
        "domain": "faq",
        "templates": [
            "Quán có wifi không {ctx}",
            "{sub} hỏi mật khẩu wifi {ctx}",
            "Cho {sub} xin pass wifi {ctx}",
            "Wifi quán tốt không {ctx}",
        ],
    },
    {
        "group_id": "g_tu_van",
        "action_gt": "tư vấn món uống",
        "domain": "faq",
        "templates": [
            "{sub} muốn tư vấn món uống {ctx}",
            "Gợi ý món uống cho {sub} {ctx}",
            "{sub} không biết chọn gì, tư vấn giúp {ctx}",
            "Nên uống gì {ctx}",
        ],
    },
    {
        "group_id": "g_thanh_toan",
        "action_gt": "hỏi thanh toán bằng thẻ",
        "domain": "faq",
        "templates": [
            "Thanh toán bằng thẻ được không {ctx}",
            "Quán nhận thẻ không {ctx}",
            "{sub} muốn trả bằng thẻ {ctx}",
            "Quán có POS không {ctx}",
        ],
    },
    {
        "group_id": "g_khuyen_mai",
        "action_gt": "hỏi khuyến mãi",
        "domain": "faq",
        "templates": [
            "{sub} hỏi về khuyến mãi {ctx}",
            "Quán có ưu đãi gì không {ctx}",
            "Chương trình giảm giá hôm nay {ctx}",
            "Coupon của quán {ctx}",
        ],
    },
]

SUBJECTS = ["anh", "em", "tôi", "mình", "chị"]
ITEM_CAFES = ["cà phê sữa đá", "bạc xỉu", "cold brew", "americano"]

CONTEXTS = [
    "vào ngày mai lúc 7h",
    "tối nay lúc 8 giờ",
    "cho 4 người",
    "cho 6 người",
    "cuối tuần này",
    "lúc 12h trưa",
    "hôm nay",
    "ngày 25 tháng 12",
    "dịp kỷ niệm",
    "cho buổi họp",
    "",
    "nhé em",
    "giúp mình với",
    "ngay bây giờ",
]


def generate() -> list:
    samples = []
    seen = set()

    for group in ACTION_GROUPS:
        templates = group["templates"]
        count = max(1, 500 // len(ACTION_GROUPS) + 5) 

        attempts = 0
        generated = 0
        while generated < count and attempts < count * 30:
            attempts += 1
            tmpl = random.choice(templates)
            sub = random.choice(SUBJECTS)
            ctx = random.choice(CONTEXTS)
            item_cafe = random.choice(ITEM_CAFES)

            inp = tmpl.format(sub=sub, ctx=ctx, item_cafe=item_cafe).strip()
            if inp in seen:
                continue
            seen.add(inp)

            samples.append({
                "input": inp,
                "action_gt": group["action_gt"],
                "group_id": group["group_id"],
                "domain": group["domain"],
            })
            generated += 1

    random.shuffle(samples)
    return samples[:500]


def main():
    out_dir = Path("data/cache_test_set")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cache_test_500.jsonl"

    samples = generate()
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    groups = Counter(s["group_id"] for s in samples)
    print(f"[data] Sinh {len(samples)} samples → {out_path}")
    for g, n in sorted(groups.items()):
        print(f"  {g}: {n}")


if __name__ == "__main__":
    main()