"""
C3: TTS Preprocessor — normalize text trước khi gửi TTS.
Đảm bảo output tự nhiên khi phát âm.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Normalization rules
# ---------------------------------------------------------------------------

class TTSPreprocessor:
    """
    Chuẩn hóa text cho TTS tiếng Việt trong môi trường F&B.

    Thứ tự áp dụng rules:
    1. Giá tiền VNĐ
    2. Đơn vị size
    3. Từ viết tắt / ký hiệu đặc biệt
    4. Số điện thoại / URL
    5. Ký tự không phát âm được
    """

    # -----------------------------------------------------------------------
    # Price rules: 49.000đ → 49k, 1.500.000đ → 1 triệu 5, ...
    # -----------------------------------------------------------------------

    def _normalize_price(self, text: str) -> str:
        # 49.000đ / 49,000đ / 49000đ → 49k
        def replace_price(m):
            raw = m.group(1).replace(".", "").replace(",", "")
            try:
                val = int(raw)
            except ValueError:
                return m.group(0)
            if val >= 1_000_000:
                millions = val // 1_000_000
                remainder = (val % 1_000_000) // 1_000
                if remainder:
                    return f"{millions} triệu {remainder} nghìn"
                return f"{millions} triệu"
            if val >= 1_000:
                k = val // 1_000
                return f"{k}k"
            return str(val)

        # Match: số (có thể có . hoặc ,) + đ hoặc đồng hoặc VND/vnđ
        text = re.sub(
            r"([\d][0-9.,]*)\s*(?:đ|đồng|VNĐ|vnđ|VND)",
            replace_price,
            text,
            flags=re.IGNORECASE,
        )
        return text

    # -----------------------------------------------------------------------
    # Size rules
    # -----------------------------------------------------------------------

    SIZE_MAP = {
        r"\bS\b": "size ét",
        r"\bM\b": "size mờ",
        r"\bL\b": "size lờ",
        r"\bXL\b": "size ếch lờ",
        r"\bXXL\b": "size hai ếch lờ",
    }

    def _normalize_size(self, text: str) -> str:
        for pattern, replacement in self.SIZE_MAP.items():
            text = re.sub(pattern, replacement, text)
        return text

    # -----------------------------------------------------------------------
    # Abbreviation rules
    # -----------------------------------------------------------------------

    ABBREV_MAP = {
        r"\bVAT\b": "thuế",
        r"\bvat\b": "thuế",
        r"\bGST\b": "thuế",
        r"\bTP\.HCM\b": "Thành phố Hồ Chí Minh",
        r"\bHCM\b": "Hồ Chí Minh",
        r"\bHN\b": "Hà Nội",
        r"\bTT\b": "trân châu",
        r"\bko\b": "không",
        r"\bk\b(?=\s|$)": "không",   # "k" standalone
        r"\bOK\b": "ô kê",
        r"\bok\b": "ô kê",
        r"\bĐT\b": "điện thoại",
        r"\bĐH\b": "đại học",
        r"\bFB\b": "ép bì",
        r"\bĐC\b": "địa chỉ",
        r"\bBH\b": "bảo hành",
        r"\bKM\b": "khuyến mãi",
        r"\bNV\b": "nhân viên",
        r"\bHĐ\b": "hóa đơn",
        r"\bSL\b": "số lượng",
    }

    def _normalize_abbreviations(self, text: str) -> str:
        for pattern, replacement in self.ABBREV_MAP.items():
            text = re.sub(pattern, replacement, text)
        return text

    # -----------------------------------------------------------------------
    # Number normalization
    # -----------------------------------------------------------------------

    def _normalize_numbers(self, text: str) -> str:
        # Số điện thoại: 0901234567 → để nguyên (TTS xử lý tốt)
        # Phần trăm: 10% → 10 phần trăm
        text = re.sub(r"(\d+)\s*%", r"\1 phần trăm", text)
        # Giờ: 7h, 7h30 → 7 giờ, 7 giờ 30
        text = re.sub(r"(\d+)h(\d+)", r"\1 giờ \2", text)
        text = re.sub(r"(\d+)h\b", r"\1 giờ", text)
        return text

    # -----------------------------------------------------------------------
    # Special characters
    # -----------------------------------------------------------------------

    SPECIAL_CHARS = {
        "&": "và",
        "@": "a còng",
        "#": "thăng",
        "*": "",
        "~": "",
        "|": "",
        "\\": "",
        "/": " hoặc ",    # A/B → A hoặc B
        "_": " ",
        "...": "…",       # normalize ellipsis
    }

    def _normalize_special_chars(self, text: str) -> str:
        for char, replacement in self.SPECIAL_CHARS.items():
            text = text.replace(char, replacement)
        # Xóa URL
        text = re.sub(r"https?://\S+", "", text)
        # Xóa email
        text = re.sub(r"\S+@\S+\.\S+", "", text)
        # Xóa ký tự không in được
        text = re.sub(r"[^\w\sÀ-ỹ.,!?:;()\-…]", "", text, flags=re.UNICODE)
        return text

    # -----------------------------------------------------------------------
    # Whitespace cleanup
    # -----------------------------------------------------------------------

    def _cleanup_whitespace(self, text: str) -> str:
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s([.,!?:;])", r"\1", text)
        return text.strip()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Áp dụng toàn bộ normalization pipeline."""
        text = self._normalize_price(text)
        text = self._normalize_size(text)
        text = self._normalize_abbreviations(text)
        text = self._normalize_numbers(text)
        text = self._normalize_special_chars(text)
        text = self._cleanup_whitespace(text)
        return text

    def __call__(self, text: str) -> str:
        return self.process(text)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_preprocessor = TTSPreprocessor()


def preprocess_for_tts(text: str) -> str:
    """Shorthand function — dùng singleton."""
    return _preprocessor.process(text)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        ("Giá là 49.000đ", "Giá là 49k"),
        ("Size M hoặc L", "size mờ hoặc size lờ"),
        ("Thuế VAT 10%", "thuế thuế 10 phần trăm"),   # VAT → thuế
        ("Mở cửa lúc 7h30", "Mở cửa lúc 7 giờ 30"),
        ("Giảm giá 20% hôm nay", "Giảm giá 20 phần trăm hôm nay"),
        ("Giá 1.500.000đ", "Giá 1 triệu 5 trăm nghìn"),
    ]
    preprocessor = TTSPreprocessor()
    print("TTS Preprocessor Test:")
    all_pass = True
    for inp, expected in cases:
        out = preprocessor.process(inp)
        ok = "✅" if expected.lower() in out.lower() or out == expected else "⚠️ "
        if ok == "⚠️ ":
            all_pass = False
        print(f"  {ok} IN : {inp}")
        print(f"     OUT: {out}")
        print(f"     EXP: {expected}")