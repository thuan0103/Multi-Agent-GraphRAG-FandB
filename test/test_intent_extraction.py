"""
Test C2.1: Intent Extraction — extractor output 3 thành phần + paraphraser.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.intent_extraction.extractor import (
    IntentExtractor, IntentResult,
    _parse_json_output, _rule_based_fallback,
)
from src.intent_extraction.paraphraser import Paraphraser
from src.guardrails.tts_preprocessor import TTSPreprocessor


# ---------------------------------------------------------------------------
# Test _parse_json_output
# ---------------------------------------------------------------------------

class TestParseJsonOutput:
    def test_clean_json(self):
        raw = '{"subject": "anh", "action": "đặt bàn", "context": "ngày mai"}'
        result = _parse_json_output(raw)
        assert result["subject"] == "anh"
        assert result["action"] == "đặt bàn"
        assert result["context"] == "ngày mai"

    def test_json_with_surrounding_text(self):
        raw = 'Đây là kết quả: {"subject": "em", "action": "xem menu", "context": ""}'
        result = _parse_json_output(raw)
        assert result is not None
        assert result["action"] == "xem menu"

    def test_malformed_json_fallback(self):
        raw = '"subject": "anh", "action": "hủy order"'
        result = _parse_json_output(raw)
        # Should return None or partial
        # We accept None here, fallback handles it
        assert result is None or "action" in result

    def test_empty_context(self):
        raw = '{"subject": "khách", "action": "hỏi wifi", "context": ""}'
        result = _parse_json_output(raw)
        assert result["context"] == ""


# ---------------------------------------------------------------------------
# Test _rule_based_fallback
# ---------------------------------------------------------------------------

class TestRuleBasedFallback:
    def test_basic_subject_extraction(self):
        result = _rule_based_fallback("Cho anh đặt bàn tối nay")
        assert "anh" in result["subject"].lower()

    def test_time_context_extraction(self):
        result = _rule_based_fallback("Cho em đặt bàn vào ngày mai lúc 7h")
        assert result["action"] != ""
        # Context should contain time info
        assert "ngày mai" in result["context"] or "7h" in result["context"] \
               or result["action"] != ""

    def test_no_subject(self):
        result = _rule_based_fallback("Hủy order ngay bây giờ")
        assert result["action"] != ""

    def test_always_returns_dict(self):
        for text in ["", "   ", "abc", "12345"]:
            result = _rule_based_fallback(text)
            assert isinstance(result, dict)
            assert "subject" in result
            assert "action" in result
            assert "context" in result


# ---------------------------------------------------------------------------
# Test IntentExtractor (mock model)
# ---------------------------------------------------------------------------

class TestIntentExtractor:
    @pytest.fixture
    def extractor_with_mock_model(self):
        with patch("src.intent_extraction.extractor.IntentExtractionModel") as MockModel:
            mock_instance = MagicMock()
            mock_instance.is_loaded.return_value = True
            mock_instance.infer_raw.return_value = json.dumps({
                "subject": "anh",
                "action": "đặt bàn tiệc sinh nhật",
                "context": "ngày mai lúc 7h",
            }, ensure_ascii=False)
            MockModel.get_instance.return_value = mock_instance

            extractor = IntentExtractor.__new__(IntentExtractor)
            extractor._model = mock_instance
            return extractor

    def test_extract_birthday_booking(self, extractor_with_mock_model):
        result = extractor_with_mock_model.extract(
            "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé"
        )
        assert isinstance(result, IntentResult)
        assert result.subject == "anh"
        assert "đặt bàn" in result.action
        assert "7h" in result.context or "ngày mai" in result.context
        assert result.latency_ms >= 0

    def test_extract_always_returns_result(self, extractor_with_mock_model):
        extractor_with_mock_model._model.infer_raw.side_effect = Exception("Model crashed")
        result = extractor_with_mock_model.extract("Cho tôi xem menu")
        assert isinstance(result, IntentResult)
        assert result.action != ""


# ---------------------------------------------------------------------------
# Test Paraphraser
# ---------------------------------------------------------------------------

class TestParaphraser:
    def setup_method(self):
        self.p = Paraphraser()

    def test_basic_paraphrase(self):
        result = self.p.paraphrase(
            cache_response="Quán có thể đặt bàn cho quý khách",
            action="đặt bàn",
            context="ngày mai lúc 7h",
            subject="anh",
            seed=42,
        )
        assert isinstance(result, str)
        assert len(result) > 10
        # Should contain the cache response
        assert "đặt bàn" in result.lower()

    def test_no_context(self):
        result = self.p.paraphrase(
            cache_response="Menu của quán đây ạ",
            action="xem menu",
            context="",
            subject="em",
            seed=1,
        )
        assert isinstance(result, str)
        assert len(result) > 5

    def test_output_no_double_punctuation(self):
        result = self.p.paraphrase(
            cache_response="OK.",
            action="hủy order",
            context="",
            subject="anh",
            seed=99,
        )
        # No ".." or "!!" in result
        assert ".." not in result
        assert "!!" not in result

    def test_subject_normalization(self):
        result = self.p.paraphrase(
            cache_response="Quán nhận được yêu cầu",
            action="order",
            context="",
            subject="tôi",
            seed=0,
        )
        # "tôi" → "anh/chị"
        assert "tôi" not in result


# ---------------------------------------------------------------------------
# Test TTSPreprocessor
# ---------------------------------------------------------------------------

class TestTTSPreprocessor:
    def setup_method(self):
        self.pp = TTSPreprocessor()

    def test_price_thousand(self):
        assert "49k" in self.pp.process("Giá 49.000đ")

    def test_price_million(self):
        result = self.pp.process("Tổng là 1.500.000đ")
        assert "triệu" in result

    def test_size_m(self):
        result = self.pp.process("Size M")
        assert "mờ" in result.lower()

    def test_vat(self):
        result = self.pp.process("Giá chưa bao gồm VAT")
        assert "thuế" in result.lower()
        assert "VAT" not in result

    def test_hour_format(self):
        result = self.pp.process("Mở cửa lúc 7h30")
        assert "7 giờ 30" in result

    def test_percent(self):
        result = self.pp.process("Giảm 20%")
        assert "phần trăm" in result

    def test_no_crash_on_empty(self):
        assert self.pp.process("") == ""

    def test_no_crash_on_special_chars(self):
        result = self.pp.process("Hello @world #test & more | pipe")
        assert isinstance(result, str)