import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"text", "label", "intent", "is_noise", "language"}
VALID_LABELS = {0, 1, 2, 3}
VALID_INTENTS = {"order", "consultant", "faq", "ignore"}
VALID_LANGUAGES = {"vi", "en", "ko"}
MIN_TEXT_LENGTH = 2
MAX_TEXT_LENGTH = 300


def validate_sample(sample: dict) -> tuple[bool, str]:
    missing = REQUIRED_FIELDS - set(sample.keys())
    if missing:
        return False, f"Missing fields: {missing}"

    if sample["label"] not in VALID_LABELS:
        return False, f"Invalid label: {sample['label']}"

    if sample["intent"] not in VALID_INTENTS:
        return False, f"Invalid intent: {sample['intent']}"

    expected = {"order": 0, "consultant": 1, "faq": 2, "ignore": 3}
    if not sample["is_noise"] and sample["label"] != expected[sample["intent"]]:
        return False, f"Label-intent mismatch: {sample['label']} vs {sample['intent']}"

    if sample["language"] not in VALID_LANGUAGES:
        return False, f"Invalid language: {sample['language']}"

    text = sample["text"].strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False, f"Text too short: '{text}'"
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long: {len(text)} chars"

    return True, ""


def validate_dataset(samples: list[dict]) -> list[dict]:
    valid, invalid = [], []

    for i, sample in enumerate(samples):
        ok, reason = validate_sample(sample)
        if ok:
            valid.append(sample)
        else:
            invalid.append((i, reason, sample.get("text", "")[:50]))

    logger.info(f"Validation: {len(valid)} valid / {len(invalid)} invalid")

    if invalid:
        logger.warning(f"First 5 invalid samples:")
        for idx, reason, text in invalid[:5]:
            logger.warning(f"  [{idx}] {reason} | text: '{text}'")

    from collections import Counter
    label_dist = Counter(s["label"] for s in valid)
    lang_dist = Counter(s["language"] for s in valid)
    hard_count = sum(1 for s in valid if s["is_noise"])

    logger.info(f"Label distribution: {dict(sorted(label_dist.items()))}")
    logger.info(f"Language distribution: {dict(lang_dist)}")
    logger.info(f"Hard samples: {hard_count} ({100*hard_count/len(valid):.1f}%)")

    return valid