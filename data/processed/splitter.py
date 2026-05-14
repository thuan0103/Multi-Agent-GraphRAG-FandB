import json
import random
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

INTENT_NAMES = ["order", "consultant", "faq", "ignore"]


def stratified_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    random.seed(seed)
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    train, val, test = [], [], []

    for label, label_samples in by_label.items():
        random.shuffle(label_samples)
        n = len(label_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(label_samples[:n_train])
        val.extend(label_samples[n_train:n_train + n_val])
        test.extend(label_samples[n_train + n_val:])
        logger.info(f"Label {INTENT_NAMES[label]}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def prepare_splits(input_path: str, output_dir: str = "data/processed") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = json.loads(Path(input_path).read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(samples)} samples")

    train, val, test = stratified_split(samples)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{name}.json"
        path.write_text(
            json.dumps(split, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"Saved {len(split)} samples to {path}")