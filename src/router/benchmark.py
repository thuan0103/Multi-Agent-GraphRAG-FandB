import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

INTENT_NAMES = ["order", "consultant", "faq", "ignore"]
INTENT_TO_LABEL = {name: i for i, name in enumerate(INTENT_NAMES)}


class RouterBenchmark:
    def __init__(self, classifier, config_path: str = "config.yaml"):
        self.classifier = classifier
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["router"]

    def run(self, test_path: str, output_dir: str = "data/processed") -> dict:
        samples = json.loads(Path(test_path).read_text())
        logger.info(f"Benchmarking on {len(samples)} samples...")

        y_true, y_pred = [], []
        latencies = []
        hard_results = []
        errors = []

        for i, sample in enumerate(samples):
            text = sample["text"]
            true_label = sample["label"]
            is_hard = sample.get("is_hard", False)

            result = self.classifier.classify(text)
            pred_intent = result["action"]
            pred_label = INTENT_TO_LABEL.get(pred_intent, 3)
            latency = result["latency_ms"]

            y_true.append(true_label)
            y_pred.append(pred_label)
            latencies.append(latency)

            if is_hard:
                hard_results.append(pred_label == true_label)

            if pred_label != true_label:
                errors.append({
                    "text": text,
                    "true": INTENT_NAMES[true_label],
                    "pred": pred_intent,
                    "is_hard": is_hard,
                    "latency_ms": latency,
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{len(samples)}")

        metrics = self._compute_metrics(y_true, y_pred, latencies, hard_results, errors)
        self._print_report(metrics)
        self._save_report(metrics, output_dir)
        return metrics

    def _compute_metrics(
        self,
        y_true: list,
        y_pred: list,
        latencies: list,
        hard_results: list,
        errors: list,
    ) -> dict:
        n = len(y_true)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n

        cm = [[0] * 4 for _ in range(4)]
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        f1_scores = {}
        for c in range(4):
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(4)) - tp
            fn = sum(cm[c][r] for r in range(4)) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores[INTENT_NAMES[c]] = {
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "support": sum(cm[c]),
            }

        macro_f1 = sum(v["f1"] for v in f1_scores.values()) / 4

        lat_arr = np.array(latencies)
        latency_stats = {
            "mean_ms": round(float(np.mean(lat_arr)), 2),
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
            "max_ms": round(float(np.max(lat_arr)), 2),
            "pct_under_200ms": round(float(np.mean(lat_arr <= 200) * 100), 2),
            "pct_under_100ms": round(float(np.mean(lat_arr <= 100) * 100), 2),
        }

        hard_accuracy = sum(hard_results) / len(hard_results) if hard_results else None

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": f1_scores,
            "confusion_matrix": cm,
            "latency": latency_stats,
            "hard_sample_accuracy": round(hard_accuracy, 4) if hard_accuracy else None,
            "total_samples": n,
            "error_analysis": errors[:50],  
            "meets_accuracy_target": accuracy >= 0.92,
            "meets_latency_target": latency_stats["p95_ms"] <= 200,
            "meets_excellent_latency": latency_stats["p95_ms"] <= 100,
            "meets_hard_sample_target": (hard_accuracy or 0) >= 0.75,
        }

    def _print_report(self, m: dict) -> None:
        print("\n" + "="*60)
        print("ROUTER BENCHMARK REPORT")
        print("="*60)
        print(f"Accuracy:        {m['accuracy']*100:.2f}% {'✓' if m['meets_accuracy_target'] else '✗'} (target: 92%)")
        print(f"Macro F1:        {m['macro_f1']*100:.2f}%")
        print(f"Hard samples:    {(m['hard_sample_accuracy'] or 0)*100:.2f}% {'✓' if m['meets_hard_sample_target'] else '✗'} (target: 75%)")
        print(f"\nLatency (p95):   {m['latency']['p95_ms']:.1f}ms", end=" ")
        if m['meets_excellent_latency']:
            print("✓ Excellent (≤100ms)")
        elif m['meets_latency_target']:
            print("✓ OK (≤200ms)")
        else:
            print("✗ Exceeded target")

        print(f"\nPer-class F1:")
        for intent, stats in m["per_class_f1"].items():
            print(f"  {intent:<12} F1={stats['f1']:.3f}  P={stats['precision']:.3f}  R={stats['recall']:.3f}  n={stats['support']}")

        print(f"\nConfusion Matrix (rows=true, cols=pred):")
        print(f"{'':>12}", end="")
        for name in INTENT_NAMES:
            print(f"{name:>12}", end="")
        print()
        for i, row in enumerate(m["confusion_matrix"]):
            print(f"{INTENT_NAMES[i]:>12}", end="")
            for val in row:
                print(f"{val:>12}", end="")
            print()
        print("="*60 + "\n")

    def _save_report(self, metrics: dict, output_dir: str) -> None:
        output = Path(output_dir) / "benchmark_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        logger.info(f"Report saved to {output}")

def profile_latency(classifier, n_runs: int = 100) -> dict:
    test_texts = [
        "Cho tôi 1 ly cà phê sữa đá",
        "Có gì ngon không?",
        "Wifi mật khẩu gì?",
        "Ừm...",
        "I'd like an iced latte",
        "What do you recommend?",
        "What time do you close?",
        "Hello",
    ]

    latencies = []
    for i in range(n_runs):
        text = test_texts[i % len(test_texts)]
        result = classifier.classify(text)
        latencies.append(result["latency_ms"])

    arr = np.array(latencies)
    report = {
        "n_runs": n_runs,
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }
    print(f"\nLatency profile ({n_runs} runs):")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report