import logging
import yaml
from src.router import IntentClassifier, RouterBenchmark, profile_latency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    test_path: str = "data/processed/test.json",
    config_path: str = "config.yaml",
) -> None:

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["router"]["model_id"] = model_path

    tmp_config = "config_eval.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    classifier = IntentClassifier(config_path=tmp_config)
    classifier.load()

    logger.info("Profiling latency...")
    profile_latency(classifier, n_runs=50)

    logger.info("Running full benchmark...")
    benchmark = RouterBenchmark(classifier, config_path=tmp_config)
    metrics = benchmark.run(test_path)

    print("\n── NGHIỆM THU A1 ──")
    print(f"Accuracy ≥ 92%:        {'✓ PASS' if metrics['meets_accuracy_target'] else '✗ FAIL'} ({metrics['accuracy']*100:.2f}%)")
    print(f"Latency p95 ≤ 200ms:   {'✓ PASS' if metrics['meets_latency_target'] else '✗ FAIL'} ({metrics['latency']['p95_ms']}ms)")
    print(f"Latency p95 ≤ 100ms:   {'✓ EXCELLENT' if metrics['meets_excellent_latency'] else '✗'}")
    print(f"Hard samples ≥ 75%:    {'✓ PASS' if metrics['meets_hard_sample_target'] else '✗ FAIL'} ({(metrics['hard_sample_accuracy'] or 0)*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/router-awq")
    parser.add_argument("--test", default="data/processed/test.json")
    args = parser.parse_args()
    evaluate(args.model, args.test)