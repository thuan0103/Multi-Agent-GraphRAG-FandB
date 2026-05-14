import os
os.environ["PYTHONUTF8"] = "1"

import argparse
import logging
import sys
sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all", "split", "train", "quantize", "evaluate"], default="all")
    parser.add_argument("--model", default="models/router-awq")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.step in ("all", "split"):
        from data.processed.splitter import prepare_splits
        prepare_splits(
            input_path="data/processed/dataset_validated.json",
            output_dir="data/processed",
        )

    if args.step in ("all", "train"):
        from training.sft_train import train
        train(args.config)

    if args.step in ("all", "quantize"):
        from training.quantize import run_all
        run_all(args.config)

    if args.step in ("all", "evaluate"):
        from training.evaluate import evaluate
        evaluate(args.model, config_path=args.config)

if __name__ == "__main__":
    main()