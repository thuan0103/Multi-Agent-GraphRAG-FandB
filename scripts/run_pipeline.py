import asyncio
import argparse
import logging
import json
from pathlib import Path
import sys

sys.path.append(".")
from src.pipeline import DataGenerator, validate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


async def main(config_path: str):
    generator = DataGenerator(config_path)

    print("Starting data generation pipeline...")
    samples = await generator.generate_all()

    print(f"Generated {len(samples)} raw samples. Validating...")
    valid_samples = validate_dataset(samples)

    output = Path("data/processed/dataset_validated.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(valid_samples, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(valid_samples)} valid samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config))