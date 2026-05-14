import json
import time
import logging
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    total_generated: int = 0
    total_valid: int = 0
    total_duplicates: int = 0
    per_intent: dict = field(default_factory=lambda: defaultdict(int))
    per_language: dict = field(default_factory=lambda: defaultdict(int))
    hard_samples: int = 0
    start_time: float = field(default_factory=time.time)
    api_calls: int = 0
    api_errors: int = 0

    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60

    def samples_per_minute(self) -> float:
        elapsed = self.elapsed_minutes()
        return self.total_valid / elapsed if elapsed > 0 else 0

    def estimated_remaining(self, target: int) -> float:
        spm = self.samples_per_minute()
        remaining = max(0, target - self.total_valid)
        return remaining / spm if spm > 0 else float("inf")


class PipelineTracker:
    def __init__(self, target: int, report_every: int = 100, stats_path: str = "data/checkpoints/stats.json"):
        self.target = target
        self.report_every = report_every
        self.stats_path = Path(stats_path)
        self.stats = PipelineStats()
        self._last_report = 0

    def update(self, samples: list[dict], duplicates: int = 0, api_errors: int = 0) -> None:
        self.stats.total_generated += len(samples) + duplicates
        self.stats.total_valid += len(samples)
        self.stats.total_duplicates += duplicates
        self.stats.api_errors += api_errors
        self.stats.api_calls += 1

        for s in samples:
            self.stats.per_intent[s["intent"]] += 1
            self.stats.per_language[s["language"]] += 1
            if s["is_hard"]:
                self.stats.hard_samples += 1

        if self.stats.total_valid - self._last_report >= self.report_every:
            self._report()
            self._last_report = self.stats.total_valid
            self._save()

    def _report(self) -> None:
        s = self.stats
        pct = 100 * s.total_valid / self.target
        eta = s.estimated_remaining(self.target)
        logger.info(
            f"Progress: {s.total_valid}/{self.target} ({pct:.1f}%) | "
            f"{s.samples_per_minute():.1f} samples/min | "
            f"ETA: {eta:.1f} min | "
            f"Dupes: {s.total_duplicates} | "
            f"Errors: {s.api_errors}"
        )

    def _save(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self.stats)
        self.stats_path.write_text(json.dumps(data, indent=2))

    def final_report(self) -> None:
        self._report()
        self._save()
        logger.info(f"Pipeline complete in {self.stats.elapsed_minutes():.1f} minutes")