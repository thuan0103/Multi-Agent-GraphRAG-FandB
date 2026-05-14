"""
C2.2 + B2.3: Tracking cache hit rate, miss rate, latency.
Thread-safe counters. Expose metrics cho health check và benchmark.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    hit_latencies_ms: List[float] = field(default_factory=list)
    miss_latencies_ms: List[float] = field(default_factory=list)
    paraphrase_latencies_ms: List[float] = field(default_factory=list)
    total_invalidations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_hit(self, latency_ms: float):
        with self._lock:
            self.hits += 1
            self.hit_latencies_ms.append(latency_ms)

    def record_miss(self, latency_ms: float):
        with self._lock:
            self.misses += 1
            self.miss_latencies_ms.append(latency_ms)

    def record_paraphrase(self, latency_ms: float):
        with self._lock:
            self.paraphrase_latencies_ms.append(latency_ms)

    def record_invalidation(self):
        with self._lock:
            self.total_invalidations += 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def avg_hit_latency(self) -> float:
        return (sum(self.hit_latencies_ms) / len(self.hit_latencies_ms)
                if self.hit_latencies_ms else 0.0)

    def avg_miss_latency(self) -> float:
        return (sum(self.miss_latencies_ms) / len(self.miss_latencies_ms)
                if self.miss_latencies_ms else 0.0)

    def p95_hit_latency(self) -> float:
        if not self.hit_latencies_ms:
            return 0.0
        s = sorted(self.hit_latencies_ms)
        return s[int(len(s) * 0.95)]

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate(), 4),
            "avg_hit_latency_ms": round(self.avg_hit_latency(), 2),
            "avg_miss_latency_ms": round(self.avg_miss_latency(), 2),
            "p95_hit_latency_ms": round(self.p95_hit_latency(), 2),
            "total_invalidations": self.total_invalidations,
            "meets_latency_target": self.p95_hit_latency() <= 100.0,
            "meets_hit_rate_target": self.hit_rate() >= 0.60,
        }

    def report(self):
        d = self.to_dict()
        print("\n" + "=" * 55)
        print("CACHE STATS REPORT")
        print("=" * 55)
        print(f"  Hits / Misses    : {d['hits']} / {d['misses']}")
        print(f"  Hit Rate         : {d['hit_rate']*100:.1f}%  (target ≥ 60%)")
        print(f"  Avg Hit Latency  : {d['avg_hit_latency_ms']:.1f}ms (target ≤ 100ms)")
        print(f"  P95 Hit Latency  : {d['p95_hit_latency_ms']:.1f}ms")
        print(f"  Avg Miss Latency : {d['avg_miss_latency_ms']:.1f}ms")
        print(f"  Invalidations    : {d['total_invalidations']}")
        print(f"  Latency OK       : {'✅' if d['meets_latency_target'] else '❌'}")
        print(f"  Hit Rate OK      : {'✅' if d['meets_hit_rate_target'] else '❌'}")
        print("=" * 55)


# Global singleton
_global_stats = CacheStats()


def get_stats() -> CacheStats:
    return _global_stats