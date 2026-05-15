import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("[SemanticCache] faiss không có, dùng numpy brute-force.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    id: str
    key_text: str               # action text (C2.2) hoặc full query (B2.3)
    response_template: str      # câu trả lời / template
    domain: str                 # "menu" | "faq" | "order" | "general"
    tags: List[str]             # dùng cho tag-based invalidation
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Embedding function (dùng sentence-transformers hoặc API)
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """
    Wrapper cho embedding model.
    Ưu tiên: local SentenceTransformer → TEI HTTP API → dummy (test mode).
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 tei_url: Optional[str] = None):
        self._tei_url = tei_url
        self._model = None
        self._dim = 384

        if tei_url:
            logger.info(f"[Embedding] Dùng TEI: {tei_url}")
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"[Embedding] Loaded local: {model_name} (dim={self._dim})")
        except Exception as e:
            logger.warning(f"[Embedding] Không load được model: {e}. Dùng dummy mode.")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        """Trả về (n, dim) float32 array."""
        if self._tei_url:
            return self._embed_tei(texts)
        if self._model:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.astype(np.float32)
        # Dummy: random unit vector (chỉ dùng trong test)
        vecs = np.random.randn(len(texts), self._dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-8)

    def _embed_tei(self, texts: List[str]) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self._tei_url}/embed",
            json={"inputs": texts},
            timeout=10,
        )
        resp.raise_for_status()
        vecs = np.array(resp.json(), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-8)


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)   # Inner Product = cosine (với unit vectors)
        else:
            self._index = None
        self._id_map: List[str] = []   # FAISS internal index → entry id

    def add(self, vec: np.ndarray, entry_id: str):
        vec = vec.reshape(1, -1).astype(np.float32)
        if self._index is not None:
            self._index.add(vec)
        else:
            # numpy fallback
            if not hasattr(self, "_vecs"):
                self._vecs = vec
            else:
                self._vecs = np.vstack([self._vecs, vec])
        self._id_map.append(entry_id)

    def search(self, vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Trả list (entry_id, score) sorted desc."""
        if not self._id_map:
            return []
        vec = vec.reshape(1, -1).astype(np.float32)
        k = min(k, len(self._id_map))

        if self._index is not None:
            scores, indices = self._index.search(vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self._id_map[idx], float(score)))
            return results
        else:
            # numpy brute-force
            sims = (self._vecs @ vec.T).flatten()
            top_k = np.argsort(sims)[::-1][:k]
            return [(self._id_map[i], float(sims[i])) for i in top_k]

    def rebuild_excluding(self, exclude_ids: set, all_vecs: Dict[str, np.ndarray]):
        """Rebuild index khi invalidate entries."""
        if self._index is not None:
            self._index = faiss.IndexFlatIP(self.dim)
        elif hasattr(self, "_vecs"):
            del self._vecs
        self._id_map = []
        for eid, vec in all_vecs.items():
            if eid not in exclude_ids:
                self.add(vec, eid)

    def __len__(self):
        return len(self._id_map)


# ---------------------------------------------------------------------------
# Main SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Thread-safe semantic cache dùng FAISS.

    Hai mode:
    - intent_mode=True  : threshold 0.92, key = action text (C2.2)
    - intent_mode=False : threshold 0.95, key = full query (B2.3)
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        threshold: float = 0.92,
        tei_url: Optional[str] = None,
    ):
        self._emb = embedding_provider or EmbeddingProvider(tei_url=tei_url)
        self._threshold = threshold
        self._lock = threading.RLock()

        self._entries: Dict[str, CacheEntry] = {}      # id → CacheEntry
        self._vecs: Dict[str, np.ndarray] = {}         # id → embedding vector
        self._index = FaissIndex(self._emb.dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, key_text: str, top_k: int = 3) -> Optional[Tuple[CacheEntry, float]]:
        """
        Tìm cache entry gần nhất.
        Trả (entry, score) nếu score ≥ threshold, else None.
        """
        t0 = time.perf_counter()
        vec = self._emb.embed([key_text])[0]

        with self._lock:
            results = self._index.search(vec, top_k)

        best = None
        best_score = 0.0
        for entry_id, score in results:
            if score >= self._threshold and score > best_score:
                best_score = score
                best = self._entries.get(entry_id)

        elapsed = (time.perf_counter() - t0) * 1000
        if best:
            with self._lock:
                self._entries[best.id].hit_count += 1
            logger.debug(f"[SemanticCache] HIT score={best_score:.3f} ({elapsed:.1f}ms)")
            return best, best_score

        logger.debug(f"[SemanticCache] MISS ({elapsed:.1f}ms)")
        return None

    def put(
        self,
        key_text: str,
        response_template: str,
        domain: str = "general",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Thêm entry mới. Trả entry_id."""
        vec = self._emb.embed([key_text])[0]
        entry_id = str(uuid.uuid4())

        entry = CacheEntry(
            id=entry_id,
            key_text=key_text,
            response_template=response_template,
            domain=domain,
            tags=tags or [domain],
        )

        with self._lock:
            self._entries[entry_id] = entry
            self._vecs[entry_id] = vec
            self._index.add(vec, entry_id)

        logger.debug(f"[SemanticCache] PUT id={entry_id[:8]} key={key_text!r}")
        return entry_id

    def invalidate_by_tag(self, tag: str) -> int:
        """
        Xóa tất cả entries có tag tương ứng.
        Dùng khi menu/FAQ thay đổi.
        """
        with self._lock:
            to_remove = {
                eid for eid, e in self._entries.items()
                if tag in e.tags
            }
            if not to_remove:
                return 0

            for eid in to_remove:
                del self._entries[eid]
                del self._vecs[eid]

            # Rebuild index
            self._index.rebuild_excluding(set(), self._vecs)
            logger.info(f"[SemanticCache] Invalidated {len(to_remove)} entries với tag={tag!r}")
            return len(to_remove)

    def invalidate_by_domain(self, domain: str) -> int:
        return self.invalidate_by_tag(domain)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._vecs.clear()
            self._index = FaissIndex(self._emb.dim)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "threshold": self._threshold,
                "index_size": len(self._index),
                "domains": list({e.domain for e in self._entries.values()}),
            }

    def __len__(self):
        return len(self._entries)