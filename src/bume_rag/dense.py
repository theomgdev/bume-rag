"""The dense channel: cosine similarity over embeddings.

A brute-force scan, because an exact search over a few hundred thousand vectors
is milliseconds and an approximate index is a dependency plus a recall loss.
When the corpus outgrows that, the measurement says so.
"""

from __future__ import annotations

import math

from bume_rag.corpus import Query, Suite
from bume_rag.embedding import Embedder, HashingEmbedder
from bume_rag.retriever import Retrieved


def normalise(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else vector


class DenseRetriever:
    """Cosine similarity, with vectors normalised once at index time."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []

    @property
    def name(self) -> str:
        return f"dense[{self.embedder.name}]"

    def index(self, suite: Suite) -> None:
        self.ids = list(suite.memories)
        if not self.ids:
            self.vectors = []
            return
        raw = self.embedder.embed([suite.memories[i].text for i in self.ids])
        self.vectors = [normalise(v) for v in raw]

    def search(self, query: Query, limit: int) -> list[Retrieved]:
        if not self.vectors:
            return []
        q = normalise(self.embedder.embed([query.text])[0])
        scored = (
            (sum(a * b for a, b in zip(q, v)), memory_id)
            for v, memory_id in zip(self.vectors, self.ids)
        )
        ranked = sorted(scored, key=lambda pair: -pair[0])[:limit]
        return [Retrieved(memory_id=m, score=s) for s, m in ranked]
