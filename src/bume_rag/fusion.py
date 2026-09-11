"""Reciprocal rank fusion over the lexical and dense channels."""

from __future__ import annotations

from bume_rag.corpus import Query, Suite
from bume_rag.retriever import Retrieved, Retriever

# k=60 is the constant from the original RRF paper, carried by every hybrid
# system this project compares against. It is not tuned here: with two channels
# it only controls how sharply rank 1 beats rank 2, and tuning it on a corpus
# this size would fit noise.
DEFAULT_K = 60

# Candidates are fused deeper than they are returned, because a memory ranked
# 30th by one channel and 2nd by the other should still surface.
DEFAULT_CANDIDATES = 50


class RRFRetriever:
    """Fuses any number of channels by rank, never by score.

    Scores are not comparable across channels — BM25 is unbounded and cosine is
    in [-1, 1] — so rank is the only thing both agree on.
    """

    def __init__(
        self,
        channels: list[Retriever],
        k: int = DEFAULT_K,
        candidates: int = DEFAULT_CANDIDATES,
        weights: list[float] | None = None,
    ) -> None:
        if not channels:
            raise ValueError("RRF needs at least one channel")
        if weights is not None and len(weights) != len(channels):
            raise ValueError("one weight per channel")
        self.channels = channels
        self.k = k
        self.candidates = candidates
        self.weights = weights or [1.0] * len(channels)

    @property
    def name(self) -> str:
        return "rrf[" + "+".join(c.name for c in self.channels) + "]"

    def index(self, suite: Suite) -> None:
        for channel in self.channels:
            channel.index(suite)

    def search(self, query: Query, limit: int) -> list[Retrieved]:
        scores: dict[str, float] = {}
        for channel, weight in zip(self.channels, self.weights):
            for position, hit in enumerate(channel.search(query, self.candidates)):
                scores[hit.memory_id] = scores.get(hit.memory_id, 0.0) + weight / (
                    self.k + position + 1
                )
        ranked = sorted(scores.items(), key=lambda pair: -pair[1])[:limit]
        return [Retrieved(memory_id=m, score=s) for m, s in ranked]
