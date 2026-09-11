"""Ranking metrics, delivery cost, and abstention.

Relevance is binary: a memory is gold or it is not. Nothing here weights a
partial match, because the corpus labels do not carry one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, m in enumerate(ranked[:k]) if m in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, m in enumerate(ranked):
        if m in gold:
            return 1.0 / (i + 1)
    return 0.0


@dataclass
class Accumulator:
    """Sums per-query scores so a group mean is a division, not a second pass."""

    n_ranked: int = 0
    ndcg: float = 0.0
    recall: float = 0.0
    rr: float = 0.0
    n_delivery: int = 0
    chars: int = 0
    words: int = 0
    latency_ms: float = 0.0
    n_abstain: int = 0
    abstained: int = 0

    def add_ranking(self, ranked: list[str], gold: set[str], k: int) -> None:
        self.n_ranked += 1
        self.ndcg += ndcg_at_k(ranked, gold, 10)
        self.recall += recall_at_k(ranked, gold, k)
        self.rr += reciprocal_rank(ranked, gold)

    def add_delivery(self, texts: list[str], latency_ms: float) -> None:
        self.n_delivery += 1
        self.chars += sum(len(t) for t in texts)
        self.words += sum(len(t.split()) for t in texts)
        self.latency_ms += latency_ms

    def add_abstention(self, returned_nothing: bool) -> None:
        self.n_abstain += 1
        self.abstained += int(returned_nothing)

    def as_row(self) -> dict[str, float | int]:
        ranked = self.n_ranked or 1
        delivered = self.n_delivery or 1
        return {
            "n": self.n_delivery,
            "ndcg@10": self.ndcg / ranked,
            "recall@k": self.recall / ranked,
            "mrr": self.rr / ranked,
            "chars": self.chars / delivered,
            "words": self.words / delivered,
            "ms": self.latency_ms / delivered,
            "abstain": (self.abstained / self.n_abstain) if self.n_abstain else float("nan"),
        }
