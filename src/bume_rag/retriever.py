"""The retriever interface every phase implements, and the baseline that returns nothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bume_rag.corpus import Query, Suite


@dataclass(frozen=True)
class Retrieved:
    memory_id: str
    score: float


class Retriever(Protocol):
    name: str

    def index(self, suite: Suite) -> None: ...

    def search(self, query: Query, limit: int) -> list[Retrieved]: ...


class NoOpRetriever:
    """Returns nothing for everything.

    This is the floor the harness is calibrated against: it must score zero on
    every ranking metric, deliver zero tokens, and abstain on every query.
    """

    name = "no-op"

    def index(self, suite: Suite) -> None:
        return None

    def search(self, query: Query, limit: int) -> list[Retrieved]:
        return []
