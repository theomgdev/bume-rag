"""Benchmark corpora: memories, labelled queries, and the language groups."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CROSS_LINGUAL = "cross"


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    language: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    language: str
    gold: tuple[str, ...] = ()
    note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


@dataclass
class Suite:
    name: str
    memories: dict[str, Memory] = field(default_factory=dict)
    queries: list[Query] = field(default_factory=list)

    def group_of(self, query: Query) -> str:
        """`en-en`, `tr-tr`, or `cross` when the query and its gold disagree.

        Unanswerable queries have no gold to compare against, so they group by
        their own language; abstention is scored separately anyway.
        """
        gold_languages = {self.memories[m].language for m in query.gold if m in self.memories}
        if not gold_languages:
            return f"{query.language}-{query.language}"
        if gold_languages == {query.language}:
            return f"{query.language}-{query.language}"
        return CROSS_LINGUAL

    def groups(self) -> dict[str, list[Query]]:
        out: dict[str, list[Query]] = {}
        for query in self.queries:
            out.setdefault(self.group_of(query), []).append(query)
        return out

    def missing_gold(self) -> list[tuple[str, str]]:
        """Query/memory id pairs where the label points at a memory we do not have."""
        return [(q.id, m) for q in self.queries for m in q.gold if m not in self.memories]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_suite(directory: Path) -> Suite:
    """Load a suite from `<directory>/memories.jsonl` and `<directory>/queries.jsonl`."""
    memories = {
        row["id"]: Memory(id=row["id"], text=row["text"], language=row["language"])
        for row in _read_jsonl(directory / "memories.jsonl")
    }
    queries = [
        Query(
            id=row["id"],
            text=row["text"],
            language=row["language"],
            gold=tuple(row.get("gold", ())),
            note=row.get("note", ""),
        )
        for row in _read_jsonl(directory / "queries.jsonl")
    ]
    suite = Suite(name=directory.name, memories=memories, queries=queries)
    dangling = suite.missing_gold()
    if dangling:
        raise ValueError(f"{directory}: gold ids with no memory: {dangling[:5]}")
    return suite
