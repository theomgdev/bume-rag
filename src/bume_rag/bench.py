"""Runs a retriever over a suite and reports per language group as well as pooled.

Pooled alone would hide a cross-lingual collapse behind two healthy monolingual
numbers, which is the failure this project expects to have, so the group
breakdown is the point of the table rather than a detail in it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from bume_rag.corpus import Suite
from bume_rag.metrics import Accumulator
from bume_rag.retriever import Retriever

POOLED = "pooled"


@dataclass
class Report:
    suite: str
    retriever: str
    k: int
    rows: dict[str, dict[str, float | int]]


def run(retriever: Retriever, suite: Suite, k: int = 10) -> Report:
    retriever.index(suite)
    groups = {name: Accumulator() for name in suite.groups()}
    pooled = Accumulator()

    for query in suite.queries:
        started = time.perf_counter()
        hits = retriever.search(query, k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        ranked = [h.memory_id for h in hits]
        texts = [suite.memories[m].text for m in ranked if m in suite.memories]
        accumulators = (groups[suite.group_of(query)], pooled)

        for acc in accumulators:
            acc.add_delivery(texts, elapsed_ms)
            if query.answerable:
                acc.add_ranking(ranked, set(query.gold), k)
            else:
                acc.add_abstention(not ranked)

    rows = {name: acc.as_row() for name, acc in sorted(groups.items())}
    rows[POOLED] = pooled.as_row()
    return Report(suite=suite.name, retriever=retriever.name, k=k, rows=rows)


COLUMNS = ("n", "ndcg@10", "recall@k", "mrr", "chars", "words", "ms", "abstain")


def format_report(report: Report) -> str:
    header = f"{report.suite} / {report.retriever} / k={report.k}"
    width = max((len(name) for name in report.rows), default=5)
    lines = [header, "group".ljust(width) + "".join(c.rjust(10) for c in COLUMNS)]
    for name, row in report.rows.items():
        cells = "".join(
            (str(row[c]) if c == "n" else f"{row[c]:.4f}").rjust(10) for c in COLUMNS
        )
        lines.append(name.ljust(width) + cells)
    lines.append("chars/words are per query and proxy for tokens until an encoder is pinned")
    return "\n".join(lines)
