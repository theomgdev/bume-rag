"""The lexical channel: SQLite FTS5 with BM25 ranking.

Zero dependencies — the stdlib build ships FTS5 (verified: sqlite 3.45.3 on
CPython 3.13.1 reports ENABLE_FTS5).
"""

from __future__ import annotations

import re
import sqlite3

from bume_rag.corpus import Query, Suite
from bume_rag.retriever import Retrieved

# FTS5's query language is not a search box. A raw user query raises rather than
# returning nothing: "npc gear-up" parses `up` as a column, "c++" as an operator.
# So terms get extracted and quoted, never passed through.
#
# \w rather than [0-9a-z_]: an ASCII class shreds `değişiklik` into five
# fragments while still splitting identifiers the same way, which silently
# breaks the Turkish half of the corpus and nothing else.
TERM = re.compile(r"\w+(?:[._]\w+)*", re.UNICODE)

# unicode61 remove_diacritics 2 folds ğşçöü onto gscou but leaves ı alone,
# because a dotless i is a distinct letter rather than a decomposable accent.
# Without this, `ciktisi` misses a memory that says `çıktısı`.
DOTLESS = str.maketrans({"ı": "i", "İ": "i"})


def terms_of(text: str) -> list[str]:
    return [t for t in TERM.findall(text.translate(DOTLESS).lower()) if len(t) > 1]


def match_expression(text: str) -> str:
    """An OR of quoted terms. Empty when the query has nothing indexable in it."""
    return " OR ".join(f'"{t}"' for t in terms_of(text))


class LexicalRetriever:
    """BM25 over FTS5.

    No stopword list: on the real corpus the gold document already ranks first
    for a query two thirds made of stopwords, because inverse document frequency
    does that job as soon as the corpus is bigger than a toy.
    """

    name = "lexical"

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "CREATE VIRTUAL TABLE memories USING fts5(id UNINDEXED, text,"
            " tokenize='unicode61 remove_diacritics 2')"
        )

    def index(self, suite: Suite) -> None:
        self.db.execute("DELETE FROM memories")
        self.db.executemany(
            "INSERT INTO memories(id, text) VALUES (?, ?)",
            [(m.id, m.text.translate(DOTLESS)) for m in suite.memories.values()],
        )

    def search(self, query: Query, limit: int) -> list[Retrieved]:
        expression = match_expression(query.text)
        if not expression:
            return []
        rows = self.db.execute(
            "SELECT id, bm25(memories) FROM memories WHERE memories MATCH ?"
            " ORDER BY bm25(memories) LIMIT ?",
            (expression, limit),
        )
        # bm25() returns a negative score, better being more negative, so the
        # sign is flipped to keep "higher is better" true across every channel.
        return [Retrieved(memory_id=row[0], score=-row[1]) for row in rows]
