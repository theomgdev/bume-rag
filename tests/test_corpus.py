import json
from pathlib import Path

import pytest

from bume_rag.corpus import Memory, Query, Suite, load_suite

SEED = Path(__file__).resolve().parents[1] / "benchmarks" / "seed"


def suite_of(memories, queries):
    return Suite(name="t", memories={m.id: m for m in memories}, queries=queries)


def test_group_follows_the_gold_language_not_the_query_language():
    s = suite_of(
        [Memory("en1", "x", "en"), Memory("tr1", "y", "tr")],
        [
            Query("a", "q", "tr", ("en1",)),
            Query("b", "q", "tr", ("tr1",)),
            Query("c", "q", "en", ("en1",)),
        ],
    )
    assert s.group_of(s.queries[0]) == "cross"
    assert s.group_of(s.queries[1]) == "tr-tr"
    assert s.group_of(s.queries[2]) == "en-en"


def test_mixed_gold_languages_count_as_cross_lingual():
    s = suite_of(
        [Memory("en1", "x", "en"), Memory("tr1", "y", "tr")],
        [Query("a", "q", "tr", ("tr1", "en1"))],
    )
    assert s.group_of(s.queries[0]) == "cross"


def test_unanswerable_query_groups_by_its_own_language():
    s = suite_of([Memory("en1", "x", "en")], [Query("a", "q", "tr", ())])
    assert s.group_of(s.queries[0]) == "tr-tr"


def test_dangling_gold_is_reported_not_silently_dropped(tmp_path):
    (tmp_path / "memories.jsonl").write_text(
        json.dumps({"id": "m1", "text": "x", "language": "en"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "q", "language": "en", "gold": ["nope"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nope"):
        load_suite(tmp_path)


def test_seed_suite_carries_all_three_groups_and_an_abstention_case():
    suite = load_suite(SEED)
    groups = suite.groups()
    assert set(groups) == {"en-en", "tr-tr", "cross"}
    assert all(len(v) >= 3 for v in groups.values())
    assert any(not q.answerable for q in suite.queries)
    assert not suite.missing_gold()
