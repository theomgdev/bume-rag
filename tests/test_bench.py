from pathlib import Path

from bume_rag.bench import POOLED, format_report, run
from bume_rag.corpus import Memory, Query, Suite, load_suite
from bume_rag.retriever import NoOpRetriever, Retrieved

SEED = Path(__file__).resolve().parents[1] / "benchmarks" / "seed"


class PerfectRetriever:
    """Returns exactly the gold, so a broken harness cannot score it as zero."""

    name = "perfect"

    def index(self, suite):
        self.suite = suite

    def search(self, query, limit):
        return [Retrieved(m, 1.0) for m in query.gold][:limit]


def test_no_op_scores_zero_and_abstains_everywhere():
    report = run(NoOpRetriever(), load_suite(SEED))
    for group, row in report.rows.items():
        assert row["ndcg@10"] == 0.0, group
        assert row["mrr"] == 0.0, group
        assert row["chars"] == 0.0, group
    assert report.rows[POOLED]["abstain"] == 1.0


def test_a_perfect_retriever_scores_one_so_zeros_mean_something():
    report = run(PerfectRetriever(), load_suite(SEED))
    assert report.rows[POOLED]["ndcg@10"] == 1.0
    assert report.rows[POOLED]["mrr"] == 1.0
    assert report.rows[POOLED]["recall@k"] == 1.0
    assert report.rows[POOLED]["chars"] > 0


def test_every_group_is_reported_plus_pooled():
    report = run(NoOpRetriever(), load_suite(SEED))
    assert set(report.rows) == {"en-en", "tr-tr", "cross", POOLED}


def test_pooled_counts_every_query_once():
    report = run(NoOpRetriever(), load_suite(SEED))
    groups = sum(row["n"] for name, row in report.rows.items() if name != POOLED)
    assert groups == report.rows[POOLED]["n"] == len(load_suite(SEED).queries)


def test_a_group_collapse_does_not_hide_in_the_pooled_average():
    """The reason the table is per group: cross-lingual failing must be visible."""

    class EnglishOnly:
        name = "en-only"

        def index(self, suite):
            self.suite = suite

        def search(self, query, limit):
            if query.language != "en":
                return []
            return [Retrieved(m, 1.0) for m in query.gold][:limit]

    report = run(EnglishOnly(), load_suite(SEED))
    assert report.rows["en-en"]["mrr"] == 1.0
    assert report.rows["cross"]["mrr"] == 0.0
    assert 0.0 < report.rows[POOLED]["mrr"] < 1.0


def test_delivery_cost_counts_returned_text_only():
    suite = Suite(
        name="t",
        memories={"a": Memory("a", "one two three", "en"), "b": Memory("b", "four", "en")},
        queries=[Query("q", "q", "en", ("a",))],
    )
    report = run(PerfectRetriever(), suite)
    assert report.rows[POOLED]["words"] == 3.0
    assert report.rows[POOLED]["chars"] == float(len("one two three"))


def test_format_report_prints_a_row_per_group():
    report = run(NoOpRetriever(), load_suite(SEED))
    text = format_report(report)
    for group in ("en-en", "tr-tr", "cross", POOLED):
        assert group in text
    assert "no-op" in text
