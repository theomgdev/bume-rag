import math

from bume_rag.metrics import Accumulator, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ndcg_rewards_rank_not_just_presence():
    gold = {"a"}
    assert ndcg_at_k(["a", "b", "c"], gold, 10) == 1.0
    second = ndcg_at_k(["b", "a", "c"], gold, 10)
    assert second == 1 / math.log2(3)
    assert second < 1.0


def test_ndcg_ideal_accounts_for_multiple_gold():
    gold = {"a", "b"}
    perfect = ndcg_at_k(["a", "b", "c"], gold, 10)
    assert perfect == 1.0
    half = ndcg_at_k(["a", "c", "d"], gold, 10)
    assert half == 1 / (1 + 1 / math.log2(3))


def test_ndcg_honours_the_cut():
    assert ndcg_at_k(["x"] * 10 + ["a"], {"a"}, 10) == 0.0


def test_recall_is_a_fraction_of_gold_found():
    assert recall_at_k(["a", "b"], {"a", "b", "c"}, 10) == 2 / 3
    assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5


def test_reciprocal_rank_takes_the_first_hit():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_empty_ranking_scores_zero_everywhere():
    assert ndcg_at_k([], {"a"}, 10) == 0.0
    assert recall_at_k([], {"a"}, 10) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_accumulator_averages_over_the_right_denominators():
    acc = Accumulator()
    acc.add_ranking(["a"], {"a"}, 10)
    acc.add_delivery(["four five"], 2.0)
    acc.add_delivery([], 4.0)
    row = acc.as_row()
    assert row["n"] == 2
    assert row["ndcg@10"] == 1.0, "ranking averages over ranked queries, not delivered ones"
    assert row["words"] == 1.0
    assert row["ms"] == 3.0
    assert math.isnan(row["abstain"])


def test_abstention_is_the_share_that_returned_nothing():
    acc = Accumulator()
    acc.add_abstention(True)
    acc.add_abstention(False)
    assert acc.as_row()["abstain"] == 0.5
