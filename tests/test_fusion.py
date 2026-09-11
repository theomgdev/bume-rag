import pytest

from bume_rag.corpus import Memory, Query, Suite
from bume_rag.dense import DenseRetriever, normalise
from bume_rag.embedding import HashingEmbedder
from bume_rag.fusion import RRFRetriever
from bume_rag.lexical import LexicalRetriever
from bume_rag.retriever import Retrieved


class Fixed:
    """A channel that returns a fixed ranking, so fusion is testable alone."""

    def __init__(self, name, ranking):
        self.name = name
        self.ranking = ranking

    def index(self, suite):
        pass

    def search(self, query, limit):
        return [Retrieved(m, 1.0 / (i + 1)) for i, m in enumerate(self.ranking)][:limit]


def ids(hits):
    return [h.memory_id for h in hits]


def test_a_document_both_channels_like_beats_one_channels_favourite():
    """The whole point of fusion: agreement outweighs a single strong opinion."""
    r = RRFRetriever([Fixed("a", ["x", "agreed"]), Fixed("b", ["y", "agreed"])])
    assert ids(r.search(Query("q", "q", "en"), 3))[0] == "agreed"


def test_scores_are_never_compared_across_channels():
    """BM25 is unbounded and cosine is in [-1,1]; only rank is shared."""
    huge = Fixed("huge", ["loser"])
    huge.search = lambda q, limit: [Retrieved("loser", 10_000.0)]
    r = RRFRetriever([huge, Fixed("b", ["winner"]), Fixed("c", ["winner"])])
    assert ids(r.search(Query("q", "q", "en"), 3))[0] == "winner"


def test_weights_shift_the_outcome():
    channels = [Fixed("a", ["from-a"]), Fixed("b", ["from-b"])]
    even = RRFRetriever(channels)
    assert ids(even.search(Query("q", "q", "en"), 2)) == ["from-a", "from-b"]
    weighted = RRFRetriever(channels, weights=[1.0, 5.0])
    assert ids(weighted.search(Query("q", "q", "en"), 2))[0] == "from-b"


def test_candidates_are_fused_deeper_than_they_are_returned():
    """A memory ranked 30th by one channel and 1st by the other must survive."""
    deep = Fixed("deep", [f"m{i}" for i in range(40)] + ["needle"])
    shallow = Fixed("shallow", ["needle"])
    r = RRFRetriever([deep, shallow], candidates=50)
    assert "needle" in ids(r.search(Query("q", "q", "en"), 5))


def test_a_channel_returning_nothing_does_not_break_fusion():
    r = RRFRetriever([Fixed("empty", []), Fixed("b", ["only"])])
    assert ids(r.search(Query("q", "q", "en"), 3)) == ["only"]


def test_limit_is_honoured():
    r = RRFRetriever([Fixed("a", [f"m{i}" for i in range(10)])])
    assert len(r.search(Query("q", "q", "en"), 3)) == 3


def test_rrf_needs_a_channel():
    with pytest.raises(ValueError):
        RRFRetriever([])


def test_weights_must_match_channels():
    with pytest.raises(ValueError):
        RRFRetriever([Fixed("a", [])], weights=[1.0, 2.0])


def test_normalise_leaves_a_zero_vector_alone():
    assert normalise([0.0, 0.0]) == [0.0, 0.0]


def test_dense_ranks_by_cosine_not_by_magnitude():
    class Stub:
        name = "stub"

        def embed(self, texts):
            table = {"q": [1.0, 0.0], "near": [10.0, 0.1], "far": [0.0, 1.0]}
            return [table[t] for t in texts]

    suite = Suite(
        name="t",
        memories={"near": Memory("near", "near", "en"), "far": Memory("far", "far", "en")},
        queries=[],
    )
    d = DenseRetriever(Stub())
    d.index(suite)
    assert ids(d.search(Query("q", "q", "en"), 2)) == ["near", "far"]


def test_dense_on_an_empty_corpus_returns_nothing():
    d = DenseRetriever(HashingEmbedder())
    d.index(Suite(name="t", memories={}, queries=[]))
    assert d.search(Query("q", "q", "en"), 5) == []


def test_hybrid_recovers_a_query_the_lexical_channel_loses_entirely():
    """The measured cross-lingual case, in miniature."""

    class Meaning:
        name = "meaning"

        def embed(self, texts):
            table = {
                "kayıt dosyaları neden yanlış klasöre gidiyor": [1.0, 0.0],
                "the save directory follows the working directory": [0.9, 0.1],
                "rsync needs checksum or ninja sees nothing": [0.0, 1.0],
            }
            return [table[t] for t in texts]

    suite = Suite(
        name="t",
        memories={
            "gold": Memory("gold", "the save directory follows the working directory", "en"),
            "other": Memory("other", "rsync needs checksum or ninja sees nothing", "en"),
        },
        queries=[],
    )
    query = Query("q", "kayıt dosyaları neden yanlış klasöre gidiyor", "tr", ("gold",))

    lexical = LexicalRetriever()
    lexical.index(suite)
    assert lexical.search(query, 10) == [], "precondition: no shared terms"

    hybrid = RRFRetriever([lexical, DenseRetriever(Meaning())])
    hybrid.index(suite)
    assert ids(hybrid.search(query, 2))[0] == "gold"
