from pathlib import Path

from bume_rag.corpus import Memory, Query, Suite
from bume_rag.lexical import LexicalRetriever, match_expression, terms_of

SEED = Path(__file__).resolve().parents[1] / "benchmarks" / "seed"


def indexed(*memories):
    r = LexicalRetriever()
    r.index(Suite(name="t", memories={m.id: m for m in memories}, queries=[]))
    return r


def top(retriever, text, language="en", limit=10):
    return [h.memory_id for h in retriever.search(Query("q", text, language), limit)]


def test_identifiers_survive_tokenisation_whole():
    assert terms_of("weight_hit_share = 6.0") == ["weight_hit_share", "6.0"]
    assert terms_of("main.cpp:687") == ["main.cpp", "687"]


def test_punctuation_that_breaks_the_fts5_parser_is_never_passed_through():
    """A raw query raises OperationalError; these are the shapes that do it."""
    for hostile in ["npc gear-up", "c++ / rust", "a AND b", '"unbalanced', "NEAR(x y)"]:
        r = indexed(Memory("m", "npc gear up rust c a b x y", "en"))
        r.search(Query("q", hostile, "en"), 10)


def test_a_query_with_nothing_indexable_returns_nothing():
    assert match_expression("?! ++ -") == ""
    r = indexed(Memory("m", "anything", "en"))
    assert top(r, "?! ++ -") == []


def test_dotless_i_is_folded_because_fts5_will_not_do_it():
    """remove_diacritics 2 folds ğşçöü but not ı, so `ciktisi` would miss `çıktısı`."""
    r = indexed(Memory("m", "UTF-8 çıktısı bozuluyor", "tr"))
    assert top(r, "ciktisi") == ["m"]
    assert top(r, "çıktısı") == ["m"]


def test_other_turkish_diacritics_fold_both_ways():
    r = indexed(Memory("m", "belleği paylaşımlı", "tr"))
    assert top(r, "bellegi") == ["m"]
    assert top(r, "paylasimli") == ["m"]


def test_score_is_higher_is_better_unlike_raw_bm25():
    r = indexed(Memory("a", "alpha beta", "en"), Memory("b", "alpha", "en"))
    hits = r.search(Query("q", "alpha", "en"), 10)
    assert all(h.score > 0 for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_a_rare_term_outranks_a_pile_of_common_ones():
    """Why there is no stopword list: idf already does that job."""
    memories = [Memory(f"f{i}", "this is a note that is not about anything", "en") for i in range(20)]
    memories.append(Memory("rare", "CODEMEM_EMBEDDING_MODEL resolves the encoder", "en"))
    r = indexed(*memories)
    assert top(r, "what happens if CODEMEM_EMBEDDING_MODEL is not set")[0] == "rare"


def test_reindexing_replaces_rather_than_appends():
    r = LexicalRetriever()
    suite = Suite(name="t", memories={"a": Memory("a", "alpha", "en")}, queries=[])
    r.index(suite)
    r.index(suite)
    assert top(r, "alpha") == ["a"]


def test_limit_is_honoured():
    r = indexed(*[Memory(f"m{i}", "alpha", "en") for i in range(10)])
    assert len(top(r, "alpha", limit=3)) == 3


def test_cross_lingual_hits_only_when_identifiers_survive_translation():
    """The measured shape of the lexical channel: it carries identifiers, not meaning."""
    r = indexed(
        Memory("id", "rsync --checksum is required or ninja sees no changes", "en"),
        Memory("prose", "the save directory follows the working directory", "en"),
    )
    assert top(r, "wsl icine kopyalarken ninja neden degisiklik gormuyor", "tr") == ["id"]
    assert top(r, "kayit dosyalari neden yanlis klasore gidiyor", "tr") == []
