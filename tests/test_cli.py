import pytest

from bume_rag.cli import build_retriever, main
from bume_rag.dense import DenseRetriever
from bume_rag.embedding import HashingEmbedder, OpenAIEmbedder
from bume_rag.fusion import RRFRetriever
from bume_rag.lexical import LexicalRetriever
from bume_rag.local import LocalEmbedder


class Args:
    def __init__(self, retriever="hybrid", embedder="auto", model="m"):
        self.retriever = retriever
        self.embedder = embedder
        self.model = model
        self.model_local = "some/repo"
        self.dimensions = None


def no_key(monkeypatch):
    monkeypatch.setattr("bume_rag.cli.find_api_key", lambda *a, **k: None)


def a_key(monkeypatch):
    monkeypatch.setattr("bume_rag.cli.find_api_key", lambda *a, **k: "k")


def local(monkeypatch, available):
    monkeypatch.setattr("bume_rag.cli.local_is_available", lambda: available)


def test_hybrid_is_the_default_and_prefers_local_over_cloud(monkeypatch):
    """Local needs no key and retrieves well, so it wins when installed."""
    a_key(monkeypatch)
    local(monkeypatch, True)
    retriever, warning = build_retriever(Args())
    assert isinstance(retriever, RRFRetriever)
    assert isinstance(retriever.channels[0], LexicalRetriever)
    assert isinstance(retriever.channels[1].embedder, LocalEmbedder)
    assert warning is None


def test_cloud_is_used_when_local_is_not_installed(monkeypatch):
    a_key(monkeypatch)
    local(monkeypatch, False)
    retriever, warning = build_retriever(Args())
    assert isinstance(retriever.channels[1].embedder, OpenAIEmbedder)
    assert warning is None


def test_neither_available_falls_back_to_offline_rather_than_failing(monkeypatch):
    """A first run has to work before anyone has read the configuration."""
    no_key(monkeypatch)
    local(monkeypatch, False)
    retriever, warning = build_retriever(Args())
    assert isinstance(retriever.channels[1].embedder, HashingEmbedder)
    assert warning is not None


def test_the_fallback_names_both_ways_out(monkeypatch):
    """Silently landing on it would look like the project retrieving badly."""
    no_key(monkeypatch)
    local(monkeypatch, False)
    _, warning = build_retriever(Args())
    assert "OMNIROUTE_API_KEY" in warning
    assert "bume-rag[local]" in warning


def test_local_works_with_no_key_at_all(monkeypatch):
    """The point of the local encoder: an API key becomes a preference."""
    no_key(monkeypatch)
    local(monkeypatch, True)
    retriever, warning = build_retriever(Args())
    assert isinstance(retriever.channels[1].embedder, LocalEmbedder)
    assert warning is None


def test_dimensions_reach_the_embedder(monkeypatch):
    a_key(monkeypatch)
    local(monkeypatch, True)
    args = Args()
    args.dimensions = 256
    retriever, _ = build_retriever(args)
    assert retriever.channels[1].embedder.dimensions == 256


def test_asking_for_cloud_without_a_key_fails_loudly(monkeypatch):
    no_key(monkeypatch)
    with pytest.raises(SystemExit, match="OMNIROUTE_API_KEY"):
        build_retriever(Args(embedder="cloud"))


def test_hashing_is_selectable_even_when_everything_else_exists(monkeypatch):
    a_key(monkeypatch)
    local(monkeypatch, True)
    retriever, warning = build_retriever(Args(embedder="hashing"))
    assert isinstance(retriever.channels[1].embedder, HashingEmbedder)
    assert warning is None


def test_single_channels_are_selectable(monkeypatch):
    a_key(monkeypatch)
    local(monkeypatch, False)
    assert isinstance(build_retriever(Args(retriever="lexical"))[0], LexicalRetriever)
    assert isinstance(build_retriever(Args(retriever="dense"))[0], DenseRetriever)


def test_the_lexical_channel_never_needs_a_key(monkeypatch):
    no_key(monkeypatch)
    local(monkeypatch, False)
    retriever, warning = build_retriever(Args(retriever="lexical"))
    assert isinstance(retriever, LexicalRetriever)
    assert warning is None


def test_bench_runs_end_to_end_offline(monkeypatch, capsys):
    no_key(monkeypatch)
    local(monkeypatch, False)
    assert main(["bench", "benchmarks/seed", "--embedder", "hashing"]) == 0
    out = capsys.readouterr().out
    for group in ("en-en", "tr-tr", "cross", "pooled"):
        assert group in out


def test_an_unknown_retriever_is_rejected():
    with pytest.raises(SystemExit):
        main(["bench", "--retriever", "telepathy"])
