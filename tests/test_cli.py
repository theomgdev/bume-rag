import pytest

from bume_rag.cli import build_retriever, main
from bume_rag.dense import DenseRetriever
from bume_rag.embedding import HashingEmbedder, OpenAIEmbedder
from bume_rag.fusion import RRFRetriever
from bume_rag.lexical import LexicalRetriever


class Args:
    def __init__(self, retriever="hybrid", embedder="auto", model="m"):
        self.retriever = retriever
        self.embedder = embedder
        self.model = model


def no_key(monkeypatch):
    monkeypatch.setattr("bume_rag.cli.find_api_key", lambda *a, **k: None)


def a_key(monkeypatch):
    monkeypatch.setattr("bume_rag.cli.find_api_key", lambda *a, **k: "k")


def test_hybrid_is_the_default_and_uses_the_cloud_when_a_key_exists(monkeypatch):
    a_key(monkeypatch)
    retriever, warning = build_retriever(Args())
    assert isinstance(retriever, RRFRetriever)
    channels = retriever.channels
    assert isinstance(channels[0], LexicalRetriever)
    assert isinstance(channels[1].embedder, OpenAIEmbedder)
    assert warning is None


def test_a_missing_key_falls_back_to_offline_rather_than_failing(monkeypatch):
    """A first run has to work before anyone has read the configuration."""
    no_key(monkeypatch)
    retriever, warning = build_retriever(Args())
    assert isinstance(retriever.channels[1].embedder, HashingEmbedder)
    assert warning is not None


def test_the_fallback_says_it_is_a_fallback(monkeypatch):
    """Silently landing on it would look like the project retrieving badly."""
    no_key(monkeypatch)
    _, warning = build_retriever(Args())
    assert "OMNIROUTE_API_KEY" in warning
    assert "hashing" in warning


def test_asking_for_cloud_without_a_key_fails_loudly(monkeypatch):
    no_key(monkeypatch)
    with pytest.raises(SystemExit, match="OMNIROUTE_API_KEY"):
        build_retriever(Args(embedder="cloud"))


def test_hashing_is_selectable_even_when_a_key_exists(monkeypatch):
    a_key(monkeypatch)
    retriever, warning = build_retriever(Args(embedder="hashing"))
    assert isinstance(retriever.channels[1].embedder, HashingEmbedder)
    assert warning is None


def test_single_channels_are_selectable(monkeypatch):
    a_key(monkeypatch)
    assert isinstance(build_retriever(Args(retriever="lexical"))[0], LexicalRetriever)
    assert isinstance(build_retriever(Args(retriever="dense"))[0], DenseRetriever)


def test_the_lexical_channel_never_needs_a_key(monkeypatch):
    no_key(monkeypatch)
    retriever, warning = build_retriever(Args(retriever="lexical"))
    assert isinstance(retriever, LexicalRetriever)
    assert warning is None


def test_bench_runs_end_to_end_offline(monkeypatch, capsys):
    no_key(monkeypatch)
    assert main(["bench", "benchmarks/seed", "--embedder", "hashing"]) == 0
    out = capsys.readouterr().out
    for group in ("en-en", "tr-tr", "cross", "pooled"):
        assert group in out


def test_an_unknown_retriever_is_rejected():
    with pytest.raises(SystemExit):
        main(["bench", "--retriever", "telepathy"])
