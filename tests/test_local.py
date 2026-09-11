import pytest

from bume_rag.corpus import Memory, Query, Suite
from bume_rag.dense import DenseRetriever
from bume_rag.local import RETRIEVAL_INSTRUCTION, LocalEmbedder, ensure_weights, truncate


class StubModel:
    """Records how it was called, so the instruction asymmetry is observable."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, prompt=None, **kwargs):
        self.calls.append((list(texts), prompt))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def stubbed(monkeypatch, **kwargs):
    embedder = LocalEmbedder(**kwargs)
    model = StubModel()
    embedder._model = model
    return embedder, model


def test_documents_get_no_instruction(monkeypatch):
    """The model was trained with instructions on queries only."""
    embedder, model = stubbed(monkeypatch)
    embedder.embed(["a document"])
    assert model.calls[0][1] is None


def test_queries_get_the_instruction(monkeypatch):
    embedder, model = stubbed(monkeypatch)
    embedder.embed_queries(["a query"])
    assert model.calls[0][1] == RETRIEVAL_INSTRUCTION


def test_the_instruction_is_configurable(monkeypatch):
    embedder, model = stubbed(monkeypatch, instruction="Instruct: custom\nQuery: ")
    embedder.embed_queries(["q"])
    assert model.calls[0][1] == "Instruct: custom\nQuery: "


def test_dense_asks_the_embedder_for_a_query_vector(monkeypatch):
    """Encoding a query like a document is the bug that costs recall silently."""
    embedder, model = stubbed(monkeypatch)
    suite = Suite(name="t", memories={"m": Memory("m", "doc", "en")}, queries=[])
    dense = DenseRetriever(embedder)
    dense.index(suite)
    dense.search(Query("q", "question", "tr"), 5)
    assert model.calls[0][1] is None, "document side"
    assert model.calls[1][1] == RETRIEVAL_INSTRUCTION, "query side"


def test_dense_still_works_with_a_symmetric_embedder():
    """Cloud and hashing embedders have no embed_queries; they must not break."""
    class Symmetric:
        name = "sym"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    suite = Suite(name="t", memories={"m": Memory("m", "doc", "en")}, queries=[])
    dense = DenseRetriever(Symmetric())
    dense.index(suite)
    assert dense.search(Query("q", "question", "en"), 5)[0].memory_id == "m"


def test_truncate_renormalises_so_cosine_stays_comparable():
    cut = truncate([3.0, 4.0, 100.0], 2)
    assert cut == pytest.approx([0.6, 0.8])
    assert sum(x * x for x in cut) == pytest.approx(1.0)


def test_truncate_is_a_no_op_when_not_asked_or_already_short():
    assert truncate([1.0, 2.0], None) == [1.0, 2.0]
    assert truncate([1.0, 2.0], 5) == [1.0, 2.0]


def test_truncate_handles_a_zero_vector():
    assert truncate([0.0, 0.0, 0.0], 2) == [0.0, 0.0]


def test_dimensions_appear_in_the_name():
    assert LocalEmbedder(repo="a/b").name == "b"
    assert LocalEmbedder(repo="a/b", dimensions=256).name == "b@256"


def test_truncation_is_applied_to_the_output(monkeypatch):
    embedder, _ = stubbed(monkeypatch, dimensions=2)
    assert len(embedder.embed(["x"])[0]) == 2


def test_a_missing_dependency_names_the_extra(monkeypatch):
    """Nobody should have to guess what to install."""
    import builtins

    real = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(RuntimeError, match=r"bume-rag\[local\]"):
        ensure_weights("a/b")


def test_weights_path_uses_forward_slashes(monkeypatch):
    """transformers 5.0.0.dev0 splits checkpoint paths on "/" at
    modeling_utils.py:4217 and raises IndexError on a Windows separator before
    the weights are read. Verified: the same snapshot loads with forward slashes.

    The module is faked rather than imported, because this test has to run in CI
    where the optional dependency is absent.
    """
    import sys
    import types

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda **kw: r"C:\cache\models--x\snapshots\abc"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert ensure_weights("a/b", quiet=True) == "C:/cache/models--x/snapshots/abc"


def fake_hub(monkeypatch, calls, cached):
    import sys
    import types

    def snapshot_download(repo_id, local_files_only=False, **kw):
        calls.append(local_files_only)
        if local_files_only and not cached:
            raise OSError("not cached")
        return rf"C:\cache\{repo_id}"

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_cached_weights_never_touch_the_network(monkeypatch, capsys):
    """Asking the hub on every run costs seconds and fails without a connection."""
    calls = []
    fake_hub(monkeypatch, calls, cached=True)
    assert ensure_weights("a/b") == "C:/cache/a/b"
    assert calls == [True], "only the offline lookup should have run"
    assert capsys.readouterr().err == "", "nothing to report when nothing is fetched"


def test_a_missing_cache_falls_through_to_a_download(monkeypatch, capsys):
    calls = []
    fake_hub(monkeypatch, calls, cached=False)
    assert ensure_weights("a/b") == "C:/cache/a/b"
    assert calls == [True, False], "offline attempt first, then the real download"
    assert "1.2 GB" in capsys.readouterr().err, "a long download announces itself"
