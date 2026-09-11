import json
import urllib.error

import pytest

from bume_rag.embedding import EmbeddingCache, HashingEmbedder, OpenAIEmbedder, find_api_key


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return urllib.error.HTTPError("u", code, "msg", {}, None)


def patch_urlopen(monkeypatch, responses):
    calls = []

    def fake(request, timeout=None):
        calls.append(json.loads(request.data))
        outcome = responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    monkeypatch.setattr("bume_rag.embedding.urllib.request.urlopen", fake)
    return calls


def body(vectors, omit_first_index=True):
    data = []
    for i, v in enumerate(vectors):
        item = {"object": "embedding", "embedding": v}
        if not (omit_first_index and i == 0):
            item["index"] = i
        data.append(item)
    return {"object": "list", "data": data}


def test_first_batch_element_has_no_index_and_that_must_not_break_it(monkeypatch):
    """Measured against the real endpoint: data[0] comes back without "index".

    Sorting the batch on that field raises KeyError, which is how this was found.
    """
    payload = body([[1.0], [2.0], [3.0]])
    assert "index" not in payload["data"][0]
    embedder = OpenAIEmbedder(api_key="k", cache=None)
    patch_urlopen(monkeypatch, [payload])
    assert embedder.embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_vectors_keep_input_order(monkeypatch):
    embedder = OpenAIEmbedder(api_key="k", cache=None)
    patch_urlopen(monkeypatch, [body([[1.0], [2.0], [3.0]])])
    assert embedder.embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_an_out_of_order_response_raises_rather_than_misaligning(monkeypatch):
    """Silently mismatched vectors would look like bad retrieval, not a bug."""
    embedder = OpenAIEmbedder(api_key="k", cache=None)
    payload = body([[1.0], [2.0]])
    payload["data"][1]["index"] = 7
    patch_urlopen(monkeypatch, [payload])
    with pytest.raises(ValueError, match="out of order"):
        embedder.embed(["a", "b"])


def test_429_is_retried_with_backoff(monkeypatch):
    slept = []
    embedder = OpenAIEmbedder(api_key="k", cache=None, sleep=slept.append)
    patch_urlopen(monkeypatch, [http_error(429), http_error(429), body([[1.0]])])
    assert embedder.embed(["a"]) == [[1.0]]
    assert slept == [1, 2]


def test_a_non_429_error_is_not_retried(monkeypatch):
    embedder = OpenAIEmbedder(api_key="k", cache=None, sleep=lambda s: None)
    patch_urlopen(monkeypatch, [http_error(410)])
    with pytest.raises(urllib.error.HTTPError):
        embedder.embed(["a"])


def test_retries_give_up_rather_than_looping_forever(monkeypatch):
    embedder = OpenAIEmbedder(api_key="k", cache=None, max_attempts=3, sleep=lambda s: None)
    patch_urlopen(monkeypatch, [http_error(429)] * 3)
    with pytest.raises(urllib.error.HTTPError):
        embedder.embed(["a"])


def test_a_missing_key_names_the_offline_option(monkeypatch):
    """api_key=None means "look it up", so a real key on the machine must not leak in."""
    monkeypatch.setattr("bume_rag.embedding.find_api_key", lambda *a, **k: None)
    embedder = OpenAIEmbedder(api_key=None, cache=None)
    assert embedder.api_key is None
    with pytest.raises(RuntimeError, match="hashing"):
        embedder.embed(["a"])


def test_cache_means_the_second_run_makes_no_request(tmp_path, monkeypatch):
    cache = EmbeddingCache(tmp_path / "c.json")
    embedder = OpenAIEmbedder(api_key="k", cache=cache)
    calls = patch_urlopen(monkeypatch, [body([[1.0]])])
    assert embedder.embed(["a"]) == [[1.0]]
    assert embedder.embed(["a"]) == [[1.0]]
    assert len(calls) == 1


def test_cache_survives_a_restart(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    first = OpenAIEmbedder(api_key="k", cache=EmbeddingCache(path))
    patch_urlopen(monkeypatch, [body([[1.0]])])
    first.embed(["a"])
    second = OpenAIEmbedder(api_key="k", cache=EmbeddingCache(path))
    calls = patch_urlopen(monkeypatch, [])
    assert second.embed(["a"]) == [[1.0]]
    assert calls == []


def test_only_uncached_texts_are_sent(tmp_path, monkeypatch):
    cache = EmbeddingCache(tmp_path / "c.json")
    embedder = OpenAIEmbedder(api_key="k", cache=cache)
    patch_urlopen(monkeypatch, [body([[1.0]])])
    embedder.embed(["a"])
    calls = patch_urlopen(monkeypatch, [body([[2.0]])])
    assert embedder.embed(["a", "b"]) == [[1.0], [2.0]]
    assert calls[0]["input"] == ["b"]


def test_duplicate_texts_are_sent_once(tmp_path, monkeypatch):
    embedder = OpenAIEmbedder(api_key="k", cache=EmbeddingCache(tmp_path / "c.json"))
    calls = patch_urlopen(monkeypatch, [body([[1.0]])])
    assert embedder.embed(["a", "a"]) == [[1.0], [1.0]]
    assert calls[0]["input"] == ["a"]


def test_batches_respect_the_batch_size(tmp_path, monkeypatch):
    embedder = OpenAIEmbedder(api_key="k", cache=EmbeddingCache(tmp_path / "c.json"), batch_size=2)
    calls = patch_urlopen(monkeypatch, [body([[1.0], [2.0]]), body([[3.0]])])
    embedder.embed(["a", "b", "c"])
    assert [len(c["input"]) for c in calls] == [2, 1]


def test_cache_key_separates_models(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.json")
    cache.put("m1", "text", [1.0])
    assert cache.get("m2", "text") is None


def test_env_var_wins_over_the_file(monkeypatch):
    monkeypatch.setenv("BUME_TEST_KEY", "from-env")
    assert find_api_key("BUME_TEST_KEY") == "from-env"


def test_hashing_embedder_is_deterministic_and_offline():
    e = HashingEmbedder()
    assert e.embed(["hello"]) == e.embed(["hello"])
    assert e.embed(["hello"]) != e.embed(["goodbye"])
    assert len(e.embed(["hello"])[0]) == 256
