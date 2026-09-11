"""Embedding providers.

Cloud by default because it is the setting that retrieves best, and offline
whenever a key is absent, so a first run works before anyone has read the
configuration. Only the standard library is imported here: an embedder is one
HTTP call, and pulling a vendor SDK in to make it would cost a dependency per
provider.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

DEFAULT_BASE_URL = "http://localhost:20128/v1"
DEFAULT_MODEL = "gemini/gemini-embedding-001"
KEY_VARIABLE = "OMNIROUTE_API_KEY"


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def find_api_key(variable: str = KEY_VARIABLE) -> str | None:
    """The environment first, then `~/.omniroute/.env`, which is where the CLI puts it."""
    key = os.environ.get(variable)
    if key:
        return key.strip()
    env_file = Path.home() / ".omniroute" / ".env"
    try:
        text = env_file.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    match = re.search(rf"^\s*{re.escape(variable)}\s*=\s*(\S+)", text, re.MULTILINE)
    return match.group(1).strip().strip("\"'") if match else None


class EmbeddingCache:
    """Keyed by model and content hash, so re-running a benchmark is free.

    Without this the rate limiter below makes iteration unaffordable, and a
    cache keyed on the text itself would store the corpus twice over.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, list[float]] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(model: str, text: str) -> str:
        return f"{model}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def get(self, model: str, text: str) -> list[float] | None:
        return self.entries.get(self.key(model, text))

    def put(self, model: str, text: str, vector: list[float]) -> None:
        self.entries[self.key(model, text)] = vector

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries), encoding="utf-8")


class OpenAIEmbedder:
    """Any OpenAI-compatible `/embeddings` endpoint.

    Measured against OmniRoute: 429 is routine rather than exceptional — a 212
    text corpus took six backoffs and 278 s to embed — so retrying is the normal
    path and anything calling this needs the cache.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
        max_attempts: int = 8,
        timeout: float = 300.0,
        sleep=time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else find_api_key()
        self.cache = cache
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.sleep = sleep

    @property
    def name(self) -> str:
        return self.model

    def _post(self, texts: Sequence[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)["data"]
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == self.max_attempts - 1:
                    raise
                self.sleep(2**attempt)
        # The first element of a batch comes back with no "index" field at all,
        # so sorting on it raises and trusting it would silently misalign the
        # rest. Position is authoritative; index is checked when present.
        for position, item in enumerate(data):
            if item.get("index", position) != position:
                raise ValueError(f"embedding {position} returned out of order")
        return [item["embedding"] for item in data]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError(
                f"no API key: set {KEY_VARIABLE} or use --embedder hashing for an offline run"
            )
        known: dict[str, list[float]] = {}
        pending = []
        for text in dict.fromkeys(texts):
            cached = self.cache.get(self.model, text) if self.cache else None
            if cached is None:
                pending.append(text)
            else:
                known[text] = cached
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            for text, vector in zip(chunk, self._post(chunk)):
                known[text] = vector
                if self.cache:
                    self.cache.put(self.model, text, vector)
        if self.cache and pending:
            self.cache.save()
        return [known[t] for t in texts]


class HashingEmbedder:
    """A deterministic offline stand-in with no model behind it.

    It exists so the dense code path is exercised in CI without a network or a
    download, and it retrieves badly on purpose: character trigrams hashed into
    buckets match spelling, not meaning, so it cannot cross a language boundary.
    Never compare a retrieval method against this and call the result a finding.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            lowered = f"  {text.lower()}  "
            for i in range(len(lowered) - 2):
                trigram = lowered[i : i + 3]
                digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[bucket] += 1.0 if digest[4] & 1 else -1.0
            vectors.append(vector)
        return vectors
