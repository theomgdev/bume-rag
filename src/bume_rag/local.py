"""The local encoder: harrier-oss-v1, downloaded on first use.

Nobody should have to find a URL to make this work, so the weights are fetched
automatically with a progress bar and cached where every other Hugging Face tool
looks for them. The download is the slow part of a first run, not a setup step
the user performs beforehand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_REPO = "microsoft/harrier-oss-v1-0.6b"

# Queries carry a one-sentence instruction and documents carry none. The
# asymmetry is how the model was trained; config_sentence_transformers.json
# ships the prompts but leaves default_prompt_name null, so a plain encode()
# silently adds nothing and the model card warns that accuracy drops.
RETRIEVAL_INSTRUCTION = (
    "Instruct: Given a question, retrieve the note that answers it\nQuery: "
)


def ensure_weights(repo: str = DEFAULT_REPO, quiet: bool = False) -> str:
    """Download the model if it is not cached, and return a usable local path.

    Returns a forward-slash path deliberately. transformers 5.0.0.dev0 splits
    checkpoint paths on "/" at modeling_utils.py:4217, so a Windows path raises
    IndexError before the weights are ever read. Verified: the same snapshot
    loads in 24 s when the separator is forward slashes.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "the local encoder needs the 'local' extra: pip install bume-rag[local]"
        ) from error

    # Cached weights are used without asking the network anything. Hitting the
    # hub on every run costs seconds, fails offline, and is the kind of latency
    # that makes a local encoder feel worse than the cloud it replaces.
    try:
        return Path(snapshot_download(repo_id=repo, local_files_only=True)).as_posix()
    except Exception:
        pass

    if not quiet:
        print(f"fetching {repo} (about 1.2 GB on first run)...", file=sys.stderr, flush=True)
    path = snapshot_download(repo_id=repo)
    return Path(path).as_posix()


class LocalEmbedder:
    """harrier-oss-v1 through sentence-transformers.

    Dimensions are truncatable: the vector is cut and renormalised, which costs
    little down to 512 and is the lever for trading accuracy against storage.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        dimensions: int | None = None,
        instruction: str = RETRIEVAL_INSTRUCTION,
        device: str | None = None,
        batch_size: int = 8,
        quiet: bool = False,
    ) -> None:
        self.repo = repo
        self.dimensions = dimensions
        self.instruction = instruction
        self.device = device
        self.batch_size = batch_size
        self.quiet = quiet
        self._model = None

    @property
    def name(self) -> str:
        suffix = f"@{self.dimensions}" if self.dimensions else ""
        return f"{self.repo.rsplit('/', 1)[-1]}{suffix}"

    @property
    def model(self):
        """Loaded on first use, because importing torch costs seconds."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "the local encoder needs the 'local' extra:"
                    " pip install bume-rag[local]"
                ) from error
            path = ensure_weights(self.repo, quiet=self.quiet)
            self._model = SentenceTransformer(
                path,
                device=self.device,
                model_kwargs={"dtype": "auto"},
                local_files_only=True,
            )
        return self._model

    def _encode(self, texts: Sequence[str], prompt: str | None) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=self.dimensions is None,
            show_progress_bar=False,
        )
        out = [list(map(float, v)) for v in vectors]
        return [truncate(v, self.dimensions) for v in out] if self.dimensions else out

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Documents: no instruction, as the model was trained."""
        return self._encode(texts, None)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self.instruction)


def truncate(vector: list[float], dimensions: int | None) -> list[float]:
    """Cut to `dimensions` and renormalise, so cosine stays comparable."""
    if not dimensions or dimensions >= len(vector):
        return vector
    cut = vector[:dimensions]
    length = sum(x * x for x in cut) ** 0.5
    return [x / length for x in cut] if length else cut
