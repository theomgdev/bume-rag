"""The `bume` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bume_rag import bench
from bume_rag.corpus import load_suite
from bume_rag.dense import DenseRetriever
from bume_rag.embedding import (
    DEFAULT_MODEL,
    KEY_VARIABLE,
    EmbeddingCache,
    HashingEmbedder,
    OpenAIEmbedder,
    find_api_key,
)
from bume_rag.fusion import RRFRetriever
from bume_rag.lexical import LexicalRetriever
from bume_rag.local import DEFAULT_REPO, LocalEmbedder

DEFAULT_SUITES = Path("benchmarks")
CACHE_PATH = Path.home() / ".bume" / "embeddings.json"

CHANNELS = ("hybrid", "lexical", "dense", "no-op")


def local_is_available() -> bool:
    """Whether the optional dependency is installed, not whether weights exist.

    Weights download on first use; the import is the thing that cannot be fixed
    at runtime.
    """
    from importlib.util import find_spec

    return find_spec("sentence_transformers") is not None


def build_embedder(args: argparse.Namespace) -> tuple[object, str | None]:
    """Local first, because it needs no key and retrieves well.

    `auto` prefers the local encoder when its dependency is installed, falls
    back to the cloud when a key is reachable, and lands on hashing only when
    neither is available. The hashing fallback is deliberately loud: it keeps a
    first run working, but silently landing on it would look like the project
    retrieving badly rather than nothing being configured.
    """
    if args.embedder == "hashing":
        return HashingEmbedder(), None
    if args.embedder == "local":
        return LocalEmbedder(repo=args.model_local, dimensions=args.dimensions), None
    if args.embedder == "cloud":
        if find_api_key() is None:
            raise SystemExit(
                f"--embedder cloud needs {KEY_VARIABLE} in the environment"
                f" or in ~/.omniroute/.env"
            )
        return OpenAIEmbedder(model=args.model, cache=EmbeddingCache(CACHE_PATH)), None
    if local_is_available():
        return LocalEmbedder(repo=args.model_local, dimensions=args.dimensions), None
    if find_api_key() is not None:
        return OpenAIEmbedder(model=args.model, cache=EmbeddingCache(CACHE_PATH)), None
    return HashingEmbedder(), (
        "no local encoder installed and no API key found, so the dense channel is"
        " the offline hashing stand-in. It matches spelling rather than meaning and"
        " cannot cross a language boundary. Install bume-rag[local] for the real"
        f" encoder, or set {KEY_VARIABLE} to use the cloud."
    )


def build_retriever(args: argparse.Namespace):
    if args.retriever == "no-op":
        from bume_rag.retriever import NoOpRetriever

        return NoOpRetriever(), None
    if args.retriever == "lexical":
        return LexicalRetriever(), None
    embedder, warning = build_embedder(args)
    dense = DenseRetriever(embedder)
    if args.retriever == "dense":
        return dense, warning
    return RRFRetriever([LexicalRetriever(), dense]), warning


def _bench(args: argparse.Namespace) -> int:
    directories = [Path(s) for s in args.suite] or sorted(
        p for p in DEFAULT_SUITES.iterdir() if (p / "queries.jsonl").exists()
    )
    if not directories:
        raise SystemExit(f"no suites found under {DEFAULT_SUITES}")
    retriever, warning = build_retriever(args)
    if warning:
        print(f"warning: {warning}\n", file=sys.stderr)
    for directory in directories:
        report = bench.run(retriever, load_suite(directory), k=args.k)
        print(bench.format_report(report))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bume")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run a retriever over benchmark suites")
    b.add_argument("suite", nargs="*", help="suite directories (default: every one under benchmarks/)")
    b.add_argument("-k", type=int, default=10, help="cut-off for recall and delivery (default 10)")
    b.add_argument("--retriever", choices=CHANNELS, default="hybrid")
    b.add_argument(
        "--embedder",
        choices=("auto", "local", "cloud", "hashing"),
        default="auto",
        help="auto prefers local, then cloud, then the offline stand-in",
    )
    b.add_argument("--model", default=DEFAULT_MODEL, help="cloud embedding model id")
    b.add_argument("--model-local", default=DEFAULT_REPO, help="local encoder repo id")
    b.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="truncate embeddings to this many dimensions (default: the model's own)",
    )
    b.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
