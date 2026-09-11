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

DEFAULT_SUITES = Path("benchmarks")
CACHE_PATH = Path.home() / ".bume" / "embeddings.json"

CHANNELS = ("hybrid", "lexical", "dense", "no-op")


def build_embedder(args: argparse.Namespace) -> tuple[object, str | None]:
    """Cloud when a key is reachable, hashing when it is not.

    The fallback is deliberately loud: it keeps a first run working, but its
    retrieval is bad enough that silently landing on it would look like the
    project failing rather than the key being absent.
    """
    if args.embedder == "hashing":
        return HashingEmbedder(), None
    if find_api_key() is None:
        if args.embedder == "cloud":
            raise SystemExit(
                f"--embedder cloud needs {KEY_VARIABLE} in the environment"
                f" or in ~/.omniroute/.env"
            )
        return HashingEmbedder(), (
            f"no {KEY_VARIABLE} found, so the dense channel is the offline hashing"
            f" stand-in. It matches spelling rather than meaning and cannot cross a"
            f" language boundary; set the key for real numbers."
        )
    return OpenAIEmbedder(model=args.model, cache=EmbeddingCache(CACHE_PATH)), None


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
        choices=("auto", "cloud", "hashing"),
        default="auto",
        help="auto uses the cloud when a key is reachable and hashing when it is not",
    )
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
