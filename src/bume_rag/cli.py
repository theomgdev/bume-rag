"""The `bume` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from bume_rag import bench
from bume_rag.corpus import load_suite
from bume_rag.lexical import LexicalRetriever
from bume_rag.retriever import NoOpRetriever

RETRIEVERS = {"no-op": NoOpRetriever, "lexical": LexicalRetriever}
DEFAULT_SUITES = Path("benchmarks")


def _bench(args: argparse.Namespace) -> int:
    directories = [Path(s) for s in args.suite] or sorted(
        p for p in DEFAULT_SUITES.iterdir() if (p / "queries.jsonl").exists()
    )
    if not directories:
        raise SystemExit(f"no suites found under {DEFAULT_SUITES}")
    retriever = RETRIEVERS[args.retriever]()
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
    b.add_argument("--retriever", choices=sorted(RETRIEVERS), default="lexical")
    b.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
