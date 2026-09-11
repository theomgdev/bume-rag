# bume-rag

A retrieval-augmented memory tool: a Python library with a CLI on top, meant to
be the store an agent reads from at the start of a turn and writes to at the end
of it.

Status: early development. What exists is a benchmark harness and a hybrid
retriever — BM25 over SQLite FTS5 fused with embeddings by reciprocal rank. The
contract is in [`AGENTS.md`](AGENTS.md) and the phase order in
[`PLAN.md`](PLAN.md).

## What it is for

Agents lose everything between sessions, and the usual fix — embed some chunks,
take the top k, paste them into the prompt — retrieves by surface similarity and
returns the wrong thing often enough to be worth replacing. The goal is a memory
that holds what was expensive to discover and hands back the part that changes
the next decision.

Two things follow from that goal. Retrieval quality is measured, not asserted:
methods land behind a benchmark that says whether they helped, and the ones that
did not help get deleted rather than kept behind a flag. And the cost that
matters is the context budget of the next turn, so precision is worth more than
recall.

## Running it

`uv run bume bench` scores every suite under `benchmarks/` and prints a table per
language group; `uv run --group dev pytest -q` runs the tests. Neither needs
configuration to start: without an API key the dense channel falls back to an
offline stand-in and says so.

That fallback is a convenience, not the product. `pip install bume-rag[local]`
gets a real encoder — harrier-oss-v1, fetched on first use with a progress bar
and never contacted again once cached — or set `OMNIROUTE_API_KEY` to use a
hosted one. `--embedder` chooses explicitly and `--dimensions` truncates the
vector when storage matters more than the last point of accuracy.

The corpus is cross-lingual on purpose — Turkish queries against English memories
are the normal case here, not an edge one — so the table breaks the groups out
rather than pooling them into an average that would hide the case most likely to
fail. There is no runtime dependency: FTS5 ships with SQLite and an embedding
call is one HTTP request.

## License

[MIT](LICENSE).
