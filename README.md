# bume-rag

A retrieval-augmented memory tool: a Python library with a CLI on top, meant to
be the store an agent reads from at the start of a turn and writes to at the end
of it.

Status: early development. There is no code in this repository yet — only the
contract in [`AGENTS.md`](AGENTS.md) and this statement of intent. Nothing here
has been benchmarked, so nothing here claims a number.

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

## Building it

Nothing to build or install yet. When there is, the instructions live in
`AGENTS.md` next to everything else an assistant needs.

## License

[MIT](LICENSE).
