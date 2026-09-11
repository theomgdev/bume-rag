# Benchmark suites

A suite is a directory holding `memories.jsonl` and `queries.jsonl`. A memory is
`{id, text, language}`; a query is `{id, text, language, gold: [memory ids],
note}`. An empty `gold` marks a query the store should refuse to answer, and
`bume bench` scores those as abstention instead of ranking.

Language groups are derived, not declared. A query lands in `cross` when its gold
memories are not all in the query's own language, and in `en-en` or `tr-tr`
otherwise, so mislabelling a group means mislabelling a memory rather than
quietly tuning the split.

`seed` is a small suite over real notes from this machine, written to exercise
the harness rather than to rank anything: thirteen memories, nineteen queries,
all three groups plus two abstention cases. It is not a measurement and no
retrieval claim should ever rest on it. The two suites that decide things —
LongMemEval for comparability and roughly a hundred hand-labelled queries over
real codemem notes for our own workload — arrive with Phase 1.
