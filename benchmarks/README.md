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
the harness rather than to rank anything: thirteen memories, twenty queries, all
three groups plus two abstention cases. It is not a measurement and no retrieval
claim should ever rest on it — but it is written in real orthography, because an
ASCII-typed Turkish corpus scored a perfect 1.0 while the lexical channel was
shredding every accented word. One query repeats another without diacritics,
which is how people actually type, and both must hit.

The suites that decide things — LongMemEval for comparability, and hand-labelled
queries over real codemem notes for our own workload — are not here yet. All 212
real notes on this machine are in English, so the `cross` group can be harvested
by writing Turkish queries against them while `tr-tr` memories have to be
authored. That makes `cross` the group that mirrors actual use and `tr-tr` the
least representative one.
