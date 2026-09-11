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

## The suite that decides things

It does not exist yet, and building it needs a decision rather than code.

The workload we serve is a memory store full of real working notes, so the
corpus has to be real working notes. All 224 on this machine are in English —
they are written for models, and the house style is English — which means the
`cross` group can be harvested directly by writing Turkish queries against them,
while `tr-tr` memories have to be authored. That makes `cross` the group that
mirrors actual use and `tr-tr` the least representative one, and it is why a
strong `tr-tr` number must never stand in for cross-lingual health.

The obstacle is that this repository is public. A scan for keys, tokens, paths,
emails and hosts found no credentials of any kind, but 25 notes carry a home
directory path, three carry email addresses, and the set spans client projects
alongside open-source ones. Three ways out, none of them free:

- a `bume corpus build` generator that reads a local memory store, with the
  corpus gitignored — nothing is published and nobody else can reproduce a
  number exactly;
- publish only the notes from work that was already public, which is a smaller
  and less varied corpus;
- redact and publish everything, which is the strongest benchmark and the
  largest disclosure.

Do not build it until that is settled, and do not quietly pick one.

Whatever lands must also avoid the trap that a previous attempt fell into.
Adding the remaining notes as distractors looked like a harder benchmark and was
actually a broken one: the `seed` memories had been written *from* those notes,
so the distractors were near-duplicates of the gold, and the dense channel was
scored wrong for returning a correct answer that carried a different id. If gold
and distractors share a source, the benchmark measures id-matching rather than
retrieval.

LongMemEval is the other suite, for comparability with published work. Also not
wired up yet.
