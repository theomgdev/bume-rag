# Implementation plan

This is the order the work happens in and the reason for that order. It is not a
promise about dates, and every number in it that came from someone else's paper
is labelled as theirs. Ours go in commit messages as we measure them.

## The thing being optimised

An agent reads from the store at the start of a turn and writes at the end. What
we are minimising is the cost of that read to the turn that follows it, which is
not the same as maximising how much we return. A chunk that does not change the
next decision is not free: input length alone degrades LLM accuracy by 13.9-85%
even when retrieval is perfect and the irrelevant tokens are masked out
(arXiv:2510.05381), and individually distracting passages cost measurable
accuracy on top of that (arXiv:2505.06914). So the headline metric is answer
accuracy against tokens delivered, and ranking metrics are diagnostics under it.

The store is multilingual and cross-lingual by default. Queries arrive in Turkish
against memories written in English, and that combination is the normal case
rather than an edge one, so it is a property of every phase below rather than a
section of its own.

## Phase 0 — measure before building

Nothing lands before there is something to measure it with, because the project
rule is that a method enters on a benchmark and not on the paper that proposed
it. Phase 0 is the benchmark harness, a no-op retriever, and the CLI skeleton.

Two corpora. LongMemEval (arXiv:2410.10813, 500 questions, five abilities:
extraction, multi-session, temporal, knowledge update, abstention) gives
comparability with published work. Roughly 100 hand-labelled queries over real
codemem notes from this machine give us the workload we actually serve, which is
short authored fragments dense with identifiers and `file:line` references. The
two disagree by design; when they do, ours decides and LongMemEval explains.

Neither is built yet, and the second one has a problem the first does not: this
repository is public and those notes are somebody's real work. A scan of the 224
notes on this machine found no keys or tokens, but 25 carry a home directory
path, three carry email addresses, and they span client projects as well as open
source ones. The options are a generator that reads a local memory store so the
corpus is never committed, publishing only the notes that were already public
work, or redacting everything and publishing it. That choice is the user's and it
blocks the corpus, not the harness — everything below is measured on `seed`,
which is thirteen memories and exists to exercise the code rather than to rank
anything.

Our set is cross-lingual, because the workload is. Users ask in Turkish about
notes written in English, so a query and the memory that answers it routinely
share no surface tokens at all. That case has to be labelled in the corpus from
the first commit rather than added once monolingual numbers look good — a
retriever tuned on same-language pairs will score well and then fail in normal
use. The set therefore carries three groups sized deliberately: same-language
English, same-language Turkish, and cross-lingual TR query against EN memory. The
harness reports every metric per group as well as pooled, because a pooled
average will hide a cross-lingual collapse behind two healthy monolingual
numbers.

The harness reports nDCG@10, Recall@k and MRR, plus tokens delivered at the
chosen cut and end-to-end latency. It also records abstention behaviour, because
LongMemEval scores it and because a store that cannot say "nothing here" pollutes
the turn it was meant to help.

Done when: `bume bench` runs both suites against a retriever that returns
nothing, and prints a table of zeros without crashing. **Landed** in `4015e3d`
against `seed`; the two real suites are still outstanding.

## Phase 1 — the honest baseline

SQLite with FTS5 for the lexical channel, a vector column for the dense channel,
fused with reciprocal rank fusion at k=60. This is deliberately the same shape as
MemX (arXiv:2603.16171), which is the closest published sibling — local-first,
SQLite, vector + keyword + RRF — and which reports LongMemEval Hit@5 of 51.6% and
MRR 0.380 at fact granularity, with temporal and multi-session below 43.6%. That
is the number to beat, and it is a far more honest target than the 88-93% figures
from systems with a hosted database behind them.

The lexical channel is not optional here and not a legacy concession. On labelled
queries BM25 scores nDCG@10 0.6611 against dense 0.6451, and RRF of the two
reaches 0.6940; each channel rescues queries the other loses entirely
(sesen.ai, 300 queries). On identifier-heavy corpora BM25 beats SOTA dense
outright (arXiv:2604.01733). Our notes are identifier-heavy.

Multilingual pulls the two channels apart, and the fusion has to be tuned knowing
that. Lexical matching degrades to near nothing across a language boundary except
on the identifiers that survive translation, while the dense channel carries
whatever cross-lingual signal exists. So the same RRF weighting cannot be right
for both same-language and cross-lingual queries, and Phase 1 measures them
separately before picking one. The observed failure in codemem is exactly this:
FTS and text overlap dominate its scoring, so a Turkish query against English
notes returns `sem:0` and falls back to nothing useful unless the caller happens
to include English domain terms.

Every model in the stack must be multilingual, not the English default, and
whatever prefix or instruction scheme it was trained with has to be applied.
codemem embeds without E5's `query:` / `passage:` prefixes and pays for it in
cross-lingual recall; the same mistake is available with any instruction-tuned
encoder. If a reranker enters in Phase 2 it must be multilingual too; an English
cross-encoder over Turkish queries is the domain mismatch of arXiv:2608.03860
with the language axis added.

Embeddings come from OmniRoute at `http://localhost:20128/v1`, OpenAI-compatible,
key from `OMNIROUTE_API_KEY` — never committed, never stored in the memory DB.
Verified working this session: `gemini/gemini-embedding-001` returns 3072 dims;
`openai/text-embedding-3-small` answered 429 under load; `nvidia/nv-embedqa-e5-v5`
answered 410. CI must not depend on any of them: rate limits and dead model IDs
make them non-deterministic, and embedding 212 texts cost six backoffs and 278 s.

The local encoder is `microsoft/harrier-oss-v1-0.6b`: 0.6B parameters, 1024 dims,
a 32k-token window, MIT, 94 languages including Turkish, and MTEB v2 69.0 on the
multilingual board. It replaces `multilingual-e5-small` in the plan because it
removes the reason the cloud had to be the default — a local model that competes
with the hosted one turns the API key into a preference rather than a
requirement. The 1.2 GB of weights are too large for CI, so the hashing stand-in
stays as the no-download path and the two are not the same thing: one is a real
encoder, the other exists to exercise the code.

It has two properties that silently cost accuracy if missed, both the same shape
as the E5 prefix bug in codemem. Queries need a one-sentence instruction —
`config_sentence_transformers.json` ships `web_search_query` and friends but sets
`default_prompt_name` to null, so plain `encode()` adds nothing and the model
card says performance degrades. Documents take no instruction at all, so the
asymmetry is the thing to get right. And pooling is last-token, not mean; the
config declares it, but anything that reimplements pooling has to match.

GGUF is a deliverable rather than a note: `llama.cpp` runtimes reach hardware and
languages that torch does not, and Q8_0 is 610 MiB against 1.12 GiB for BF16.
It lands as its own phase behind the same rule as everything else — the
quantisation that ships is the one measured on our suites, not the one with the
best table elsewhere. Quantisation error and retrieval quality are different
questions, and published throughput numbers say nothing about either.

The GPU is shared, so `nvidia-smi` gets checked before anything loads a model
locally, and the local path must keep working on CPU.

Done when: Phase 0 harness reports a real number for hybrid, dense-only and
lexical-only, broken out per language group, and the commit message carries all
of them. **Landed** across `ac9dc12`, `1f6b603` and `b210410`.

### Where Phase 1 actually got to

Everything here is nDCG@10 on `seed`, which is thirteen memories and twenty
queries. It is enough to show a channel is broken and nowhere near enough to
rank two that both work, so these are directional and the real corpus replaces
them.

| retriever | pooled | cross |
| --- | --- | --- |
| no-op | 0.0000 | 0.0000 |
| lexical (BM25) | 0.7812 | 0.5000 |
| hybrid, hashing stand-in | 0.8519 | 0.6389 |
| dense, harrier local | 0.8788 | 0.6363 |
| hybrid, harrier local | 0.8860 | 0.7196 |
| hybrid, cloud embeddings | 0.9590 | 0.8770 |

Three things are unresolved and should not be guessed at later.

Whether the lexical channel earns its place is still open. Fusing it with cloud
embeddings changed nothing at all — RRF scored identically to dense alone — while
fusing it with the local encoder lifted the cross-lingual group from 0.6363 to
0.7196. An attempt to settle it on a harder corpus, by adding the other 212 notes
as distractors, produced a contaminated result: the seed memories were written
*from* those notes, so the distractors are near-duplicates of the gold and dense
was marked wrong for returning a correct answer under a different id. Measured
cosine between the rank-1 result and the gold text was 0.90, 0.83, 0.75. A clean
corpus is the only thing that answers this.

Abstention is unsolved and no threshold will solve it. The query nothing in the
corpus answers scores *above* three genuine answers on both channels — BM25 2.269
against gold scores of 1.867 to 2.359, and cosine 0.5627 against a genuine range
of 0.5854 to 0.8654. The false positive sits inside the true distribution on
both, which is the argument for ranked-list truncation in Phase 3 rather than a
cutoff.

The cloud still beats the local encoder on the cross-lingual group, 0.8770 to
0.6363, and that is the group that matters most here. Recall@10 is 1.0 for both,
so the gold is always retrieved and merely ranked worse. Whether that survives a
real corpus decides whether the API key is optional in practice or only in
principle.

## Phase 2 — contextual indexing and reranking

The single highest-leverage published change is contextual retrieval: an LLM
writes one or two sentences situating each unit in its source before it is
embedded *and* before it is indexed lexically. Anthropic measured top-20 failure
falling 5.7% → 3.7% with contextual embeddings, → 2.9% adding contextual BM25,
→ 1.9% adding a cross-encoder rerank. Those are their corpora, not ours, and our
units are already short authored notes rather than fragments torn out of a
document, so the gain may be much smaller here. That is exactly why it is a
phase with a measurement and not an assumption.

Generation goes through OmniRoute. `scala-ingenii` resolved to `claude-opus-5`
when tested this session; the write path pins a model explicitly rather than
relying on whatever the alias points at that week.

Reranking is a candidate, not a given: an MS MARCO-trained cross-encoder *reduced*
precision on a scientific corpus because domain mismatch outweighed the stronger
query-passage interaction (arXiv:2608.03860). It ships only if it wins here.

Done when: contextual and non-contextual indexes are compared on both suites. If
contextualisation does not win, the code for it is deleted rather than left
behind a flag.

## Phase 3 — deciding how much to return

Fixed top-k is the wrong shape for a precision-first store, because the right
number of results is a property of the score distribution and not a constant.
Ranked-list truncation — autocut, elbow and relatives — produced more factual
answers than fixed k in arXiv:2506.19512. This is where the precision rule stops
being a slogan and becomes a function.

Paired with it: separating evidence extraction from policy execution. On
MemoryAgentBench FactConsolidation every one of 22 reported systems scored ≤7% on
multi-hop despite the task stating its own freshness rule, and the fix was not
better retrieval but splitting "find the evidence" from "apply the answer
policy": +10.8pp average, +21pp at 262K context, with the policy executor itself
worth only 2.0pp of that (arXiv:2606.01435). Whether that survives contact with
our workload is a Phase 3 measurement.

If the cross-lingual group is still the weak one after Phase 2, this is where
query-side translation gets tested: send the query to both channels in its own
language and in the corpus language, and fuse. It is deliberately later than the
encoder and the fusion weights, because it adds a model call to every read and
should only buy its latency if a cheaper fix has already failed. Note that
query expansion measured *limited* benefit for precise queries while
contextualisation gave consistent gains (arXiv:2604.01733), so the expectation
here is modest.

## Phase 3.5 — GGUF

The local encoder in Phase 1 is a torch checkpoint, which means a multi-gigabyte
install to run a 0.6B model and no path onto hardware torch does not serve.
`llama.cpp` runtimes take GGUF, and quantisation makes the weights incidental:
Q8_0 is 610 MiB against 1.12 GiB for BF16.

GGUF is the second backend and ONNX is not a third. Both would need the same
work — a runtime, a tokeniser path, last-token pooling reimplemented, and a
correctness check against the torch vectors — and two of them means every future
change is made three times. GGUF wins because it is where quantisation below
8-bit actually lives, because `llama.cpp` reaches CPUs and GPUs that
onnxruntime's provider matrix does not, and because it is the format the person
running this already uses. ONNX earns its place if something concrete demands
it, not because it exists.

Q8_0 is the default without a sweep. Quantisation error at 8 bits is small
enough to treat as settled — the interesting question is whether it holds for
*embeddings*, where the output is a 1024-dimensional direction rather than a
token choice, so one comparison against the torch vectors on our suites answers
it. If Q8_0 holds, it ships and nothing else is measured. If it does not, that
is a finding worth the time. Producing a table of every quantisation is
leaderboard work and this is not a leaderboard.

The instruction prefix and last-token pooling have to survive the port, because
a runtime that pools differently produces vectors that look fine and rank wrong.

## Phase 4 — the write path

Origin gets bound when a memory is written, not inferred when it is read. This is
not speculative hardening: content-based and lineage-based trust are both
malleable, because an agent's own summarisation, a trusted-tool echo, and
manufactured corroboration all launder an untrusted origin into a trusted-looking
one, with up to 68% attack success against existing defences (arXiv:2606.24322).
A store whose notes steer later decisions needs the cheap half of that — record
where a memory came from at write time, and never let a later summarisation
upgrade it.

Consolidation and forgetting also live here. Localised maintenance measured more
cost-efficient than global reorganisation across 12 memory systems
(arXiv:2606.24775), so superseding is an edit to one record, not a reindex.

## What is deliberately not in the plan

No knowledge graph until a benchmark asks for one. GraphRAG frequently
underperforms vanilla RAG on real tasks (arXiv:2506.05690), Mem0's graph variant
bought about 2% over flat Mem0, and the one clear win — HippoRAG's up-to-20% — is
specifically multi-hop over entities. If our multi-hop numbers stall in Phase 3,
that is the evidence that opens this door. Nothing else does.

No chunker until the corpus needs one. Memories arrive as short authored units;
optimal chunking is task-dependent to the point of reversing sign between
in-corpus and in-document retrieval (arXiv:2602.16974).

No agentic grep-versus-vector rewrite on the strength of either 2026 paper. They
disagree: grep generally beat vector on a LongMemEval sample (arXiv:2605.15184),
while semantic search beat a grep subagent 65.2% to 46.2% at half the cost on
SWE-QA, with 41.8% of the agentic failures happening silently at the planner
hand-off (arXiv:2608.01507). The harness settles it for our workload or it stays
unsettled.

No second implementation kept alongside a winner. When a comparison answers, the
loser is deleted.

No English-only model anywhere in the stack, including in CI. That is not a
future feature to bolt on; a monolingual encoder makes the cross-lingual group
unmeasurable, and unmeasured is how it stays broken.

## Invariants

Every phase adds its own test command and its runtime to `AGENTS.md` in the same
change that adds the code. Anything running in CI runs on CPU without network.
Benchmark numbers go in commit messages. Claims about a run that did not happen
do not go anywhere.
