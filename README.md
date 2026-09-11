# Keel

A keel is what keeps a vessel upright and tracking straight. This is one
`AGENTS.md` file that does the same for the assistants working in your
repository — start a project with it and the help stays help instead of turning
into volume.

Copy [`AGENTS.md`](AGENTS.md) into your repo, replace the project specifics
section with what is true of yours, and you are done. You can also use this
repository as a GitHub template, or fork it if you want to track changes as the
file grows.

## Saying you use it

If Keel is worth telling people about, the README is where they will see it, not
the commit log. Paste this near the top of yours:

```markdown
[![follows Keel](https://img.shields.io/badge/follows-Keel-1f6feb)](https://github.com/theomgdev/keel)
```

[![follows Keel](https://img.shields.io/badge/follows-Keel-1f6feb)](https://github.com/theomgdev/keel)

Nothing depends on it. The provenance that matters is already in your
`AGENTS.md`, which says where its general half came from, and that is the file
anyone debugging your project's habits will actually open.

## Why one file, and why this name for it

`AGENTS.md` is read by most assistants and belongs to none of them. The
alternative is what repositories are actually filling up with: a `CLAUDE.md`
next to a `CURSOR.md` next to a `GEMINI.md`, each one noise to everyone using a
different tool, all of them drifting out of sync with each other. One file is
the contract. Everything vendor-specific stays local and gitignored, and this
repo's [`.gitignore`](.gitignore) already does that for you.

## The one rule

Maximise the value to garbage ratio. Its measurable form is that everything you
write for a change — commit message, pull request body, comments, markdown —
has to come out shorter than the code that change contains. Two lines of code do
not get twenty lines of explanation. Along the time axis the same ratio reads
as value over time, so an hour spent re-deriving something already known is
garbage too.

Everything else in the file follows from that: keep diffs surgical, separate
inspecting from mutating, ground every symbol in real tool output, find root
causes before retrying, never claim a test result you did not observe, remember
that passing tests are not proof of correctness, and be able to defend the
change without the model.

## Where this comes from

Mostly not from taste. Nearly every rule traces to something maintainers are
actually complaining about — the projects that closed bug bounties or started
auto-closing outside pull requests, the research that named the three properties
of slop, the measurements showing duplicated code climbing and refactoring
collapsing. The file gets updated as that record grows, which is the
maintenance this repository exists to do.

The file itself was written with LLM assistance, reviewed and owned by
[@theomgdev](https://github.com/theomgdev), and every commit here carries an
`Assisted-by:` trailer saying so. A repository that asks for disclosure and does
not practise it is not worth copying.

<details>
<summary>Sources</summary>

- [Open source maintainers are drowning in AI-generated pull requests](https://thenewstack.io/ai-generated-code-crisis/) — The New Stack
- [AI is burning out the people who keep open source alive](https://www.coderabbit.ai/blog/ai-is-burning-out-the-people-who-keep-open-source-alive) — CodeRabbit
- [GitHub eyes restrictions on pull requests](https://www.infoworld.com/article/4127156/github-eyes-restrictions-on-pull-requests-to-rein-in-ai-based-code-deluge-on-maintainers.html) — InfoWorld
- [Agent pull requests are everywhere. Here's how to review them](https://github.blog/ai-and-ml/generative-ai/agent-pull-requests-are-everywhere-heres-how-to-review-them/) — GitHub Blog
- [AI Agents Cheat on Pull Requests. I Mined 327 of Them to Prove It](https://dev.to/moonrunnerkc/ai-agents-cheat-on-pull-requests-i-mined-327-of-them-to-prove-it-43ij) — DEV
- [Why Slop Matters](https://arxiv.org/abs/2601.06060) — Kommers et al., where the three prototypical properties are named
- [“We Permit the Use of AI, but […]”](https://arxiv.org/abs/2609.07542) — Hora, Robbes and Zacchiroli; 281 AI policies: 83% permit assisted code, 49% require disclosure, 67% require high human involvement, and no trailer convention has emerged
- [Assisted-by: How open source projects are drawing the line on AI contributions](https://allthingsopen.org/articles/open-source-ai-contributions-assisted-by-git-trailer-standard) — All Things Open, on the trailer and the DCO line
- [AI Coding Assistants](https://docs.kernel.org/process/coding-assistants.html) — Linux kernel: agents must not add `Signed-off-by`; `Assisted-by:` is the trailer
- [rust-lang/rust is adopting an LLM policy](https://blog.rust-lang.org/inside-rust/2026/08/05/rust-langrust-is-adopting-an-llm-policy/) — analyse, do not create; disclosure attaches to publication, not to private use
- [Fighting AI Slop](https://actualbudget.org/blog/fighting-ai-slop/) — Actual Budget's contributor policy
- [“An Endless Stream of AI Slop”](https://arxiv.org/abs/2603.27249) — Baltes, Cheong and Treude; 1,154 Reddit and Hacker News posts, the complaint record this file is built from
- [Recent Frontier Models Are Reward Hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/) — METR; o3 stubs the evaluator, overwrites equality, and being told not to cheat barely moves the rate
- [We Have a Package for You!](https://arxiv.org/abs/2406.10279) — Spracklen et al., USENIX Security '25; 576,000 samples, hallucinated package names as a supply-chain attack
- [The Maintainability Gap](https://www.gitclear.com/the_ai_code_quality_maintainability_gap) — GitClear; 623 million changes, refactoring −70%, duplicated blocks +81%, error-masking +47%
- [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://arxiv.org/abs/2507.09089) — METR RCT; 16 developers, 246 real tasks, 19% slower, believed they were 20% faster
- [2026 GenAI Code Security Report](https://www.veracode.com/blog/2026-genai-code-security-report-ai-risk/) — syntax pass rate ~95%, security pass rate still ~55%; XSS and log injection are the holes
- [Instruction Adherence in Coding Agent Configuration Files](https://arxiv.org/abs/2605.10039) — McMillan; file size, position and architecture did not move compliance, and extra instructions make the real ones harder to follow
- [A Taxonomy of Inefficiencies in LLM-Generated Code](https://arxiv.org/pdf/2503.06327) — arXiv
- [When to Stop? Towards Efficient Code Generation in LLMs with Excess Token Prevention](https://arxiv.org/pdf/2407.20042) — arXiv
- [What Is AI Slop? Detect & Prevent Low-Quality AI Code](https://larridin.com/developer-productivity-hub/what-is-ai-slop-detect-prevent-low-quality-ai-code) — Larridin
- [AI Code Looks Right. That's the Problem.](https://www.aviator.co/blog/how-to-avoid-ai-code-slop/) — Aviator
- [The most common thing that makes agentic code ugly is the overuse of comments](https://news.ycombinator.com/item?id=43929768) — Hacker News
- [Customize your AI-generated git commit messages](https://devblogs.microsoft.com/visualstudio/customize-your-ai-generated-git-commit-messages/) — Visual Studio Blog
- [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) — prior art, and the reason this one is tool-agnostic

</details>

## Contributing

If you have watched an assistant do something that wasted a reviewer's time and
the file does not already cover it, open an issue or a pull request. Real
complaints from real review queues are what this is built from. The file is also
held to its own rule, so anything added has to earn its length.

## License

[MIT](LICENSE). Copy it, change it, ship it.
