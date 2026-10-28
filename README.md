# Persona Experiment

A mini replication of **"Playing Pretend: Expert Personas Don't Improve Factual
Accuracy"** (Prompting Science Report 4, Wharton GAIL; [arXiv:2512.05858](https://arxiv.org/abs/2512.05858)),
plus a follow-up on what expert instructions cost.

**Question:** does telling a model it is an expert make it answer more accurately,
and what does it cost?
**Answer so far:** no accuracy gain. With reasoning on, the expert prompt can spend
more tokens and money (Ling +36% versus the neutral control at 5,000 questions),
but that cost is model-specific.

Four models answer the same 5,000 paired MMLU-Pro questions under five system
prompts: no prompt, a length-matched neutral instruction, and short, generic, and
long expert instructions.

## Method

- **Benchmark:** graduate-level questions sampled from the MMLU-Pro test set
  (physics, chemistry, biology, computer science, engineering, math, economics,
  law, history), seeded and reproducible (`seed=42`). Samples live in
  `data/question_samples/mmlu_pro_sample_<n>.json` for n in 50, 100, 500, 1000,
  2000, 2500, 5000; the runner defaults to the 50-question rung.
- **Models:** `anthropic/claude-haiku-5.5`, `deepseek/deepseek-v4-flash`,
  `inclusionai/ling-3.0-flash`, and `z-ai/glm-5.3-flash` through OpenRouter at
  5,000 questions, reasoning xhigh. The runner defaults to
  `inclusionai/ling-3.0-flash`, thinking off by default (pass
  `--reasoning-effort` to enable).
- **Design:** paired, so every question is answered under every condition and any
  difference is attributable to the system prompt. The five conditions are in
  `data/condition_variants/expert_variants.json`, passed with
  `--conditions-file`.
- **Outcomes:** answer accuracy, plus token usage (prompt and reasoning).
  Cost is tokens times price and speed is tokens per unit time, so tokens are the
  underlying quantity recorded for every call.

### Conditions

`data/condition_variants/expert_variants.json`, with measured system-prompt
lengths:

| Condition | Prompt | Tokens |
|---|---|---|
| baseline | *(none)* | 0 |
| neutral | "You are a helpful assistant. You are always glad to help with whatever question the user asks. You try to be accurate, you think the problem through carefully, and you answer clearly." | 38 |
| expert_short | "You are an expert." | 6 |
| expert | "You are a world-class expert in physics, chemistry, biology, computer science, engineering, mathematics, economics, law, and history. You have decades of experience and you never make mistakes." | 39 |
| expert_long | The expert prompt plus "You have decades of experience across academia and industry, you have published widely and taught at the highest level... Approach every question with the care and rigor of a seasoned specialist in the field." | 72 |

The neutral control is a non-expert instruction of the same length as the expert
prompt, so any difference between them reflects content rather than length.

## Results

Every number below comes from the run files listed in `tables/runs.csv`, and every
figure from `figures/`. Raw run files are not committed (1.4 GB of JSON); see
[Reproducing the tables and figures](#reproducing-the-tables-and-figures).

### Expert wording variants, 5,000 questions, reasoning xhigh

Four models, five conditions, the same 5,000 paired questions. Cells are accuracy
over all calls, with the number of answered calls in parentheses: a lower answered
count means more empty or truncated replies, which cost accuracy but say nothing
about knowledge.

| Model | baseline | neutral | expert_short | expert | expert_long |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 88.4% (4,984) | 87.9% (4,984) | 88.0% (4,985) | 88.0% (4,982) | 88.0% (4,987) |
| Deepseek V4 Flash | 77.9% (4,447) | 83.7% (4,695) | 83.2% (4,677) | 83.6% (4,667) | 83.0% (4,626) |
| Ling 3.0 Flash | 81.7% (4,726) | 81.9% (4,750) | 81.2% (4,696) | 80.8% (4,656) | 81.5% (4,683) |
| GLM 5.3 Flash | 84.7% (4,927) | 84.9% (4,920) | 85.1% (4,936) | 84.7% (4,923) | 84.9% (4,921) |

Mean reasoning tokens and total run cost per condition:

| Model | baseline | neutral | expert_short | expert | expert_long |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 882 / $2.39 | 899 / $2.45 | 934 / $2.52 | 863 / $2.36 | 970 / $2.66 |
| Deepseek V4 Flash | 7,047 / $1.09 | 4,901 / $0.76 | 5,162 / $0.80 | 5,080 / $0.79 | 5,343 / $0.84 |
| Ling 3.0 Flash | 2,158 / $1.04 | 1,923 / $0.94 | 2,363 / $1.11 | 2,608 / $1.21 | 2,562 / $1.19 |
| GLM 5.3 Flash | 1,258 / $1.26 | 1,279 / $1.28 | 1,218 / $1.22 | 1,294 / $1.29 | 1,314 / $1.31 |

Three things follow from the tables.

**No expert wording improves accuracy significantly.** Against the length-matched
neutral control, the generic expert prompt moves accuracy by +0.1 pp (Claude),
-0.1 pp (Deepseek), -1.1 pp (Ling, McNemar p = 0.002), and -0.2 pp (GLM). Claude's
shortest expert prompt is the best case at +0.2 pp. Those gains are noise-sized;
the two significant comparisons are both negative (Ling expert at -1.1 pp and
Deepseek expert_long at -0.8 pp, p = 0.043).

**The Deepseek baseline is a truncation artifact, not a persona effect.** 545 of
its 5,000 baseline replies hit the 32,768-token cap and returned nothing, far more
than any prompted condition (neutral truncates least, at 298). Its all-calls
accuracy therefore rises 5.9 pp from baseline to neutral (p < 1e-36) while answered-call accuracy barely
moves, and against neutral the expert prompt is -0.1 pp (p = 0.79). Treat any
claim that a system prompt helps Deepseek as an answered-rate effect.

**The expert prompt's token cost is model-specific.** Against neutral, reasoning
tokens change by +36% (Ling), +4% (Deepseek), +1% (GLM), and -4% (Claude). There is
no general "expert personas cost more" rule, only a consistent "expert personas do
not answer better" one.

![Cross-model comparison](figures/model-comparison-5000q.png)

Standalone versions of each panel are in `figures/model-comparison-5000q-accuracy.png`,
`-cost.png`, `-reasoning.png`, `-latency.png`, and `-accuracy-vs-cost.png`.
Per-run token distributions are in `figures/runs/`.

### Caveats

- The four models ran behind different upstream providers (Anthropic
  native, Baidu fp8 for Deepseek, open-inference fp4 for GLM, and Ling's default
  OpenRouter pool). Model and provider are therefore confounded in the cross-model
  panels; the within-model condition comparisons are paired and unaffected.
- One seeded draw per run at temperature 0 (seed 861), so
  run-to-run token use varies and the cost magnitude is approximate.
- Replies that hit the 32,768-token cap are recorded as blanks and never retried.
  That is a harness limit, not a model property, and it is why the answered count
  is reported beside every accuracy.
- Token effects are heavy-tailed: the mean increase is much larger than the median
  for the same condition.
- Latency is logged but is confounded by run order and provider load, so it is not
  a clean speed measurement.
- Results are about answer accuracy and token use. Personas can still change tone,
  style, and formatting, which this does not measure.

## Usage

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/):

```bash
make setup        # uv sync
```

### API key

`scripts/run_experiment.py` reads `OPENROUTER_API_KEY_PROJECT` from the
environment. Create a git-ignored `.env` with that variable and load it before
running:

```bash
set -a; . ./.env; set +a
```

### Run the experiment

```bash
# plan only, no network calls
.venv/bin/python scripts/run_experiment.py --dry-run --limit 2

# smoke test: 5 questions x 5 conditions = 25 calls
.venv/bin/python scripts/run_experiment.py \
  --conditions-file data/condition_variants/expert_variants.json --limit 5

# full run on the 50-question sample, reasoning off (the default)
.venv/bin/python scripts/run_experiment.py \
  --conditions-file data/condition_variants/expert_variants.json

# the writeup run: 5,000 questions, reasoning on
.venv/bin/python scripts/run_experiment.py \
  --questions-file data/question_samples/mmlu_pro_sample_5000.json \
  --conditions-file data/condition_variants/expert_variants.json \
  --reasoning-effort xhigh

# continue the most recent run for this model
.venv/bin/python scripts/run_experiment.py --resume
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--model` | `inclusionai/ling-3.0-flash` | OpenRouter model slug |
| `--questions-file` | `data/question_samples/mmlu_pro_sample_50.json` | question JSON to run |
| `--limit N` | all 50 | first N questions (N x conditions calls) |
| `--workers N` | 20 | calls in flight at once |
| `--max-tokens N` | 32768 | completion budget (thinking plus answer) |
| `--reasoning-effort` | `none` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; `none` disables thinking |
| `--reasoning-max-tokens N` | none | cap reasoning tokens instead of setting an effort |
| `--provider` | any | pin a specific upstream provider |
| `--temperature` / `--seed` | `0.0` / `861` | sampling |
| `--out FILE` / `--resume` | none | write an explicit file, or continue the newest run for the model (mutually exclusive) |
| `--conditions-file FILE` | the built-in four | JSON `{name: system_prompt}` mapping; the study uses `data/condition_variants/expert_variants.json` |
| `--dry-run` / `--force` | off | plan only / override the resume-compatibility check |

Tasks are scheduled question-major, so all conditions run interleaved across a run
rather than in contiguous blocks. That keeps latency from being confounded with
provider load drifting over time.

Output: each run writes one file named for its question count, reasoning level, and
start time (UTC): `results/{slug}__{count}q__{reasoning}__{timestamp}.json`, where
`{reasoning}` is the effort level (`none`, `high`, ...) or `maxtokens{N}`. `results/`
is git-ignored and holds the raw runs; `make report` turns them into the committed
tables and figures.

Behavior: retries 429/5xx and network errors with backoff (5 attempts, 180s
timeout), saves every 10 calls, aborts after 3 consecutive failures (exit 2), and
on Ctrl-C saves and exits 130. A single progress bar shows calls done, failures,
and cumulative cost. Empty or truncated answers are recorded as non-answers and
never retried.

### Workers

`--workers` is the number of calls in flight (default 20). The right value is
provider-bound; start at 20 and raise it while the progress bar's fail count stays
flat. `ling-3.0-flash` serves from a small provider pool and can rate-limit, so
watch for 429 backoff.

### Analyze results

```bash
.venv/bin/python scripts/analyze_results.py results/<file>.json                 # summary
.venv/bin/python scripts/analyze_results.py results/<file>.json --questions     # per question
.venv/bin/python scripts/analyze_results.py results/<file>.json --categories    # by MMLU category
.venv/bin/python scripts/analyze_results.py results/<file>.json --reasoning     # reasoning stats
.venv/bin/python scripts/analyze_results.py results/<file>.json --reference neutral  # comparisons vs neutral
.venv/bin/python scripts/analyze_results.py results/<file>.json --confidence 0.99  # 99% intervals
.venv/bin/python scripts/analyze_results.py A.json B.json --compare             # side by side
.venv/bin/python scripts/analyze_results.py results/<file>.json --json          # machine-readable
.venv/bin/python scripts/analyze_results.py results/<file>.json --format-errors out.tsv
.venv/bin/python scripts/analyze_results.py results/<file>.json --credit-format-errors
.venv/bin/python scripts/analyze_results.py results/<file>.json --no-significance
.venv/bin/python scripts/analyze_results.py results/<file>.json --plot out.png
```

Views combine; `--json` suppresses the text views, and partial (in-progress) files
are handled. The summary prints, in order:

- **Counts and accuracy** per condition: `Correct`, `Wrong`, `Blank(len)`,
  `Blank(stop)`, `FmtErr`, `Missing`, `Acc(ans)` (answered only), `Acc(all)`, and
  `Cost`.
- **Total cost** across conditions.
- **Token usage vs the reference:** mean prompt and reasoning tokens, the `R med` column
  (median reasoning tokens, far below the mean because these counts are skewed), the
  mean deltas (`dPrompt`, `dReason`), `dR med` (median of the per-question reasoning
  differences), and the reasoning-token ratio with a 95% interval and a sign-test
  p-value. The interval is a delta-method interval on the log mean ratio; the sign
  test asks whether the per-question differences move consistently in one direction,
  so it can be significant while the mean-ratio interval includes 1. Intervals are
  95% by default; `--confidence 0.99` widens them to 99%. With reasoning off the
  reasoning columns are zero and the ratio and sign-test p-value show `n/a`.
  Completion tokens are deliberately not
  reported: the visible answer is a single letter, and the provider's completion and
  reasoning fields disagree with each other (reasoning exceeds completion in a large
  fraction of calls), so reasoning tokens are the measurable quantity. The
  reference is baseline by default; `--reference neutral` uses the length-matched
  non-expert control, which isolates the expert wording effect. The
  paired accuracy test and answer-change table use the same reference.
- **Paired significance:** each condition against the reference (baseline unless
  `--reference` says otherwise) with the exact McNemar
  test on discordant pairs, showing `Fixed`, `Broke`, the discordant total, the net
  difference in percentage points, a 95% interval (or the level set by
  `--confidence`), and p. This is the only place the
  per-condition net is reported; the old standalone deltas line was removed because
  it duplicated this column. Printed by default; `--no-significance` hides it, and
  `--json` carries it under `paired_vs_reference` (with the chosen reference in the
  top-level `reference` field), and the accuracy delta as `delta_pp_vs_reference`.
- **Answer changes vs the reference** (answered pairs only): `Changed`, `W->R`, `R->W`,
  `W->W`, `Same`.

Entries that name no option are split two ways. `Blank(len)` and `Blank(stop)` are
both empty completions, separated by finish reason: `Blank(len)` hit the output cap
(`finish_reason: length`, the model was still thinking when it ran out of budget)
and `Blank(stop)` stopped with no content at all (`finish_reason: stop`). `FmtErr`
means the completion is non-blank but holds no option letter, for example
`2.5 gallons`. All three are kept out of wrong answers. The answer is never
recovered from the reasoning trace, since a real user reads the completion.

`--plot OUT.png` writes a figure of the reasoning-token distribution for one input
file (matplotlib, a declared dependency). A box-and-whiskers panel sits on top, and
below it there is one density panel per condition on a shared log axis, so the shapes
can be compared without overlapping curves:

- **Box and whiskers** for every condition on one log axis: the box is the IQR, the
  line is the median, the diamond is the mean, and dots are values beyond 1.5 IQR.
  Baseline sits at the top. This is the clearest read on median versus mean and on
  how far the tail runs.
- **Density**, one panel per condition, with that condition's median (solid) and mean
  (dashed) marked and the median and mean printed in the title.
- **Accuracy against cost**, one labeled point per condition with accuracy over all
  calls on the y axis and that condition's run cost on the x axis. Every call was
  paid for, so blanks and truncations count against the condition here; the summary
  table also reports the answered-calls rate.

Reasoning tokens are the only token metric plotted, for the reason above. It takes
exactly one input file.

`--format-errors OUT` dumps every `FmtErr` row to a tab-separated file with
columns condition, id, category, gold, the raw reply, `regex_letter`,
`would_be_correct`, finish reason, provider, reasoning tokens, and the question.
The `regex_letter` and `would_be_correct` columns match the reply against the
option texts by word-boundary
regex after lowercasing and stripping LaTeX, so `3.54 eV` matches an option written
`3.54 $\mathrm{eV}$`, `100` does not match inside `1000`, and `8.0` does not match
inside `18.0`. It is a heuristic, not a parser.

`--credit-format-errors` applies that match to scoring: a format error whose reply
matches the gold option counts as correct. The summary adds a `Credit` column and
a scoring banner, the per-category table follows the same rule, and `--json` adds a
`credit_format_errors` flag and a per-condition `credited` count.

### Compare models

`scripts/compare_models.py` selects the latest complete 5,000-question run for
each model and writes one combined four-panel PNG plus five standalone panel
PNGs: accuracy, total cost by condition, mean reasoning tokens, average call
latency, and accuracy against cost. It reads the run files directly, so no
preprocessing is needed.

```bash
.venv/bin/python scripts/compare_models.py
.venv/bin/python scripts/compare_models.py --list
.venv/bin/python scripts/compare_models.py --conditions baseline,expert
.venv/bin/python scripts/compare_models.py --models anthropic/claude-haiku-5.5,z-ai/glm-5.3-flash
```

The default output is `results/model-comparison-5000q.png`; the standalone panels
are written beside it as `-accuracy.png`, `-cost.png`, `-reasoning.png`,
`-latency.png`, and `-accuracy-vs-cost.png`. Use `--results-dir` to point at another results tree,
`--out` to choose the combined PNG path, and `--no-split` to skip the standalone
panels. Complete runs are required by default; `--allow-partial` includes in-progress
files and counts missing calls as wrong. `make report` (below) calls this script and
writes the figures into `figures/` instead.

### Reproducing the tables and figures

`scripts/make_report.py` rebuilds every artifact the writeup cites from the raw
runs:

```bash
make report        # or: .venv/bin/python scripts/make_report.py
```

It writes:

- `tables/runs.csv`: one row per experiment run and condition (correct, answered,
  blanks split by cap versus empty stop, format errors, missing, accuracy over
  answered calls and over all calls, prompt and reasoning token means, median
  reasoning tokens, cost). Provider probe files (`spike_*`) are skipped.
- `tables/model-comparison-5000q.csv`: the four-model table.
- `tables/paired-5000q.csv`: exact McNemar tests against baseline and against the
  neutral control.
- `figures/model-comparison-5000q.png` plus five standalone panels, and one
  token-distribution figure per complete run that recorded reasoning tokens in
  `figures/runs/`.

Point it at another tree with `--results-dir`, change the headline rung with
`--questions` and `--reasoning`, and skip the per-run plots with `--no-run-figures`.
The raw `results/` tree is not tracked: the run JSON files alone are over 1 GB and
the largest single file is 504 MB, so the committed artifacts are the small,
deterministic outputs above.

### Regenerate the question sample

`scripts/sample_mmlu.py` samples from the parquet and writes the question JSON. Its
default count is 50, and the default output name carries the count
(`data/question_samples/mmlu_pro_sample_<n>.json`) so a new rung does not overwrite
an existing one, and it confirms before writing (`--yes` skips the prompt):

```bash
# write the 50-question rung the runner reads by default
.venv/bin/python scripts/sample_mmlu.py --yes

# another dataset: custom paths, count, seed, categories, and column names
.venv/bin/python scripts/sample_mmlu.py \
  --input /path/to/other.parquet --output /path/to/sample.json \
  --n 100 --seed 7 --categories all --yes
```

`--categories all` skips filtering; a comma list keeps only those (for example
`--categories math,physics`). `--question-col`, `--options-col`, `--answer-col`,
and `--category-col` map the source columns; the answer column must hold a 0-based
index into the options. The default input is `data/mmlu_pro_test.parquet`.

## Development

```bash
make                # list targets
make setup          # create/sync the .venv (uv sync)
make test           # unit tests with coverage
make lint           # ruff check
make format         # ruff format
make format-noedit  # verify formatting without writing
make check          # lint, then test
make coverage-html  # write an HTML coverage report to htmlcov/
make precommit      # format-noedit + lint + test (run by the git hook)
make install-hooks  # point git at the tracked .githooks/pre-commit
make report         # regenerate tables/ and figures/ from results/
make clean          # remove caches
```

Tests live in `tests/` (stdlib `unittest`, run under `coverage`).

## Files

- `scripts/run_experiment.py`: OpenRouter runner (built-in fallback conditions;
  `--conditions-file` for the five-condition set)
- `scripts/analyze_results.py`: CLI analysis (summary, tokens, significance,
  per-question, per-category, reasoning stats, compare, format-error dump, `--json`)
- `scripts/compare_models.py`: cross-model four-panel comparison of 5,000-question runs
- `scripts/make_report.py`: regenerates `tables/` and `figures/` from the raw runs
- `scripts/run_data.py`: shared result loading and statistics used by the CLIs
- `scripts/sample_mmlu.py`: seeded sampler from the MMLU-Pro parquet (needs pyarrow)
- `data/question_samples/`: the question samples (`mmlu_pro_sample_<n>.json`)
- `data/condition_variants/`: the five system-prompt conditions
- `data/mmlu_pro_test.parquet`: source parquet (git-ignored)
- `tables/`: committed derived statistics (`runs.csv`, model comparison, paired tests)
- `figures/`: committed comparison and per-run figures
- `results/`: raw run JSON files (git-ignored)
- `tests/`: unit tests for the runner, sampler, analyzer, and report generator
- `Makefile`, `pyproject.toml`: tasks and project/config

The original local runner (`experiment.py`) and earlier cloud script
(`experiment_openrouter_think.py`) have been replaced by `scripts/run_experiment.py`.
