# Persona Experiment

A mini replication of **"Playing Pretend: Expert Personas Don't Improve Factual
Accuracy"** (Prompting Science Report 4, Wharton GAIL; [arXiv:2512.05858](https://arxiv.org/abs/2512.05858)),
plus a follow-up on what expert instructions cost.

**Question:** does telling a model it is an expert make it answer more accurately,
and what does it cost?
**Answer so far:** no accuracy gain. With reasoning on, expert instructions also
spend more tokens (and therefore more money and time). Low-knowledge personas both
cost more and hurt accuracy.

The work is split into two phases:

- **Phase 1** - the original four-condition replication: `baseline`, `expert`,
  `layperson`, `toddler`.
- **Phase 2** - expert-instruction variants: `baseline`, `neutral` (length-matched
  non-expert control), `expert_short`, `expert` (the phase 1 generic prompt),
  `expert_long`.

## Method

- **Benchmark:** graduate-level questions sampled from the MMLU-Pro test set
  (physics, chemistry, biology, computer science, engineering, math, economics,
  law, history), seeded and reproducible (`seed=42`). Samples live in
  `data/question_samples/mmlu_pro_sample_<n>.json` for n in 50, 100, 500, 1000,
  2000, 5000; the runner defaults to the 50-question rung.
- **Models:** the current runner defaults to `inclusionai/ling-3.0-flash` via
  OpenRouter, thinking off by default (pass `--reasoning-effort` to enable). The
  earliest results below come from local Ollama runs (`qwen3.5:9b`,
  `qwen3.5:0.8b`, temperature 0, thinking disabled).
- **Design:** paired, so every question is answered under every condition and any
  difference is attributable to the system prompt. Conditions are the built-in
  four unless `--conditions-file` supplies another set.
- **Outcomes:** answer accuracy, plus token usage (prompt and reasoning).
  Cost is tokens times price and speed is tokens per unit time, so tokens are the
  underlying quantity recorded for every call.

### Conditions

Built-in (phase 1):

| Condition | System prompt |
|---|---|
| baseline | *(none)* |
| expert | "You are a world-class expert in physics, chemistry, biology, computer science, engineering, mathematics, economics, law, and history. You have decades of experience and you never make mistakes." |
| layperson | "You are a layperson with no specialized training in any technical or academic subject." |
| toddler | "You are a toddler. You are still learning the basics of the world and have almost no knowledge of science, engineering, law, or mathematics." |

Phase 2 (`data/condition_variants/phase_2_conditions_expert_variants.json`), with
measured system-prompt lengths:

| Condition | System prompt tokens |
|---|---|
| baseline | 0 |
| neutral | 38 |
| expert_short | 6 ("You are an expert.") |
| expert | 39 (same prompt as phase 1) |
| expert_long | 72 |

The neutral control is a non-expert instruction of the same length as the expert
prompt, so any difference between them reflects content rather than length.

## Results

**Phase 2 (expert wording variants) has not been run yet.** Its conditions are in
`data/condition_variants/phase_2_conditions_expert_variants.json` and `results/phase2/`
is empty. The results below are phase 1 and the original local runs.

### Phase 1 on OpenRouter, 5,000 questions

Reasoning off (`--reasoning-effort none`): accuracy is flat across conditions
(59.8%, 59.7%, 59.2%, 59.8% for baseline, expert, layperson, toddler; all
McNemar p >= 0.07), and because completions are about two tokens the conditions
cost the same. Nothing to separate.

Reasoning on (`--reasoning-effort xhigh`), 5,000 questions:

| Condition | Accuracy | Net vs baseline | Reasoning tokens | Token ratio | Total cost |
|---|---|---|---|---|---|
| baseline | 82.0% | - | 2152 | 1.00 | $1.03 |
| expert | 81.6% | -0.4pp (p=0.31) | 2595 | 1.21 [1.16, 1.25] | $1.20 |
| layperson | 81.4% | -0.6pp (p=0.08) | 2294 | 1.07 [1.03, 1.10] | $1.09 |
| toddler | 81.0% | -0.9pp (p=0.01) | 2306 | 1.07 [1.03, 1.11] | $1.09 |

Read this as: the expert persona buys no accuracy and costs about 17% more, almost
entirely extra reasoning tokens (mean +443 per call, median only +22, so the mean
is tail-driven). The low-knowledge personas cost a little more and lose accuracy,
and the toddler loss is significant. The effect sizes are not fully stable across
runs of different size, so treat the token penalty as "real in direction, roughly
7 to 17% in magnitude".

### Original local Ollama replication (44 questions)

| Condition | qwen3.5:9b | qwen3.5:0.8b |
|---|---|---|
| baseline | **19/44 (43%)** | **10/44 (23%)** |
| expert | 20/44 (45%) | 9/44 (20%) |
| layperson | 18/44 (41%) | 9/44 (20%) |
| toddler | 18/44 (41%) | 8/44 (18%) |

Across both models and 88 answers the expert persona's net accuracy effect was
zero; the low-knowledge personas lowered accuracy. Archived under `results/old/`.
These runs used a 44-question sample that is no longer committed; the current
sample ladder starts at 50.

### Caveats

- Single model and single provider (OpenRouter, usually Novita), one seeded draw
  per run, temperature 0. Run-to-run token use varies, so the cost magnitude is
  approximate.
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
environment. The repo ships a git-ignored `.env`; load it before running:

```bash
set -a; . ./.env; set +a
```

### Run the experiment

```bash
# plan only, no network calls
.venv/bin/python scripts/run_experiment.py --dry-run --limit 2

# phase 1 smoke test: 5 questions x 4 conditions = 20 calls
.venv/bin/python scripts/run_experiment.py --limit 5

# phase 1 full run on the 50-question sample, reasoning off (the default)
.venv/bin/python scripts/run_experiment.py

# phase 1 with reasoning on
.venv/bin/python scripts/run_experiment.py --reasoning-effort xhigh

# phase 2: expert wording variants, 5,000 questions, reasoning on
.venv/bin/python scripts/run_experiment.py \
  --questions-file data/question_samples/mmlu_pro_sample_5000.json \
  --conditions-file data/condition_variants/phase_2_conditions_expert_variants.json \
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
| `--conditions-file FILE` | the built-in four | JSON `{name: system_prompt}` mapping to run instead of the defaults |
| `--dry-run` / `--force` | off | plan only / override the resume-compatibility check |

Tasks are scheduled question-major, so all conditions run interleaved across a run
rather than in contiguous blocks. That keeps latency from being confounded with
provider load drifting over time.

Output: each run writes one file named for its question count, reasoning level, and
start time (UTC): `results/{slug}__{count}q__{reasoning}__{timestamp}.json`, where
`{reasoning}` is the effort level (`none`, `high`, ...) or `maxtokens{N}`. `results/`
is git-ignored. Phase 1 results are kept in `results/phase1/` and phase 2 in
`results/phase2/`; write there with `--out results/phase2/<name>.json`.

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
- **Token usage vs baseline:** mean prompt and reasoning tokens, the `R med` column
  (median reasoning tokens, far below the mean because these counts are skewed), the
  mean deltas (`dPrompt`, `dReason`), `dR med` (median of the per-question reasoning
  differences), and the reasoning-token ratio with a 95% interval and a p-value. The
  interval is a delta-method interval on the log ratio; the p-value is the exact sign
  test on the per-question differences. With reasoning off the reasoning columns are
  zero and the ratio and p-value show `n/a`. Completion tokens are deliberately not
  reported: the visible answer is a single letter, and the provider's completion and
  reasoning fields disagree with each other (reasoning exceeds completion in a large
  fraction of calls), so reasoning tokens are the measurable quantity.
- **Paired significance:** each condition against baseline with the exact McNemar
  test on discordant pairs, showing `Fixed`, `Broke`, the discordant total, the net
  difference in percentage points, a 95% interval, and p. This is the only place the
  per-condition net is reported; the old standalone deltas line was removed because
  it duplicated this column. Printed by default; `--no-significance` hides it, and
  `--json` carries it under `paired_vs_baseline`.
- **Answer changes vs baseline** (answered pairs only): `Changed`, `W->R`, `R->W`,
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
The last two of those match the reply against the option texts by word-boundary
regex after lowercasing and stripping LaTeX, so `3.54 eV` matches an option written
`3.54 $\mathrm{eV}$`, `100` does not match inside `1000`, and `8.0` does not match
inside `18.0`. It is a heuristic, not a parser.

`--credit-format-errors` applies that match to scoring: a format error whose reply
matches the gold option counts as correct. The summary adds a `Credit` column and
a scoring banner, the per-category table follows the same rule, and `--json` adds a
`credit_format_errors` flag and a per-condition `credited` count.

### Regenerate the question sample

`scripts/sample_mmlu.py` samples from the parquet and writes the question JSON. Its
default count is 44, and the default output name carries the count
(`data/question_samples/mmlu_pro_sample_<n>.json`) so a new rung does not overwrite
an existing one, and it confirms before writing (`--yes` skips the prompt):

```bash
# write the 50-question rung the runner reads by default
.venv/bin/python scripts/sample_mmlu.py --n 50 --yes

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
make precommit      # format-noedit + lint + test (run by the git hook)
make install-hooks  # point git at the tracked .githooks/pre-commit
make clean          # remove caches
```

The Makefile header also mentions `check` and `coverage-html`; those two are not
actually defined, so use `precommit` and `coverage` directly. Tests live in
`tests/` (stdlib `unittest`, run under `coverage`).

## Files

- `scripts/run_experiment.py`: OpenRouter runner (built-in four conditions;
  `--conditions-file` for phase 2 sets)
- `scripts/analyze_results.py`: CLI analysis (summary, tokens, significance,
  per-question, per-category, reasoning stats, compare, format-error dump, `--json`)
- `scripts/sample_mmlu.py`: seeded sampler from the MMLU-Pro parquet (needs pyarrow)
- `data/question_samples/`: the question samples (`mmlu_pro_sample_<n>.json`)
- `data/condition_variants/`: phase 2 condition sets
- `data/mmlu_pro_test.parquet`: source parquet (git-ignored)
- `results/old/`: original local Ollama runs
- `results/phase1/`: phase 1 OpenRouter runs
- `results/phase2/`: phase 2 runs (empty so far)
- `tests/`: unit tests for the runner, sampler, and analyzer
- `Makefile`, `pyproject.toml`: tasks and project/config

The original local runner (`experiment.py`) and earlier cloud script
(`experiment_openrouter_think.py`) have been replaced by `scripts/run_experiment.py`.
