# Persona Experiment

A mini replication of **"Playing Pretend: Expert Personas Don't Improve Factual Accuracy"**
(Prompting Science Report 4, Wharton GAIL Generative AI Lab; [arXiv:2512.05858](https://arxiv.org/abs/2512.05858)).

**Question:** does telling a model it is an expert make it answer more accurately?
**Answer (this experiment):** no. Personas change behavior, not capability.

The results below are from the original local Ollama runs (`qwen3.5:9b`,
`qwen3.5:0.8b`). The experiment now runs through `scripts/run_experiment.py`
against OpenRouter models with configurable thinking; see [Usage](#usage).

## Method

- **Benchmark:** 44 graduate-level questions sampled from the MMLU-Pro test set
  (science, engineering, law, economics, math, history, biology, chemistry,
  physics), seeded and reproducible (`seed=42`). Committed at
  `data/mmlu_pro_sample.json`.
- **Models:** `qwen3.5:9b` and `qwen3.5:0.8b` via Ollama for the results below
  (temperature 0, thinking disabled). The current runner defaults to
  `inclusionai/ling-3.1-flash` on OpenRouter with thinking enabled.
- **Design:** paired — every question is answered under all four conditions, so
  any accuracy difference is attributable to the system prompt.

### Conditions

| Condition | System prompt |
|---|---|
| baseline | *(none)* |
| expert | "You are a world-class expert in physics, chemistry, biology, computer science, engineering, mathematics, economics, law, and history. You have decades of experience and you never make mistakes." |
| layperson | "You are a layperson with no specialized training in any technical or academic subject." |
| toddler | "You are a toddler. You are still learning the basics of the world and have almost no knowledge of science, engineering, law, or mathematics." |

## Results

### Accuracy by condition (44 questions)

| Condition | qwen3.5:9b | qwen3.5:0.8b |
|---|---|---|
| baseline | **19/44 (43%)** | **10/44 (23%)** |
| expert | 20/44 (45%) | 9/44 (20%) |
| layperson | 18/44 (41%) | 9/44 (20%) |
| toddler | 18/44 (41%) | 8/44 (18%) |

### Deltas vs baseline

| Condition | qwen3.5:9b | qwen3.5:0.8b |
|---|---|---|
| expert | +2pp | −3pp |
| layperson | −2pp | −3pp |
| toddler | −2pp | −5pp |

### Interpretation

- On the 9b model, the "world-class expert who never makes mistakes" prompt
  changed answers on **5 of 44** questions and flipped exactly **1** from wrong
  to right. The model was still wrong on 24 of 44.
- On the 0.8b model, the same expert prompt **broke two previously-correct
  answers** and lowered accuracy by 3pp.
- Across both models and 88 question-answers, the expert persona's net effect
  on accuracy was **zero**.
- Personas are not inert: the toddler persona changed 12 answers on the 9b
  model and broke two answers that had been correct (Q27, Q30). Negative-
  capability personas actively degrade performance, matching the study.
- The direction matches the original study: expert personas do not improve
  factual accuracy; low-knowledge personas can hurt it.

### Caveats (read before citing)

- Small sample (44 questions, single seeded draw), single run per condition,
  temperature 0. The ±2-5pp deltas are within noise at this n.
- Results are for accuracy of answers only. Personas can legitimately change
  tone, style, and formatting — the study's explicit caveat.
- Local 9b/0.8b models are far below frontier capability; the absolute scores
  are not the point, the persona effect (or lack of it) is.

## Usage

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). Build the environment:

```bash
make setup        # uv sync
```

### Run the experiment

`scripts/run_experiment.py` runs the four conditions over the sample against
OpenRouter. It reads `OPENROUTER_API_KEY` from the environment or `~/.bashrc`.

```bash
# plan only, no network calls
.venv/bin/python scripts/run_experiment.py --dry-run --limit 2

# smoke test: 5 questions x 4 conditions = 20 calls
.venv/bin/python scripts/run_experiment.py --limit 5 --workers 8

# full run: 44 questions x 4 conditions = 176 calls
.venv/bin/python scripts/run_experiment.py --workers 8

# thinking off
.venv/bin/python scripts/run_experiment.py --reasoning-effort none

# continue the most recent run for this model
.venv/bin/python scripts/run_experiment.py --resume
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--model` | `inclusionai/ling-3.0-flash` | OpenRouter model slug |
| `--questions-file` | `data/mmlu_pro_sample.json` | question JSON to run |
| `--limit N` | all 44 | first N questions (N x 4 calls) |
| `--workers N` | 8 | calls in flight at once |
| `--max-tokens N` | 32768 | completion budget (thinking plus answer) |
| `--reasoning-effort` | `high` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; `none` disables thinking |
| `--reasoning-max-tokens N` | none | cap reasoning tokens instead of setting an effort |
| `--provider` | any | pin a specific upstream provider |
| `--temperature` / `--seed` | `0.0` / `861` | sampling |
| `--out FILE` / `--resume` | none | write an explicit file, or continue the newest run for the model (mutually exclusive) |
| `--dry-run` / `--force` | off | plan only / override the resume-compatibility check |

Each run writes one file named for its question count, reasoning level, and
start time (UTC): `results/{slug}__{count}q__{reasoning}__{timestamp}.json`,
where `{count}` is the number of questions and `{reasoning}` is the effort level
(`none`, `high`, and so on) or `maxtokens{N}` for the max-tokens mode.
`results/` is git-ignored, so outputs stay local.

The runner retries 429/5xx responses with backoff (5 attempts), saves every 10
calls, aborts after 3 consecutive failures (exit 2), and on Ctrl-C saves and
exits 130. A live progress bar shows overall and per-condition progress, rate,
and the running fail count. Empty or truncated answers are recorded as
non-answers and never retried.

### Workers

`--workers` is the number of calls in flight (default 8). The right value is
provider-bound and has not been tuned for `ling-3.1-flash`, which rate-limits
intermittently (429s) on its upstream pool. Start at 8 and raise it while the
progress bar's fail count stays flat.

### Analyze results

```bash
.venv/bin/python scripts/analyze_results.py results/<file>.json              # summary
.venv/bin/python scripts/analyze_results.py results/<file>.json --questions  # per question
.venv/bin/python scripts/analyze_results.py results/<file>.json --categories # by MMLU category
.venv/bin/python scripts/analyze_results.py results/<file>.json --reasoning  # reasoning stats
.venv/bin/python scripts/analyze_results.py A.json B.json --compare          # side by side
```

Views combine, and `--json` emits a machine-readable summary. Partial
(in-progress) files are handled, and non-answers (`?`) are counted separately
from wrong answers.

### Regenerate the question sample

`scripts/sample_mmlu.py` samples from a parquet and writes the question JSON.
The default output name carries the count (`data/mmlu_pro_sample_<n>.json`) so it
does not overwrite the committed `data/mmlu_pro_sample.json` the runner reads,
and it confirms before writing (`--yes` skips the prompt):

```bash
# reproduce the committed sample (study categories, seed=42, 44 questions)
.venv/bin/python scripts/sample_mmlu.py

# write straight to the file the runner reads
.venv/bin/python scripts/sample_mmlu.py -o data/mmlu_pro_sample.json --yes
```

For another dataset, pass the paths, count, seed, and categories, and remap the
column names (defaults match the MMLU-Pro schema):

```bash
.venv/bin/python scripts/sample_mmlu.py \
  --input /path/to/other.parquet --output /path/to/sample.json \
  --n 100 --seed 7 --categories all --yes
```

`--categories all` skips category filtering; a comma list keeps only those
categories (for example `--categories math,physics`). `--question-col`,
`--options-col`, `--answer-col`, and `--category-col` map the source columns,
and the answer column must hold a 0-based index into the options.

## Development

```bash
make                # list targets
make setup          # create/sync the .venv (uv sync)
make test           # unit tests with coverage
make coverage-html  # HTML coverage report in htmlcov/
make lint           # ruff check
make format         # ruff format
make check          # lint + test
make precommit      # format-check + lint + test
make install-hooks  # enable the tracked .githooks/pre-commit hook
```

Tests live in `tests/` (stdlib `unittest`, executed under `coverage`).

## Files

- `scripts/run_experiment.py`: OpenRouter runner for the four-condition protocol
- `scripts/analyze_results.py`: CLI analysis of results files (summary, deltas, per-question, per-category, reasoning stats, compare, `--json`)
- `scripts/sample_mmlu.py`: seeded sampler from the MMLU-Pro parquet (needs pyarrow)
- `data/mmlu_pro_sample.json`: the 44 questions used
- `tests/test_run_experiment.py`: unit tests for the runner
- `Makefile`: developer tasks (test, lint, format, coverage, hooks)
- `results/`: run outputs (git-ignored); the original local result files are archived in `results/old/`

The original local runner (`experiment.py`) and the earlier cloud script
(`experiment_openrouter_think.py`) have been replaced by
`scripts/run_experiment.py`, which also absorbed the OpenRouter re-run.
