#!/usr/bin/env python3
"""Regenerate committed tables and figures from the raw results tree.

The raw run JSON files under results/ stay out of git. This script derives the
small, reviewable artifacts that are committed: long-format run statistics,
the cross-model and paired tables, and the comparison and per-run figures.

Usage:
  python scripts/make_report.py
  python scripts/make_report.py --questions 500 --reasoning xhigh
  python scripts/make_report.py --no-run-figures
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_results as ar  # noqa: E402
import compare_models as cm  # noqa: E402
import run_data as rd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
# Provider probe files (spike_*) are calibration runs, not experiment results.
SKIP_PREFIXES = ("spike_",)
RUN_COLUMNS = [
    "file",
    "model",
    "provider",
    "questions",
    "reasoning",
    "complete",
    "condition",
    "correct",
    "answered",
    "blank",
    "blank_len",
    "blank_stop",
    "format_errors",
    "missing",
    "acc_all",
    "acc_answered",
    "mean_prompt_tokens",
    "mean_reasoning_tokens",
    "median_reasoning_tokens",
    "cost",
]
MODEL_COMPARISON_COLUMNS = [
    "model",
    "label",
    "provider",
    "questions",
    "condition",
    "n",
    "correct",
    "answered",
    "blank_len",
    "blank_stop",
    "format_errors",
    "acc_all",
    "acc_answered",
    "cost",
    "mean_reasoning_tokens",
    "mean_latency_s",
]
PAIRED_COLUMNS = [
    "file",
    "model",
    "reasoning",
    "questions",
    "reference",
    "condition",
    "paired",
    "fixed",
    "broke",
    "discordant",
    "net_pp",
    "ci_low_pp",
    "ci_high_pp",
    "p",
]


@dataclass(frozen=True)
class RunInfo:
    path: Path
    file: str
    model: str
    provider: str | None
    questions: int
    reasoning: str
    complete: bool
    stats: dict[str, dict]


def reasoning_label(config):
    """Short reasoning label from a run config, or "?" when unknown."""
    if not isinstance(config, dict):
        return "?"
    mode = config.get("reasoning_mode")
    value = config.get("reasoning_value")
    if mode == "effort" and value is not None:
        return "none" if value == "none" else str(value)
    if mode == "max_tokens" and value is not None:
        return f"maxtokens{value}"
    return "?"


def _relative_to_root(path):
    return Path(os.path.relpath(Path(path).resolve(), REPO_ROOT))


def _run_sort_key(run):
    return (run.model, run.questions, run.reasoning, run.file)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def _write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    return len(rows)


def discover_runs(results_dir):
    """Summarize every readable run JSON in the results directory."""
    runs = []
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return runs
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith(SKIP_PREFIXES):
            continue
        try:
            data = rd.load_file(str(path))
        except (OSError, ValueError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        questions = data["questions"]
        config = data["meta"].get("config")
        provider = config.get("provider") if isinstance(config, dict) else None
        complete = all(len(entries) == len(questions) for entries in data["results"].values())
        stats = {
            condition: rd.cond_stats(data["results"], questions, condition)
            for condition in data["results"]
        }
        runs.append(
            RunInfo(
                path=path,
                file=str(_relative_to_root(path)),
                model=data["meta"]["model"],
                provider=provider,
                questions=len(questions),
                reasoning=reasoning_label(config),
                complete=complete,
                stats=stats,
            )
        )
    return sorted(runs, key=_run_sort_key)


def write_runs_csv(runs, path):
    """Write one row per run and condition; return the row count."""
    rows = []
    for run in sorted(runs, key=_run_sort_key):
        for condition in sorted(run.stats):
            stats = run.stats[condition]
            rows.append(
                {
                    "file": run.file,
                    "model": run.model,
                    "provider": run.provider,
                    "questions": run.questions,
                    "reasoning": run.reasoning,
                    "complete": run.complete,
                    "condition": condition,
                    "correct": stats["correct"],
                    "answered": stats["answered"],
                    "blank": stats["blank"],
                    "blank_len": stats["blank_len"],
                    "blank_stop": stats["blank_stop"],
                    "format_errors": stats["format_errors"],
                    "missing": stats["missing"],
                    "acc_all": stats["acc_all"],
                    "acc_answered": stats["acc_answered"],
                    "mean_prompt_tokens": stats["mean_prompt_tokens"],
                    "mean_reasoning_tokens": stats["mean_reasoning_tokens"],
                    "median_reasoning_tokens": stats["median_reasoning_tokens"],
                    "cost": stats["cost"],
                }
            )
    return _write_csv(path, RUN_COLUMNS, rows)


def write_model_comparison_csv(runs, path):
    """Write one row per model and condition from compare_models runs."""
    model_order = {model: i for i, model in enumerate(cm.MODEL_ORDER)}
    condition_order = {condition: i for i, condition in enumerate(cm.CONDITIONS)}
    rows = []
    for run in runs:
        for condition in cm.CONDITIONS:
            metrics = run.metrics[condition]
            rows.append(
                {
                    "model": run.model,
                    "label": run.label,
                    "provider": run.provider,
                    "questions": run.questions,
                    "condition": condition,
                    "n": metrics["n"],
                    "correct": metrics["correct"],
                    "answered": metrics["answered"],
                    "blank_len": metrics["blank_len"],
                    "blank_stop": metrics["blank_stop"],
                    "format_errors": metrics["format_errors"],
                    "acc_all": metrics["acc_all"],
                    "acc_answered": metrics["acc_answered"],
                    "cost": metrics["cost"],
                    "mean_reasoning_tokens": metrics["reasoning"],
                    "mean_latency_s": metrics["latency"],
                }
            )
    rows.sort(
        key=lambda row: (
            model_order.get(row["model"], len(model_order)),
            condition_order.get(row["condition"], len(condition_order)),
        )
    )
    return _write_csv(path, MODEL_COMPARISON_COLUMNS, rows)


def write_paired_csv(runs, path, run_info):
    """Paired comparisons against baseline and the neutral control."""
    by_path = {Path(info.path).resolve(): info for info in run_info}
    rows = []
    for run in runs:
        info = by_path.get(Path(run.path).resolve())
        if info is None:
            continue
        try:
            data = rd.load_file(str(run.path))
        except (OSError, ValueError) as exc:
            print(f"skipping {run.path}: {exc}", file=sys.stderr)
            continue
        results = data["results"]
        references = [reference for reference in ("baseline", "neutral") if reference in results]
        for reference in references:
            for condition in data["order"]:
                if condition == reference:
                    continue
                stats = rd.paired_stats(results, data["questions"], condition, reference=reference)
                if stats is None:
                    continue
                rows.append(
                    {
                        "file": info.file,
                        "model": run.model,
                        "reasoning": info.reasoning,
                        "questions": info.questions,
                        "reference": reference,
                        "condition": condition,
                        "paired": stats["paired"],
                        "fixed": stats["fixed"],
                        "broke": stats["broke"],
                        "discordant": stats["discordant"],
                        "net_pp": stats["net_pp"],
                        "ci_low_pp": stats["ci_low_pp"],
                        "ci_high_pp": stats["ci_high_pp"],
                        "p": stats["p"],
                    }
                )
    rows.sort(key=lambda row: (row["model"], row["reference"], row["condition"]))
    return _write_csv(path, PAIRED_COLUMNS, rows)


def has_reasoning_tokens(data):
    return any(
        value > 0
        for condition in data["order"]
        for value in rd.token_series(data, condition, "reasoning_tokens")
    )


def write_comparison_figures(runs, figures_dir, questions, reasoning, dpi):
    """Write the combined comparison figure and its five panels."""
    if not runs:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"model-comparison-{questions}q.png"
    cm.plot_comparison(runs, out_path, reasoning=reasoning, dpi=dpi)
    panels = cm.plot_individual_panels(runs, out_path, reasoning=reasoning, dpi=dpi)
    return [out_path, *panels]


def write_run_figures(runs, figures_dir):
    """Write one token plot per complete run with reasoning data.

    Figures for runs that no longer exist are removed, so the directory always
    mirrors the current results tree.
    """
    written = []
    selected = [run for run in sorted(runs, key=_run_sort_key) if run.complete]
    for run in selected:
        try:
            data = rd.load_file(str(run.path))
        except (OSError, ValueError) as exc:
            print(f"skipping {run.path}: {exc}", file=sys.stderr)
            continue
        if not has_reasoning_tokens(data):
            continue
        runs_dir = figures_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        out_path = runs_dir / f"{run.path.stem}.png"
        ar.plot_tokens(data, str(out_path))
        written.append(out_path)
    if written:
        stale = set((figures_dir / "runs").glob("*.png")) - set(written)
        for path in sorted(stale):
            path.unlink()
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate committed tables and figures from the results tree",
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--tables-dir", default=str(REPO_ROOT / "tables"))
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--questions", type=int, default=5000)
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument(
        "--no-run-figures",
        action="store_true",
        help="skip the per-run token plots",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="include runs with missing entries in the comparison",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="write empty tables instead of failing when no runs are found",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    runs = discover_runs(results_dir)
    if not runs and not args.allow_empty:
        print(
            f"error: no readable runs under {results_dir}; nothing written "
            "(pass --allow-empty to write empty tables)",
            file=sys.stderr,
        )
        return 1

    runs_path = tables_dir / "runs.csv"
    runs_rows = write_runs_csv(runs, runs_path)
    print(f"Wrote {runs_path} ({runs_rows} rows)")

    model_runs = []
    if results_dir.is_dir():
        model_runs, skipped = cm.discover_runs(
            results_dir,
            questions=args.questions,
            reasoning=args.reasoning,
            allow_partial=args.allow_partial,
        )
        for path, reason in skipped:
            print(f"skipped {path}: {reason}", file=sys.stderr)

    comparison_path = tables_dir / f"model-comparison-{args.questions}q.csv"
    comparison_rows = write_model_comparison_csv(model_runs, comparison_path)
    print(f"Wrote {comparison_path} ({comparison_rows} rows)")

    paired_path = tables_dir / f"paired-{args.questions}q.csv"
    paired_rows = write_paired_csv(model_runs, paired_path, runs)
    print(f"Wrote {paired_path} ({paired_rows} rows)")

    comparison_figures = write_comparison_figures(
        model_runs,
        figures_dir,
        args.questions,
        args.reasoning,
        args.dpi,
    )
    if comparison_figures:
        print(f"Wrote {len(comparison_figures)} comparison figures to {figures_dir}")
    else:
        print(f"No complete {args.questions}-question runs; skipping comparison figures")

    if args.no_run_figures:
        print("Skipped run figures (--no-run-figures)")
    else:
        run_figures = write_run_figures(runs, figures_dir)
        print(f"Wrote {len(run_figures)} run figures to {figures_dir / 'runs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
