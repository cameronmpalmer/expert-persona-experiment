#!/usr/bin/env python3
"""Cross-model comparison plots for the persona experiment.

Selects the latest complete phase 2 run for each model at a given question count
and reasoning level, then writes a four-panel PNG:

  1. accuracy by condition
  2. total cost by condition
  3. mean reasoning tokens by condition, log scale
  4. accuracy against cost, one point per model and condition

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --list
  python scripts/compare_models.py --models anthropic/claude-haiku-5.5,z-ai/glm-5.3-flash
  python scripts/compare_models.py --conditions baseline,expert,expert_long
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_data as rd  # noqa: E402

PHASE2_CONDITIONS = ("neutral", "baseline", "expert_short", "expert", "expert_long")

MODEL_ORDER = (
    "anthropic/claude-haiku-5.5",
    "deepseek/deepseek-v4-flash",
    "inclusionai/ling-3.0-flash",
    "z-ai/glm-5.3-flash",
)
MODEL_LABELS = {
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "deepseek/deepseek-v4-flash": "Deepseek V4 Flash",
    "inclusionai/ling-3.0-flash": "Ling 3.0 Flash",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
}
CONDITION_MARKERS = {
    "baseline": "o",
    "neutral": "s",
    "expert_short": "^",
    "expert": "D",
    "expert_long": "v",
}

_MODEL_RE = re.compile(r'"model"\s*:\s*"([^"]+)"')


class RunSummary(NamedTuple):
    path: Path
    model: str
    label: str
    provider: str | None
    questions: int
    metrics: dict[str, dict[str, float | int | None]]


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.rsplit("/", 1)[-1])


def read_model_slug(path: Path) -> str:
    """Read the model slug without parsing a potentially huge results file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(8192)
    match = _MODEL_RE.search(head)
    if match:
        return match.group(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("model", "?")


def candidate_paths(results_dir: Path, questions: int, reasoning: str) -> list[Path]:
    pattern = f"*{questions}q*{reasoning}*.json"
    return sorted(
        results_dir.rglob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def sort_runs(runs: list[RunSummary]) -> list[RunSummary]:
    order = {model: i for i, model in enumerate(MODEL_ORDER)}
    return sorted(runs, key=lambda run: (order.get(run.model, len(order)), run.label))


def discover_runs(
    results_dir: Path,
    *,
    questions: int = 5000,
    reasoning: str = "xhigh",
    conditions: tuple[str, ...] = PHASE2_CONDITIONS,
    models: tuple[str, ...] | None = None,
    allow_partial: bool = False,
) -> tuple[list[RunSummary], list[tuple[Path, str]]]:
    """Pick the latest complete run per model and summarize its conditions.

    Candidates are considered newest first, so older runs for a model are skipped
    after reading only the model slug. A run qualifies when it has the requested
    question count, every requested condition, and (unless ``allow_partial``) an
    entry for every question in every condition.
    """
    selected: dict[str, RunSummary] = {}
    skipped: list[tuple[Path, str]] = []
    for path in candidate_paths(results_dir, questions, reasoning):
        try:
            model = read_model_slug(path)
        except (OSError, ValueError) as exc:
            skipped.append((path, f"unreadable ({exc})"))
            continue
        if models and not any(token.lower() in model.lower() for token in models):
            continue
        if model in selected:
            continue
        try:
            data = rd.load_file(str(path))
        except (OSError, ValueError) as exc:
            skipped.append((path, f"unreadable ({exc})"))
            continue
        run_questions = data["questions"]
        if len(run_questions) != questions:
            skipped.append((path, f"{len(run_questions)} questions, wanted {questions}"))
            continue
        missing = [cond for cond in conditions if cond not in data["results"]]
        if missing:
            skipped.append((path, f"missing conditions: {', '.join(missing)}"))
            continue
        short = [
            cond for cond in conditions if len(data["results"].get(cond, {})) < len(run_questions)
        ]
        if short and not allow_partial:
            skipped.append((path, f"incomplete: {', '.join(short)}"))
            continue
        metrics = {}
        for cond in conditions:
            stats = rd.cond_stats(data["results"], run_questions, cond)
            metrics[cond] = {
                "n": stats["n"],
                "correct": stats["correct"],
                "answered": stats["answered"],
                "blank_len": stats["blank_len"],
                "blank_stop": stats["blank_stop"],
                "format_errors": stats["format_errors"],
                "acc_all": stats["acc_all"],
                "acc_answered": stats["acc_answered"],
                "cost": stats["cost"],
                "reasoning": stats["mean_reasoning_tokens"],
                "prompt": stats["mean_prompt_tokens"],
                "latency": rd.mean_latency(data["results"], cond),
            }
        config = data["meta"].get("config")
        provider = config.get("provider") if isinstance(config, dict) else None
        selected[model] = RunSummary(
            path=path,
            model=model,
            label=model_label(model),
            provider=provider,
            questions=len(run_questions),
            metrics=metrics,
        )
    return sort_runs(list(selected.values())), skipped


def condition_label(condition: str) -> str:
    return condition.replace("_", "\n")


def _optional(value: float | int | None, scale: float = 1.0) -> float:
    return math.nan if value is None else float(value) * scale


def usd_tick(value: float, _position: int) -> str:
    if value >= 10:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.2f}"
    return f"${value:,.3f}"


def _model_handles(runs: list[RunSummary], colours):
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=colours(i % 10), marker="o", linewidth=1.8, label=run.label)
        for i, run in enumerate(runs)
    ]


def _pareto_handle():
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        color="0.25",
        linestyle="--",
        linewidth=1.4,
        label="Pareto frontier",
    )


def _condition_handles(conditions: tuple[str, ...]):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color="0.3",
            marker=CONDITION_MARKERS.get(condition, "o"),
            linestyle="None",
            markersize=7,
            label=condition,
        )
        for condition in conditions
    ]


def _plot_accuracy(axis, runs, conditions, colours) -> None:
    x = list(range(len(conditions)))
    for index, run in enumerate(runs):
        accuracy = [_optional(run.metrics[condition]["acc_all"], 100) for condition in conditions]
        axis.plot(
            x,
            accuracy,
            color=colours(index % 10),
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=run.label,
        )
    axis.set_xticks(x, [condition_label(condition) for condition in conditions])
    axis.set_ylabel("accuracy, all calls (%)")
    axis.set_title("Accuracy by condition")
    values = [_optional(run.metrics[cond]["acc_all"], 100) for run in runs for cond in conditions]
    finite = [value for value in values if not math.isnan(value)]
    if finite:
        axis.set_ylim(min(finite) - 1.5, max(finite) + 1.5)
    axis.grid(True, alpha=0.25, linewidth=0.5)


def _plot_cost(axis, runs, conditions, colours) -> None:
    from matplotlib.ticker import FuncFormatter

    x = list(range(len(conditions)))
    for index, run in enumerate(runs):
        cost = [_optional(run.metrics[condition]["cost"]) for condition in conditions]
        axis.plot(
            x,
            cost,
            color=colours(index % 10),
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=run.label,
        )
    axis.set_xticks(x, [condition_label(condition) for condition in conditions])
    axis.set_ylabel("total cost (USD)")
    axis.set_title("Total cost by condition")
    axis.yaxis.set_major_formatter(FuncFormatter(usd_tick))
    axis.margins(y=0.12)
    axis.grid(True, alpha=0.25, linewidth=0.5)


def _plot_reasoning(axis, runs, conditions, colours) -> None:
    from matplotlib.ticker import FuncFormatter

    x = list(range(len(conditions)))
    for index, run in enumerate(runs):
        token_count = [_optional(run.metrics[condition]["reasoning"]) for condition in conditions]
        axis.plot(
            x,
            token_count,
            color=colours(index % 10),
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=run.label,
        )
    axis.set_xticks(x, [condition_label(condition) for condition in conditions])
    axis.set_yscale("log")
    axis.set_ylabel("mean reasoning tokens (log)")
    axis.set_title("Reasoning tokens by condition")
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_minor_formatter(formatter)
    axis.grid(True, alpha=0.25, linewidth=0.5)


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated points for lower cost and higher accuracy."""
    ordered = sorted(points, key=lambda point: (point[0], -point[1]))
    frontier = []
    best_accuracy = float("-inf")
    for cost, accuracy in ordered:
        if accuracy > best_accuracy:
            frontier.append((cost, accuracy))
            best_accuracy = accuracy
    return frontier


def _plot_scatter(axis, runs, conditions, colours) -> None:
    from matplotlib.ticker import FuncFormatter

    points = []
    for index, run in enumerate(runs):
        colour = colours(index % 10)
        for condition in conditions:
            metric = run.metrics[condition]
            total_cost = metric["cost"]
            accuracy_value = metric["acc_all"]
            if total_cost is None or accuracy_value is None or total_cost <= 0:
                continue
            accuracy = accuracy_value * 100
            points.append((total_cost, accuracy))
            axis.scatter(
                total_cost,
                accuracy,
                color=colour,
                marker=CONDITION_MARKERS.get(condition, "o"),
                s=70,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    frontier = pareto_frontier(points)
    if len(frontier) >= 2:
        axis.plot(
            [point[0] for point in frontier],
            [point[1] for point in frontier],
            color="0.25",
            linestyle="--",
            linewidth=1.4,
            zorder=2,
        )
    axis.set_xlabel("total cost (USD)")
    axis.set_ylabel("accuracy, all calls (%)")
    axis.set_title("Accuracy vs cost (up and to the left is better)")
    axis.xaxis.set_major_formatter(FuncFormatter(usd_tick))
    axis.margins(x=0.08, y=0.08)
    axis.grid(True, alpha=0.25, linewidth=0.5)


def _plot_latency(axis, runs, conditions, colours) -> None:
    from matplotlib.ticker import FuncFormatter

    x = list(range(len(conditions)))
    for index, run in enumerate(runs):
        latency = [_optional(run.metrics[condition]["latency"]) for condition in conditions]
        axis.plot(
            x,
            latency,
            color=colours(index % 10),
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=run.label,
        )
    axis.set_xticks(x, [condition_label(condition) for condition in conditions])
    axis.set_yscale("log")
    axis.set_ylabel("average call latency (s, log)")
    axis.set_title("Average call latency by condition")
    formatter = FuncFormatter(lambda value, _: f"{value:,.1f}")
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_minor_formatter(formatter)
    axis.grid(True, alpha=0.25, linewidth=0.5)


def _footer(fig, fontsize: float = 8.5) -> None:
    fig.text(
        0.5,
        0.012,
        "accuracy = correct / all calls; cost = total usage.cost per condition; "
        "reasoning = mean reasoning tokens",
        ha="center",
        fontsize=fontsize,
        color="0.35",
    )


def plot_comparison(
    runs: list[RunSummary],
    out_path: Path,
    conditions: tuple[str, ...] = PHASE2_CONDITIONS,
    *,
    reasoning: str = "xhigh",
    dpi: int = 160,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("plot_comparison needs at least one run")

    colours = plt.get_cmap("tab10")
    fig, grid = plt.subplots(2, 2, figsize=(14, 10))
    axes = grid.ravel()
    _plot_accuracy(axes[0], runs, conditions, colours)
    _plot_cost(axes[1], runs, conditions, colours)
    _plot_reasoning(axes[2], runs, conditions, colours)
    _plot_scatter(axes[3], runs, conditions, colours)

    fig.legend(
        handles=_model_handles(runs, colours),
        loc="upper center",
        ncol=min(len(runs), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
        fontsize=9,
    )
    axes[3].legend(
        handles=_condition_handles(conditions) + [_pareto_handle()],
        title="condition / frontier",
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )
    fig.suptitle(
        f"Cross-model comparison: MMLU-Pro {runs[0].questions:,} questions, reasoning {reasoning}",
        fontsize=14,
        y=0.995,
    )
    _footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_individual_panels(
    runs: list[RunSummary],
    out_path: Path,
    conditions: tuple[str, ...] = PHASE2_CONDITIONS,
    *,
    reasoning: str = "xhigh",
    dpi: int = 160,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("plot_individual_panels needs at least one run")

    colours = plt.get_cmap("tab10")
    model_handles = _model_handles(runs, colours)
    condition_handles = _condition_handles(conditions)
    panels = (
        ("accuracy", "Accuracy by condition", _plot_accuracy),
        ("cost", "Total cost by condition", _plot_cost),
        ("reasoning", "Reasoning tokens by condition", _plot_reasoning),
        ("latency", "Average call latency by condition", _plot_latency),
        ("accuracy-vs-cost", "Accuracy vs cost (up and to the left is better)", _plot_scatter),
    )
    paths = []
    for slug, title, draw in panels:
        fig, axis = plt.subplots(figsize=(9, 6))
        draw(axis, runs, conditions, colours)
        axis.set_title(
            f"{title}\nMMLU-Pro {runs[0].questions:,} questions, reasoning {reasoning}",
            fontsize=11,
        )
        if slug == "accuracy-vs-cost":
            first = axis.legend(handles=model_handles, loc="upper left", fontsize=8, framealpha=0.9)
            axis.add_artist(first)
            axis.legend(
                handles=condition_handles + [_pareto_handle()],
                title="condition / frontier",
                loc="lower right",
                fontsize=8,
                title_fontsize=8,
                framealpha=0.9,
            )
        else:
            axis.legend(handles=model_handles, loc="best", fontsize=8, framealpha=0.9)
        _footer(fig, fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 0.98))
        panel_path = out_path.with_name(f"{out_path.stem}-{slug}{out_path.suffix}")
        fig.savefig(panel_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(panel_path)
    return paths


def print_summary(runs: list[RunSummary], conditions: tuple[str, ...]) -> None:
    print(f"{'model':24s} {'condition':12s} {'acc(all)':>9s} {'total $':>10s} {'reason':>9s}")
    for run in runs:
        for condition in conditions:
            metric = run.metrics[condition]
            accuracy = (
                f"{metric['acc_all'] * 100:8.1f}%" if metric["acc_all"] is not None else "      n/a"
            )
            cost = f"${metric['cost']:9.2f}" if metric["cost"] is not None else "      n/a"
            token_count = (
                f"{metric['reasoning']:9.0f}" if metric["reasoning"] is not None else "      n/a"
            )
            print(f"{run.label:24s} {condition:12s} {accuracy:>9s} {cost:>12s} {token_count:>9s}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy, cost, and reasoning tokens across model runs",
    )
    parser.add_argument("--results-dir", default="results/phase2")
    parser.add_argument("--questions", type=int, default=5000)
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument(
        "--conditions",
        default=",".join(PHASE2_CONDITIONS),
        help="comma-separated condition names",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model slug substrings to include",
    )
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="print the selected runs and metrics without plotting",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="include runs with missing entries; missing calls count as wrong",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="write only the combined figure, not the four individual panels",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    if not conditions:
        raise SystemExit("--conditions is empty")
    models = (
        tuple(part.strip() for part in args.models.split(",") if part.strip())
        if args.models
        else None
    )
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"results directory not found: {results_dir}")

    runs, skipped = discover_runs(
        results_dir,
        questions=args.questions,
        reasoning=args.reasoning,
        conditions=conditions,
        models=models,
        allow_partial=args.allow_partial,
    )
    if not runs:
        for path, reason in skipped:
            print(f"skipped {path}: {reason}", file=sys.stderr)
        raise SystemExit(f"no complete {args.questions}-question runs found in {results_dir}")

    if args.list_only:
        print_summary(runs, conditions)
        for path, reason in skipped:
            print(f"skipped {path}: {reason}", file=sys.stderr)
        return 0

    out_path = (
        Path(args.out)
        if args.out
        else results_dir.parent / f"model-comparison-{args.questions}q.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(
        runs,
        out_path,
        conditions,
        reasoning=args.reasoning,
        dpi=args.dpi,
    )
    print(f"Selected {len(runs)} models: {', '.join(run.label for run in runs)}")
    print(f"Wrote {out_path}")
    if not args.no_split:
        panel_paths = plot_individual_panels(
            runs,
            out_path,
            conditions,
            reasoning=args.reasoning,
            dpi=args.dpi,
        )
        print(f"Wrote {len(panel_paths)} individual panels:")
        for panel_path in panel_paths:
            print(f"  {panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
