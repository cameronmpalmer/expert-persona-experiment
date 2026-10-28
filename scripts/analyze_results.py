#!/usr/bin/env python3
"""Clean CLI analysis of persona experiment results.

Handles both result file shapes:
  - local Ollama runs (experiment.py):           entries have "thinking"
  - OpenRouter runs (experiment_openrouter_think.py): entries have "reasoning"

An entry that names no option is split in two. A blank completion field (empty
`raw`) is a non-answer, and blanks are further split by finish reason into
`Blank(len)` (hit the output cap, `finish_reason: length`) and `Blank(stop)` (stopped
with no content). A non-blank completion that still does not contain an option
letter is a format parsing error: the model answered in a shape the parser could
not read, for example "2.5 gallons" in place of a letter. None of these are
recovered from the reasoning trace, since a real user reads the completion.
Partial results files (run in progress) are handled gracefully - missing entries
are reported per condition.

Usage:
  python3 analyze_results.py FILE [FILE ...]           default summary per file
  python3 analyze_results.py FILE --questions          per-question matrix
  python3 analyze_results.py FILE --categories         accuracy by MMLU category
  python3 analyze_results.py FILE --reasoning          thinking/reasoning stats
  python3 analyze_results.py FILE --reference neutral  comparisons vs neutral
  python3 analyze_results.py FILE --confidence 0.99    99% confidence intervals
  python3 analyze_results.py FILE1 FILE2 --compare     side-by-side per condition
  python3 analyze_results.py FILE --json               machine-readable summary
  python3 analyze_results.py FILE --format-errors OUT  dump format errors to OUT (TSV)
  python3 analyze_results.py FILE --credit-format-errors  score matched format errors as correct
  python3 analyze_results.py FILE --plot OUT.png       write a token-distribution plot

All views can be combined; --json suppresses the text views. --format-errors
writes one row per format parsing error (condition, question id, category, gold
letter, the model's raw reply, finish reason, provider, reasoning tokens, and
the question text) and can be combined with any view. --credit-format-errors
counts a format error as correct when its reply matches the gold option, so
accuracy reflects answers given in the wrong shape. A paired significance table
(McNemar exact on discordant pairs) prints by default; pass --no-significance to
hide it. The comparison tables compare against baseline by default; pass
--reference neutral to use the length-matched non-expert control instead, which
isolates the expert wording effect in phase 2. Intervals are 95% by default;
pass --confidence 0.99 (or 99) for 99% intervals.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_data import (  # noqa: E402
    DEFAULT_ORDER,
    FORMAT_ERROR_COLUMNS,
    REASON_KEYS,
    change_matrix,
    classify_entry,
    cond_stats,
    detect_api,
    entry_is_correct,
    format_error_rows,
    load_file,
    match_option_index,
    mcnemar_exact,
    mean_latency,
    median_reasoning_diff,
    normalize_option,
    paired_stats,
    question_options,
    reason_ratio_stats,
    reason_text,
    reasoning_stats,
    thinking_label,
    token_series,
    write_format_errors,
    z_for_confidence,
)

__all__ = [
    "DEFAULT_ORDER",
    "FORMAT_ERROR_COLUMNS",
    "REASON_KEYS",
    "change_matrix",
    "classify_entry",
    "cond_stats",
    "detect_api",
    "entry_is_correct",
    "format_error_rows",
    "load_file",
    "match_option_index",
    "mcnemar_exact",
    "mean_latency",
    "median_reasoning_diff",
    "normalize_option",
    "paired_stats",
    "question_options",
    "reason_ratio_stats",
    "reason_text",
    "reasoning_stats",
    "thinking_label",
    "token_series",
    "write_format_errors",
    "z_for_confidence",
]

# ---------------------------------------------------------------- plots ---


def plot_tokens(d, out_path):
    """Write a PNG comparing reasoning-token cost and accuracy across conditions.

    A box-and-whiskers panel (median, IQR, whiskers, mean) sits on top, followed by
    one density panel per condition on a shared log axis, and an accuracy-versus-cost
    scatter at the bottom (accuracy over all calls, the deployment-facing rate).
    Reasoning tokens are the quantity that matters here; the
    visible answer is a single letter, so completion tokens add nothing, and the
    provider reports completion and reasoning inconsistently anyway.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = d["order"]
    key = "reasoning_tokens"
    cols = 3
    dens_rows = math.ceil(len(order) / cols)
    total_rows = 1 + dens_rows + 1
    cmap = matplotlib.colormaps["tab10"]
    fig = plt.figure(figsize=(13, 3.6 + 2.7 * total_rows))
    gs = fig.add_gridspec(total_rows, cols, height_ratios=[1.3] + [1.0] * dens_rows + [1.2])
    ax_box = fig.add_subplot(gs[0, :])
    suptitle = f"{os.path.basename(d['meta']['path']).split('__')[0]}  n={len(d['questions'])}"

    base_vals = token_series(d, "baseline", key)
    if not base_vals or max(base_vals) <= 0:
        ax_box.set_title("Reasoning tokens: no data")
        ax_box.axis("off")
        for i in range(len(order)):
            fig.add_subplot(gs[1 + i // cols, i % cols]).axis("off")
    else:
        # Box and whiskers: readable at a glance even when the tails are long.
        # Zero values cannot be drawn on a log axis, so they are dropped here.
        data = []
        for cond in order:
            vals = [v for v in token_series(d, cond, key) if v > 0]
            data.append(vals if vals else [np.nan])
        ax_box.boxplot(
            data,
            orientation="horizontal",
            positions=range(len(order)),
            widths=0.6,
            whis=1.5,
            showmeans=True,
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.35},
            meanprops={
                "marker": "D",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            medianprops={"linewidth": 1.6},
        )
        ax_box.set_xscale("log")
        ax_box.set_yticks(range(len(order)), labels=order)
        ax_box.invert_yaxis()  # baseline (position 0) at the top
        ax_box.set_title("Reasoning tokens: box and whiskers")
        ax_box.set_xlabel("tokens (log scale)")
        ax_box.grid(True, axis="x", alpha=0.25, linewidth=0.5)

        positive = [v for c in order for v in token_series(d, c, key) if v > 0]
        hi = max(positive) if positive else 1
        bins = np.logspace(0, np.log10(max(hi, 10)), 40)
        hist = []
        for cond in order:
            vals = [v for v in token_series(d, cond, key) if v > 0]
            counts, _ = np.histogram(vals, bins=bins, density=True)
            hist.append(counts)
        ymax = max((h.max() for h in hist), default=1.0)

        for i, cond in enumerate(order):
            ax = fig.add_subplot(gs[1 + i // cols, i % cols])
            color = cmap(i % 10)
            ax.stairs(hist[i], bins, color=color, fill=True, alpha=0.25, linewidth=1.6)
            vals = token_series(d, cond, key)
            med, mean = np.median(vals), np.mean(vals)
            if med > 0:
                ax.axvline(med, color=color, linestyle="-", linewidth=1.3)
            if mean > 0:
                ax.axvline(mean, color=color, linestyle="--", linewidth=1.3)
            ax.set_xscale("log")
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(0, ymax * 1.15)
            ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
            ax.set_title(f"{cond}   median {med:.0f}, mean {mean:.0f}", fontsize=10)
            ax.tick_params(labelsize=8)
            if i % cols == 0:
                ax.set_ylabel("density", fontsize=9)
            if i // cols == dens_rows - 1:
                ax.set_xlabel("tokens (log scale)", fontsize=9)

    # Accuracy against spend. Up and to the left is the good corner: more accurate
    # and cheaper. Accuracy is over all calls, so every call that was paid for counts,
    # including blanks and truncations that yielded no answer.
    ax_cost = fig.add_subplot(gs[total_rows - 1, :])
    ax_cost.set_title("Accuracy vs cost (up and to the left is better)")
    ax_cost.set_xlabel("run cost in USD (per condition)")
    ax_cost.set_ylabel("accuracy, all calls (%)")
    for i, cond in enumerate(order):
        s = cond_stats(d["results"], d["questions"], cond)
        acc = (s["acc_all"] or 0.0) * 100
        ax_cost.scatter(s["cost"], acc, s=80, color=cmap(i % 10), edgecolor="black", zorder=3)
        ax_cost.annotate(
            cond,
            (s["cost"], acc),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )
    ax_cost.grid(True, alpha=0.25, linewidth=0.5)
    ax_cost.margins(x=0.15, y=0.2)

    fig.suptitle(
        f"{suptitle}\nbox = IQR, line = median, diamond = mean, dots = values beyond 1.5xIQR",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- output ---


def fmt_pct(v):
    return f"{v:.1%}" if v is not None else "  n/a"


def cond_col_width(order, minimum=11):
    """Width of the Condition column, widened for longer condition names.

    Names like `expert_short` are longer than the old fixed width, which pushed
    the following columns out of line.
    """
    return max([minimum, *(len(c) for c in order)])


def fmt_usd(v):
    return f"${v:.4f}" if v is not None else "   n/a"


def fmt_int(v):
    return f"{v:.0f}" if v is not None else "n/a"


def fmt_delta(cond_val, base_val):
    """Signed mean delta for a token row, or n/a when either side is missing."""
    if cond_val is None or base_val is None:
        return f"{'n/a':>9}"
    delta = cond_val - base_val
    if abs(delta) < 0.5:  # avoid a signed "-0" in the output
        delta = 0.0
    return f"{delta:>+9.0f}"


def render_tokens(d, stats=None, reference="baseline", confidence=0.95):
    """Token usage per condition, with deltas against a reference condition.

    Cost and speed both follow from tokens: cost is tokens times price, speed
    is tokens per unit time. The means here use every entry that recorded usage.
    The default reference is baseline; neutral is the length-matched control
    for phase 2 wording comparisons.
    """
    results, order, questions = d["results"], d["order"], d["questions"]
    if stats is None:
        stats = {cond: cond_stats(results, questions, cond) for cond in order}
    base = stats.get(reference)
    if base is None:
        print()
        print(f"Token usage: reference condition '{reference}' not found; skipping.")
        return
    if base["mean_reasoning_tokens"] is None:
        print()
        print(f"Token usage: reference condition '{reference}' has no reasoning tokens; skipping.")
        return
    ci_label = f"{confidence * 100:g}% CI"
    print()
    print(f"Token usage vs {reference} (tokens per call; means, plus median reasoning):")
    print(
        f"{ci_label} = mean reasoning-token ratio; "
        "sign p = exact sign test on per-question differences."
    )
    w = cond_col_width(order)
    hdr = (
        f"{'Condition':<{w}}{'Prompt':>8}{'Reason':>8}{'R med':>7}"
        f"{'dPrompt':>9}{'dReason':>9}{'dR med':>8}"
        f"{'Reason x':>10}{ci_label:>14}{'sign p':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cond in order:
        s = stats[cond]
        row = (
            f"{cond:<{w}}{fmt_int(s['mean_prompt_tokens']):>8}"
            f"{fmt_int(s['mean_reasoning_tokens']):>8}"
            f"{fmt_int(s['median_reasoning_tokens']):>7}"
        )
        if cond == reference:
            row += f"{'-':>9}{'-':>9}{'-':>8}{'1.00':>10}{'-':>14}{'-':>9}"
        else:
            rr = reason_ratio_stats(
                results,
                questions,
                cond,
                reference=reference,
                confidence=confidence,
            )
            ratio_str = f"{rr['ratio']:.2f}" if rr else "n/a"
            ci_str = f"[{rr['ci_low']:.2f},{rr['ci_high']:.2f}]" if rr else "n/a"
            p_str = (f"{rr['p']:.3f}" if rr["p"] >= 0.001 else "<0.001") if rr else "n/a"
            md = median_reasoning_diff(results, questions, cond, reference=reference)
            md_str = f"{md:>+8.0f}" if md is not None else f"{'n/a':>8}"
            row += (
                fmt_delta(s["mean_prompt_tokens"], base["mean_prompt_tokens"])
                + fmt_delta(s["mean_reasoning_tokens"], base["mean_reasoning_tokens"])
                + md_str
                + f"{ratio_str:>10}{ci_str:>14}{p_str:>9}"
            )
        print(row)


def render_significance(d, credit_format_errors=False, reference="baseline", confidence=0.95):
    if reference not in d["results"]:
        print()
        print(f"Paired test: reference condition '{reference}' not found; skipping.")
        return
    order = [c for c in d["order"] if c != reference]
    if not order:
        return
    print()
    note = " (format errors credited)" if credit_format_errors else ""
    ci_label = f"{confidence * 100:g}% CI"
    print(f"Paired test vs {reference} (McNemar exact on discordant pairs){note}:")
    w = cond_col_width(d["order"])
    hdr = f"{'Condition':<{w}}{'Fixed':>7}{'Broke':>7}{'Disc':>6}{'Net':>10}{ci_label:>20}{'p':>9}"
    print(hdr)
    print("-" * len(hdr))
    for cond in order:
        s = paired_stats(
            d["results"],
            d["questions"],
            cond,
            credit_format_errors,
            reference=reference,
            confidence=confidence,
        )
        if not s:
            continue
        ci = f"[{s['ci_low_pp']:+.2f}, {s['ci_high_pp']:+.2f}]"
        p = f"{s['p']:.3f}" if s["p"] >= 0.001 else "<0.001"
        print(
            f"{cond:<{w}}{s['fixed']:>7}{s['broke']:>7}{s['discordant']:>6}"
            f"{s['net_pp']:>+9.2f}pp{ci:>20}{p:>9}"
        )


def render_summary(
    d,
    credit_format_errors=False,
    significance=True,
    reference="baseline",
    confidence=0.95,
):
    meta, results, order, questions = d["meta"], d["results"], d["order"], d["questions"]
    think = thinking_label(meta)
    w = cond_col_width(order)
    print(f"File:   {meta['path']}")
    print(
        f"Model:  {meta['model']} | Think: {think} | API: {detect_api(results)} | n={len(questions)}"
    )
    if credit_format_errors:
        print("Scoring: format errors credited when the reply matches the gold option")
    print()
    hdr = f"{'Condition':<{w}}{'Correct':>8}{'Wrong':>7}{'Blank(len)':>11}{'Blank(stop)':>12}{'FmtErr':>8}"
    if credit_format_errors:
        hdr += f"{'Credit':>8}"
    hdr += f"{'Missing':>9}{'Acc(ans)':>11}{'Acc(all)':>10}{'Cost':>11}"
    print(hdr)
    print("-" * len(hdr))
    stats = {}
    for cond in order:
        s = cond_stats(results, questions, cond, credit_format_errors=credit_format_errors)
        stats[cond] = s
        row = (
            f"{cond:<{w}}{s['correct']:>8}{s['wrong']:>7}{s['blank_len']:>11}"
            f"{s['blank_stop']:>12}{s['format_errors']:>8}"
        )
        if credit_format_errors:
            row += f"{s['credited']:>8}"
        row += (
            f"{s['missing']:>9}{fmt_pct(s['acc_answered']):>11}{fmt_pct(s['acc_all']):>10}"
            f"{fmt_usd(s['cost']):>11}"
        )
        print(row)

    total_cost = sum(s["cost"] for s in stats.values())
    cost_entries = sum(s["cost_entries"] for s in stats.values())
    missing_cost = sum(s["entries"] - s["cost_entries"] for s in stats.values())
    cost_line = f"Total cost: {fmt_usd(total_cost)} ({cost_entries} entries with usage"
    cost_line += f", {missing_cost} without)" if missing_cost else ")"
    print(cost_line)

    render_tokens(d, stats, reference=reference, confidence=confidence)

    if significance:
        render_significance(
            d,
            credit_format_errors=credit_format_errors,
            reference=reference,
            confidence=confidence,
        )

    print()
    if reference not in results:
        print(f"Answer changes: reference condition '{reference}' not found; skipping.")
        return
    print(f"Answer changes vs {reference} (answered pairs only):")
    hdr2 = f"{'Condition':<{w}}{'Changed':>8}{'W->R':>7}{'R->W':>7}{'W->W':>7}{'Same':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    for cond in order:
        if cond == reference:
            continue
        m = change_matrix(results, cond, reference=reference)
        print(
            f"{cond:<{w}}{m['changed']:>8}{m['WtoR']:>7}{m['RtoW']:>7}{m['WtoW']:>7}{m['same']:>7}"
        )


def render_reasoning(d):
    results, order, questions = d["results"], d["order"], d["questions"]
    w = cond_col_width(order)
    print("Reasoning/thinking stats (per condition):")
    hdr = f"{'Condition':<{w}}{'With':>7}{'Of':>6}{'Cover':>9}{'Mean len':>10}{'Max len':>9}{'Mean(corr)':>12}{'Mean(wrong)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for cond in order:
        r = reasoning_stats(results, questions, cond)
        if not r:
            print(f"{cond:<{w}}{'no reasoning captured':<60}")
            continue
        print(
            f"{cond:<{w}}{r['with_reasoning']:>7}{r['total_entries']:>6}"
            f"{r['with_reasoning'] / r['total_entries']:>9.0%}"
            f"{r['mean_len']:>10.0f}{r['max_len']:>9.0f}"
            f"{fmt_pct_len(r['mean_len_correct']):>12}{fmt_pct_len(r['mean_len_wrong']):>12}"
        )
    print()


def fmt_pct_len(v):
    return f"{v:.0f}" if v is not None else "n/a"


def render_questions(d):
    results, order, questions = d["results"], d["order"], d["questions"]
    cw = cond_col_width(order, minimum=10)
    print(f"{'ID':<6}{'Category':<20}{'Correct':<8}", end="")
    for cond in order:
        print(f"{cond:<{cw}}", end="")
    print()
    print("-" * (6 + 20 + 8 + cw * len(order)))
    for q in questions:
        qid = q["id"]
        cat = q.get("category", "?")
        correct_letter = chr(65 + q["answer_index"])
        print(f"{qid:<6}{cat:<20}{correct_letter:<8}", end="")
        base_ans = results.get("baseline", {}).get(qid, {}).get("answer")
        for cond in order:
            e = results.get(cond, {}).get(qid)
            if not e:
                print(f"{'-':<{cw}}", end="")
                continue
            cls = classify_entry(e)
            if cls == "blank":
                label = "trunc" if e.get("finish_reason") == "length" else "empty"
                print(f"{label:<{cw}}", end="")
            elif cls == "format":
                print(f"{'fmt':<{cw}}", end="")
            else:
                ans = e["answer"]
                mark = "" if ans == base_ans else "*"
                print(f"{ans + mark:<{cw}}", end="")
        print()


def render_categories(d, credit_format_errors=False):
    results, order, questions = d["results"], d["order"], d["questions"]
    by_id = {q["id"]: q for q in questions}
    cats = {}
    for q in questions:
        cat = q.get("category", "?")
        cats.setdefault(cat, []).append(q["id"])
    print(f"{'Category':<20}{'n':>4}", end="")
    cw = cond_col_width(order, minimum=12)
    for cond in order:
        print(f"{cond:>{cw}}", end="")
    print()
    print("-" * (24 + cw * len(order)))
    for cat, qids in sorted(cats.items()):
        print(f"{cat:<20}{len(qids):>4}", end="")
        for cond in order:
            entries = results.get(cond, {})
            correct = sum(
                1
                for qid in qids
                if entries.get(qid)
                and entry_is_correct(entries[qid], by_id[qid], credit_format_errors)
            )
            print(f"{correct:>7}/{len(qids):<5}", end="")
        print()


def render_compare(files, credit_format_errors=False):
    w = cond_col_width([c for d in files for c in d["order"]])
    print(f"{'Condition':<{w}}", end="")
    for d in files:
        name = os.path.basename(d["meta"]["path"])
        print(f"{name[:22]:>24}", end="")
    print()
    print("-" * (w + 24 * len(files)))
    for cond in files[0]["order"]:
        print(f"{cond:<{w}}", end="")
        for d in files:
            if cond not in d["results"]:
                print(f"{'n/a':>24}", end="")
                continue
            s = cond_stats(
                d["results"], d["questions"], cond, credit_format_errors=credit_format_errors
            )
            label = fmt_pct(s["acc_all"])
            if s["blank"] or s["format_errors"]:
                label += f" (b{s['blank']} f{s['format_errors']})"
            print(f"{label:>24}", end="")
        print()
    print(f"{'Cost':<{w}}", end="")
    for d in files:
        total = sum(
            cond_stats(
                d["results"], d["questions"], cond, credit_format_errors=credit_format_errors
            )["cost"]
            for cond in d["order"]
        )
        print(f"{fmt_usd(total):>24}", end="")
    print()


def render_json(
    files,
    credit_format_errors=False,
    significance=True,
    reference="baseline",
    confidence=0.95,
):
    out = []
    for d in files:
        results, order, questions = d["results"], d["order"], d["questions"]
        entry = {
            "file": d["meta"]["path"],
            "model": d["meta"]["model"],
            "think": d["meta"]["think"],
            "api": detect_api(results),
            "n": len(questions),
            "reference": reference,
            "credit_format_errors": credit_format_errors,
            "conditions": {},
        }
        base_acc = None
        if reference in results:
            base_stats = cond_stats(
                results, questions, reference, credit_format_errors=credit_format_errors
            )
            base_acc = base_stats["acc_all"]
        for cond in order:
            s = cond_stats(results, questions, cond, credit_format_errors=credit_format_errors)
            s["delta_pp_vs_reference"] = None
            if base_acc is not None and s["acc_all"] is not None:
                s["delta_pp_vs_reference"] = round((s["acc_all"] - base_acc) * 100, 1)
            s["paired_vs_reference"] = (
                paired_stats(
                    results,
                    questions,
                    cond,
                    credit_format_errors,
                    reference=reference,
                    confidence=confidence,
                )
                if significance and cond != reference
                else None
            )
            entry["conditions"][cond] = s
        out.append(entry)
    print(json.dumps(out, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze persona experiment results.")
    parser.add_argument("files", metavar="FILE", nargs="*", help="result JSON file(s)")
    parser.add_argument("--questions", action="store_true", help="per-question matrix")
    parser.add_argument("--categories", action="store_true", help="accuracy by MMLU category")
    parser.add_argument("--reasoning", action="store_true", help="thinking/reasoning stats")
    parser.add_argument("--compare", action="store_true", help="side-by-side per condition")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    parser.add_argument("--format-errors", metavar="OUT", default=None)
    parser.add_argument("--credit-format-errors", action="store_true")
    parser.add_argument("--plot", metavar="OUT", default=None)
    parser.add_argument("--reference", metavar="COND", default="baseline")
    parser.add_argument("--confidence", metavar="LEVEL", default="0.95")
    parser.add_argument("--no-significance", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    try:
        confidence = float(args.confidence)
    except ValueError:
        print("error: --confidence must be a number, e.g. 0.95 or 99", file=sys.stderr)
        sys.exit(1)
    if 1 < confidence < 2:
        print(
            "error: --confidence must be between 0 and 1, or between 2 and 100",
            file=sys.stderr,
        )
        sys.exit(1)
    if confidence > 1:
        confidence /= 100
    if not 0 < confidence < 1:
        print(
            "error: --confidence must be between 0 and 1, or between 2 and 100",
            file=sys.stderr,
        )
        sys.exit(1)
    reference = args.reference
    credit = args.credit_format_errors
    significance = not args.no_significance
    paths = args.files
    if not paths:
        print(__doc__)
        sys.exit(1)

    files = []
    for p in paths:
        if not os.path.exists(p):
            print(f"error: no such file: {p}", file=sys.stderr)
            sys.exit(1)
        files.append(load_file(p))

    if args.format_errors is not None:
        n = write_format_errors(files, args.format_errors)
        print(f"Wrote {n} format errors to {args.format_errors}")

    if args.plot is not None:
        if len(files) != 1:
            print("error: --plot takes exactly one input file", file=sys.stderr)
            sys.exit(1)
        plot_tokens(files[0], args.plot)
        print(f"Wrote plot to {args.plot}")

    if args.json:
        render_json(
            files,
            credit_format_errors=credit,
            significance=significance,
            reference=reference,
            confidence=confidence,
        )
        return

    for i, d in enumerate(files):
        if i:
            print()
        if args.compare and len(files) > 1:
            continue  # compare rendered once below
        render_summary(
            d,
            credit_format_errors=credit,
            significance=significance,
            reference=reference,
            confidence=confidence,
        )
        if args.reasoning:
            print()
            render_reasoning(d)
        if args.categories:
            print()
            render_categories(d, credit_format_errors=credit)
        if args.questions:
            print()
            render_questions(d)

    if args.compare and len(files) > 1:
        print()
        render_compare(files, credit_format_errors=credit)


if __name__ == "__main__":
    main()
