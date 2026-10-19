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
hide it.
"""

import csv
import json
import math
import os
import re
import statistics
import sys

DEFAULT_ORDER = [
    "baseline",
    "neutral",
    "expert_short",
    "expert",
    "expert_long",
    "layperson",
    "toddler",
]
REASON_KEYS = ("reasoning", "thinking")
FORMAT_ERROR_COLUMNS = [
    "condition",
    "id",
    "category",
    "gold",
    "answer_text",
    "regex_letter",
    "would_be_correct",
    "finish_reason",
    "provider",
    "reasoning_tokens",
    "question",
]


# ---------------------------------------------------------------- loading ---


def load_file(path):
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    results = data.get("results", {})
    order = [c for c in DEFAULT_ORDER if c in results]
    for c in results:
        if c not in order:
            order.append(c)
    meta = {
        "path": path,
        "model": data.get("model", "?"),
        "think": data.get("think"),
        "api": data.get("api"),
        "config": data.get("config"),
    }
    return {"meta": meta, "questions": questions, "results": results, "order": order}


def detect_api(results):
    # Explicit api field wins (only the OpenRouter script writes it)
    for entries in results.values():
        for e in entries.values():
            if "reasoning" in e:
                return "openrouter"
            if "thinking" in e:
                return "ollama"
    # Only the local Ollama script omits the api field
    return "ollama"


def thinking_label(meta):
    """Thinking setting for the header, e.g. "on (xhigh)" or "off".

    Local Ollama files record an explicit `think` boolean. OpenRouter files do
    not, so derive it from the run config (thinking is off only when the reasoning
    effort is `none`). "?" means the setting could not be determined.
    """
    if meta.get("think") is not None:
        return "on" if meta["think"] else "off"
    config = meta.get("config")
    if not isinstance(config, dict):
        return "?"
    mode = config.get("reasoning_mode")
    value = config.get("reasoning_value")
    if mode is None:
        return "?"
    if mode == "effort" and value == "none":
        return "off"
    if mode == "max_tokens" and value is not None:
        return f"on (max_tokens {value})"
    return f"on ({value})" if value is not None else "on"


def reason_text(entry):
    for k in REASON_KEYS:
        v = entry.get(k)
        if v:
            return v
    return ""


def classify_entry(entry):
    """Classify one entry as answered, blank, or a format parsing error.

    answered - the parser found an option letter
    blank    - the completion field is empty, so there was nothing to parse
    format   - the completion is non-blank but holds no option letter
    """
    if entry.get("answer") != "?":
        return "answered"
    if (entry.get("raw") or "").strip():
        return "format"
    return "blank"


def question_options(question):
    """The option texts for a question, tolerating a JSON-encoded string."""
    options = question.get("options") or []
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except ValueError:
            options = [options]
    return options


def entry_is_correct(entry, question, credit_format_errors=False):
    """Correctness of one entry, optionally crediting matched format errors."""
    if entry.get("correct"):
        return True
    if not credit_format_errors or classify_entry(entry) != "format":
        return False
    return (
        match_option_index(entry.get("raw"), question_options(question)) == question["answer_index"]
    )


def cond_stats(results, questions, cond, credit_format_errors=False):
    entries = results.get(cond, {})
    n = len(questions)
    correct = sum(1 for e in entries.values() if e.get("correct"))
    blank_len = sum(
        1
        for e in entries.values()
        if classify_entry(e) == "blank" and e.get("finish_reason") == "length"
    )
    blank_total = sum(1 for e in entries.values() if classify_entry(e) == "blank")
    blank_stop = blank_total - blank_len
    blank = blank_total
    format_errors = sum(1 for e in entries.values() if classify_entry(e) == "format")
    credited = 0
    if credit_format_errors:
        credited = sum(
            1
            for q in questions
            if (e := entries.get(q["id"])) is not None
            and classify_entry(e) == "format"
            and entry_is_correct(e, q, credit_format_errors=True)
        )
        correct += credited
        format_errors -= credited
    answered = len(entries) - blank - format_errors
    wrong = answered - correct
    missing = n - len(entries)
    costs = [
        (e.get("usage") or {}).get("cost")
        for e in entries.values()
        if (e.get("usage") or {}).get("cost") is not None
    ]

    def token_values(key):
        vals = []
        for e in entries.values():
            v = (e.get("usage") or {}).get(key)
            if v is not None:
                vals.append(v)
        return vals

    pv = token_values("prompt_tokens")
    rv = token_values("reasoning_tokens")
    return {
        "n": n,
        "entries": len(entries),
        "correct": correct,
        "wrong": wrong,
        "answered": answered,
        "blank": blank,
        "blank_len": blank_len,
        "blank_stop": blank_stop,
        "format_errors": format_errors,
        "credited": credited,
        "missing": missing,
        "cost": sum(costs),
        "cost_entries": len(costs),
        "mean_prompt_tokens": (sum(pv) / len(pv)) if pv else None,
        "mean_reasoning_tokens": (sum(rv) / len(rv)) if rv else None,
        "median_prompt_tokens": statistics.median(pv) if pv else None,
        "median_reasoning_tokens": statistics.median(rv) if rv else None,
        "acc_answered": (correct / answered) if answered else None,
        "acc_all": (correct / n) if n else None,
    }


def mcnemar_exact(fixed, broke):
    """Two-sided exact McNemar (sign test) p-value on discordant pairs.

    Falls back to a continuity-corrected normal approximation above 1000
    discordant pairs, where the exact binomial becomes slow.
    """
    m = fixed + broke
    if m == 0:
        return 1.0
    if m > 1000:
        z = (abs(fixed - broke) - 1) / math.sqrt(m)
        return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2))))
    k = min(fixed, broke)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m)


def reason_ratio_stats(results, questions, cond):
    """Reasoning-token ratio of a condition against baseline, with CI and p.

    Ratio is mean(condition) / mean(baseline) on shared questions. The 95%
    interval is a delta-method interval on the log ratio (fast and stable for
    these heavy-tailed counts). The p-value is the exact sign test on the
    per-question differences, so it answers whether reasoning moved at all.
    Returns None when baseline reasoning is zero, since the ratio is undefined.
    """
    base = results.get("baseline", {})
    entries = results.get(cond, {})
    xs, ys = [], []
    for q in questions:
        qid = q["id"]
        if qid not in base or qid not in entries:
            continue
        x = (entries[qid].get("usage") or {}).get("reasoning_tokens")
        y = (base[qid].get("usage") or {}).get("reasoning_tokens")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    if my <= 0:
        return None
    higher = sum(1 for a, b in zip(xs, ys, strict=True) if a > b)
    lower = sum(1 for a, b in zip(xs, ys, strict=True) if a < b)
    vx = sum((a - mx) ** 2 for a in xs) / (n - 1)
    vy = sum((b - my) ** 2 for b in ys) / (n - 1)
    cxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / (n - 1)
    var_log = max(0.0, vx / (n * mx * mx) + vy / (n * my * my) - 2 * cxy / (n * mx * my))
    se_log = math.sqrt(var_log)
    ratio = mx / my
    return {
        "ratio": ratio,
        "ci_low": ratio * math.exp(-1.96 * se_log),
        "ci_high": ratio * math.exp(1.96 * se_log),
        "higher": higher,
        "lower": lower,
        "p": mcnemar_exact(higher, lower),
    }


def median_reasoning_diff(results, questions, cond):
    """Median of the per-question reasoning-token difference vs baseline.

    Robust companion to the mean difference, which a few long traces can carry.
    Returns None when there are no paired questions with reasoning tokens.
    """
    base = results.get("baseline", {})
    entries = results.get(cond, {})
    diffs = []
    for q in questions:
        qid = q["id"]
        if qid not in base or qid not in entries:
            continue
        x = (entries[qid].get("usage") or {}).get("reasoning_tokens")
        y = (base[qid].get("usage") or {}).get("reasoning_tokens")
        if x is None or y is None:
            continue
        diffs.append(x - y)
    return statistics.median(diffs) if diffs else None


def paired_stats(results, questions, cond, credit_format_errors=False):
    """Paired comparison of one condition against baseline on shared questions.

    Returns fixed (baseline wrong, condition right), broke (the reverse), the
    discordant total, the net difference in percentage points, a 95% interval
    from the McNemar normal approximation, and the exact McNemar p-value.
    """
    base = results.get("baseline", {})
    entries = results.get(cond, {})
    fixed = broke = paired = 0
    for q in questions:
        qid = q["id"]
        if qid not in base or qid not in entries:
            continue
        paired += 1
        b = entry_is_correct(base[qid], q, credit_format_errors)
        c = entry_is_correct(entries[qid], q, credit_format_errors)
        if c and not b:
            fixed += 1
        elif b and not c:
            broke += 1
    if not paired:
        return None
    disc = fixed + broke
    net = (fixed - broke) / paired
    se = math.sqrt(disc - (fixed - broke) ** 2 / paired) / paired if disc else 0.0
    return {
        "paired": paired,
        "fixed": fixed,
        "broke": broke,
        "discordant": disc,
        "net_pp": net * 100,
        "ci_low_pp": (net - 1.96 * se) * 100,
        "ci_high_pp": (net + 1.96 * se) * 100,
        "p": mcnemar_exact(fixed, broke),
    }


def change_matrix(results, cond):
    base = results.get("baseline", {})
    entries = results.get(cond, {})
    m = {"changed": 0, "WtoR": 0, "RtoW": 0, "WtoW": 0, "same": 0, "pairs": 0}
    for qid, e in entries.items():
        if qid not in base:
            continue
        b = base[qid]
        if classify_entry(e) != "answered" or classify_entry(b) != "answered":
            continue  # only answered pairs
        m["pairs"] += 1
        if e["answer"] == b["answer"]:
            m["same"] += 1
        else:
            m["changed"] += 1
            m[("R" if b["correct"] else "W") + "to" + ("R" if e["correct"] else "W")] += 1
    return m


def reasoning_stats(results, questions, cond):
    entries = results.get(cond, {})
    texts = []
    for e in entries.values():
        t = reason_text(e)
        if t.strip():
            texts.append((len(t), e.get("correct"), e.get("answer")))
    if not texts:
        return None
    lens = [t[0] for t in texts]
    by_correct = {
        True: [t[0] for t in texts if t[1] is True and t[2] != "?"],
        False: [t[0] for t in texts if t[1] is False and t[2] != "?"],
    }  # t[2] is the parsed answer; "?" means no option letter was found
    out = {
        "with_reasoning": len(texts),
        "total_entries": len(entries),
        "mean_len": sum(lens) / len(lens),
        "max_len": max(lens),
    }
    for c, name in ((True, "correct"), (False, "wrong")):
        ls = by_correct[c]
        out[f"mean_len_{name}"] = (sum(ls) / len(ls)) if ls else None
    return out


# ------------------------------------------------------- format error dump ---


def normalize_option(text):
    """Lowercase an option or reply and strip LaTeX so text forms match.

    Options often carry math markup (`3.54 $\\mathrm{eV}$`) while the model's
    reply is plain (`3.54 eV`); both reduce to `3.54 ev` here.
    """
    s = str(text).lower()
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # \mathrm, \text, \frac, ...
    s = re.sub(r"[$}{]", " ", s)  # math delimiters and braces
    s = s.replace("\\", " ")
    return re.sub(r"\s+", " ", s).strip(" .,:;")


def match_option_index(reply, options):
    """Index of the option whose text the reply matches, or None.

    Word-boundary regex on normalized text, so `100` does not match inside
    `1000` and `8.0` does not match inside `18.0`.
    """
    reply_norm = normalize_option(reply)
    if not reply_norm:
        return None
    for i, opt in enumerate(options):
        opt_norm = normalize_option(opt)
        if not opt_norm:
            continue
        if re.search(r"(?<!\w)" + re.escape(opt_norm) + r"(?!\w)", reply_norm):
            return i
    return None


def format_error_rows(files):
    """One row per format parsing error across all files and conditions."""
    rows = []
    for d in files:
        for cond in d["order"]:
            entries = d["results"].get(cond, {})
            for q in d["questions"]:
                e = entries.get(q["id"])
                if not e or classify_entry(e) != "format":
                    continue
                usage = e.get("usage") or {}
                gold = chr(65 + q["answer_index"])
                options = question_options(q)
                idx = match_option_index(e.get("raw"), options)
                rows.append(
                    {
                        "condition": cond,
                        "id": q["id"],
                        "category": q.get("category", "?"),
                        "gold": gold,
                        "answer_text": " ".join((e.get("raw") or "").split()),
                        "regex_letter": chr(65 + idx) if idx is not None else "-",
                        "would_be_correct": (
                            "yes" if idx is not None and chr(65 + idx) == gold else "no"
                        ),
                        "finish_reason": e.get("finish_reason"),
                        "provider": e.get("provider"),
                        "reasoning_tokens": usage.get("reasoning_tokens"),
                        "question": " ".join((q.get("question") or "").split()),
                    }
                )
    return rows


def write_format_errors(files, path):
    """Write format parsing errors to a tab-separated file; return the count."""
    rows = format_error_rows(files)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(FORMAT_ERROR_COLUMNS)
        for r in rows:
            writer.writerow([r[c] for c in FORMAT_ERROR_COLUMNS])
    return len(rows)


# ---------------------------------------------------------------- plots ---


def token_series(d, cond, key):
    """Token counts for one condition and usage key, skipping missing values."""
    out = []
    for e in d["results"].get(cond, {}).values():
        v = (e.get("usage") or {}).get(key)
        if v is not None:
            out.append(v)
    return out


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


def render_tokens(d, stats=None):
    """Token usage per condition, with deltas against baseline.

    Cost and speed both follow from tokens: cost is tokens times price, speed
    is tokens per unit time. The means here use every entry that recorded usage.
    """
    results, order, questions = d["results"], d["order"], d["questions"]
    if stats is None:
        stats = {cond: cond_stats(results, questions, cond) for cond in order}
    base = stats.get("baseline")
    if not base or base["mean_reasoning_tokens"] is None:
        return
    print()
    print("Token usage vs baseline (tokens per call; means, plus median reasoning):")
    w = cond_col_width(order)
    hdr = (
        f"{'Condition':<{w}}{'Prompt':>8}{'Reason':>8}{'R med':>7}"
        f"{'dPrompt':>9}{'dReason':>9}{'dR med':>8}"
        f"{'Reason x':>10}{'95% CI':>14}{'p':>9}"
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
        if cond == "baseline":
            row += f"{'-':>9}{'-':>9}{'-':>8}{'1.00':>10}{'-':>14}{'-':>9}"
        else:
            rr = reason_ratio_stats(results, questions, cond)
            ratio_str = f"{rr['ratio']:.2f}" if rr else "n/a"
            ci_str = f"[{rr['ci_low']:.2f},{rr['ci_high']:.2f}]" if rr else "n/a"
            p_str = (f"{rr['p']:.3f}" if rr["p"] >= 0.001 else "<0.001") if rr else "n/a"
            md = median_reasoning_diff(results, questions, cond)
            md_str = f"{md:>+8.0f}" if md is not None else f"{'n/a':>8}"
            row += (
                fmt_delta(s["mean_prompt_tokens"], base["mean_prompt_tokens"])
                + fmt_delta(s["mean_reasoning_tokens"], base["mean_reasoning_tokens"])
                + md_str
                + f"{ratio_str:>10}{ci_str:>14}{p_str:>9}"
            )
        print(row)


def render_significance(d, credit_format_errors=False):
    order = [c for c in d["order"] if c != "baseline"]
    if not order:
        return
    print()
    note = " (format errors credited)" if credit_format_errors else ""
    print(f"Paired test vs baseline (McNemar exact on discordant pairs){note}:")
    w = cond_col_width(d["order"])
    hdr = f"{'Condition':<{w}}{'Fixed':>7}{'Broke':>7}{'Disc':>6}{'Net':>10}{'95% CI':>20}{'p':>9}"
    print(hdr)
    print("-" * len(hdr))
    for cond in order:
        s = paired_stats(d["results"], d["questions"], cond, credit_format_errors)
        if not s:
            continue
        ci = f"[{s['ci_low_pp']:+.2f}, {s['ci_high_pp']:+.2f}]"
        p = f"{s['p']:.3f}" if s["p"] >= 0.001 else "<0.001"
        print(
            f"{cond:<{w}}{s['fixed']:>7}{s['broke']:>7}{s['discordant']:>6}"
            f"{s['net_pp']:>+9.2f}pp{ci:>20}{p:>9}"
        )


def render_summary(d, credit_format_errors=False, significance=True):
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

    render_tokens(d, stats)

    if significance:
        render_significance(d, credit_format_errors=credit_format_errors)

    print()
    print("Answer changes vs baseline (answered pairs only):")
    hdr2 = f"{'Condition':<{w}}{'Changed':>8}{'W->R':>7}{'R->W':>7}{'W->W':>7}{'Same':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    for cond in order[1:]:
        m = change_matrix(results, cond)
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


def render_json(files, credit_format_errors=False, significance=True):
    out = []
    for d in files:
        results, order, questions = d["results"], d["order"], d["questions"]
        entry = {
            "file": d["meta"]["path"],
            "model": d["meta"]["model"],
            "think": d["meta"]["think"],
            "api": detect_api(results),
            "n": len(questions),
            "credit_format_errors": credit_format_errors,
            "conditions": {},
        }
        base_stats = cond_stats(
            results, questions, "baseline", credit_format_errors=credit_format_errors
        )
        base_acc = base_stats["acc_all"]
        for cond in order:
            s = cond_stats(results, questions, cond, credit_format_errors=credit_format_errors)
            s["deltas_pp_vs_baseline"] = None
            if base_acc is not None and s["acc_all"] is not None:
                s["deltas_pp_vs_baseline"] = round((s["acc_all"] - base_acc) * 100, 1)
            s["paired_vs_baseline"] = (
                paired_stats(results, questions, cond, credit_format_errors)
                if significance and cond != "baseline"
                else None
            )
            entry["conditions"][cond] = s
        out.append(entry)
    print(json.dumps(out, indent=2))


def main():
    args = sys.argv[1:]
    fmt_path = None
    if "--format-errors" in args:
        i = args.index("--format-errors")
        if i + 1 >= len(args):
            print("error: --format-errors needs an output path", file=sys.stderr)
            sys.exit(1)
        fmt_path = args[i + 1]
        del args[i : i + 2]
    plot_path = None
    if "--plot" in args:
        i = args.index("--plot")
        if i + 1 >= len(args):
            print("error: --plot needs an output path", file=sys.stderr)
            sys.exit(1)
        plot_path = args[i + 1]
        del args[i : i + 2]
    credit = "--credit-format-errors" in args
    significance = "--no-significance" not in args
    flags = {
        f: f in args for f in ("--questions", "--categories", "--reasoning", "--compare", "--json")
    }
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(__doc__)
        sys.exit(1)

    files = []
    for p in paths:
        if not os.path.exists(p):
            print(f"error: no such file: {p}", file=sys.stderr)
            sys.exit(1)
        files.append(load_file(p))

    if fmt_path is not None:
        n = write_format_errors(files, fmt_path)
        print(f"Wrote {n} format errors to {fmt_path}")

    if plot_path is not None:
        if len(files) != 1:
            print("error: --plot takes exactly one input file", file=sys.stderr)
            sys.exit(1)
        plot_tokens(files[0], plot_path)
        print(f"Wrote plot to {plot_path}")

    if flags["--json"]:
        render_json(files, credit_format_errors=credit, significance=significance)
        return

    for i, d in enumerate(files):
        if i:
            print()
        if flags["--compare"] and len(files) > 1:
            continue  # compare rendered once below
        render_summary(d, credit_format_errors=credit, significance=significance)
        if flags["--reasoning"]:
            print()
            render_reasoning(d)
        if flags["--categories"]:
            print()
            render_categories(d, credit_format_errors=credit)
        if flags["--questions"]:
            print()
            render_questions(d)

    if flags["--compare"] and len(files) > 1:
        print()
        render_compare(files, credit_format_errors=credit)


if __name__ == "__main__":
    main()
