#!/usr/bin/env python3
"""Shared data layer for the persona experiment result files.

This module is deliberately free of matplotlib and CLI rendering. It loads run
JSON files, classifies entries, and computes the statistics used by the analysis,
comparison, and report scripts.
"""

import csv
import json
import math
import re
import statistics

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


def z_for_confidence(confidence: float) -> float:
    """Two-sided normal critical value for a confidence level."""
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2)


def reason_ratio_stats(results, questions, cond, reference="baseline", confidence=0.95):
    """Reasoning-token ratio of a condition against a reference, with CI and p.

    Ratio is mean(condition) / mean(reference) on shared questions. The 95%
    interval is a delta-method interval on the log ratio (fast and stable for
    these heavy-tailed counts). The p-value is the exact sign test on the
    per-question differences, so it answers whether reasoning moved at all.
    Returns None when reference reasoning is zero, since the ratio is undefined.
    """
    base = results.get(reference, {})
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
    if mx <= 0 or my <= 0:
        return None
    higher = sum(1 for a, b in zip(xs, ys, strict=True) if a > b)
    lower = sum(1 for a, b in zip(xs, ys, strict=True) if a < b)
    vx = sum((a - mx) ** 2 for a in xs) / (n - 1)
    vy = sum((b - my) ** 2 for b in ys) / (n - 1)
    cxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / (n - 1)
    var_log = max(0.0, vx / (n * mx * mx) + vy / (n * my * my) - 2 * cxy / (n * mx * my))
    se_log = math.sqrt(var_log)
    ratio = mx / my
    z = z_for_confidence(confidence)
    return {
        "ratio": ratio,
        "ci_low": ratio * math.exp(-z * se_log),
        "ci_high": ratio * math.exp(z * se_log),
        "higher": higher,
        "lower": lower,
        "p": mcnemar_exact(higher, lower),
    }


def median_reasoning_diff(results, questions, cond, reference="baseline"):
    """Median of the per-question reasoning-token difference vs a reference.

    Robust companion to the mean difference, which a few long traces can carry.
    Returns None when there are no paired questions with reasoning tokens.
    """
    base = results.get(reference, {})
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


def paired_stats(
    results,
    questions,
    cond,
    credit_format_errors=False,
    reference="baseline",
    confidence=0.95,
):
    """Paired comparison of one condition against a reference on shared questions.

    Returns fixed (reference wrong, condition right), broke (the reverse), the
    discordant total, the net difference in percentage points, a confidence
    interval from the McNemar normal approximation, and the exact McNemar p-value.
    """
    base = results.get(reference, {})
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
    z = z_for_confidence(confidence)
    return {
        "paired": paired,
        "fixed": fixed,
        "broke": broke,
        "discordant": disc,
        "net_pp": net * 100,
        "ci_low_pp": (net - z * se) * 100,
        "ci_high_pp": (net + z * se) * 100,
        "p": mcnemar_exact(fixed, broke),
    }


def change_matrix(results, cond, reference="baseline"):
    base = results.get(reference, {})
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


def token_series(d, cond, key):
    """Token counts for one condition and usage key, skipping missing values."""
    out = []
    for e in d["results"].get(cond, {}).values():
        v = (e.get("usage") or {}).get(key)
        if v is not None:
            out.append(v)
    return out


def mean_latency(results, cond):
    """Mean latency in seconds over entries that recorded one, or None."""
    latencies = [
        entry.get("latency_s")
        for entry in results.get(cond, {}).values()
        if entry.get("latency_s") is not None
    ]
    return (sum(latencies) / len(latencies)) if latencies else None
