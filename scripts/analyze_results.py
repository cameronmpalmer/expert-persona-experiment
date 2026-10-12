#!/usr/bin/env python3
"""Clean CLI analysis of persona experiment results.

Handles both result file shapes:
  - local Ollama runs (experiment.py):           entries have "thinking"
  - OpenRouter runs (experiment_openrouter_think.py): entries have "reasoning"

Non-answers ("?") are always counted separately from wrong answers. Partial
results files (run in progress) are handled gracefully - missing entries are
reported per condition.

Usage:
  python3 analyze_results.py FILE [FILE ...]           default summary per file
  python3 analyze_results.py FILE --questions          per-question matrix
  python3 analyze_results.py FILE --categories         accuracy by MMLU category
  python3 analyze_results.py FILE --reasoning          thinking/reasoning stats
  python3 analyze_results.py FILE1 FILE2 --compare     side-by-side per condition
  python3 analyze_results.py FILE --json               machine-readable summary

All views can be combined; --json suppresses the text views.
"""

import json
import os
import sys

DEFAULT_ORDER = ["baseline", "expert", "layperson", "toddler"]
REASON_KEYS = ("reasoning", "thinking")


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


def reason_text(entry):
    for k in REASON_KEYS:
        v = entry.get(k)
        if v:
            return v
    return ""


def cond_stats(results, questions, cond):
    entries = results.get(cond, {})
    n = len(questions)
    correct = sum(1 for e in entries.values() if e.get("correct"))
    nonans = sum(1 for e in entries.values() if e.get("answer") == "?")
    answered = len(entries) - nonans
    wrong = answered - correct
    missing = n - len(entries)
    return {
        "n": n,
        "entries": len(entries),
        "correct": correct,
        "wrong": wrong,
        "nonans": nonans,
        "missing": missing,
        "acc_answered": (correct / answered) if answered else None,
        "acc_all": (correct / n) if n else None,
    }


def change_matrix(results, cond):
    base = results.get("baseline", {})
    entries = results.get(cond, {})
    m = {"changed": 0, "WtoR": 0, "RtoW": 0, "RtoR": 0, "WtoW": 0, "same": 0, "pairs": 0}
    for qid, e in entries.items():
        if qid not in base:
            continue
        b = base[qid]
        if e.get("answer") == "?" or b.get("answer") == "?":
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
    }
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


# ---------------------------------------------------------------- output ---


def fmt_pct(v):
    return f"{v:.1%}" if v is not None else "  n/a"


def render_summary(d):
    meta, results, order, questions = d["meta"], d["results"], d["order"], d["questions"]
    think = {True: "on", False: "off"}.get(
        meta["think"], "off"
    )  # think runs always record think: true
    print(f"File:   {meta['path']}")
    print(
        f"Model:  {meta['model']} | Think: {think} | API: {detect_api(results)} | n={len(questions)}"
    )
    print()
    hdr = f"{'Condition':<11}{'Correct':>8}{'Wrong':>7}{'Non-ans':>9}{'Missing':>9}{'Acc(ans)':>11}{'Acc(all)':>10}"
    print(hdr)
    print("-" * len(hdr))
    stats = {}
    for cond in order:
        s = cond_stats(results, questions, cond)
        stats[cond] = s
        print(
            f"{cond:<11}{s['correct']:>8}{s['wrong']:>7}{s['nonans']:>9}{s['missing']:>9}"
            f"{fmt_pct(s['acc_answered']):>11}{fmt_pct(s['acc_all']):>10}"
        )

    base = stats.get("baseline")
    if base and base["acc_all"] is not None:
        print()
        print("Deltas vs baseline (percentage points; '?' counted as non-answer):")
        for cond in order[1:]:
            s = stats[cond]
            if s["acc_all"] is None:
                continue
            delta = (s["acc_all"] - base["acc_all"]) * 100
            print(f"  {cond:<11}{delta:+5.1f}pp")

    print()
    print("Answer changes vs baseline (answered pairs only):")
    hdr2 = f"{'Condition':<11}{'Changed':>8}{'W->R':>7}{'R->W':>7}{'R->R':>7}{'W->W':>7}{'Same':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    for cond in order[1:]:
        m = change_matrix(results, cond)
        print(
            f"{cond:<11}{m['changed']:>8}{m['WtoR']:>7}{m['RtoW']:>7}{m['RtoR']:>7}{m['WtoW']:>7}{m['same']:>7}"
        )


def render_reasoning(d):
    results, order, questions = d["results"], d["order"], d["questions"]
    print("Reasoning/thinking stats (per condition):")
    hdr = f"{'Condition':<11}{'With':>7}{'Of':>6}{'Cover':>9}{'Mean len':>10}{'Max len':>9}{'Mean(corr)':>12}{'Mean(wrong)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for cond in order:
        r = reasoning_stats(results, questions, cond)
        if not r:
            print(f"{cond:<11}{'no reasoning captured':<60}")
            continue
        print(
            f"{cond:<11}{r['with_reasoning']:>7}{r['total_entries']:>6}"
            f"{r['with_reasoning'] / r['total_entries']:>9.0%}"
            f"{r['mean_len']:>10.0f}{r['max_len']:>9.0f}"
            f"{fmt_pct_len(r['mean_len_correct']):>12}{fmt_pct_len(r['mean_len_wrong']):>12}"
        )
    print()


def fmt_pct_len(v):
    return f"{v:.0f}" if v is not None else "n/a"


def render_questions(d):
    results, order, questions = d["results"], d["order"], d["questions"]
    print(f"{'ID':<6}{'Category':<20}{'Correct':<8}", end="")
    for cond in order:
        print(f"{cond:<10}", end="")
    print()
    print("-" * (6 + 20 + 8 + 10 * len(order)))
    for q in questions:
        qid = q["id"]
        cat = q.get("category", "?")
        correct_letter = chr(65 + q["answer_index"])
        print(f"{qid:<6}{cat:<20}{correct_letter:<8}", end="")
        base_ans = results.get("baseline", {}).get(qid, {}).get("answer")
        for cond in order:
            e = results.get(cond, {}).get(qid)
            if not e:
                print(f"{'-':<10}", end="")
                continue
            ans = e.get("answer", "?")
            if ans == "?":
                print(f"{'?':<10}", end="")
            else:
                mark = "" if ans == base_ans else "*"
                print(f"{ans + mark:<10}", end="")
        print()


def render_categories(d):
    results, order, questions = d["results"], d["order"], d["questions"]
    cats = {}
    for q in questions:
        cat = q.get("category", "?")
        cats.setdefault(cat, []).append(q["id"])
    print(f"{'Category':<20}{'n':>4}", end="")
    for cond in order:
        print(f"{cond:>12}", end="")
    print()
    print("-" * (24 + 12 * len(order)))
    for cat, qids in sorted(cats.items()):
        print(f"{cat:<20}{len(qids):>4}", end="")
        for cond in order:
            entries = results.get(cond, {})
            correct = sum(1 for qid in qids if entries.get(qid, {}).get("correct"))
            print(f"{correct:>7}/{len(qids):<5}", end="")
        print()


def render_compare(files):
    print(f"{'Condition':<11}", end="")
    for d in files:
        name = os.path.basename(d["meta"]["path"])
        print(f"{name[:22]:>24}", end="")
    print()
    print("-" * (11 + 24 * len(files)))
    for cond in files[0]["order"]:
        print(f"{cond:<11}", end="")
        for d in files:
            if cond not in d["results"]:
                print(f"{'n/a':>24}", end="")
                continue
            s = cond_stats(d["results"], d["questions"], cond)
            label = fmt_pct(s["acc_all"])
            if s["nonans"]:
                label += f" ({s['nonans']}?)"
            print(f"{label:>24}", end="")
        print()


def render_json(files):
    out = []
    for d in files:
        results, order, questions = d["results"], d["order"], d["questions"]
        entry = {
            "file": d["meta"]["path"],
            "model": d["meta"]["model"],
            "think": d["meta"]["think"],
            "api": detect_api(results),
            "n": len(questions),
            "conditions": {},
        }
        for cond in order:
            s = cond_stats(results, questions, cond)
            s["deltas_pp_vs_baseline"] = None
            base = results.get("baseline", {})
            base_acc = (
                (sum(1 for e in base.values() if e.get("correct")) / len(questions))
                if questions
                else None
            )
            if base_acc is not None and s["acc_all"] is not None:
                s["deltas_pp_vs_baseline"] = round((s["acc_all"] - base_acc) * 100, 1)
            entry["conditions"][cond] = s
        out.append(entry)
    print(json.dumps(out, indent=2))


def main():
    args = sys.argv[1:]
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

    if flags["--json"]:
        render_json(files)
        return

    for i, d in enumerate(files):
        if i:
            print()
        if flags["--compare"] and len(files) > 1:
            continue  # compare rendered once below
        render_summary(d)
        if flags["--reasoning"]:
            print()
            render_reasoning(d)
        if flags["--categories"]:
            print()
            render_categories(d)
        if flags["--questions"]:
            print()
            render_questions(d)

    if flags["--compare"] and len(files) > 1:
        print()
        render_compare(files)


if __name__ == "__main__":
    main()
