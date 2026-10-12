#!/usr/bin/env python3
"""Sample a reproducible subset of questions from an MMLU-Pro style parquet.

Defaults reproduce the committed study sample: the nine study categories,
seed 42, 44 questions from data/mmlu_pro_test.parquet. Point --input/--output at
other parquet files and use --categories and the --*-col flags to sample custom
datasets.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SEED = 42
N = 44
# The study's category list. mmlu_pro_test.parquet spells the fourth one
# "computer science" (with a space), so "computer_science" never matches there;
# keeping it verbatim reproduces the committed sample.
STUDY_CATEGORIES = {
    "physics",
    "chemistry",
    "biology",
    "computer science",
    "engineering",
    "law",
    "economics",
    "math",
    "history",
}


def find_repo_root(start):
    """Return the nearest directory at or above `start` containing pyproject.toml."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


ROOT = find_repo_root(Path(__file__).resolve().parent)
DATA_DIR = ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "mmlu_pro_test.parquet"


def default_output(n):
    """Default output file: data/mmlu_pro_sample_<n>.json (the count is in the name)."""
    return DATA_DIR / f"mmlu_pro_sample_{n}.json"


def parse_categories(spec):
    """Parse a --categories value into a set, or None to keep every category."""
    spec = (spec or "").strip()
    if not spec or spec.lower() == "all":
        return None
    return {c.strip() for c in spec.split(",") if c.strip()}


def is_usable(options, answer):
    """True if a row has at least two options and an in-range 0-based answer index."""
    return (
        isinstance(options, (list, tuple))
        and len(options) >= 2
        and isinstance(answer, int)
        and 0 <= answer < len(options)
    )


def valid_rows(rows, categories, columns):
    """Return (filtered, valid): rows in the wanted categories, then usable rows."""
    filtered = [r for r in rows if not categories or r.get(columns["category"]) in categories]
    valid = [r for r in filtered if is_usable(r.get(columns["options"]), r.get(columns["answer"]))]
    return filtered, valid


def build_sample(valid, n, seed, columns):
    """Shuffle with `seed`, take `n`, sort by category, and assign zero-padded ids."""
    rng = random.Random(seed)
    rng.shuffle(valid)
    chosen = valid[:n]
    chosen.sort(key=lambda r: r.get(columns["category"]) or "")
    return [
        {
            "id": f"{i + 1:02d}",
            "category": r.get(columns["category"]),
            "question": r.get(columns["question"]),
            "options": list(r.get(columns["options"])),
            "answer_index": r.get(columns["answer"]),
        }
        for i, r in enumerate(chosen)
    ]


def confirm_write(count, out_path, exists):
    """Ask before writing; return False when there is no usable answer."""
    if exists:
        print(f"{out_path} already exists and will be overwritten.")
    try:
        answer = input(f"Write {count} questions to {out_path}? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Sample questions from a parquet dataset.")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="source parquet file")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="output JSON file (default data/mmlu_pro_sample_<n>.json)",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the write confirmation (needed for non-interactive use)",
    )
    p.add_argument("-n", "--n", type=int, default=N, help="number of questions to sample")
    p.add_argument("--seed", type=int, default=SEED, help="shuffle seed")
    p.add_argument(
        "--categories",
        default=",".join(sorted(STUDY_CATEGORIES)),
        help="comma-separated categories to include, or 'all'",
    )
    p.add_argument("--question-col", default="question", help="column holding the question text")
    p.add_argument("--options-col", default="options", help="column holding the options list")
    p.add_argument("--answer-col", default="answer_index", help="column holding the 0-based answer")
    p.add_argument("--category-col", default="category", help="column holding the category")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input parquet not found: {input_path}")

    columns = {
        "question": args.question_col,
        "options": args.options_col,
        "answer": args.answer_col,
        "category": args.category_col,
    }
    rows = pq.read_table(input_path).to_pylist()
    filtered, valid = valid_rows(rows, parse_categories(args.categories), columns)
    print(f"rows={len(rows)} filtered={len(filtered)} valid={len(valid)}")

    sample = build_sample(valid, args.n, args.seed, columns)
    if not sample:
        raise SystemExit(
            "no valid rows matched; check --input, --categories, and the --*-col flags"
        )

    out_path = Path(args.output) if args.output else default_output(args.n)
    if not args.yes and not confirm_write(len(sample), out_path, out_path.exists()):
        raise SystemExit("aborted; nothing written (pass --yes to skip the prompt)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"wrote {len(sample)} questions to {out_path}")
    print("Category distribution:", dict(Counter(r["category"] for r in sample)))
    for r in sample:
        print(f"  {r['id']} [{r['category']:>16}] ans={r['answer_index']} {r['question'][:75]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
