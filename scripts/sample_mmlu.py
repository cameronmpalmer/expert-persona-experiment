#!/usr/bin/env python3
"""Sample a reproducible 44-question MMLU-Pro test subset (science/engineering/law
categories, mirroring the Playing Pretend study). Reads the parquet via pyarrow."""

import json
import random
from collections import Counter

import pyarrow.parquet as pq

SEED = 42
N = 44
CATEGORIES = {
    "physics",
    "chemistry",
    "biology",
    "computer_science",
    "engineering",
    "law",
    "economics",
    "math",
    "history",
}

t = pq.read_table("mmlu_pro_test.parquet")
data = t.to_pylist()

filtered = [r for r in data if r.get("category") in CATEGORIES]
valid = [
    r
    for r in filtered
    if isinstance(r.get("options"), list)
    and len(r["options"]) >= 2
    and isinstance(r.get("answer_index"), int)
    and 0 <= r["answer_index"] < len(r["options"])
]
print(f"rows={len(data)} filtered={len(filtered)} valid={len(valid)}")

rng = random.Random(SEED)
rng.shuffle(valid)
sample = valid[:N]
sample.sort(key=lambda r: r.get("category", ""))

out = [
    {
        "id": f"{i + 1:02d}",
        "category": r.get("category"),
        "question": r["question"],
        "options": r["options"],
        "answer_index": r["answer_index"],
    }
    for i, r in enumerate(sample)
]

with open("mmlu_pro_sample.json", "w") as f:
    json.dump(out, f, indent=2)

print("Category distribution:", dict(Counter(r["category"] for r in out)))
for r in out:
    print(f"  {r['id']} [{r['category']:>16}] ans={r['answer_index']} {r['question'][:75]}...")
