#!/usr/bin/env python3
"""OpenRouter runner for the "Playing Pretend" persona experiment.

Runs the existing four-condition system-prompt protocol (baseline / expert /
layperson / toddler) over the 44-question MMLU-Pro sample against an
OpenAI-compatible OpenRouter endpoint, with thinking enabled.

Design notes:
  - output path is derived from the model slug, so --model can never silently
    reuse another model's results file
  - each entry captures usage, cost, finish_reason, provider, the model the
    provider actually returned, latency, and any error text, so truncation and
    provider routing are analyzable
  - repo root is discovered by walking up from this module to the nearest
    pyproject.toml, so it always reads data/ and writes results/ correctly
  - --max-tokens defaults to 32768 so thinking plus the answer both fit; a
    truncated answer is recorded as a real outcome (not retried into a false
    "non-answer")
  - the request asks OpenRouter to include usage data, and the post-run summary
    totals cost alongside tokens
  - at most `workers` calls are in flight at once, so an abort or Ctrl-C stops
    promptly instead of draining a fully submitted queue

Stdlib only. No network calls happen during --dry-run.

Usage:
  export OPENROUTER_API_KEY=sk-or-...   # also read from ~/.bashrc if present
  python scripts/run_experiment.py                     # full run
  python scripts/run_experiment.py --limit 5           # smoke test
  python scripts/run_experiment.py --dry-run --limit 2 # plan only, no calls
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "inclusionai/ling-3.1-flash"


def find_repo_root(start):
    """Return the nearest directory at or above `start` containing pyproject.toml.

    Resolves the repo root from scripts/ (scripts -> repo).
    """
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


ROOT = find_repo_root(Path(__file__).resolve().parent)
QUESTIONS_FILE = ROOT / "data" / "mmlu_pro_sample.json"
RESULTS_DIR = ROOT / "results"

REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_WORKERS = 8
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
SAVE_EVERY = 10

# Copied verbatim from the original scripts so results are comparable.
CONDITIONS = {
    "baseline": "",
    "expert": "You are a world-class expert in physics, chemistry, biology, computer science, engineering, mathematics, economics, law, and history. You have decades of experience and you never make mistakes.",
    "layperson": "You are a layperson with no specialized training in any technical or academic subject.",
    "toddler": "You are a toddler. You are still learning the basics of the world and have almost no knowledge of science, engineering, law, or mathematics.",
}

USER_INSTRUCTION = (
    "Answer the following multiple-choice question. Reply with only the letter "
    "of the correct answer, e.g. 'A'. Do not include any explanation.\n\n"
)


def utc_now():
    """UTC ISO-8601 timestamp without importing datetime."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slugify(model):
    """Filesystem-safe model slug: lowercase, '/' and ':' become '-'."""
    return model.lower().replace("/", "-").replace(":", "-")


def output_path(model):
    """Default results file for a model: results/{slug}__think.json."""
    return RESULTS_DIR / f"{slugify(model)}__think.json"


def load_api_key():
    """Read OPENROUTER_API_KEY from env, else the export line in ~/.bashrc.

    The key is never printed or logged.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    bashrc = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc):
        with open(bashrc) as f:
            for line in f:
                m = re.match(r"\s*export\s+OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"]+)", line)
                if m:
                    return m.group(1).strip()
    raise SystemExit("OPENROUTER_API_KEY not found in env or ~/.bashrc")


def build_user_message(question, options):
    letters = [chr(65 + i) for i in range(len(options))]
    lines = [f"Question: {question}"]
    lines += [f"{letter}) {opt}" for letter, opt in zip(letters, options)]
    return USER_INSTRUCTION + "\n".join(lines)


def parse_letter(text, n_options):
    letters = [chr(65 + i) for i in range(n_options)]
    pat = re.compile(r"\b([" + "".join(letters) + r"])\b", re.IGNORECASE)
    m = pat.search(text or "")
    return m.group(1).upper() if m else "?"


def extract_usage(data):
    """Pull token/cost accounting out of an OpenRouter response dict."""
    usage = (data or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
    }


def config_signature(model, max_tokens, reasoning_mode, reasoning_value, temperature, seed, provider):
    """Configuration identity used to guard resume against silent drift."""
    return {
        "model": slugify(model),
        "max_tokens": int(max_tokens),
        "reasoning_mode": reasoning_mode,
        "reasoning_value": reasoning_value,
        "temperature": float(temperature),
        "seed": int(seed),
        "provider": provider or None,
    }


def signature_mismatches(stored, current):
    """Keys whose values differ between two signatures (empty list == match)."""
    keys = sorted(set(stored or {}) | set(current or {}))
    return [k for k in keys if (stored or {}).get(k) != (current or {}).get(k)]


def resume_check(existing, config, force=False):
    """Return an error message if resuming an existing file is unsafe, else None.

    Fails closed: a file with a missing/non-dict `config` cannot be verified
    against `config`, so it refuses unless `force`. A config that differs from
    `config` also refuses. `existing=None` (no file) is always fine.
    """
    if force or existing is None:
        return None
    if not isinstance(existing, dict):
        return "existing results file is not a JSON object"
    stored = existing.get("config")
    if not isinstance(stored, dict):
        return ("existing results file has no usable 'config' signature; cannot "
                "verify it matches this run")
    mismatch = signature_mismatches(stored, config)
    if mismatch:
        return ("existing results file was produced with a different config "
                f"(differing keys: {', '.join(mismatch)})")
    return None


def build_payload(model, system_prompt, user_content, temperature, max_tokens, seed,
                  reasoning_mode, reasoning_value, provider=None):
    """Build the OpenRouter chat-completions payload.

    reasoning_mode is "effort" (send reasoning.effort) or "max_tokens" (send
    reasoning.max_tokens). Thinking is always on (exclude=False).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    reasoning = {"exclude": False}
    if reasoning_mode == "max_tokens":
        reasoning["max_tokens"] = reasoning_value
    else:
        reasoning["effort"] = reasoning_value
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning": reasoning,
        "usage": {"include": True},
    }
    if provider:
        payload["provider"] = {"only": [provider], "allow_fallbacks": False}
    return payload


def post_chat(api_key, payload):
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def call_with_retries(api_key, payload):
    """POST with exponential backoff on 429/5xx and transient network errors.

    Other HTTP errors raise immediately (caller records them).
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return post_chat(api_key, payload)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            if e.code == 429 or e.code >= 500:
                last_err = f"HTTP {e.code}: {body[:200]}"
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
        except Exception as e:  # network/parse failures are retried
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))
            continue
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")


def error_entry(error):
    return {
        "answer": "?",
        "correct": False,
        "raw": "",
        "reasoning": "",
        "finish_reason": None,
        "usage": None,
        "provider": None,
        "returned_model": None,
        "latency_s": None,
        "error": error,
        "ts": utc_now(),
    }


def run_task(api_key, model, q, system_prompt, *, max_tokens, temperature, seed,
             reasoning_mode, reasoning_value, provider):
    """Execute one (condition, question) call and return its result entry.

    Empty/truncated answers are recorded as-is (finish_reason preserved); they
    are never retried, so truncation stays distinct from a wrong answer.
    """
    payload = build_payload(
        model, system_prompt, build_user_message(q["question"], q["options"]),
        temperature, max_tokens, seed, reasoning_mode, reasoning_value, provider,
    )
    t0 = time.time()
    try:
        data = call_with_retries(api_key, payload)
    except Exception as e:
        return error_entry(f"{type(e).__name__}: {e}")
    latency = time.time() - t0
    choices = data.get("choices") or [{}]
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning") or ""
    answer = parse_letter(content, len(q["options"]))
    return {
        "answer": answer,
        "correct": answer == chr(65 + q["answer_index"]),
        "raw": content,
        "reasoning": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "usage": extract_usage(data),
        "provider": data.get("provider"),
        "returned_model": data.get("model"),
        "latency_s": latency,
        "error": None,
        "ts": utc_now(),
    }


def save(out_path, model, config, questions, results):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "model": model,
        "api": "openrouter",
        "config": config,
        "questions": questions,
        "conditions": CONDITIONS,
        "results": results,
    }
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, out_path)


def print_summary(results, questions):
    n = len(questions)
    conds = list(CONDITIONS)
    acc = {}
    print(f"\n=== Answers by condition (n={n}) ===")
    for cond in conds:
        present = [results.get(cond, {}).get(q["id"]) for q in questions]
        present = [e for e in present if e]
        correct = sum(1 for e in present if e.get("correct"))
        nonans = sum(1 for e in present if e.get("answer") == "?")
        wrong = len(present) - correct - nonans
        acc[cond] = correct / n if n else 0.0
        print(f"  {cond:<10} correct={correct:>2}  wrong={wrong:>2}  "
              f"non-answer={nonans:>2}  ({acc[cond]:.0%} correct)")

    base = acc.get("baseline", 0.0)
    print("\n=== Deltas vs baseline (pp) ===")
    for cond in conds:
        if cond == "baseline":
            continue
        print(f"  {cond:<10} {acc[cond] - base:+.1%}")

    prompt_tokens = completion_tokens = reasoning_tokens = 0
    reasoning_vals = []
    costs = []
    length_count = 0
    for cond in conds:
        for e in results.get(cond, {}).values():
            if not e:
                continue
            if e.get("finish_reason") == "length":
                length_count += 1
            usage = e.get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
            rv = usage.get("reasoning_tokens")
            if rv is not None:
                reasoning_tokens += rv
                reasoning_vals.append(rv)
            if usage.get("cost") is not None:
                costs.append(usage["cost"])
    median_reasoning = statistics.median(reasoning_vals) if reasoning_vals else 0
    total_cost = f"{sum(costs):.6f}" if costs else "n/a"

    print("\n=== Tokens ===")
    print(f"  total prompt={prompt_tokens}  completion={completion_tokens}  "
          f"reasoning={reasoning_tokens}  median reasoning={median_reasoning}")
    print(f"  total cost: {total_cost}")
    print(f"  finish_reason == 'length' count: {length_count}")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="OpenRouter persona-experiment runner (thinking on).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--reasoning-effort", choices=REASONING_EFFORTS,
                   default=DEFAULT_REASONING_EFFORT)
    p.add_argument("--reasoning-max-tokens", type=int, default=None)
    p.add_argument("--provider", default=None)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")

    if args.reasoning_max_tokens is not None:
        reasoning_mode = "max_tokens"
        reasoning_value = args.reasoning_max_tokens
    else:
        reasoning_mode = "effort"
        reasoning_value = args.reasoning_effort

    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    if args.limit is not None:
        questions = questions[:args.limit]
    n = len(questions)
    if n == 0:
        raise SystemExit("no questions loaded")

    config = config_signature(
        args.model, args.max_tokens, reasoning_mode, reasoning_value,
        args.temperature, args.seed, args.provider,
    )
    out_path = Path(args.out) if args.out else output_path(args.model)

    if args.dry_run:
        planned = n * len(CONDITIONS)
        example_cond = "expert" if "expert" in CONDITIONS else next(iter(CONDITIONS))
        example_payload = build_payload(
            args.model, CONDITIONS[example_cond],
            build_user_message(questions[0]["question"], questions[0]["options"]),
            args.temperature, args.max_tokens, args.seed,
            reasoning_mode, reasoning_value, args.provider,
        )
        print("=== DRY RUN (no network calls) ===")
        print(f"Model:            {args.model}")
        print(f"Questions:        {n}")
        print(f"Conditions:       {len(CONDITIONS)} ({', '.join(CONDITIONS)})")
        print(f"Planned calls:    {planned}")
        print(f"Reasoning:        mode={reasoning_mode} value={reasoning_value}")
        print(f"Resolved out:     {out_path.resolve()}")
        print(f"Example payload (condition='{example_cond}', question id={questions[0]['id']}):")
        print(json.dumps(example_payload, indent=2))
        return 0

    api_key = load_api_key()

    results = {cond: {} for cond in CONDITIONS}
    existing = None
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if not args.force:
                print(f"Refusing to resume: could not read existing results file {out_path} "
                      f"({type(e).__name__}: {e}).", file=sys.stderr)
                print("Re-run with --force to start fresh, or pass --out to a new path.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"Warning: ignoring unreadable results file {out_path} "
                  f"({type(e).__name__}).", file=sys.stderr)
            existing = None

        problem = resume_check(existing, config, args.force)
        if problem:
            print(f"Refusing to resume: {problem}.", file=sys.stderr)
            print("Re-run with --force to extend it anyway, or pass --out to a new path.",
                  file=sys.stderr)
            sys.exit(1)

    if isinstance(existing, dict):
        for cond, entries in (existing.get("results") or {}).items():
            results.setdefault(cond, {})
            if isinstance(entries, dict):
                results[cond].update(entries)
        already = sum(len(v) for v in results.values())
        print(f"Resuming: {already} entries already done")

    tasks = []
    for cond in CONDITIONS:
        for q in questions:
            if q["id"] in results[cond]:
                continue
            tasks.append((cond, q))

    print(f"Model: {args.model} [THINK reasoning={reasoning_mode}:{reasoning_value}] | "
          f"n={n} | {len(tasks)} tasks remaining | temp={args.temperature} | "
          f"workers={args.workers} | max_tokens={args.max_tokens} | out={out_path}")

    def worker(task):
        cond, q = task
        entry = run_task(
            api_key, args.model, q, CONDITIONS[cond],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            reasoning_mode=reasoning_mode,
            reasoning_value=reasoning_value,
            provider=args.provider,
        )
        return cond, q, entry

    t0 = time.time()
    done = 0
    total = len(tasks)
    consecutive_failures = 0
    remaining = iter(tasks)
    pending = {}
    executor = ThreadPoolExecutor(max_workers=args.workers)
    exit_code = 0

    def submit_more():
        # Keep at most `workers` calls in flight, so abort/interrupt never has to
        # drain an already-fully-submitted queue.
        while len(pending) < args.workers:
            try:
                nxt = next(remaining)
            except StopIteration:
                return
            pending[executor.submit(worker, nxt)] = nxt

    try:
        submit_more()
        while pending:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                task = pending.pop(fut)
                cond, q, entry = fut.result()
                results[cond][q["id"]] = entry
                done += 1
                consecutive_failures = consecutive_failures + 1 if entry.get("error") else 0

                if done % SAVE_EVERY == 0 or done == total:
                    save(out_path, args.model, config, questions, results)
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{elapsed:5.0f}s] {done}/{total} done ({rate:.2f}/s)", flush=True)

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    exit_code = 2
                    break
            if exit_code:
                break
            submit_more()
    except KeyboardInterrupt:
        exit_code = 130

    if exit_code:
        # Stop scheduling immediately; cancel anything still queued rather than
        # waiting for up to a full queue of slow calls.
        save(out_path, args.model, config, questions, results)
        executor.shutdown(wait=False, cancel_futures=True)
        if exit_code == 2:
            print(f"\nABORT: {consecutive_failures} consecutive failures - API may be "
                  "wedged. Progress saved; rerun to resume.", file=sys.stderr, flush=True)
        else:
            print(f"\nInterrupted - progress saved to {out_path} ({done}/{total} done). "
                  "Rerun to resume.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    executor.shutdown(wait=True)
    save(out_path, args.model, config, questions, results)
    print(f"\nModel: {args.model} [THINK] | MMLU-Pro sample n={n} | temp={args.temperature} | "
          f"elapsed {time.time() - t0:.0f}s")
    print_summary(results, questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
