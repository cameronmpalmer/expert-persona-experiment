"""Tests for the cross-model comparison script."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_models.py"
spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

CONDS = cm.PHASE2_CONDITIONS


def question(index):
    return {
        "id": str(index),
        "category": "x",
        "options": ["a", "b"],
        "answer_index": 0,
    }


def make_doc(model, n=2, conditions=CONDS, complete=True, omit=None):
    questions = [question(i + 1) for i in range(n)]
    results = {}
    for condition in conditions:
        if omit and condition == omit:
            continue
        count = n if complete else max(0, n - 1)
        entries = {}
        for i in range(count):
            qid = str(i + 1)
            entries[qid] = {
                "answer": "A",
                "correct": i == 0,
                "raw": "A",
                "latency_s": 1.0 + i,
                "usage": {
                    "cost": 0.001 * (i + 1),
                    "prompt_tokens": 10,
                    "reasoning_tokens": 100 * (i + 1),
                },
            }
        results[condition] = entries
    return {
        "model": model,
        "api": "openrouter",
        "questions": questions,
        "results": results,
        "config": {"provider": "test"},
    }


def write_doc(root, name, doc, mtime):
    path = Path(root) / name
    path.write_text(json.dumps(doc, indent=2))
    os.utime(path, (mtime, mtime))
    return path


def make_run(model, label, accuracy, total_cost, reasoning):
    metrics = {}
    for condition in CONDS:
        metrics[condition] = {
            "n": 100,
            "correct": int(accuracy * 100),
            "acc_all": accuracy,
            "cost": total_cost,
            "reasoning": reasoning,
            "prompt": 10,
            "latency": 10.0,
        }
    return cm.RunSummary(
        path=Path(f"{model}.json"),
        model=model,
        label=label,
        provider="test",
        questions=100,
        metrics=metrics,
    )


class TestDiscoverRuns(unittest.TestCase):
    def test_latest_complete_run_per_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = write_doc(
                tmp,
                "a__2q__xhigh__20261001-000000.json",
                make_doc("model/a"),
                1000,
            )
            new = write_doc(
                tmp,
                "a__2q__xhigh__20261002-000000.json",
                make_doc("model/a"),
                2000,
            )
            complete = write_doc(
                tmp,
                "b__2q__xhigh__20261003-000000.json",
                make_doc("model/b"),
                3000,
            )
            write_doc(
                tmp,
                "c__2q__xhigh__20261004-000000.json",
                make_doc("model/c", complete=False),
                4000,
            )
            write_doc(
                tmp,
                "d__2q__xhigh__20261005-000000.json",
                make_doc("model/d", omit="neutral"),
                5000,
            )
            write_doc(
                tmp,
                "e__500q__xhigh__20261006-000000.json",
                make_doc("model/e"),
                6000,
            )

            runs, skipped = cm.discover_runs(
                Path(tmp),
                questions=2,
                reasoning="xhigh",
                conditions=CONDS,
            )

            by_model = {run.model: run for run in runs}
            self.assertEqual(set(by_model), {"model/a", "model/b"})
            self.assertEqual(by_model["model/a"].path, new)
            self.assertNotEqual(by_model["model/a"].path, old)
            self.assertEqual(by_model["model/b"].path, complete)
            reasons = " ".join(reason for _, reason in skipped)
            self.assertIn("incomplete", reasons)
            self.assertIn("missing conditions", reasons)

            baseline = by_model["model/a"].metrics["baseline"]
            self.assertAlmostEqual(baseline["acc_all"], 0.5)
            self.assertAlmostEqual(baseline["cost"], 0.003)
            self.assertAlmostEqual(baseline["reasoning"], 150)
            self.assertAlmostEqual(baseline["latency"], 1.5)

    def test_model_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_doc(tmp, "a__2q__xhigh__20261001-000000.json", make_doc("model/a"), 1000)
            write_doc(tmp, "b__2q__xhigh__20261002-000000.json", make_doc("model/b"), 2000)
            runs, _ = cm.discover_runs(
                Path(tmp),
                questions=2,
                reasoning="xhigh",
                conditions=CONDS,
                models=("model/b",),
            )
            self.assertEqual([run.model for run in runs], ["model/b"])

    def test_partial_allowed_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_doc(
                tmp,
                "a__2q__xhigh__20261001-000000.json",
                make_doc("model/a", complete=False),
                1000,
            )
            runs, skipped = cm.discover_runs(
                Path(tmp),
                questions=2,
                reasoning="xhigh",
                conditions=CONDS,
                allow_partial=True,
            )
            self.assertEqual([run.model for run in runs], ["model/a"])
            self.assertEqual(skipped, [])


class TestParetoFrontier(unittest.TestCase):
    def test_keeps_non_dominated_points(self):
        points = [(1, 80), (2, 85), (3, 84), (4, 90), (5, 89)]
        self.assertEqual(
            cm.pareto_frontier(points),
            [(1, 80), (2, 85), (4, 90)],
        )

    def test_higher_accuracy_wins_at_same_cost(self):
        self.assertEqual(
            cm.pareto_frontier([(1, 80), (1, 85), (2, 84)]),
            [(1, 85)],
        )

    def test_equal_accuracy_at_higher_cost_is_dominated(self):
        self.assertEqual(cm.pareto_frontier([(1, 80), (2, 80)]), [(1, 80)])

    def test_empty(self):
        self.assertEqual(cm.pareto_frontier([]), [])


class TestPlot(unittest.TestCase):
    def test_writes_png(self):
        runs = [
            make_run("model/a", "Model A", 0.80, 1.5, 1000),
            make_run("model/b", "Model B", 0.85, 2.0, 2000),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "comparison.png"
            cm.plot_comparison(runs, out, CONDS, reasoning="xhigh", dpi=80)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_writes_individual_panels(self):
        runs = [
            make_run("model/a", "Model A", 0.80, 1.5, 1000),
            make_run("model/b", "Model B", 0.85, 2.0, 2000),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "comparison.png"
            paths = cm.plot_individual_panels(runs, out, CONDS, reasoning="xhigh", dpi=80)
            self.assertEqual(len(paths), 5)
            names = {path.name for path in paths}
            self.assertEqual(
                names,
                {
                    "comparison-accuracy.png",
                    "comparison-cost.png",
                    "comparison-reasoning.png",
                    "comparison-latency.png",
                    "comparison-accuracy-vs-cost.png",
                },
            )
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)

    def test_subset_of_conditions(self):
        runs = [make_run("model/a", "Model A", 0.80, 1.5, 1000)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "comparison.png"
            cm.plot_comparison(
                runs,
                out,
                ("baseline", "expert"),
                reasoning="xhigh",
                dpi=80,
            )
            self.assertTrue(out.exists())


class TestLabels(unittest.TestCase):
    def test_known_model_label(self):
        self.assertEqual(cm.model_label("z-ai/glm-5.3-flash"), "GLM 5.3 Flash")

    def test_deepseek_label_uses_lowercase_s(self):
        self.assertEqual(
            cm.model_label("deepseek/deepseek-v4-flash"),
            "Deepseek V4 Flash",
        )

    def test_fallback_model_label(self):
        self.assertEqual(cm.model_label("vendor/model-name"), "model-name")

    def test_condition_label_wraps_underscore(self):
        self.assertEqual(cm.condition_label("expert_short"), "expert\nshort")


if __name__ == "__main__":
    unittest.main()
