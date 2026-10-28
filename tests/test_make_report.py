"""Tests for the report regeneration script."""

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_make_report():
    path = SCRIPTS / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mr = load_make_report()

CONDS_P1 = ("baseline", "expert", "layperson", "toddler")
CONDS_P2 = mr.cm.PHASE2_CONDITIONS


def question(index):
    return {
        "id": str(index),
        "category": "x",
        "options": ["a", "b"],
        "answer_index": 0,
    }


def make_doc(model, n=2, conditions=CONDS_P2, complete=True):
    questions = [question(i + 1) for i in range(n)]
    results = {}
    for condition in conditions:
        count = n if complete else max(0, n - 1)
        entries = {}
        for i in range(count):
            entries[str(i + 1)] = {
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
        "config": {
            "provider": "test",
            "reasoning_mode": "effort",
            "reasoning_value": "xhigh",
        },
    }


def write_doc(root, name, doc):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def build_tree(tmp):
    results = Path(tmp) / "results"
    write_doc(
        results / "phase1",
        "ling__2q__xhigh__20261001-000000.json",
        make_doc("model/a", conditions=CONDS_P1),
    )
    write_doc(
        results / "phase2",
        "ling__2q__xhigh__20261002-000000.json",
        make_doc("inclusionai/ling-3.0-flash"),
    )
    return results


class TestReasoningLabel(unittest.TestCase):
    def test_effort_none_is_none(self):
        self.assertEqual(
            mr.reasoning_label({"reasoning_mode": "effort", "reasoning_value": "none"}),
            "none",
        )

    def test_effort_value_is_preserved(self):
        self.assertEqual(
            mr.reasoning_label({"reasoning_mode": "effort", "reasoning_value": "xhigh"}),
            "xhigh",
        )

    def test_max_tokens_has_prefix(self):
        self.assertEqual(
            mr.reasoning_label({"reasoning_mode": "max_tokens", "reasoning_value": 512}),
            "maxtokens512",
        )

    def test_unknown_config_is_question_mark(self):
        self.assertEqual(mr.reasoning_label(None), "?")
        self.assertEqual(mr.reasoning_label({}), "?")


class TestDiscoverRuns(unittest.TestCase):
    def test_discovers_both_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            runs = mr.discover_runs(results)

        self.assertEqual([run.phase for run in runs], ["phase1", "phase2"])
        by_phase = {run.phase: run for run in runs}
        self.assertEqual(by_phase["phase1"].model, "model/a")
        self.assertEqual(by_phase["phase1"].questions, 2)
        self.assertEqual(by_phase["phase1"].reasoning, "xhigh")
        self.assertTrue(by_phase["phase1"].complete)
        self.assertTrue(by_phase["phase1"].file.endswith("ling__2q__xhigh__20261001-000000.json"))
        self.assertEqual(by_phase["phase2"].model, "inclusionai/ling-3.0-flash")
        self.assertEqual(by_phase["phase2"].provider, "test")
        self.assertTrue(by_phase["phase2"].complete)

    def test_probe_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            write_doc(
                results / "phase2",
                "spike_model__provider.json",
                make_doc("model/spike"),
            )
            runs = mr.discover_runs(results)

        self.assertEqual(len(runs), 2)
        self.assertFalse(any("spike" in run.file for run in runs))

    def test_unreadable_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            (results / "phase1" / "broken.json").write_text("{")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                runs = mr.discover_runs(results)

        self.assertEqual(len(runs), 2)
        self.assertIn("skipping", stderr.getvalue())


class TestWriters(unittest.TestCase):
    def test_runs_csv_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            runs = mr.discover_runs(results)
            out = Path(tmp) / "tables" / "runs.csv"
            count = mr.write_runs_csv(runs, out)

            with open(out, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, mr.RUN_COLUMNS)
                rows = list(reader)

        self.assertEqual(count, 9)
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["phase"], "phase1")
        self.assertEqual(rows[0]["questions"], "2")
        self.assertEqual(rows[0]["complete"], "yes")
        self.assertEqual(rows[0]["condition"], "baseline")

    def test_model_comparison_csv_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            runs, skipped = mr.cm.discover_runs(results / "phase2", questions=2, reasoning="xhigh")
            out = Path(tmp) / "tables" / "model-comparison-2q.csv"
            count = mr.write_model_comparison_csv(runs, out)

            with open(out, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, mr.MODEL_COMPARISON_COLUMNS)
                rows = list(reader)

        self.assertEqual(skipped, [])
        self.assertEqual(count, 5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["model"], "inclusionai/ling-3.0-flash")
        self.assertEqual(rows[0]["condition"], "neutral")
        self.assertEqual(rows[0]["n"], "2")
        self.assertEqual(rows[0]["mean_latency_s"], "1.5")

    def test_paired_csv_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            runs = mr.discover_runs(results)
            out = Path(tmp) / "tables" / "phase1-2q-paired.csv"
            count = mr.write_paired_csv(runs, out, 2)

            with open(out, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, mr.PAIRED_COLUMNS)
                rows = list(reader)

        self.assertEqual(count, 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["reference"] == "baseline" for row in rows))
        self.assertTrue(all(row["paired"] == "2" for row in rows))

    def test_phase2_paired_csv_uses_both_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            runs = mr.discover_runs(results)
            model_runs, _ = mr.cm.discover_runs(results / "phase2", questions=2, reasoning="xhigh")
            out = Path(tmp) / "tables" / "phase2-2q-paired.csv"
            count = mr.write_phase2_paired_csv(model_runs, out, runs)

            with open(out, newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(count, 8)
        self.assertEqual({row["reference"] for row in rows}, {"baseline", "neutral"})
        self.assertTrue(all(row["phase"] == "phase2" for row in rows))
        self.assertTrue(all(row["questions"] == "2" for row in rows))
        self.assertTrue(all(row["reasoning"] == "xhigh" for row in rows))

    def test_paired_csv_is_header_only_without_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tables" / "phase1-5000q-paired.csv"
            count = mr.write_paired_csv([], out, 5000)

            with open(out, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, mr.PAIRED_COLUMNS)
                self.assertEqual(list(reader), [])

        self.assertEqual(count, 0)


class TestRunFigures(unittest.TestCase):
    def test_names_shared_stems_per_phase_and_removes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            doc = make_doc("model/a", conditions=CONDS_P1)
            write_doc(results / "phase1", "same__2q__xhigh.json", doc)
            write_doc(results / "phase2", "same__2q__xhigh.json", doc)
            figures = Path(tmp) / "figures"
            stale = figures / "runs" / "gone.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old")

            def fake_plot(_data, path):
                Path(path).write_bytes(b"png")

            with mock.patch.object(mr.ar, "plot_tokens", side_effect=fake_plot):
                written = mr.write_run_figures(mr.discover_runs(results), figures)

            names = sorted(path.name for path in written)
            self.assertEqual(names, ["phase1-same__2q__xhigh.png", "phase2-same__2q__xhigh.png"])
            self.assertFalse(stale.exists())


class TestMain(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = mr.parse_args([])
        self.assertEqual(args.results_dir, str(mr.REPO_ROOT / "results"))
        self.assertEqual(args.tables_dir, str(mr.REPO_ROOT / "tables"))
        self.assertEqual(args.figures_dir, str(mr.REPO_ROOT / "figures"))
        self.assertEqual(args.questions, 5000)
        self.assertEqual(args.reasoning, "xhigh")
        self.assertEqual(args.dpi, 160)
        self.assertFalse(args.no_run_figures)
        self.assertFalse(args.allow_partial)

    def test_empty_results_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            tables = Path(tmp) / "tables"
            figures = Path(tmp) / "figures"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = mr.main(
                    [
                        "--results-dir",
                        str(results),
                        "--tables-dir",
                        str(tables),
                        "--figures-dir",
                        str(figures),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertFalse(tables.exists())
            self.assertFalse(figures.exists())
            self.assertIn("no readable runs", stderr.getvalue())

    def test_no_run_figures_writes_tables_and_skips_run_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = build_tree(tmp)
            tables = Path(tmp) / "tables"
            figures = Path(tmp) / "figures"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(mr.cm, "plot_comparison") as plot_comparison,
                mock.patch.object(mr.cm, "plot_individual_panels", return_value=[]) as plot_panels,
                mock.patch.object(mr.ar, "plot_tokens") as plot_tokens,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = mr.main(
                    [
                        "--results-dir",
                        str(results),
                        "--tables-dir",
                        str(tables),
                        "--figures-dir",
                        str(figures),
                        "--questions",
                        "2",
                        "--no-run-figures",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue((tables / "runs.csv").exists())
            self.assertTrue((tables / "model-comparison-2q.csv").exists())
            self.assertTrue((tables / "phase1-2q-paired.csv").exists())
            self.assertFalse((figures / "runs").exists())
            plot_comparison.assert_called_once()
            plot_panels.assert_called_once()
            plot_tokens.assert_not_called()
            self.assertIn("Skipped run figures", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
