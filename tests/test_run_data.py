"""Tests for the shared results data layer."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_data as rd  # noqa: E402


def load_analyze_results():
    path = SCRIPTS / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("analyze_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ar = load_analyze_results()


class TestMeanLatency(unittest.TestCase):
    def test_averages_recorded_latencies(self):
        results = {
            "baseline": {
                "1": {"latency_s": 1.0},
                "2": {"latency_s": 3.0},
                "3": {"answer": "A"},
            }
        }
        self.assertEqual(rd.mean_latency(results, "baseline"), 2.0)

    def test_returns_none_when_no_latencies(self):
        self.assertIsNone(rd.mean_latency({"baseline": {"1": {}}}, "baseline"))

    def test_returns_none_for_unknown_condition(self):
        self.assertIsNone(rd.mean_latency({}, "baseline"))


class TestReasonRatioGuard(unittest.TestCase):
    def test_zero_condition_reasoning_returns_none(self):
        questions = [{"id": "1"}, {"id": "2"}]
        results = {
            "baseline": {
                "1": {"usage": {"reasoning_tokens": 100}},
                "2": {"usage": {"reasoning_tokens": 200}},
            },
            "expert": {
                "1": {"usage": {"reasoning_tokens": 0}},
                "2": {"usage": {"reasoning_tokens": 0}},
            },
        }
        self.assertIsNone(rd.reason_ratio_stats(results, questions, "expert", reference="baseline"))


class TestAnalyzeResultsReexports(unittest.TestCase):
    def test_moved_names_still_resolve_through_analyze_results(self):
        self.assertIs(ar.load_file, rd.load_file)
        self.assertIs(ar.cond_stats, rd.cond_stats)
        self.assertIs(ar.classify_entry, rd.classify_entry)
        self.assertIs(ar.mean_latency, rd.mean_latency)
        self.assertIs(ar.DEFAULT_ORDER, rd.DEFAULT_ORDER)
        self.assertIs(ar.FORMAT_ERROR_COLUMNS, rd.FORMAT_ERROR_COLUMNS)

    def test_analyze_results_exposes_render_entry_points(self):
        self.assertTrue(callable(ar.plot_tokens))
        self.assertTrue(callable(ar.render_summary))
        self.assertTrue(callable(ar.main))


if __name__ == "__main__":
    unittest.main()
