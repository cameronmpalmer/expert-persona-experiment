"""Tests for the blank-versus-format-error split in analyze_results.py."""

import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_results.py"
spec = importlib.util.spec_from_file_location("analyze_results", SCRIPT)
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def write_results(tmpdir, results, questions=None, model="test/model", think=False, name="r.json"):
    if questions is None:
        questions = [{"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0}]
    doc = {"model": model, "think": think, "questions": questions, "results": results}
    path = Path(tmpdir) / name
    path.write_text(json.dumps(doc))
    return str(path)


Q = {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0}


class TestClassifyEntry(unittest.TestCase):
    def test_blank_completion_is_non_answer(self):
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": ""}), "blank")
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": "   \n "}), "blank")

    def test_missing_raw_is_blank(self):
        self.assertEqual(ar.classify_entry({"answer": "?"}), "blank")
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": None}), "blank")

    def test_error_entry_with_blank_raw_is_blank(self):
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": "", "error": "Timeout"}), "blank")

    def test_parsed_letter_is_answered(self):
        self.assertEqual(ar.classify_entry({"answer": "B", "raw": "B"}), "answered")
        self.assertEqual(ar.classify_entry({"answer": "C", "raw": "The answer is C."}), "answered")

    def test_non_blank_unparsed_is_format_error(self):
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": "2.5 gallons"}), "format")
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": "0.67"}), "format")
        self.assertEqual(ar.classify_entry({"answer": "?", "raw": "48 percent"}), "format")

    def test_blank_and_format_are_mutually_exclusive(self):
        for entry in ({"answer": "?", "raw": ""}, {"answer": "?", "raw": "17"}):
            self.assertIn(ar.classify_entry(entry), ("blank", "format"))
            self.assertNotEqual(ar.classify_entry(entry), "answered")


class TestCondStats(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "2", "category": "x", "options": ["a", "b"], "answer_index": 1},
            {"id": "3", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "4", "category": "x", "options": ["a", "b"], "answer_index": 0},
        ]
        self.entries = {
            "1": {"answer": "A", "correct": True, "raw": "A"},
            "2": {"answer": "A", "correct": False, "raw": "A"},
            "3": {"answer": "?", "correct": False, "raw": "2.5 gallons"},
            "4": {"answer": "?", "correct": False, "raw": ""},
        }

    def test_counts_split_blanks_from_format_errors(self):
        s = ar.cond_stats({"baseline": self.entries}, self.questions, "baseline")
        self.assertEqual(s["correct"], 1)
        self.assertEqual(s["wrong"], 1)
        self.assertEqual(s["format_errors"], 1)
        self.assertEqual(s["blank"], 1)
        self.assertEqual(s["answered"], 2)

    def test_accuracy_denominators_unchanged(self):
        s = ar.cond_stats({"baseline": self.entries}, self.questions, "baseline")
        self.assertAlmostEqual(s["acc_all"], 0.25)
        self.assertAlmostEqual(s["acc_answered"], 0.5)

    def test_partial_file_reports_missing(self):
        partial = {"1": self.entries["1"]}
        s = ar.cond_stats({"baseline": partial}, self.questions, "baseline")
        self.assertEqual(s["missing"], 3)


class TestChangeMatrix(unittest.TestCase):
    def test_skips_blank_and_format_pairs(self):
        results = {
            "baseline": {
                "1": {"answer": "A", "correct": True, "raw": "A"},
                "2": {"answer": "?", "correct": False, "raw": "12"},
                "3": {"answer": "B", "correct": False, "raw": "B"},
            },
            "expert": {
                "1": {"answer": "B", "correct": False, "raw": "B"},
                "2": {"answer": "A", "correct": True, "raw": "A"},
                "3": {"answer": "?", "correct": False, "raw": ""},
            },
        }
        m = ar.change_matrix(results, "expert")
        self.assertEqual(m["pairs"], 1)
        self.assertEqual(m["RtoW"], 1)


class TestRender(unittest.TestCase):
    def test_summary_shows_blank_and_format_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_results(
                tmp,
                {
                    "baseline": {
                        "1": {"answer": "A", "correct": True, "raw": "A"},
                    }
                },
                questions=[Q],
            )
            d = ar.load_file(path)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d)
            out = buf.getvalue()
            self.assertIn("Blank", out)
            self.assertIn("FmtErr", out)

    def test_json_exposes_blank_and_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_results(
                tmp,
                {
                    "baseline": {
                        "1": {"answer": "?", "correct": False, "raw": ""},
                        "2": {"answer": "?", "correct": False, "raw": "12"},
                        "3": {"answer": "A", "correct": True, "raw": "A"},
                    }
                },
                questions=[
                    Q,
                    {"id": "2", "category": "x", "options": ["a", "b"], "answer_index": 0},
                    {"id": "3", "category": "x", "options": ["a", "b"], "answer_index": 0},
                ],
            )
            d = ar.load_file(path)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_json([d])
            payload = json.loads(buf.getvalue())
            cond = payload[0]["conditions"]["baseline"]
            self.assertEqual(cond["blank"], 1)
            self.assertEqual(cond["format_errors"], 1)
            self.assertEqual(cond["correct"], 1)


class TestCost(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "2", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "3", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "4", "category": "x", "options": ["a", "b"], "answer_index": 0},
        ]
        self.entries = {
            "1": {"answer": "A", "correct": True, "raw": "A", "usage": {"cost": 0.001}},
            "2": {"answer": "?", "correct": False, "raw": "", "usage": {"cost": 0.002}},
            "3": {"answer": "A", "correct": True, "raw": "A", "usage": None},
            "4": {"answer": "A", "correct": True, "raw": "A"},
        }

    def test_cost_sums_usage_and_skips_missing(self):
        s = ar.cond_stats({"baseline": self.entries}, self.questions, "baseline")
        self.assertAlmostEqual(s["cost"], 0.003)
        self.assertEqual(s["cost_entries"], 2)

    def test_summary_shows_total_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_results(tmp, {"baseline": self.entries}, questions=self.questions)
            d = ar.load_file(path)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d)
            out = buf.getvalue()
            self.assertIn("Total cost", out)
            self.assertIn("$0.0030", out)

    def test_json_includes_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_results(tmp, {"baseline": self.entries}, questions=self.questions)
            d = ar.load_file(path)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_json([d])
            cond = json.loads(buf.getvalue())[0]["conditions"]["baseline"]
            self.assertAlmostEqual(cond["cost"], 0.003)
            self.assertEqual(cond["cost_entries"], 2)

    def test_compare_shows_total_cost_per_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = [{"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0}]
            a = write_results(
                tmp,
                {
                    "baseline": {
                        "1": {"answer": "A", "correct": True, "raw": "A", "usage": {"cost": 0.003}}
                    }
                },
                questions=q,
                name="a.json",
            )
            b = write_results(
                tmp,
                {
                    "baseline": {
                        "1": {"answer": "A", "correct": True, "raw": "A", "usage": {"cost": 0.007}}
                    }
                },
                questions=q,
                name="b.json",
            )
            files = [ar.load_file(a), ar.load_file(b)]
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_compare(files)
            out = buf.getvalue()
            self.assertIn("Cost", out)
            self.assertIn("$0.0030", out)
            self.assertIn("$0.0070", out)


class TestFormatErrorDump(unittest.TestCase):
    def _fixture(self, tmp):
        questions = [
            {
                "id": "1",
                "category": "physics",
                "options": ["a", "b"],
                "answer_index": 1,
                "question": "What is 2+2?",
            },
            {
                "id": "2",
                "category": "physics",
                "options": ["a", "b"],
                "answer_index": 0,
                "question": "What is the unit of mass?",
            },
            {
                "id": "3",
                "category": "math",
                "options": ["a", "b"],
                "answer_index": 0,
                "question": "How many gallons?",
            },
        ]
        results = {
            "baseline": {
                "1": {"answer": "B", "correct": True, "raw": "B"},
                "2": {"answer": "?", "correct": False, "raw": ""},
                "3": {
                    "answer": "?",
                    "correct": False,
                    "raw": "2.5 gallons",
                    "finish_reason": "stop",
                    "provider": "Novita",
                    "usage": {"reasoning_tokens": 0},
                },
            }
        }
        return write_results(tmp, results, questions=questions, name="a.json")

    def test_rows_only_include_format_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            rows = ar.format_error_rows([d])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "baseline")
            self.assertEqual(rows[0]["id"], "3")
            self.assertEqual(rows[0]["answer_text"], "2.5 gallons")
            self.assertEqual(rows[0]["gold"], "A")
            self.assertEqual(rows[0]["question"], "How many gallons?")

    def test_write_creates_tsv_with_header_and_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            out = os.path.join(tmp, "fmt.tsv")
            n = ar.write_format_errors([d], out)
            self.assertEqual(n, 1)
            text = Path(out).read_text()
            header = text.splitlines()[0]
            self.assertIn("condition", header)
            self.assertIn("answer_text", header)
            self.assertIn("2.5 gallons", text)
            self.assertNotIn("What is 2+2?", text)

    def test_cli_flag_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture(tmp)
            out = os.path.join(tmp, "cli.tsv")
            argv = sys.argv
            sys.argv = ["analyze_results.py", path, "--format-errors", out]
            try:
                import io
                from contextlib import redirect_stdout

                buf = io.StringIO()
                with redirect_stdout(buf):
                    ar.main()
            finally:
                sys.argv = argv
            self.assertTrue(os.path.exists(out))
            self.assertIn("2.5 gallons", Path(out).read_text())

    def test_regex_flags_wrong_option_as_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = [
                {
                    "id": "1",
                    "category": "physics",
                    "options": ["6.20 eV", "3.54 $\\mathrm{eV}$", "2.33 eV"],
                    "answer_index": 0,
                    "question": "Q",
                }
            ]
            res = {"baseline": {"1": {"answer": "?", "correct": False, "raw": "3.54 eV"}}}
            d = ar.load_file(write_results(tmp, res, questions=q, name="w.json"))
            row = ar.format_error_rows([d])[0]
            self.assertEqual(row["regex_letter"], "B")
            self.assertEqual(row["would_be_correct"], "no")

    def test_regex_flags_gold_option_as_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = [
                {
                    "id": "1",
                    "category": "physics",
                    "options": ["6.20 eV", "3.54 $\\mathrm{eV}$", "2.33 eV"],
                    "answer_index": 1,
                    "question": "Q",
                }
            ]
            res = {"baseline": {"1": {"answer": "?", "correct": False, "raw": "3.54 eV"}}}
            d = ar.load_file(write_results(tmp, res, questions=q, name="g.json"))
            row = ar.format_error_rows([d])[0]
            self.assertEqual(row["regex_letter"], "B")
            self.assertEqual(row["would_be_correct"], "yes")

    def test_regex_no_match_is_dash_and_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            row = [r for r in ar.format_error_rows([d]) if r["id"] == "3"][0]
            self.assertEqual(row["regex_letter"], "-")
            self.assertEqual(row["would_be_correct"], "no")


class TestCreditFormatErrors(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 1},
            {"id": "2", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "3", "category": "x", "options": ["a", "b"], "answer_index": 0},
            {"id": "4", "category": "x", "options": ["a", "b"], "answer_index": 1},
            {"id": "5", "category": "x", "options": ["a", "b"], "answer_index": 0},
        ]
        self.entries = {
            "1": {"answer": "B", "correct": True, "raw": "B"},
            "2": {"answer": "B", "correct": False, "raw": "B"},
            "3": {"answer": "?", "correct": False, "raw": "a"},
            "4": {"answer": "?", "correct": False, "raw": "a"},
            "5": {"answer": "?", "correct": False, "raw": ""},
        }

    def test_strict_ignores_creditable_format_errors(self):
        s = ar.cond_stats({"baseline": self.entries}, self.questions, "baseline")
        self.assertEqual(s["correct"], 1)
        self.assertEqual(s["format_errors"], 2)
        self.assertEqual(s["blank"], 1)
        self.assertEqual(s["credited"], 0)
        self.assertAlmostEqual(s["acc_all"], 0.2)

    def test_credit_marks_matching_format_error_correct(self):
        s = ar.cond_stats(
            {"baseline": self.entries}, self.questions, "baseline", credit_format_errors=True
        )
        self.assertEqual(s["correct"], 2)
        self.assertEqual(s["credited"], 1)
        self.assertEqual(s["format_errors"], 1)
        self.assertEqual(s["blank"], 1)
        self.assertEqual(s["wrong"], 1)
        self.assertAlmostEqual(s["acc_all"], 0.4)
        self.assertAlmostEqual(s["acc_answered"], 2 / 3)

    def test_cli_flag_changes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = [
                {
                    "id": "1",
                    "category": "x",
                    "options": ["a", "b"],
                    "answer_index": 0,
                    "question": "?",
                },
                {
                    "id": "2",
                    "category": "x",
                    "options": ["a", "b"],
                    "answer_index": 0,
                    "question": "?",
                },
                {
                    "id": "3",
                    "category": "x",
                    "options": ["a", "b"],
                    "answer_index": 0,
                    "question": "?",
                },
                {
                    "id": "4",
                    "category": "x",
                    "options": ["a", "b"],
                    "answer_index": 0,
                    "question": "?",
                },
            ]
            res = {
                "baseline": {
                    "1": {"answer": "A", "correct": True, "raw": "A"},
                    "2": {"answer": "B", "correct": False, "raw": "B"},
                    "3": {"answer": "?", "correct": False, "raw": "a"},
                    "4": {"answer": "?", "correct": False, "raw": "b"},
                }
            }
            path = write_results(tmp, res, questions=q, name="c.json")
            import io
            from contextlib import redirect_stdout

            def run(*extra):
                argv = sys.argv
                sys.argv = ["analyze_results.py", path, *extra]
                buf = io.StringIO()
                try:
                    with redirect_stdout(buf):
                        ar.main()
                finally:
                    sys.argv = argv
                return buf.getvalue()

            strict = run()
            lenient = run("--credit-format-errors")
            self.assertIn("25.0%", strict)
            self.assertIn("50.0%", lenient)
            self.assertIn("credited when the reply matches", lenient)


class TestSignificance(unittest.TestCase):
    def _fixture(self, tmp):
        q = [
            {
                "id": str(i),
                "category": "x",
                "options": ["a", "b"],
                "answer_index": 0,
                "question": "?",
            }
            for i in range(1, 5)
        ]
        res = {
            "baseline": {
                "1": {"answer": "A", "correct": True, "raw": "A"},
                "2": {"answer": "A", "correct": True, "raw": "A"},
                "3": {"answer": "A", "correct": False, "raw": "A"},
                "4": {"answer": "A", "correct": False, "raw": "A"},
            },
            "expert": {
                "1": {"answer": "A", "correct": False, "raw": "A"},
                "2": {"answer": "A", "correct": True, "raw": "A"},
                "3": {"answer": "A", "correct": True, "raw": "A"},
                "4": {"answer": "A", "correct": True, "raw": "A"},
            },
        }
        return write_results(tmp, res, questions=q, name="s.json")

    def test_no_deltas_section(self):
        # The "Deltas vs baseline" block duplicated the paired test's Net column,
        # so it was removed; the net still appears there.
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d)
            out = buf.getvalue()
            self.assertNotIn("Deltas vs baseline", out)
            self.assertIn("Paired test vs baseline", out)

    def test_paired_stats_counts_and_net(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            s = ar.paired_stats(d["results"], d["questions"], "expert")
            self.assertEqual(s["fixed"], 2)
            self.assertEqual(s["broke"], 1)
            self.assertEqual(s["discordant"], 3)
            self.assertAlmostEqual(s["net_pp"], 25.0)
            self.assertEqual(s["p"], 1.0)

    def test_summary_shows_mcnemar_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d)
            self.assertIn("McNemar", buf.getvalue())

    def test_no_significance_flag_hides_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture(tmp)
            import io
            from contextlib import redirect_stdout

            def run(*extra):
                argv = sys.argv
                sys.argv = ["analyze_results.py", path, *extra]
                buf = io.StringIO()
                try:
                    with redirect_stdout(buf):
                        ar.main()
                finally:
                    sys.argv = argv
                return buf.getvalue()

            self.assertIn("McNemar", run())
            self.assertNotIn("McNemar", run("--no-significance"))

    def test_json_includes_paired_vs_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_json([d])
            cond = json.loads(buf.getvalue())[0]["conditions"]["expert"]
            self.assertEqual(cond["paired_vs_baseline"]["discordant"], 3)
            self.assertAlmostEqual(cond["paired_vs_baseline"]["net_pp"], 25.0)


class TestTokenUsage(unittest.TestCase):
    def _fixture(self, tmp):
        q = [
            {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0, "question": "?"},
            {"id": "2", "category": "x", "options": ["a", "b"], "answer_index": 0, "question": "?"},
        ]
        res = {
            "baseline": {
                "1": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0},
                },
                "2": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0},
                },
            },
            "expert": {
                "1": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {
                        "prompt_tokens": 139,
                        "completion_tokens": 300,
                        "reasoning_tokens": 250,
                    },
                },
                "2": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {
                        "prompt_tokens": 139,
                        "completion_tokens": 100,
                        "reasoning_tokens": 50,
                    },
                },
            },
        }
        return write_results(tmp, res, questions=q, name="t.json")

    def test_cond_stats_reports_mean_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            s = ar.cond_stats(d["results"], d["questions"], "expert")
            self.assertEqual(s["mean_prompt_tokens"], 139)
            self.assertEqual(s["mean_completion_tokens"], 200)
            self.assertEqual(s["mean_reasoning_tokens"], 150)

    def test_summary_has_token_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d, significance=False)
            out = buf.getvalue()
            self.assertIn("Token usage", out)
            self.assertIn("+39", out)  # prompt delta
            self.assertIn("+150", out)  # reasoning delta


class TestReasonRatio(unittest.TestCase):
    def _fixture(self, tmp):
        q = [
            {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0, "question": "?"},
            {"id": "2", "category": "x", "options": ["a", "b"], "answer_index": 0, "question": "?"},
        ]
        res = {
            "baseline": {
                "1": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"reasoning_tokens": 100},
                },
                "2": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"reasoning_tokens": 100},
                },
            },
            "expert": {
                "1": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"reasoning_tokens": 200},
                },
                "2": {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"reasoning_tokens": 200},
                },
            },
        }
        return write_results(tmp, res, questions=q, name="rr.json")

    def test_ratio_and_sign_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            s = ar.reason_ratio_stats(d["results"], d["questions"], "expert")
            self.assertAlmostEqual(s["ratio"], 2.0)
            self.assertEqual(s["higher"], 2)
            self.assertEqual(s["lower"], 0)
            self.assertEqual(s["p"], 0.5)

    def test_ratio_undefined_when_baseline_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = [
                {
                    "id": "1",
                    "category": "x",
                    "options": ["a", "b"],
                    "answer_index": 0,
                    "question": "?",
                }
            ]
            res = {
                "baseline": {
                    "1": {
                        "answer": "A",
                        "correct": True,
                        "raw": "A",
                        "usage": {"reasoning_tokens": 0},
                    }
                },
                "expert": {
                    "1": {
                        "answer": "A",
                        "correct": True,
                        "raw": "A",
                        "usage": {"reasoning_tokens": 50},
                    }
                },
            }
            d = ar.load_file(write_results(tmp, res, questions=q, name="z.json"))
            self.assertIsNone(ar.reason_ratio_stats(d["results"], d["questions"], "expert"))

    def test_summary_shows_ratio_ci_and_p(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d, significance=False)
            out = buf.getvalue()
            self.assertIn("95% CI", out)
            self.assertIn("2.00", out)  # ratio
            self.assertIn("[2.00,2.00]", out)  # zero-variance CI
            self.assertIn("0.5", out)  # sign-test p


class TestTokenMedians(unittest.TestCase):
    def _fixture(self, tmp):
        q = [
            {
                "id": str(i),
                "category": "x",
                "options": ["a", "b"],
                "answer_index": 0,
                "question": "?",
            }
            for i in range(1, 5)
        ]
        base_reason = [10, 10, 10, 10]
        expert_reason = [10, 20, 30, 400]
        res = {
            "baseline": {
                str(i + 1): {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"reasoning_tokens": base_reason[i]},
                }
                for i in range(4)
            },
            "expert": {
                str(i + 1): {
                    "answer": "A",
                    "correct": True,
                    "raw": "A",
                    "usage": {"reasoning_tokens": expert_reason[i]},
                }
                for i in range(4)
            },
        }
        return write_results(tmp, res, questions=q, name="m.json")

    def test_median_reasoning_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            s = ar.cond_stats(d["results"], d["questions"], "expert")
            self.assertEqual(s["median_reasoning_tokens"], 25)

    def test_paired_median_reasoning_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            self.assertEqual(ar.median_reasoning_diff(d["results"], d["questions"], "expert"), 15)

    def test_summary_shows_mean_and_median_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = ar.load_file(self._fixture(tmp))
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                ar.render_summary(d, significance=False)
            out = buf.getvalue()
            self.assertIn("med", out)
            self.assertIn("+105", out)  # mean reasoning diff
            self.assertIn("+15", out)  # median reasoning diff


class TestHeaderAndAlignment(unittest.TestCase):
    def _or_doc(self, tmp, name="h.json"):
        q = [
            {"id": "1", "category": "x", "options": ["a", "b"], "answer_index": 0, "question": "?"}
        ]
        res = {
            "baseline": {"1": {"answer": "A", "correct": True, "raw": "A"}},
            "expert": {"1": {"answer": "A", "correct": True, "raw": "A"}},
            "expert_short": {"1": {"answer": "A", "correct": True, "raw": "A"}},
        }
        doc = {
            "model": "m",
            "api": "openrouter",
            "config": {"reasoning_mode": "effort", "reasoning_value": "xhigh"},
            "questions": q,
            "conditions": {},
            "results": res,
        }
        path = Path(tmp) / name
        path.write_text(json.dumps(doc))
        return str(path)

    def _render(self, path, significance=True):
        import io
        from contextlib import redirect_stdout

        d = ar.load_file(path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ar.render_summary(d, significance=significance)
        return buf.getvalue()

    def test_think_label_derives_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._render(self._or_doc(tmp))
            self.assertIn("Think: on (xhigh)", out)

    def test_cost_column_aligns_with_long_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._render(self._or_doc(tmp))
            rows = [
                line
                for line in out.splitlines()
                if line.startswith(("baseline", "expert")) and "$" in line
            ]
            self.assertGreaterEqual(len(rows), 3)
            self.assertEqual(len({line.index("$") for line in rows}), 1)

    def test_significance_net_aligns_with_long_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._render(self._or_doc(tmp))
            rows = [line for line in out.splitlines() if re.search(r"[+-]\d+\.\d+pp", line)]
            self.assertGreaterEqual(len(rows), 1)
            starts = {re.search(r"[+-]\d+\.\d+pp", line).start() for line in rows}
            self.assertEqual(len(starts), 1)


class TestThinkingLabel(unittest.TestCase):
    def test_explicit_think_wins(self):
        self.assertEqual(ar.thinking_label({"think": True}), "on")
        self.assertEqual(ar.thinking_label({"think": False}), "off")

    def test_effort_levels(self):
        self.assertEqual(
            ar.thinking_label({"config": {"reasoning_mode": "effort", "reasoning_value": "xhigh"}}),
            "on (xhigh)",
        )
        self.assertEqual(
            ar.thinking_label({"config": {"reasoning_mode": "effort", "reasoning_value": "none"}}),
            "off",
        )

    def test_max_tokens_mode(self):
        self.assertEqual(
            ar.thinking_label(
                {"config": {"reasoning_mode": "max_tokens", "reasoning_value": 4096}}
            ),
            "on (max_tokens 4096)",
        )

    def test_unknown(self):
        self.assertEqual(ar.thinking_label({}), "?")


if __name__ == "__main__":
    unittest.main()
