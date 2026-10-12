"""Unit tests for scripts/sample_mmlu.py. No parquet or network access."""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_mmlu.py"
_SPEC = importlib.util.spec_from_file_location("sample_mmlu", _SCRIPT)
sm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sm)

_COLS = {"question": "q", "options": "opts", "answer": "ans", "category": "cat"}


class TestParseCategories(unittest.TestCase):
    def test_all_and_empty_mean_no_filter(self):
        self.assertIsNone(sm.parse_categories("all"))
        self.assertIsNone(sm.parse_categories("ALL"))
        self.assertIsNone(sm.parse_categories(""))
        self.assertIsNone(sm.parse_categories(None))

    def test_comma_list_becomes_a_set(self):
        self.assertEqual(sm.parse_categories("math, physics ,math"), {"math", "physics"})


class TestValidRows(unittest.TestCase):
    def _rows(self):
        return [
            {"cat": "math", "q": "keep", "opts": ["a", "b"], "ans": 0},
            {"cat": "law", "q": "other-category", "opts": ["a", "b"], "ans": 1},
            {"cat": "math", "q": "one-option", "opts": ["a"], "ans": 0},
            {"cat": "math", "q": "answer-out-of-range", "opts": ["a", "b"], "ans": 5},
            {"cat": "math", "q": "answer-not-int", "opts": ["a", "b"], "ans": "a"},
        ]

    def test_filters_category_then_usability(self):
        filtered, valid = sm.valid_rows(self._rows(), {"math"}, _COLS)
        self.assertEqual(len(filtered), 4)
        self.assertEqual([r["q"] for r in valid], ["keep"])

    def test_no_categories_keeps_every_category(self):
        filtered, valid = sm.valid_rows(self._rows(), None, _COLS)
        self.assertEqual(len(filtered), 5)
        self.assertEqual(len(valid), 2)


class TestBuildSample(unittest.TestCase):
    def _rows(self):
        return [
            {"cat": "physics", "q": f"p{i}", "opts": ["a", "b"], "ans": 0} for i in range(3)
        ] + [{"cat": "biology", "q": f"b{i}", "opts": ["a", "b"], "ans": 1} for i in range(2)]

    def test_ids_are_padded_and_output_is_sorted_by_category(self):
        sample = sm.build_sample(self._rows(), 5, 42, _COLS)
        self.assertEqual([r["id"] for r in sample], ["01", "02", "03", "04", "05"])
        cats = [r["category"] for r in sample]
        self.assertEqual(cats, sorted(cats))
        self.assertEqual(set(sample[0]), {"id", "category", "question", "options", "answer_index"})
        self.assertEqual(sample[0]["options"], ["a", "b"])

    def test_same_seed_same_sample(self):
        first = sm.build_sample(self._rows(), 3, 42, _COLS)
        second = sm.build_sample(self._rows(), 3, 42, _COLS)
        self.assertEqual(first, second)

    def test_n_larger_than_rows_returns_everything(self):
        self.assertEqual(len(sm.build_sample(self._rows(), 99, 1, _COLS)), 5)


class TestDefaultOutput(unittest.TestCase):
    def test_name_includes_the_count(self):
        self.assertEqual(sm.default_output(100).name, "mmlu_pro_sample_100.json")
        self.assertEqual(sm.default_output(44), sm.DATA_DIR / "mmlu_pro_sample_44.json")


class TestConfirmWrite(unittest.TestCase):
    def test_affirmative(self):
        with mock.patch("builtins.input", return_value="y"):
            self.assertTrue(sm.confirm_write(3, Path("x.json"), False))

    def test_default_is_no(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertFalse(sm.confirm_write(3, Path("x.json"), False))

    def test_eof_is_no(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(sm.confirm_write(3, Path("x.json"), False))


if __name__ == "__main__":
    unittest.main()
