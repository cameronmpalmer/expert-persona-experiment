"""Unit tests for scripts/run_experiment.py. No network access."""

import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_experiment", _SCRIPT)
rx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rx)


class TestParseLetter(unittest.TestCase):
    def test_single_letter(self):
        self.assertEqual(rx.parse_letter("A", 4), "A")

    def test_letter_inside_sentence(self):
        self.assertEqual(rx.parse_letter("The answer is C.", 4), "C")

    def test_letter_with_paren(self):
        self.assertEqual(rx.parse_letter("B) because it works", 4), "B")

    def test_no_letter_is_question_mark(self):
        self.assertEqual(rx.parse_letter("no letter here", 4), "?")

    def test_empty_is_question_mark(self):
        self.assertEqual(rx.parse_letter("", 4), "?")

    def test_ignores_letters_outside_option_range(self):
        # 2 options -> only A and B count; "D" must not match.
        self.assertEqual(rx.parse_letter("D", 2), "?")

    def test_lowercase_letter_is_normalized(self):
        self.assertEqual(rx.parse_letter("the answer is c", 4), "C")


class TestModelSlugAndPaths(unittest.TestCase):
    def test_slug_for_default_model(self):
        self.assertEqual(rx.slugify("inclusionai/ling-3.1-flash"), "inclusionai-ling-3.1-flash")

    def test_slug_replaces_colon(self):
        self.assertEqual(rx.slugify("qwen3.5:9b"), "qwen3.5-9b")

    def test_reasoning_tag_effort_is_the_level(self):
        self.assertEqual(rx.reasoning_tag("effort", "none"), "none")
        self.assertEqual(rx.reasoning_tag("effort", "high"), "high")

    def test_reasoning_tag_max_tokens(self):
        self.assertEqual(rx.reasoning_tag("max_tokens", 4096), "maxtokens4096")

    def test_output_path_uses_reasoning_tag_and_timestamp(self):
        path = rx.output_path("inclusionai/ling-3.1-flash", "high", "20261012-153000")
        expected = rx.ROOT / "results" / "inclusionai-ling-3.1-flash__high__20261012-153000.json"
        self.assertEqual(path, expected)

    def test_output_path_tag_without_timestamp(self):
        expected = rx.ROOT / "results" / "inclusionai-ling-3.1-flash__none.json"
        self.assertEqual(rx.output_path("inclusionai/ling-3.1-flash", "none"), expected)

    def test_run_stamp_format(self):
        when = time.strptime("2026-10-12 15:30:00", "%Y-%m-%d %H:%M:%S")
        self.assertEqual(rx.run_stamp(when), "20261012-153000")

    def test_questions_file_lives_under_repo_root(self):
        self.assertEqual(rx.QUESTIONS_FILE, rx.ROOT / "data" / "mmlu_pro_sample.json")


class TestLatestResultsPath(unittest.TestCase):
    def _touch(self, path, mtime):
        path.write_text("{}")
        os.utime(path, (mtime, mtime))

    def test_matches_legacy_think_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            older = d / "m__think__20260101-000000.json"
            newer = d / "m__think__20260202-000000.json"
            self._touch(older, 1000)
            self._touch(newer, 2000)
            self.assertEqual(rx.latest_results_path("m", "effort", "high", d), newer)

    def test_matches_new_tag_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            high = d / "m__high__20260202-000000.json"
            none = d / "m__none__20260101-000000.json"
            self._touch(high, 2000)
            self._touch(none, 1000)
            self.assertEqual(rx.latest_results_path("m", "effort", "high", d), high)
            self.assertEqual(rx.latest_results_path("m", "effort", "none", d), none)

    def test_thinking_off_ignores_think_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            think = d / "m__think__20260202-000000.json"
            plain = d / "m__20260101-000000.json"
            self._touch(think, 2000)
            self._touch(plain, 1000)
            self.assertEqual(rx.latest_results_path("m", "effort", "none", d), plain)

    def test_ignores_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._touch(d / "other__think__20260202-000000.json", 2000)
            self.assertIsNone(rx.latest_results_path("m", "effort", "high", d))

    def test_none_when_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(rx.latest_results_path("m", "effort", "high", Path(tmp)))


class TestParseArgs(unittest.TestCase):
    def test_questions_file_defaults_to_none(self):
        self.assertIsNone(rx.parse_args([]).questions_file)

    def test_questions_file_flag(self):
        self.assertEqual(
            rx.parse_args(["--questions-file", "/tmp/q.json"]).questions_file, "/tmp/q.json"
        )


class TestExtractUsage(unittest.TestCase):
    def test_full_response_with_reasoning_and_provider(self):
        response = {
            "id": "gen-abc",
            "model": "inclusionai/ling-3.1-flash",
            "provider": "InclusionAI",
            "choices": [{"finish_reason": "stop", "message": {"content": "C", "reasoning": "..."}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 800,
                "total_tokens": 920,
                "cost": 0.0,
                "completion_tokens_details": {"reasoning_tokens": 650},
            },
        }
        self.assertEqual(
            rx.extract_usage(response),
            {
                "prompt_tokens": 120,
                "completion_tokens": 800,
                "reasoning_tokens": 650,
                "total_tokens": 920,
                "cost": 0.0,
            },
        )

    def test_missing_cost_and_details_are_none(self):
        response = {"usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}}
        usage = rx.extract_usage(response)
        self.assertIsNone(usage["cost"])
        self.assertIsNone(usage["reasoning_tokens"])

    def test_missing_usage_is_all_none(self):
        usage = rx.extract_usage({})
        self.assertEqual(
            usage,
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
                "cost": None,
            },
        )


class TestConfigSignature(unittest.TestCase):
    def _sig(self, **overrides):
        kwargs = {
            "model": "inclusionai/ling-3.1-flash",
            "max_tokens": 32768,
            "reasoning_mode": "effort",
            "reasoning_value": "high",
            "temperature": 0.0,
            "seed": 42,
            "provider": None,
        }
        kwargs.update(overrides)
        return rx.config_signature(**kwargs)

    def test_equal_signatures_match(self):
        self.assertEqual(rx.signature_mismatches(self._sig(), self._sig()), [])

    def test_provider_change_detected(self):
        self.assertEqual(
            rx.signature_mismatches(self._sig(), self._sig(provider="Together")),
            ["provider"],
        )

    def test_reasoning_mode_change_detected(self):
        mismatch = rx.signature_mismatches(
            self._sig(), self._sig(reasoning_mode="max_tokens", reasoning_value=4096)
        )
        self.assertIn("reasoning_mode", mismatch)
        self.assertIn("reasoning_value", mismatch)

    def test_max_tokens_change_detected(self):
        self.assertEqual(
            rx.signature_mismatches(self._sig(), self._sig(max_tokens=8192)), ["max_tokens"]
        )

    def test_temperature_change_detected(self):
        self.assertEqual(
            rx.signature_mismatches(self._sig(), self._sig(temperature=0.7)), ["temperature"]
        )

    def test_seed_change_detected(self):
        self.assertEqual(rx.signature_mismatches(self._sig(), self._sig(seed=1)), ["seed"])

    def test_model_slug_used_in_signature(self):
        self.assertEqual(self._sig()["model"], "inclusionai-ling-3.1-flash")


class TestResumeCheck(unittest.TestCase):
    def _config(self):
        return rx.config_signature(
            "inclusionai/ling-3.1-flash", 32768, "effort", "high", 0.0, 42, None
        )

    def test_no_file_is_allowed(self):
        self.assertIsNone(rx.resume_check(None, self._config()))

    def test_matching_config_is_allowed(self):
        self.assertIsNone(rx.resume_check({"config": self._config()}, self._config()))

    def test_missing_config_refuses(self):
        msg = rx.resume_check({"model": "legacy", "results": {}}, self._config())
        self.assertIsNotNone(msg)
        self.assertIn("config", msg)

    def test_non_dict_config_refuses(self):
        self.assertIsNotNone(rx.resume_check({"config": "nope"}, self._config()))

    def test_non_dict_file_refuses(self):
        self.assertIsNotNone(rx.resume_check([1, 2, 3], self._config()))

    def test_mismatched_config_names_differing_keys(self):
        stored = rx.config_signature(
            "inclusionai/ling-3.1-flash", 8192, "effort", "high", 0.0, 42, None
        )
        msg = rx.resume_check({"config": stored}, self._config())
        self.assertIsNotNone(msg)
        self.assertIn("max_tokens", msg)

    def test_force_bypasses_all_checks(self):
        self.assertIsNone(rx.resume_check({"model": "legacy"}, self._config(), force=True))
        self.assertIsNone(rx.resume_check([1, 2], self._config(), force=True))


class TestBuildPayload(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = {
            "model": "inclusionai/ling-3.1-flash",
            "system_prompt": "You are an expert.",
            "user_content": "Q?",
            "temperature": 0.0,
            "max_tokens": 32768,
            "seed": 42,
            "reasoning_mode": "effort",
            "reasoning_value": "high",
            "provider": None,
        }
        kwargs.update(overrides)
        return rx.build_payload(**kwargs)

    def test_reasoning_on_by_default(self):
        payload = self._build()
        self.assertEqual(payload["reasoning"], {"exclude": False, "effort": "high"})
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][-1]["role"], "user")
        self.assertEqual(payload["usage"], {"include": True})

    def test_no_provider_by_default(self):
        self.assertNotIn("provider", self._build())

    def test_provider_pinning(self):
        payload = self._build(provider="Together")
        self.assertEqual(payload["provider"], {"only": ["Together"], "allow_fallbacks": False})

    def test_reasoning_max_tokens_switches_field(self):
        payload = self._build(reasoning_mode="max_tokens", reasoning_value=4096)
        self.assertEqual(payload["reasoning"], {"exclude": False, "max_tokens": 4096})
        self.assertNotIn("effort", payload["reasoning"])

    def test_reasoning_none_disables_thinking(self):
        payload = self._build(reasoning_value="none")
        self.assertEqual(payload["reasoning"], {"enabled": False})

    def test_empty_persona_omits_system_message(self):
        payload = self._build(system_prompt="")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")


class TestThinkingEnabled(unittest.TestCase):
    def test_effort_none_is_off(self):
        self.assertFalse(rx.thinking_enabled("effort", "none"))

    def test_effort_value_is_on(self):
        self.assertTrue(rx.thinking_enabled("effort", "high"))

    def test_max_tokens_mode_is_on(self):
        self.assertTrue(rx.thinking_enabled("max_tokens", 4096))


class TestErrorEntry(unittest.TestCase):
    def test_error_entry_is_a_non_answer(self):
        entry = rx.error_entry("HTTP 400: bad request")
        self.assertEqual(entry["answer"], "?")
        self.assertFalse(entry["correct"])
        self.assertEqual(entry["error"], "HTTP 400: bad request")


class TestRunTask(unittest.TestCase):
    Q = {"id": "01", "question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": 0}

    def test_truncated_empty_answer_is_non_answer_and_not_retried(self):
        response = {
            "model": "inclusionai/ling-3.1-flash",
            "provider": "InclusionAI",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning": "long thought..."},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 32768,
                "total_tokens": 32778,
                "cost": 0.0,
                "completion_tokens_details": {"reasoning_tokens": 32700},
            },
        }
        calls = []

        def fake_call(api_key, payload):
            calls.append(payload)
            return response

        original = rx.call_with_retries
        rx.call_with_retries = fake_call
        try:
            entry = rx.run_task(
                "fake-key",
                "inclusionai/ling-3.1-flash",
                self.Q,
                "",
                max_tokens=32768,
                temperature=0.0,
                seed=42,
                reasoning_mode="effort",
                reasoning_value="high",
                provider=None,
            )
        finally:
            rx.call_with_retries = original

        self.assertEqual(len(calls), 1)  # not retried
        self.assertEqual(entry["answer"], "?")
        self.assertFalse(entry["correct"])
        self.assertEqual(entry["finish_reason"], "length")
        self.assertIsNone(entry["error"])
        self.assertEqual(entry["provider"], "InclusionAI")


class _FakeBar:
    def __init__(self):
        self.postfix = None
        self.advanced = 0
        self.closed = False

    def set_postfix_str(self, s):
        self.postfix = s

    def update(self, n):
        self.advanced += n

    def close(self):
        self.closed = True


class TestProgress(unittest.TestCase):
    def test_no_work_returns_no_bar(self):
        self.assertIsNone(rx.start_progress(0))

    def test_update_progress_none_bar_is_noop(self):
        rx.update_progress(None, ok=1, fail=0, cost=0.0, t0=time.time(), done=1)

    def test_update_progress_advances_and_reports(self):
        bar = _FakeBar()
        rx.update_progress(bar, ok=3, fail=1, cost=0.0123, t0=time.time() - 10, done=4)
        self.assertEqual(bar.advanced, 1)
        self.assertIn("ok=3", bar.postfix)
        self.assertIn("fail=1", bar.postfix)
        self.assertIn("$0.0123", bar.postfix)
        self.assertNotIn("last=", bar.postfix)

    def test_close_progress_closes_and_ignores_none(self):
        bar = _FakeBar()
        rx.close_progress(bar)
        self.assertTrue(bar.closed)
        rx.close_progress(None)  # must not raise


if __name__ == "__main__":
    unittest.main()
