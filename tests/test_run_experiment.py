"""Unit tests for scripts/run_experiment.py. No network access."""

import importlib.util
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

    def test_output_path_for_default_model(self):
        expected = rx.ROOT / "results" / "inclusionai-ling-3.1-flash__think.json"
        self.assertEqual(rx.output_path("inclusionai/ling-3.1-flash"), expected)

    def test_questions_file_lives_under_repo_root(self):
        self.assertEqual(rx.QUESTIONS_FILE, rx.ROOT / "data" / "mmlu_pro_sample.json")


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

    def test_empty_persona_omits_system_message(self):
        payload = self._build(system_prompt="")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")


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
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "", "reasoning": "long thought..."},
            }],
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
                "fake-key", "inclusionai/ling-3.1-flash", self.Q, "",
                max_tokens=32768, temperature=0.0, seed=42,
                reasoning_mode="effort", reasoning_value="high", provider=None,
            )
        finally:
            rx.call_with_retries = original

        self.assertEqual(len(calls), 1)  # not retried
        self.assertEqual(entry["answer"], "?")
        self.assertFalse(entry["correct"])
        self.assertEqual(entry["finish_reason"], "length")
        self.assertIsNone(entry["error"])
        self.assertEqual(entry["provider"], "InclusionAI")


if __name__ == "__main__":
    unittest.main()
