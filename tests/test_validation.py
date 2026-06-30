"""Tests for ValidationEngine and ValidationResult (no model required)."""
import pytest

from validation.engine import ValidationResult


class TestValidationResult:
    def test_pass_result(self):
        r = ValidationResult(
            passed=True,
            prompt="What is 2+2?",
            reference_text="2+2 is 4.",
            gamr_text="2+2 is 4.",
            reference_tokens=[1, 2, 3],
            gamr_tokens=[1, 2, 3],
        )
        assert r.passed is True
        assert r.mismatch_index is None

    def test_fail_result(self):
        r = ValidationResult(
            passed=False,
            prompt="What is 2+2?",
            reference_text="2+2 is 4.",
            gamr_text="2+2 is 5.",
            reference_tokens=[1, 2, 3],
            gamr_tokens=[1, 2, 99],
            mismatch_index=2,
        )
        assert r.passed is False
        assert r.mismatch_index == 2

    def test_summary_pass_contains_pass(self):
        r = ValidationResult(
            passed=True, prompt="p", reference_text="r", gamr_text="g",
        )
        assert "PASS" in r.summary()

    def test_summary_fail_contains_fail(self):
        r = ValidationResult(
            passed=False, prompt="p", reference_text="r", gamr_text="g",
            mismatch_index=0,
        )
        assert "FAIL" in r.summary()

    def test_summary_shows_prompt(self):
        r = ValidationResult(
            passed=True, prompt="hello world", reference_text="", gamr_text="",
        )
        assert "hello world" in r.summary()


class TestValidationReport:
    def test_write_report_creates_files(self, tmp_path):
        from validation.report import write_report
        r = ValidationResult(
            passed=True,
            prompt="test prompt",
            reference_text="ref output",
            gamr_text="ref output",
            reference_tokens=[1, 2, 3],
            gamr_tokens=[1, 2, 3],
            reference_time_s=1.5,
            gamr_time_s=2.1,
        )
        write_report(r, tmp_path)
        assert (tmp_path / "summary.md").exists()
        assert (tmp_path / "metrics.json").exists()

    def test_metrics_json_has_required_keys(self, tmp_path):
        import orjson
        from validation.report import write_report
        r = ValidationResult(
            passed=False, prompt="p", reference_text="r", gamr_text="g",
            mismatch_index=1, reference_time_s=1.0, gamr_time_s=1.0,
        )
        write_report(r, tmp_path)
        data = orjson.loads((tmp_path / "metrics.json").read_bytes())
        assert "validation_passed" in data
        assert "reference_time_s" in data
        assert "gamr_time_s" in data
