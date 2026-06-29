"""Tests for run directory manager."""
import pytest
from runtime.run_manager import create_run_dir


class TestCreateRunDir:
    def test_creates_run_001_first(self, tmp_path):
        run_dir = create_run_dir(tmp_path)
        assert run_dir.name == "Run_001"
        assert run_dir.is_dir()

    def test_increments_on_second_call(self, tmp_path):
        first = create_run_dir(tmp_path)
        second = create_run_dir(tmp_path)
        assert first.name == "Run_001"
        assert second.name == "Run_002"

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        run_dir = create_run_dir(nested)
        assert run_dir.is_dir()
        assert run_dir.name == "Run_001"

    def test_gap_in_numbering_uses_next(self, tmp_path):
        # Simulate Run_001 and Run_003 already existing
        (tmp_path / "Run_001").mkdir()
        (tmp_path / "Run_003").mkdir()
        run_dir = create_run_dir(tmp_path)
        assert run_dir.name == "Run_004"

    def test_does_not_overwrite_existing(self, tmp_path):
        first = create_run_dir(tmp_path)
        sentinel = first / "sentinel.txt"
        sentinel.write_text("do not delete")
        create_run_dir(tmp_path)  # Run_002
        assert sentinel.exists(), "Run_001 contents must not be touched"
