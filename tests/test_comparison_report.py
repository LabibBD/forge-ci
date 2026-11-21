"""Regression report tests."""

import json
from pathlib import Path

import pytest

from forge_ci.comparison_report import ComparisonReportError, render_comparison_report


def _payload(passed: bool) -> dict:
    return {
        "baseline_run_id": "baseline-123",
        "candidate_run_id": "candidate-456",
        "baseline_success_rate": 1.0,
        "candidate_success_rate": 1.0,
        "success_rate_delta": 0.0,
        "baseline_mean_steps": 5.0,
        "candidate_mean_steps": 5.0 if passed else 9.0,
        "mean_steps_delta": 0.0 if passed else 4.0,
        "max_success_rate_drop": 0.0,
        "max_mean_steps_increase": 1.0,
        "gate_passed": passed,
        "reasons": [] if passed else ["Mean-step increase 4.0 exceeds 1.0."],
    }


@pytest.mark.parametrize("passed", [True, False])
def test_renders_pass_and_fail(tmp_path: Path, passed: bool) -> None:
    (tmp_path / "comparison.json").write_text(json.dumps(_payload(passed)))
    result = render_comparison_report(tmp_path)
    document = result.report_path.read_text(encoding="utf-8")
    assert result.gate_passed is passed
    assert "Baseline: <strong>baseline-123</strong>" in document
    assert "5.00 →" in document
    assert ("REGRESSION DETECTED" in document) is not passed


def test_escapes_untrusted_text(tmp_path: Path) -> None:
    payload = _payload(False)
    payload["candidate_run_id"] = '<img src=x onerror="alert(1)">'
    payload["reasons"] = ["<script>alert(1)</script>"]
    (tmp_path / "comparison.json").write_text(json.dumps(payload))
    document = render_comparison_report(tmp_path).report_path.read_text()
    assert "&lt;img" in document
    assert "&lt;script&gt;" in document
    assert "<script>" not in document


def test_invalid_or_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ComparisonReportError, match="not found"):
        render_comparison_report(tmp_path)
    (tmp_path / "comparison.json").write_text("invalid")
    with pytest.raises(ComparisonReportError, match="Unable to read"):
        render_comparison_report(tmp_path)
    (tmp_path / "comparison.json").write_text("[]")
    with pytest.raises(ComparisonReportError, match="JSON object"):
        render_comparison_report(tmp_path)


def test_does_not_overwrite_json(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    content = json.dumps(_payload(True))
    path.write_text(content)
    with pytest.raises(ComparisonReportError, match="cannot overwrite"):
        render_comparison_report(tmp_path, output_path=path)
    assert path.read_text() == content
