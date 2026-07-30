"""Tests for the FORGE-CI command-line interface."""

from pathlib import Path

from typer.testing import CliRunner

from forge_ci.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "FORGE-CI 0.1.0" in result.stdout


def test_validate_command(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"

    config_path.write_text(
        "name: smoke-test\nseed: 7\nepisodes: 3\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["validate", str(config_path)],
    )

    assert result.exit_code == 0
    assert "Configuration valid" in result.stdout
    assert "Episodes: 3" in result.stdout


def test_evaluate_passes_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pass.yaml"

    config_path.write_text(
        "name: pass-test\nepisodes: 3\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert "Gate: PASS" in result.stdout

    summaries = list(
        (tmp_path / "runs").glob("*/summary.json")
    )

    assert len(summaries) == 1


def test_evaluate_returns_two_when_gate_fails(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fail.yaml"

    config_path.write_text(
        """name: fail-test
episodes: 2
environment:
  goal: 5
  max_steps: 2
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(config_path),
            "-o",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 2
    assert "Gate: FAIL" in result.stdout
