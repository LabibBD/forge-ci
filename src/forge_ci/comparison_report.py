"""Safe standalone HTML reporting for baseline-versus-candidate regression comparisons."""

import html
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from forge_ci.models import ComparisonResult


class ComparisonReportError(ValueError):
    """Raised if a comparison report cannot be generated."""


@dataclass(frozen=True)
class ComparisonReport:
    report_path: Path
    gate_passed: bool


def render_comparison_report(
    candidate_dir: Path,
    output_path: Path | None = None,
) -> ComparisonReport:
    """Render an existing comparison.json without rerunning an experiment."""
    input_path = candidate_dir / "comparison.json"
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ComparisonReportError(f"Comparison artifact not found: {input_path}") from None
    except (OSError, JSONDecodeError) as exc:
        raise ComparisonReportError(f"Unable to read {input_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ComparisonReportError("Comparison artifact must contain a JSON object.")

    try:
        result = ComparisonResult.model_validate(payload)
    except ValidationError as exc:
        raise ComparisonReportError(f"Invalid comparison artifact:\n{exc}") from exc

    output_path = (
        output_path if output_path is not None else candidate_dir / "comparison_report.html"
    )
    if output_path.resolve() == input_path.resolve():
        raise ComparisonReportError("HTML output cannot overwrite comparison.json.")

    baseline = html.escape(result.baseline_run_id, quote=True)
    candidate = html.escape(result.candidate_run_id, quote=True)
    verdict = "PASS" if result.gate_passed else "REGRESSION DETECTED"
    verdict_class = "pass" if result.gate_passed else "fail"
    reasons = (
        "".join(f"<li>{html.escape(reason, quote=True)}</li>" for reason in result.reasons)
        if result.reasons
        else "<li>No configured regression threshold was exceeded.</li>"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
  <title>FORGE-CI | Regression Comparison</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ background: #0b1220; color: #edf2ff; margin: 0; line-height: 1.5; }}
    main {{ max-width: 960px; padding: 52px 24px; margin: auto; }}
    .eyebrow {{ color: #62d9c7; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }}
    h1 {{ font-size: clamp(2rem, 5vw, 3.4rem); margin: 8px 0 14px; }}
    .muted {{ color: #acbbd6; }}
    .runs {{ overflow-wrap: anywhere; }}
    .badge {{ display: inline-block; border-radius: 999px; font-weight: 800; padding: 10px 18px; }}
    .pass {{ background: #143e36; color: #8bf3cb; }}
    .fail {{ background: #54252b; color: #ffc0c2; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 28px 0; }}
    .card {{ background: #131f34; border: 1px solid #293a58; border-radius: 18px; padding: 22px; }}
    .label {{ color: #aebcda; font-size: .87rem; }}
    .value {{ display: block; font-size: 1.75rem; margin: 6px 0; font-weight: 800; }}
    ul {{ margin: 0; padding-left: 22px; }}
    footer {{ color: #acbbd6; font-size: .85rem; margin-top: 30px; }}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">FORGE-CI • Policy evaluation</div>
  <h1>Regression comparison</h1>
  <p class="runs muted">Baseline: <strong>{baseline}</strong><br>
    Candidate: <strong>{candidate}</strong></p>
  <p><span class="badge {verdict_class}">{verdict}</span></p>
  <section class="grid" aria-label="Comparison metrics">
    <div class="card"><div class="label">Success rate, baseline → candidate</div>
      <strong class="value">{result.baseline_success_rate:.1%} → {result.candidate_success_rate:.1%}</strong>
      <div class="muted">Delta: {result.success_rate_delta:+.1%}; allowed drop: {result.max_success_rate_drop:.1%}</div></div>
    <div class="card"><div class="label">Mean steps, baseline → candidate</div>
      <strong class="value">{result.baseline_mean_steps:.2f} → {result.candidate_mean_steps:.2f}</strong>
      <div class="muted">Delta: {result.mean_steps_delta:+.2f}; allowed increase: {result.max_mean_steps_increase:.2f}</div></div>
  </section>
  <section class="card"><h2>Gate decision</h2><ul>{reasons}</ul></section>
  <footer>Generated from validated comparison.json. Static report, no external assets.</footer>
</main>
</body>
</html>
"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
    except OSError as exc:
        raise ComparisonReportError(f"Unable to write report {output_path}: {exc}") from exc

    return ComparisonReport(report_path=output_path, gate_passed=result.gate_passed)
