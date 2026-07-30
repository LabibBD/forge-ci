"""Self-contained HTML reporting for robustness envelopes."""

import html
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from forge_ci.boundary_models import (
    BoundarySearchSummary,
)
from forge_ci.envelope_models import (
    EnvelopeDimensionResult,
    RobustnessEnvelopeSummary,
)
from forge_ci.failure_analysis import (
    FailureAnalysisSummary,
)


class EnvelopeReportError(ValueError):
    """Raised when envelope artifacts cannot be rendered."""


@dataclass(frozen=True)
class EnvelopeReport:
    """Artifacts produced by report generation."""

    report_path: Path
    data_path: Path


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON file and require an object root."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EnvelopeReportError(f"Required report artifact not found: {path}") from None
    except (OSError, JSONDecodeError) as exc:
        raise EnvelopeReportError(f"Could not read {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise EnvelopeReportError(f"Report artifact must contain a JSON object: {path}")

    return payload


def _relative_artifact_path(
    root: Path,
    path: Path,
) -> str:
    """Return a safe POSIX path relative to the envelope."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()

    if not resolved_path.is_relative_to(resolved_root):
        raise EnvelopeReportError(f"Artifact escapes the envelope directory: {path}")

    return resolved_path.relative_to(resolved_root).as_posix()


def _format_number(
    value: float | int,
) -> str:
    """Format report values without unnecessary zeroes."""

    if isinstance(value, int):
        return str(value)

    return f"{value:.6f}".rstrip("0").rstrip(".")


def _chart_svg(
    dimensions: list[EnvelopeDimensionResult],
) -> str:
    """Create an inline SVG robustness-boundary chart."""

    label_width = 280
    chart_width = 680
    right_padding = 40
    row_height = 54

    width = label_width + chart_width + right_padding

    height = 45 + row_height * len(dimensions)

    rows: list[str] = []

    for index, dimension in enumerate(dimensions):
        y = 40 + index * row_height

        normalized = dimension.normalized_boundary_position

        filled_width = max(
            2.0,
            chart_width * normalized,
        )

        name = html.escape(dimension.name)
        percentage = f"{normalized:.1%}"

        rows.append(
            f"""
            <text
                x="0"
                y="{y + 19}"
                class="chart-label"
            >{name}</text>

            <rect
                x="{label_width}"
                y="{y}"
                width="{chart_width}"
                height="26"
                rx="6"
                class="chart-track"
            />

            <rect
                x="{label_width}"
                y="{y}"
                width="{filled_width:.2f}"
                height="26"
                rx="6"
                class="chart-fill"
            />

            <text
                x="{label_width + chart_width + 10}"
                y="{y + 19}"
                class="chart-value"
            >{percentage}</text>
            """
        )

    return f"""
    <svg
        class="robustness-chart"
        viewBox="0 0 {width} {height}"
        role="img"
        aria-label="Normalized robustness boundaries"
    >
        {"".join(rows)}
    </svg>
    """


def _dimension_card(
    dimension: EnvelopeDimensionResult,
    *,
    failure: FailureAnalysisSummary,
    counterexample_path: str,
    failure_analysis_path: str,
    clusters_path: str,
) -> str:
    """Render one detailed envelope dimension."""

    clusters = failure.clusters

    cluster_html = ""

    if clusters:
        cluster = clusters[0]

        cluster_html = f"""
        <div class="diagnosis-grid">
            <div>
                <span class="field-label">Classification</span>
                <strong>{html.escape(cluster.failure_type)}</strong>
            </div>
            <div>
                <span class="field-label">Confidence</span>
                <strong>{cluster.mean_confidence:.1%}</strong>
            </div>
            <div>
                <span class="field-label">Severity</span>
                <strong>{cluster.mean_severity:.3f}</strong>
            </div>
            <div>
                <span class="field-label">Failed episodes</span>
                <strong>{failure.failed_episodes}/{failure.total_episodes}</strong>
            </div>
        </div>
        """

    evidence_items: list[str] = []

    if failure.failures:
        for evidence in failure.failures[0].evidence:
            evidence_items.append(f"<li>{html.escape(evidence)}</li>")

    evidence_html = ""

    if evidence_items:
        evidence_html = f"""
        <div class="evidence">
            <h4>Representative evidence</h4>
            <ul>
                {"".join(evidence_items)}
            </ul>
        </div>
        """

    return f"""
    <article class="dimension-card">
        <div class="dimension-heading">
            <div>
                <span class="rank-badge">
                    {dimension.normalized_boundary_position:.1%}
                </span>
                <h3>{html.escape(dimension.name)}</h3>
            </div>

            <span class="status-badge">
                {"CONVERGED" if dimension.converged else "INCOMPLETE"}
            </span>
        </div>

        <p class="dimension-subtitle">
            {html.escape(dimension.parameter)}
            ·
            {html.escape(dimension.direction)}
        </p>

        <div class="metric-grid">
            <div>
                <span class="field-label">Baseline</span>
                <strong>
                    {_format_number(dimension.baseline_parameter_value)}
                </strong>
            </div>

            <div>
                <span class="field-label">Largest passing magnitude</span>
                <strong>
                    {_format_number(dimension.largest_passing_magnitude)}
                </strong>
            </div>

            <div>
                <span class="field-label">Smallest failing magnitude</span>
                <strong>
                    {_format_number(dimension.smallest_failing_magnitude)}
                </strong>
            </div>

            <div>
                <span class="field-label">Applied pass → fail</span>
                <strong>
                    {_format_number(dimension.passing_applied_value)}
                    →
                    {_format_number(dimension.failing_applied_value)}
                </strong>
            </div>

            <div>
                <span class="field-label">Boundary width</span>
                <strong>
                    {_format_number(dimension.boundary_width)}
                </strong>
            </div>

            <div>
                <span class="field-label">Diagnosis</span>
                <strong>
                    {html.escape(dimension.dominant_failure_type or "unclassified")}
                </strong>
            </div>
        </div>

        {cluster_html}
        {evidence_html}

        <div class="artifact-links">
            <a href="{html.escape(counterexample_path)}">
                Counterexample YAML
            </a>

            <a href="{html.escape(failure_analysis_path)}">
                Failure analysis JSON
            </a>

            <a href="{html.escape(clusters_path)}">
                Failure clusters CSV
            </a>
        </div>
    </article>
    """


def render_envelope_report(
    envelope_dir: Path,
    output_path: Path | None = None,
) -> EnvelopeReport:
    """Render a self-contained HTML report for an envelope."""

    envelope_dir = envelope_dir.resolve()

    summary_payload = _load_json_object(envelope_dir / "envelope_summary.json")

    try:
        summary = RobustnessEnvelopeSummary.model_validate(summary_payload)
    except ValidationError as exc:
        raise EnvelopeReportError(f"Invalid envelope summary:\n{exc}") from exc

    dimension_payloads: list[dict[str, Any]] = []
    cards: list[str] = []

    for dimension in summary.dimensions:
        search_dir = envelope_dir / dimension.search_directory

        boundary_payload = _load_json_object(search_dir / "boundary_summary.json")

        try:
            boundary = BoundarySearchSummary.model_validate(boundary_payload)
        except ValidationError as exc:
            raise EnvelopeReportError(
                f"Invalid boundary summary for {dimension.name}:\n{exc}"
            ) from exc

        counterexample_path = search_dir / boundary.counterexample_config

        failure_run = search_dir / boundary.counterexample_run_directory

        failure_analysis_path = failure_run / "failure_analysis.json"

        failure_payload = _load_json_object(failure_analysis_path)

        try:
            failure = FailureAnalysisSummary.model_validate(failure_payload)
        except ValidationError as exc:
            raise EnvelopeReportError(
                f"Invalid failure analysis for {dimension.name}:\n{exc}"
            ) from exc

        clusters_path = failure_run / "failure_clusters.csv"

        for required_path in (
            counterexample_path,
            clusters_path,
        ):
            if not required_path.exists():
                raise EnvelopeReportError(f"Required report artifact not found: {required_path}")

        counterexample_link = _relative_artifact_path(
            envelope_dir,
            counterexample_path,
        )

        failure_analysis_link = _relative_artifact_path(
            envelope_dir,
            failure_analysis_path,
        )

        clusters_link = _relative_artifact_path(
            envelope_dir,
            clusters_path,
        )

        cards.append(
            _dimension_card(
                dimension,
                failure=failure,
                counterexample_path=counterexample_link,
                failure_analysis_path=(failure_analysis_link),
                clusters_path=clusters_link,
            )
        )

        dimension_payloads.append(
            {
                "summary": dimension.model_dump(mode="json"),
                "boundary": boundary.model_dump(mode="json"),
                "failure_analysis": failure.model_dump(mode="json"),
                "artifacts": {
                    "counterexample": (counterexample_link),
                    "failure_analysis": (failure_analysis_link),
                    "failure_clusters": (clusters_link),
                },
            }
        )

    report_payload = {
        "schema_version": 1,
        "envelope": summary.model_dump(mode="json"),
        "dimensions": dimension_payloads,
    }

    embedded_json = json.dumps(
        report_payload,
        indent=2,
        sort_keys=True,
    ).replace("</", "<\\/")

    output_path = (
        output_path.resolve()
        if output_path is not None
        else envelope_dir / "robustness_report.html"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = output_path.parent / "report_data.json"

    weakest = html.escape(summary.weakest_dimension)

    chart = _chart_svg(summary.dimensions)

    document = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >
    <title>
        FORGE-CI Robustness Report
    </title>

    <style>
        :root {{
            color-scheme: light dark;
            --background: #0b1020;
            --surface: #121a2f;
            --surface-raised: #18233d;
            --text: #edf2ff;
            --muted: #9faed0;
            --accent: #66d9c8;
            --accent-strong: #36b8a5;
            --border: #2a3858;
            --warning: #ffcc66;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background:
                radial-gradient(
                    circle at top right,
                    #1a3152,
                    transparent 36rem
                ),
                var(--background);
            color: var(--text);
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
            line-height: 1.55;
        }}

        main {{
            width: min(1180px, calc(100% - 40px));
            margin: 0 auto;
            padding: 52px 0 80px;
        }}

        h1,
        h2,
        h3,
        h4,
        p {{
            margin-top: 0;
        }}

        h1 {{
            max-width: 760px;
            margin-bottom: 14px;
            font-size: clamp(2.3rem, 6vw, 4.8rem);
            line-height: 1;
            letter-spacing: -0.055em;
        }}

        h2 {{
            margin-bottom: 22px;
            font-size: 1.65rem;
        }}

        h3 {{
            display: inline;
            margin-left: 10px;
            font-size: 1.25rem;
        }}

        a {{
            color: var(--accent);
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .eyebrow {{
            margin-bottom: 12px;
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }}

        .lede {{
            max-width: 760px;
            color: var(--muted);
            font-size: 1.08rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin: 34px 0;
        }}

        .summary-card,
        .chart-panel,
        .dimension-card {{
            border: 1px solid var(--border);
            border-radius: 18px;
            background:
                linear-gradient(
                    145deg,
                    rgba(255, 255, 255, 0.035),
                    transparent
                ),
                var(--surface);
            box-shadow:
                0 18px 45px rgba(0, 0, 0, 0.22);
        }}

        .summary-card {{
            padding: 22px;
        }}

        .summary-card span,
        .field-label {{
            display: block;
            margin-bottom: 5px;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }}

        .summary-card strong {{
            font-size: 1.35rem;
        }}

        .chart-panel {{
            margin-bottom: 36px;
            padding: 26px;
            overflow-x: auto;
        }}

        .robustness-chart {{
            min-width: 850px;
            width: 100%;
        }}

        .chart-label,
        .chart-value {{
            fill: var(--text);
            font-family: inherit;
            font-size: 14px;
        }}

        .chart-value {{
            fill: var(--muted);
            font-weight: 700;
        }}

        .chart-track {{
            fill: var(--surface-raised);
        }}

        .chart-fill {{
            fill: var(--accent-strong);
        }}

        .dimension-list {{
            display: grid;
            gap: 18px;
        }}

        .dimension-card {{
            padding: 26px;
        }}

        .dimension-heading {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
        }}

        .rank-badge,
        .status-badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.05em;
        }}

        .rank-badge {{
            padding: 6px 9px;
            background: var(--accent);
            color: #061815;
        }}

        .status-badge {{
            padding: 7px 11px;
            border: 1px solid var(--accent-strong);
            color: var(--accent);
        }}

        .dimension-subtitle {{
            margin: 10px 0 22px;
            color: var(--muted);
            font-family:
                ui-monospace,
                SFMono-Regular,
                Menlo,
                monospace;
        }}

        .metric-grid,
        .diagnosis-grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
        }}

        .metric-grid > div,
        .diagnosis-grid > div {{
            border-radius: 12px;
            background: var(--surface-raised);
            padding: 15px;
        }}

        .diagnosis-grid {{
            margin-top: 14px;
        }}

        .evidence {{
            margin-top: 20px;
            border-left: 3px solid var(--warning);
            padding: 4px 0 4px 18px;
        }}

        .evidence h4 {{
            margin-bottom: 6px;
        }}

        .evidence ul {{
            margin: 0;
            padding-left: 20px;
            color: var(--muted);
        }}

        .artifact-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin-top: 22px;
            padding-top: 18px;
            border-top: 1px solid var(--border);
            font-size: 0.9rem;
            font-weight: 700;
        }}

        footer {{
            margin-top: 42px;
            color: var(--muted);
            font-size: 0.85rem;
        }}

        @media (max-width: 640px) {{
            main {{
                width: min(100% - 24px, 1180px);
                padding-top: 34px;
            }}

            .dimension-heading {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}
    </style>
</head>

<body>
<main>
    <header>
        <p class="eyebrow">
            FORGE-CI · Failure Intelligence
        </p>

        <h1>
            Robot Robustness Envelope
        </h1>

        <p class="lede">
            Deterministic failure-boundary discovery across
            controller and environment dimensions. Smaller
            normalized boundaries indicate weaker robustness.
        </p>
    </header>

    <section class="summary-grid">
        <div class="summary-card">
            <span>Envelope</span>
            <strong>
                {html.escape(summary.envelope_name)}
            </strong>
        </div>

        <div class="summary-card">
            <span>Weakest dimension</span>
            <strong>{weakest}</strong>
        </div>

        <div class="summary-card">
            <span>Dimensions analyzed</span>
            <strong>{len(summary.dimensions)}</strong>
        </div>

        <div class="summary-card">
            <span>Search status</span>
            <strong>
                {"All converged" if summary.all_converged else "Incomplete"}
            </strong>
        </div>
    </section>

    <section class="chart-panel">
        <h2>Normalized failure boundaries</h2>
        {chart}
    </section>

    <section>
        <h2>Ranked robustness dimensions</h2>

        <div class="dimension-list">
            {"".join(cards)}
        </div>
    </section>

    <footer>
        Generated by FORGE-CI. Report data is embedded below
        and also written to report_data.json.
    </footer>
</main>

<script
    id="forge-report-data"
    type="application/json"
>
{embedded_json}
</script>
</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )

    data_path.write_text(
        json.dumps(
            report_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return EnvelopeReport(
        report_path=output_path,
        data_path=data_path,
    )
