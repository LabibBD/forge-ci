"""Matched-seed Panda command-disturbance sensitivity study.

Each condition uses the standard FORGE-CI runner and comparison gate.
The sweep is discrete, not a continuous or certified robustness boundary.
"""

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from statistics import fmean
from tempfile import mkdtemp

import mujoco
import numpy as np

from forge_ci.comparison import compare_runs
from forge_ci.config import ExperimentConfig, load_config
from forge_ci.runner import run_evaluation

LEVELS = {
    "bias": ("command_bias_rad", [0.005, 0.01, 0.02, 0.04, 0.08, 0.12]),
    "delay": ("command_delay_steps", [25, 50, 75, 100, 150, 250]),
    "noise": ("command_noise_std_rad", [0.002, 0.005, 0.01, 0.015, 0.02, 0.03]),
}


def build_conditions(base: ExperimentConfig):
    """Keep the task, target seeds, and gate fixed; change one command fault."""
    payload = base.model_dump(mode="python")
    yield "nominal", 0, ExperimentConfig.model_validate(payload)
    for axis, (parameter, values) in LEVELS.items():
        for value in values:
            case = base.model_dump(mode="python")
            case["name"] = f"panda-{axis}-{value}"
            case["policy"][parameter] = value
            yield axis, value, ExperimentConfig.model_validate(case)


def read_episodes(path: Path):
    return [
        json.loads(line)
        for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def targets(rows):
    return {
        row["seed"]: tuple(row["diagnostics"][f"target_{a}_m"] for a in "xyz")
        for row in rows
    }


def summarize(axis, value, run, rows, comparison=None):
    settled = [
        row["diagnostics"]["settling_time_s"]
        for row in rows
        if row["success"] and "settling_time_s" in row["diagnostics"]
    ]
    errors = [row["diagnostics"]["final_position_error_m"] * 1000 for row in rows]
    return {
        "axis": axis,
        "level": value,
        "episodes": len(rows),
        "successes": run.summary.successes,
        "success_rate": run.summary.success_rate,
        "mean_final_error_mm": fmean(errors),
        "max_final_error_mm": max(errors),
        "settled_episodes": len(settled),
        "mean_settling_s_among_successes": fmean(settled) if settled else None,
        "mean_completion_steps": run.summary.mean_steps,
        "absolute_gate_passed": run.summary.gate_passed,
        "regression_gate_passed": comparison.gate_passed if comparison else None,
        "run_directory": run.run_dir.name,
        "regression_reasons": comparison.reasons if comparison else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=Path("runs/panda-sensitivity"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")

    base = load_config(Path("configs/panda_nominal.yaml"))
    base.episodes = args.episodes
    base.seed = args.seed
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="study-", dir=args.output_root))
    provenance = {
        "forge_ci_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_revision": base.environment.model_revision,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "episodes_per_condition": args.episodes,
        "base_config": base.model_dump(mode="json"),
        "levels": LEVELS,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Evidence directory: {root}", flush=True)
    results = []
    baseline = None
    baseline_targets = None
    for axis, value, config in build_conditions(base):
        run = run_evaluation(config, root / "evaluations")
        rows = read_episodes(run.run_dir)
        if axis == "nominal":
            baseline = run
            baseline_targets = targets(rows)
            comparison = None
        else:
            if targets(rows) != baseline_targets:
                raise RuntimeError("Matched-seed target mismatch")
            comparison = compare_runs(baseline.run_dir, run.run_dir)
        entry = summarize(axis, value, run, rows, comparison)
        results.append(entry)
        print(
            f"{axis:8} {value:7g} | "
            f"{entry['successes']}/{entry['episodes']} success | "
            f"{entry['mean_final_error_mm']:.3f} mm | "
            f"{entry['mean_completion_steps']:.1f} steps | "
            f"relative gate: {entry['regression_gate_passed']}",
            flush=True,
        )

    boundaries = {}
    for axis in LEVELS:
        cases = [r for r in results if r["axis"] == axis]
        first_failure = next(
            (r for r in cases if not r["regression_gate_passed"]), None
        )
        boundaries[axis] = {
            "first_tested_relative_gate_failure": (
                first_failure["level"] if first_failure else None
            ),
            "tested_levels": [r["level"] for r in cases],
            "gate_pattern": [r["regression_gate_passed"] for r in cases],
        }
    summary = {
        "conditions": len(results),
        "total_episodes": sum(r["episodes"] for r in results),
        "nominal_absolute_gate_passed": results[0]["absolute_gate_passed"],
        "discrete_boundaries": boundaries,
        "results": results,
    }
    (root / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "study_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Evidence directory: {root}", flush=True)
    return 0 if summary["nominal_absolute_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
