"""Compare nominal IK tracking with bounded Cartesian feedback under matched faults."""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from statistics import fmean
from tempfile import mkdtemp

import mujoco

from forge_ci.comparison import compare_runs
from forge_ci.config import ExperimentConfig, load_config
from forge_ci.runner import run_evaluation

BIAS_LEVELS = [0.0, 0.02, 0.04, 0.08, 0.12]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--gain", type=float, default=0.12)
    parser.add_argument("--output-root", type=Path, default=Path("runs/panda-feedback"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or not 0 < args.gain <= 0.3:
        parser.error("Invalid episodes, seed or feedback gain")
    base = load_config(Path("configs/panda_nominal.yaml"))
    base.seed = args.seed
    base.episodes = args.episodes
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="study-", dir=args.output_root))
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": base.environment.model_revision,
        "mujoco_version": mujoco.__version__,
        "python_version": platform.python_version(),
        "seed": args.seed,
        "episodes_per_condition": args.episodes,
        "gain": args.gain,
        "bias_levels_rad": BIAS_LEVELS,
        "base_config": base.model_dump(mode="json"),
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Evidence directory: {root}", flush=True)
    results = []
    for bias in BIAS_LEVELS:
        baseline = None
        baseline_targets = None
        for label, gain in [("original", 0.0), ("feedback", args.gain)]:
            payload = base.model_dump(mode="python")
            payload["name"] = f"panda-{label}-bias-{bias}"
            payload["policy"]["command_bias_rad"] = bias
            payload["policy"]["cartesian_feedback_gain"] = gain
            config = ExperimentConfig.model_validate(payload)
            run = run_evaluation(config, root / "evaluations")
            episodes = [
                json.loads(line)
                for line in (run.run_dir / "episodes.jsonl").read_text().splitlines()
            ]
            targets = {
                row["seed"]: tuple(row["diagnostics"][f"target_{a}_m"] for a in "xyz")
                for row in episodes
            }
            errors = [row["diagnostics"]["final_position_error_m"] * 1000 for row in episodes]
            settled = [
                row["diagnostics"]["settling_time_s"]
                for row in episodes
                if row["success"] and "settling_time_s" in row["diagnostics"]
            ]
            comparison = None
            if label == "original":
                baseline = run
                baseline_targets = targets
            else:
                if targets != baseline_targets:
                    raise RuntimeError("Targets differ between controllers")
                comparison = compare_runs(baseline.run_dir, run.run_dir)
            entry = {
                "controller": label,
                "bias_rad": bias,
                "episodes": len(episodes),
                "successes": run.summary.successes,
                "success_rate": run.summary.success_rate,
                "mean_final_error_mm": fmean(errors),
                "max_final_error_mm": max(errors),
                "mean_settling_s_among_successes": fmean(settled) if settled else None,
                "mean_completion_steps": run.summary.mean_steps,
                "absolute_gate_passed": run.summary.gate_passed,
                "relative_gate_passed": comparison.gate_passed if comparison else None,
                "relative_gate_reasons": comparison.reasons if comparison else [],
                "run_directory": run.run_dir.name,
            }
            results.append(entry)
            print(
                f"bias {bias:.3f} | {label:8} | "
                f"{entry['successes']}/{entry['episodes']} success | "
                f"{entry['mean_final_error_mm']:.3f} mm | "
                f"{entry['mean_completion_steps']:.1f} steps | "
                f"relative gate: {entry['relative_gate_passed']}",
                flush=True,
            )
    summary = {"total_episodes": sum(row["episodes"] for row in results), "results": results}
    (root / "feedback_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Evidence directory: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
