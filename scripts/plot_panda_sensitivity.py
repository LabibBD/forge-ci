"""Plot actual Panda sensitivity measurements from a completed study."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", type=Path)
    args = parser.parse_args()
    root = args.study_directory
    summary = json.loads((root / "study_summary.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    rows = summary["results"]
    nominal = next(row for row in rows if row["axis"] == "nominal")
    axis_labels = {
        "bias": "Joint-1 command bias (rad)",
        "delay": "Command delay (ms)",
        "noise": "Command noise std. dev. (rad)",
    }
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    metrics = [
        ("success_rate", "Success rate (%)", 100),
        ("mean_final_error_mm", "Mean final error (mm)", 1),
        ("mean_completion_steps", "Mean completion steps", 1),
    ]
    for col, (axis, label) in enumerate(axis_labels.items()):
        cases = [nominal] + [row for row in rows if row["axis"] == axis]
        # Delay levels are simulation steps, converted using actual timestep.
        if axis == "delay":
            run = root / "evaluations" / nominal["run_directory"]
            first = json.loads((run / "episodes.jsonl").read_text().splitlines()[0])
            factor = first["diagnostics"]["timestep_s"] * 1000
        else:
            factor = 1
        xs = [row["level"] * factor for row in cases]
        for row_index, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row_index, col]
            ys = [row[metric] * scale for row in cases]
            ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=4)
            failed = [
                (x, y)
                for x, y, case in zip(xs, ys, cases, strict=True)
                if case["regression_gate_passed"] is False
            ]
            if failed:
                ax.scatter(*zip(*failed, strict=True), marker="x", s=45,
                           label="Relative gate failed")
            ax.set_ylabel(ylabel)
            ax.set_xlabel(label)
            ax.grid(alpha=0.2)
            if metric == "success_rate":
                ax.set_ylim(-5, 105)
            if metric == "mean_final_error_mm":
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.set_ylim(0, max(12.0, max(ys) * 1.12))
            if metric == "mean_completion_steps":
                limit = provenance["base_config"]["gate"]["max_mean_steps_increase"]
                ax.axhline(nominal[metric] + limit, linestyle="--", linewidth=0.8,
                           label="Relative step limit")
            if failed or metric == "mean_completion_steps":
                ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        f"FORGE-CI | Franka Panda command sensitivity\n"
        f"{summary['conditions']} conditions, {summary['total_episodes']} simulated episodes; "
        f"{provenance['episodes_per_condition']} matched seeds per condition",
        fontsize=14,
    )
    output = root / "panda_sensitivity.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
