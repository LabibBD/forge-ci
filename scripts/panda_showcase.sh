#!/usr/bin/env bash
# Compare one nominal reaching policy against three command-channel fault scenarios.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_root="${1:-runs/panda-showcase}"
mkdir -p "$output_root"
root="$(mktemp -d "$output_root/panda-XXXXXXXX")"
echo "Panda experiment directory: $root"

run_case() {
    local scenario="$1"
    local status=0
    forgeci evaluate "configs/panda_${scenario}.yaml" --output-dir "$root/$scenario" || status=$?
    if [[ "$status" != 0 && "$status" != 2 ]]; then
        return "$status"
    fi
    if [[ "$scenario" == nominal && "$status" != 0 ]]; then
        echo "Nominal reaching failed; stop before interpreting disturbances." >&2
        return 1
    fi
}
run_case nominal
baseline="$(find "$root/nominal" -mindepth 1 -maxdepth 1 -type d -print -quit)"
test -n "$baseline"

for scenario in bias delay noise; do
    run_case "$scenario"
    candidate="$(find "$root/$scenario" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    test -n "$candidate"
    status=0
    forgeci compare "$baseline" "$candidate" || status=$?
    if [[ "$status" != 0 && "$status" != 2 ]]; then
        exit "$status"
    fi
    status=0
    forgeci render-comparison "$candidate" || status=$?
    if [[ "$status" != 0 && "$status" != 2 ]]; then
        exit "$status"
    fi
done

python - "$root" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
summary = {}
for scenario in ("nominal", "bias", "delay", "noise"):
    run = next((root / scenario).iterdir())
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    errors = [r["diagnostics"]["final_position_error_m"] for r in rows]
    settled = [r["diagnostics"]["settling_time_s"] for r in rows
               if "settling_time_s" in r["diagnostics"]]
    info = json.loads((run / "summary.json").read_text())
    summary[scenario] = {
        "episodes": len(rows),
        "successes": info["successes"],
        "mean_completion_steps": info["mean_steps"],
        "mean_final_error_mm": 1000 * mean(errors),
        "max_final_error_mm": 1000 * max(errors),
        "settled_episodes": len(settled),
        "mean_settling_seconds_among_settled": mean(settled) if settled else None,
        "run_directory": str(run),
    }
    if scenario != "nominal":
        comparison = json.loads((run / "comparison.json").read_text())
        summary[scenario]["regression_gate_passed"] = comparison["gate_passed"]
        summary[scenario]["regression_reasons"] = comparison["reasons"]
    else:
        baseline_targets = {r["seed"]: tuple(r["diagnostics"][f"target_{a}_m"]
                            for a in "xyz") for r in rows}
        continue
    candidate_targets = {r["seed"]: tuple(r["diagnostics"][f"target_{a}_m"]
                         for a in "xyz") for r in rows}
    assert candidate_targets == baseline_targets, "Target mismatch"
(root / "panda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"Evidence directory: {root}")
PY
