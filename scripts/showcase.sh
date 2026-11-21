#!/usr/bin/env bash
# Produce a self-contained regression-report demo from two deterministic evaluations.
set -euo pipefail

output_root="${1:-runs/showcase}"
mkdir -p "$output_root"
showcase_dir="$(mktemp -d "$output_root/forgeci-XXXXXXXX")"

echo "FORGE-CI showcase: $showcase_dir"
echo "Running deterministic baseline..."
forgeci evaluate configs/smoke.yaml --output-dir "$showcase_dir/baseline"

baseline_run="$(
  find "$showcase_dir/baseline" -mindepth 1 -maxdepth 1 -type d -print -quit
)"
if [[ -z "$baseline_run" ]]; then
  echo "No baseline run was produced." >&2
  exit 1
fi

echo "Running deliberately slower policy..."
forgeci evaluate configs/alternating_candidate.yaml --output-dir "$showcase_dir/candidate"

candidate_run="$(
  find "$showcase_dir/candidate" -mindepth 1 -maxdepth 1 -type d -print -quit
)"
if [[ -z "$candidate_run" ]]; then
  echo "No candidate run was produced." >&2
  exit 1
fi

echo "Verifying regression gate detects the slower policy..."
set +e
forgeci compare "$baseline_run" "$candidate_run"
comparison_status=$?
set -e

if [[ "$comparison_status" -ne 2 ]]; then
  echo "Expected comparison to exit 2, got $comparison_status" >&2
  exit 1
fi

echo "Rendering standalone comparison HTML..."
set +e
forgeci render-comparison "$candidate_run"
report_status=$?
set -e

if [[ "$report_status" -ne 2 ]]; then
  echo "Expected report to exit 2 for a regression, got $report_status" >&2
  exit 1
fi

test -s "$candidate_run/comparison.json"
test -s "$candidate_run/comparison_report.html"

# Check the regression is the intended slow-policy failure, not an unrelated failure.
python - "$candidate_run/comparison.json" <<'PYVERIFY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["gate_passed"] is False
assert report["success_rate_delta"] == 0
assert report["mean_steps_delta"] > report["max_mean_steps_increase"]
assert len(report["reasons"]) > 0
print("Verified regression caused by increased mean episode length.")
PYVERIFY

echo
echo "SHOWCASE READY"
echo "HTML report: $candidate_run/comparison_report.html"
echo "Structured results: $candidate_run/comparison.json"
echo "Baseline artifacts: $baseline_run"
echo "Candidate artifacts: $candidate_run"
