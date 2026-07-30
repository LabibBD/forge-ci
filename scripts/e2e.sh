#!/usr/bin/env bash
set -euo pipefail

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "Running baseline evaluation..."

forgeci evaluate \
  configs/smoke.yaml \
  --output-dir "$workdir/baseline"

baseline_run=$(
  find "$workdir/baseline" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d |
  sort |
  tail -n 1
)

echo "Running identical candidate..."

forgeci evaluate \
  configs/smoke.yaml \
  --output-dir "$workdir/candidate-pass"

candidate_pass_run=$(
  find "$workdir/candidate-pass" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d |
  sort |
  tail -n 1
)

echo "Checking passing comparison..."

forgeci compare \
  "$baseline_run" \
  "$candidate_pass_run"

echo "Running deliberately slower candidate..."

forgeci evaluate \
  configs/alternating_candidate.yaml \
  --output-dir "$workdir/candidate-fail"

candidate_fail_run=$(
  find "$workdir/candidate-fail" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d |
  sort |
  tail -n 1
)

echo "Checking regression detection..."

set +e

forgeci compare \
  "$baseline_run" \
  "$candidate_fail_run"

comparison_exit_code=$?

set -e

if [[ "$comparison_exit_code" -ne 2 ]]; then
  echo "Expected regression exit code 2."
  echo "Received exit code: $comparison_exit_code"
  exit 1
fi

if [[ ! -f "$candidate_fail_run/comparison.json" ]]; then
  echo "Expected comparison.json was not generated."
  exit 1
fi

echo "End-to-end regression detection passed."
