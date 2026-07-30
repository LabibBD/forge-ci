#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-runs/release-demo}"

if [[ -z "$output_root" || "$output_root" == "/" ]]; then
  echo "Unsafe output directory: '$output_root'" >&2
  exit 1
fi

if ! command -v forgeci >/dev/null 2>&1; then
  echo "forgeci is unavailable. Activate .venv first." >&2
  exit 1
fi

echo "==> Running repository checks"
./scripts/check.sh

echo
echo "==> Discovering multidimensional robustness envelope"
rm -rf "$output_root"

forgeci discover-envelope \
  configs/mujoco_multidimensional_envelope.yaml \
  --output-dir "$output_root"

envelope_run=$(
  find "$output_root" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d |
  sort |
  tail -n 1
)

if [[ -z "$envelope_run" ]]; then
  echo "No envelope run was generated." >&2
  exit 1
fi

echo
echo "==> Rendering robustness report"
forgeci render-report "$envelope_run"

report_path="$envelope_run/robustness_report.html"
data_path="$envelope_run/report_data.json"

test -s "$report_path"
test -s "$data_path"

echo
echo "FORGE-CI demo completed successfully."
echo "Envelope: $envelope_run"
echo "HTML report: $report_path"
echo "Report data: $data_path"
