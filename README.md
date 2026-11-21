# FORGE-CI

**Continuous evaluation, regression gating, and failure intelligence for robot-learning systems.**

FORGE-CI is a Python CLI and CI toolkit for evaluating robot controllers in deterministic test environments. It records structured episode results, compares candidate policies against baselines, detects performance regressions, sweeps simulator disturbances, discovers failure boundaries, and generates standalone HTML diagnostics.

The project includes a fast LineWorld test environment and a MuJoCo position-servo reaching task. It is a research/portfolio prototype, **not** a claim of production deployment or validation on physical robots.

## What it does

| Capability | Function |
| --- | --- |
| Reproducible evaluation | Seeded episode execution, YAML validation, machine-readable artifacts |
| Regression gates | Compare success rates and mean episode lengths; exit code 2 for regression |
| Disturbance sweeps | Check robustness under multiple simulated perturbations |
| Failure diagnosis | Categorize failed episodes and cluster failure evidence |
| Boundary search | Find the smallest tested disturbance that causes a policy to fail |
| Robustness envelopes | Search and rank failure boundaries across six disturbance dimensions |
| HTML reporting | Generate standalone robustness and regression comparison reports |
| Automation | Run pytest, Ruff, integration evaluations and artifact assertions in GitHub Actions |

## Get started

Requires Python **3.12+**. MuJoCo examples use the optional `sim` dependency.

```bash
git clone https://github.com/LabibBD/forge-ci.git
cd forge-ci
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,sim]"
./scripts/check.sh
```

### 1. Run an evaluation

```bash
forgeci validate configs/smoke.yaml
forgeci evaluate configs/smoke.yaml --output-dir runs/demo-baseline
```

FORGE-CI creates a timestamped run directory with a manifest, episode results and an aggregate summary.

### 2. Reproduce a regression

```bash
./scripts/e2e.sh
```

The script runs a known-good baseline, checks an identical candidate, then evaluates a deliberately slower candidate. It **expects** the comparison to fail with exit code 2, proving that the CI gate catches a regression rather than merely producing successful runs.

To generate the HTML comparison report from a saved candidate directory:

```bash
forgeci render-comparison runs/my-candidate-run
# Writes runs/my-candidate-run/comparison_report.html
```

The candidate directory must contain a `comparison.json` produced by `forgeci compare BASELINE_DIR CANDIDATE_DIR`. The report can be opened locally, without a web server or external assets.

### 3. Analyze simulated controller robustness

```bash
forgeci sweep configs/mujoco_robustness_sweep.yaml --output-dir runs/sweeps
forgeci discover-boundary configs/mujoco_bias_boundary.yaml --output-dir runs/boundaries
forgeci discover-envelope configs/mujoco_multidimensional_envelope.yaml --output-dir runs/envelopes
```

The multidimensional envelope explores **positive/negative target bias, controller gain loss, actuator delay, control noise, and joint damping**. It generates a ranked `envelope_summary.json`, `robustness_envelope.csv`, and individual counterexamples.

To render the existing robustness dashboard:

```bash
forgeci render-report PATH_TO_ENVELOPE_RUN
# Writes robustness_report.html and report_data.json
```

## CI and verification

The workflow in `.github/workflows/ci.yml` runs automated checks, end-to-end regression detection, MuJoCo runs, failure classification, boundary discovery, six-dimensional robustness envelope verification, and HTML report assertions.

```bash
./scripts/check.sh
./scripts/e2e.sh
```

Exit-code convention: **0 = passed**, **1 = invalid input or runtime error**, **2 = policy/evaluation regression gate failed**. Deliberately failing policy evaluations are verified as expected outcomes in integration tests.

The regression HTML report is produced from a validated `comparison.json`, escapes untrusted text, applies a restrictive content security policy, and contains no external JavaScript or third-party assets.

## Repository map

- `src/forge_ci/cli.py`: command-line entrypoints
- `src/forge_ci/runner.py`: deterministic evaluation
- `src/forge_ci/comparison.py`: baseline/candidate regression checks
- `src/forge_ci/comparison_report.py`: standalone comparison reporting
- `src/forge_ci/failure_analysis.py`: episode failure classification
- `src/forge_ci/boundary_search.py`: failure threshold discovery
- `src/forge_ci/envelope_search.py`: multidimensional robustness envelope
- `src/forge_ci/reporting.py`: HTML robustness visualization
- `src/forge_ci/simulators/`: LineWorld/MuJoCo integration
- `tests/`: automated checks, including report generation
- `configs/`: reproducible scenario definitions

## Limitations

- The included controller demonstrations are synthetic, and reported failure boundaries apply to these configured simulations.
- This does not certify safety or generalize to physical deployments or arbitrary learned policies.
- Determinism depends on the simulator, machine and software versions; this is not a guarantee of identical results across all hardware.
- The comparison gate currently emphasizes success rate and mean step count; statistical confidence estimation and large-scale distributed evaluation are out of scope.

**Maintainer:** Quazi Mostahid Labib | Tsinghua University
