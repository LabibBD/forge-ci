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

## One-command showcase

This command creates a **real, reproducible regression demonstration** using two
seeded LineWorld policies. It saves both run artifacts, the comparison JSON,
and a self-contained HTML report (the intentionally slower policy is expected
to fail its relative regression gate).

```bash
bash scripts/showcase.sh
# Optionally choose the parent directory:
bash scripts/showcase.sh runs/my-demo
```

The command prints the complete path to `comparison_report.html`. Open that
HTML file in a browser for a screenshot or share the CI evaluation artifact
with reviewers. The CI workflow also runs this exact demo and uploads its
output under `runs/ci-showcase/`. No MuJoCo installation is required
for this specific showcase, although the full CI suite uses it.

**Interpretation:** the demonstration compares two simple deterministic
controllers and verifies a *performance regression*, not a real-world robot
failure. The dashboard reports success-rate and mean-step changes against
explicit thresholds.

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

## Franka Panda reaching experiment

The articulated position-only reaching adapter uses the external
[MuJoCo Menagerie Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/eea4c7431b63da8278b0ff31638e0b0e941cec0e/franka_emika_panda)
at a pinned revision. Model assets are not redistributed by FORGE-CI.
From the repository root, install the simulator extra and check out the model:

```bash
git clone --no-checkout --filter=blob:none https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout init --cone
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
git -C mujoco_menagerie checkout --detach eea4c7431b63da8278b0ff31638e0b0e941cec0e
python -m pip install -e ".[dev,sim]"
bash scripts/panda_showcase.sh
```

The four cases use the same five seeds and Cartesian targets, with nominal,
first-joint command bias, command delay, and command noise conditions.
The experiment observes each episode for six simulated seconds.
With `step_metric: settling_or_horizon`, successful episodes report the
first step of the final uninterrupted interval within position and velocity
tolerances after the commanded trajectory ends; failures report the full
observation horizon. The raw simulated step count is retained in diagnostics.
The comparison permits a 50-step mean increase (100 ms at the 2 ms timestep).
Legacy Panda configs without the metric field retain horizon semantics and
cannot be compared directly against the new configs.

This is a simulated position-only IK and joint-position control experiment,
not a learned policy, full-pose control, or hardware validation.
The existing slider-specific robustness boundary search is unchanged.

To view a nominal trial:

```bash
python scripts/panda_reach_probe.py --model mujoco_menagerie/franka_emika_panda/scene.xml --view
```

### Expanded Panda disturbance sensitivity study

After installing the pinned model as described above, run:

```bash
python scripts/panda_sensitivity.py --episodes 20 --seed 1000
# Use the unique evidence directory printed by the script:
python -m pip install matplotlib
python scripts/plot_panda_sensitivity.py runs/panda-sensitivity/STUDY_DIRECTORY
```

This evaluates 19 conditions (nominal and six levels each of joint-1 command
bias, command delay, and command noise) over 20 matched seeds per condition.
It saves standard FORGE-CI episode artifacts and comparisons, study JSON/CSV,
and environment provenance. The plotting script reads these measured results.
A reported first failing tested level is a **discrete sensitivity result**,
not a certified or continuous robustness boundary. Nonmonotonic outcomes
remain visible, and conclusions about relative regressions require a valid
nominal baseline. Only a local family of reaching targets is sampled;
the study does not characterize the entire Panda workspace.

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
