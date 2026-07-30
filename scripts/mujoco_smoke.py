"""Run the standalone FORGE-CI MuJoCo probe."""

from forge_ci.simulators.mujoco_probe import run_probe


def main() -> None:
    """Execute and display one deterministic probe."""

    result = run_probe(seed=42)

    print(f"Seed: {result.seed}")
    print(f"Initial position: {result.initial_position:.6f}")
    print(f"Target position: {result.target_position:.6f}")
    print(f"Final position: {result.final_position:.6f}")
    print(f"Final velocity: {result.final_velocity:.6f}")
    print(f"Steps: {result.steps}")
    print(f"Result: {'PASS' if result.success else 'FAIL'}")

    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
