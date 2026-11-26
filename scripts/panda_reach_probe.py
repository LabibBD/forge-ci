"""Run or visualize the Panda reaching simulator used by FORGE-CI."""

import argparse
import json
from pathlib import Path

from forge_ci.simulators.panda_reach import run_panda_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feedback-gain", type=float, default=0.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    args = parser.parse_args()
    result = run_panda_episode(
        args.model,
        seed=args.seed,
        view=args.view,
        cartesian_feedback_gain=args.feedback_gain,
        command_bias_rad=args.bias,
        command_delay_steps=args.delay,
        command_noise_std_rad=args.noise,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
