"""Render the actual final MuJoCo states for two matched reaching trials."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from forge_ci.simulators.panda_reach import run_panda_episode


def render_state(model, data, target, tool):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.34, 0.05, 0.42]
    camera.distance = 1.45
    camera.azimuth = 135
    camera.elevation = -22
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 800)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 600)
    with mujoco.Renderer(model, height=600, width=800) as renderer:
        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        # Mark the actual Cartesian target and tool point in the simulated world.
        for pos, radius, rgba in (
            (target, 0.015, [0.1, 0.85, 0.2, 0.65]),
            (tool, 0.008, [0.95, 0.55, 0.1, 1.0]),
        ):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Rendering scene capacity exceeded")
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius),
                pos,
                np.eye(3).ravel(),
                np.array(rgba, dtype=np.float32),
            )
            scene.ngeom += 1
        return renderer.render().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--bias", type=float, default=0.12)
    parser.add_argument("--gain", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    frames = []
    results = []
    for gain in (0.0, args.gain):
        def capture(model, data, target, tool):
            frames.append(render_state(model, data, target, tool))

        result = run_panda_episode(
            args.model,
            seed=args.seed,
            command_bias_rad=args.bias,
            cartesian_feedback_gain=gain,
            final_state_callback=capture,
        )
        results.append(result)

    if results[0]["target_xyz_m"] != results[1]["target_xyz_m"]:
        raise RuntimeError("Matched trials must share the same target")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.80, bottom=0.14, wspace=0.04)
    for ax, image, result, label in zip(
        axes, frames, results, ("Original IK controller", "Cartesian feedback"), strict=True
    ):
        ax.imshow(image)
        ax.axis("off")
        error_mm = result["final_position_error_m"] * 1000
        verdict = "PASS" if result["passed"] else "FAIL"
        ax.set_title(f"{label}\n{error_mm:.3f} mm final error  |  {verdict}", fontsize=13)
    fig.suptitle("FORGE-CI | Franka Panda reaching", fontsize=18, weight="semibold", y=0.97)
    fig.text(
        0.5, 0.885,
        f"Matched target, seed {args.seed} | Joint-1 command bias: {args.bias:g} rad",
        ha="center", fontsize=11,
    )
    fig.text(
        0.5, 0.075,
        "Green: target   •   Orange: measured tool point   |   Final state at 6 s",
        ha="center", fontsize=10,
    )
    fig.text(
        0.5, 0.035,
        "MuJoCo Menagerie simulation | 10 mm position tolerance | github.com/LabibBD/forge-ci",
        ha="center", fontsize=9,
    )
    fig.savefig(args.output_dir / "panda_comparison.png", dpi=180)
    plt.close(fig)
    for label, frame in zip(("original", "feedback"), frames, strict=True):
        plt.imsave(args.output_dir / f"{label}.png", frame)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(args.output_dir / "panda_comparison.png")


if __name__ == "__main__":
    main()
