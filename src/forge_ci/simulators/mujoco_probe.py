"""Minimal deterministic MuJoCo physics probe."""

from dataclasses import dataclass

import mujoco
import numpy as np

MODEL_XML = """
<mujoco model="forge_ci_slider">
  <option timestep="0.005" gravity="0 0 0"/>

  <worldbody>
    <geom
      name="track"
      type="box"
      pos="0.5 0 -0.06"
      size="0.55 0.08 0.02"
      rgba="0.4 0.4 0.4 1"
      contype="0"
      conaffinity="0"
    />

    <body name="slider" pos="0 0 0">
      <joint
        name="slide"
        type="slide"
        axis="1 0 0"
        limited="true"
        range="0 1"
        damping="4"
      />

      <geom
        name="agent"
        type="sphere"
        size="0.05"
        mass="1"
        rgba="0.2 0.4 1 1"
      />
    </body>

    <site
      name="target"
      type="sphere"
      pos="0.8 0 0"
      size="0.03"
      rgba="0.2 1 0.2 1"
    />
  </worldbody>

  <actuator>
    <position
      name="slider_position"
      joint="slide"
      kp="40"
      ctrllimited="true"
      ctrlrange="0 1"
    />
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class ProbeResult:
    """Result of one deterministic MuJoCo simulator probe."""

    seed: int
    success: bool
    steps: int
    initial_position: float
    final_position: float
    final_velocity: float
    target_position: float


def run_probe(seed: int = 42) -> ProbeResult:
    """Run a deterministic position-control episode in MuJoCo."""

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    random_generator = np.random.default_rng(seed)

    initial_position = float(
        random_generator.uniform(0.0, 0.1)
    )

    target_position = 0.8

    data.qpos[0] = initial_position
    data.qvel[0] = 0.0

    mujoco.mj_forward(model, data)

    success = False
    completed_steps = 0

    for _ in range(1000):
        completed_steps += 1
        data.ctrl[0] = target_position

        mujoco.mj_step(model, data)

        position_error = abs(
            float(data.qpos[0]) - target_position
        )

        velocity = abs(float(data.qvel[0]))

        if position_error <= 0.01 and velocity <= 0.02:
            success = True
            break

    return ProbeResult(
        seed=seed,
        success=success,
        steps=completed_steps,
        initial_position=initial_position,
        final_position=float(data.qpos[0]),
        final_velocity=float(data.qvel[0]),
        target_position=target_position,
    )
