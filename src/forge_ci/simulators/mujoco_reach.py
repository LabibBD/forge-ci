"""Deterministic MuJoCo reaching environment."""

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

MODEL_XML_TEMPLATE = """
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
        damping="__JOINT_DAMPING__"
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
      kp="__CONTROLLER_KP__"
      ctrllimited="true"
      ctrlrange="0 1"
    />
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class MujocoReachResult:
    """Result of one MuJoCo reaching episode."""

    seed: int
    success: bool
    steps: int

    initial_position: float
    final_position: float
    final_velocity: float

    target_position: float
    commanded_target: float
    controller_kp: float

    actuator_delay_steps: int
    control_noise_std: float
    joint_damping: float

    initial_position_error: float
    final_position_error: float
    mean_position_error: float
    peak_abs_velocity: float
    overshoot_count: int
    control_saturation_fraction: float

    failure_reason: str | None


def _build_model_xml(
    *,
    controller_kp: float,
    joint_damping: float,
) -> str:
    """Insert selected physics and controller parameters."""

    return (
        MODEL_XML_TEMPLATE.replace(
            "__CONTROLLER_KP__",
            f"{controller_kp:.12g}",
        ).replace(
            "__JOINT_DAMPING__",
            f"{joint_damping:.12g}",
        )
    )


def run_reach_episode(
    *,
    seed: int,
    target_position: float,
    max_steps: int,
    position_tolerance: float,
    velocity_tolerance: float,
    initial_position_low: float,
    initial_position_high: float,
    controller_kp: float,
    target_bias: float,
    actuator_delay_steps: int,
    control_noise_std: float,
    joint_damping: float,
) -> MujocoReachResult:
    """Run one deterministic disturbed MuJoCo episode."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if controller_kp <= 0.0:
        raise ValueError("controller_kp must be positive.")

    if actuator_delay_steps < 0:
        raise ValueError(
            "actuator_delay_steps cannot be negative."
        )

    if control_noise_std < 0.0:
        raise ValueError(
            "control_noise_std cannot be negative."
        )

    if joint_damping <= 0.0:
        raise ValueError("joint_damping must be positive.")

    if initial_position_low > initial_position_high:
        raise ValueError(
            "Initial-position lower bound cannot exceed "
            "the upper bound."
        )

    model = mujoco.MjModel.from_xml_string(
        _build_model_xml(
            controller_kp=controller_kp,
            joint_damping=joint_damping,
        )
    )

    data = mujoco.MjData(model)

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "slide",
    )

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "slider_position",
    )

    if joint_id < 0 or actuator_id < 0:
        raise RuntimeError(
            "Required MuJoCo model elements were not found."
        )

    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])

    random_generator = np.random.default_rng(seed)

    initial_position = float(
        random_generator.uniform(
            initial_position_low,
            initial_position_high,
        )
    )

    commanded_target = float(
        np.clip(
            target_position + target_bias,
            0.0,
            1.0,
        )
    )

    delayed_commands: deque[float] = deque(
        [initial_position] * actuator_delay_steps
    )

    mujoco.mj_resetData(model, data)

    data.qpos[qpos_address] = initial_position
    data.qvel[dof_address] = 0.0

    mujoco.mj_forward(model, data)

    initial_position_error = abs(
        initial_position - target_position
    )

    previous_signed_error = (
        initial_position - target_position
    )

    completed_steps = 0
    success = False

    position_error_sum = 0.0
    peak_abs_velocity = 0.0
    overshoot_count = 0
    saturated_commands = 0

    for _ in range(max_steps):
        completed_steps += 1

        command_noise = float(
            random_generator.normal(
                loc=0.0,
                scale=control_noise_std,
            )
        )

        noisy_command = float(
            np.clip(
                commanded_target + command_noise,
                0.0,
                1.0,
            )
        )

        if actuator_delay_steps > 0:
            delayed_commands.append(noisy_command)
            applied_command = delayed_commands.popleft()
        else:
            applied_command = noisy_command

        if (
            applied_command <= 1e-12
            or applied_command >= 1.0 - 1e-12
        ):
            saturated_commands += 1

        data.ctrl[actuator_id] = applied_command

        mujoco.mj_step(model, data)

        current_position = float(
            data.qpos[qpos_address]
        )

        current_velocity = float(
            data.qvel[dof_address]
        )

        signed_error = (
            current_position - target_position
        )

        position_error = abs(signed_error)
        velocity = abs(current_velocity)

        position_error_sum += position_error

        peak_abs_velocity = max(
            peak_abs_velocity,
            velocity,
        )

        if (
            signed_error != 0.0
            and previous_signed_error != 0.0
            and (
                np.sign(signed_error)
                != np.sign(previous_signed_error)
            )
        ):
            overshoot_count += 1

        if signed_error != 0.0:
            previous_signed_error = signed_error

        if (
            position_error <= position_tolerance
            and velocity <= velocity_tolerance
        ):
            success = True
            break

    final_position = float(data.qpos[qpos_address])
    final_velocity = float(data.qvel[dof_address])

    final_position_error = abs(
        final_position - target_position
    )

    mean_position_error = (
        position_error_sum / completed_steps
    )

    control_saturation_fraction = (
        saturated_commands / completed_steps
    )

    return MujocoReachResult(
        seed=seed,
        success=success,
        steps=completed_steps,
        initial_position=initial_position,
        final_position=final_position,
        final_velocity=final_velocity,
        target_position=target_position,
        commanded_target=commanded_target,
        controller_kp=controller_kp,
        actuator_delay_steps=actuator_delay_steps,
        control_noise_std=control_noise_std,
        joint_damping=joint_damping,
        initial_position_error=initial_position_error,
        final_position_error=final_position_error,
        mean_position_error=mean_position_error,
        peak_abs_velocity=peak_abs_velocity,
        overshoot_count=overshoot_count,
        control_saturation_fraction=(
            control_saturation_fraction
        ),
        failure_reason=(
            None if success else "target_not_reached"
        ),
    )
