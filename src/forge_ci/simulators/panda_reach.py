"""Deterministic position-only Panda reaching using an external Menagerie model."""

import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np


def object_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing required model object: {name}")
    return index


def run_panda_episode(
    model_path: Path,
    *,
    seed: int = 42,
    view: bool = False,
    duration_s: float = 6.0,
    position_tolerance_m: float = 0.01,
    speed_tolerance_m_s: float = 0.02,
    command_bias_rad: float = 0.0,
    command_delay_steps: int = 0,
    command_noise_std_rad: float = 0.0,
) -> dict:
    """Evaluate the same target distribution with optional command-channel faults."""
    if duration_s < 2.0 or command_delay_steps < 0 or command_noise_std_rad < 0:
        raise ValueError("Invalid duration, delay or command noise.")
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    joint_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 8)
    ]
    actuator_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
        for i in range(1, 8)
    ]
    qpos = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    hand = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    home = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    lower = model.jnt_range[joint_ids, 0] + 0.02
    upper = model.jnt_range[joint_ids, 1] - 0.02
    # The hand frame is at the wrist; this fixed local offset approximates the
    # gripper tool centre between the fingertips. Only position is controlled.
    tool_offset = np.array([0.0, 0.0, 0.1034])

    def tip(d):
        return d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ tool_offset

    def reset(d):
        mujoco.mj_resetDataKeyframe(model, d, home)
        mujoco.mj_forward(model, d)

    reset(data)
    initial_q = data.qpos[qpos].copy()
    initial_tip = tip(data).copy()
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
    reference = mujoco.MjData(model)
    reset(reference)
    # Sample a nearby known-reachable configuration without using its joint
    # values as the final command; inverse kinematics starts from home.
    offset = np.array([0.30, 0.20, -0.18, -0.25, 0.15, 0.25, 0.10])
    offset += rng.uniform(-0.04, 0.04, size=7)
    reference.qpos[qpos] = np.clip(initial_q + offset, lower, upper)
    mujoco.mj_forward(model, reference)
    target = tip(reference).copy()

    ik = mujoco.MjData(model)
    reset(ik)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(250):
        error = target - tip(ik)
        if np.linalg.norm(error) < 0.0005:
            break
        mujoco.mj_jac(model, ik, jacp, jacr, tip(ik), hand)
        jac = jacp[:, dofs]
        delta = jac.T @ np.linalg.solve(
            jac @ jac.T + 0.035**2 * np.eye(3), error
        )
        ik.qpos[qpos] = np.clip(
            ik.qpos[qpos] + np.clip(delta, -0.10, 0.10), lower, upper
        )
        mujoco.mj_forward(model, ik)

    ik_error = float(np.linalg.norm(target - tip(ik)))
    if ik_error > 0.002:
        raise RuntimeError(f"Inverse kinematics did not converge: {ik_error:.6f} m")
    command = ik.qpos[qpos].copy()

    if view:
        import mujoco.viewer as mj_viewer

        context = mj_viewer.launch_passive(model, data)
    else:
        context = nullcontext(None)

    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.35, 0.0, 0.45]
                viewer.cam.distance = 1.7
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.018, 0.018, 0.018]),
                    target,
                    np.eye(3).ravel(),
                    np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
                )
                viewer.user_scn.ngeom = 1

        dt = model.opt.timestep
        steps = int(np.ceil(duration_s / dt))
        delayed = deque([initial_q.copy() for _ in range(command_delay_steps)])
        settling_step = None
        peak_error = 0.0
        error_sum = 0.0
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                raise RuntimeError("Viewer closed before the trial finished")
            fraction = min((step + 1) * dt / 2.0, 1.0)
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            requested = initial_q + blend * (command - initial_q)
            # Bias only the first arm joint; delay/noise affect all seven commands.
            requested[0] += command_bias_rad * blend
            if command_noise_std_rad:
                requested += noise_rng.normal(0.0, command_noise_std_rad, 7)
            requested = np.clip(requested, lower, upper)
            if command_delay_steps:
                delayed.append(requested.copy())
                requested = delayed.popleft()
            data.ctrl[actuator_ids] = requested
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            tool = tip(data)
            error_now = float(np.linalg.norm(target - tool))
            mujoco.mj_jac(model, data, jacp, jacr, tool, hand)
            speed_now = float(np.linalg.norm(jacp @ data.qvel))
            peak_error = max(peak_error, error_now)
            error_sum += error_now
            if not np.isfinite(error_now) or not np.isfinite(speed_now):
                raise RuntimeError("Non-finite Panda simulation state")
            if (
                fraction >= 1.0
                and error_now <= position_tolerance_m
                and speed_now <= speed_tolerance_m_s
            ):
                if settling_step is None:
                    settling_step = step + 1
            else:
                settling_step = None
            if viewer is not None and (step + 1) % 10 == 0:
                viewer.sync()
                time.sleep(10 * dt)

        mujoco.mj_forward(model, data)
        final_tip = tip(data).copy()
        final_error = float(np.linalg.norm(target - final_tip))
        mujoco.mj_jac(model, data, jacp, jacr, final_tip, hand)
        final_speed = float(np.linalg.norm(jacp @ data.qvel))
        passed = final_error <= position_tolerance_m and final_speed <= speed_tolerance_m_s
        result = {
            "model": "Franka Emika Panda (MuJoCo Menagerie)",
            "seed": seed,
            "steps": steps,
            "command_bias_rad": command_bias_rad,
            "command_delay_steps": command_delay_steps,
            "command_noise_std_rad": command_noise_std_rad,
            "mean_position_error_m": error_sum / steps,
            "peak_position_error_m": peak_error,
            "settling_time_s": settling_step * dt if settling_step is not None else None,
            "simulation_seconds": float(data.time),
            "initial_tool_xyz_m": initial_tip.tolist(),
            "target_xyz_m": target.tolist(),
            "final_tool_xyz_m": final_tip.tolist(),
            "target_displacement_m": float(np.linalg.norm(target - initial_tip)),
            "ik_error_m": ik_error,
            "final_position_error_m": final_error,
            "final_tool_speed_m_s": final_speed,
            "position_tolerance_m": position_tolerance_m,
            "speed_tolerance_m_s": speed_tolerance_m_s,
            "passed": bool(passed),
        }
        if viewer is not None:
            viewer.sync()
            print("Trial finished. Close the viewer to exit.", flush=True)
            while viewer.is_running():
                time.sleep(0.1)

    return result
