"""SONIC reference-motion cycle in MuJoCo with viewer (planner-free).

No planner. Loads a recorded reference motion from disk and feeds it (alongside
a synthetic 'stand-still' idle reference) through the same encoder + decoder +
MuJoCo PD stack as `run_idle_planner_decoder_mujoco.py`. Each run alternates:

    idle (T_idle s) -> motion -> idle -> motion -> ... -> idle

so you can repeatedly watch a motion play out and the controller readapt back
to the default standing pose between takes.

Reference motions live in `gear_sonic_deploy/reference/example/<name>/` and are
the per-channel CSVs the deploy `MotionDataReader` consumes (50 Hz, IsaacLab
joint order, body 0 = pelvis):
    joint_pos.csv   (T, 29)
    joint_vel.csv   (T, 29)
    body_pos.csv    (T, 14*3)   -> we keep body 0 (pelvis) only
    body_quat.csv   (T, 14*4)   -> we keep body 0 (pelvis) only, wxyz

The synthetic idle reference is a flat copy of `default_angles` (IsaacLab order)
with identity quat and pelvis at `default_height`, mimicking what the planner's
IDLE mode produces.

Run:
    ./sonic_investigation/scripts/run_motion_cycle_mujoco.sh --motion <name>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_decoder import run_decoder  # noqa: E402
from run_encoder_modes import run_g1  # noqa: E402
from run_idle_planner_decoder_mujoco import (  # noqa: E402
    ACTION_SCALE_MJ,
    CONTROL_DT,
    DEFAULT_ANGLES_IL,
    DEFAULT_ANGLES_MJ,
    DEFAULT_HEIGHT,
    DEPLOY_ROOT,
    G1_NUM_MOTOR,
    HISTORY_LEN,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    SCENE_XML,
    SIM_DT,
    build_encoder_inputs,
    pd_step,
    quat_conj,
    quat_mul,
    quat_rotate,
)

ENCODER_PATH = DEPLOY_ROOT / "policy/release/model_encoder.onnx"
DECODER_PATH = DEPLOY_ROOT / "policy/release/model_decoder.onnx"
REFERENCE_ROOT = DEPLOY_ROOT / "reference/example"


# ----- motion loading ------------------------------------------------------


def list_motions() -> list[str]:
    if not REFERENCE_ROOT.is_dir():
        return []
    return sorted(p.name for p in REFERENCE_ROOT.iterdir() if (p / "joint_pos.csv").exists())


def _zero_yaw_motion(body_quat: np.ndarray) -> np.ndarray:
    """Yaw-align the motion so frame 0's pelvis has zero heading.

    Mirrors deploy's `apply_delta_heading` (g1_deploy_onnx_ref.cpp:589-602):
        apply_delta_heading = init_heading · inv(data_heading)
    where `data_heading` is the yaw-only quat of frame 0's pelvis. Because we
    enter motion playback with the robot at ~identity yaw (init_heading ≈ I),
    this reduces to a static pre-multiplication by inv(data_heading). Without
    it the encoder's `motion_anchor` (= inv(base_quat) · motion_root_quat) sees
    the recordings' ~−90° yaw offset (IsaacLab world frame), and the decoder
    snaps the robot to match within ~one tick of motion phase entry.
    """
    q0 = body_quat[0]
    x_rot = quat_rotate(q0, np.array([1.0, 0.0, 0.0]))  # (q0 ⊗ x̂) — same as deploy calc_heading_d
    yaw = float(np.arctan2(x_rot[1], x_rot[0]))
    yaw_inv = np.array([np.cos(-yaw / 2.0), 0.0, 0.0, np.sin(-yaw / 2.0)])
    return np.stack([quat_mul(yaw_inv, q) for q in body_quat], axis=0)


def load_reference_motion(motion_dir: Path) -> dict[str, np.ndarray]:
    """Load a deploy-format motion (50 Hz, IsaacLab order). Yaw-aligns frame 0 to identity."""

    def _read(name: str) -> np.ndarray:
        return np.loadtxt(motion_dir / name, delimiter=",", skiprows=1, dtype=np.float64)

    jp = _read("joint_pos.csv")  # (T, 29)
    jv = _read("joint_vel.csv")  # (T, 29)
    bp = _read("body_pos.csv")  # (T, 14*3) — body 0 = pelvis
    bq = _read("body_quat.csv")  # (T, 14*4) wxyz, body 0 = pelvis
    if jp.shape[1] != G1_NUM_MOTOR or jv.shape[1] != G1_NUM_MOTOR:
        raise RuntimeError(f"unexpected joint dim in {motion_dir} (jp={jp.shape}, jv={jv.shape})")

    return {
        "joint_pos": jp,
        "joint_vel": jv,
        "body_pos": bp[:, 0:3],
        "body_quat": _zero_yaw_motion(bq[:, 0:4]),
    }


def make_idle_motion(num_frames: int) -> dict[str, np.ndarray]:
    """Synthesize a 'stand still' reference: default angles, identity quat."""
    return {
        "joint_pos": np.tile(DEFAULT_ANGLES_IL, (num_frames, 1)),
        "joint_vel": np.zeros((num_frames, G1_NUM_MOTOR), dtype=np.float64),
        "body_pos": np.tile(np.array([0.0, 0.0, DEFAULT_HEIGHT]), (num_frames, 1)),
        "body_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_frames, 1)),
    }


# ----- one control tick ----------------------------------------------------


def control_step(
    enc: ort.InferenceSession,
    dec: ort.InferenceSession,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: dict[str, np.ndarray],
    cur_frame: int,
    hist: dict[str, np.ndarray],
    last_action_il: np.ndarray,
    sim_steps_per_control: int,
) -> np.ndarray:
    """Run encoder + decoder + 4 PD substeps. Updates `hist` in place; returns new action (IL)."""

    base_quat = data.qpos[3:7].copy()
    qpos_jnt_mj = data.qpos[7 : 7 + G1_NUM_MOTOR].copy()
    qvel_jnt_mj = data.qvel[6 : 6 + G1_NUM_MOTOR].copy()
    qpos_jnt_il = qpos_jnt_mj[MUJOCO_TO_ISAACLAB]
    qvel_jnt_il = qvel_jnt_mj[MUJOCO_TO_ISAACLAB]
    base_omega_body = data.qvel[3:6].copy()
    grav_body = quat_rotate(quat_conj(base_quat), np.array([0.0, 0.0, -1.0]))

    hist["q"] = np.roll(hist["q"], -1, axis=0)
    hist["q"][-1] = qpos_jnt_il - DEFAULT_ANGLES_IL
    hist["dq"] = np.roll(hist["dq"], -1, axis=0)
    hist["dq"][-1] = qvel_jnt_il
    hist["act"] = np.roll(hist["act"], -1, axis=0)
    hist["act"][-1] = last_action_il
    hist["omega"] = np.roll(hist["omega"], -1, axis=0)
    hist["omega"][-1] = base_omega_body
    hist["grav"] = np.roll(hist["grav"], -1, axis=0)
    hist["grav"][-1] = grav_body

    enc_in = build_encoder_inputs(motion, cur_frame, base_quat)
    token = run_g1(enc, enc_in["motion_jp"], enc_in["motion_jv"], enc_in["motion_anchor"])

    action_il = run_decoder(dec, token, hist["omega"], hist["q"], hist["dq"], hist["act"], hist["grav"])

    q_target_mj = DEFAULT_ANGLES_MJ + action_il[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_MJ
    pd_step(model, data, q_target_mj, sim_steps_per_control)
    return action_il.astype(np.float64)


# ----- phase runner --------------------------------------------------------


def run_phase(
    enc, dec, model, data, viewer, motion, hist, last_action_il, duration_s, sim_steps_per_control, label, wall_t0
):
    """Play `motion` (clamping at the last frame) for `duration_s`. Returns updated last_action."""
    n_ticks = int(round(duration_s / CONTROL_DT))
    T = motion["joint_pos"].shape[0]
    print(f"[{label}] {duration_s:.2f}s ({n_ticks} ticks), motion length = {T} frames ({T / 50.0:.2f}s)")
    for tick in range(n_ticks):
        if not viewer.is_running():
            return last_action_il, False
        cur_frame = min(tick, T - 1)
        last_action_il = control_step(
            enc, dec, model, data, motion, cur_frame, hist, last_action_il, sim_steps_per_control
        )
        viewer.sync()
        sleep = (wall_t0[0] + CONTROL_DT) - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        wall_t0[0] += CONTROL_DT
    return last_action_il, True


# ----- main ----------------------------------------------------------------


def main() -> None:
    motions = list_motions()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--motion",
        required=True,
        choices=motions if motions else None,
        help="Reference motion directory name under gear_sonic_deploy/reference/example/",
    )
    parser.add_argument("--idle-duration", type=float, default=3.0, help="Seconds of idle phase between motions.")
    parser.add_argument(
        "--motion-duration",
        type=float,
        default=None,
        help="Seconds for each motion phase (default: motion length / 50 Hz).",
    )
    parser.add_argument("--num-cycles", type=int, default=3, help="Number of motion phases. Idle wraps both ends.")
    args = parser.parse_args()

    enc = ort.InferenceSession(str(ENCODER_PATH), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(DECODER_PATH), providers=["CPUExecutionProvider"])

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    sim_steps_per_control = int(round(CONTROL_DT / SIM_DT))

    data.qpos[:3] = (0.0, 0.0, DEFAULT_HEIGHT)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7 : 7 + G1_NUM_MOTOR] = DEFAULT_ANGLES_MJ
    mujoco.mj_forward(model, data)

    motion_ref = load_reference_motion(REFERENCE_ROOT / args.motion)
    motion_dur = args.motion_duration if args.motion_duration is not None else motion_ref["joint_pos"].shape[0] / 50.0
    idle_motion = make_idle_motion(max(int(args.idle_duration * 50), 50))
    print(f"[loaded] motion '{args.motion}': {motion_ref['joint_pos'].shape[0]} frames ({motion_dur:.2f}s)")

    hist = {
        "q": np.tile(DEFAULT_ANGLES_IL - DEFAULT_ANGLES_IL, (HISTORY_LEN, 1)),  # zero deviation
        "dq": np.zeros((HISTORY_LEN, G1_NUM_MOTOR)),
        "act": np.zeros((HISTORY_LEN, G1_NUM_MOTOR)),
        "omega": np.zeros((HISTORY_LEN, 3)),
        "grav": np.tile(np.array([0.0, 0.0, -1.0]), (HISTORY_LEN, 1)),
    }
    last_action_il = np.zeros(G1_NUM_MOTOR)

    print("[viewer] launching… close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("pelvis").id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 120, -15
        wall_t0 = [time.monotonic()]

        for cycle in range(args.num_cycles):
            print(f"\n=== cycle {cycle + 1}/{args.num_cycles} ===")
            last_action_il, ok = run_phase(
                enc, dec, model, data, viewer, idle_motion, hist, last_action_il,
                args.idle_duration, sim_steps_per_control, "idle", wall_t0,
            )
            if not ok:
                break
            last_action_il, ok = run_phase(
                enc, dec, model, data, viewer, motion_ref, hist, last_action_il,
                motion_dur, sim_steps_per_control, args.motion, wall_t0,
            )
            if not ok:
                break
        else:
            run_phase(
                enc, dec, model, data, viewer, idle_motion, hist, last_action_il,
                args.idle_duration, sim_steps_per_control, "idle (final)", wall_t0,
            )

    print("[viewer] closed.")


if __name__ == "__main__":
    main()
