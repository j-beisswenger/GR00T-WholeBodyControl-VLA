"""End-to-end: AMASS SMPL-X motion → GMR retarget → SONIC g1/robot encoder.

Single-process pipeline (no CSV intermediate). Loads an AMASS .npz, runs GMR
retargeting to the Unitree G1 at 50 Hz, builds the deploy-format motion dict
in memory, and drives SONIC's encoder + decoder + MuJoCo PD loop using the
**robot encoder path** (mode_id=0 → run_g1).

Prereq (one-time):
    uv pip install -e GMR/

Run (with the uv venv active or via uv run):
    .venv/bin/python sonic_investigation/scripts/mujoco_viewer_amass_gmr_g1_encoder_decoder.py \
        --smplx_file sonic_investigation/data/amass/SFU/0005/0005_Walking001_stageii.npz

Encoder asymmetry recap (see findings/encoder_inputs.md):
  * `motion_joint_positions_*` is fed RAW (IsaacLab order, no offset).
  * Decoder side `his_body_joint_positions_*` is fed `q − default_angles`.
  * GMR.retarget() returns qpos in MuJoCo joint order; we permute to IsaacLab.
  * GMR's qpos[3:7] is wxyz (matches deploy CSV `body_quat` convention).
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
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from sonic_decoder_onnx import run_decoder  # noqa: E402
from sonic_encoder_onnx_3modes import run_g1  # noqa: E402
from mujoco_viewer_synthetic_idle_encoder_decoder import (  # noqa: E402
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
from mujoco_viewer_csv_motion_g1_encoder_decoder import _zero_yaw_motion, make_idle_motion  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
GMR_ROOT = REPO_ROOT / "GMR"
DEFAULT_SMPLX_FOLDER = REPO_ROOT / "sonic_investigation/data"
DEFAULT_SMPLX_FILE = (
    REPO_ROOT / "sonic_investigation/data/amass/SFU/0005/0005_Walking001_stageii.npz"
)

ENCODER_PATH = DEPLOY_ROOT / "policy/release/model_encoder.onnx"
DECODER_PATH = DEPLOY_ROOT / "policy/release/model_decoder.onnx"

CONTROL_HZ = 1.0 / CONTROL_DT  # 50.0
NUM_BODIES = 14  # body_quat layout (only body 0 / pelvis is read)


# ----- AMASS → deploy-format motion ----------------------------------------


def retarget_amass(smplx_file: Path, smplx_folder: Path) -> tuple[np.ndarray, float]:
    """GMR retarget. Returns (qpos_seq, fps).

    qpos_seq: (T, 36) = [root_xyz(3), root_wxyz(4), dof_mj(29)].
    """
    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        str(smplx_file), smplx_folder
    )
    # GMR's `frame_skip = int(src_fps / tgt_fps)` only hits 50 Hz when src is a
    # multiple of 50 (e.g. 100 Hz). For other rates (120, 60, …) we ask GMR for
    # the nearest-faster integer-skip rate, then resample to exactly 50 Hz below.
    smplx_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=int(CONTROL_HZ)
    )
    print(f"[gmr] retargeting source: {len(smplx_frames)} frames @ {aligned_fps:.2f} Hz "
          f"(human height {actual_human_height:.2f} m)")

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
    )
    qpos_list = [retargeter.retarget(frame) for frame in smplx_frames]
    return np.stack(qpos_list, axis=0), float(aligned_fps)


def resample_to_50hz(
    root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof_il: np.ndarray, src_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample to exactly CONTROL_HZ. Linear interp on translations/dofs, SLERP on quats."""
    if abs(src_fps - CONTROL_HZ) < 1e-6:
        return root_pos, root_quat_wxyz, dof_il

    T_src = root_pos.shape[0]
    duration = (T_src - 1) / src_fps
    T_tgt = int(np.floor(duration * CONTROL_HZ)) + 1
    t_src = np.arange(T_src) / src_fps
    t_tgt = np.arange(T_tgt) / CONTROL_HZ
    t_tgt = np.minimum(t_tgt, t_src[-1])

    rp = np.stack([np.interp(t_tgt, t_src, root_pos[:, k]) for k in range(3)], axis=1)
    dof = np.stack([np.interp(t_tgt, t_src, dof_il[:, k]) for k in range(dof_il.shape[1])], axis=1)
    # scipy Rotation uses xyzw; GMR/SONIC use wxyz.
    quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    slerp = Slerp(t_src, R.from_quat(quat_xyzw))
    rq_xyzw = slerp(t_tgt).as_quat()
    rq_wxyz = rq_xyzw[:, [3, 0, 1, 2]]
    return rp, rq_wxyz, dof


def build_deploy_motion(qpos_seq: np.ndarray, src_fps: float) -> dict[str, np.ndarray]:
    """Convert GMR qpos sequence into the dict shape that build_encoder_inputs() expects."""
    T_src = qpos_seq.shape[0]
    if qpos_seq.shape[1] != 7 + G1_NUM_MOTOR:
        raise RuntimeError(f"unexpected qpos width {qpos_seq.shape[1]}, expected {7 + G1_NUM_MOTOR}")

    root_pos_src = qpos_seq[:, 0:3]
    root_quat_src = qpos_seq[:, 3:7]
    dof_mj = qpos_seq[:, 7:]
    dof_il_src = dof_mj[:, MUJOCO_TO_ISAACLAB]

    root_pos, root_quat_wxyz, dof_il = resample_to_50hz(
        root_pos_src, root_quat_src, dof_il_src, src_fps
    )
    T = dof_il.shape[0]
    if T_src != T:
        print(f"[resample] {T_src} @ {src_fps:.2f} Hz → {T} @ {CONTROL_HZ:.0f} Hz")

    dt = 1.0 / CONTROL_HZ
    dof_vel_il = np.zeros_like(dof_il)
    if T >= 3:
        dof_vel_il[1:-1] = (dof_il[2:] - dof_il[:-2]) / (2.0 * dt)
        dof_vel_il[0] = (dof_il[1] - dof_il[0]) / dt
        dof_vel_il[-1] = (dof_il[-1] - dof_il[-2]) / dt
    elif T == 2:
        dof_vel_il[:] = (dof_il[1] - dof_il[0]) / dt

    body_pos = np.zeros((T, NUM_BODIES * 3), dtype=np.float64)
    body_pos[:, 0:3] = root_pos
    body_quat = np.zeros((T, NUM_BODIES * 4), dtype=np.float64)
    body_quat[:, 0:4] = root_quat_wxyz

    # Yaw-align frame 0 to identity, mirroring deploy's apply_delta_heading.
    aligned_pelvis_quat = _zero_yaw_motion(body_quat[:, 0:4])

    return {
        "joint_pos": dof_il,
        "joint_vel": dof_vel_il,
        "body_pos": body_pos[:, 0:3],
        "body_quat": aligned_pelvis_quat,
    }


# ----- one control tick (matches mujoco_viewer_csv_motion_g1_encoder_decoder.control_step) -----


def control_step(enc, dec, model, data, motion, cur_frame, hist, last_action_il, sim_steps_per_control):
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


def run_phase(enc, dec, model, data, viewer, motion, hist, last_action_il,
              duration_s, sim_steps_per_control, label, wall_t0):
    n_ticks = int(round(duration_s / CONTROL_DT))
    T = motion["joint_pos"].shape[0]
    print(f"[{label}] {duration_s:.2f}s ({n_ticks} ticks), motion length = {T} frames ({T / CONTROL_HZ:.2f}s)")
    for tick in range(n_ticks):
        if not viewer.is_running():
            return last_action_il, False
        cur_frame = min(tick, T - 1)
        last_action_il = control_step(enc, dec, model, data, motion, cur_frame, hist, last_action_il, sim_steps_per_control)
        viewer.sync()
        sleep = (wall_t0[0] + CONTROL_DT) - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        wall_t0[0] += CONTROL_DT
    return last_action_il, True


# ----- main ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx_file", type=Path, default=DEFAULT_SMPLX_FILE,
                        help=f"Path to AMASS SMPL-X .npz file. Default: {DEFAULT_SMPLX_FILE}")
    parser.add_argument("--smplx_folder", type=Path, default=DEFAULT_SMPLX_FOLDER,
                        help=f"Folder containing smplx/SMPLX_*.pkl. Default: {DEFAULT_SMPLX_FOLDER}")
    parser.add_argument("--idle_duration", type=float, default=2.0,
                        help="Seconds of stand-still idle before/between motion playback.")
    parser.add_argument("--num_cycles", type=int, default=1,
                        help="Times to play the motion (idle wraps both ends).")
    args = parser.parse_args()

    if not args.smplx_file.exists():
        sys.exit(f"smplx_file not found: {args.smplx_file}")
    if not (args.smplx_folder / "smplx" / "SMPLX_NEUTRAL.pkl").exists():
        sys.exit(f"SMPLX_NEUTRAL.pkl not found under {args.smplx_folder}/smplx/")

    qpos_seq, fps = retarget_amass(args.smplx_file, args.smplx_folder)
    motion_ref = build_deploy_motion(qpos_seq, fps)
    motion_dur = motion_ref["joint_pos"].shape[0] / CONTROL_HZ
    idle_motion = make_idle_motion(max(int(args.idle_duration * CONTROL_HZ), 50))
    print(f"[loaded] motion: {motion_ref['joint_pos'].shape[0]} frames ({motion_dur:.2f}s)")

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

    hist = {
        "q": np.zeros((HISTORY_LEN, G1_NUM_MOTOR)),
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
                motion_dur, sim_steps_per_control, args.smplx_file.stem, wall_t0,
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
