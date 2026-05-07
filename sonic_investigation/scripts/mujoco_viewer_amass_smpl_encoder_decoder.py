"""SONIC AMASS-motion cycle in MuJoCo with viewer (planner-free, SMPL/'human' encoder).

Same idle ↔ motion cycle as `mujoco_viewer_csv_motion_g1_encoder_decoder.py`, but the reference
motion is an AMASS .npz fed through encoder mode 2 (`smpl`, the human encoder)
instead of mode 0 (`g1`, the robot encoder).

Pipeline per AMASS file:
  1. SMPL-X forward (`smplx` package) on AMASS axis-angle pose params → 24-joint
     positions per frame in the Z-up world frame the AMASS data already lives in.
     SMPL-X joints 0–21 == SMPL joints 0–21; for SMPL joints 22/23 (L/R hand we
     don't have in SMPL-X) we use SMPL-X 28/43 (left/right `middle1`, base of
     the middle-finger metacarpal — the closest hand-center proxy).
  2. Resample 120 Hz → 50 Hz to match the SONIC control rate.
  3. Yaw-zero frame 0 so the encoder's `smpl_anchor_orientation` lines up with
     the robot starting at identity heading (same idea as `_zero_yaw_motion`
     for the g1 encoder).
  4. Each control tick: build SMPL encoder inputs (10 consecutive future frames
     at step 1) → `run_smpl` → 64D token → `run_decoder` → MuJoCo PD step.

Idle phase reuses the synthetic stand-still reference (pelvis at default height,
identity quat, default joint angles) but routed through the SMPL encoder: 24
joints frozen at the T-pose → world-frame joint positions for a standing person.

Run:
    ./sonic_investigation/scripts/mujoco_viewer_amass_smpl_encoder_decoder.sh --motion <path.npz>
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
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from sonic_decoder_onnx import run_decoder  # noqa: E402
from sonic_encoder_onnx_3modes import run_smpl  # noqa: E402
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
    pd_step,
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_to_mat,
)

ENCODER_PATH = DEPLOY_ROOT / "policy/release/model_encoder.onnx"
DECODER_PATH = DEPLOY_ROOT / "policy/release/model_decoder.onnx"

# Encoder mode 2 (`smpl`) reads `_step1` (consecutive) windows: 10 frames at
# stride 1 (= 0.2 s lookahead at 50 Hz). Matches deploy
# `smpl_joints_10frame_step1` (g1_deploy_onnx_ref.cpp:1748).
SMPL_LOOKAHEAD_N = 10
SMPL_LOOKAHEAD_STEP = 1

# G1 wrist joints in IsaacLab order. Matches
# `wrist_joint_isaaclab_order_in_isaaclab_index` in the deploy
# (policy_parameters.hpp:88) — fed as `motion_joint_positions_wrists_10frame_step1`
# in SMPL mode (g1_deploy_onnx_ref.cpp:1739). AMASS has no G1 joint angles, so we
# leave these at default (==0 in IL order — DEFAULT_ANGLES_IL[23..28] are all 0).
WRIST_JOINTS_IL = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)

# SMPL-X joint indices that produce the 24 SMPL joints expected by the encoder.
# 0–21 line up with SMPL exactly. SMPL's L_Hand/R_Hand (22/23) don't exist in
# SMPL-X — we substitute SMPL-X left_middle1 (28) and right_middle1 (43), which
# sit at the center of the palm and are the closest geometric match to where
# SMPL's hand joint lives. Index list is from `smplx.joint_names.JOINT_NAMES`.
SMPLX_TO_SMPL24 = np.array(list(range(22)) + [28, 43], dtype=np.int64)

# SMPL "base rotation": axis-angle root_orient bakes in a Y-up→Z-up swap so the
# rest T-pose stands upright in world. Conjugating this constant out of the root
# quat leaves a "facing direction" rotation in the same convention as the robot
# pelvis quat (Z up, x = forward). Mirrors `rotations.remove_smpl_base_rot`
# (gear_sonic/isaac_utils/rotations.py:704). Without this the encoder's
# `smpl_anchor_orientation` is dominated by the constant 120° base rotation
# instead of the actual heading delta vs the robot.
SMPL_BASE_ROT_WXYZ = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
SMPL_BASE_ROT_CONJ_WXYZ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)


# ----- AMASS / SMPL-X loading ---------------------------------------------


def axis_angle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle → (N, 4) quaternion (wxyz). Returns identity for ‖aa‖→0."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(angle > 1e-8, angle, 1.0)
    half = angle * 0.5
    sin_over_a = np.where(angle > 1e-8, np.sin(half) / safe, 0.5)
    xyz = aa * sin_over_a
    w = np.cos(half)
    return np.concatenate([w, xyz], axis=-1)


def _resample(arr: np.ndarray, n_src: int, n_tgt: int) -> np.ndarray:
    """Linear interpolation along axis 0 from n_src frames to n_tgt frames."""
    if n_src == n_tgt:
        return arr.copy()
    t = np.linspace(0.0, n_src - 1, n_tgt)
    f0 = np.floor(t).astype(np.int64)
    f1 = np.minimum(f0 + 1, n_src - 1)
    a = (t - f0).reshape(-1, *([1] * (arr.ndim - 1)))
    return (1 - a) * arr[f0] + a * arr[f1]


def _resample_quat(q: np.ndarray, n_src: int, n_tgt: int) -> np.ndarray:
    """Linear interp of quats with sign-flip + renorm. Good enough at 120→50 Hz."""
    if n_src == n_tgt:
        return q.copy()
    t = np.linspace(0.0, n_src - 1, n_tgt)
    f0 = np.floor(t).astype(np.int64)
    f1 = np.minimum(f0 + 1, n_src - 1)
    a = (t - f0).reshape(-1, 1)
    q0 = q[f0]
    q1 = q[f1]
    sign = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -1.0, 1.0)
    qi = (1 - a) * q0 + a * (q1 * sign)
    return qi / np.linalg.norm(qi, axis=-1, keepdims=True)


def load_amass_motion(motion_path: Path, smplx_dir: Path, target_fps: int = 50) -> dict[str, np.ndarray]:
    """Load AMASS .npz, run SMPL-X forward, resample to `target_fps`. Z-up world frame.

    Returns:
        smpl_joints: (T, 24, 3) — pelvis-centered joints (= world joints − pelvis),
                     world axes (Z up). Rotated into body-local later by
                     quat_inv(root_quat).
        smpl_root_quat: (T, 4) wxyz — SMPL root in Z-up world frame, with
                        `remove_smpl_base_rot` applied so frame 0 reads as a
                        nearly-pure yaw of the body's facing direction (matches
                        training-side `smpl_root_quat_w` convention).
        body_pos: (T, 3) — pelvis position in world (kept for diagnostics).
    """
    import smplx  # imported lazily so the module is usable for `--list` etc.

    data = np.load(str(motion_path), allow_pickle=True)
    src_fps = float(data["mocap_frame_rate"])
    gender = str(data["gender"])
    n_src = data["pose_body"].shape[0]
    n_betas = int(data["num_betas"]) if "num_betas" in data else data["betas"].shape[-1]

    body = smplx.create(
        str(smplx_dir),
        "smplx",
        gender=gender,
        use_pca=False,
        num_betas=n_betas,
        batch_size=n_src,
    )
    betas = torch.tensor(np.asarray(data["betas"][:n_betas])).float().view(1, -1).expand(n_src, -1)
    out = body(
        betas=betas,
        global_orient=torch.tensor(np.asarray(data["root_orient"])).float(),
        body_pose=torch.tensor(np.asarray(data["pose_body"])).float(),
        transl=torch.tensor(np.asarray(data["trans"])).float(),
    )
    joints_world = out.joints.detach().numpy()[:, SMPLX_TO_SMPL24, :].astype(np.float64)  # (T, 24, 3)
    pelvis_world = joints_world[:, 0, :].copy()  # (T, 3)
    smpl_joints_pc = joints_world - pelvis_world[:, None, :]  # (T, 24, 3) pelvis-centered
    smpl_root_quat_raw = axis_angle_to_quat_wxyz(np.asarray(data["root_orient"], dtype=np.float64))
    # Strip the SMPL Y-up→Z-up base rotation so the resulting quat is comparable
    # to the robot's pelvis quat (training-side `remove_smpl_base_rot`).
    smpl_root_quat = np.stack([quat_mul(q, SMPL_BASE_ROT_CONJ_WXYZ) for q in smpl_root_quat_raw], axis=0)

    n_tgt = max(1, int(round(n_src * target_fps / src_fps)))
    smpl_joints_pc = _resample(smpl_joints_pc, n_src, n_tgt)
    pelvis_world = _resample(pelvis_world, n_src, n_tgt)
    smpl_root_quat = _resample_quat(smpl_root_quat, n_src, n_tgt)

    return {
        "smpl_joints": smpl_joints_pc,
        "smpl_root_quat": smpl_root_quat,
        "body_pos": pelvis_world,
        "src_fps": src_fps,
        "name": motion_path.stem,
    }


def make_idle_smpl_motion(num_frames: int, t_pose_joints_pc: np.ndarray) -> dict[str, np.ndarray]:
    """Synthetic stand-still SMPL reference: T-pose joints (pelvis-centered), identity root."""
    return {
        "smpl_joints": np.tile(t_pose_joints_pc, (num_frames, 1, 1)),
        "smpl_root_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_frames, 1)),
        "body_pos": np.tile(np.array([0.0, 0.0, DEFAULT_HEIGHT]), (num_frames, 1)),
        "src_fps": 50.0,
        "name": "idle",
    }


def compute_t_pose_joints(smplx_dir: Path) -> np.ndarray:
    """SMPL-X T-pose joints in the body-local frame the encoder learned. (24, 3).

    Used as the synthetic 'stand still' reference for the idle phase. The idle
    motion plays with `smpl_root_quat = identity`, so when the encoder rotates
    these joints by `quat_inv(identity)` they're left as-is — meaning we must
    return joints already in the body-local frame an AMASS frame would land in
    after `quat_inv(remove_smpl_base_rot(root_quat))`.

    SMPL T-pose with zero root_orient lives in the SMPL Y-up local frame:
    +X = body-left, +Y = body-up, +Z = body-rear. The base rotation
    [0.5, 0.5, 0.5, 0.5] is a 120° permutation that maps (X,Y,Z) → (Y,Z,X), i.e.
    SMPL-X→body-Y, SMPL-Y→body-Z, SMPL-Z→body-X. After applying it: head along
    body +Z, left wrist along body +Y, body forward = body -X — matching what
    AMASS produces in body-local after `remove_smpl_base_rot` + quat_inv.

    Neutral betas are fine here — the idle T-pose doesn't model a specific
    subject and the encoder cares about joint topology, not body-shape scale.
    """
    import smplx

    body = smplx.create(str(smplx_dir), "smplx", gender="neutral", use_pca=False, num_betas=10, batch_size=1)
    out = body(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 63),
        transl=torch.zeros(1, 3),
    )
    j = out.joints.detach().numpy()[0, SMPLX_TO_SMPL24, :].astype(np.float64)
    j = j - j[0:1]  # pelvis-centered
    R_base = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return j @ R_base.T


# ----- yaw alignment ------------------------------------------------------


# def _zero_yaw_smpl_motion(smpl_root_quat: np.ndarray) -> np.ndarray:
#     """Yaw-align frame 0 to identity heading (mirrors `_zero_yaw_motion` in run_motion_cycle).

#     `smpl_anchor_orientation = inv(robot_anchor) · smpl_root_quat`. The robot
#     starts at identity yaw; without this alignment the encoder sees AMASS's
#     arbitrary initial heading and the decoder snaps the robot to face that way
#     within a tick of motion entry.
#     """
#     q0 = smpl_root_quat[0]
#     x_rot = quat_rotate(q0, np.array([1.0, 0.0, 0.0]))
#     yaw = float(np.arctan2(x_rot[1], x_rot[0]))
#     yaw_inv = np.array([np.cos(-yaw / 2.0), 0.0, 0.0, np.sin(-yaw / 2.0)])
#     return np.stack([quat_mul(yaw_inv, q) for q in smpl_root_quat], axis=0)


def _zero_yaw_smpl_motion(smpl_root_quat: np.ndarray, smpl_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw-align frame 0 to identity heading for both root quats and world joints."""
    q0 = smpl_root_quat[0]
    x_rot = quat_rotate(q0, np.array([1.0, 0.0, 0.0]))
    yaw = float(np.arctan2(x_rot[1], x_rot[0]))
    yaw_inv = np.array([np.cos(-yaw / 2.0), 0.0, 0.0, np.sin(-yaw / 2.0)])

    # 1. Rotate the root quaternions
    aligned_root_quat = np.stack([quat_mul(yaw_inv, q) for q in smpl_root_quat], axis=0)

    # 2. Rotate the pelvis-centered world joints by the same yaw
    # Flatten to (T*24, 3) to apply quat_rotate easily, then reshape back
    flat_joints = smpl_joints.reshape(-1, 3)
    aligned_flat_joints = np.stack([quat_rotate(yaw_inv, pos) for pos in flat_joints])
    aligned_joints = aligned_flat_joints.reshape(smpl_joints.shape)

    return aligned_root_quat, aligned_joints


# ----- SMPL encoder input builder -----------------------------------------


def build_smpl_encoder_inputs(
    motion: dict[str, np.ndarray], cur_frame: int, base_quat_wxyz: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the 3 non-zero SMPL-mode inputs for `run_smpl` from the reference motion.

    Mirrors training-side `smpl_joints_multi_future_local` / `smpl_root_ori_b_mf`
    (gear_sonic/envs/manager_env/mdp/observations.py:1716, 1625):

        smpl_joints_local[k]      = quat_apply(quat_inv(root_q[f]), joints_pc[f])
        smpl_anchor_orientation[k] = first 2 cols of mat(quat_inv(robot_anchor)·root_q[f])

    Both are filled per future frame at stride `SMPL_LOOKAHEAD_STEP=1` (= 10
    consecutive 50 Hz frames). Frames past motion end clamp to the last frame
    (matches the deploy gather behavior).
    """
    T = motion["smpl_joints"].shape[0]
    smpl_joints = np.zeros((SMPL_LOOKAHEAD_N, 24, 3), dtype=np.float64)
    smpl_anchor = np.zeros((SMPL_LOOKAHEAD_N, 6), dtype=np.float64)

    for k in range(SMPL_LOOKAHEAD_N):
        f = min(cur_frame + k * SMPL_LOOKAHEAD_STEP, T - 1)
        rq = motion["smpl_root_quat"][f]
        rq_inv = quat_conj(rq)
        # Rotate each pelvis-centered joint into body-local frame.
        for j in range(24):
            smpl_joints[k, j] = quat_rotate(rq_inv, motion["smpl_joints"][f, j])
        # 6D anchor: first 2 columns of rotmat(inv(robot_anchor) · smpl_root).
        rel = quat_mul(quat_conj(base_quat_wxyz), rq)
        smpl_anchor[k] = quat_to_mat(rel)[:, :2].reshape(-1)

    # Wrist G1 joints: AMASS has none → zero deviation from default angles
    # (DEFAULT_ANGLES_IL[23..28] is all zeros, so this is also raw value 0).
    wrist_jp = np.zeros((SMPL_LOOKAHEAD_N, 6), dtype=np.float64)

    return {"smpl_joints": smpl_joints, "smpl_anchor": smpl_anchor, "wrist_jp": wrist_jp}


# ----- one control tick ---------------------------------------------------


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
    """Encoder (mode 2 = smpl) + decoder + 4 PD substeps. Updates `hist` in place."""
    base_quat = data.qpos[3:7].copy()
    qpos_jnt_mj = data.qpos[7 : 7 + G1_NUM_MOTOR].copy()
    qvel_jnt_mj = data.qvel[6 : 6 + G1_NUM_MOTOR].copy()
    qpos_jnt_il = qpos_jnt_mj[MUJOCO_TO_ISAACLAB]
    qvel_jnt_il = qvel_jnt_mj[MUJOCO_TO_ISAACLAB]
    base_omega_body = data.qvel[3:6].copy()
    grav_body = quat_rotate(quat_conj(base_quat), np.array([0.0, 0.0, -1.0]))

    hist["q"] = np.roll(hist["q"], -1, axis=0)
    hist["q"][-1] = qpos_jnt_il - DEFAULT_ANGLES_IL  # decoder-side: deviation
    hist["dq"] = np.roll(hist["dq"], -1, axis=0)
    hist["dq"][-1] = qvel_jnt_il
    hist["act"] = np.roll(hist["act"], -1, axis=0)
    hist["act"][-1] = last_action_il
    hist["omega"] = np.roll(hist["omega"], -1, axis=0)
    hist["omega"][-1] = base_omega_body
    hist["grav"] = np.roll(hist["grav"], -1, axis=0)
    hist["grav"][-1] = grav_body

    enc_in = build_smpl_encoder_inputs(motion, cur_frame, base_quat)
    token = run_smpl(enc, enc_in["smpl_joints"], enc_in["smpl_anchor"], enc_in["wrist_jp"])
    action_il = run_decoder(dec, token, hist["omega"], hist["q"], hist["dq"], hist["act"], hist["grav"])
    q_target_mj = DEFAULT_ANGLES_MJ + action_il[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_MJ
    pd_step(model, data, q_target_mj, sim_steps_per_control)
    return action_il.astype(np.float64)


# ----- phase runner -------------------------------------------------------


def run_phase(
    enc, dec, model, data, viewer, motion, hist, last_action_il, duration_s, sim_steps_per_control, label, wall_t0
):
    n_ticks = int(round(duration_s / CONTROL_DT))
    T = motion["smpl_joints"].shape[0]
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


# ----- main ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", required=True, type=Path, help="Path to AMASS .npz motion file.")
    parser.add_argument(
        "--smplx-dir",
        type=Path,
        default=Path("/Users/jbeisswenger/Downloads/models"),
        help="Parent directory containing a smplx/ subfolder with SMPLX_{NEUTRAL,MALE,FEMALE}.npz.",
    )
    parser.add_argument("--idle-duration", type=float, default=3.0, help="Seconds of idle phase between motions.")
    parser.add_argument(
        "--motion-duration",
        type=float,
        default=None,
        help="Seconds for each motion phase (default: motion length at 50 Hz).",
    )
    parser.add_argument("--num-cycles", type=int, default=3, help="Number of motion phases. Idle wraps both ends.")
    args = parser.parse_args()

    if not args.motion.is_file():
        raise SystemExit(f"motion not found: {args.motion}")
    if not (args.smplx_dir / "smplx").is_dir():
        raise SystemExit(f"smplx model dir not found: {args.smplx_dir}/smplx (expected SMPLX_*.npz inside).")

    print(f"[load] AMASS: {args.motion.name}")
    motion_ref = load_amass_motion(args.motion, args.smplx_dir, target_fps=int(round(1.0 / CONTROL_DT)))

    # Fix: update both the root quats AND the joint positions
    motion_ref["smpl_root_quat"], motion_ref["smpl_joints"] = _zero_yaw_smpl_motion(
        motion_ref["smpl_root_quat"], motion_ref["smpl_joints"]
    )

    motion_dur = (
        args.motion_duration if args.motion_duration is not None else motion_ref["smpl_joints"].shape[0] / 50.0
    )
    print(
        f"[loaded] '{motion_ref['name']}': {motion_ref['smpl_joints'].shape[0]} frames @50Hz "
        f"(src {motion_ref['src_fps']:.0f} Hz, {motion_dur:.2f}s)"
    )

    t_pose = compute_t_pose_joints(args.smplx_dir)
    idle_motion = make_idle_smpl_motion(max(int(args.idle_duration * 50), 50), t_pose)

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
                enc,
                dec,
                model,
                data,
                viewer,
                idle_motion,
                hist,
                last_action_il,
                args.idle_duration,
                sim_steps_per_control,
                "idle",
                wall_t0,
            )
            if not ok:
                break
            last_action_il, ok = run_phase(
                enc,
                dec,
                model,
                data,
                viewer,
                motion_ref,
                hist,
                last_action_il,
                motion_dur,
                sim_steps_per_control,
                motion_ref["name"],
                wall_t0,
            )
            if not ok:
                break
        else:
            run_phase(
                enc,
                dec,
                model,
                data,
                viewer,
                idle_motion,
                hist,
                last_action_il,
                args.idle_duration,
                sim_steps_per_control,
                "idle (final)",
                wall_t0,
            )

    print("[viewer] closed.")


if __name__ == "__main__":
    main()
