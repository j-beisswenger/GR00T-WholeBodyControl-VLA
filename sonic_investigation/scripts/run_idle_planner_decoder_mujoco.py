"""SONIC IDLE-mode rollout in MuJoCo with viewer (planner-free).

Wires just the encoder + decoder ONNX pair against a synthetic stand-still
reference motion — no planner ONNX, no resampling, no allowed-tokens mask:

  * (synthetic motion)             → flat tile of `default_angles` (IL order),
                                     identity quat, zero velocities
  * model_encoder.onnx   (50 Hz)   → 64-D universal token from that reference
  * model_decoder.onnx   (50 Hz)   → 29-D action (joint targets, IsaacLab order)
  * MuJoCo PD            (200 Hz)  → physics with kp/kd from policy_parameters.hpp

The deploy's `planner_sonic.onnx` in IDLE mode produces a near-constant copy
of `default_angles` with sub-millimeter sway; for a stand-still demo the
constant tile is indistinguishable. To re-introduce the planner (e.g. for
walk/run/squat modes that need a real reference motion), see `run_planner_idle`
in the git history of this file or use `run_motion_cycle_mujoco.py` for
playback of recorded motions.

⚠ Encoder vs decoder joint-position convention (read this before editing!)
─────────────────────────────────────────────────────────────────────────
The two networks are NOT symmetric in what they expect for joint positions:

  * Encoder `motion_joint_positions_*`: RAW joint angles (IsaacLab order),
    taken straight from the planner motion. The deploy stores
    `planner_motion_50hz_.JointPositions` as raw qpos values
    (localmotion_kplanner.hpp:494, no default-angle subtraction).

  * Decoder `his_body_joint_positions_*`: DEVIATION from default angles,
    i.e. `q - default_angles` in IsaacLab order. The deploy logs this at
    g1_deploy_onnx_ref.cpp:2827:
        body_q[i] = motor_state[mujoco_to_isaaclab[i]].q()
                  - default_angles[mujoco_to_isaaclab[i]];
    and `GatherHisBodyJointPositions` pipes that straight into the decoder.

Symptom of getting the decoder side wrong (feeding raw q): at t=0 the robot
sits at `default_angles` (knee 0.669, ankle −0.363, hip −0.312). The decoder
reads those as a huge "error from zero" and outputs corrective torques that
extend the legs and tip the robot — observed as either an instant fall or a
slow squat. See the `DEFAULT_ANGLES_IL` constant and the two `hist_q` write
sites for the exact subtraction.

Velocities, last actions, base ang-vel, and gravity are NOT default-offset
in either network — they are the values as-is.

Run:
    ./sonic_investigation/scripts/run_idle_planner_decoder_mujoco.sh
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Reuse the I/O layout helpers from the existing investigation scripts.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_decoder import run_decoder  # noqa: E402
from run_encoder_modes import run_g1  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"

ENCODER_PATH = DEPLOY_ROOT / "policy/release/model_encoder.onnx"
DECODER_PATH = DEPLOY_ROOT / "policy/release/model_decoder.onnx"
SCENE_XML = DEPLOY_ROOT / "g1/scene_29dof.xml"


# ----- constants from gear_sonic_deploy/.../policy_parameters.hpp ----------
# Verbatim copies; names match the C++ source for grep-ability.

G1_NUM_MOTOR = 29

# MuJoCo joint i ↔ IsaacLab joint ISAACLAB_TO_MUJOCO[i] (despite the name).
# i.e. ISAACLAB_TO_MUJOCO[mj_idx] = isaaclab_idx; MUJOCO_TO_ISAACLAB[isaaclab_idx] = mj_idx.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

# default_angles[] in MuJoCo order (policy_parameters.hpp:210)
DEFAULT_ANGLES_MJ = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,  # left leg
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,  # right leg
        0.0,
        0.0,
        0.0,  # waist
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,  # left arm
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,  # right arm
    ],
    dtype=np.float64,
)

# Same constants reordered into IsaacLab joint order. Deploy logs
# body_q[i] = motor_q[mujoco_to_isaaclab[i]] - default_angles[mujoco_to_isaaclab[i]]
# at g1_deploy_onnx_ref.cpp:2827, i.e. deviation from default_angles in IL order.
# The decoder's `his_body_joint_positions_*` reads back this deviation, NOT the
# raw joint angles. Failing to subtract this offset makes the decoder see a
# huge initial pose error and drives the robot into a squat/fall.
DEFAULT_ANGLES_IL = DEFAULT_ANGLES_MJ[MUJOCO_TO_ISAACLAB]

# action_scale = 0.25 * effort_limit / stiffness   (in MuJoCo order)
NATURAL_FREQ = 10.0 * 2.0 * np.pi
S_5020 = 0.003609725 * NATURAL_FREQ**2
S_7520_14 = 0.010177520 * NATURAL_FREQ**2
S_7520_22 = 0.025101925 * NATURAL_FREQ**2
S_4010 = 0.00425 * NATURAL_FREQ**2
D_5020 = 2.0 * 2.0 * 0.003609725 * NATURAL_FREQ
D_7520_14 = 2.0 * 2.0 * 0.010177520 * NATURAL_FREQ
D_7520_22 = 2.0 * 2.0 * 0.025101925 * NATURAL_FREQ
D_4010 = 2.0 * 2.0 * 0.00425 * NATURAL_FREQ
E_5020, E_7520_14, E_7520_22, E_4010 = 25.0, 88.0, 139.0, 5.0

ACTION_SCALE_MJ = np.array(
    [
        0.25 * E_7520_22 / S_7520_22,
        0.25 * E_7520_22 / S_7520_22,
        0.25 * E_7520_14 / S_7520_14,
        0.25 * E_7520_22 / S_7520_22,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_7520_22 / S_7520_22,
        0.25 * E_7520_22 / S_7520_22,
        0.25 * E_7520_14 / S_7520_14,
        0.25 * E_7520_22 / S_7520_22,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_7520_14 / S_7520_14,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_4010 / S_4010,
        0.25 * E_4010 / S_4010,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_5020 / S_5020,
        0.25 * E_4010 / S_4010,
        0.25 * E_4010 / S_4010,
    ],
    dtype=np.float64,
)

KPS_MJ = np.array(
    [
        S_7520_22,
        S_7520_22,
        S_7520_14,
        S_7520_22,
        2.0 * S_5020,
        2.0 * S_5020,
        S_7520_22,
        S_7520_22,
        S_7520_14,
        S_7520_22,
        2.0 * S_5020,
        2.0 * S_5020,
        S_7520_14,
        2.0 * S_5020,
        2.0 * S_5020,
        S_5020,
        S_5020,
        S_5020,
        S_5020,
        S_5020,
        S_4010,
        S_4010,
        S_5020,
        S_5020,
        S_5020,
        S_5020,
        S_5020,
        S_4010,
        S_4010,
    ],
    dtype=np.float64,
)
KDS_MJ = np.array(
    [
        D_7520_22,
        D_7520_22,
        D_7520_14,
        D_7520_22,
        2.0 * D_5020,
        2.0 * D_5020,
        D_7520_22,
        D_7520_22,
        D_7520_14,
        D_7520_22,
        2.0 * D_5020,
        2.0 * D_5020,
        D_7520_14,
        2.0 * D_5020,
        2.0 * D_5020,
        D_5020,
        D_5020,
        D_5020,
        D_5020,
        D_5020,
        D_4010,
        D_4010,
        D_5020,
        D_5020,
        D_5020,
        D_5020,
        D_5020,
        D_4010,
        D_4010,
    ],
    dtype=np.float64,
)


# ----- timing & idle-readapt constants from g1_deploy_onnx_ref.cpp ---------

SIM_DT = 0.005  # 200 Hz physics
CONTROL_DT = 0.02  # 50 Hz encoder+decoder
HISTORY_LEN = 10  # 10-tick `_step1` history windows for the decoder
LOOKAHEAD_N = 10  # 10 frames in the encoder's `_step5` lookahead
LOOKAHEAD_STEP = 5  # every 5 control ticks
DEFAULT_HEIGHT = 0.788740  # PlannerConfig::default_height (localmotion_kplanner.hpp:217)


# ----- quaternion helpers (wxyz convention to match deploy) ----------------


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=a.dtype,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q (wxyz)."""
    p = np.array([0.0, v[0], v[1], v[2]], dtype=v.dtype)
    r = quat_mul(quat_mul(q, p), quat_conj(q))
    return r[1:]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=q.dtype,
    )


# ----- synthetic stand-still reference motion ------------------------------


def make_idle_motion(num_frames: int = 100) -> dict[str, np.ndarray]:
    """Build a flat 50-Hz 'stand still' reference: default angles + identity quat.

    Replaces the planner+resample pipeline for IDLE: the deploy planner in
    `LocomotionMode::IDLE` produces a near-constant copy of `default_angles`
    with sub-millimeter sway (it never re-plans — `is_static_motion_mode(IDLE)`
    short-circuits the replan predicate at g1_deploy_onnx_ref.cpp:3692). For a
    stand-still demo a constant tile is indistinguishable, and skipping the
    planner ONNX drops ~15 ms of startup and ~80 lines of input-binding code.

    `joint_pos` is in IsaacLab order (matches `motion_joint_positions_*` raw
    convention). `body_quat` is identity (wxyz). `body_pos` is at the planner's
    default standing height.
    """
    return {
        "joint_pos": np.tile(DEFAULT_ANGLES_IL, (num_frames, 1)),
        "joint_vel": np.zeros((num_frames, G1_NUM_MOTOR), dtype=np.float64),
        "body_pos": np.tile(np.array([0.0, 0.0, DEFAULT_HEIGHT]), (num_frames, 1)),
        "body_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_frames, 1)),
    }


# ----- encoder/decoder input builders --------------------------------------


def build_encoder_inputs(
    motion: dict[str, np.ndarray], cur_frame: int, base_quat_wxyz: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the 3 non-zero inputs to `run_g1` from the planner reference.

    For each of `LOOKAHEAD_N` future frames at stride `LOOKAHEAD_STEP`, clamps to
    the last available frame (matches the deploy gather behavior when
    `t + k·step` exceeds motion length, see GatherMotion*MultiFrame).
    """
    T = motion["joint_pos"].shape[0]
    motion_jp = np.zeros((LOOKAHEAD_N, G1_NUM_MOTOR), dtype=np.float64)
    motion_jv = np.zeros((LOOKAHEAD_N, G1_NUM_MOTOR), dtype=np.float64)
    motion_anchor = np.zeros((LOOKAHEAD_N, 6), dtype=np.float64)

    for k in range(LOOKAHEAD_N):
        f = min(cur_frame + k * LOOKAHEAD_STEP, T - 1)
        motion_jp[k] = motion["joint_pos"][f]
        motion_jv[k] = motion["joint_vel"][f]
        # anchor = first 2 columns of rotmat(inv(base_quat) · ref_root_quat),
        # row-major flatten to 6D (g1_deploy_onnx_ref.cpp:679-683).
        rel = quat_mul(quat_conj(base_quat_wxyz), motion["body_quat"][f])
        rmat = quat_to_mat(rel)
        motion_anchor[k] = rmat[:, :2].reshape(-1)
    return {"motion_jp": motion_jp, "motion_jv": motion_jv, "motion_anchor": motion_anchor}


# ----- MuJoCo PD step ------------------------------------------------------


def pd_step(model: mujoco.MjModel, data: mujoco.MjData, q_target_mj: np.ndarray, steps: int) -> None:
    """Run `steps` physics ticks holding the same joint target."""
    qpos_idx = np.arange(7, 7 + G1_NUM_MOTOR)
    qvel_idx = np.arange(6, 6 + G1_NUM_MOTOR)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    for _ in range(steps):
        q = data.qpos[qpos_idx]
        dq = data.qvel[qvel_idx]
        data.ctrl[:G1_NUM_MOTOR] = np.clip(
            KPS_MJ * (q_target_mj - q) + KDS_MJ * (-dq),
            lo,
            hi,
        )
        mujoco.mj_step(model, data)


# ----- main loop -----------------------------------------------------------


def main() -> None:
    enc = ort.InferenceSession(str(ENCODER_PATH), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(DECODER_PATH), providers=["CPUExecutionProvider"])

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    sim_steps_per_control = int(round(CONTROL_DT / SIM_DT))  # 4

    # Drop the robot at the default standing pose.
    data.qpos[:3] = (0.0, 0.0, DEFAULT_HEIGHT)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7 : 7 + G1_NUM_MOTOR] = DEFAULT_ANGLES_MJ
    mujoco.mj_forward(model, data)

    # Synthetic stand-still reference (no planner). Cursor clamps at last frame
    # past `T`, so any T ≥ encoder lookahead window is sufficient.
    motion = make_idle_motion(num_frames=100)
    T = motion["joint_pos"].shape[0]
    print(f"[idle] {T} frames @ 50 Hz ({T / 50.0:.2f}s of synthetic stand-still reference)")

    # 10-tick history buffers, all in IsaacLab joint order, oldest-first.
    # `hist_q` stores `q - default_angles_il` (deviation), matching what the
    # deploy logs into StateLogger.body_q (cpp:2827) and what the decoder reads
    # back as `his_body_joint_positions_10frame_step1`.
    hist_q = np.tile(motion["joint_pos"][0] - DEFAULT_ANGLES_IL, (HISTORY_LEN, 1))
    hist_dq = np.zeros((HISTORY_LEN, G1_NUM_MOTOR))
    hist_act = np.zeros((HISTORY_LEN, G1_NUM_MOTOR))  # raw decoder outputs
    hist_omega = np.zeros((HISTORY_LEN, 3))
    hist_grav = np.tile(np.array([0.0, 0.0, -1.0]), (HISTORY_LEN, 1))

    cur_frame = 0
    last_action_isaaclab = np.zeros(G1_NUM_MOTOR)  # decoder raw output (IsaacLab)

    print("[viewer] launching… close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("pelvis").id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 120, -20
        wall_t0 = time.monotonic()

        tick = 0
        while viewer.is_running():
            # ----- read state in both MuJoCo and IsaacLab orders -----
            base_quat = data.qpos[3:7].copy()  # wxyz
            qpos_jnt_mj = data.qpos[7 : 7 + G1_NUM_MOTOR].copy()
            qvel_jnt_mj = data.qvel[6 : 6 + G1_NUM_MOTOR].copy()
            qpos_jnt_il = qpos_jnt_mj[MUJOCO_TO_ISAACLAB]
            qvel_jnt_il = qvel_jnt_mj[MUJOCO_TO_ISAACLAB]
            base_omega_body = data.qvel[3:6].copy()  # free joint: body-frame ω
            grav_body = quat_rotate(quat_conj(base_quat), np.array([0.0, 0.0, -1.0]))

            # ----- shift history buffers (oldest drops out) -----
            # Deploy: body_q = motor_q[mj→il] - default_angles[mj→il]  (cpp:2827).
            hist_q = np.roll(hist_q, -1, axis=0)
            hist_q[-1] = qpos_jnt_il - DEFAULT_ANGLES_IL
            hist_dq = np.roll(hist_dq, -1, axis=0)
            hist_dq[-1] = qvel_jnt_il
            hist_act = np.roll(hist_act, -1, axis=0)
            hist_act[-1] = last_action_isaaclab
            hist_omega = np.roll(hist_omega, -1, axis=0)
            hist_omega[-1] = base_omega_body
            hist_grav = np.roll(hist_grav, -1, axis=0)
            hist_grav[-1] = grav_body

            # ----- encoder forward (G1 mode = 0; teleop/SMPL fields stay 0) -----
            enc_in = build_encoder_inputs(motion, cur_frame, base_quat)
            token = run_g1(enc, enc_in["motion_jp"], enc_in["motion_jv"], enc_in["motion_anchor"])

            # ----- decoder forward -----
            action_il = run_decoder(
                dec,
                token,
                hist_omega,
                hist_q,
                hist_dq,
                hist_act,
                hist_grav,
            )
            last_action_isaaclab = action_il.astype(np.float64)

            # ----- IsaacLab raw action → MuJoCo joint targets (cpp:3120-3122) -----
            #   q_target_mj[i] = default_angles[i] + action_il[ISAACLAB_TO_MUJOCO[i]] * scale[i]
            q_target_mj = DEFAULT_ANGLES_MJ + action_il[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_MJ

            # ----- physics: 4 sim steps per control tick -----
            pd_step(model, data, q_target_mj, sim_steps_per_control)
            viewer.sync()

            # ----- advance playback cursor (clamp at last frame in IDLE) -----
            if cur_frame < T - 1:
                cur_frame += 1

            # ----- pace to wall clock -----
            sleep = (wall_t0 + (tick + 1) * CONTROL_DT) - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            tick += 1

            if tick % 50 == 0:
                print(f"  t={tick * CONTROL_DT:5.2f}s  frame={cur_frame:3d}/{T-1}  " f"z={data.qpos[2]:.3f}m")

    print("[viewer] closed.")


if __name__ == "__main__":
    main()
