#!/usr/bin/env python3
"""Convert Humanoid Everyday dataset to SONIC VLA LeRobot v2.1 format.

Standalone script — no dependency on GR00T-WholeBodyControl / gear_sonic.
Requires only: numpy, pyarrow, scipy, tqdm, onnxruntime, pinocchio, huggingface_hub.

Reads G1 episodes from the Humanoid Everyday dataset (30Hz, raw joints),
upsamples to 50Hz (linear interp for joints, SLERP for quaternions),
computes end-effector state via pinocchio FK (using bundled G1 URDF),
runs the SONIC ONNX encoder offline to produce 64-dim motion tokens,
and writes the result as a LeRobot v2.1 dataset at 50Hz.
Videos are also resampled from 30fps to 50fps via ffmpeg.

Two dataset variants:
  --subset locomanip  -> 879 G1 locomanipulation episodes
  --subset all        -> 4064 G1 episodes (all categories)

The SONIC encoder ONNX model is auto-downloaded from HuggingFace (nvidia/GEAR-SONIC)
on first run.

Usage:
    pip install -r requirements.txt
    python convert.py \\
        --source-dir /path/to/humanoid_everyday \\
        --output-dir /path/to/output_dataset \\
        --subset locomanip
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Lightweight pinocchio FK wrapper (replaces gear_sonic dependency)
# ---------------------------------------------------------------------------

class G1RobotModel:
    """Minimal FK wrapper matching SONIC's gear_sonic.data.robot_model.RobotModel.

    Loads the G1 URDF (bundled in g1_model_data/) with pinocchio, applies the same
    joint limit overrides as SONIC's G1SupplementalInfo, and clips joints before FK
    — identical behaviour to the original RobotModel.cache_forward_kinematics().
    """

    # SONIC supplemental_info overrides for joints where URDF limits differ
    _LIMIT_OVERRIDES = {
        "left_shoulder_roll_joint": (0.19, 2.2515),
        "right_shoulder_roll_joint": (-2.2515, -0.19),
        "right_hand_index_0_joint": (-1.57079632, 0),
        "right_hand_index_1_joint": (-1.74532925, 0),
        "right_hand_middle_0_joint": (-1.57079632, 0),
        "right_hand_middle_1_joint": (-1.74532925, 0),
        "right_hand_thumb_1_joint": (-0.72431163, 1.04719755),
        "right_hand_thumb_2_joint": (0, 1.74532925),
    }

    def __init__(self, urdf_path: str):
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.lower_limits = self.model.lowerPositionLimit.copy()
        self.upper_limits = self.model.upperPositionLimit.copy()
        for name, (lo, hi) in self._LIMIT_OVERRIDES.items():
            jid = self.model.getJointId(name)
            idx = self.model.joints[jid].idx_q
            self.lower_limits[idx] = lo
            self.upper_limits[idx] = hi

    @property
    def num_dofs(self) -> int:
        return self.model.nq

    def cache_forward_kinematics(self, q: np.ndarray):
        q = np.clip(q, self.lower_limits + 1e-6, self.upper_limits - 1e-6)
        self._pin.framesForwardKinematics(self.model, self.data, q)

    def frame_placement(self, frame_name: str):
        frame_id = self.model.getFrameId(frame_name)
        return self.data.oMf[frame_id].copy()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_FPS = 30
TARGET_FPS = 50
NUM_BODY_JOINTS = 29
NUM_HAND_JOINTS_PER_HAND = 7
NUM_HAND_JOINTS = 14  # both hands
NUM_FULL_JOINTS = 43  # body 29 + left_hand 7 + right_hand 7
ENCODER_INPUT_DIM = 1762
ENCODER_OUTPUT_DIM = 64

# Multi-frame encoder params (G1 mode)
NUM_FUTURE_FRAMES = 10
FRAME_STEP = 5  # at 50Hz, step=5 -> 0.1s intervals

# SONIC 29-DOF body joint order (from g1_supplemental_info.py):
# [0:6]   left_leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
# [6:12]  right_leg: same
# [12:15] waist: yaw, roll, pitch
# [15:22] left_arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
# [22:29] right_arm: same
#
# Humanoid Everyday:
# leg_joints (15): left_leg(6) + right_leg(6) + waist(3) -> matches SONIC order
# arm_joints (14): left_arm(7) + right_arm(7) -> matches SONIC order

# Hand joint reordering: HE uses thumb(3)+index(2)+middle(2),
# SONIC uses index(2)+middle(2)+thumb(3)
HAND_REMAP = [3, 4, 5, 6, 0, 1, 2]
# Inverse remap: SONIC order -> HE/URDF order (for pinocchio FK)
HAND_REMAP_INV = [4, 5, 6, 0, 1, 2, 3]

# ONNX encoder input layout (1762D total):
# All observations are concatenated in this order for the release model.
# For G1 mode (mode_id=0), only marked observations are filled, rest are zero.
ENCODER_OBS_LAYOUT = [
    ("encoder_mode_4", 4, True),  # [mode_id, 0, 0, 0]
    ("motion_joint_positions_10frame_step5", 290, True),   # 29 * 10
    ("motion_joint_velocities_10frame_step5", 290, True),  # 29 * 10
    ("motion_root_z_position_10frame_step5", 10, False),
    ("motion_root_z_position", 1, False),
    ("motion_anchor_orientation", 6, False),
    ("motion_anchor_orientation_10frame_step5", 60, True),  # 6 * 10
    ("motion_joint_positions_lowerbody_10frame_step5", 120, False),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120, False),
    ("vr_3point_local_target", 9, False),
    ("vr_3point_local_orn_target", 12, False),
    ("smpl_joints_10frame_step1", 720, False),
    ("smpl_anchor_orientation_10frame_step1", 60, False),
    ("motion_joint_positions_wrists_10frame_step1", 60, False),
]

# Pre-compute offsets for the G1-required observations
_G1_JOINT_POS_OFFSET = 4  # after encoder_mode_4
_G1_JOINT_VEL_OFFSET = 4 + 290  # after joint_pos
_G1_ANCHOR_ORI_OFFSET = 4 + 290 + 290 + 10 + 1 + 6  # skip z_pos and single ori


# ---------------------------------------------------------------------------
# Joint mapping
# ---------------------------------------------------------------------------


def map_joints_to_sonic29(leg_joints: np.ndarray, arm_joints: np.ndarray) -> np.ndarray:
    """Map Humanoid Everyday joints to SONIC 29-DOF order.

    Args:
        leg_joints: (N, 15) - left_leg(6) + right_leg(6) + waist(3)
        arm_joints: (N, 14) - left_arm(7) + right_arm(7)

    Returns:
        (N, 29) in SONIC order
    """
    return np.concatenate([
        leg_joints[:, 0:6],    # left_leg
        leg_joints[:, 6:12],   # right_leg
        leg_joints[:, 12:15],  # waist
        arm_joints[:, 0:7],    # left_arm
        arm_joints[:, 7:14],   # right_arm
    ], axis=1)


def remap_hand_joints(hand_joints_he: np.ndarray) -> np.ndarray:
    """Remap hand joints from HE order (thumb,index,middle) to SONIC order (index,middle,thumb).

    Args:
        hand_joints_he: (N, 7) per hand

    Returns:
        (N, 7) in SONIC order
    """
    return hand_joints_he[:, HAND_REMAP]


# ---------------------------------------------------------------------------
# Upsampling (30Hz -> 50Hz)
# ---------------------------------------------------------------------------


def upsample_linear(data: np.ndarray, fps_source: int, fps_target: int) -> np.ndarray:
    """Linearly interpolate data from fps_source to fps_target.

    Args:
        data: (T_src, D)
        fps_source: source frame rate
        fps_target: target frame rate

    Returns:
        (T_tgt, D) upsampled data
    """
    T_src = data.shape[0]
    duration = (T_src - 1) / fps_source
    T_tgt = int(round(duration * fps_target)) + 1

    t_src = np.linspace(0, duration, T_src)
    t_tgt = np.linspace(0, duration, T_tgt)

    # Vectorized linear interpolation
    result = np.empty((T_tgt, data.shape[1]), dtype=np.float64)
    for d in range(data.shape[1]):
        result[:, d] = np.interp(t_tgt, t_src, data[:, d])
    return result


def upsample_quaternions(quats: np.ndarray, fps_source: int, fps_target: int) -> np.ndarray:
    """SLERP interpolate quaternions (wxyz) from fps_source to fps_target.

    Args:
        quats: (T_src, 4) scalar-first quaternions
        fps_source: source frame rate
        fps_target: target frame rate

    Returns:
        (T_tgt, 4) interpolated quaternions (wxyz)
    """
    T_src = quats.shape[0]
    duration = (T_src - 1) / fps_source
    T_tgt = int(round(duration * fps_target)) + 1

    t_src = np.linspace(0, duration, T_src)
    t_tgt = np.linspace(0, duration, T_tgt)

    # scipy uses xyzw convention
    quats_xyzw = quats[:, [1, 2, 3, 0]]
    rots = R.from_quat(quats_xyzw)
    slerp = Slerp(t_src, rots)
    result_rots = slerp(t_tgt)
    result_xyzw = result_rots.as_quat()
    return result_xyzw[:, [3, 0, 1, 2]]  # back to wxyz


# ---------------------------------------------------------------------------
# Projected gravity
# ---------------------------------------------------------------------------


def compute_projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Compute projected gravity vector from base quaternion.

    Args:
        base_quat_wxyz: (4,) wxyz quaternion

    Returns:
        (3,) projected gravity in body frame
    """
    gravity_world = np.array([0.0, 0.0, -1.0])
    quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)
    return rot.inv().apply(gravity_world).astype(np.float32)


# ---------------------------------------------------------------------------
# Anchor orientation (quat -> 6D rotation)
# ---------------------------------------------------------------------------


def quat_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 6D rotation (first 2 cols of rotation matrix, row-wise).

    Args:
        q_wxyz: (4,) or (N, 4)

    Returns:
        (6,) or (N, 6)
    """
    q = np.atleast_2d(q_wxyz)
    q_xyzw = q[:, [1, 2, 3, 0]]
    rot_mat = R.from_quat(q_xyzw).as_matrix()  # (N, 3, 3)
    # First 2 columns, row-wise: [[r00,r01], [r10,r11], [r20,r21]]
    rot_6d = rot_mat[:, :, :2].reshape(-1, 6)
    if q_wxyz.ndim == 1:
        return rot_6d[0]
    return rot_6d


def compute_anchor_orientation(base_quat_wxyz: np.ndarray, ref_quat_wxyz: np.ndarray) -> np.ndarray:
    """Compute anchor orientation: relative rotation from base to reference as 6D.

    anchor_ori = rot6d(quat_inv(base) * ref)

    Args:
        base_quat_wxyz: (4,) current frame base quaternion
        ref_quat_wxyz: (4,) reference frame body quaternion

    Returns:
        (6,) 6D rotation representation
    """
    base_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
    ref_xyzw = ref_quat_wxyz[[1, 2, 3, 0]]
    base_rot = R.from_quat(base_xyzw)
    ref_rot = R.from_quat(ref_xyzw)
    rel_rot = base_rot.inv() * ref_rot
    mat = rel_rot.as_matrix()
    return mat[:, :2].reshape(6)


# ---------------------------------------------------------------------------
# Compute joint velocities via finite differences
# ---------------------------------------------------------------------------


def compute_joint_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute joint velocities from positions using finite differences.

    Args:
        positions: (T, D) joint positions
        dt: time step

    Returns:
        (T, D) joint velocities
    """
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / dt
    velocities[0] = velocities[1]  # repeat first velocity
    return velocities


# ---------------------------------------------------------------------------
# ONNX encoder
# ---------------------------------------------------------------------------


def load_encoder(model_path: str):
    """Load ONNX encoder session, using GPU if available."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 4

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        print("  Using GPU (CUDAExecutionProvider)")
        session = ort.InferenceSession(
            model_path, opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        print("  Using CPU only")
        session = ort.InferenceSession(model_path, opts)
    return session


def build_encoder_input_g1(
    joint_positions_50hz: np.ndarray,
    joint_velocities_50hz: np.ndarray,
    base_quats_50hz: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    """Build the 1762D encoder input for G1 mode at a given frame.

    For the G1 encoder, we need:
    - encoder_mode_4: [0, 0, 0, 0] (mode_id=0 is G1)
    - motion_joint_positions_10frame_step5: 10 future frames * 29 joints
    - motion_joint_velocities_10frame_step5: 10 future frames * 29 joints
    - motion_anchor_orientation_10frame_step5: 10 future frames * 6D rotation

    All other observations are zero-filled.

    Args:
        joint_positions_50hz: (T, 29) joint positions at 50Hz
        joint_velocities_50hz: (T, 29) joint velocities at 50Hz
        base_quats_50hz: (T, 4) base quaternions at 50Hz (wxyz)
        frame_idx: current frame index

    Returns:
        (1762,) encoder input vector
    """
    T = len(joint_positions_50hz)
    obs = np.zeros(ENCODER_INPUT_DIM, dtype=np.float32)

    # encoder_mode_4: first element = mode_id (0 for G1)
    obs[0] = 0.0  # G1 mode

    # Gather 10 future frames with step=5
    for fi in range(NUM_FUTURE_FRAMES):
        target_frame = min(frame_idx + fi * FRAME_STEP, T - 1)

        # Joint positions
        pos_start = _G1_JOINT_POS_OFFSET + fi * NUM_BODY_JOINTS
        obs[pos_start:pos_start + NUM_BODY_JOINTS] = joint_positions_50hz[target_frame]

        # Joint velocities
        vel_start = _G1_JOINT_VEL_OFFSET + fi * NUM_BODY_JOINTS
        obs[vel_start:vel_start + NUM_BODY_JOINTS] = joint_velocities_50hz[target_frame]

        # Anchor orientation: relative rotation from current base to future reference
        anchor_ori = compute_anchor_orientation(
            base_quats_50hz[frame_idx],
            base_quats_50hz[target_frame],
        )
        ori_start = _G1_ANCHOR_ORI_OFFSET + fi * 6
        obs[ori_start:ori_start + 6] = anchor_ori

    return obs


def encode_episode(
    session,
    joint_positions_50hz: np.ndarray,
    joint_velocities_50hz: np.ndarray,
    base_quats_50hz: np.ndarray,
) -> np.ndarray:
    """Run ONNX encoder on all frames of an episode.

    Args:
        session: ONNX runtime session
        joint_positions_50hz: (T, 29)
        joint_velocities_50hz: (T, 29)
        base_quats_50hz: (T, 4)

    Returns:
        (T, 64) motion tokens
    """
    T = len(joint_positions_50hz)
    tokens = np.zeros((T, ENCODER_OUTPUT_DIM), dtype=np.float64)

    # Process one frame at a time (ONNX model expects batch_size=1)
    for i in range(T):
        obs = build_encoder_input_g1(
            joint_positions_50hz, joint_velocities_50hz,
            base_quats_50hz, i,
        )
        out = session.run(["encoded_tokens"], {"obs_dict": obs[None]})[0]  # (1, 64)
        tokens[i] = out[0]

    return tokens


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------


def load_source_episodes(source_dir: str, subset: str):
    """Load episode metadata, filtered by subset.

    Args:
        source_dir: path to humanoid_everyday dataset
        subset: "locomanip" or "all"

    Returns:
        List of episode dicts, task descriptions dict
    """
    episodes = []
    with open(os.path.join(source_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            episodes.append(json.loads(line))

    tasks = {}
    with open(os.path.join(source_dir, "meta", "tasks.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_index"]] = t

    # Filter to G1 only
    g1_episodes = [e for e in episodes if e.get("robot_type") == "g1"]

    if subset == "locomanip":
        filtered = []
        for ep in g1_episodes:
            task_idx = ep["tasks"][0]
            if task_idx in tasks and tasks[task_idx].get("category") == "Locomanip":
                filtered.append(ep)
        g1_episodes = filtered

    print(f"Selected {len(g1_episodes)} episodes ({subset})")
    return g1_episodes, tasks


def load_episode_data(source_dir: str, episode: dict):
    """Load a single episode's parquet data.

    Returns dict with numpy arrays for each field.
    """
    ep_idx = episode["episode_index"]
    chunk = ep_idx // 1000
    parquet_path = os.path.join(
        source_dir, "data", f"chunk-{chunk:03d}", f"episode_{ep_idx:06d}.parquet"
    )
    table = pq.read_table(parquet_path)

    data = {}
    data["leg_joints"] = np.array(table["observation.leg_joints"].to_pylist(), dtype=np.float64)
    data["arm_joints"] = np.array(table["observation.arm_joints"].to_pylist(), dtype=np.float64)
    data["hand_joints"] = np.array(table["observation.hand_joints"].to_pylist(), dtype=np.float64)
    data["imu_quat"] = np.array(table["observation.imu.quaternion"].to_pylist(), dtype=np.float64)
    data["odom_quat"] = np.array(table["observation.odometry.quat"].to_pylist(), dtype=np.float64)
    data["action"] = np.array(table["action"].to_pylist(), dtype=np.float64)
    data["frame_index"] = np.array(table["frame_index"].to_pylist(), dtype=np.int64)
    data["timestamp"] = np.array(table["timestamp"].to_pylist(), dtype=np.float64)

    return data


# ---------------------------------------------------------------------------
# Video handling
# ---------------------------------------------------------------------------


def get_video_path(source_dir: str, episode_index: int) -> str:
    """Get the source video path for an episode."""
    chunk = episode_index // 1000
    return os.path.join(
        source_dir, "videos", f"chunk-{chunk:03d}",
        "egocentric", f"episode_{episode_index:06d}.mp4"
    )


def copy_or_resample_video(
    src_video: str, dst_video: str, target_fps: int = TARGET_FPS
):
    """Copy video, resampling to target_fps if needed.

    Falls back to simple copy if ffmpeg is not available.
    """
    os.makedirs(os.path.dirname(dst_video), exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_video,
                "-r", str(target_fps),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                dst_video,
            ],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ffmpeg not available or failed - just copy the video as-is
        shutil.copy2(src_video, dst_video)


# ---------------------------------------------------------------------------
# Convert single episode
# ---------------------------------------------------------------------------


def compute_eef_state(robot_model, joints_43_urdf_order: np.ndarray) -> np.ndarray:
    """Compute end-effector state (14D) via pinocchio FK for all frames.

    Matches SONIC's data collection: runs FK on left_wrist_yaw_link and
    right_wrist_yaw_link with joint clipping (same as RobotModel.cache_forward_kinematics).

    Args:
        robot_model: G1RobotModel instance
        joints_43_urdf_order: (T, 43) joint positions in URDF/pinocchio order
            (body 29 + left_hand 7 in HE order + right_hand 7 in HE order)

    Returns:
        (T, 14) = left_pos(3) + left_quat_wxyz(4) + right_pos(3) + right_quat_wxyz(4)
    """
    T = len(joints_43_urdf_order)
    eef = np.zeros((T, 14), dtype=np.float64)

    for t in range(T):
        robot_model.cache_forward_kinematics(joints_43_urdf_order[t])

        for side_idx, frame_name in enumerate(["left_wrist_yaw_link", "right_wrist_yaw_link"]):
            se3 = robot_model.frame_placement(frame_name)
            pos = se3.translation  # (3,)
            rot_mat = se3.rotation  # (3, 3)
            # Convert rotation matrix to wxyz quaternion
            quat_xyzw = R.from_matrix(rot_mat).as_quat()  # scipy: xyzw
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

            offset = side_idx * 7
            eef[t, offset:offset + 3] = pos
            eef[t, offset + 3:offset + 7] = quat_wxyz

    return eef


def convert_episode(
    source_dir: str,
    episode: dict,
    tasks: dict,
    encoder_session,
    output_ep_idx: int,
    robot_model=None,
) -> dict:
    """Convert a single episode from HE format to SONIC VLA format.

    Returns:
        dict with:
            - "frames": list of frame dicts (one per 50Hz timestep)
            - "episode_meta": episode metadata
            - "task_description": str
            - "num_frames_50hz": int
            - "source_ep_idx": int
    """
    ep_idx = episode["episode_index"]
    task_idx = episode["tasks"][0]
    task_desc = tasks.get(task_idx, {}).get("description", "")

    # Load source data (30Hz)
    data = load_episode_data(source_dir, episode)

    # Map to SONIC 29-DOF body joints
    joints_29_30hz = map_joints_to_sonic29(data["leg_joints"], data["arm_joints"])

    # Base quaternion (use IMU quat - it's the body frame orientation)
    base_quats_30hz = data["imu_quat"]  # (T, 4) wxyz

    # Upsample body joints and quats to 50Hz
    joints_29_50hz = upsample_linear(joints_29_30hz, SOURCE_FPS, TARGET_FPS)
    base_quats_50hz = upsample_quaternions(base_quats_30hz, SOURCE_FPS, TARGET_FPS)

    # Compute velocities at 50Hz (body only, for encoder)
    dt_50hz = 1.0 / TARGET_FPS
    joint_vels_50hz = compute_joint_velocities(joints_29_50hz, dt_50hz)

    # --- Hand observation joints (from observation.hand_joints) ---
    # These are the measured/state hand joints, remapped to SONIC order
    left_hand_obs_30hz = remap_hand_joints(data["hand_joints"][:, :7])
    right_hand_obs_30hz = remap_hand_joints(data["hand_joints"][:, 7:14])
    left_hand_obs_50hz = upsample_linear(left_hand_obs_30hz, SOURCE_FPS, TARGET_FPS)
    right_hand_obs_50hz = upsample_linear(right_hand_obs_30hz, SOURCE_FPS, TARGET_FPS)

    # --- Hand action joints (from action field) ---
    # action layout: left_hand(7) + right_hand(7) + left_arm(7) + right_arm(7) = 28D
    # These are the commanded hand targets, remapped to SONIC order
    left_hand_act_30hz = remap_hand_joints(data["action"][:, :7])
    right_hand_act_30hz = remap_hand_joints(data["action"][:, 7:14])
    left_hand_act_50hz = upsample_linear(left_hand_act_30hz, SOURCE_FPS, TARGET_FPS)
    right_hand_act_50hz = upsample_linear(right_hand_act_30hz, SOURCE_FPS, TARGET_FPS)

    # Build 43D observation.state matching deployment format:
    # left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + left_hand(7) + right_arm(7) + right_hand(7)
    joints_43_50hz = np.concatenate([
        joints_29_50hz[:, :22],   # left_leg + right_leg + waist + left_arm
        left_hand_obs_50hz,        # left_hand
        joints_29_50hz[:, 22:29],  # right_arm
        right_hand_obs_50hz,       # right_hand
    ], axis=1)

    # --- Compute FK for eef_state ---
    if robot_model is not None:
        # FK needs joints in URDF order (hands in HE/thumb,index,middle order)
        # Un-remap SONIC hand joints back to URDF order for pinocchio
        left_hand_urdf_50hz = left_hand_obs_50hz[:, HAND_REMAP_INV]
        right_hand_urdf_50hz = right_hand_obs_50hz[:, HAND_REMAP_INV]
        joints_43_urdf = np.concatenate([
            joints_29_50hz, left_hand_urdf_50hz, right_hand_urdf_50hz,
        ], axis=1)
        eef_state_50hz = compute_eef_state(robot_model, joints_43_urdf)
    else:
        T_50 = len(joints_43_50hz)
        eef_state_50hz = np.zeros((T_50, 14), dtype=np.float64)

    # Convert body joints from SONIC/MuJoCo grouped order to IsaacLab interleaved
    # order before encoding — the encoder was trained with IsaacLab joint order.
    # MUJOCO_TO_ISAACLAB: il_array = mj_array[:, MUJOCO_TO_ISAACLAB]
    _MJ_TO_IL = np.array(
        [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
         5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=np.int64)
    joints_29_il_50hz = joints_29_50hz[:, _MJ_TO_IL]
    joint_vels_il_50hz = joint_vels_50hz[:, _MJ_TO_IL]

    # Run ONNX encoder (IsaacLab joint order)
    tokens_50hz = encode_episode(
        encoder_session,
        joints_29_il_50hz,
        joint_vels_il_50hz,
        base_quats_50hz,
    )

    T = len(joints_43_50hz)
    init_base_quat = base_quats_50hz[0]

    # Compute delta_heading: yaw change between consecutive frames
    heading_changes = np.zeros((T, 1), dtype=np.float64)
    for t in range(1, T):
        r_prev = R.from_quat(base_quats_50hz[t - 1][[1, 2, 3, 0]])
        r_curr = R.from_quat(base_quats_50hz[t][[1, 2, 3, 0]])
        euler_prev = r_prev.as_euler("zyx")
        euler_curr = r_curr.as_euler("zyx")
        dyaw = euler_curr[0] - euler_prev[0]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        heading_changes[t, 0] = dyaw

    # Build frames
    frames = []
    for t in range(T):
        proj_grav = compute_projected_gravity(base_quats_50hz[t])

        # action.wbc = next frame's full 43D state (or current for last frame)
        next_t = min(t + 1, T - 1)

        frame = {
            "observation.state": joints_43_50hz[t].astype(np.float64),
            "observation.eef_state": eef_state_50hz[t].astype(np.float64),
            "observation.root_orientation": base_quats_50hz[t].astype(np.float64),
            "observation.projected_gravity": proj_grav.astype(np.float64),
            "observation.cpp_rotation_offset": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "observation.init_base_quat": init_base_quat.astype(np.float64),
            "action.motion_token": tokens_50hz[t].astype(np.float64),
            "action.wbc": joints_43_50hz[next_t].astype(np.float64),
            "teleop.left_hand_joints": left_hand_act_50hz[t].astype(np.float32),
            "teleop.right_hand_joints": right_hand_act_50hz[t].astype(np.float32),
            "teleop.delta_heading": heading_changes[t].astype(np.float64),
        }
        frames.append(frame)

    return {
        "frames": frames,
        "task_description": task_desc,
        "num_frames_50hz": T,
        "source_ep_idx": ep_idx,
        "output_ep_idx": output_ep_idx,
    }


# ---------------------------------------------------------------------------
# Write dataset
# ---------------------------------------------------------------------------


CHUNKS_SIZE = 1000

# Zero-pad teleop fields not available from this dataset
_ZERO_FIELDS = {
    "teleop.smpl_joints": np.zeros(72, dtype=np.float32),
    "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
    "teleop.body_quat_w": np.zeros(4, dtype=np.float32),
    "teleop.target_body_orientation": np.zeros(6, dtype=np.float32),
    "teleop.smpl_frame_index": np.array([0], dtype=np.int64),
    "teleop.left_wrist_joints": np.zeros(3, dtype=np.float32),
    "teleop.right_wrist_joints": np.zeros(3, dtype=np.float32),
    "teleop.stream_mode": np.array([0], dtype=np.int32),
    "teleop.planner_mode": np.array([0], dtype=np.int32),
    "teleop.planner_movement": np.zeros(3, dtype=np.float32),
    "teleop.planner_facing": np.zeros(3, dtype=np.float32),
    "teleop.planner_speed": np.zeros(1, dtype=np.float32),
    "teleop.planner_height": np.zeros(1, dtype=np.float32),
    "teleop.vr_3pt_position": np.zeros(9, dtype=np.float32),
    "teleop.vr_3pt_orientation": np.zeros(18, dtype=np.float32),
}


def write_episode(output_path: Path, result: dict, task_idx: int,
                  total_frames: int, source_dir: str):
    """Write a single episode's parquet + video. Returns episode metadata dict and frame count."""
    import pyarrow as pa

    ep_idx = result["output_ep_idx"]
    chunk = ep_idx // CHUNKS_SIZE
    chunk_dir = output_path / "data" / f"chunk-{chunk:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"

    frames = result["frames"]
    T = len(frames)

    columns = {}
    feature_keys = list(frames[0].keys())
    for key in feature_keys:
        columns[key] = [frames[t][key].tolist() for t in range(T)]

    for key, val in _ZERO_FIELDS.items():
        columns[key] = [val.tolist()] * T

    columns["frame_index"] = list(range(T))
    columns["timestamp"] = [t / TARGET_FPS for t in range(T)]
    columns["episode_index"] = [ep_idx] * T
    columns["index"] = list(range(total_frames, total_frames + T))
    columns["task_index"] = [task_idx] * T
    columns["next.done"] = [False] * (T - 1) + [True]

    table = pa.table(columns)
    pq.write_table(table, parquet_path)

    # Copy/resample video
    src_ep_idx = result["source_ep_idx"]
    src_video = get_video_path(source_dir, src_ep_idx)
    dst_video = output_path / "videos" / f"chunk-{chunk:03d}" / "ego_view" / f"episode_{ep_idx:06d}.mp4"

    if os.path.exists(src_video):
        try:
            copy_or_resample_video(src_video, str(dst_video), TARGET_FPS)
        except subprocess.CalledProcessError as e:
            print(f"Warning: ffmpeg failed for episode {src_ep_idx}: {e.stderr[:200] if e.stderr else ''}")
            os.makedirs(os.path.dirname(dst_video), exist_ok=True)
            shutil.copy2(src_video, dst_video)
    else:
        print(f"Warning: video not found for episode {src_ep_idx}")

    ep_meta = {
        "episode_index": ep_idx,
        "tasks": [task_idx],
        "length": T,
        "dataset_from_index": total_frames,
        "dataset_to_index": total_frames + T - 1,
        "robot_type": "g1",
        "instruction": result["task_description"],
    }
    return ep_meta, T


def write_dataset_meta(
    output_dir: str,
    episodes_meta: list,
    unique_tasks: dict,
    tasks_source: dict,
    total_frames: int,
    total_episodes: int,
):
    """Write metadata files (info.json, modality.json, tasks.jsonl, episodes.jsonl)."""
    output_path = Path(output_dir)
    (output_path / "meta").mkdir(parents=True, exist_ok=True)

    # Write meta/tasks.jsonl
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        for desc, idx in sorted(unique_tasks.items(), key=lambda x: x[1]):
            category = "unknown"
            for orig_task in tasks_source.values():
                if orig_task.get("description") == desc:
                    category = orig_task.get("category", "unknown")
                    break
            json.dump({
                "task_index": idx,
                "task": desc[:80],
                "category": category,
                "description": desc,
            }, f)
            f.write("\n")

    # Write meta/episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            json.dump(ep, f)
            f.write("\n")

    num_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE
    features = {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(TARGET_FPS),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float64",
            "shape": [NUM_FULL_JOINTS],
            "names": _get_full_joint_names(),
        },
        "observation.eef_state": {
            "dtype": "float64",
            "shape": [14],
            "names": ["left_wrist_pos", "left_wrist_abs_quat", "right_wrist_pos", "right_wrist_abs_quat"],
        },
        "observation.root_orientation": {
            "dtype": "float64",
            "shape": [4],
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": [3],
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "observation.cpp_rotation_offset": {
            "dtype": "float64",
            "shape": [4],
            "names": ["rot_offset_qw", "rot_offset_qx", "rot_offset_qy", "rot_offset_qz"],
        },
        "observation.init_base_quat": {
            "dtype": "float64",
            "shape": [4],
            "names": ["init_base_qw", "init_base_qx", "init_base_qy", "init_base_qz"],
        },
        "action.motion_token": {
            "dtype": "float64",
            "shape": [64],
            "names": "motion_token",
        },
        "action.wbc": {
            "dtype": "float64",
            "shape": [NUM_FULL_JOINTS],
            "names": _get_full_joint_names(),
        },
        "teleop.delta_heading": {
            "dtype": "float64",
            "shape": [1],
            "names": ["delta_heading"],
        },
        "teleop.left_hand_joints": {
            "dtype": "float32",
            "shape": [7],
            "names": "left_hand_joints",
        },
        "teleop.right_hand_joints": {
            "dtype": "float32",
            "shape": [7],
            "names": "right_hand_joints",
        },
        "teleop.smpl_joints": {"dtype": "float32", "shape": [72], "names": "smpl_joints"},
        "teleop.smpl_pose": {"dtype": "float32", "shape": [63], "names": "smpl_pose"},
        "teleop.body_quat_w": {"dtype": "float32", "shape": [4], "names": "body_quat_w"},
        "teleop.target_body_orientation": {"dtype": "float32", "shape": [6], "names": [
            "target_body_r00", "target_body_r10", "target_body_r01",
            "target_body_r11", "target_body_r02", "target_body_r12",
        ]},
        "teleop.smpl_frame_index": {"dtype": "int64", "shape": [1], "names": ["smpl_frame_index"]},
        "teleop.left_wrist_joints": {"dtype": "float32", "shape": [3], "names": [
            "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        ]},
        "teleop.right_wrist_joints": {"dtype": "float32", "shape": [3], "names": [
            "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        ]},
        "teleop.stream_mode": {"dtype": "int32", "shape": [1], "names": ["stream_mode"]},
        "teleop.planner_mode": {"dtype": "int32", "shape": [1], "names": ["locomotion_mode"]},
        "teleop.planner_movement": {"dtype": "float32", "shape": [3], "names": [
            "movement_x", "movement_y", "movement_z",
        ]},
        "teleop.planner_facing": {"dtype": "float32", "shape": [3], "names": [
            "facing_x", "facing_y", "facing_z",
        ]},
        "teleop.planner_speed": {"dtype": "float32", "shape": [1], "names": ["speed"]},
        "teleop.planner_height": {"dtype": "float32", "shape": [1], "names": ["height"]},
        "teleop.vr_3pt_position": {"dtype": "float32", "shape": [9], "names": [
            "lwrist_x", "lwrist_y", "lwrist_z",
            "rwrist_x", "rwrist_y", "rwrist_z",
            "neck_x", "neck_y", "neck_z",
        ]},
        "teleop.vr_3pt_orientation": {"dtype": "float32", "shape": [18], "names": [
            "lwrist_r00", "lwrist_r10", "lwrist_r01", "lwrist_r11", "lwrist_r02", "lwrist_r12",
            "rwrist_r00", "rwrist_r10", "rwrist_r01", "rwrist_r11", "rwrist_r02", "rwrist_r12",
            "neck_r00", "neck_r10", "neck_r01", "neck_r11", "neck_r02", "neck_r12",
        ]},
        # Standard LeRobot index columns
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }

    # Write meta/info.json
    info = {
        "codebase_version": "2.1",
        "robot_type": "g1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks),
        "total_videos": total_episodes,
        "total_chunks": num_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": TARGET_FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/ego_view/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Write meta/modality.json
    modality = _build_modality_config()
    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    print(f"\nDataset metadata written to {output_dir}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Tasks: {len(unique_tasks)}")


def _get_full_joint_names():
    """Return SONIC 43-DOF joint names: body(29) + left_hand(7) + right_hand(7) in SONIC order."""
    return [
        # Left leg (6)
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        # Right leg (6)
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        # Waist (3)
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        # Left arm (7)
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        # Left hand (7, SONIC order: index, middle, thumb)
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        # Right arm (7)
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        # Right hand (7, SONIC order: index, middle, thumb)
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    ]


def _build_modality_config():
    """Build modality.json matching SONIC VLA schema."""
    # Joint group slices matching the 43-DOF order:
    # left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + left_hand(7) + right_arm(7) + right_hand(7)
    return {
        "state": {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
            "left_wrist_pos": {"start": 0, "end": 3, "original_key": "observation.eef_state"},
            "left_wrist_abs_quat": {"start": 3, "end": 7, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
            "right_wrist_pos": {"start": 7, "end": 10, "original_key": "observation.eef_state"},
            "right_wrist_abs_quat": {"start": 10, "end": 14, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
            "root_orientation": {"start": 0, "end": 4, "original_key": "observation.root_orientation", "rotation_type": "quaternion"},
            "projected_gravity": {"start": 0, "end": 3, "original_key": "observation.projected_gravity"},
            "cpp_rotation_offset": {"start": 0, "end": 4, "original_key": "observation.cpp_rotation_offset", "rotation_type": "quaternion"},
            "init_base_quat": {"start": 0, "end": 4, "original_key": "observation.init_base_quat", "rotation_type": "quaternion"},
        },
        "action": {
            "delta_heading": {"start": 0, "end": 1, "original_key": "teleop.delta_heading"},
            "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
            "smpl_joints": {"start": 0, "end": 72, "original_key": "teleop.smpl_joints"},
            "smpl_pose": {"start": 0, "end": 63, "original_key": "teleop.smpl_pose"},
            "body_quat_w": {"start": 0, "end": 4, "original_key": "teleop.body_quat_w", "rotation_type": "quaternion"},
            "target_body_orientation": {"start": 0, "end": 6, "original_key": "teleop.target_body_orientation", "rotation_type": "rotation_6d"},
            "left_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.left_hand_joints"},
            "right_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.right_hand_joints"},
            "left_wrist_joints": {"start": 0, "end": 3, "original_key": "teleop.left_wrist_joints"},
            "right_wrist_joints": {"start": 0, "end": 3, "original_key": "teleop.right_wrist_joints"},
            "stream_mode": {"start": 0, "end": 1, "original_key": "teleop.stream_mode"},
            "planner_mode": {"start": 0, "end": 1, "original_key": "teleop.planner_mode"},
            "planner_movement": {"start": 0, "end": 3, "original_key": "teleop.planner_movement"},
            "planner_facing": {"start": 0, "end": 3, "original_key": "teleop.planner_facing"},
            "planner_speed": {"start": 0, "end": 1, "original_key": "teleop.planner_speed"},
            "planner_height": {"start": 0, "end": 1, "original_key": "teleop.planner_height"},
            "vr_3pt_position": {"start": 0, "end": 9, "original_key": "teleop.vr_3pt_position"},
            "vr_3pt_orientation": {"start": 0, "end": 18, "original_key": "teleop.vr_3pt_orientation", "rotation_type": "rotation_6d"},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


# ---------------------------------------------------------------------------
# Download encoder
# ---------------------------------------------------------------------------


def download_encoder(model_dir: str) -> str:
    """Download SONIC encoder ONNX model from nvidia/GEAR-SONIC on HuggingFace.

    Saves to <model_dir>/model_encoder.onnx (default: <script_dir>/encoder/).
    Skips download if the file already exists.
    """
    model_path = os.path.join(model_dir, "model_encoder.onnx")
    if os.path.exists(model_path):
        print(f"Encoder already exists: {model_path}")
        return model_path

    print("Downloading SONIC encoder from HuggingFace...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        "nvidia/GEAR-SONIC", "model_encoder.onnx",
        local_dir=model_dir,
    )
    print(f"Downloaded encoder to: {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------


def _init_models(encoder_path):
    """Load encoder and robot model (called once per worker)."""
    encoder_session = load_encoder(encoder_path)
    _urdf_path = os.path.join(_SCRIPT_DIR, "g1_model_data", "g1_29dof_with_hand.urdf")
    try:
        robot_model = G1RobotModel(_urdf_path)
    except Exception:
        robot_model = None
    return encoder_session, robot_model


def _convert_sequential(args, episodes, tasks, encoder_path):
    """Convert episodes sequentially, writing each to disk immediately."""
    encoder_session, robot_model = _init_models(encoder_path)
    print("Loading ONNX encoder...")
    inp = encoder_session.get_inputs()[0]
    out = encoder_session.get_outputs()[0]
    print(f"  Input: {inp.name} shape={inp.shape}")
    print(f"  Output: {out.name} shape={out.shape}")
    if robot_model:
        print(f"  Robot model loaded: {robot_model.num_dofs} DOFs")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unique_tasks = {}
    task_counter = 0
    episodes_meta = []
    total_frames = 0

    for i, episode in enumerate(tqdm(episodes, desc="Converting episodes")):
        output_ep_idx = args.start_episode + i
        try:
            result = convert_episode(
                args.source_dir, episode, tasks, encoder_session, output_ep_idx,
                robot_model=robot_model,
            )
        except Exception as e:
            print(f"\nError converting episode {episode['episode_index']}: {e}")
            import traceback
            traceback.print_exc()
            continue

        desc = result["task_description"]
        if desc not in unique_tasks:
            unique_tasks[desc] = task_counter
            task_counter += 1
        task_idx = unique_tasks[desc]

        ep_meta, num_frames = write_episode(
            output_path, result, task_idx, total_frames, args.source_dir,
        )
        episodes_meta.append(ep_meta)
        total_frames += num_frames

    return episodes_meta, unique_tasks, total_frames


# Global worker state (initialized once per process via pool initializer)
_worker_encoder = None
_worker_robot_model = None


def _worker_init(encoder_path):
    """Pool initializer: load models once per worker process."""
    global _worker_encoder, _worker_robot_model
    _worker_encoder, _worker_robot_model = _init_models(encoder_path)


def _worker_convert(job):
    """Convert a single episode in a worker process."""
    source_dir, episode, tasks, output_ep_idx = job
    return convert_episode(
        source_dir, episode, tasks, _worker_encoder, output_ep_idx,
        robot_model=_worker_robot_model,
    )


def _convert_parallel(args, episodes, tasks, encoder_path):
    """Convert episodes in parallel, writing each to disk as it completes."""
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    from multiprocessing import Pool

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jobs = [
        (args.source_dir, ep, tasks, args.start_episode + i)
        for i, ep in enumerate(episodes)
    ]

    unique_tasks = {}
    task_counter = 0
    episodes_meta = []
    total_frames = 0

    print(f"Converting {len(jobs)} episodes with {args.num_workers} workers...")
    with Pool(args.num_workers, initializer=_worker_init, initargs=(encoder_path,)) as pool:
        for result in tqdm(
            pool.imap(_worker_convert, jobs),
            total=len(jobs),
            desc="Converting episodes",
        ):
            if result is None:
                continue

            desc = result["task_description"]
            if desc not in unique_tasks:
                unique_tasks[desc] = task_counter
                task_counter += 1
            task_idx = unique_tasks[desc]

            ep_meta, num_frames = write_episode(
                output_path, result, task_idx, total_frames, args.source_dir,
            )
            episodes_meta.append(ep_meta)
            total_frames += num_frames

    return episodes_meta, unique_tasks, total_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert Humanoid Everyday to SONIC VLA format")
    parser.add_argument("--source-dir", required=True, help="Path to humanoid_everyday dataset")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--subset", choices=["locomanip", "all"], default="locomanip",
                        help="Which episodes to include")
    parser.add_argument("--encoder-dir", default=os.path.join(_SCRIPT_DIR, "encoder"),
                        help="Directory for ONNX encoder model (auto-downloaded)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes (for testing)")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="Start from this episode index (for resuming)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers for episode conversion")
    args = parser.parse_args()

    # Download encoder
    encoder_path = download_encoder(args.encoder_dir)

    # Load episodes
    episodes, tasks = load_source_episodes(args.source_dir, args.subset)

    if args.max_episodes:
        episodes = episodes[:args.start_episode + args.max_episodes]
    episodes = episodes[args.start_episode:]

    if args.num_workers > 1:
        episodes_meta, unique_tasks, total_frames = _convert_parallel(args, episodes, tasks, encoder_path)
    else:
        episodes_meta, unique_tasks, total_frames = _convert_sequential(args, episodes, tasks, encoder_path)

    # Write metadata (parquet + video already written incrementally)
    write_dataset_meta(
        args.output_dir, episodes_meta, unique_tasks, tasks,
        total_frames, len(episodes_meta),
    )

    print(f"\n--- Done: {len(episodes_meta)} episodes, {total_frames} frames ---")


if __name__ == "__main__":
    main()
