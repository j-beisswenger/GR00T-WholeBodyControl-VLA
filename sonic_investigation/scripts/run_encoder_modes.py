"""Run the released SONIC encoder ONNX in each of its 3 modes.

The deploy encoder is a single ONNX (1762 -> 64). The active mode is signaled
via the first 4 entries (`encoder_mode_4`); inputs not used by a mode are
zero-filled. Layout below mirrors the order in
`gear_sonic_deploy/policy/release/observation_config.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

# (name, dim) in deploy YAML order. Offsets derived by cumulative sum.
# Two slots (motion_root_z_position*) are intentionally always zero — no deploy
# mode lists them in `required_observations`, and the deploy C++ also leaves
# them zero (g1_deploy_onnx_ref.cpp:2008,2089).
LAYOUT: list[tuple[str, int]] = [
    ("encoder_mode_4",                                   4),
    ("motion_joint_positions_10frame_step5",           290),
    ("motion_joint_velocities_10frame_step5",          290),
    ("motion_root_z_position_10frame_step5",            10),  # always zero in all 3 modes
    ("motion_root_z_position",                           1),  # always zero in all 3 modes
    ("motion_anchor_orientation",                        6),
    ("motion_anchor_orientation_10frame_step5",         60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5",120),
    ("vr_3point_local_target",                           9),
    ("vr_3point_local_orn_target",                      12),
    ("smpl_joints_10frame_step1",                      720),
    ("smpl_anchor_orientation_10frame_step1",           60),
    ("motion_joint_positions_wrists_10frame_step1",     60),
]
TOTAL_DIM = sum(d for _, d in LAYOUT)  # 1762

_offset, SLICES = 0, {}
for _name, _d in LAYOUT:
    SLICES[_name] = slice(_offset, _offset + _d)
    _offset += _d


def _pack(parts: dict[str, np.ndarray]) -> np.ndarray:
    """Place each named array at its slot; everything else stays zero."""
    x = np.zeros(TOTAL_DIM, dtype=np.float32)
    for name, arr in parts.items():
        sl = SLICES[name]
        if arr.size != sl.stop - sl.start:
            raise ValueError(f"{name}: got {arr.size}, expected {sl.stop - sl.start}")
        x[sl] = arr.astype(np.float32, copy=False)
    return x


def _mode_id(mode_id: int) -> np.ndarray:
    return np.array([mode_id, 0, 0, 0], dtype=np.float32)


def _check(arr: np.ndarray, expected: tuple[int, ...], name: str) -> np.ndarray:
    """Assert array shape (not just size) — guards against e.g. (29, 10) vs (10, 29)."""
    if arr.shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {arr.shape}")
    return arr


def _infer(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "obs_dict":
        raise RuntimeError(
            f"Expected single 'obs_dict' input, got {[i.name for i in inputs]}"
        )
    return session.run(None, {inputs[0].name: x[None, :]})[0][0]  # (64,)


# --- per-mode entry points -------------------------------------------------

def run_g1(
    session: ort.InferenceSession,
    motion_joint_pos_10x29: np.ndarray,        # (10, 29) -> 290, IsaacLab joint order
    motion_joint_vel_10x29: np.ndarray,        # (10, 29) -> 290, IsaacLab joint order
    motion_anchor_ori_10x6: np.ndarray,        # (10, 6)  -> 60
) -> np.ndarray:
    _check(motion_joint_pos_10x29,  (10, 29), "motion_joint_pos_10x29")
    _check(motion_joint_vel_10x29,  (10, 29), "motion_joint_vel_10x29")
    _check(motion_anchor_ori_10x6,  (10, 6),  "motion_anchor_ori_10x6")
    x = _pack({
        "encoder_mode_4":                            _mode_id(0),
        "motion_joint_positions_10frame_step5":      motion_joint_pos_10x29.reshape(-1),
        "motion_joint_velocities_10frame_step5":     motion_joint_vel_10x29.reshape(-1),
        "motion_anchor_orientation_10frame_step5":   motion_anchor_ori_10x6.reshape(-1),
    })
    return _infer(session, x)


def run_teleop(
    session: ort.InferenceSession,
    motion_joint_pos_lower_10x12: np.ndarray,  # (10, 12) -> 120, see joint-order note below
    motion_joint_vel_lower_10x12: np.ndarray,  # (10, 12) -> 120, same order as positions
    vr_3point_pos_3x3: np.ndarray,             # (3, 3)   -> 9    [Lwrist, Rwrist, head] xyz
    vr_3point_orn_3x4: np.ndarray,             # (3, 4)   -> 12   quat wxyz, same point order
    motion_anchor_ori_6: np.ndarray,           # (6,)     -> 6
) -> np.ndarray:
    """
    Lower-body joint ordering: deploy uses MUJOCO order, *not* IsaacLab order.
    The 12 joints are picked via `lower_body_joint_mujoco_order_in_isaaclab_index`
    = {0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18}  (g1_deploy_onnx_ref.cpp:1735-1738,
    policy_parameters.hpp:92). If you have a 29-vector in IsaacLab order, slice
    with that index list — do NOT use the natural slice {0,1,3,4,6,7,9,10,13,14,17,18}.
    """
    _check(motion_joint_pos_lower_10x12, (10, 12), "motion_joint_pos_lower_10x12")
    _check(motion_joint_vel_lower_10x12, (10, 12), "motion_joint_vel_lower_10x12")
    _check(vr_3point_pos_3x3,            (3, 3),   "vr_3point_pos_3x3")
    _check(vr_3point_orn_3x4,            (3, 4),   "vr_3point_orn_3x4")
    _check(motion_anchor_ori_6,          (6,),     "motion_anchor_ori_6")
    x = _pack({
        "encoder_mode_4":                                  _mode_id(1),
        "motion_joint_positions_lowerbody_10frame_step5":  motion_joint_pos_lower_10x12.reshape(-1),
        "motion_joint_velocities_lowerbody_10frame_step5": motion_joint_vel_lower_10x12.reshape(-1),
        "vr_3point_local_target":                          vr_3point_pos_3x3.reshape(-1),
        "vr_3point_local_orn_target":                      vr_3point_orn_3x4.reshape(-1),
        "motion_anchor_orientation":                       motion_anchor_ori_6.reshape(-1),
    })
    return _infer(session, x)


def run_smpl(
    session: ort.InferenceSession,
    smpl_joints_10x24x3: np.ndarray,           # (10, 24, 3) -> 720, SMPL joint order
    smpl_anchor_ori_10x6: np.ndarray,          # (10, 6)     -> 60
    wrist_joint_pos_10x6: np.ndarray,          # (10, 6)     -> 60, IsaacLab idx 23..28
) -> np.ndarray:
    """
    Wrist joints are IsaacLab indices {23,24,25,26,27,28} (natural consecutive
    slice — `wrist_joint_isaaclab_order_in_isaaclab_index`,
    g1_deploy_onnx_ref.cpp:1739, policy_parameters.hpp:88).
    """
    _check(smpl_joints_10x24x3,  (10, 24, 3), "smpl_joints_10x24x3")
    _check(smpl_anchor_ori_10x6, (10, 6),     "smpl_anchor_ori_10x6")
    _check(wrist_joint_pos_10x6, (10, 6),     "wrist_joint_pos_10x6")
    x = _pack({
        "encoder_mode_4":                              _mode_id(2),
        "smpl_joints_10frame_step1":                   smpl_joints_10x24x3.reshape(-1),
        "smpl_anchor_orientation_10frame_step1":       smpl_anchor_ori_10x6.reshape(-1),
        "motion_joint_positions_wrists_10frame_step1": wrist_joint_pos_10x6.reshape(-1),
    })
    return _infer(session, x)


ENCODER_PATH = (
    Path(__file__).resolve().parents[2]
    / "gear_sonic_deploy/policy/release/model_encoder.onnx"
)


if __name__ == "__main__":
    sess = ort.InferenceSession(str(ENCODER_PATH), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    tok_g1 = run_g1(
        sess,
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 6)),
    )
    tok_teleop = run_teleop(
        sess,
        rng.standard_normal((10, 12)),
        rng.standard_normal((10, 12)),
        rng.standard_normal((3, 3)),
        rng.standard_normal((3, 4)),
        rng.standard_normal((6,)),
    )
    tok_smpl = run_smpl(
        sess,
        rng.standard_normal((10, 24, 3)),
        rng.standard_normal((10, 6)),
        rng.standard_normal((10, 6)),
    )
    print("g1     token_state:", tok_g1.shape, tok_g1[:4])
    print("teleop token_state:", tok_teleop.shape, tok_teleop[:4])
    print("smpl   token_state:", tok_smpl.shape, tok_smpl[:4])
