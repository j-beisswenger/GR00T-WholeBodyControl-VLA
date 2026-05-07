"""Run the released SONIC action decoder ONNX.

The deploy decoder is a single ONNX (994 -> 29). Unlike the encoder it has no
mode switching: it consumes the encoder's 64D `token_state` plus 930D of robot
proprioception/action history, and emits 29D normalised joint actions.

Layout below mirrors the order in
`gear_sonic_deploy/policy/release/observation_config.yaml` `observations:` block
(NOT the `encoder_observations` list — that's the encoder side).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

# (name, dim) in deploy YAML order. Offsets derived by cumulative sum.
LAYOUT: list[tuple[str, int]] = [
    ("token_state",                              64),  # encoder output (`encoded_tokens`)
    ("his_base_angular_velocity_10frame_step1",  30),  # IMU ω (3) × 10 ticks
    ("his_body_joint_positions_10frame_step1",  290),  # 29 joints × 10 ticks
    ("his_body_joint_velocities_10frame_step1", 290),  # 29 joints × 10 ticks
    ("his_last_actions_10frame_step1",          290),  # 29 actions × 10 ticks
    ("his_gravity_dir_10frame_step1",            30),  # gravity dir (3) × 10 ticks
]
TOTAL_DIM = sum(d for _, d in LAYOUT)  # 994
ACTION_DIM = 29

_offset, SLICES = 0, {}
for _name, _d in LAYOUT:
    SLICES[_name] = slice(_offset, _offset + _d)
    _offset += _d


def _pack(parts: dict[str, np.ndarray]) -> np.ndarray:
    """Place each named array at its slot. All names must be present (no zero defaults)."""
    if set(parts) != set(SLICES):
        missing = set(SLICES) - set(parts)
        extra = set(parts) - set(SLICES)
        raise ValueError(f"missing={sorted(missing)} extra={sorted(extra)}")
    x = np.empty(TOTAL_DIM, dtype=np.float32)
    for name, arr in parts.items():
        sl = SLICES[name]
        if arr.size != sl.stop - sl.start:
            raise ValueError(f"{name}: got {arr.size}, expected {sl.stop - sl.start}")
        x[sl] = arr.astype(np.float32, copy=False)
    return x


def _check(arr: np.ndarray, expected: tuple[int, ...], name: str) -> np.ndarray:
    """Assert array shape (not just size)."""
    if arr.shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {arr.shape}")
    return arr


def _infer(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "obs_dict":
        raise RuntimeError(
            f"Expected single 'obs_dict' input, got {[i.name for i in inputs]}"
        )
    return session.run(None, {inputs[0].name: x[None, :]})[0][0]  # (29,)


# --- entry point -----------------------------------------------------------

def run_decoder(
    session: ort.InferenceSession,
    token_state_64: np.ndarray,                # (64,)        from encoder output
    his_base_ang_vel_10x3: np.ndarray,         # (10, 3)      most-recent first or last? see note
    his_body_joint_pos_10x29: np.ndarray,      # (10, 29)     IsaacLab order, q − default_angles (NOT raw)
    his_body_joint_vel_10x29: np.ndarray,      # (10, 29)     IsaacLab order, raw velocities
    his_last_actions_10x29: np.ndarray,        # (10, 29)     IsaacLab order, raw decoder outputs
    his_gravity_dir_10x3: np.ndarray,          # (10, 3)      body frame
) -> np.ndarray:
    """
    Inputs are 10-frame `_step1` history windows: 10 consecutive 50 Hz ticks
    (=0.2 s). The deploy gatherer for `his_*` reads from a ring buffer
    (`state_logger_->GetLatest(N)`); see `g1_deploy_onnx_ref.cpp` `GatherHis*`
    for exact frame ordering. Mirror that order when constructing inputs.

    Reshape is C-order (frames-major): index `(t, j)` lands at flat slot
    `t * J + j`, matching the deploy multi-frame pack pattern.

    ⚠ `his_body_joint_pos_10x29` must be `q − default_angles` in IsaacLab order,
    not raw `q`. The deploy logs this offset at `g1_deploy_onnx_ref.cpp:2827`:
        body_q[i] = motor_q[mujoco_to_isaaclab[i]] - default_angles[mujoco_to_isaaclab[i]]
    Forgetting it makes the decoder see a huge initial pose error at default
    stance. Note this is the *opposite* convention from the encoder's
    `motion_joint_positions_*` input, which is fed raw.
    """
    _check(token_state_64,           (64,),    "token_state_64")
    _check(his_base_ang_vel_10x3,    (10, 3),  "his_base_ang_vel_10x3")
    _check(his_body_joint_pos_10x29, (10, 29), "his_body_joint_pos_10x29")
    _check(his_body_joint_vel_10x29, (10, 29), "his_body_joint_vel_10x29")
    _check(his_last_actions_10x29,   (10, 29), "his_last_actions_10x29")
    _check(his_gravity_dir_10x3,     (10, 3),  "his_gravity_dir_10x3")
    x = _pack({
        "token_state":                              token_state_64,
        "his_base_angular_velocity_10frame_step1":  his_base_ang_vel_10x3.reshape(-1),
        "his_body_joint_positions_10frame_step1":   his_body_joint_pos_10x29.reshape(-1),
        "his_body_joint_velocities_10frame_step1":  his_body_joint_vel_10x29.reshape(-1),
        "his_last_actions_10frame_step1":           his_last_actions_10x29.reshape(-1),
        "his_gravity_dir_10frame_step1":            his_gravity_dir_10x3.reshape(-1),
    })
    return _infer(session, x)


DECODER_PATH = (
    Path(__file__).resolve().parents[2]
    / "gear_sonic_deploy/policy/release/model_decoder.onnx"
)


if __name__ == "__main__":
    sess = ort.InferenceSession(str(DECODER_PATH), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    action = run_decoder(
        sess,
        rng.standard_normal((64,)),
        rng.standard_normal((10, 3)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 3)),
    )
    print("decoder action:", action.shape, action[:8], "...")
