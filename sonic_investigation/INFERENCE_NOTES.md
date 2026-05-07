# GEAR-SONIC Inference Notes

General reference for how the released SONIC ONNX trio (encoder + decoder +
planner) is wired in the C++ deploy stack
[`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`](../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/).
All shape and layout claims here are verified against that source.

The Python scripts that exercise these models live in [`scripts/`](scripts/):

| Script | What it does |
|---|---|
| [`sonic_encoder_onnx_3modes.py`](scripts/sonic_encoder_onnx_3modes.py) | Single-shot ONNX encoder run in g1/teleop/smpl modes (1762 → 64). |
| [`sonic_decoder_onnx.py`](scripts/sonic_decoder_onnx.py) | Single-shot ONNX decoder run (994 → 29). |
| [`sonic_encoder_decoder_pytorch_unquantized.py`](scripts/sonic_encoder_decoder_pytorch_unquantized.py) | PyTorch counterparts of both, with optional `quantize=False` to bypass FSQ. |
| [`mujoco_viewer_synthetic_idle_encoder_decoder.py`](scripts/mujoco_viewer_synthetic_idle_encoder_decoder.py) | End-to-end IDLE rollout in MuJoCo with viewer (synthetic stand-still reference + encoder + decoder + PD). |
| [`mujoco_viewer_csv_motion_g1_encoder_decoder.py`](scripts/mujoco_viewer_csv_motion_g1_encoder_decoder.py) | MuJoCo viewer playback of a recorded CSV reference motion through the G1/robot encoder. |
| [`mujoco_viewer_amass_smpl_encoder_decoder.py`](scripts/mujoco_viewer_amass_smpl_encoder_decoder.py) | MuJoCo viewer playback of an AMASS .npz through the SMPL/human encoder. |
| [`mujoco_viewer_amass_gmr_g1_encoder_decoder.py`](scripts/mujoco_viewer_amass_gmr_g1_encoder_decoder.py) | MuJoCo viewer playback of an AMASS .npz retargeted via GMR through the G1/robot encoder. |
| [`plot_amass_smpl_vs_g1_encoder_tokens.py`](scripts/plot_amass_smpl_vs_g1_encoder_tokens.py) | Overlay 64-D encoder tokens from the SMPL and G1 paths across random AMASS clips. |

---

## 1. The model

SONIC is a motion-tracking foundation model for the Unitree G1. Three ONNX
components ship in the HuggingFace release `nvidia/GEAR-SONIC`:

| ONNX file | Role | Input | Output |
|---|---|---|---|
| [`model_encoder.onnx`](../gear_sonic_deploy/policy/release/model_encoder.onnx) | Hybrid encoder → 64-D universal token | `obs_dict [1, 1762]` | `encoded_tokens [1, 64]` |
| [`model_decoder.onnx`](../gear_sonic_deploy/policy/release/model_decoder.onnx) | Tracking policy (control decoder) | `obs_dict [1, 994]` | `action [1, 29]` (joint targets, rad) |
| [`planner_sonic.onnx`](../gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx) | Kinematic planner (only used at runtime when there's no pre-recorded motion) | `target_vel, mode, movement_direction, facing_direction, …` | `mujoco_qpos [1, 64, 36]` + `num_pred_frames [1]` |

**The token is quantized.** Encoder outputs are multiples of 1/16 = 0.0625 per
dim. Each dim is effectively a 5-bit code. So the "universal token" is a 64-dim
discrete latent, not a continuous embedding.

`mujoco_viewer_synthetic_idle_encoder_decoder.py` uses both ONNX components plus
MuJoCo physics. The standalone `sonic_encoder_onnx_3modes.py` / `sonic_decoder_onnx.py`
exercise the encoder/decoder in isolation with synthetic inputs.

---

## 2. Control rates

From [`g1_deploy_onnx_ref.cpp:2015-2016`](../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp#L2015-L2016):

```cpp
control_dt_(0.02),  // 50 Hz control loop
planner_dt_(0.1),   // 10 Hz planner loop
```

Per second of operation:

- **50× encoder forward passes** (sequential with policy — its output is a policy input)
- **50× decoder forward passes** (produces motor commands)
- **10× planner forward passes** (generates 0.8–2.4 s of reference motion, output length chosen by a learned head)

In my scripts I run the encoder+decoder pair at 50 Hz exactly like the C++ loop.
I do not run the planner — I feed motion CSV frames directly in its place.

---

## 3. Encoder input — 1762 dims

Order is set by [`observation_config.yaml`](../gear_sonic_deploy/policy/release/observation_config.yaml)
and the registry at [`g1_deploy_onnx_ref.cpp:1578-1620`](../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp#L1578-L1620).

| # | Field | Dim | What it is |
|---|---|---|---|
| 1 | `encoder_mode_4` | 4 | One-hot-ish mode: `[mode_id, 0, 0, 0]`. `mode_id=0` for G1 retarget, 1 for teleop, 2 for SMPL. |
| 2 | `motion_joint_positions_10frame_step5` | 290 | 10 future motion frames, every 5th tick → 10 × 29 joints. |
| 3 | `motion_joint_velocities_10frame_step5` | 290 | Same 10 frames, 29 joint velocities. |
| 4 | `motion_root_z_position_10frame_step5` | 10 | Root z-height at those 10 future frames. |
| 5 | `motion_root_z_position` | 1 | Current root z-height. |
| 6 | `motion_anchor_orientation` | 6 | First 2 cols of 3×3 rotmat (row-major) of `inv(base_quat) · ref_root_quat`, current frame. |
| 7 | `motion_anchor_orientation_10frame_step5` | 60 | Same 6-D anchor, for 10 future frames. |
| 8 | `motion_joint_positions_lowerbody_10frame_step5` | 120 | 10 frames × 12 lower-body joints (IsaacLab idx `[0,3,6,9,13,17,1,4,7,10,14,18]`). |
| 9 | `motion_joint_velocities_lowerbody_10frame_step5` | 120 | Same 12 joints, velocities. |
| 10 | `vr_3point_local_target` | 9 | Teleop: 3 VR points × xyz. **Zero in G1 mode** (script fills zeros). |
| 11 | `vr_3point_local_orn_target` | 12 | Teleop: 3 quats (wxyz). **Zero in G1 mode**. |
| 12 | `smpl_joints_10frame_step1` | 720 | SMPL mode: 10 frames × 24 SMPL joints × xyz. **Zero in G1 mode**. |
| 13 | `smpl_anchor_orientation_10frame_step1` | 60 | SMPL mode: same anchor formula on SMPL. **Zero in G1 mode**. |
| 14 | `motion_joint_positions_wrists_10frame_step1` | 60 | 10 frames × 6 wrist joints (IsaacLab idx `[23..28]`). |

Total: 4 + 290 + 290 + 10 + 1 + 6 + 60 + 120 + 120 + 9 + 12 + 720 + 60 + 60 = **1762** ✓

**Lookahead horizon.** The longest look-ahead is 10 × 5 = 50 ticks ≈ 1 s. When
`t + k·step` exceeds motion length, indices are clamped to the last frame.

---

## 4. Decoder input — 994 dims

Same YAML order. All `his_*` fields are *past* history ending at the current tick.

| # | Field | Dim | What it is |
|---|---|---|---|
| 1 | `token_state` | 64 | Encoder output from this same tick. |
| 2 | `his_base_angular_velocity_10frame_step1` | 30 | 10 past ticks × 3 body-angular-vel. |
| 3 | `his_body_joint_positions_10frame_step1` | 290 | 10 past ticks × 29 joint **deviations** `q − default_angles` (IsaacLab order). See gotcha below. |
| 4 | `his_body_joint_velocities_10frame_step1` | 290 | 10 past ticks × 29 robot joint velocities (raw). |
| 5 | `his_last_actions_10frame_step1` | 290 | 10 past *policy outputs* (closed-loop on the decoder itself). |
| 6 | `his_gravity_dir_10frame_step1` | 30 | 10 past ticks × 3. Gravity `[0,0,-1]` rotated into body frame via `quat_rotate(conj(base_quat), g)`. |

Total: 64 + 30 + 290 + 290 + 290 + 30 = **994** ✓

**History horizon.** 10 ticks × 1 step ≈ 0.2 s of past state/actions.

> ⚠ **Joint-position offset gotcha (decoder side).** Row 3 is *not* raw joint
> angles. The deploy logs `body_q[i] = motor_q[mujoco_to_isaaclab[i]] -
> default_angles[mujoco_to_isaaclab[i]]` (`g1_deploy_onnx_ref.cpp:2827`) and
> `GatherHisBodyJointPositions` pipes that straight into the decoder. The
> *encoder* side (`motion_joint_positions_*`) is the opposite — fed raw, no
> offset (`localmotion_kplanner.hpp:494`). Mixing these up makes the decoder
> see a huge initial pose error at default stance and squat/fall the robot.

---

## 5. What `mujoco_viewer_synthetic_idle_encoder_decoder.py` does

Closed-loop IDLE rollout of planner + encoder + decoder on top of MuJoCo:

```
qpos_30hz ← planner.run(IDLE, identity context, current qpos)        # once
motion    ← resample to 50 Hz, joints in IsaacLab order               # once

for t in 0..∞:
    state    ← read MuJoCo qpos / qvel / base_quat
    history  ← roll(history); history[-1] ← (q − default_angles, dq, last_action, ω, gravity_in_body)
    enc_x    ← build 1762-D from motion[t…t+45 step 5]
    token    ← encoder.run(enc_x)                                     # (64,) quantized
    pol_x    ← build 994-D from token + history
    action   ← decoder.run(pol_x)                                     # (29,) IsaacLab-order
    q_target ← default_angles + action[isaaclab_to_mujoco] · scale    # MJ-order PD target
    for 4 sim sub-steps: data.ctrl ← clip(kp·(q_target−q) + kd·(−dq))
                          mj_step(model, data)
```

50 Hz control, 200 Hz physics (4 sub-steps), planner runs **once** at startup
because `is_static_motion_mode(IDLE)` short-circuits the replan in the deploy
(`g1_deploy_onnx_ref.cpp:3692`). Observation and gather rules match
[`g1_deploy_onnx_ref.cpp:380-1500`](../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp#L380).

### Run

```bash
.venv_sim/bin/python sonic_investigation/scripts/mujoco_viewer_synthetic_idle_encoder_decoder.py
```

(See also the wrapper script `mujoco_viewer_synthetic_idle_encoder_decoder.sh`.)

### Simplifications vs. the real control stack

Skipped for clarity: input/joystick thread, ZMQ/ROS2 plumbing, hand controllers,
motion recording, multi-motion blend, `apply_delta_heading` frame correction
(safe to drop in IDLE because robot and motion both start at identity yaw).

---

## 6. Where to go from here

- **Other locomotion modes.** Change `mode` (currently 0 = IDLE) and add a
  10 Hz re-plan loop — the deploy replans at 10 Hz for non-static modes.
- **Wire up the real pipeline.** The C++ deploy handles real-time threading,
  TensorRT acceleration, and hardware I/O. Use `gear_sonic_deploy/deploy.sh`
  for end-to-end evaluation.
