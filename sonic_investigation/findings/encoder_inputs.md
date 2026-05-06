# SONIC Encoder Inputs — by mode

## TL;DR

At deploy, the encoder is **a single ONNX model**, not three. It always consumes the superset of every mode's observations (1762D in the released config) and emits a 64D `token_state`. "Modes" are realised by zero-filling the observations a given mode doesn't use, with the active mode signaled to the network via `encoder_mode_4`.

On the training side, `UniversalTokenModule` does build *separate* per-mode MLP encoders (g1 / teleop / smpl). For deployment those per-mode pathways are packed into one ONNX whose input is the union of all per-mode inputs.

Sources:
- Deploy config (authoritative for runtime): `gear_sonic_deploy/policy/release/observation_config.yaml`
- Training config: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
- Encoder module: `gear_sonic/trl/modules/universal_token_modules.py:260-307`
- Default actor/critic: `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml:37-55`
- Observation reference: `docs/source/references/observation_config.md`

---

## Deploy encoder I/O dimensionality

Single ONNX with input = sum of all entries under `encoder.encoder_observations` in `gear_sonic_deploy/policy/release/observation_config.yaml:28-56`:

| Observation | Dim |
|---|---:|
| `encoder_mode_4` | 4 |
| `motion_joint_positions_10frame_step5` | 290 |
| `motion_joint_velocities_10frame_step5` | 290 |
| `motion_root_z_position_10frame_step5` | 10 |
| `motion_root_z_position` | 1 |
| `motion_anchor_orientation` | 6 |
| `motion_anchor_orientation_10frame_step5` | 60 |
| `motion_joint_positions_lowerbody_10frame_step5` | 120 |
| `motion_joint_velocities_lowerbody_10frame_step5` | 120 |
| `vr_3point_local_target` | 9 |
| `vr_3point_local_orn_target` | 12 |
| `smpl_joints_10frame_step1` | 720 |
| `smpl_anchor_orientation_10frame_step1` | 60 |
| `motion_joint_positions_wrists_10frame_step1` | 60 |
| **Total input** | **1762** |
| **Output (`token_state`)** | **64** |

So the deploy graph is **1762 → 64**. All three "modes" share these same input/output tensors; only the values change (zero-filled where unused).

---

## Per-mode active inputs

The `encoder_modes:` block in the deploy YAML lists which observations are *real* in each mode. Everything else from the 1762D superset is zero-filled. All three modes also carry `encoder_mode_4` so the network knows which mode is active.

**`encoder_mode_4` layout (4D): `[mode_id, 0, 0, 0]` — *not* one-hot.** The integer mode id (cast to float) goes in slot 0; slots 1-3 are zero pads. Source: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:1687-1693`. Concretely:

| Mode | `encoder_mode_4` |
|---|---|
| `g1` (robot)    | `[0, 0, 0, 0]` |
| `teleop` (hybrid) | `[1, 0, 0, 0]` |
| `smpl` (human)   | `[2, 0, 0, 0]` |

---

## Robot encoder — `g1`, `mode_id: 0`
Full-body G1 joint reference, no human-side input.

| Observation | Dim | Description |
|---|---|---|
| `encoder_mode_4` | 4 | `[0, 0, 0, 0]` |
| `motion_joint_positions_10frame_step5` | 290 | 29 joints × 10 future frames @ 0.1 s |
| `motion_joint_velocities_10frame_step5` | 290 | 29 joints × 10 future frames @ 0.1 s |
| `motion_anchor_orientation_10frame_step5` | 60 | 6D root-orientation diff (body frame) × 10 frames |

> ⚠ **Encoder vs decoder asymmetry.** `motion_joint_positions_*` is fed **raw**
> (no default-angle subtraction; `localmotion_kplanner.hpp:494`). The decoder's
> `his_body_joint_positions_*` is fed the **deviation** `q − default_angles`
> (`g1_deploy_onnx_ref.cpp:2827`). See `findings/decoder_inputs.md` for the
> failure mode if you mix these up.

## Hybrid encoder — `teleop`, `mode_id: 1`
Lower body from G1 motion reference, upper body from VR 3-point teleop targets.

| Observation | Dim | Description |
|---|---|---|
| `encoder_mode_4` | 4 | `[1, 0, 0, 0]` |
| `motion_joint_positions_lowerbody_10frame_step5` | 120 | 12 lower-body joints × 10 frames |
| `motion_joint_velocities_lowerbody_10frame_step5` | 120 | 12 lower-body joints × 10 frames |
| `vr_3point_local_target` | 9 | left wrist / right wrist / head xyz, in root frame |
| `vr_3point_local_orn_target` | 12 | 3 quaternions (wxyz), in root frame |
| `motion_anchor_orientation` | 6 | Current-frame 6D root-orientation diff |

## Human encoder — `smpl`, `mode_id: 2`
SMPL human-body kinematics, plus G1 wrist joints (SMPL doesn't articulate wrists/dex hands).

| Observation | Dim | Description |
|---|---|---|
| `encoder_mode_4` | 4 | `[2, 0, 0, 0]` |
| `smpl_joints_10frame_step1` | 720 | 24 SMPL joints × 3 × 10 consecutive frames |
| `smpl_anchor_orientation_10frame_step1` | 60 | 6D root-orientation diff × 10 frames |
| `motion_joint_positions_wrists_10frame_step1` | 60 | 6 wrist joints × 10 frames |

---

## Encoder output → policy

The encoder emits a 64D `token_state` (set by `encoder.dimension` in the deploy YAML). The policy's full input concatenates `token_state` with proprioception/history (see `observation_config.yaml:5-23`). The deploy YAML comment header claims a total of 436D = 64+12+116+116+116+12, which would imply 4-frame history variants — but the YAML actually lists `_10frame_step1` names, which `docs/source/references/observation_config.md:268-272` says are 290D each. Worth double-checking against the ONNX policy input size; the comment header may be stale.

Robot proprioception is **not** routed through the encoders; it bypasses to the decoder/policy directly (see `gear_sonic/trl/modules/universal_token_modules.py:330-334`).

---

## Training-side note

Training uses slightly different observation names for the SMPL encoder (`smpl_joints_multi_future_local_nonflat`, `smpl_root_ori_b_multi_future`, `joint_pos_multi_future_wrist_for_smpl`) — see `sonic_release.yaml:88-104`. They encode the same three concepts as the deploy mode definitions above; the deploy YAML is what is fed at runtime.
