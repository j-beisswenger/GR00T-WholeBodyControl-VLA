# SONIC Decoder Inputs

## TL;DR

The deploy decoder is **a single ONNX model**: input `obs_dict` of shape `(1, 994)` float, output `action` of shape `(1, 29)` float. It is mode-agnostic — there is no per-mode switching at decode time; the encoder already absorbed the mode, and what the decoder sees is just `[token_state (64), proprioception history (930)]`.

Verified empirically:

```
INPUTS:  obs_dict       shape=[1, 994]  type=tensor(float)
OUTPUTS: action         shape=[1, 29]   type=tensor(float)
```

(`onnxruntime.InferenceSession` on `gear_sonic_deploy/policy/release/model_decoder.onnx`.)

Sources:
- Deploy config (authoritative for runtime): `gear_sonic_deploy/policy/release/observation_config.yaml`
- Training decoder config: `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml`
- Training actor/critic config: `gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml`
- Decoder module wiring: `gear_sonic/trl/modules/universal_token_modules.py:316-379`
- Observation reference: `docs/source/references/observation_config.md`

---

## Deploy decoder I/O dimensionality

Single ONNX with input = sum of all enabled entries under `observations:` in `gear_sonic_deploy/policy/release/observation_config.yaml:5-23`, in YAML order:

| Observation | Dim | Source |
|---|---:|---|
| `token_state` | 64 | encoder output (`encoded_tokens`) |
| `his_base_angular_velocity_10frame_step1` | 30 | IMU ω, 10 consecutive ticks (3 × 10) |
| `his_body_joint_positions_10frame_step1` | 290 | joint position **deviations from default angles** (`q − default_angles`), IsaacLab order, 10 ticks (29 × 10) |
| `his_body_joint_velocities_10frame_step1` | 290 | joint velocities (raw, not offset), IsaacLab order, 10 ticks (29 × 10) |
| `his_last_actions_10frame_step1` | 290 | previous decoder outputs (raw IsaacLab-order action), 10 ticks (29 × 10) |
| `his_gravity_dir_10frame_step1` | 30 | gravity direction in body frame, 10 ticks (3 × 10) |
| **Total input** | **994** | |
| **Output (`action`)** | **29** | per-joint normalised action targets, IsaacLab order |

So the deploy graph is **994 → 29**. All observations are stacked at 50 Hz, step=1 (i.e. last 0.2 s of history at 10 frames).

> ⚠ **Joint-position offset gotcha.** `his_body_joint_positions_*` is *not* raw `q`; the deploy logs `body_q[i] = motor_q[mujoco_to_isaaclab[i]] - default_angles[mujoco_to_isaaclab[i]]` (`g1_deploy_onnx_ref.cpp:2827`) and that is what the decoder reads back via `GatherHisBodyJointPositions`. The encoder side is the *opposite*: `motion_joint_positions_*` is fed raw (no default subtraction — see `localmotion_kplanner.hpp:494`). Forgetting this offset on the decoder side makes the network see a huge initial pose error at default stance (knee 0.669, ankle −0.363, …) and drives the robot into a squat/fall.

> Note: the YAML header comment on line 3 claims `Total dimension: 436 (64+12+116+116+116+12)`, which would correspond to the `_4frame_step1` history variants. That comment is **stale** — the YAML actually lists `_10frame_step1` names, and the ONNX confirms 994D. Worth a future PR to fix.

---

## What the encoder mode does *not* affect

Unlike the encoder (which has three mode-specific input subsets glued into one 1762D superset), the decoder has no per-mode pathway. Once the encoder emits its 64D `token_state`, the decoder treats it as a black-box motion command and combines it with the robot's own history. So:

- The decoder ONNX is the **same graph** whether the user is in `g1`, `teleop`, or `smpl` mode.
- There is **no `encoder_mode_4` input on the decoder**.
- The mode-conditioning effectively lives inside the encoder's MLP routing — the policy itself only ever sees a token vector + proprio.

This is why the policy can stay invariant across mode switches and why upstream changes (planner / VR / SMPL retarget) only have to touch the encoder side.

---

## Training-side wiring

Per `all_mlp_v1.yaml:11-12`, the SONIC actor backbone instantiates two decoders:

| Decoder config | Inputs | Outputs | Role |
|---|---|---|---|
| `g1_dyn` (`g1_dyn_mlp.yaml`) | `token_flattened`, `proprioception` | `action` (29D) | **Action policy — exported as `model_decoder.onnx`** |
| `g1_kin` (`g1_kin_mf_mlp.yaml`) | `token` | `command_multi_future_nonflat`, `motion_anchor_ori_b_mf_nonflat` | Kinematic reconstruction head, used as auxiliary loss only |

Only `g1_dyn` is shipped to deploy. `g1_kin` exists to encourage the latent token to remain a faithful summary of the future motion, by asking it to reconstruct the reference trajectory; it is dropped at export.

Dimension bookkeeping at the `g1_dyn` interface:
- `token_flattened` = `token_dim × max_num_tokens` = `num_fsq_levels × max_num_tokens` = 32 × 2 = **64** (matches the encoder output).
- `proprioception` = sum over `proprioception_features = ["actor_obs"]`. With `actor_prop_history_length=10` and the deploy YAML's history entries, this sums to 30+290+290+290+30 = **930**.
- 64 + 930 = **994** ✓

So the deploy decoder layout `[token_state, his_base_ang_vel, his_joint_pos, his_joint_vel, his_last_actions, his_gravity_dir]` mirrors the training input `[token_flattened, proprioception]` after the proprioception sub-vectors are concatenated in the YAML's stated order.

---

## Output (`action`)

29D vector of normalised joint-action targets, in IsaacLab joint ordering (same ordering used everywhere else in the deploy code — see `docs/source/references/observation_config.md` joint-related entries). Downstream low-level control denormalises and applies these as PD targets to the robot. The action is also written back into `last_actions` for the next tick's history slot.
