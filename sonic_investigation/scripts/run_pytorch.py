"""Run the SONIC encoder and decoder from the released PyTorch checkpoint.

PyTorch counterpart of `run_encoder_modes.py` + `run_decoder.py`. Useful for:
  - bypassing the FSQ quantizer (`quantize=False`) to get continuous tokens,
  - reading intermediate activations,
  - cross-checking the deploy ONNX numerically.

Each per-mode encoder MLP takes only its own modality's training-side inputs
(NOT the 1762D deploy superset). Mode is selected by *which encoder we run*;
there is no `encoder_mode_4` channel on the PyTorch side — that's a deploy-time
concatenation artefact.

IMPORTANT — training inputs ≠ deploy inputs.
  The deploy ONNX exposes obs like `motion_joint_positions_10frame_step5`
  (joint angles). The PyTorch encoders consume training-side obs like
  `command_multi_future_nonflat` (reference body LINK POSITIONS in body frame).
  These are *different* observations of the same motion, with different
  dimensions and semantics. Do not reuse deploy-style joint vectors as-is;
  see `gear_sonic/envs/manager_env/mdp/observations.py` for the obs formulas.

Per-mode encoder input layout (pre-flat tensor, training obs in YAML order):

  g1 (640D total):
    command_multi_future_nonflat           580D  body link positions × 10 future frames
    motion_anchor_ori_b_mf_nonflat          60D  6D root-orientation diff × 10 frames

  teleop (267D total):
    command_multi_future_lower_body        240D  lower-body link positions × 10 frames
    vr_3point_local_target                   9D  [Lwrist, Rwrist, head] xyz, body frame
    vr_3point_local_orn_target              12D  3 quaternions wxyz, body frame
    motion_anchor_ori_b                      6D  current-frame 6D root-ori diff

  smpl (840D total):
    smpl_joints_multi_future_local_nonflat 720D  24 SMPL joints × 3 × 10 frames
    smpl_root_ori_b_multi_future            60D  6D × 10 frames
    joint_pos_multi_future_wrist_for_smpl   60D  6 G1 wrist joints × 10 frames

Decoder input: 994D = token_state(64) + history(930). Output: 29D action.
The decoder *does* match the deploy YAML's `observations:` block exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = REPO_ROOT / "sonic_release" / "last.pt"

# FSQ config from gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml
NUM_FSQ_LEVELS = 32       # codebook dim per token (== levels per dim)
MAX_NUM_TOKENS = 2        # number of tokens per latent
TOKEN_FLAT_DIM = NUM_FSQ_LEVELS * MAX_NUM_TOKENS  # 64
ACTION_DIM = 29

ENCODER_INPUT_DIMS = {"g1": 640, "teleop": 267, "smpl": 840}
DECODER_INPUT_DIM = 994


# --- model construction from raw state dict --------------------------------

def _build_mlp(state_dict: dict, prefix: str, activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    """Reconstruct an MLP from `<prefix>.module.{0,2,4,...}.{weight,bias}` keys.

    Linear layers live at even indices in the saved Sequential; activations are
    interleaved between them.
    """
    sub = f"{prefix}.module."
    indices = sorted({int(k[len(sub):].split(".")[0]) for k in state_dict if k.startswith(sub)})
    layers: list[nn.Module] = []
    for i, idx in enumerate(indices):
        w = state_dict[f"{prefix}.module.{idx}.weight"]
        b = state_dict[f"{prefix}.module.{idx}.bias"]
        out_f, in_f = w.shape
        lin = nn.Linear(in_f, out_f)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:  # no activation after the last linear
            layers.append(activation())
    return nn.Sequential(*layers).eval()


def load_models(ckpt_path: Path = CKPT_PATH):
    """Load encoders {g1, teleop, smpl} and decoders {g1_dyn, g1_kin} + FSQ from the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["policy_state_dict"]
    encoders = {n: _build_mlp(sd, f"actor_module.encoders.{n}") for n in ("g1", "teleop", "smpl")}
    decoders = {n: _build_mlp(sd, f"actor_module.decoders.{n}") for n in ("g1_dyn", "g1_kin")}

    # FSQ from vector_quantize_pytorch — parameterless, matches deploy ONNX
    # output lattice (multiples of 1/16 for L=32).
    from vector_quantize_pytorch import FSQ
    fsq = FSQ(levels=[NUM_FSQ_LEVELS] * NUM_FSQ_LEVELS).eval()
    return encoders, decoders, fsq


# --- shared helpers --------------------------------------------------------

def _to_tensor(arr, expected_dim: int, name: str) -> torch.Tensor:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    if a.size != expected_dim:
        raise ValueError(f"{name}: expected flat dim {expected_dim}, got {a.size}")
    return torch.from_numpy(a)


def _quantize(latent: torch.Tensor, fsq) -> torch.Tensor:
    """Apply FSQ to a (B, 64) latent.

    Reshapes (B, 64) → (B, MAX_NUM_TOKENS=2, NUM_FSQ_LEVELS=32) per the
    universal_token_modules.py path, FSQ per-token, then flattens back.
    """
    B = latent.shape[0]
    z = latent.view(B, MAX_NUM_TOKENS, NUM_FSQ_LEVELS)
    quantized, _ = fsq(z)
    return quantized.reshape(B, TOKEN_FLAT_DIM)


# --- entry points ----------------------------------------------------------

def run_encoder(
    encoders,
    fsq,
    mode: str,
    x_flat: np.ndarray,
    quantize: bool = True,
) -> np.ndarray:
    """Run a single per-mode encoder MLP, optionally with FSQ.

    Args:
        mode: one of 'g1', 'teleop', 'smpl'.
        x_flat: pre-flat input of dim ENCODER_INPUT_DIMS[mode]. Built by
            concatenating that mode's training-side obs in the order documented
            at the top of this file. Shape (D,) or (B, D).
        quantize: if False, return raw 64D latent (continuous floats); if True,
            return FSQ-quantized 64D token (multiples of 1/16, matching deploy).

    Returns:
        np.ndarray of shape (64,) or (B, 64).
    """
    if mode not in encoders:
        raise KeyError(f"unknown mode '{mode}', expected one of {list(encoders)}")
    expected = ENCODER_INPUT_DIMS[mode]

    arr = np.asarray(x_flat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze = True
    else:
        squeeze = False
    if arr.shape[-1] != expected:
        raise ValueError(f"mode={mode!r}: expected last dim {expected}, got {arr.shape[-1]}")

    x = torch.from_numpy(arr)
    with torch.no_grad():
        latent = encoders[mode](x)                           # (B, 64)
        out = _quantize(latent, fsq) if quantize else latent
    out_np = out.numpy()
    return out_np[0] if squeeze else out_np


def run_decoder(
    decoders,
    token_state_64: np.ndarray,             # (64,)    from any encoder (post-FSQ)
    his_base_ang_vel_10x3: np.ndarray,      # (10, 3)  body frame
    his_body_joint_pos_10x29: np.ndarray,   # (10, 29) IsaacLab order, q − default_angles (NOT raw)
    his_body_joint_vel_10x29: np.ndarray,   # (10, 29) IsaacLab order, raw velocities
    his_last_actions_10x29: np.ndarray,     # (10, 29) IsaacLab order, raw decoder outputs
    his_gravity_dir_10x3: np.ndarray,       # (10, 3)  body frame
) -> np.ndarray:
    """Run the action decoder (994 -> 29). Inputs match the deploy YAML's
    `observations:` block one-to-one (no naming divergence here).

    ⚠ `his_body_joint_pos_10x29` must be `q − default_angles` (IsaacLab order),
    not raw `q`. Deploy reference: `g1_deploy_onnx_ref.cpp:2827`. This is the
    *opposite* convention from the encoder's `motion_joint_positions_*`, which
    is fed raw. Mixing these up makes the decoder see a huge initial pose error
    at default stance.
    """
    parts = [
        _to_tensor(token_state_64,           64,   "token_state_64"),
        _to_tensor(his_base_ang_vel_10x3,    30,   "his_base_ang_vel_10x3"),
        _to_tensor(his_body_joint_pos_10x29, 290,  "his_body_joint_pos_10x29"),
        _to_tensor(his_body_joint_vel_10x29, 290,  "his_body_joint_vel_10x29"),
        _to_tensor(his_last_actions_10x29,   290,  "his_last_actions_10x29"),
        _to_tensor(his_gravity_dir_10x3,     30,   "his_gravity_dir_10x3"),
    ]
    x = torch.cat(parts).unsqueeze(0)        # (1, 994)
    with torch.no_grad():
        action = decoders["g1_dyn"](x)
    return action.squeeze(0).numpy()


# --- smoke test ------------------------------------------------------------

if __name__ == "__main__":
    encoders, decoders, fsq = load_models()
    rng = np.random.default_rng(0)

    print("ENCODER (3 modes × 2 quantization settings):")
    for mode, dim in ENCODER_INPUT_DIMS.items():
        for quantize in (True, False):
            tok = run_encoder(encoders, fsq, mode, rng.standard_normal(dim), quantize=quantize)
            tag = "FSQ on " if quantize else "FSQ OFF"
            print(f"  {mode:7s} {tag} -> {tok.shape}  {tok[:4]}")

    print("\nDECODER (994 -> 29):")
    tok_for_decoder = run_encoder(encoders, fsq, "g1", rng.standard_normal(640), quantize=True)
    action = run_decoder(
        decoders,
        tok_for_decoder,
        rng.standard_normal((10, 3)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 29)),
        rng.standard_normal((10, 3)),
    )
    print(f"  action -> {action.shape}  {action[:8]} ...")
