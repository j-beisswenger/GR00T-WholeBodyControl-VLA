#!/usr/bin/env python3
"""
Plot AMASS universal tokens comparing the SMPL encoder and the G1 encoder.

This script crawls an AMASS root directory, selects a random subset of
motions, feeds them through both the Human (SMPL) encoder path and the
Robot (GMR->G1) encoder path, and overlays their resulting 64-D tokens
for direct comparison.

Run:
    python scripts/plot_amass.py --amass-dir data/amass --smplx-dir data
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the verified, working logic from the sibling scripts
from run_encoder_modes import run_smpl, run_g1
from run_idle_planner_decoder_mujoco import build_encoder_inputs, DEPLOY_ROOT
from run_amass_smpl_mujoco import load_amass_motion, _zero_yaw_smpl_motion, build_smpl_encoder_inputs
from amass_to_sonic import retarget_amass, build_deploy_motion

ENCODER_PATH = DEPLOY_ROOT / "policy/release/model_encoder.onnx"
TOKEN_DIM = 64


def plot_overlay(name, traces, outfile):
    """Overlay multiple (label, color, t, z[T,64]) tokens in one figure."""
    rows, cols = 8, 8
    fig = plt.figure(figsize=(2.0 * cols, 1.4 * rows + 2.4))
    gs = fig.add_gridspec(nrows=9, ncols=cols, height_ratios=[1] * rows + [1.8], hspace=0.35, wspace=0.18)

    first_ax = None
    for d in range(TOKEN_DIM):
        r, c = divmod(d, cols)
        ax = fig.add_subplot(gs[r, c], sharex=first_ax, sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        for label, color, t, z in traces:
            ax.plot(t, z[:, d], color=color, lw=0.9, label=label if d == 0 else None)
        ax.set_title(f"dim {d}", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-1.05, 1.05)

    ax_s = fig.add_subplot(gs[rows, :])
    for label, color, t, z in traces:
        dz = np.linalg.norm(np.diff(z, axis=0), axis=1)
        ax_s.plot(t[1:], dz, lw=1.0, color=color, label=label)
    ax_s.set_xlabel("time [s]", fontsize=9)
    ax_s.set_ylabel("||Δtoken||", fontsize=8)
    ax_s.set_title("latent jump size per tick", fontsize=9)
    ax_s.grid(True, alpha=0.3)
    ax_s.tick_params(labelsize=7)
    ax_s.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"GEAR-SONIC universal token  —  {name}", fontsize=11, y=0.995)
    fig.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amass-dir", required=True, type=Path, help="Root directory containing AMASS .npz files.")
    parser.add_argument(
        "--smplx-dir", required=True, type=Path, help="Parent directory containing the smplx/ subfolder."
    )
    parser.add_argument("--outdir", type=Path, default=Path("plots"), help="Directory to save the resulting plots.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of random motions to plot.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[onnx] Loading encoder from {ENCODER_PATH}")
    enc = ort.InferenceSession(str(ENCODER_PATH), providers=["CPUExecutionProvider"])

    # We use a static identity quaternion for the robot's base.
    # This acts as our "open-loop" anchor assumption for purely comparing encoded outputs.
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # ---------------------------------------------------------
    # Gather & Sample Motions
    # ---------------------------------------------------------
    print(f"\n[setup] Scanning {args.amass_dir} for .npz files...")
    all_motions = list(args.amass_dir.rglob("*.npz"))

    if not all_motions:
        print(f"Error: No .npz files found in {args.amass_dir}")
        sys.exit(1)

    num_to_select = min(args.num_samples, len(all_motions))
    selected_motions = random.sample(all_motions, num_to_select)
    print(f"[setup] Found {len(all_motions)} motions. Randomly selected {num_to_select} for plotting.")

    # ---------------------------------------------------------
    # Process Selected Motions
    # ---------------------------------------------------------
    for i, motion_path in enumerate(selected_motions, 1):
        print(f"\n=== Processing Motion {i}/{num_to_select}: {motion_path.name} ===")
        traces = []

        # 1. SMPL Path (Mode 2)
        print(f"  [smpl] Processing human encoder path...")
        try:
            motion_smpl = load_amass_motion(motion_path, args.smplx_dir, target_fps=50)
            motion_smpl["smpl_root_quat"], motion_smpl["smpl_joints"] = _zero_yaw_smpl_motion(
                motion_smpl["smpl_root_quat"], motion_smpl["smpl_joints"]
            )

            T_smpl = motion_smpl["smpl_joints"].shape[0]
            tokens_smpl = []
            for t in range(T_smpl):
                enc_in = build_smpl_encoder_inputs(motion_smpl, t, base_quat)
                tok = run_smpl(enc, enc_in["smpl_joints"], enc_in["smpl_anchor"], enc_in["wrist_jp"])
                tokens_smpl.append(tok)

            tokens_smpl = np.stack(tokens_smpl)
            t_smpl = np.arange(T_smpl) / 50.0
            traces.append(("Human (SMPL, mode 2)", "tab:blue", t_smpl, tokens_smpl))
            print(f"    -> Success: T={T_smpl}, range=[{tokens_smpl.min():+.2f}, {tokens_smpl.max():+.2f}]")
        except Exception as e:
            print(f"    -> Failed SMPL path: {e}")

        # 2. G1 Path (Mode 0)
        print(f"  [g1] Retargeting AMASS for robot encoder via GMR...")
        try:
            qpos_seq, fps = retarget_amass(motion_path, args.smplx_dir)
            motion_g1 = build_deploy_motion(qpos_seq, fps)

            T_g1 = motion_g1["joint_pos"].shape[0]
            tokens_g1 = []
            for t in range(T_g1):
                enc_in = build_encoder_inputs(motion_g1, t, base_quat)
                tok = run_g1(enc, enc_in["motion_jp"], enc_in["motion_jv"], enc_in["motion_anchor"])
                tokens_g1.append(tok)

            tokens_g1 = np.stack(tokens_g1)
            t_g1 = np.arange(T_g1) / 50.0
            traces.append(("Robot (GMR→G1, mode 0)", "tab:orange", t_g1, tokens_g1))
            print(f"    -> Success: T={T_g1}, range=[{tokens_g1.min():+.2f}, {tokens_g1.max():+.2f}]")
        except Exception as e:
            print(f"    -> Failed G1 path: {e}")

        # 3. Plotting
        if traces:
            # Create a somewhat unique filename by incorporating the parent folder name
            file_prefix = f"{motion_path.parent.name}_{motion_path.stem}"
            outfile = args.outdir / f"{file_prefix}_tokens.png"
            plot_overlay(f"{motion_path.parent.name}/{motion_path.name}", traces, outfile)
        else:
            print(f"  [plot] No traces successfully generated for {motion_path.name}. Skipping plot.")

    print("\n[done] All selected motions processed.")


if __name__ == "__main__":
    main()
