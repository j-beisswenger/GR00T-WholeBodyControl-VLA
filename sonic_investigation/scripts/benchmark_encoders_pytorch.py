#!/usr/bin/env python3
"""Benchmark PyTorch SONIC encoders + GMR retargeting on an AMASS motion.

Times three things on one AMASS clip:
  1. GMR retargeting (smplx -> unitree_g1) — full motion, ms/frame.
  2. PyTorch G1 (robot) encoder — one 640D frame tiled into a batch.
  3. PyTorch SMPL (human) encoder — one 840D frame tiled into a batch.

The encoders use training-side observations (640D / 840D), which require the
full FK pipeline to derive from a motion. To keep the script short, the
encoder benchmarks use a single shape-correct synthetic frame (seeded from
the motion path) tiled into the batch. The motion file is used genuinely
for GMR retargeting timing and to report the motion length.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from sonic_encoder_decoder_pytorch_unquantized import ENCODER_INPUT_DIMS, load_models

REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MOTION = (
    REPO_ROOT
    / "sonic_investigation/data/amass/diverse_dynamics_for_investigations"
    / "CMU_02_04_stageii.npz"
)
DEFAULT_SMPLX_DIR = REPO_ROOT / "sonic_investigation/data"
CONTROL_HZ = 50


def time_gmr_retarget(motion_path: Path, smplx_dir: Path) -> tuple[float, int]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    smplx_data, body_model, smplx_output, h = load_smplx_file(str(motion_path), smplx_dir)
    frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=CONTROL_HZ)
    retargeter = GMR(actual_human_height=h, src_human="smplx", tgt_robot="unitree_g1")
    _ = retargeter.retarget(frames[0])  # warm-up
    t0 = time.perf_counter()
    for f in frames:
        _ = retargeter.retarget(f)
    return time.perf_counter() - t0, len(frames)


def bench_encoder(enc, dim, batch_size, device, num_iters, warmup, seed):
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal(dim).astype(np.float32)
    batch = torch.from_numpy(np.tile(frame, (batch_size, 1))).to(device)
    enc = enc.to(device)
    use_cuda = device.startswith("cuda")
    with torch.no_grad():
        for _ in range(warmup):
            _ = enc(batch)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = enc(batch)
        if use_cuda:
            torch.cuda.synchronize()
        total = time.perf_counter() - t0
    return total, total / num_iters * 1000, total / num_iters / batch_size * 1e6, batch_size * num_iters / total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="AMASS .npz file.")
    p.add_argument("--smplx-dir", type=Path, default=DEFAULT_SMPLX_DIR, help="Parent dir containing smplx/.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data = np.load(args.motion)
    raw_T = data["trans"].shape[0]
    fps_key = "mocap_frame_rate" if "mocap_frame_rate" in data.files else "mocap_framerate"
    raw_fps = float(data[fps_key])
    print(f"[motion] {args.motion.name}")
    print(f"[motion] raw: {raw_T} frames @ {raw_fps:.1f} Hz ({raw_T/raw_fps:.2f} s)")

    print(f"\n[gmr] retargeting smplx -> unitree_g1 at {CONTROL_HZ} Hz ...")
    try:
        gmr_s, gmr_T = time_gmr_retarget(args.motion, args.smplx_dir)
        print(f"[gmr] retargeted: {gmr_T} frames @ {CONTROL_HZ} Hz")
        print(
            f"[gmr] loop: {gmr_s*1000:.1f} ms total | "
            f"{gmr_s/gmr_T*1000:.2f} ms/frame | {gmr_T/gmr_s:.0f} frames/s"
        )
    except Exception as e:
        print(f"[gmr] FAILED ({type(e).__name__}: {e}) — skipping retargeting timing")

    print(f"\n[load] PyTorch encoders from sonic_release/last.pt")
    try:
        encoders, _, _ = load_models()
    except FileNotFoundError as e:
        print(f"[load] checkpoint missing ({e}) — skipping encoder benchmarks")
        return
    seed = abs(hash(str(args.motion))) % (2**32)

    for mode in ("g1", "smpl"):
        dim = ENCODER_INPUT_DIMS[mode]
        total, per_batch_ms, per_sample_us, sps = bench_encoder(
            encoders[mode], dim, args.batch_size, args.device, args.num_iters, args.warmup, seed
        )
        label = {"g1": "G1 (robot)", "smpl": "SMPL (human)"}[mode]
        print(
            f"\n[bench] {label}  device={args.device}  batch={args.batch_size}  "
            f"input={dim}D  iters={args.num_iters} (+{args.warmup} warmup)"
        )
        print(f"  total      = {total*1000:.1f} ms")
        print(f"  per-batch  = {per_batch_ms:.3f} ms")
        print(f"  per-sample = {per_sample_us:.2f} us")
        print(f"  throughput = {sps:,.0f} samples/s")


if __name__ == "__main__":
    main()
