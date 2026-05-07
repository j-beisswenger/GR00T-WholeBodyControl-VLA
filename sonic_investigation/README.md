# sonic_investigation

Working directory for investigating the SONIC policy in this repo. Not part of the published code — just notes, scripts, and reference assets.

## Layout

- `sonic_pipeline.svg` — pipeline diagram (reference figure).
- `INFERENCE_NOTES.md` — running notes on how SONIC inference is wired.
- `findings/` — written-up findings, one file per topic.
  - `encoder_inputs.md` — what is fed into the encoder in `g1` (robot), `teleop` (hybrid), and `smpl` (human) modes.
  - `decoder_inputs.md` — what is fed into the (mode-agnostic) action decoder.
- `scripts/`
  - `sonic_encoder_onnx_3modes.py` — runs the deploy ONNX encoder in g1/teleop/smpl modes (1762 → 64).
  - `sonic_decoder_onnx.py` — runs the deploy ONNX action decoder (994 → 29).
  - `sonic_encoder_decoder_pytorch_unquantized.py` — runs the same encoders/decoder from the PyTorch checkpoint, with `quantize=False` to bypass FSQ. Requires `sonic_release/last.pt` (download via `python download_from_hf.py --training --no-smpl`).
  - `mujoco_viewer_synthetic_idle_encoder_decoder.py` (+ `.sh` wrapper) — end-to-end IDLE rollout in MuJoCo with viewer: synthetic stand-still reference + encoder + decoder + 200 Hz PD physics, mirroring the C++ deploy IDLE path. Read the module docstring before editing — there's a non-obvious encoder/decoder asymmetry in joint-position conventions that bites every time.
  - `mujoco_viewer_csv_motion_g1_encoder_decoder.py` (+ `.sh` wrapper) — same MuJoCo+encoder+decoder stack, but plays back a recorded CSV reference motion (idle ↔ motion cycle) through the G1/robot encoder.
  - `mujoco_viewer_amass_smpl_encoder_decoder.py` (+ `.sh` wrapper) — same stack, but reference is an AMASS .npz fed through the SMPL/human encoder (mode 2).
  - `mujoco_viewer_amass_gmr_g1_encoder_decoder.py` (+ `.sh` wrapper) — same stack, but reference is an AMASS .npz retargeted via GMR to the G1 and fed through the G1/robot encoder (mode 0).
  - `plot_amass_smpl_vs_g1_encoder_tokens.py` — overlay 64-D encoder tokens from the SMPL and G1 paths across a random subset of AMASS clips.
