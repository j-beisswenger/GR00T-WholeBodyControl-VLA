# sonic_investigation

Working directory for investigating the SONIC policy in this repo. Not part of the published code — just notes, scripts, and reference assets.

## Layout

- `sonic_pipeline.svg` — pipeline diagram (reference figure).
- `INFERENCE_NOTES.md` — running notes on how SONIC inference is wired.
- `findings/` — written-up findings, one file per topic.
  - `encoder_inputs.md` — what is fed into the encoder in `g1` (robot), `teleop` (hybrid), and `smpl` (human) modes.
  - `decoder_inputs.md` — what is fed into the (mode-agnostic) action decoder.
- `scripts/`
  - `run_encoder_modes.py` — runs the deploy ONNX encoder in g1/teleop/smpl modes (1762 → 64).
  - `run_decoder.py` — runs the deploy ONNX action decoder (994 → 29).
  - `run_pytorch.py` — runs the same encoders/decoder from the PyTorch checkpoint, with `quantize=False` to bypass FSQ. Requires `sonic_release/last.pt` (download via `python download_from_hf.py --training --no-smpl`).
  - `run_idle_planner_decoder_mujoco.py` (+ `.sh` wrapper) — end-to-end IDLE rollout in MuJoCo with viewer: planner + encoder + decoder + 200 Hz PD physics, mirroring the C++ deploy IDLE path. Read the module docstring before editing — there's a non-obvious encoder/decoder asymmetry in joint-position conventions that bites every time.
