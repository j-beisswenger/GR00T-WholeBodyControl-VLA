"""Default initial poses for VLA inference.

These arrays are sent to the C++ control loop when the user presses 'i'
to move the robot to a known starting configuration before inference begins.

WARNING: The initial motion token below is specific to the SONIC checkpoint used
during training. Different SONIC checkpoints encode different latent spaces, so
this token will produce a different (and likely incorrect) pose if you switch to
a different SONIC checkpoint. When changing the SONIC checkpoint, you MUST update
LATENT_INITIAL_MOTION_TOKEN to a value that corresponds to a known safe standing
pose in the new checkpoint's latent space.
"""

import numpy as np

# 64-dim motion token for a stable standing pose.
# CHECKPOINT-SPECIFIC: this value must be updated if the SONIC checkpoint changes.
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [
        -0.0625,  0.0000, -0.0625, -0.1250, -0.1875, -0.0625,  0.1875,
         0.2500,  0.1875, -0.1250,  0.0625, -0.0625, -0.2500, -0.2500,
        -0.3125, -0.0625,  0.0000, -0.0625, -0.1250, -0.1875,  0.0000,
        -0.2500,  0.0000, -0.2500, -0.0625,  0.0625,  0.1250, -0.1250,
         0.2500,  0.1875,  0.2500, -0.1250,  0.1250,  0.1875, -0.0625,
         0.0000, -0.1875, -0.1875,  0.2500,  0.0000,  0.0000, -0.1250,
         0.0625,  0.0000, -0.0625, -0.0625,  0.1875, -0.0625,  0.0000,
         0.0625,  0.1250,  0.0625,  0.1250,  0.0625,  0.1250,  0.0000,
         0.1250,  0.1875,  0.0000,  0.0000,  0.0625,  0.0625,  0.1875,
         0.0625,
    ],
    dtype=np.float32,
)

# 64-dim motion token for a stable standing pose, SONIC v1.1's latent space (encoder 1751-D,
# decoder 994-D -- see data/egostandard/README.md in the parent repo). NOT interchangeable with
# LATENT_INITIAL_MOTION_TOKEN above: the two networks are trained separately and a v1.0 point
# means nothing to the v1.1 decoder (confirmed -- feeding the OLD token to a v1.1-decoder robot
# during bring-up moved to an unverified pose that happened to not look obviously wrong; that is
# luck, not correctness).
#
# DERIVED, not measured on hardware: encoded a synthetic "stand still" reference (DEFAULT_MJ
# stance, zero joint velocity, identity base orientation, G1 mode) through the v1.1 encoder ONNX,
# using the validated v1.1 input layout in deploy/sim/sonic_roundtrip.py (LAYOUT_V11 -- NOTE this
# is not a naive reuse of the v1.0 layout: v1.1 drops two always-zero padding slots AND swaps the
# order of the two anchor-orientation slots relative to v1.0, both cross-checked there against
# v1.1's own model_config.yaml). VERIFIED by decoding this exact token with the v1.1 decoder ONNX
# and running it in MuJoCo (deploy/sim SonicSim, 200 Hz PD) for 5s: pelvis height held
# 0.753-0.768 m, matching the 0.757-0.787 m band this repo already treats as healthy v1.0 idle.
# Recipe: /tmp/.../derive_v11_idle.py this session -- re-derive with
# sonic_roundtrip.Encoder + SPECS["v1.1"] if the v1.1 checkpoint ever changes.
LATENT_INITIAL_MOTION_TOKEN_V1_1 = np.array(
    [
        0.1250, -0.1875, -0.0625, -0.1250,  0.0625,  0.0000,  0.1250,
       -0.0625, -0.1250,  0.0000,  0.0000, -0.1250,  0.2500,  0.1250,
        0.0625,  0.0000,  0.0000,  0.0000, -0.0625,  0.0000, -0.0625,
        0.0000, -0.0625,  0.1250, -0.1875,  0.2500, -0.1250,  0.0625,
        0.1250, -0.2500,  0.1875,  0.0000, -0.0625,  0.1250,  0.1250,
        0.0625,  0.0625,  0.1250,  0.2500,  0.0000,  0.0000,  0.0625,
        0.0625, -0.1250,  0.1875,  0.2500,  0.2500, -0.0625,  0.0000,
       -0.2500,  0.0625, -0.0625, -0.1875, -0.1250,  0.1250,  0.0000,
        0.2500,  0.1250,  0.0625,  0.0625,  0.0000,  0.0000,  0.1250,
        0.0000,
    ],
    dtype=np.float32,
)
