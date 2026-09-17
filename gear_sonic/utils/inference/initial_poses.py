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

# 64-dim motion token for a stable standing pose, SONIC v1.1's latent space (encoder 1751-D,
# decoder 994-D). This branch's checkpoints are all trained on the v1.1 closed-loop
# sonic_tracked corpora (egostandard/humanoid_everyday/leverb/psi/unifolm), which is NOT
# interchangeable with the old v1.0 token: the two decoder networks are trained separately and
# a v1.0 point means nothing to a v1.1 decoder.
#
# LOW-HEAT VARIANT, not the plain DEFAULT_MJ-derived idle: encoded a synthetic "stand still"
# reference (DEFAULT_MJ stance, zero joint velocity, identity base orientation, G1 mode) with
# BOTH elbow joints overridden to 1.2 rad (DEFAULT_MJ's own elbow value is 0.6 rad) through the
# v1.1 encoder ONNX, using the validated v1.1 input layout in deploy/sim/sonic_roundtrip.py
# (LAYOUT_V11). Chosen deliberately straighter than DEFAULT_MJ's arm-forward stance to reduce
# the shoulder/elbow holding torque (and therefore motor heat) during long idle holds between
# real-robot deploys -- the decoder is a closed-loop policy, not a literal token->pose lookup,
# so it does not reproduce the encoded reference exactly; it pulls partway back toward its
# trained distribution. Empirically the encoded target and the decoder's SETTLED value track
# each other but are not equal (checked at three targets before settling on this one: 0.8 rad
# settled at L0.734/R0.712, 0.9 rad settled at L0.877/R0.822, 1.2 rad -- this one -- settled at
# L1.139/R1.231, i.e. actually overshooting the request on the right arm).
#
# VERIFIED by decoding this exact token with the v1.1 decoder ONNX and running it in MuJoCo
# (deploy/sim SonicSim, 200 Hz PD) for 20s with a live on-screen viewer (visually confirmed: no
# self-collision, stable stand, straighter arms): pelvis height held 0.756-0.762 m (the plain
# DEFAULT_MJ-derived v1.1 idle held 0.753-0.768 m over 5s -- same healthy band), final elbow
# angles L1.139/R1.231 rad (vs DEFAULT_MJ's 0.6 rad and the old idle token's own settled
# ~0.556/0.615 rad).
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [
        0.0625, -0.3750,  0.0000, -0.0625,  0.0625,  0.0000,  0.1250,
        0.0000, -0.1875,  0.2500,  0.0000, -0.0625,  0.4375,  0.0625,
        0.0000,  0.0000, -0.0625, -0.0625,  0.0625, -0.1250, -0.0625,
        0.0000, -0.0625,  0.1250, -0.4375,  0.3750, -0.1250,  0.0000,
        0.1250, -0.3750,  0.1250,  0.0000, -0.0625,  0.0625,  0.3125,
        0.0000,  0.0000,  0.2500,  0.1875,  0.0625,  0.0000,  0.1250,
        0.0000, -0.1250,  0.2500,  0.4375,  0.4375, -0.0625,  0.0000,
       -0.2500,  0.0000, -0.1250, -0.4375, -0.1875,  0.0625, -0.0625,
        0.4375,  0.1250, -0.0625,  0.1250,  0.0625,  0.0000,  0.0625,
       -0.0625,
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
