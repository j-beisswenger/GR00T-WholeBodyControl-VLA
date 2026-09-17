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

# 64-dim SONIC v1.1 idle token DERIVED FROM THE HUMANOID EVERYDAY START POSE (2026-09-17).
# The tracked HE corpus (sonic_tracked/humanoid_everyday, 4064 episodes) starts every episode in
# one consistent stance (per-joint std 0.02-0.10 rad, no clusters): straighter legs than
# DEFAULT_MJ (knees 0.41 rad vs 0.67, pelvis ~2 cm higher) and arms hanging with more bent
# elbows (0.89/0.92 rad vs 0.60). Idling in the corpus's own start pose puts the policy's first
# observation where its training episodes begin. Target joints (MuJoCo order, rad) are in the
# parent repo at deploy/sim/pi05/assets/he_start_pose/he_start_pose_abs_q29_mj.npy (the median
# of frame 0 over all episodes); render vs DEFAULT_MJ next to it.
#
# DERIVED with deploy/sim/pi05/derive_idle_token.py: the target stance repeated over the 10
# future frames, zero joint velocity, identity base orientation, G1 mode, through the v1.1
# encoder ONNX using the validated v1.1 layout (deploy/sim/sonic_roundtrip.py LAYOUT_V11) -- the
# same recipe as the two tokens below. VERIFIED by holding this exact token under the v1.1
# decoder in MuJoCo (SonicSim, 200 Hz PD) for 20 s from DEFAULT_MJ: stable, final-second joint
# jitter 0.01 deg, pelvis z 0.779 m (vs 0.760 m for the low-heat token: the legs are straighter),
# settled joints within 1.0 deg mean / 4.8 deg max (left elbow) of the target. The low-heat token
# below settles 4.0 deg mean / 17.8 deg max from it (elbows, knees). Byte-identical to
# deploy/sim/pi05/assets/initial_pose_token_v11_he.npy, which the sim's idle uses.
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [
         0.1250, -0.2500,  0.0000,  0.0000,  0.0625,  0.0000,  0.1250,
         0.0000, -0.0625,  0.1250,  0.0000, -0.1250,  0.3750,  0.1250,
        -0.0625,  0.0000,  0.0625,  0.0000,  0.0625, -0.1250, -0.0625,
        -0.0625,  0.0000,  0.1250, -0.3125,  0.3125, -0.0625,  0.0625,
         0.1875, -0.3125,  0.1875,  0.0000, -0.0625,  0.0625,  0.2500,
         0.0625, -0.0625,  0.2500,  0.1875,  0.0000,  0.0000,  0.0625,
         0.0625, -0.0625,  0.2500,  0.3750,  0.3750, -0.0625,  0.0000,
        -0.2500,  0.0000, -0.0625, -0.3125, -0.1875,  0.0625, -0.0625,
         0.3750,  0.1250,  0.0000,  0.1875, -0.0625,  0.0000,  0.1250,
         0.0000,
    ],
    dtype=np.float32,
)

# Previous 'i' token (2026-09-16 -> 2026-09-17): the low-heat variant, kept for reference.
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
LATENT_INITIAL_MOTION_TOKEN_V1_1_LOWHEAT = np.array(
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
