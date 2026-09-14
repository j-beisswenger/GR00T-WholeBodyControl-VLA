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
# a v1.0 point means nothing to a v1.1 decoder. Swapped in from main's LATENT_INITIAL_MOTION_
# TOKEN_V1_1 (deploy/GR00T-WholeBodyControl/gear_sonic/utils/inference/initial_poses.py), which
# is decoder-latent-space-specific, not checkpoint-specific -- reusable across every v1.1
# checkpoint on this branch, same as it is on main.
#
# DERIVED, not measured on hardware: encoded a synthetic "stand still" reference (DEFAULT_MJ
# stance, zero joint velocity, identity base orientation, G1 mode) through the v1.1 encoder
# ONNX, using the validated v1.1 input layout in deploy/sim/sonic_roundtrip.py (LAYOUT_V11 --
# NOTE this is not a naive reuse of the v1.0 layout: v1.1 drops two always-zero padding slots
# AND swaps the order of the two anchor-orientation slots relative to v1.0, both cross-checked
# there against v1.1's own model_config.yaml). VERIFIED by decoding this exact token with the
# v1.1 decoder ONNX and running it in MuJoCo (deploy/sim SonicSim, 200 Hz PD) for 5s: pelvis
# height held 0.753-0.768 m, matching the 0.757-0.787 m band this repo already treats as
# healthy v1.0 idle.
LATENT_INITIAL_MOTION_TOKEN = np.array(
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
