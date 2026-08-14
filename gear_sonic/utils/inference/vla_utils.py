"""Utility functions for VLA inference.

Includes action processing, observation preparation, latency compensation,
and inference scheduling logic.
"""

from typing import Any, Dict

import numpy as np

from gear_sonic.data.robot_model.robot_model import RobotModel


def concat_action(robot_model: RobotModel, goal: Dict[str, Any]) -> Dict[str, Any]:
    """Process the action dict from the policy into a flat dict.

    Strips ``action.`` prefixes from keys (if present) and returns the result.

    Args:
        robot_model: RobotModel instance (unused for latent actions, kept for API compat).
        goal: Action dict from policy.

    Returns:
        Processed action dict with prefixes stripped.
    """
    processed_goal = {}
    for key, value in goal.items():
        processed_goal[key.replace("action.", "")] = value
    return processed_goal


# SONIC stance pose per state group, in each group's own joint order. Must stay identical to
# gr00t/data/sonic_q_convention.py:DEFAULT_ANGLES_BY_GROUP (training) and to
# deploy/real/common/sonic_constants.py:DEFAULT_MJ (the pi0.5 bridge). These three are the same
# 29 numbers; a drift between them does not raise, it degrades the state channel silently.
# Groups absent here are never offset: gravity is a direction, hands are outside the token space.
DEFAULT_ANGLES_BY_GROUP = {
    "left_leg": np.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0], dtype=np.float32),
    "right_leg": np.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0], dtype=np.float32),
    "waist": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "left_arm": np.array([0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32),
    "right_arm": np.array([0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32),
}


def prepare_observation_for_eval(
    robot_model: RobotModel, obs: dict, state_q_dev: bool = False
) -> dict:
    """Split whole-body ``q`` into per-joint-group state keys for the policy.

    Populates ``obs["state"]`` with ``left_arm``, ``right_arm``, ``waist``,
    ``left_leg``, ``right_leg``, ``left_hand``, ``right_hand`` sub-keys
    using the nested dict format expected by ``Gr00tPolicy``.

    With ``state_q_dev`` the five body groups are emitted as ``q - default_angles`` rather than
    raw ``q``. This MUST match how the checkpoint was trained. Models trained after the
    ``state_q_convention`` change consume q_dev, which is also what the SONIC decoder and the
    pi0.5 bridge use; the offset is 0.669 rad at the knee and 0.363 at the ankle, so getting it
    backwards biases exactly the joints that carry balance. It defaults to False because every
    checkpoint trained before that change consumes absolute q -- flipping the default would break
    each of them on the next run. Turn it on for models trained with ``state_q_convention`` set.

    Hands are never offset -- the 14 hand DOF are outside the SONIC token space.

    Args:
        robot_model: RobotModel instance.
        obs: Observation dict containing ``"q"`` key and a ``"state"`` sub-dict.
        state_q_dev: Emit body joints as q - default_angles (see above).

    Returns:
        Modified observation dict with ``obs["state"]`` populated.
    """
    assert "q" in obs, "q is not in the observation"

    whole_q = obs["q"]
    assert whole_q.shape[-1] == robot_model.num_joints, "q has wrong shape"

    if "state" not in obs:
        obs["state"] = {}

    for group in ("left_arm", "right_arm", "waist", "left_leg", "right_leg"):
        values = whole_q[..., robot_model.get_joint_group_indices(group)]
        if state_q_dev:
            default = DEFAULT_ANGLES_BY_GROUP[group]
            assert values.shape[-1] == default.shape[0], (
                f"state group {group!r} has width {values.shape[-1]}, expected {default.shape[0]}"
            )
            values = values - default
        obs["state"][group] = values

    obs["state"]["left_hand"] = whole_q[..., robot_model.get_joint_group_indices("left_hand")]
    obs["state"]["right_hand"] = whole_q[..., robot_model.get_joint_group_indices("right_hand")]

    return obs


def calculate_latency_compensated_index(
    inference_delay: float, control_freq: float, action_horizon: int
) -> int:
    """Calculate the starting action index compensating for inference latency.

    When inference completes, some time has elapsed, so we skip the first few
    actions that are now "stale" and start from a later index in the chunk.

    Args:
        inference_delay: Time elapsed since inference started (seconds).
        control_freq: Control loop frequency (Hz), e.g. 20.
        action_horizon: Total number of actions in the chunk, e.g. 16.

    Returns:
        Starting index (0 to action_horizon-1) for the action chunk.
    """
    raw_index = np.round(inference_delay * control_freq)
    return int(np.clip(raw_index, 0, action_horizon - 1))


def should_trigger_new_inference(
    cached_chunk_exists: bool,
    inference_thread_running: bool,
    time_since_last_inference: float,
    inference_interval: float,
) -> bool:
    """Determine if a new inference should be triggered.

    Args:
        cached_chunk_exists: Whether we have a cached action chunk.
        inference_thread_running: Whether inference is currently running.
        time_since_last_inference: Time elapsed since last inference started (seconds).
        inference_interval: Minimum time between inferences (seconds).

    Returns:
        True if new inference should start.
    """
    # The busy check comes FIRST. With it second, a missing cached chunk re-dispatched on every
    # 20 ms tick while the first inference was still in flight: the worker drains the (maxsize=1)
    # queue at the START of a request, so a second one queued behind it and landed ~20 ticks
    # later carrying a tail built before the first chunk existed -- an anchor no longer in force,
    # which then overwrote the good chunk. `i` clears the cache, so this fired precisely at the
    # idle -> VLA handover: the largest discontinuity in a run, and the one RTC most needs to
    # get right.
    if inference_thread_running:
        return False
    if not cached_chunk_exists:
        return True
    return time_since_last_inference >= inference_interval


def build_prev_chunk_tail(
    cached_action_chunk: Any,
    action_chunk_index: int,
    last_published_token: Any,
    holding: bool,
    action_horizon: int | None = None,
) -> "np.ndarray | None":
    """The motion tokens this robot will execute if no new chunk ever arrives.

    This is ``A_prev`` for real-time chunking (arXiv:2506.07339): the plan currently in force,
    aligned so that ``tail[k]`` lands on the same controller tick as index ``k`` of the chunk
    being generated. The policy server needs it to make the next chunk continuous with what is
    actually being executed; it cannot derive it, because only the robot knows what it ran.

    Three situations, one rule:

    * **Running** -> ``motion_token[action_chunk_index:]``. Shrinks as the chunk is consumed,
      down to a single token once ``action_chunk_index`` hits the clamp at ``horizon - 1`` --
      which is correct, because from then on the loop re-publishes that last token.
    * **Holding** (paused, or just after 'i') -> ``[last_published_token]``. The cached chunk is
      NOT usable here: while paused, inference keeps landing and replaces the cache with plans
      that were never executed, while the robot physically holds the last token it published.
      Slicing the cache would describe a plan that never ran.
    * **Nothing published yet** -> ``None``; there is no previous plan to be continuous with.

    Padding to the model horizon and the guidance mask are deliberately left to the server: they
    depend on ``H`` and on the mask span, which are policy-side, whereas the tail is a fact about
    the robot. Note the server should right-pad by REPEATING the last row, since that is exactly
    what this loop does when a chunk runs out.

    Args:
        cached_action_chunk: The processed action dict currently being executed, or None.
        action_chunk_index: Index of the next token to publish from that chunk.
        last_published_token: The (D,) token most recently sent to the controller, or None.
        holding: True when the loop is not publishing from the chunk (paused / after 'i').

    Returns:
        (T, D) float32 tokens with T >= 1, or None if there is nothing to report.
    """
    if holding or cached_action_chunk is None:
        if last_published_token is None:
            return None
        return np.asarray(last_published_token, dtype=np.float32).reshape(1, -1)

    # Same two-key lookup as run_vla_inference.get_action_field; inlined to avoid importing
    # from the script (which imports this module).
    tokens = cached_action_chunk.get("motion_token")
    if tokens is None:
        tokens = cached_action_chunk.get("action.motion_token")
    if tokens is None:
        if last_published_token is None:
            return None
        return np.asarray(last_published_token, dtype=np.float32).reshape(1, -1)

    tokens = np.asarray(tokens, dtype=np.float32)
    while tokens.ndim > 2:  # (B, T, D) -> (T, D)
        tokens = tokens[0]
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        return None

    # Clamp against the horizon the publish loop actually walks, NOT the chunk length. They
    # differ when --action-horizon is left at its 40 default while the model emits 50: the loop
    # then stops at index 39 and re-publishes it forever, so advertising tokens 40..49 as "what
    # I will execute" would anchor the guidance to ticks that never happen.
    last = tokens.shape[0] if action_horizon is None else min(int(action_horizon), tokens.shape[0])
    last = max(last, 1)
    start = int(np.clip(action_chunk_index, 0, last - 1))
    return tokens[start:last]


def conservative_delay_ticks(delay_buffer, control_freq: float, action_horizon: int,
                             outlier_s: float = 1.0, percentile: float = 90.0,
                             fallback_ticks: int = 3) -> int:
    """Inference delay in controller ticks, estimated with an upper quantile.

    Real-time chunking freezes the first ``d`` actions of a new chunk to the previous plan,
    because those ticks will have elapsed before the chunk lands. The error is asymmetric, so
    the estimate is deliberately biased high: under-estimating ``d`` leaves an already-executed
    tick unfrozen and the discontinuity comes back, while over-estimating costs reactivity.

    This is the "estimate the next inference delay conservatively" step of Algorithm 1
    (arXiv:2506.07339), but as a **high quantile rather than the MAX**. A max is the most
    conservative statistic available and also the least robust: a single slow sample becomes the
    answer, and -- because it is re-read from the buffer rather than re-measured -- it is
    returned bit-identically on every subsequent request until it ages out. On the real robot one
    600 ms spike (~3x typical) pinned d at 30 ticks for a full buffer's worth of inferences,
    roughly 10 s, during which half of every chunk was frozen to a plan the robot had already
    finished executing. The tell is a *constant* d in the logs; a live measurement jitters.

    p90 keeps the upward bias -- it still sits above typical latency, so the seam stays closed --
    while a lone outlier in a buffer of 20 falls outside the quantile entirely and is ignored.
    Sustained latency growth still moves it, once more than a tenth of the window is slow.

    Args:
        delay_buffer: Recent measured inference delays in seconds (e.g. a deque).
        control_freq: Controller rate in Hz -- the token publish rate, 50 for SONIC.
        action_horizon: Chunk length, used to bound the result.
        outlier_s: Samples above this are dropped as non-predictive (see below).
        percentile: Quantile of the retained samples to use, in [0, 100].
        fallback_ticks: d to use when no sample is usable. Deliberately small but NOT zero.

    Returns:
        d in ticks, ``fallback_ticks`` when no delay has been observed yet.

    Note:
        The invariant that matters is ``d_pred >= d_act``, not equality. The robot enters the new
        chunk at the slot matching its *actual* delay, so over-predicting means it starts inside
        the pinned region and simply follows the old plan a little longer -- continuous, just less
        reactive. Under-predicting drops it past the pin, into slots the model generated on the
        assumption that the robot had already diverged onto the new plan; it had not, and that is
        the seam. Hence the upward bias, and hence a non-zero fallback: returning 0 when the buffer
        holds nothing usable pins nothing at all, which is a guaranteed seam at precisely the
        moment the estimate is least trustworthy.
    """
    # Drop non-predictive outliers first. The guided sampler is a separate XLA program, so the
    # first request carrying a previous chunk pays a one-off JIT compile of tens of seconds.
    # A compile is not a prediction of the next delay, so it does not belong in the estimate --
    # and at small buffer occupancy it would still drag the quantile up even though p90 alone
    # already discards a single outlier once the buffer is reasonably full.
    delays = [d for d in delay_buffer if d is not None and d <= outlier_s]
    if not delays:
        # Fail SAFE, not open. This fires on a cold buffer and when every sample is an outlier
        # (a stall long enough that RTC cannot rescue it anyway -- the chunk is only 0.8 s). The
        # old behaviour returned 0, i.e. "pin nothing", handing back the chunk-boundary
        # discontinuity exactly when the delay is least understood. A few ticks of pin keeps the
        # join closed and costs almost no reactivity if the true delay turns out to be small.
        return int(np.clip(fallback_ticks, 0, max(action_horizon - 1, 0)))
    estimate = float(np.percentile(delays, percentile))
    return calculate_latency_compensated_index(estimate, control_freq, action_horizon)
