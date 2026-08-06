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


def prepare_observation_for_eval(robot_model: RobotModel, obs: dict) -> dict:
    """Split whole-body ``q`` into per-joint-group state keys for the policy.

    Populates ``obs["state"]`` with ``left_arm``, ``right_arm``, ``waist``,
    ``left_leg``, ``right_leg``, ``left_hand``, ``right_hand`` sub-keys
    using the nested dict format expected by ``Gr00tPolicy``.

    Args:
        robot_model: RobotModel instance.
        obs: Observation dict containing ``"q"`` key and a ``"state"`` sub-dict.

    Returns:
        Modified observation dict with ``obs["state"]`` populated.
    """
    assert "q" in obs, "q is not in the observation"

    whole_q = obs["q"]
    assert whole_q.shape[-1] == robot_model.num_joints, "q has wrong shape"

    if "state" not in obs:
        obs["state"] = {}

    obs["state"]["left_arm"] = whole_q[..., robot_model.get_joint_group_indices("left_arm")]
    obs["state"]["right_arm"] = whole_q[..., robot_model.get_joint_group_indices("right_arm")]
    obs["state"]["waist"] = whole_q[..., robot_model.get_joint_group_indices("waist")]
    obs["state"]["left_leg"] = whole_q[..., robot_model.get_joint_group_indices("left_leg")]
    obs["state"]["right_leg"] = whole_q[..., robot_model.get_joint_group_indices("right_leg")]
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
    if not cached_chunk_exists:
        return True
    if inference_thread_running:
        return False
    return time_since_last_inference >= inference_interval


def build_prev_chunk_tail(
    cached_action_chunk: Any,
    action_chunk_index: int,
    last_published_token: Any,
    holding: bool,
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

    start = int(np.clip(action_chunk_index, 0, tokens.shape[0] - 1))
    return tokens[start:]


def conservative_delay_ticks(delay_buffer, control_freq: float, action_horizon: int) -> int:
    """Inference delay in controller ticks, estimated pessimistically.

    Real-time chunking freezes the first ``d`` actions of a new chunk to the previous plan,
    because those ticks will have elapsed before the chunk lands. Under-estimating ``d`` leaves
    an already-executed tick unfrozen and the discontinuity comes back; over-estimating only
    costs reactivity (the new plan takes effect a little later). So take the MAX over recent
    delays, per Algorithm 1 of arXiv:2506.07339 ("estimate the next inference delay
    conservatively"), not the mean.

    Args:
        delay_buffer: Recent measured inference delays in seconds (e.g. a deque).
        control_freq: Controller rate in Hz -- the token publish rate, 50 for SONIC.
        action_horizon: Chunk length, used to bound the result.

    Returns:
        d in ticks, 0 when no delay has been observed yet.
    """
    delays = [d for d in delay_buffer if d is not None]
    if not delays:
        return 0
    return calculate_latency_compensated_index(max(delays), control_freq, action_horizon)
