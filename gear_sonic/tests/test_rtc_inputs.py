"""Robot-side inputs for real-time chunking (arXiv:2506.07339).

The policy server cannot derive either of these: only the robot knows what it actually executed
and how long inference took. Getting `prev_chunk_tail` misaligned by even one tick reintroduces
exactly the chunk-boundary discontinuity RTC exists to remove, and it would do so silently --
the robot would still walk, just less smoothly -- so the alignment is pinned down here.
"""

import numpy as np

from gear_sonic.utils.inference.vla_utils import (
    build_prev_chunk_tail,
    conservative_delay_ticks,
    should_trigger_new_inference,
)

HORIZON, DIM = 50, 64


def _chunk(prefix="action."):
    """A (1, H, D) chunk whose row t is filled with the value t, so alignment is checkable."""
    tokens = np.tile(np.arange(HORIZON, dtype=np.float32)[:, None], (1, DIM))
    return {f"{prefix}motion_token": tokens[None]}


def _held(value=7.0):
    return np.full(DIM, value, dtype=np.float32)


def test_running_tail_is_the_unexecuted_remainder_and_is_aligned():
    tail = build_prev_chunk_tail(_chunk(), 20, _held(), holding=False)
    assert tail.shape == (HORIZON - 20, DIM)
    # tail[k] must be the token for the same tick as index k of the chunk being generated.
    assert tail[0][0] == 20.0
    assert tail[-1][0] == HORIZON - 1


def test_tail_shrinks_to_one_token_at_the_clamp():
    # The publish loop clamps at horizon-1 and re-sends that token, so a one-row tail is the
    # honest answer, not a degenerate case.
    tail = build_prev_chunk_tail(_chunk(), HORIZON - 1, _held(), holding=False)
    assert tail.shape == (1, DIM)
    assert tail[0][0] == HORIZON - 1


def test_paused_reports_the_held_token_not_the_cached_chunk():
    # While paused, inference keeps landing and replaces the cache with plans that were never
    # executed; the robot is physically holding the last token it published. Slicing the cache
    # here would describe a plan that never ran.
    tail = build_prev_chunk_tail(_chunk(), 20, _held(3.5), holding=True)
    assert tail.shape == (1, DIM)
    assert np.all(tail[0] == 3.5)


def test_after_initial_pose_there_is_no_cache_but_still_a_plan():
    # 'i' clears the cache and holds the idle token; that token is what the next chunk must be
    # continuous with.
    tail = build_prev_chunk_tail(None, 0, _held(1.25), holding=True)
    assert tail.shape == (1, DIM)
    assert np.all(tail[0] == 1.25)


def test_nothing_published_yet_reports_no_plan():
    assert build_prev_chunk_tail(None, 0, None, holding=True) is None
    assert build_prev_chunk_tail(None, 0, None, holding=False) is None


def test_unprefixed_key_and_unbatched_chunk_are_accepted():
    assert build_prev_chunk_tail(_chunk(prefix=""), 10, _held(), holding=False).shape == (40, DIM)
    flat = {"motion_token": _chunk()["action.motion_token"][0]}  # (H, D), no batch dim
    assert build_prev_chunk_tail(flat, 10, _held(), holding=False).shape == (40, DIM)


def test_index_past_the_end_is_clamped_not_an_empty_tail():
    tail = build_prev_chunk_tail(_chunk(), HORIZON + 5, _held(), holding=False)
    assert tail.shape == (1, DIM)


def test_delay_estimate_is_the_max_not_the_mean():
    # Under-estimating d leaves an already-executed tick unfrozen and the seam comes back;
    # over-estimating only costs reactivity. Algorithm 1 takes max over the buffer.
    assert conservative_delay_ticks([0.02, 0.30, 0.04], 50, HORIZON) == 15
    assert conservative_delay_ticks([0.02], 50, HORIZON) == 1


def test_delay_estimate_handles_cold_start_and_is_bounded_by_the_horizon():
    assert conservative_delay_ticks([], 50, HORIZON) == 0
    # Within the plausible band it is still clamped to the horizon.
    assert conservative_delay_ticks([0.9], 50, HORIZON, outlier_s=1.0) == 45


def test_jit_compile_outlier_does_not_pin_the_delay_estimate():
    # The guided sampler is a separate XLA program, so the first request carrying a previous
    # chunk pays a one-off compile of tens of seconds. As a plain MAX over the buffer that
    # would pin d at the clamp for the next `maxlen` inferences and freeze the robot on a
    # nearly-constant plan. A compile is not a prediction of the next delay.
    steady = [0.05, 0.06, 0.05]
    assert conservative_delay_ticks([60.0, *steady], 50, HORIZON) == 3   # == max(steady)
    assert conservative_delay_ticks([60.0], 50, HORIZON) == 0            # nothing usable yet


def test_tail_is_clamped_to_the_publish_horizon_not_the_chunk_length():
    # --action-horizon 40 against a 50-token model: the loop stops at 39 and re-publishes it,
    # so tokens 40..49 must not be advertised as "what I will execute".
    tail = build_prev_chunk_tail(_chunk(), 20, _held(), holding=False, action_horizon=40)
    assert tail.shape == (20, DIM)
    assert tail[-1][0] == 39.0


def test_no_redispatch_while_an_inference_is_already_running():
    # Regression: with the cache-empty check first, a missing chunk re-dispatched every tick
    # while the first inference was in flight, and the second request carried a stale anchor.
    # 'i' clears the cache, so this fired at the idle -> VLA handover.
    assert not should_trigger_new_inference(
        cached_chunk_exists=False, inference_thread_running=True,
        time_since_last_inference=0.02, inference_interval=0.4,
    )
    # Still dispatches immediately when nothing is running and there is no plan.
    assert should_trigger_new_inference(
        cached_chunk_exists=False, inference_thread_running=False,
        time_since_last_inference=0.02, inference_interval=0.4,
    )
