"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> blend smoothly to initial pose (or snap if no prior token) and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

import collections
import os
import pathlib
from dataclasses import dataclass
import queue
import threading
import time

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    build_prev_chunk_tail,
    calculate_latency_compensated_index,
    concat_action,
    conservative_delay_ticks,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    initial_pose_blend_duration: float = 2.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    # Real-time chunking (arXiv:2506.07339)
    rtc: bool = True
    """Send the previous action chunk's remaining tail and the inference delay to the policy
    server, so it can generate the next chunk continuously with the one being executed. Costs
    one extra array per request. A server that does not implement RTC ignores the options and
    behaves exactly as before, so this is safe to leave on."""

    rtc_delay_buffer_size: int = 20
    """How many recent inference delays to keep. `d` is the p90 over this window. Needs to be
    large enough that a single slow sample sits outside the quantile -- at 20, one outlier is
    ignored outright; at 10 it still carries about a tenth of the estimate."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    # WIDTH 6 OR 7. 7 = dex3, the space the C++ deploy actuates over DDS. 6 = Inspire, which
    # nothing downstream actuates (those hands are Modbus, driven in this process) -- but the
    # values still have to travel, because the data exporter reads its hand columns off this
    # message and a VLA run has no teleop stream to fall back on. Rejecting 6 here is what left
    # `teleop.left/right_hand_joints` all-zero in every recorded VLA episode: the caller nulled
    # them rather than crash, so the policy's own hand commands were never written down.
    #
    # Sent under `vla_*` names, NOT `left_hand_joints`. The C++ ZMQEndpointInterface validates
    # the canonical names against its dex3 shape and logs two lines per tick when they are 6
    # wide -- at 50 Hz that is 100 lines/s of stderr inside a realtime control loop, which is a
    # cost the loop should not pay for a field it then discards. Unknown field names are skipped
    # by the header walk without comment, so the values reach the exporter and the controller
    # never sees them.
    for name, val in (("vla_left_hand_joints", left_hand_joints),
                      ("vla_right_hand_joints", right_hand_joints)):
        if val is None:
            continue
        val = np.asarray(val, dtype=np.float32)
        if val.ndim == 1:
            if val.shape[0] not in (6, 7):
                raise ValueError(
                    f"{name} must have shape [6] (Inspire) or [7] (dex3), got {val.shape}"
                )
            val = val.reshape(1, -1)
        pose_data[name] = val

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str, required: bool = True):
    """Get action field from dict, checking both with and without 'action.' prefix.

    Body-only models (and robots without hands) emit no hand fields, so callers
    pass ``required=False`` for those and get ``None`` instead of an error.
    """
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    if not required:
        return None
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


def select_action_step(array, index: int):
    """Reduce a ``(B, T, D)`` or ``(T, D)`` action array to the entry at ``index``.

    Returns ``None`` for absent optional fields, so hand joints stay unset all
    the way through to the ZMQ message.
    """
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2:
        array = array[min(index, array.shape[0] - 1)]
    return array


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------



_BODY_GROUPS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")   # 6+6+3+7+7 = 29
_DEFAULT_MJ = None


def _to_q_dev(observation):
    """Subtract the SONIC stance from the body groups, in place.

    WHY: the GR00T checkpoints are trained on q_dev. `roboxperience_converter/convert.py`
    writes `observation.state = remap(q_dev)[29] + gravity_body[3]`, with
    `q_dev = q_il - DEFAULT_ANGLES_IL`, and their dataset statistics agree (knee mean -0.046,
    not the +0.669 an absolute stance reads). Nothing in the GR00T serving path subtracts it --
    unlike pi0.5, whose bridge does it in `build_state32` -- so the robot has to.

    Groups only, never gravity or the hands. Each group already arrives in SONIC intra-group
    order, so concatenating the five is exactly the DEFAULT_MJ layout: no permutation.

    NB the SONIC DECODER's own 994-d observation is a different vector and is already
    stance-subtracted by the C++ deploy; this does not touch it.
    """
    global _DEFAULT_MJ
    if _DEFAULT_MJ is None:
        import sys
        repo = pathlib.Path(__file__).resolve().parents[5]        # .../humanoid-vla
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from deploy.real.common.sonic_constants import DEFAULT_MJ
        _DEFAULT_MJ = DEFAULT_MJ
    off = 0
    for g in _BODY_GROUPS:
        v = observation["state"].get(g)
        if v is None:
            raise KeyError(f"SONIC_STATE_Q=dev needs observation['state']['{g}']")
        n = np.asarray(v).shape[-1]
        observation["state"][g] = np.asarray(v, np.float32) - _DEFAULT_MJ[off:off + n]
        off += n
    if off != 29:
        raise ValueError(f"body groups summed to {off} dof, expected 29")
    return observation


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
    inspire_reader=None,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    # INSPIRE hands: overwrite the dex3-shaped slices with what the hands actually report.
    # `prepare_observation_for_eval` builds left_hand/right_hand from whole_q, i.e. 7 dex3 slots
    # fed by the rt/dex3/* topics -- which nothing publishes on a G1 wearing Inspire hands. The
    # pi0.5 bridge dispatches on width (7 = dex3, 6 = Inspire), so replacing them with the real
    # 6-DOF vectors is all that is needed for --hand-proprio inspire. Left untouched when the
    # reader is absent or the hands are unreachable: stale-but-shaped beats a fabricated pose.
    if inspire_reader is not None:
        hands = inspire_reader.read()
        if hands is not None:
            left6, right6 = hands
            if os.environ.get("SONIC_HAND_SPACE", "inspire") == "dex3":
                # GR00T handtoken checkpoints want 7 dex3 joints per hand and their server does
                # no retargeting; pi0.5's bridge does it itself from the 6-DOF vector. Same
                # codec either way -- only the side that calls it differs.
                from gear_sonic.utils.inference.inspire_hands import to_dex3

                left, right = to_dex3(left6, right6)
            else:
                left, right = left6, right6
            observation["state"]["left_hand"] = left[np.newaxis, np.newaxis]
            observation["state"]["right_hand"] = right[np.newaxis, np.newaxis]

    if os.environ.get("SONIC_STATE_Q", "absolute") == "dev":
        observation = _to_q_dev(observation)

    return observation


def run_policy_inference_and_process(policy, observation, robot_model, options=None):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Args:
        options: Optional dict forwarded verbatim to the server. Carries the real-time-chunking
            inputs (``prev_chunk_tail``, ``delay_ticks``); the server ignores it if it does not
            implement RTC, so this stays compatible with a stock policy server.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, info = policy.get_action(observation, options)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        # The server bounds tokens to the FSQ range before returning, so this check on the
        # RECEIVED chunk can no longer fire on its own. It reports the pre-clip magnitude
        # instead; use it when present, so a diverging chunk is still rejected. This matters
        # specifically under real-time chunking, whose guidance is what can blow tokens up
        # while making the join look perfect.
        token_max = (info or {}).get("token_max_preclip")
        token_max = float(token_max) if token_max is not None else float(np.abs(action[motion_key]).max())
        if token_max > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({token_max:.4f}) > 1.25. Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                # The queue item is the request options captured by the main loop AT DISPATCH
                # (see the trigger site). For real-time chunking these must be sampled when
                # inference starts, not when it returns: `prev_chunk_tail` has to be sliced at
                # the index being executed right now, so that tail[k] lines up with index k of
                # the chunk about to be generated. Legacy callers may still enqueue None.
                dispatch_time, options = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                # Measure from DISPATCH, not from here. `prev_chunk_tail` was sliced in the main
                # loop at dispatch, so tail[k] is the tick of dispatch+k; timing from after
                # prepare_obs_fn() would omit the observation-build time and under-estimate the
                # delay, which shifts the chunk earlier than the guidance placed it -- the
                # direction that brings the chunk-boundary jump back.
                inference_start_time = dispatch_time
                processed_action = inference_fn(observation, options)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # INSPIRE hand proprio. Off unless SONIC_INSPIRE_HANDS=1, because it is only correct on a G1
    # that actually wears Inspire hands: it replaces the (dex3-shaped, unpublished) left_hand /
    # right_hand state with the 6-DOF vectors read over Modbus. Env-gated rather than a CLI flag
    # because launch_inference.py builds its pane command from a fixed string.
    inspire_reader = None
    warned_hand_width = False
    if os.environ.get("SONIC_INSPIRE_HANDS", "0") == "1":
        from gear_sonic.utils.inference.inspire_hands import InspireHandReader

        inspire_reader = InspireHandReader()
        inspire_reader.open_hands()      # start from REST, as the policy's state assumes
        print("[inspire] hands ENABLED (SONIC_INSPIRE_HANDS=1): proprio in, targets out",
              flush=True)

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        nonlocal last_sent_motion_token
        print("Moving to initial pose")
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        # The controller holds the last token it received, so from here the robot is executing
        # this one until the policy loop resumes. Record it: it is the previous plan that
        # real-time chunking makes the first post-'i' chunk continuous with.
        last_sent_motion_token = np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32).copy()
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(
                np.float32
            )
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate
    # Latches the "waiting for the first chunk" notice so it prints once per stall rather than on
    # every 20 ms tick. Cleared whenever a chunk lands, so each new stall reports itself once.
    logged_awaiting_chunk = False
    # The pause path now ticks at the full control rate, so its notice needs throttling in time
    # rather than by the loop period that used to gate it.
    PAUSE_NOTICE_INTERVAL_S = 2.0
    last_pause_notice = 0.0

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    # Real-time chunking: recent inference delays (seconds). `d` is the p90 over this window --
    # biased high, because under-estimating leaves an already-executed tick unfrozen and the
    # chunk-boundary discontinuity returns, but not the MAX, which let one slow sample own the
    # estimate for a whole buffer's worth of inferences. See vla_utils.conservative_delay_ticks.
    rtc_delay_buffer: collections.deque = collections.deque(maxlen=config.rtc_delay_buffer_size)

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            zmq_frame_counter = 0
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk, reset frame counter")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
                inspire_reader=inspire_reader,
            ),
            lambda obs, options=None: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
                options=options,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                rtc_delay_buffer.append(inference_delay)  # feeds the conservative `d` estimate
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                logged_awaiting_chunk = False
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s)"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                # Build the RTC options HERE, at dispatch. `prev_chunk_tail` must be sliced at
                # the index being executed right now so that tail[k] lands on the same tick as
                # index k of the chunk about to be generated; sampling it when the reply arrives
                # would be off by the whole inference delay.
                request_options = None
                if config.rtc:
                    prev_tail = build_prev_chunk_tail(
                        cached_action_chunk=cached_action_chunk,
                        action_chunk_index=action_chunk_index,
                        last_published_token=last_sent_motion_token,
                        # While paused (and right after 'i') the cache holds chunks that were
                        # never executed -- the robot is holding the last token it published.
                        holding=pause_loop,
                        action_horizon=config.action_horizon,
                    )
                    if prev_tail is not None:
                        request_options = {
                            "prev_chunk_tail": prev_tail,
                            "delay_ticks": conservative_delay_ticks(
                                rtc_delay_buffer,
                                config.action_publish_rate,
                                config.action_horizon,
                            ),
                        }
                try:
                    inference_queue.put_nowait((time.monotonic(), request_options))
                except queue.Full:
                    pass

            if pause_loop:
                # Tick at the SAME cadence as the running path. Inference keeps firing while
                # paused, and the measured delay includes the wait for this loop to collect the
                # result -- so a slower pause loop does not merely idle, it inflates every delay
                # sample taken while paused and feeds the fiction into the RTC estimate. Measured
                # on the robot: 0.120 s running became 0.200-0.300 s paused, purely from a 0.2 s
                # sleep here. `_sleep_remaining` also keeps one definition of the cadence, so this
                # cannot drift from --action-publish-rate.
                if time.monotonic() - last_pause_notice >= PAUSE_NOTICE_INTERVAL_S:
                    print("Paused (policy loop idle, C++ loop still running)", flush=True)
                    last_pause_notice = time.monotonic()
                _sleep_remaining(t_start, loop_period)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    # Normal at the 'k' -> 'i' -> 'p' handover: 'i' clears the cache, so resuming
                    # leaves one inference delay with no plan to publish. The controller holds the
                    # last token it received (the initial pose), and real-time chunking anchors the
                    # first chunk to exactly that token, so this is a defined state, not a fault.
                    if not logged_awaiting_chunk:
                        if last_sent_motion_token is None:
                            print("Waiting for the first action chunk (nothing published yet).",
                                  flush=True)
                        else:
                            print("Waiting for the first action chunk; the controller is holding "
                                  "the last token published (after 'i', the initial pose).",
                                  flush=True)
                        logged_awaiting_chunk = True
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]

                    # Optional: body-only models emit no hand joints. Left as
                    # None, they are simply omitted from the v4 pose message.
                    left_hand_joints = select_action_step(
                        get_action_field(
                            processed_action, "left_hand_joints", required=False
                        ),
                        current_idx,
                    )
                    right_hand_joints = select_action_step(
                        get_action_field(
                            processed_action, "right_hand_joints", required=False
                        ),
                        current_idx,
                    )

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    # INSPIRE hands: the v4 message carries these targets, but the C++ deploy
                    # only actuates dex3 over DDS -- our hands are Modbus devices, so nothing
                    # downstream would move them. Drive them here, from the same per-tick target
                    # that goes on the wire, so the two can never disagree. Rate-limited inside.
                    if inspire_reader is not None and left_hand_joints is not None:
                        inspire_reader.write(left_hand_joints, right_hand_joints)

                    # 6-DOF Inspire targets now RIDE the v4 message (pack_latent_action_message
                    # takes 6 or 7). Nothing downstream actuates them -- the C++ deploy drives
                    # dex3 over DDS and these hands are Modbus, already written above -- but the
                    # data exporter reads its hand columns off this message, so dropping them is
                    # what made `teleop.left/right_hand_joints` all-zero in every VLA recording.
                    # The width travels with the values, so a consumer can tell the two spaces
                    # apart; see run_data_exporter._vla_hand_joints.
                    if (left_hand_joints is not None
                            and np.asarray(left_hand_joints).shape[-1] == 6
                            and inspire_reader is None and not warned_hand_width):
                        print("[inspire] server is sending 6-DOF Inspire hand targets but "
                              "SONIC_INSPIRE_HANDS is not set, so nothing drives the hands "
                              "-- recording them, not actuating them", flush=True)
                        warned_hand_width = True

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    last_sent_motion_token = motion_token.copy()
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
