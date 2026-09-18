"""Read the INSPIRE hands over Modbus TCP, in the layout the pi0.5 SONIC bridge expects.

WHY THIS EXISTS
---------------
The controller models hands as **dex3**: `G1ThreeFingerHand` (7 joints per hand) fed from the
DDS topics `rt/dex3/{left,right}/state` (`decoupled_wbc/.../state_processor.py`). Our G1 wears
**Inspire** hands, which are not on DDS at all -- they are Modbus TCP devices on the robot LAN.
So `prepare_observation_for_eval` slices `left_hand`/`right_hand` out of `whole_q` as 7 dex3
slots that no hardware is publishing, and the policy receives stale or zero values.

This module reads the Inspire hands directly and returns the **6 actuated DOF per hand** that
`deploy/real/pi05/server.py --hand-proprio inspire` wants. The bridge dispatches on WIDTH: 7 per
hand means dex3, 6 means Inspire, so simply overwriting the two state keys is enough.

CONVENTIONS (all verified against the sources named, except where flagged)
-------------------------------------------------------------------------
* Transport: one Modbus TCP client per hand, port 6000. Register 1546 holds the 6 measured
  finger positions; 1486 is the write side. Source: `g1-runner/src/examples/test_hands.py`.
* Register order is `[pinky, ring, middle, index, thumb_bend, thumb_rotate]` -- the REVERSE of
  the codec's `[thumb_yaw, thumb_bend, index, middle, ring, pinky]`
  (`humanoid-vla/data/README_HAND.md`, `deploy/real/common/hand_codec.py`).
* Register range is 0..1000 with **0 = closed, 1000 = open**, i.e. INVERTED w.r.t. joint angle,
  the same inversion the MuJoCo model uses (`hand_codec.inspire_rad_to_ctrl`).
* Angles are returned in radians in the URDF/MJCF joint convention, scaled by each joint's upper
  limit. The bridge applies the distal/proximal thumb rescale itself (`joint_to_codec`), so do
  NOT apply it here.

⚠ TWO THINGS TO VERIFY ON HARDWARE before trusting the numbers (both are cheap: move one finger
  at a time and watch which slot answers):
  1. the register order above -- guessing it wrong MIRRORS the hand, thumb <-> pinky;
  2. whether register 1486/1546 addresses the thumb's PROXIMAL joint (as the URDF does) or its
     DISTAL one (as the codec does). If distal, `THUMB_BEND_IS_DISTAL = True` below.
"""
from __future__ import annotations

import collections
import os
import pathlib
import threading
import time

import numpy as np

LEFT_HOST = os.environ.get("SONIC_INSPIRE_LEFT_HOST", "192.168.123.210")
RIGHT_HOST = os.environ.get("SONIC_INSPIRE_RIGHT_HOST", "192.168.123.211")
PORT = int(os.environ.get("SONIC_INSPIRE_PORT", "6000"))

REG_ACTUAL = 1546          # 6 measured finger positions (read)
REG_TARGET = 1486          # 6 finger setpoints (write)
N_FINGERS = 6
CTRL_MAX = 1000.0

# Modbus slot -> codec slot. Modbus is [pinky, ring, middle, index, thumb_bend, thumb_rotate];
# the codec is [thumb_yaw, thumb_bend, index, middle, ring, pinky] -- an exact reversal.
CODEC_FROM_MODBUS = np.array([5, 4, 3, 2, 1, 0])
MODBUS_FROM_CODEC = np.argsort(CODEC_FROM_MODBUS)   # the same reversal, written out

# Upper joint limits (rad) in CODEC slot order, read off the G1 mode15 MJCF. Lower limit is 0.
INSPIRE_LIMIT = np.array([1.1641, 0.5864, 1.4381, 1.4381, 1.4381, 1.4381], np.float32)

# Set True only if hardware testing shows the register drives the thumb's DISTAL joint; then the
# value is divided by 2.4 to reach the URDF's proximal convention (see data/README_HAND.md).
THUMB_BEND_IS_DISTAL = os.environ.get("SONIC_INSPIRE_THUMB_DISTAL", "0") == "1"
THUMB_BEND_SCALE = 2.4


def _rad_to_ctrl(q6) -> np.ndarray:
    """6 joint angles (rad, codec slot order, 0 = open) -> 6 Modbus registers.

    Exact inverse of `_ctrl_to_rad`: undo the thumb scale if the register is distal, normalise
    by each joint's limit, then INVERT (0 = closed, 1000 = open) and reorder to Modbus.
    """
    q = np.asarray(q6, np.float32)[:N_FINGERS].copy()
    if THUMB_BEND_IS_DISTAL:
        q[1] = q[1] * THUMB_BEND_SCALE
    frac = np.clip(q / INSPIRE_LIMIT, 0.0, 1.0)
    ctrl = np.rint(CTRL_MAX * (1.0 - frac))[MODBUS_FROM_CODEC]
    return np.clip(ctrl, 0, CTRL_MAX).astype(int)


def _ctrl_to_rad(regs) -> np.ndarray:
    """6 Modbus registers -> 6 joint angles (rad, codec slot order, 0 = open)."""
    ctrl = np.asarray(regs, np.float32)[:N_FINGERS][CODEC_FROM_MODBUS]
    q = INSPIRE_LIMIT * (1.0 - np.clip(ctrl, 0.0, CTRL_MAX) / CTRL_MAX)
    if THUMB_BEND_IS_DISTAL:
        q[1] = q[1] / THUMB_BEND_SCALE
    return q.astype(np.float32)


def _humanoid_vla_root():
    """Walk up to the humanoid-vla checkout that holds the shared deploy constants.

    Searched rather than counted: this file and run_vla_inference.py sit at different depths,
    and a hardcoded parents[N] silently resolves to the wrong directory from one of them.
    """
    here = pathlib.Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "deploy" / "real" / "common" / "sonic_constants.py").exists():
            return cand
    raise RuntimeError(
        f"cannot find the humanoid-vla root above {here}; SONIC_HAND_SPACE=dex3 needs "
        "deploy/real/common from the parent repo")


_CODEC = None


def to_dex3(left6, right6):
    """Inspire (6+6, joint convention) -> the 14-d dex3 vector, split 7 + 7.

    The GR00T handtoken checkpoints declare `left_hand`/`right_hand` as **7 dex3 joints** (state
    46 = 29 body + gravity(3) + 7 + 7), and their server consumes the state groups directly --
    there is no bridge to retarget for us, unlike pi0.5's `--hand-proprio inspire`. So do the
    same retarget here: encode the live Inspire pose, decode it in dex3 space. Identical call to
    the one the pi0.5 bridge makes, and identical to how training built the block
    (`_dex3_current`: decode the hand token with the dex3 decoder, frame 0).

    Without this, the raw 6-DOF Inspire vector gets sent straight into a state slot the
    checkpoint's normalization stats expect to be 7-wide -- IndexError: boolean index did not
    match indexed array along dimension 1; dimension is 6 but corresponding boolean dimension
    is 7 (normalize_values_minmax's mask is sized off the 7-wide dex3 stats).

    ORDER: the returned vector is the codec's INDEX-FIRST dex3 order, which is what the HE
    corpora store (`data/README_HAND.md`: "Humanoid-Everyday observation.state[:, 29:43] is
    already in the right order"). It is NOT permuted to the robot's thumb-first URDF order --
    these values never came from the robot's joints, they were generated in codec space.
    """
    global _CODEC
    if _CODEC is None:
        import sys
        repo = _humanoid_vla_root()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from deploy.real.common.hand_codec import HandCodec
        _CODEC = HandCodec()
    d = _CODEC.inspire_to_dex3(np.concatenate([np.asarray(left6, np.float32),
                                               np.asarray(right6, np.float32)]))
    return d[:7].astype(np.float32), d[7:].astype(np.float32)


class InspireHandReader:
    """Lazily-connected reader for both hands. Never raises into the control loop.

    ALL Modbus I/O happens on ONE dedicated thread (`_loop`, `SONIC_INSPIRE_HZ`, default 30).
    `read()` and `write()` are pure shared-memory accesses that never touch the socket:
    `read()` returns the newest values the thread managed to fetch, `write()` parks a target the
    thread picks up on its next tick (latest-wins, never queued).

    WHY A THREAD, not just a rate limit. These calls used to run inline, and pymodbus's TCP
    client is synchronous, so a hand that went quiet cost its full socket timeout -- seconds --
    charged to whichever loop called it. That landed in two bad places at once:
      * `read()` runs inside the INFERENCE worker, whose elapsed time IS the `delay_ticks` fed
        to real-time chunking. A stalled hand therefore inflated the RTC delay estimate and
        tripped the "delay > trained max; clamping" path, i.e. a hand fault degraded BODY
        control.
      * `write()` runs in the 50 Hz publish loop, so a stall there stalled body publishing
        outright.
    Rate-limiting (the previous fix) only reduced how OFTEN that exposure occurred; it could not
    remove it, because a due tick still blocked. Decoupling removes it: a stalled hand now costs
    stale values in the shared slot for the duration, and nothing else waits.

    Read and write failures share one `_fails` counter, and `_give_up` latches the hands off for
    the rest of the run after 3 consecutive failures -- unchanged, but now it stops the thread
    rather than the control loop.
    """

    def __init__(self, left_host: str = LEFT_HOST, right_host: str = RIGHT_HOST, port: int = PORT):
        self._hosts = (left_host, right_host)
        self._port = port
        self._clients = None
        self._warned = False
        self._unit_kw = None      # "slave" (pymodbus 3.x) or "device_id" (4.x); detected once
        self._fails = 0
        self._disabled = False
        self._announced = False

        # --- the shared slots between the Modbus thread and the control path ---------------
        self._hz = float(os.environ.get("SONIC_INSPIRE_HZ", "30"))
        self._period = 1.0 / self._hz
        self._lock = threading.Lock()
        self._latest = None       # newest successful read (left6, right6); None until first
        self._latest_t = 0.0      # time.monotonic() of that read, for staleness reporting
        self._target = None       # newest commanded (left6, right6); latest-wins, never queued
        self._thread = None
        self._stop = threading.Event()
        self._wrote_once = threading.Event()   # open_hands() waits on this at startup
        # Per-operation durations, so a stall can be attributed to a SPECIFIC hand rather than
        # to "the hands". Each is a window of recent seconds; `stats()` reduces them.
        self._t = {k: collections.deque(maxlen=200)
                   for k in ("read_left", "read_right", "write_left", "write_right", "tick")}
        self._overruns = 0        # ticks that took longer than the period

    def _connect(self) -> bool:
        if self._disabled:
            return False
        if self._clients is not None:
            return True
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            if not self._warned:
                print("[inspire] pymodbus not installed in this venv -- hand proprio disabled "
                      "(pip install pymodbus)", flush=True)
                self._warned = True
            return False
        clients = []
        for host in self._hosts:
            c = ModbusTcpClient(host, port=self._port)
            if not c.connect():
                if not self._warned:
                    print(f"[inspire] cannot reach hand at {host}:{self._port} -- hand proprio "
                          "disabled", flush=True)
                    self._warned = True
                for done in clients:
                    done.close()
                return False
            clients.append(c)
        self._clients = clients
        if self._unit_kw is None:
            # pymodbus renamed the unit-id argument between 3.x and 4.x. Pick whichever this
            # install accepts rather than guessing -- the wrong one raises on every call.
            import inspect
            params = inspect.signature(ModbusTcpClient.read_holding_registers).parameters
            self._unit_kw = "device_id" if "device_id" in params else "slave"
        if not self._announced:
            print(f"[inspire] hands connected: {self._hosts[0]} / {self._hosts[1]} "
                  f"(unit kwarg: {self._unit_kw})", flush=True)
            self._announced = True
        return True

    def _give_up(self, why: str) -> None:
        """Stop after repeated failures instead of reconnecting on every tick.

        A reconnect storm is worse than being off: it printed a line per tick and eventually
        got the hands to refuse connections outright.
        """
        self._clients = None
        self._fails += 1
        if self._fails >= 3 and not self._disabled:
            self._disabled = True
            print(f"[inspire] giving up after {self._fails} failures ({why}); hands DISABLED "
                  "for this run", flush=True)

    # --- the Modbus thread ----------------------------------------------------------------

    def start(self) -> None:
        """Spawn the Modbus thread. Idempotent; called lazily by read()/write()."""
        if self._thread is not None or self._disabled:
            return
        self._thread = threading.Thread(target=self._loop, name="inspire-modbus", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Own the sockets; read both hands and push the latest target, every `_period`.

        DAEMON + latest-wins by design. Nothing here is allowed to make the caller wait: a hand
        that stalls for its full socket timeout costs stale values in `_latest` for that long,
        and the control loop keeps publishing body joints at 50 Hz throughout.
        """
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self._connect():
                # Not reachable (or pymodbus missing / given up). Don't spin: _connect already
                # printed once, and _give_up latches _disabled.
                if self._disabled:
                    return
                self._stop.wait(self._period)
                continue

            left = self._read_one(0, "read_left")
            right = self._read_one(1, "read_right")
            if left is not None and right is not None:
                with self._lock:
                    self._latest, self._latest_t = (left, right), time.monotonic()
                self._fails = 0

            with self._lock:
                target = self._target
            if target is not None:
                ok_l = self._write_one(0, target[0], "write_left")
                ok_r = self._write_one(1, target[1], "write_right")
                if ok_l and ok_r:
                    self._wrote_once.set()
                    self._fails = 0

            dt = time.monotonic() - t0
            self._t["tick"].append(dt)
            if dt > self._period:
                self._overruns += 1
            self._stop.wait(max(0.0, self._period - dt))

    def _read_one(self, idx: int, key: str):
        c = self._clients[idx] if self._clients else None
        if c is None:
            return None
        t0 = time.monotonic()
        try:
            rr = c.read_holding_registers(REG_ACTUAL, count=N_FINGERS, **{self._unit_kw: 1})
        except Exception as exc:                          # transport hiccup: drop the sample
            self._t[key].append(time.monotonic() - t0)
            if self._fails == 0:
                print(f"[inspire] read failed ({exc})", flush=True)
            self._give_up(str(exc))
            return None
        self._t[key].append(time.monotonic() - t0)
        if rr is None or getattr(rr, "isError", lambda: True)():
            self._give_up("modbus error response")
            return None
        return _ctrl_to_rad(rr.registers)

    def _write_one(self, idx: int, q6, key: str) -> bool:
        c = self._clients[idx] if self._clients else None
        if c is None:
            return False
        t0 = time.monotonic()
        try:
            rr = c.write_registers(REG_TARGET, _rad_to_ctrl(q6).tolist(), **{self._unit_kw: 1})
        except Exception as exc:
            self._t[key].append(time.monotonic() - t0)
            if self._fails == 0:
                print(f"[inspire] write failed ({exc})", flush=True)
            self._give_up(str(exc))
            return False
        self._t[key].append(time.monotonic() - t0)
        if rr is not None and getattr(rr, "isError", lambda: False)():
            self._give_up("modbus error response")
            return False
        return True

    # --- the control-path API. NEITHER of these touches Modbus. ---------------------------

    def read(self):
        """Latest hand reading, or None if there has never been one. NEVER blocks.

        Returns whatever the Modbus thread last managed to read. During a device stall this
        returns the pre-stall values (stale, not wrong) instead of holding up the caller --
        which matters because this runs inside the INFERENCE worker, and that worker's elapsed
        time is measured as `delay_ticks` and fed to real-time chunking. A blocking read here
        used to charge a multi-second Modbus timeout straight to the RTC delay estimate.
        """
        self.start()
        with self._lock:
            return self._latest

    def write(self, left6, right6) -> bool:
        """Hand the Modbus thread a new target. NEVER blocks; returns False only when disabled.

        Latest-wins: a target set between two thread ticks replaces the previous one rather than
        queueing, so the hands always track the newest command and can never fall behind by a
        backlog. Called from the 50 Hz publish loop while the thread runs at `SONIC_INSPIRE_HZ`.
        """
        if self._disabled:
            return False
        self.start()
        with self._lock:
            self._target = (np.asarray(left6, np.float32).copy(),
                            np.asarray(right6, np.float32).copy())
        return True

    def open_hands(self, wait_s: float = 1.0) -> bool:
        """Park both hands OPEN -- the rest pose the policy's state assumes at episode start.

        Unlike the control-path writes this one WAITS (briefly) for the thread to confirm, since
        it runs at startup/'i' rather than in the loop, and the caller uses the return value to
        report whether the hands actually moved.
        """
        self._wrote_once.clear()
        if not self.write(np.zeros(N_FINGERS, np.float32), np.zeros(N_FINGERS, np.float32)):
            return False
        return self._wrote_once.wait(timeout=wait_s)

    def stats(self) -> dict:
        """Per-hand Modbus timing for the diagnostics line. Cheap; safe to call every chunk.

        Reports last and max over the recent window for each of the four operations separately,
        because "the hands are slow" is not actionable -- which hand, and reading or writing, is.
        """
        def _ms(key):
            d = self._t[key]
            return (0.0, 0.0) if not d else (d[-1] * 1e3, max(d) * 1e3)
        with self._lock:
            age = (time.monotonic() - self._latest_t) if self._latest is not None else float("nan")
        out = {"age_ms": age * 1e3, "hz": self._hz, "overruns": self._overruns,
               "disabled": self._disabled}
        for k in ("read_left", "read_right", "write_left", "write_right", "tick"):
            out[k] = _ms(k)
        return out

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for c in self._clients or []:
            c.close()
        self._clients = None
