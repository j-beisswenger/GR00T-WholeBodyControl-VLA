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


class _HandChannel:
    """ONE hand: its own socket, its own thread, its own failure latch, its own timings.

    The two hands are separate TCP endpoints (192.168.123.210 / .211) -- separate devices, not
    two unit-ids on a shared bus -- so nothing about them needs serialising. Giving each its own
    thread buys three things over a single shared one:

      1. ISOLATION IN TIME. A shared thread reads left then right, so a left hand stalling for
         its socket timeout also froze the right hand's values for that long. Now it doesn't.
      2. ISOLATION IN FAILURE. `_fails`/`_disabled` used to be shared, so three failures on ONE
         hand latched BOTH off for the rest of the run. Now a dead left hand leaves the right
         one running.
      3. OVERLAP. pymodbus's socket calls are blocking but release the GIL, so two threads
         genuinely overlap. A tick used to cost read_l + read_r + write_l + write_r summed;
         now each hand costs read + write and the two run concurrently, so the wall-clock tick
         is the max of the two hands rather than the sum of all four.
    """

    def __init__(self, name: str, host: str, port: int, period: float):
        self.name, self.host, self.port = name, host, port
        self._period = period
        self._client = None
        self._unit_kw = None       # "slave" (pymodbus 3.x) or "device_id" (4.x); detected once
        self._warned = False
        self._announced = False
        self._fails = 0
        self._disabled = False

        self._lock = threading.Lock()
        self._latest = None        # newest successful read (6,), None until the first one
        self._latest_t = 0.0
        self._target = None        # newest commanded (6,); latest-wins, never queued
        self._thread = None
        self._stop = threading.Event()
        self._wrote_once = threading.Event()
        self._t = {k: collections.deque(maxlen=200) for k in ("read", "write", "tick")}
        self._overruns = 0

    # --- thread side ----------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None or self._disabled:
            return
        self._thread = threading.Thread(target=self._loop, name=f"inspire-{self.name}",
                                        daemon=True)
        self._thread.start()

    def _connect(self) -> bool:
        if self._disabled:
            return False
        if self._client is not None:
            return True
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            if not self._warned:
                print("[inspire] pymodbus not installed in this venv -- hand proprio disabled "
                      "(pip install pymodbus)", flush=True)
                self._warned = True
            return False
        c = ModbusTcpClient(self.host, port=self.port)
        if not c.connect():
            if not self._warned:
                print(f"[inspire] cannot reach {self.name} hand at {self.host}:{self.port} -- "
                      "that hand disabled", flush=True)
                self._warned = True
            return False
        self._client = c
        if self._unit_kw is None:
            # pymodbus renamed the unit-id argument between 3.x and 4.x. Pick whichever this
            # install accepts rather than guessing -- the wrong one raises on every call.
            import inspect
            params = inspect.signature(ModbusTcpClient.read_holding_registers).parameters
            self._unit_kw = "device_id" if "device_id" in params else "slave"
        if not self._announced:
            print(f"[inspire] {self.name} hand connected: {self.host} "
                  f"(unit kwarg: {self._unit_kw})", flush=True)
            self._announced = True
        return True

    def _give_up(self, why: str) -> None:
        """Latch THIS hand off after repeated failures; the other hand is untouched.

        A reconnect storm is worse than being off: it printed a line per tick and eventually got
        the hands to refuse connections outright.
        """
        self._client = None
        self._fails += 1
        if self._fails >= 3 and not self._disabled:
            self._disabled = True
            print(f"[inspire] giving up on the {self.name} hand after {self._fails} failures "
                  f"({why}); {self.name} DISABLED for this run", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self._connect():
                if self._disabled:
                    return
                self._stop.wait(self._period)
                continue

            q = self._read_once()
            if q is not None:
                with self._lock:
                    self._latest, self._latest_t = q, time.monotonic()
                self._fails = 0

            with self._lock:
                target = self._target
            if target is not None and self._write_once(target):
                self._wrote_once.set()
                self._fails = 0

            dt = time.monotonic() - t0
            self._t["tick"].append(dt)
            if dt > self._period:
                self._overruns += 1
            self._stop.wait(max(0.0, self._period - dt))

    def _read_once(self):
        t0 = time.monotonic()
        try:
            rr = self._client.read_holding_registers(REG_ACTUAL, count=N_FINGERS,
                                                     **{self._unit_kw: 1})
        except Exception as exc:                      # transport hiccup: drop the sample
            self._t["read"].append(time.monotonic() - t0)
            if self._fails == 0:
                print(f"[inspire] {self.name} read failed ({exc})", flush=True)
            self._give_up(str(exc))
            return None
        self._t["read"].append(time.monotonic() - t0)
        if rr is None or getattr(rr, "isError", lambda: True)():
            self._give_up("modbus error response")
            return None
        return _ctrl_to_rad(rr.registers)

    def _write_once(self, q6) -> bool:
        t0 = time.monotonic()
        try:
            rr = self._client.write_registers(REG_TARGET, _rad_to_ctrl(q6).tolist(),
                                              **{self._unit_kw: 1})
        except Exception as exc:
            self._t["write"].append(time.monotonic() - t0)
            if self._fails == 0:
                print(f"[inspire] {self.name} write failed ({exc})", flush=True)
            self._give_up(str(exc))
            return False
        self._t["write"].append(time.monotonic() - t0)
        if rr is not None and getattr(rr, "isError", lambda: False)():
            self._give_up("modbus error response")
            return False
        return True

    # --- control-path side: shared memory only, never the socket --------------------------

    def read(self):
        self.start()
        with self._lock:
            return self._latest

    def write(self, q6) -> bool:
        if self._disabled:
            return False
        self.start()
        with self._lock:
            self._target = np.asarray(q6, np.float32).copy()
        return True

    def stats(self) -> dict:
        def _ms(key):
            d = self._t[key]
            return (0.0, 0.0) if not d else (d[-1] * 1e3, max(d) * 1e3)
        with self._lock:
            age = (time.monotonic() - self._latest_t) if self._latest is not None else float("nan")
        return {"age_ms": age * 1e3, "overruns": self._overruns, "disabled": self._disabled,
                "read": _ms("read"), "write": _ms("write"), "tick": _ms("tick")}

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None


class InspireHandReader:
    """Both Inspire hands, each on its OWN Modbus thread, behind shared latest-value slots.

    `read()` and `write()` are pure shared-memory accesses that never touch a socket: `read()`
    returns the newest values the threads managed to fetch, `write()` parks targets they pick up
    on their next tick (latest-wins, never queued). Each hand runs at `SONIC_INSPIRE_HZ`
    (default 30) independently of the other -- see `_HandChannel` for why per-hand rather than
    one shared thread.

    WHY THREADS AT ALL, not just a rate limit. These calls used to run inline, and pymodbus's
    TCP client is synchronous, so a hand that went quiet cost its full socket timeout -- seconds
    -- charged to whichever loop called it. That landed in two bad places at once:
      * `read()` runs inside the INFERENCE worker, whose elapsed time IS the `delay_ticks` fed
        to real-time chunking. A stalled hand therefore inflated the RTC delay estimate and
        tripped the "delay > trained max; clamping" path, i.e. a hand fault degraded BODY
        control.
      * `write()` runs in the 50 Hz publish loop, so a stall there stalled body publishing.
    Rate-limiting only reduced how OFTEN that exposure occurred; a due tick still blocked.
    Decoupling removes it: a stalled hand costs stale values in its slot, and nothing waits.
    """

    def __init__(self, left_host: str = LEFT_HOST, right_host: str = RIGHT_HOST,
                 port: int = PORT):
        self._hz = float(os.environ.get("SONIC_INSPIRE_HZ", "30"))
        period = 1.0 / self._hz
        self.left = _HandChannel("left", left_host, port, period)
        self.right = _HandChannel("right", right_host, port, period)
        self._chans = (self.left, self.right)

    def read(self):
        """Latest (left6, right6), or None unless BOTH hands have a reading. NEVER blocks.

        Both-or-nothing on purpose: the caller writes these into the policy's hand state as a
        pair, and half a pose with the other half stale-from-boot is a fabricated posture. A
        single dead hand therefore drops hand proprio entirely, which is what it did before.
        """
        l, r = self.left.read(), self.right.read()
        return None if l is None or r is None else (l, r)

    def write(self, left6, right6) -> bool:
        """Park new targets for both hands. NEVER blocks. False only if BOTH are disabled."""
        ok_l = self.left.write(left6)
        ok_r = self.right.write(right6)
        return ok_l or ok_r

    def open_hands(self, wait_s: float = 1.0) -> bool:
        """Park both hands OPEN -- the rest pose the policy's state assumes at episode start.

        Unlike the control-path writes this one WAITS (briefly, bounded) for confirmation, since
        it runs at startup/'i' rather than in the loop and the caller reports whether the hands
        actually moved. Waits on both hands concurrently, so a slow one costs its own time, not
        the sum.
        """
        zeros = np.zeros(N_FINGERS, np.float32)
        for c in self._chans:
            c._wrote_once.clear()
        if not self.write(zeros, zeros):
            return False
        deadline = time.monotonic() + wait_s
        ok = True
        for c in self._chans:
            if c._disabled:
                ok = False
                continue
            ok &= c._wrote_once.wait(timeout=max(0.0, deadline - time.monotonic()))
        return ok

    def start(self) -> None:
        for c in self._chans:
            c.start()

    def stats(self) -> dict:
        """Per-hand Modbus timing for the diagnostics line. Cheap; safe to call every chunk.

        Per hand and per direction, because "the hands are slow" is not actionable -- which
        hand, and reading or writing, is.
        """
        l, r = self.left.stats(), self.right.stats()
        return {
            "hz": self._hz,
            "disabled": l["disabled"] and r["disabled"],
            "left": l, "right": r,
            # kept flat as well so callers can format without knowing the nesting
            "read_left": l["read"], "write_left": l["write"],
            "read_right": r["read"], "write_right": r["write"],
            "age_ms": max(l["age_ms"], r["age_ms"]),
            "overruns": l["overruns"] + r["overruns"],
            "tick": (max(l["tick"][0], r["tick"][0]), max(l["tick"][1], r["tick"][1])),
        }

    def close(self) -> None:
        for c in self._chans:
            c.close()
