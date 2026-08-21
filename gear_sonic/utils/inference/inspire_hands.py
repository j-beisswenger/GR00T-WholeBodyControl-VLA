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

import os
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


class InspireHandReader:
    """Lazily-connected reader for both hands. Never raises into the control loop.

    `read()` returns `(left6, right6)` in radians, or `None` if either hand is unavailable --
    the caller then leaves the observation untouched rather than feeding the policy a guess.
    """

    def __init__(self, left_host: str = LEFT_HOST, right_host: str = RIGHT_HOST, port: int = PORT):
        self._hosts = (left_host, right_host)
        self._port = port
        self._clients = None
        self._warned = False
        self._write_period = 1.0 / float(os.environ.get("SONIC_INSPIRE_WRITE_HZ", "25"))
        self._last_write = 0.0
        self._unit_kw = None      # "slave" (pymodbus 3.x) or "device_id" (4.x); detected once
        self._fails = 0
        self._disabled = False
        self._announced = False

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

    def read(self):
        if not self._connect():
            return None
        out = []
        for c in self._clients:
            try:
                rr = c.read_holding_registers(REG_ACTUAL, count=N_FINGERS,
                                              **{self._unit_kw: 1})
            except Exception as exc:                      # transport hiccup: drop the sample
                if self._fails == 0:
                    print(f"[inspire] read failed ({exc})", flush=True)
                self._give_up(str(exc))
                return None
            if rr is None or getattr(rr, "isError", lambda: True)():
                self._give_up("modbus error response")
                return None
            out.append(_ctrl_to_rad(rr.registers))
        self._fails = 0
        return out[0], out[1]

    def write(self, left6, right6) -> bool:
        """Command both hands. Returns False if the write did not reach the hardware.

        Rate-limited: the control loop publishes at 50 Hz, far faster than these servos need or
        than Modbus TCP round-trips comfortably sustain, and a backed-up socket would stall the
        loop it is called from. `SONIC_INSPIRE_WRITE_HZ` (default 25) sets the ceiling; ticks
        in between are dropped, not queued, so the hands always track the LATEST target.
        """
        if not self._connect():
            return False
        now = time.monotonic()
        if now - self._last_write < self._write_period:
            return True                                   # skipped by design, not a failure
        self._last_write = now
        for c, q in zip(self._clients, (left6, right6)):
            regs = _rad_to_ctrl(q)
            try:
                rr = c.write_registers(REG_TARGET, regs.tolist(), **{self._unit_kw: 1})
            except Exception as exc:
                if self._fails == 0:
                    print(f"[inspire] write failed ({exc})", flush=True)
                self._give_up(str(exc))
                return False
            if rr is not None and getattr(rr, "isError", lambda: False)():
                self._give_up("modbus error response")
                return False
        return True

    def open_hands(self) -> bool:
        """Park both hands OPEN -- the rest pose the policy's state assumes at episode start."""
        return self.write(np.zeros(N_FINGERS, np.float32), np.zeros(N_FINGERS, np.float32))

    def close(self) -> None:
        for c in self._clients or []:
            c.close()
        self._clients = None
