"""EgoStandard reprojection, vendored from the `egocam` tool (Downloads/egocam/egocam.py).

Kept byte-for-byte in the geometry so a camera_calib.json produced by egocam's calibrate.py
works here unchanged. Only the reprojection is vendored; capture lives in drivers/egocam.py.

Original module docstring:

Reproject a calibrated source camera (Logitech Brio 500 by default) into the EgoStandard
training camera.

EgoStandard: rectified pinhole, 480 x 364 px, fx = fy = 197, cx = 240, cy = 182
(101.2 deg x 85.5 deg FOV). One precomputed cv2.remap does undistortion, FOV matching,
aspect ratio and centring.

    cam = EgoCam.from_file("camera_calib.json")
    rgb = cam.remap(bgr_frame)          # (364, 480, 3) uint8 RGB

Naming: "phone" in identifiers below just means the source camera.

Dependencies: numpy, opencv-python. No GUI calls in this module.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

EGO_W, EGO_H = 480, 364
EGO_SIZE = (EGO_W, EGO_H)
K_EGO = np.array([[197.0, 0.0, 240.0], [0.0, 197.0, 182.0], [0.0, 0.0, 1.0]])

# A target ray counts as covered only if projecting it into the phone image and
# undistorting that pixel again lands back on the ray. Outside the calibrated field
# the distortion polynomial can fold back into the image and fake coverage.
_ROUNDTRIP_TOL = 0.02


# --------------------------------------------------------------------------- calibration io


def load_calib(path: str | os.PathLike) -> dict:
    with open(path) as f:
        c = json.load(f)
    c["K"] = np.asarray(c["K"], dtype=np.float64).reshape(3, 3)
    c["D"] = np.asarray(c["D"], dtype=np.float64).ravel()
    c["image_size"] = tuple(int(v) for v in c["image_size"])
    if c["model"] not in ("standard", "fisheye"):
        raise ValueError(f"unknown camera model {c['model']!r}")
    return c


def save_calib(path: str | os.PathLike, calib: dict) -> None:
    out = dict(calib)
    out["K"] = np.asarray(calib["K"]).reshape(3, 3).tolist()
    out["D"] = np.asarray(calib["D"]).ravel().tolist()
    out["image_size"] = [int(v) for v in calib["image_size"]]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def nominal_calib(image_size: tuple[int, int], hfov_deg: float) -> dict:
    """Distortion-free guess from a spec-sheet horizontal FOV. For smoke tests only."""
    w, h = image_size
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {
        "model": "standard",
        "K": np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]]),
        "D": np.zeros(5),
        "image_size": (w, h),
        "rms": None,
        "uncalibrated": True,
    }


# --------------------------------------------------------------------------- geometry helpers


def undistort_to_rays(pts: np.ndarray, calib: dict) -> np.ndarray:
    """Source pixels (N, 2) -> normalised pinhole coords (N, 2) in the source camera frame."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if calib["model"] == "fisheye":
        out = cv2.fisheye.undistortPoints(p, calib["K"], calib["D"].reshape(4, 1))
    else:
        out = cv2.undistortPoints(p, calib["K"], calib["D"])
    return out.reshape(-1, 2)


def pitch_rotation(pitch_deg: float) -> np.ndarray:
    """R mapping phone-camera coords to a virtual camera pitched DOWN by pitch_deg.

    Camera frame is x right, y down, z forward, so looking down is a rotation about x
    that moves the optical axis towards +y.
    """
    a = math.radians(pitch_deg)
    c, s = math.cos(a), math.sin(a)
    # Rows are the virtual camera's axes expressed in phone coordinates.
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def phone_fov(calib: dict) -> dict:
    """Measured FOV of the source stream (degrees), from the calibrated edge/corner rays."""
    w, h = calib["image_size"]
    cx, cy = calib["K"][0, 2], calib["K"][1, 2]
    pts = np.array(
        [[0, cy], [w - 1, cy], [cx, 0], [cx, h - 1], [0, 0], [w - 1, h - 1], [w - 1, 0], [0, h - 1]],
        dtype=np.float64,
    )
    r = undistort_to_rays(pts, calib)
    ang = lambda v: math.degrees(math.atan(float(np.linalg.norm(v))))  # noqa: E731
    diag = max(ang(r[4]) + ang(r[5]), ang(r[6]) + ang(r[7]))
    return {"h": ang(r[0]) + ang(r[1]), "v": ang(r[2]) + ang(r[3]), "d": diag}


# --------------------------------------------------------------------------- the reprojector


class EgoCam:
    """Precomputed source camera -> EgoStandard remap."""

    def __init__(self, calib: dict, pitch_deg: float = 0.0):
        self.calib = calib
        self.pitch_deg = float(pitch_deg)
        K, D = calib["K"], calib["D"]
        R = pitch_rotation(pitch_deg)
        if calib["model"] == "fisheye":
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D.reshape(4, 1), R, K_EGO, EGO_SIZE, cv2.CV_32FC1
            )
        else:
            m1, m2 = cv2.initUndistortRectifyMap(K, D, R, K_EGO, EGO_SIZE, cv2.CV_32FC1)

        self.valid = self._valid_mask(m1, m2, R)
        # Uncovered pixels render black instead of folded-back garbage.
        m1 = np.where(self.valid, m1, -1.0).astype(np.float32)
        m2 = np.where(self.valid, m2, -1.0).astype(np.float32)
        # Fixed-point maps are ~2x faster in cv2.remap and bit-identical across x86/ARM.
        self._map1, self._map2 = cv2.convertMaps(m1, m2, cv2.CV_16SC2)

        self.coverage = float(self.valid.mean())
        self.phone_fov = phone_fov(calib)
        self.achieved_fov = self._achieved_fov()

    @classmethod
    def from_file(cls, path: str | os.PathLike = "camera_calib.json", pitch_deg: float = 0.0) -> "EgoCam":
        return cls(load_calib(path), pitch_deg)

    # -- per-frame

    def remap_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if (w, h) != self.calib["image_size"]:
            raise ValueError(
                f"frame is {w}x{h} but calibration is for "
                f"{self.calib['image_size'][0]}x{self.calib['image_size'][1]}; recalibrate this mode"
            )
        return cv2.remap(
            frame_bgr, self._map1, self._map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    def remap(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR source frame (as read by cv2) -> 480 x 364 RGB uint8."""
        return cv2.cvtColor(self.remap_bgr(frame_bgr), cv2.COLOR_BGR2RGB)

    # -- diagnostics

    def _valid_mask(self, m1: np.ndarray, m2: np.ndarray, R: np.ndarray) -> np.ndarray:
        w, h = self.calib["image_size"]
        inside = (m1 >= 0) & (m1 <= w - 1) & (m2 >= 0) & (m2 <= h - 1)
        u, v = np.meshgrid(np.arange(EGO_W, dtype=np.float64), np.arange(EGO_H, dtype=np.float64))
        rays = np.stack([(u - K_EGO[0, 2]) / K_EGO[0, 0], (v - K_EGO[1, 2]) / K_EGO[1, 1], np.ones_like(u)], -1)
        rays = rays @ R  # R^T applied to each row vector: virtual camera -> phone camera
        front = rays[..., 2] > 1e-6
        want = rays[..., :2] / np.where(front, rays[..., 2], 1.0)[..., None]
        ok = inside & front
        got = np.full_like(want, np.inf)
        if ok.any():
            got[ok] = undistort_to_rays(np.stack([m1[ok], m2[ok]], -1), self.calib)
        err = np.linalg.norm(got - want, axis=-1)
        return ok & (err < _ROUNDTRIP_TOL * (1.0 + np.linalg.norm(want, axis=-1)))

    def _achieved_fov(self) -> dict:
        cx, cy, f = K_EGO[0, 2], K_EGO[1, 2], K_EGO[0, 0]
        row = np.flatnonzero(self.valid[int(cy)])
        col = np.flatnonzero(self.valid[:, int(cx)])
        span = lambda idx, c, n: (  # noqa: E731
            0.0
            if idx.size == 0
            else math.degrees(math.atan((idx.max() + 0.5 - c) / f) - math.atan((idx.min() - 0.5 - c) / f))
        )
        return {"h": span(row, cx, EGO_W), "v": span(col, cy, EGO_H)}

    def black_edges(self) -> dict:
        """Fraction of each output border (outer 2 px) that has no source pixel."""
        inv = ~self.valid
        return {
            "top": float(inv[:2].mean()),
            "bottom": float(inv[-2:].mean()),
            "left": float(inv[:, :2].mean()),
            "right": float(inv[:, -2:].mean()),
        }

    def report(self) -> str:
        c, pf, af = self.calib, self.phone_fov, self.achieved_fov
        rms = c.get("rms")
        lines = [
            f"model          : {c['model']}" + ("   *** UNCALIBRATED nominal guess ***" if c.get("uncalibrated") else ""),
            f"stream         : {c['image_size'][0]} x {c['image_size'][1]}",
            "calib RMS      : " + ("n/a" if rms is None else f"{rms:.3f} px  [{'PASS' if rms < 0.5 else 'FAIL'} < 0.5]"),
            f"camera FOV     : {pf['h']:.1f} x {pf['v']:.1f} deg  (diag {pf['d']:.1f})",
            "target FOV     : 101.2 x 85.5 deg  (diag 113.6)",
            f"achieved FOV   : {af['h']:.1f} x {af['v']:.1f} deg   pitch {self.pitch_deg:+.1f} deg",
            f"coverage       : {100 * self.coverage:.2f} %  [{'PASS' if self.coverage >= 0.99 else 'FAIL'} >= 99]",
        ]
        if self.coverage < 0.99:
            edges = ", ".join(f"{k} {100 * v:.0f}%" for k, v in self.black_edges().items() if v > 0.01)
            lines.append(f"black borders  : {edges}")
        return "\n".join(lines)
