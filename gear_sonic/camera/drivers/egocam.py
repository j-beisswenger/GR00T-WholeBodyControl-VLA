"""Ego-view camera reprojected into the EgoStandard training camera (phone / webcam over UVC).

For checkpoints trained on EgoStandard ego video (e.g. `egostandard_only_*`): a rectified pinhole
of 480 x 364 px, fx = fy = 197, cx = 240, cy = 182 (101.2 x 85.5 deg). The robot's RealSense is a
different camera altogether (~69 x 42 deg, 640 x 480), so those checkpoints see an unfamiliar
view through it. This driver reads a wide source camera and remaps every frame into the
EgoStandard intrinsics, so the policy receives the camera it was trained on.

Source: any UVC node. A Pixel 7 in Android "Webcam" USB mode appears as "Pixel 7: Android Webcam";
its **640x480 MJPG mode is the full 4:3 sensor view** (the 16:9 modes crop the top and bottom),
and its ultrawide at 4:3 is ~102 x 85 deg, i.e. it covers EgoStandard almost exactly. The lens
is chosen ON THE PHONE (webcam preview, zoom 0.5x): UVC exposes no zoom control.

Geometry comes from a calibration file (egocam's calibrate.py output, which also models
distortion) or, if none is given, from a distortion-free pinhole with an assumed horizontal FOV --
reasonable for phones, whose camera pipeline already outputs rectilinear frames, but it should be
measured. The startup report prints coverage: anything below ~99 % means black borders the
policy never saw in training.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gear_sonic.camera import egocam_reproject as er
from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition

DEFAULT_NAMES = ("Android Webcam", "Brio", "Logi")


class EgoCamConfig:
    source: str | None = None           # /dev/videoN, index, or None = find by name
    size: tuple = (640, 480)            # 4:3 = full phone sensor view
    fps: int = 30
    calib: str | None = None            # egocam camera_calib.json for THIS device + mode
    hfov_deg: float | None = None       # nominal pinhole HFOV if no calib
    pitch_deg: float = 0.0              # virtual camera pitch (little margin with a ~102 deg lens)


def find_node(names=DEFAULT_NAMES) -> str | None:
    nodes = []
    for d in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        try:
            name = Path(d, "name").read_text().strip()
            if int(Path(d, "index").read_text()) != 0:     # index 1 = UVC metadata node
                continue
        except (OSError, ValueError):
            continue
        nodes.append((name, "/dev/" + os.path.basename(d)))
    for sub in names:
        for name, dev in nodes:
            if sub.lower() in name.lower():
                return dev
    return None


class EgoCamSensor(Sensor):
    def __init__(self, config: EgoCamConfig = EgoCamConfig(),
                 mount_position: str = CameraMountPosition.EGO_VIEW.value, device: str | None = None):
        self.config, self.mount_position = config, mount_position
        src = device or config.source or find_node()
        if src is None:
            raise RuntimeError(f"[egocam] no UVC camera matching {DEFAULT_NAMES}; is the phone in "
                               "USB 'Webcam' mode? Pass --ego-view-device-id /dev/videoN otherwise.")
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"[egocam] cannot open {src!r}")
        w, h = config.size
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame = None
        for _ in range(30):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError(f"[egocam] {src!r} opened but delivers no frames")
        got = (frame.shape[1], frame.shape[0])

        if config.calib:
            calib = er.load_calib(config.calib)
        elif config.hfov_deg:
            calib = er.nominal_calib(got, config.hfov_deg)
        else:
            raise RuntimeError("[egocam] needs --egocam-calib <camera_calib.json> or --egocam-hfov <deg>")
        if tuple(calib["image_size"]) != got:
            raise RuntimeError(f"[egocam] stream is {got[0]}x{got[1]} but the calibration is for "
                               f"{calib['image_size'][0]}x{calib['image_size'][1]}; recalibrate this mode")
        self.cam = er.EgoCam(calib, pitch_deg=config.pitch_deg)
        print(f"[{mount_position}] egocam on {src} ({got[0]}x{got[1]} MJPG) -> EgoStandard "
              f"{er.EGO_W}x{er.EGO_H}")
        print("  " + self.cam.report().replace("\n", "\n  "), flush=True)

    def read(self) -> dict[str, Any] | None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[{self.mount_position}] egocam read failed", flush=True)
            return None
        return {"timestamps": {self.mount_position: time.time()},
                "images": {self.mount_position: self.cam.remap(frame)}}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema
        return ImageMessageSchema(timestamps=data["timestamps"], images=data["images"]).serialize()

    def observation_space(self):
        return None

    def close(self):
        if self.cap is not None:
            self.cap.release()
