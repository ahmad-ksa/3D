"""Face box + landmarks for LatentSync, with Google MediaPipe Face Landmarker (Apache-2.0).

LatentSync was written for insightface's 106-point layout. Only a few of those points are used:
  * 43,48,49,50,51  = brow on the IMAGE-left, 101..105 = brow on the IMAGE-right
  * 74,77,83,86     = nose (centre of the three alignment points)
  * 73,74           = top of the nose bridge (half-face line)
  * 0..32           = jaw contour (only min/max are used -> crop box)
We fill exactly those slots from MediaPipe's 478-point mesh; the other slots are not read.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
             "float16/1/face_landmarker.task")

# MediaPipe mesh indices (subject's right side appears on the IMAGE-left)
BROW_IMG_LEFT = [(70, 46), (63, 53), (105, 52), (66, 65), (107, 55)]      # (upper, lower) pairs
BROW_IMG_RIGHT = [(300, 276), (293, 283), (334, 282), (296, 295), (336, 285)]
NOSE_BRIDGE_TOP = 168
NOSE_BRIDGE_MID = 6
NOSE_UPPER = 197
NOSE_WING_LEFT = 98
NOSE_WING_RIGHT = 327
NOSE_TIP = 4
# jaw line from the image-left temple, down to the chin, up to the image-right temple
JAW = [127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
       377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356]


def model_path() -> Path:
    from studio3d import config
    return config.MODELS_DIR / "video" / "face_landmarker.task"


def ensure_model(retry_minutes: float = 2.0) -> Path:
    """Download the 3.6 MB MediaPipe face model once (2-minute retry on network errors)."""
    p = model_path()
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + retry_minutes * 60
    while True:
        try:
            tmp = p.with_suffix(".part")
            with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(p)
            return p
        except Exception:  # noqa: BLE001
            if time.time() >= deadline:
                raise
            time.sleep(10)


def _resample(points: np.ndarray, n: int) -> np.ndarray:
    """Evenly re-sample a polyline to n points."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] <= 0:
        return np.repeat(points[:1], n, axis=0)
    u = np.linspace(0, t[-1], n)
    return np.stack([np.interp(u, t, points[:, 0]), np.interp(u, t, points[:, 1])], axis=1)


def to_106(mesh_xy: np.ndarray) -> np.ndarray:
    """MediaPipe mesh (N,2) pixel coords -> insightface-style (106,2) array (used slots filled)."""
    lmk = np.zeros((106, 2), np.float64)
    jaw = _resample(mesh_xy[JAW], 33)            # slot 16 = chin (middle of the line)
    lmk[0:33] = jaw
    for slot, (a, b) in zip([43, 48, 49, 50, 51], BROW_IMG_LEFT):
        lmk[slot] = (mesh_xy[a] + mesh_xy[b]) / 2
    for slot, (a, b) in zip([101, 102, 103, 104, 105], BROW_IMG_RIGHT):
        lmk[slot] = (mesh_xy[a] + mesh_xy[b]) / 2
    lmk[73] = mesh_xy[NOSE_BRIDGE_TOP]
    lmk[74] = mesh_xy[NOSE_BRIDGE_MID]
    lmk[77] = mesh_xy[NOSE_WING_LEFT]
    lmk[83] = mesh_xy[NOSE_WING_RIGHT]
    lmk[86] = mesh_xy[NOSE_TIP]
    lmk[72] = mesh_xy[NOSE_UPPER]
    return lmk


# same subset + box logic as LatentSync's insightface detector
LMK_ADAPT_ORIGIN_ORDER = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17, 43, 48, 49, 51, 50,
                          102, 103, 104, 105, 101, 73, 74, 86]


def box_from_106(lmk: np.ndarray, frame_w: int, frame_h: int):
    lmk = np.round(lmk).astype(np.int_)
    half = np.mean([lmk[74], lmk[73]], axis=0)
    sub = lmk[LMK_ADAPT_ORIGIN_ORDER]
    dist = np.max(sub[:, 1]) - half[1]
    upper = half[1] - dist
    x1, y1, x2, y2 = int(np.min(sub[:, 0])), int(upper), int(np.max(sub[:, 0])), int(np.max(sub[:, 1]))
    y2 += int((x2 - x1) * 0.1)
    x1 -= int((x2 - x1) * 0.05)
    x2 += int((x2 - x1) * 0.05)
    return (max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)), lmk


class FaceDetector:
    """Drop-in for latentsync.utils.face_detector.FaceDetector: __call__(rgb frame) -> (bbox, lmk106)."""

    def __init__(self, device="cuda"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE, num_faces=4,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, frame, threshold=0.5):
        h, w = frame.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[..., :3]))
        res = self._lm.detect(img)
        best, best_size = None, 0
        for face in res.face_landmarks or []:
            xy = np.array([[p.x * w, p.y * h] for p in face], np.float64)
            x0, y0 = xy.min(axis=0)
            x1, y1 = xy.max(axis=0)
            bw, bh = x1 - x0, y1 - y0
            if bw < 50 or bh < 60 or not (0.2 < bw / max(bh, 1) < 1.5):
                continue
            if bw * bh > best_size:
                best, best_size = xy, bw * bh
        if best is None:
            return None, None
        return box_from_106(to_106(best), w, h)
