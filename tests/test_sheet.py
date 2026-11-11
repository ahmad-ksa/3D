import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio3d import sheet  # noqa: E402


def _figure(d, x, wide, facing=0, bg=(245, 238, 225)):
    w = 120 if wide else 50
    d.rectangle([x - w // 2, 60, x + w // 2, 120], fill=(230, 120, 150))   # arms/torso
    d.ellipse([x - 25, 10, x + 25, 60], fill=(60, 40, 30))                 # head
    d.rectangle([x - 15, 120, x + 15, 260], fill=(230, 160, 140))          # legs
    d.rectangle([x - 15 + facing * 14, 255, x + 15 + facing * 14, 270], fill=(200, 80, 120))  # shoes


def _sheet(order):
    im = Image.new("RGB", (900, 300), (245, 238, 225))
    d = ImageDraw.Draw(im)
    for i, (wide, facing) in enumerate(order):
        _figure(d, 110 + i * 225, wide, facing)
    return im


def test_detect_and_assign_row():
    # front, back, side facing right, side facing left (like a typical turnaround sheet)
    im = _sheet([(True, 0), (True, 0), (False, 1), (False, -1)])
    boxes = sheet.detect(im)
    assert len(boxes) == 4 and boxes[0][0] < boxes[1][0] < boxes[2][0] < boxes[3][0]
    views = sheet.split(im, "auto")
    assert set(views) == {"front", "back", "left", "right"}
    assert sheet.facing(views["left"]) < 0 < sheet.facing(views["right"])
    sizes = {v.size for v in views.values()}
    assert sizes == {(768, 768)}


def test_layout_and_boxes():
    im = _sheet([(True, 0), (True, 0), (False, -1), (False, 1)])
    v = sheet.split(im, "row")
    assert len(v) == 4
    boxes = [(0, 0, 225, 300), (225, 0, 450, 300), (450, 0, 675, 300), (675, 0, 900, 300)]
    v = sheet.split(im, "boxes", boxes)
    assert list(v) == ["front", "back", "left", "right"]
    # the same scale: the two front/back crops have the same figure height
    hs = [np.where(sheet._foreground(v[k]).any(axis=1))[0].ptp() for k in ("front", "back")]
    assert abs(hs[0] - hs[1]) <= 2
