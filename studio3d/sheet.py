"""Character sheets: split ONE image that shows the 4 sides of a character into 4 view images.

* auto: find the figures on the background (works for a row of 4, a 2x2 grid, ...)
* layout: equal split (row of 4, 2x2 grid, column of 4)
* boxes: user-drawn rectangles

All crops are placed on same-size white squares at the SAME scale, which multi-view
generation needs (a back view must not be bigger than the front view).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

VIEWS = ("front", "back", "left", "right")


def _bg_color(img: Image.Image):
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return tuple(int(c) for c in np.median(border, axis=0))


def _foreground(img: Image.Image) -> np.ndarray:
    """Mask of non-background pixels (background = colour of the image border)."""
    rgba = img.convert("RGBA")
    a = np.asarray(rgba).astype(np.float32)
    if a[..., 3].min() < 250:  # transparent background
        return a[..., 3] > 30
    rgb = a[..., :3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(rgb - bg).max(axis=2)
    return diff > 28


def _components(mask: np.ndarray, merge_px: int):
    from scipy import ndimage
    grown = ndimage.binary_dilation(mask, iterations=max(1, merge_px))
    labels, n = ndimage.label(grown)
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        ys, xs = sl
        area = int(mask[sl].sum())
        boxes.append([xs.start, ys.start, xs.stop, ys.stop, area])
    return boxes


def _order(boxes):
    """Reading order: rows top-to-bottom, then left-to-right."""
    if not boxes:
        return boxes
    hs = [b[3] - b[1] for b in boxes]
    row_tol = 0.5 * float(np.median(hs))
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if abs((b[1] + b[3]) / 2 - (cur[-1][1] + cur[-1][3]) / 2) <= row_tol:
            cur.append(b)
        else:
            rows.append(cur)
            cur = [b]
    rows.append(cur)
    return [b for r in rows for b in sorted(r, key=lambda b: b[0])]


def detect(img: Image.Image, count: int = 4) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the `count` biggest figures, in reading order."""
    w, h = img.size
    mask = _foreground(img)
    if mask.mean() < 0.005:
        raise ValueError("لم أجد شخصيات واضحة على الخلفية")
    for frac in (0.004, 0.008, 0.015, 0.025):  # merge hair strands/hands into their figure
        boxes = _components(mask, int(frac * max(w, h)))
        boxes = [b for b in boxes if b[4] > 0.01 * mask.sum()]
        if len(boxes) >= count:
            break
    if len(boxes) < count:
        raise ValueError(f"وجدت {len(boxes)} شخصيات فقط — استخدم التقسيم بالتخطيط أو التحديد اليدوي")
    boxes = sorted(boxes, key=lambda b: -b[4])[:count]
    return [tuple(b[:4]) for b in _order(boxes)]


def layout_boxes(img: Image.Image, layout: str) -> list[tuple[int, int, int, int]]:
    w, h = img.size
    if layout == "row":
        return [(i * w // 4, 0, (i + 1) * w // 4, h) for i in range(4)]
    if layout == "column":
        return [(0, i * h // 4, w, (i + 1) * h // 4) for i in range(4)]
    if layout == "grid":
        return [(c * w // 2, r * h // 2, (c + 1) * w // 2, (r + 1) * h // 2) for r in range(2) for c in range(2)]
    raise ValueError(layout)


def _tight(img: Image.Image, box):
    """Shrink a box to the figure inside it (for layout / hand-drawn boxes)."""
    crop = img.crop(box)
    m = _foreground(crop)
    if m.sum() < 50:
        return box
    ys, xs = np.where(m)
    return (box[0] + xs.min(), box[1] + ys.min(), box[0] + xs.max() + 1, box[1] + ys.max() + 1)


def crops(img: Image.Image, boxes, tighten: bool = True, size: int = 768, margin: float = 0.08):
    """Cut the boxes and put each on a white square, all at the same scale, feet aligned."""
    img = img.convert("RGB")
    bg = _bg_color(img)
    boxes = [_tight(img, b) if tighten else b for b in boxes]
    side = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
    scale = size * (1 - 2 * margin) / side
    out = []
    for b in boxes:
        c = img.crop(b)
        c = c.resize((max(1, round(c.width * scale)), max(1, round(c.height * scale))), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), bg)
        x = (size - c.width) // 2
        y = size - round(size * margin) - c.height  # same ground line for every view
        canvas.paste(c, (x, y))
        out.append(canvas)
    return out


def facing(img: Image.Image) -> float:
    """Which way a side-view figure faces: < 0 = towards the left of the image, > 0 = right.
    Shoe tips point forward, so the feet stick out on the facing side of the legs."""
    m = _foreground(img)
    rows = np.where(m.any(axis=1))[0]
    if len(rows) < 10:
        return 0.0
    top, bottom = rows.min(), rows.max()
    hgt = bottom - top
    feet = m[bottom - int(0.04 * hgt): bottom + 1]
    shins = m[bottom - int(0.16 * hgt): bottom - int(0.08 * hgt)]
    fx = np.where(feet.any(axis=0))[0]
    sx = np.where(shins.any(axis=0))[0]
    if not len(fx) or not len(sx):
        return 0.0
    feet_c = (fx.min() + fx.max()) / 2
    shin_c = (sx.min() + sx.max()) / 2
    return float((feet_c - shin_c) / max(1, m.shape[1]))


def guess_views(images) -> dict:
    """Map the 4 crops to views. Front/back show the full width (arms), side views are
    narrower: the two widest are front then back, the other two left then right."""
    widths = []
    for im in images:
        m = _foreground(im)
        xs = np.where(m.any(axis=0))[0]
        widths.append((xs.max() - xs.min()) if len(xs) else 0)
    order = sorted(range(len(images)), key=lambda i: -widths[i])
    wide, narrow = sorted(order[:2]), sorted(order[2:])
    if widths[order[1]] < 1.15 * widths[order[2]]:  # all similar (e.g. A-pose): keep reading order
        wide, narrow = [0, 1], [2, 3]
    # Hunyuan3D convention: the "left" view (camera on the character's left) shows the
    # character facing the LEFT of the image.
    fa, fb = facing(images[narrow[0]]), facing(images[narrow[1]])
    if fa > 0.004 and fb < -0.004:
        narrow = [narrow[1], narrow[0]]
    idx = {"front": wide[0], "back": wide[1], "left": narrow[0], "right": narrow[1]}
    return {v: images[i] for v, i in idx.items()}


def split(img: Image.Image, mode: str = "auto", boxes=None) -> dict:
    """mode: auto | row | grid | column | boxes. Returns {view: PIL.Image}."""
    if mode == "auto":
        bx = detect(img)
        return guess_views(crops(img, bx, tighten=False))
    if mode == "boxes":
        if not boxes or len(boxes) != 4:
            raise ValueError("حدّد 4 مربعات")
        ims = crops(img, boxes)
        return dict(zip(VIEWS, ims))  # hand-drawn boxes are taken in the order front, back, left, right
    return guess_views(crops(img, layout_boxes(img, mode)))
