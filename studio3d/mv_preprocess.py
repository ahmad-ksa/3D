"""Multi-view input preparation with ONE shared scale for all views.

Hunyuan3D's stock multi-view processor fits every view to the square on its own (scale =
size / max(width, height) of that view). For a character in A-pose that is harmless (height
is the biggest side in every view), but in a T-pose the front/back views are limited by the arm
span and the side views by the height, so front/back get shrunk compared with the sides and the
model receives four views of different sizes -> broken shape. Here every view uses the same
scale (taken from the biggest figure), so a T-pose keeps consistent proportions.
"""
import numpy as np
import torch
from PIL import Image

VIEW2IDX = {"front": 0, "left": 1, "back": 2, "right": 3}


def _rgba(image) -> np.ndarray:
    if isinstance(image, str):
        image = Image.open(image)
    return np.asarray(image.convert("RGBA"))


def _bbox(alpha: np.ndarray):
    ys, xs = np.nonzero(alpha > 0)
    if len(ys) == 0:
        raise ValueError("input image is empty")
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def compose(views: dict, out_size: int, border_ratio: float = 0.15) -> dict:
    """views {tag: PIL/array RGBA} -> {tag: (rgb uint8 [S,S,3], mask uint8 [S,S])}, shared scale."""
    arrs = {k: _rgba(v) for k, v in views.items()}
    boxes = {k: _bbox(a[..., 3]) for k, a in arrs.items()}
    # every view lives on a canvas of the same size, the largest side of any figure fills it
    big = max(max(y1 - y0, x1 - x0) for y0, y1, x0, x1 in boxes.values())
    canvas = max(max(a.shape[:2]) for a in arrs.values())
    desired = int(canvas * (1 - border_ratio))
    scale = desired / big
    out = {}
    for k, a in arrs.items():
        y0, y1, x0, x1 = boxes[k]
        crop = Image.fromarray(a[y0:y1, x0:x1])
        w2 = max(1, int((x1 - x0) * scale))
        h2 = max(1, int((y1 - y0) * scale))
        crop = np.asarray(crop.resize((w2, h2), Image.LANCZOS))
        res = np.zeros((canvas, canvas, 4), np.uint8)
        top = (canvas - h2) // 2
        left = (canvas - w2) // 2
        res[top:top + h2, left:left + w2] = crop
        m = res[..., 3:].astype(np.float32) / 255
        rgb = (res[..., :3] * m + 255 * (1 - m)).clip(0, 255).astype(np.uint8)
        rgb = np.asarray(Image.fromarray(rgb).resize((out_size, out_size), Image.BICUBIC))
        mask = np.asarray(Image.fromarray((m[..., 0] * 255).astype(np.uint8)).resize(
            (out_size, out_size), Image.NEAREST))
        out[k] = (rgb, mask)
    return out


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.array(a)).float() / 255.0 * 2 - 1
    if t.ndim == 2:
        t = t[..., None]
    return t.permute(2, 0, 1).unsqueeze(0)


class ConsistentMVImageProcessor:
    """Drop-in replacement for hy3dgen's MVImageProcessorV2 (same outputs)."""
    return_view_idx = True

    def __init__(self, size=512, border_ratio=None):
        self.size = size
        self.border_ratio = border_ratio

    def __call__(self, image_dict, border_ratio=0.15, to_tensor=True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
        comp = compose(image_dict, self.size, border_ratio)
        items = sorted((VIEW2IDX[k], v) for k, v in comp.items())
        idxs = tuple(i for i, _ in items)
        if to_tensor:
            image = torch.cat([_to_tensor(rgb) for _, (rgb, _m) in items], 0).unsqueeze(0)
            mask = torch.cat([_to_tensor(m) for _, (_r, m) in items], 0).unsqueeze(0)
        else:
            image = [rgb for _, (rgb, _m) in items]
            mask = [m for _, (_r, m) in items]
        return {"image": image, "mask": mask, "view_idxs": idxs}
