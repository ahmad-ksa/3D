import numpy as np
import torch
from PIL import Image, ImageDraw

from studio3d.mv_preprocess import ConsistentMVImageProcessor, compose


def _figure(w_frac, h_frac, size=600):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, h = int(size * w_frac), int(size * h_frac)
    d.rectangle([(size - w) // 2, (size - h) // 2, (size + w) // 2, (size + h) // 2], fill=(200, 50, 50, 255))
    return img


def _height(mask):
    rows = np.where((mask > 127).any(axis=1))[0]
    return rows.max() - rows.min() + 1


def test_tpose_views_keep_the_same_height():
    # T-pose: front/back as wide as tall*1.0+, sides narrow; all are the same height in reality
    views = {"front": _figure(0.9, 0.6), "back": _figure(0.9, 0.6),
             "left": _figure(0.2, 0.6), "right": _figure(0.2, 0.6)}
    out = compose(views, 512)
    hs = [_height(m) for _, m in out.values()]
    assert max(hs) - min(hs) <= 2          # same body height in every view
    # stock hy3dgen would scale the narrow side views up to fill the square instead
    assert hs[0] < 0.7 * 512


def test_processor_output_shapes():
    views = {"front": _figure(0.9, 0.6), "left": _figure(0.2, 0.6), "back": _figure(0.9, 0.6)}
    out = ConsistentMVImageProcessor(512)(views)
    assert out["image"].shape == (1, 3, 3, 512, 512)
    assert out["mask"].shape == (1, 3, 1, 512, 512)
    assert out["view_idxs"] == (0, 1, 2)
    assert out["image"].min() >= -1 and out["image"].max() <= 1
    assert isinstance(out["image"], torch.Tensor)
