import numpy as np
from PIL import Image

from studio3d import engine, poses


def _bright(img):
    a = np.asarray(img).max(axis=2) > 40
    ys, xs = np.where(a)
    return xs.min(), xs.max(), ys.min(), ys.max()


def test_tpose_skeleton_is_wide_apose_narrower():
    t = poses.skeleton_image("t", 512)
    a = poses.skeleton_image("a", 512)
    tx0, tx1, ty0, ty1 = _bright(t)
    ax0, ax1, _, _ = _bright(a)
    assert tx0 > 5 and tx1 < 507                 # hands inside the image
    assert (tx1 - tx0) > 0.85 * (ty1 - ty0)      # T-pose: arm span about the body height
    assert (ax1 - ax0) < (tx1 - tx0)             # A-pose: arms lower -> narrower
    k = poses.keypoints("t")
    assert abs(k[4][1] - k[2][1]) < 1e-6 and abs(k[7][1] - k[5][1]) < 1e-6  # wrists at shoulder height


def test_text_to_image_uses_pose_controlnet(monkeypatch):
    calls = {}

    class FakePipe:
        def __call__(self, **kw):
            calls.update(kw)
            return ([Image.new("RGB", (8, 8))],)

    monkeypatch.setattr(engine, "_load_t2i_pose", lambda: FakePipe())
    monkeypatch.setattr(engine, "_load_t2i", lambda: (_ for _ in ()).throw(AssertionError("wrong pipe")))
    engine.MANAGER._free()
    engine.text_to_image("a man in a suit", 1, pose="t")
    assert isinstance(calls["control_image"], Image.Image)
    assert calls["prompt"].startswith("正面视角")          # front view first (CLIP reads 77 tokens)
    assert "T字姿势" in calls["prompt"] and "背面" in calls["negative_prompt"]
    engine.MANAGER._free()
