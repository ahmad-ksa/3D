"""End-to-end UI flow with the AI models mocked (runs on any CPU machine)."""
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO3D_HOME", str(tmp_path))
    from studio3d import config, engine, ui
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")

    def fake_shape(image, kind, preset, seed, **kw):
        assert kind in ("multiview", "single")
        if kind == "multiview":
            assert set(image) == {"front", "left"}
        return trimesh.creation.icosphere(4)

    def fake_texture(mesh, image, max_faces=40000):
        m = mesh.copy()
        uv = np.random.rand(len(m.vertices), 2)
        tex = Image.new("RGB", (64, 64), (200, 50, 50))
        m.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
        return m

    monkeypatch.setattr(engine, "generate_shape", fake_shape)
    monkeypatch.setattr(engine, "texture_mesh", fake_texture)
    monkeypatch.setattr(engine, "prepare_image", lambda img, rb: img.convert("RGBA"))

    img = Image.new("RGB", (64, 64), "white")
    progress = lambda *a, **k: None  # noqa: E731
    opts = ["متوازن", True, True, True, False, 2, 1000, 120, 5, False, ["glb", "obj", "stl", "3mf"]]
    preview, files, info = ui._run("multiview", {"front": img, "back": None, "left": img, "right": None},
                                   None, *opts, progress)
    assert preview.endswith("model_textured.glb") and os.path.exists(preview)
    names = {Path(f).name for f in files}
    assert {"model.stl", "model.3mf", "model_white.glb"} <= names
    assert any(n.startswith("model_obj") or n == "model.obj" for n in names)
    loaded = trimesh.load(preview, force="mesh")
    assert np.isclose(loaded.bounding_box.extents.max(), 120, atol=0.5)
    jobs = ui._jobs()
    assert len(jobs) == 1 and ui.show_job(jobs[0])[0].endswith("model_textured.glb")

    # single image, no texture
    preview, files, _ = ui._run("single", {"front": img}, None,
                                "سريع", False, True, True, False, 0, 0, 0, 1, False, ["glb"], progress)
    assert preview.endswith("model.glb")


def test_texture_network_failure_keeps_shape(tmp_path, monkeypatch):
    from studio3d import config, engine, pipeline
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(engine, "generate_shape", lambda *a, **k: trimesh.creation.icosphere(3))
    monkeypatch.setattr(engine, "prepare_image", lambda img, rb: img.convert("RGBA"))

    def boom(*a, **k):
        raise RuntimeError("Something wrong while loading C:/models/x")
    monkeypatch.setattr(engine, "texture_mesh", boom)
    img = Image.new("RGB", (32, 32), "white")
    res = pipeline.run("single", {"front": img}, None,
                       pipeline.Options(texture=True, randomize=False, formats=["glb", "stl"]))
    assert res["warning"] and "الإنترنت" in res["warning"]
    assert Path(res["preview"]).exists() and res["preview"].endswith("model.glb")
    assert (Path(res["job_dir"]) / "model_white.glb").exists()
