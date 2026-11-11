"""Quad remesh + auto-rig through Blender. Uses a real Blender if found, or a Python
interpreter that has the `bpy` module (set STUDIO3D_BLENDER). Skipped otherwise."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio3d import blender  # noqa: E402

pytestmark = pytest.mark.skipif(blender.find_blender() is None, reason="Blender / bpy not available")


def humanoid():
    def cap(r, h, axis, center):
        c = trimesh.creation.capsule(height=h, radius=r, count=[16, 16])
        if axis == "y":
            c.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        if axis == "x":
            c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        c.apply_translation(center)
        return c
    parts = [cap(0.18, 0.55, "y", [0, 1.15, 0]),
             trimesh.creation.icosphere(3, radius=0.14).apply_translation([0, 1.62, 0]),
             cap(0.08, 0.78, "y", [0.11, 0.48, 0]), cap(0.08, 0.78, "y", [-0.11, 0.48, 0]),
             cap(0.06, 0.62, "x", [0.52, 1.38, 0]), cap(0.06, 0.62, "x", [-0.52, 1.38, 0])]
    m = trimesh.util.concatenate(parts)
    v = m.voxelized(0.02).fill().marching_cubes
    v.apply_scale(0.02)
    return v


def test_quad_and_rig(tmp_path):
    src = tmp_path / "human.glb"
    humanoid().export(src)
    q = blender.quad_remesh(str(src), tmp_path / "q", target_faces=3000, bake_size=0)
    assert q["quads"] == q["faces"] and 1000 < q["faces"] < 6000
    assert {Path(f).suffix for f in q["files"]} == {".obj", ".fbx", ".glb"}
    obj_text = Path(next(f for f in q["files"] if f.endswith(".obj"))).read_text()
    assert any(len(ln.split()) == 5 for ln in obj_text.splitlines() if ln.startswith("f "))  # real quads

    r = blender.auto_rig(next(f for f in q["files"] if f.endswith(".glb")), tmp_path / "r")
    assert r["bones"] == 19 and not r["a_pose"]
    assert r["arm_test_moved_fraction"] > 0.1
    for f in r["files"]:
        assert Path(f).stat().st_size > 10000


def test_pipeline_with_quad_and_rig(tmp_path, monkeypatch):
    from studio3d import config, engine, pipeline
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(engine, "generate_shape", lambda *a, **k: humanoid())
    monkeypatch.setattr(engine, "prepare_image", lambda img, rb: img.convert("RGBA"))
    res = pipeline.run("single", {"front": Image.new("RGB", (32, 32))}, None,
                       pipeline.Options(texture=False, randomize=False, formats=["glb"],
                                        topology="quad", rig=True, faces=3000))
    names = {Path(f).name for f in res["files"]}
    assert {"model_quad.obj", "model_quad.fbx", "model_rigged.fbx", "model_rigged.glb"} <= names, names
    assert res["warning"] is None and res["stats"]["quads"] == res["stats"]["faces"]
    meta = json.loads((Path(res["job_dir"]) / "meta.json").read_text(encoding="utf-8"))
    assert meta["rig"]["bones"] == 19


def test_missing_blender_gives_clear_warning(tmp_path, monkeypatch):
    from studio3d import config, engine, pipeline
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(blender, "find_blender", lambda: None)
    monkeypatch.setattr(engine, "generate_shape", lambda *a, **k: trimesh.creation.icosphere(3))
    monkeypatch.setattr(engine, "prepare_image", lambda img, rb: img.convert("RGBA"))
    res = pipeline.run("single", {"front": Image.new("RGB", (32, 32))}, None,
                       pipeline.Options(randomize=False, formats=["glb"], topology="quad", rig=True))
    assert "Blender" in res["warning"] and Path(res["preview"]).exists()


def _wav(path, seconds=1.5):
    import wave
    rate = 16000
    t = np.arange(int(rate * seconds)) / rate
    sig = (0.5 * np.clip(np.sin(2 * np.pi * 3 * t), 0, 1) * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(sig.tobytes())


def test_lip_sync(tmp_path, monkeypatch):
    from studio3d import config, pipeline
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    src = tmp_path / "human.glb"
    humanoid().export(src)
    r = blender.auto_rig(str(src), tmp_path / "r")
    rigged = next(f for f in r["files"] if f.endswith(".glb"))
    wav = tmp_path / "v.wav"
    _wav(wav)
    res = pipeline.lip_sync(rigged, str(wav))
    names = {Path(f).name for f in res["files"]}
    assert {"model_talking.glb", "model_talking.fbx", "model_talking.blend", "voice.wav"} <= names
    assert res["stats"]["frames"] == 45 and res["stats"]["jaw_vertices"] > 0
