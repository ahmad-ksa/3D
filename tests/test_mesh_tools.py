import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio3d import mesh_tools  # noqa: E402


def _noisy():
    main = trimesh.creation.icosphere(subdivisions=4)
    speck = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    speck.apply_translation([3, 0, 0])
    return trimesh.util.concatenate([main, speck])


def test_clean_removes_floaters():
    out = mesh_tools.clean(_noisy())
    assert len(out.split(only_watertight=False)) == 1


def test_reduce_faces():
    m = trimesh.creation.icosphere(subdivisions=5)
    out = mesh_tools.reduce_faces(m, 2000)
    assert len(out.faces) <= 2100


def test_normalize_size_and_ground():
    m = trimesh.creation.box(extents=[2, 4, 1])
    out = mesh_tools.normalize(m, target_size=100, on_ground=True)
    assert np.isclose(out.bounding_box.extents.max(), 100)
    assert np.isclose(out.bounds[0][1], 0)


def test_process_pipeline():
    out = mesh_tools.process(_noisy(), target_faces=1000, smooth_iters=3, holes=True, size_mm=50)
    assert len(out.faces) <= 1100


def test_exports(tmp_path):
    m = mesh_tools.normalize(trimesh.creation.icosphere(), 100)
    for fmt in ["glb", "obj", "stl", "ply", "3mf"]:
        p = Path(mesh_tools.export(m, tmp_path, "model", fmt))
        assert p.exists() and p.stat().st_size > 0, fmt
    # STL must be Z-up: lowest point at z=0 after Y-up -> Z-up rotation
    stl = trimesh.load(tmp_path / "model.stl")
    assert np.isclose(stl.bounds[0][2], 0, atol=1e-4)


def test_hf_patch_and_ckpt_cleanup(tmp_path, monkeypatch):
    import huggingface_hub as hh
    from studio3d import config
    seen = {}
    monkeypatch.setattr(hh, "snapshot_download", lambda *a, **k: seen.update(k) or "x")
    config.patch_hf_downloads()
    hh.snapshot_download(repo_id="r", allow_patterns=["a/*"])
    assert "*.ckpt" in seen["ignore_patterns"]

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snap = tmp_path / "hub" / "models--tencent--X" / "snapshots" / "abc" / "sub"
    snap.mkdir(parents=True)
    (snap / "model.fp16.ckpt").write_bytes(b"0" * 2048)
    (snap / "model.fp16.safetensors").write_bytes(b"1")
    (snap / "only.ckpt").write_bytes(b"2")
    config.remove_ckpt_duplicates()
    assert not (snap / "model.fp16.ckpt").exists()
    assert (snap / "only.ckpt").exists() and (snap / "model.fp16.safetensors").exists()
