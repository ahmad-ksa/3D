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


def _fresh_patch(monkeypatch, fake):
    import huggingface_hub as hh
    from studio3d import config
    monkeypatch.setattr(hh, "snapshot_download", fake)
    monkeypatch.setattr(config, "_RETRY_DELAYS", [0])
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    config.patch_hf_downloads()
    return hh, config


def test_hf_patch_skips_duplicates(monkeypatch):
    seen = {}
    hh, _ = _fresh_patch(monkeypatch, lambda *a, **k: seen.update(k) or "x")
    hh.snapshot_download(repo_id="r", allow_patterns=["a/*"])
    assert "*.ckpt" in seen["ignore_patterns"]
    assert any("paint" in p and "unet" in p for p in seen["ignore_patterns"])
    assert "hunyuan3d-dit-*/model.safetensors" in seen["ignore_patterns"]


def test_hf_patch_retries_network_errors(monkeypatch):
    import requests
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 4:
            try:
                raise OSError("[Errno 11001] getaddrinfo failed")
            except OSError as inner:
                raise requests.exceptions.ConnectionError("Max retries exceeded") from inner
        return "done"
    hh, _ = _fresh_patch(monkeypatch, flaky)
    assert hh.snapshot_download(repo_id="r") == "done" and calls["n"] == 4


def test_hf_patch_does_not_retry_other_errors(monkeypatch):
    import pytest
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        raise ValueError("repository not found")
    hh, _ = _fresh_patch(monkeypatch, bad)
    with pytest.raises(ValueError):
        hh.snapshot_download(repo_id="r")
    assert calls["n"] == 1


def test_hf_patch_gives_up_after_budget(monkeypatch):
    import pytest
    import requests
    hh, config = _fresh_patch(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(
        requests.exceptions.ReadTimeout("Read timed out.")))
    monkeypatch.setattr(config, "DOWNLOAD_RETRY_MINUTES", 0)
    with pytest.raises(requests.exceptions.ReadTimeout):
        hh.snapshot_download(repo_id="r")


def test_duplicate_cleanup(tmp_path, monkeypatch):
    from studio3d import config
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snap = tmp_path / "hub" / "models--tencent--X" / "snapshots" / "abc" / "sub"
    snap.mkdir(parents=True)
    (snap / "model.fp16.ckpt").write_bytes(b"0" * 2048)
    (snap / "model.fp16.safetensors").write_bytes(b"1")
    (snap / "only.ckpt").write_bytes(b"2")
    unet = tmp_path / "hub" / "models--tencent--X" / "snapshots" / "abc" / "hunyuan3d-paint-v2-0-turbo" / "unet"
    unet.mkdir(parents=True)
    (unet / "diffusion_pytorch_model.bin").write_bytes(b"b")
    (unet / "diffusion_pytorch_model.safetensors").write_bytes(b"s" * 100)
    dit = tmp_path / "hub" / "models--tencent--X" / "snapshots" / "abc" / "hunyuan3d-dit-v2-0"
    dit.mkdir(parents=True)
    (dit / "model.safetensors").write_bytes(b"f" * 64)
    (dit / "model.fp16.safetensors").write_bytes(b"h" * 32)
    config.remove_ckpt_duplicates()
    assert not (dit / "model.safetensors").exists() and (dit / "model.fp16.safetensors").exists()
    assert not (snap / "model.fp16.ckpt").exists()
    assert (snap / "only.ckpt").exists() and (snap / "model.fp16.safetensors").exists()
    assert (unet / "diffusion_pytorch_model.bin").exists()
    assert not (unet / "diffusion_pytorch_model.safetensors").exists()
