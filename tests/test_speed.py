"""Speed fixes: no internet when models are on disk, GPU hand-over between processes,
no random init for fully-loaded parts, stale stop flag."""
import json
import os

import torch

from studio3d import config, engine


def test_local_only_forces_offline(monkeypatch):
    import huggingface_hub as hh
    config.patch_hf_downloads()
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return "/x"

    # the patch wraps the original; swap the original by re-patching around a fake
    monkeypatch.setattr(hh, "snapshot_download", fake)
    fake._studio3d = False
    config.patch_hf_downloads()
    with config.local_only(True):
        hh.snapshot_download(repo_id="a/b")
    assert seen.get("local_files_only") is True
    seen.clear()
    hh.snapshot_download(repo_id="a/b", allow_patterns=["x/*"])
    assert not seen.get("local_files_only")


def test_no_random_init_skips_work_and_restores():
    import torch.nn.init as init
    before = init.normal_
    with engine.no_random_init():
        lin = torch.nn.Linear(4, 4)
        assert init.normal_ is not before
    assert init.normal_ is before
    # weights are then fully replaced by a strict load
    lin.load_state_dict({"weight": torch.ones(4, 4), "bias": torch.zeros(4)}, strict=True)
    assert torch.equal(lin.weight, torch.ones(4, 4))
    # outside the context, init works again
    a = torch.nn.Linear(64, 64).weight
    assert a.abs().sum() > 0


def test_claim_gpu_asks_other_process_and_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_OWNER", tmp_path / "owner.json")
    monkeypatch.setattr(engine, "_REQUEST", tmp_path / "req")
    other = os.getpid() + 1
    (tmp_path / "owner.json").write_text(json.dumps({"pid": other}))
    monkeypatch.setattr(engine, "_pid_alive", lambda pid: True)

    import threading
    import time

    def owner_frees():  # the other process notices the request and frees its models
        for _ in range(50):
            if (tmp_path / "req").exists():
                (tmp_path / "owner.json").unlink()
                return
            time.sleep(0.05)

    th = threading.Thread(target=owner_frees)
    th.start()
    t = time.time()
    engine.claim_gpu(timeout=5)
    th.join()
    assert time.time() - t < 4
    assert not (tmp_path / "req").exists()


def test_claim_gpu_ignores_dead_or_self(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_OWNER", tmp_path / "owner.json")
    monkeypatch.setattr(engine, "_REQUEST", tmp_path / "req")
    (tmp_path / "owner.json").write_text(json.dumps({"pid": os.getpid()}))
    engine.claim_gpu(timeout=5)
    assert not (tmp_path / "req").exists()
    (tmp_path / "owner.json").write_text(json.dumps({"pid": 999999}))
    monkeypatch.setattr(engine, "_pid_alive", lambda pid: False)
    engine.claim_gpu(timeout=5)
    assert not (tmp_path / "req").exists()


def test_watcher_frees_idle_models_on_request(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(engine, "_OWNER", tmp_path / "owner.json")
    monkeypatch.setattr(engine, "_REQUEST", tmp_path / "req")
    m = engine.ModelManager()
    m.get(("x",), lambda: object())
    assert json.loads((tmp_path / "owner.json").read_text())["pid"] == os.getpid()
    (tmp_path / "req").write_text(json.dumps({"pid": os.getpid() + 1}))
    for _ in range(40):
        if m._obj is None and not (tmp_path / "owner.json").exists():
            break
        time.sleep(0.1)
    assert m._obj is None and not (tmp_path / "owner.json").exists()


def test_stale_stop_does_not_kill_next_tool(monkeypatch, tmp_path):
    from studio3d import blender, pipeline
    engine.request_cancel()
    seen = {}

    def fake_quad(model_path, job, **k):
        seen["cancel_at_start"] = engine.CANCEL.is_set()
        raise RuntimeError("stop here")

    monkeypatch.setattr(blender, "quad_remesh", fake_quad)
    try:
        pipeline.quad(str(tmp_path / "m.glb"))
    except RuntimeError:
        pass
    assert seen["cancel_at_start"] is False


def test_fast_shape_loader_only_skips_init_for_strict_parts(monkeypatch):
    """The DiT and image encoder are skipped (fully loaded after); the VAE (loose load) is not."""
    import hy3dgen.shapegen.pipelines as hp
    import torch.nn.init as init
    calls = []
    real_normal = init.normal_

    def fake_instantiate(cfg, **kw):
        calls.append((cfg["target"], init.normal_ is not real_normal))
        return torch.nn.Linear(1, 1)

    class FakePipe:
        @classmethod
        def from_pretrained(cls, *a, **k):
            for t in ("hy3dgen.shapegen.models.Hunyuan3DDiT", "hy3dgen.shapegen.models.ShapeVAE",
                      "hy3dgen.shapegen.models.autoencoders.ShapeVAE",
                      "hy3dgen.shapegen.models.conditioner.SingleImageEncoder",
                      "hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler"):
                hp.instantiate_from_config({"target": t})
            p = cls()
            p.image_processor = None
            return p

        def enable_flashvdm(self, **k):
            pass

    import hy3dgen.shapegen as hs
    monkeypatch.setattr(hp, "instantiate_from_config", fake_instantiate)
    monkeypatch.setattr(hs, "Hunyuan3DDiTFlowMatchingPipeline", FakePipe)
    monkeypatch.setattr(engine, "shape_ready", lambda *a: True)
    monkeypatch.setattr(engine, "_checkpoint_parts", lambda *a: {"model", "vae", "conditioner"})
    engine._load_shape("single", "turbo")
    got = dict(calls)
    assert got["hy3dgen.shapegen.models.Hunyuan3DDiT"] is True
    assert got["hy3dgen.shapegen.models.conditioner.SingleImageEncoder"] is True
    assert got["hy3dgen.shapegen.models.ShapeVAE"] is False
    assert got["hy3dgen.shapegen.models.autoencoders.ShapeVAE"] is False
    assert got["hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler"] is False
    assert hp.instantiate_from_config is fake_instantiate      # restored
    # checkpoint without conditioner weights -> encoder keeps its normal init
    calls.clear()
    monkeypatch.setattr(engine, "_checkpoint_parts", lambda *a: {"model", "vae"})
    engine._load_shape("single", "turbo")
    assert dict(calls)["hy3dgen.shapegen.models.conditioner.SingleImageEncoder"] is False
