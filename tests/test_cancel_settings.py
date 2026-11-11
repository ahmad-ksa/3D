import sys
import threading
import time
from pathlib import Path

import pytest
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio3d import engine  # noqa: E402


def test_cancellable_bar_stops_loops_and_updates():
    from tqdm import tqdm
    engine.reset_cancel()
    seen = []
    with pytest.raises(engine.Cancelled):
        for i in engine._CancellableBar(tqdm(range(100), disable=True)):
            seen.append(i)
            if i == 3:
                engine.request_cancel()
    assert seen == [0, 1, 2, 3]
    bar = engine._CancellableBar(tqdm(total=10, disable=True))
    with pytest.raises(engine.Cancelled):
        bar.update(1)
    engine.reset_cancel()


def test_pipeline_cancel_releases_locks(tmp_path, monkeypatch):
    from studio3d import config, pipeline
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(engine, "prepare_image", lambda img, rb: img.convert("RGBA"))

    def stopped(*a, **k):
        engine.request_cancel()
        engine.check_cancel()
    monkeypatch.setattr(engine, "generate_shape", stopped)
    img = Image.new("RGB", (32, 32))
    with pytest.raises(engine.Cancelled):
        pipeline.run("single", {"front": img}, None, pipeline.Options(randomize=False, formats=["glb"]))
    # the next job runs normally (flag reset, locks free)
    monkeypatch.setattr(engine, "generate_shape", lambda *a, **k: trimesh.creation.icosphere(2))
    res = pipeline.run("single", {"front": img}, None, pipeline.Options(randomize=False, formats=["glb"]))
    assert Path(res["preview"]).exists()


def test_blender_job_is_killed_on_cancel(tmp_path, monkeypatch):
    from studio3d import blender, config
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    slow = tmp_path / "python_slow"  # pretend 'blender' (a python) that never finishes
    slow.write_text("#!/bin/sh\nsleep 60\n")
    slow.chmod(0o755)
    monkeypatch.setattr(blender, "find_blender", lambda: str(slow))
    monkeypatch.setattr(blender, "_is_python", lambda exe: True)
    engine.reset_cancel()
    threading.Timer(1.0, engine.request_cancel).start()
    t = time.time()
    with pytest.raises(engine.Cancelled):
        blender.run_script("auto_rig", {"input": "x"})
    assert time.time() - t < 10
    engine.reset_cancel()


def test_last_options_saved_and_restored(tmp_path, monkeypatch):
    from studio3d import config, ui
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    ui.build()  # registers the option widgets of the 3 generation tabs
    values = ["عالي الجودة", True, False, True, True, 5, 20000, 120.0, 77, False, ["glb", "fbx"], "quad", True]
    ui._save_last_options(values)
    restored = ui.load_last_options()
    assert len(restored) == 3 * len(values)
    assert restored[: len(values)] == values
