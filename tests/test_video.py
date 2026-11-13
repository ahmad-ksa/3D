"""2D video from a storyboard: everything except the big AI models (those need the user's GPU)."""
import json
import os
import subprocess
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio3d import engine
from studio3d.video import face_mp, media, runner
from studio3d.video import storyboard as sb


def _wav(p: Path, seconds=2.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return p


def _clip(p: Path, seconds: float, w=160, h=96, fps=24, color="red", audio: Path | None = None):
    args = [media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s={w}x{h}:r={fps}:d={seconds}"]
    if audio:
        args += ["-i", str(audio), "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)]
    subprocess.run(args, check=True)
    return p


def _storyboard(tmp: Path, zipped=True):
    d = tmp / "sb_src"
    d.mkdir()
    Image.new("RGB", (640, 360), (200, 50, 50)).save(d / "01.png")
    Image.new("RGB", (640, 360), (50, 50, 200)).save(d / "02.png")
    _wav(d / "01.wav", 2.0)
    data = {"title": "قصة test", "resolution": "480p", "scenes": [
        {"image": "01.png", "prompt": "she waves", "audio": "01.wav"},
        {"image": "02.png", "description": "zoom in", "seconds": 3},
    ]}
    (d / "storyboard.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if not zipped:
        return d / "storyboard.json"
    z = tmp / "story.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in d.iterdir():
            zf.write(f, f"folder/{f.name}")
    return z


def test_storyboard_zip_and_json(tmp_path):
    s = sb.load(_storyboard(tmp_path), tmp_path / "job")
    assert [x.index for x in s.scenes] == [1, 2]
    assert s.scenes[0].audio and s.scenes[0].audio.exists()
    assert s.scenes[1].prompt == "zoom in" and s.scenes[1].seconds == 3
    (tmp_path / "b").mkdir()
    j = sb.load(_storyboard(tmp_path / "b", zipped=False))
    assert j.scenes[0].image.name == "01.png"


def test_storyboard_errors(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"scenes": [{"image": "missing.png"}]}), encoding="utf-8")
    with pytest.raises(sb.StoryboardError):
        sb.load(tmp_path / "s.json")
    (tmp_path / "e.json").write_text(json.dumps({"scenes": []}), encoding="utf-8")
    with pytest.raises(sb.StoryboardError):
        sb.load(tmp_path / "e.json")
    bad = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../../escape.json", "{}")
    with pytest.raises(sb.StoryboardError):
        sb.load(bad, tmp_path / "job")


def test_frame_size_keeps_aspect_and_multiple_of_32():
    w, h = sb.frame_size(1920, 1080, "480p")
    assert w % 32 == 0 and h % 32 == 0 and abs(w / h - 16 / 9) < 0.1
    w2, h2 = sb.frame_size(1080, 1920, "720p")
    assert h2 > w2 and w2 * h2 > w * h
    assert runner.wan_frames(5) == 121 and runner.wan_frames(0.2) == 25 and (runner.wan_frames(2.3) - 1) % 4 == 0


def test_template_is_a_valid_storyboard(tmp_path):
    z = sb.write_template(tmp_path / "t.zip")
    s = sb.load(z, tmp_path / "job")
    assert len(s.scenes) == 2 and s.scenes[0].audio is not None


def test_media_normalize_and_concat(tmp_path):
    a = _clip(tmp_path / "a.mp4", 2.0, color="red")
    voice = _wav(tmp_path / "v.wav", 1.5)
    b = _clip(tmp_path / "b.mp4", 1.5, color="blue", audio=voice)
    na = media.normalize_clip(a, tmp_path / "na.mp4", 192, 112)
    nb = media.normalize_clip(b, tmp_path / "nb.mp4", 192, 112, seconds=1.5, audio=b)
    out = media.concat([na, nb], tmp_path / "out.mp4")
    assert abs(media.duration(out) - 3.5) < 0.3
    assert abs(media.duration(media.to_wav16k(voice, tmp_path / "v16.wav")) - 1.5) < 0.1


def test_face_landmarks_map_to_latentsync_layout():
    # fake 478-point mesh: a face centred at (200, 220) in a 400x400 frame
    rng = np.random.default_rng(0)
    xy = rng.uniform(150, 250, size=(478, 2))
    for i, (a, b) in enumerate(face_mp.BROW_IMG_LEFT):
        xy[a] = (140 + i * 8, 160)
        xy[b] = (140 + i * 8, 170)
    for i, (a, b) in enumerate(face_mp.BROW_IMG_RIGHT):
        xy[a] = (220 + i * 8, 160)
        xy[b] = (220 + i * 8, 170)
    xy[face_mp.NOSE_BRIDGE_TOP] = (200, 175)
    xy[face_mp.NOSE_BRIDGE_MID] = (200, 195)
    jaw = face_mp.JAW
    for k, idx in enumerate(jaw):  # U shape, chin lowest at the middle
        u = k / (len(jaw) - 1) * 2 - 1
        xy[idx] = (200 + 90 * u, 300 - 80 * u * u)
    lmk = face_mp.to_106(xy)
    left = lmk[[43, 48, 49, 51, 50]].mean(axis=0)
    right = lmk[101:106].mean(axis=0)
    assert left[0] < right[0]                      # template expects image-left brow first
    assert abs(lmk[16][1] - 300) < 2               # chin in the middle of the contour
    (x1, y1, x2, y2), _ = face_mp.box_from_106(lmk, 400, 400)
    assert x1 < 120 and x2 > 280 and y2 > 300 and y1 < 160


def _fake_batch_worker(calls, no_face=()):
    """Stands in for the worker process: one call per phase, a list of scene jobs each."""

    def fake_worker(cmd, args, job, on_progress, on_item=None):
        calls.append((cmd, args))
        items = []
        for k, a in enumerate(args["jobs"]):
            on_progress((k + 0.5) / len(args["jobs"]), "…")
            if cmd == "i2v":
                assert (a["frames"] - 1) % 4 == 0 and a["width"] % 32 == 0
                _clip(Path(a["out"]), (a["frames"] - 1) / 24, a["width"], a["height"])
                it = {"index": a["index"], "out": a["out"], "ok": True}
            elif a["index"] in no_face:
                it = {"index": a["index"], "ok": False, "no_face": True, "error": "no face"}
            else:  # lipsync: video looped to the voice + the voice muxed in
                secs = media.duration(a["audio"])
                _clip(Path(a["out"]), secs, 160, 96, fps=25, audio=Path(a["audio"]))
                it = {"index": a["index"], "out": a["out"], "ok": True}
            items.append(it)
            if on_item:
                on_item(it)
        return {"ok": True, "items": items}

    return fake_worker


def test_runner_makes_one_video_from_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.config, "OUTPUT_DIR", tmp_path / "out")
    calls, previews = [], []
    seen = []
    r = runner.generate(_storyboard(tmp_path), "480p", progress=lambda p, d: seen.append(p),
                        worker=_fake_batch_worker(calls), on_scene=lambda i, f: previews.append((i, f)))
    # ONE worker for the motion of every scene, ONE for all lip syncs (models load once each)
    assert [c[0] for c in calls] == ["i2v", "lipsync"]
    assert [j["index"] for j in calls[0][1]["jobs"]] == [1, 2] and len(calls[1][1]["jobs"]) == 1
    assert [i for i, _ in previews] == [1, 2] and all(Path(f).exists() for _, f in previews)
    assert Path(r["video"]).exists()
    assert abs(r["seconds"] - 5.0) < 0.5           # 2 s talking scene + 3 s scene
    assert r["scenes"][0]["talking"] and not r["scenes"][1]["talking"]
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_runner_scene_without_face_keeps_voice(tmp_path, monkeypatch):
    """No face for the lip-sync model in one scene: that scene keeps its motion + voice, the video is made."""
    monkeypatch.setattr(engine.config, "OUTPUT_DIR", tmp_path / "out")
    r = runner.generate(_storyboard(tmp_path), "480p", worker=_fake_batch_worker([], no_face={1}))
    s1 = r["scenes"][0]
    assert not s1["talking"] and s1.get("warning") and abs(s1["seconds"] - 2.0) < 0.3
    assert abs(r["seconds"] - 5.0) < 0.5


def test_runner_long_silent_scene_is_not_cut(tmp_path, monkeypatch):
    """A silent 7 s scene: the 5 s motion clip is repeated, the scene is not cut to 5 s."""
    monkeypatch.setattr(engine.config, "OUTPUT_DIR", tmp_path / "out")
    img = tmp_path / "a.png"
    Image.new("RGB", (320, 180), (200, 80, 20)).save(img)
    path = sb.from_inputs([img], "walks | 7", None, False, tmp_path / "w")
    r = runner.generate(path, "480p", worker=_fake_batch_worker([]))
    assert abs(r["seconds"] - 7.0) < 0.3


def test_runner_stops_on_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.config, "OUTPUT_DIR", tmp_path / "out")

    def fake_worker(cmd, args, job, on_progress, on_item=None):
        engine.request_cancel()
        engine.check_cancel()

    with pytest.raises(engine.Cancelled):
        runner.generate(_storyboard(tmp_path), "480p", worker=fake_worker)
    engine.reset_cancel()


def test_worker_reports_errors_as_json(tmp_path):
    args = tmp_path / "a.json"
    args.write_text(json.dumps({"cmd": "nope", "result": str(tmp_path / "r.json")}), encoding="utf-8")
    from studio3d.video import worker
    assert worker.main([str(args)]) == 1
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["ok"] is False


_VIDEO_LIBS = os.environ.get("STUDIO3D_VIDEO_LIBS")
_needs_libs = pytest.mark.skipif(not (_VIDEO_LIBS and Path(_VIDEO_LIBS, "diffusers").exists()),
                                 reason="needs the video diffusers (video_libs)")

# Runs in a fresh process, exactly like the real worker (newer diffusers first on the path).
_WORKER_CHECK = r"""
import sys, torch, numpy as np
from studio3d.video import worker
worker._setup_paths()
import diffusers
assert tuple(int(x) for x in diffusers.__version__.split(".")[:2]) >= (0, 35), diffusers.__version__
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler, WanImageToVideoPipeline
from PIL import Image
# a tiny random Wan2.2-style model (expand_timesteps, no image encoder) through the worker's code path
vae = AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                       temperal_downsample=[False, True, True], latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
tr = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=4,
                           out_channels=4, text_dim=16, freq_dim=16, ffn_dim=16, num_layers=1, rope_max_seq_len=32)
sch = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=8.0)
pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=sch, transformer=tr,
                               image_encoder=None, image_processor=None, expand_timesteps=True)
class Tok:
    def __call__(self, prompt, **k):
        n = k["max_length"]; m = torch.zeros(len(prompt), n, dtype=torch.long); m[:, :5] = 1
        return type("T", (), {"input_ids": torch.zeros(len(prompt), n, dtype=torch.long), "attention_mask": m})()
class TE(torch.nn.Module):
    dtype = torch.float32
    def forward(self, ids, mask):
        return type("O", (), {"last_hidden_state": torch.randn(ids.shape[0], ids.shape[1], 16)})()
class Enc:
    tokenizer = Tok(); text_encoder = TE(); _execution_device = torch.device("cpu")
emb = WanImageToVideoPipeline._get_t5_prompt_embeds(Enc(), "a cat waves", 1, 512, torch.device("cpu"), torch.float32)
img = sys.argv[1]
Image.new("RGB", (80, 48), (120, 30, 200)).save(img)
frames = worker.run_i2v(pipe, {"image": img, "width": 64, "height": 64, "frames": 9, "steps": 2}, emb)
assert len(frames) == 9 and np.asarray(frames[0]).shape == (64, 64, 3)
# Wan2.2 VAE (patch_size=2): tiling must stay off (diffusers 0.35 tiled encode skips patchify)
vae22 = AutoencoderKLWan(base_dim=8, z_dim=4, in_channels=12, out_channels=12, patch_size=2,
                         dim_mult=[1, 1, 1, 1], num_res_blocks=1, temperal_downsample=[False, True, True],
                         latents_mean=[0.0] * 4, latents_std=[1.0] * 4, is_residual=True)
assert worker._maybe_tiling(vae22) is False and not vae22.use_tiling
assert worker._maybe_tiling(vae) is True
x = torch.randn(1, 3, 5, 288, 288)
vae22.encode(x)          # full-size frame encodes through patchify without the tiling bug
print("WORKER-OK")
"""


@_needs_libs
def test_worker_video_code_path_with_tiny_models(tmp_path):
    import sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", _WORKER_CHECK, str(tmp_path / "in.png")], cwd=root,
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, PYTHONPATH=str(root)))
    assert "WORKER-OK" in r.stdout, r.stderr[-2000:]


def test_story_text_and_boxes_make_a_storyboard(tmp_path):
    lines = sb.parse_story_text("1. the girl waves | 5\n\nالمشهد 2: she sings (4ث)\nzoom in")
    assert lines == [("the girl waves", 5.0), ("she sings", 4.0), ("zoom in", None)]
    imgs = []
    for i in range(2):
        p = tmp_path / f"i{i}.png"
        Image.new("RGB", (320, 180), (40 * i, 80, 120)).save(p)
        imgs.append(p)
    song = _wav(tmp_path / "song.wav", 9.0)
    path = sb.from_inputs(imgs, "the girl waves | 5\nshe sings\nzoom in", song, True, tmp_path / "w")
    s = sb.load(path)
    assert len(s.scenes) == 3
    assert s.scenes[2].image == imgs[1].resolve()            # last image reused
    lens = [media.duration(x.audio) for x in s.scenes]
    assert abs(lens[0] - 5) < 0.2 and abs(lens[1] - 2) < 0.2 and abs(lens[2] - 2) < 0.2  # song split
    assert all(x.lipsync for x in s.scenes)
    # music only (no lip sync) -> the scene still lasts as long as its music
    path2 = sb.from_inputs(imgs[:1], "", song, False, tmp_path / "w2")
    s2 = sb.load(path2)
    assert len(s2.scenes) == 1 and not s2.scenes[0].lipsync


def test_runner_music_scene_loops_motion(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.config, "OUTPUT_DIR", tmp_path / "out")
    img = tmp_path / "a.png"
    Image.new("RGB", (320, 180), (200, 80, 20)).save(img)
    song = _wav(tmp_path / "song.wav", 7.0)
    path = sb.from_inputs([img], "dance", song, False, tmp_path / "w")
    calls = []
    fake_worker = _fake_batch_worker(calls)

    r = runner.generate(path, "480p", worker=fake_worker)
    assert [c[0] for c in calls] == ["i2v"]                       # no lip sync for music
    assert abs(r["seconds"] - 7.0) < 0.4          # 5 s motion looped under 7 s of music


def test_story_lines_pick_image_by_character_name(tmp_path):
    a = tmp_path / "الأب.png"
    b = tmp_path / "Rehad ريهاد.png"
    for p in (a, b):
        Image.new("RGB", (320, 180), (90, 90, 90)).save(p)
    song = _wav(tmp_path / "song.wav", 6.0)
    path = sb.from_inputs([a, b], "[ريهاد] she sings | 2\n[الاب] he plays guitar | 1.5\n[ريهاد] she jumps | 2.5",
                          song, True, tmp_path / "w")
    s = sb.load(path)
    assert [x.image.name for x in s.scenes] == [b.name, a.name, b.name]
    assert s.scenes[0].prompt == "she sings"
    assert abs(media.duration(s.scenes[1].audio) - 1.5) < 0.1


def test_worker_stops_when_the_app_is_gone(tmp_path):
    """A worker whose app was closed must exit (it would keep ~15 GB RAM + the GPU)."""
    import sys
    import time
    args = tmp_path / "a.json"
    args.write_text(json.dumps({"cmd": "sleep_test", "parent_pid": 999999, "result": str(tmp_path / "r.json")}),
                    encoding="utf-8")
    code = ("import sys, time; from studio3d.video import worker; "
            "worker.COMMANDS['sleep_test'] = lambda a, c: time.sleep(30) or {}; "
            "sys.exit(worker.main([sys.argv[1]]))")
    root = Path(__file__).resolve().parents[1]
    t = time.time()
    p = subprocess.run([sys.executable, "-c", code, str(args)], cwd=root, timeout=60,
                       env=dict(os.environ, PYTHONPATH=str(root)))
    assert p.returncode == 3 and time.time() - t < 20


def _run_check(script: str, tmp_path: Path, marker: str):
    import sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "tests" / script), str(tmp_path)], cwd=root,
                       capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, PYTHONPATH=str(root), STUDIO3D_HOME=str(tmp_path / "home")))
    assert marker in r.stdout, (r.stdout[-2000:] + r.stderr[-3000:])


@_needs_libs
def test_worker_loads_wan_in_8bit_once_for_all_scenes(tmp_path):
    """Tiny FastWan-shaped folder on disk: 8-bit loader == from_pretrained + layerwise casting (bit-exact),
    still exact with block streaming, text encoder 8-bit and really freed, 2 scenes with ONE model load."""
    _run_check("_video_loader_check.py", tmp_path, "LOADER-OK")


@_needs_libs
def test_worker_lipsync_end_to_end_with_tiny_models(tmp_path):
    """Real LatentSync pipeline + worker: ffmpeg by name in a folder with spaces/Arabic, voice longer than
    the clip (looped), face lost in some frames (nearest face used), no face at all (scene reported)."""
    _run_check("_video_lipsync_check.py", tmp_path, "LIPSYNC-OK")


def test_run_worker_relays_progress_and_finished_scenes(tmp_path, monkeypatch):
    """The real run_worker with a stand-in worker process: progress lines, finished scenes, result."""
    import sys
    if sys.platform == "win32":
        pytest.skip("uses a shell script as the worker")
    script = tmp_path / "fake_python"
    script.write_text(
        "#!/bin/sh\n"
        "eval \"args=\\${$#}\"\n"
        "echo '@@P 0.5 half'\n"
        "echo '@@S {\"index\": 3, \"out\": \"x.mp4\", \"ok\": true}'\n"
        f"{sys.executable} -c \"import json,sys; a=json.load(open(sys.argv[1])); "
        "open(a['result'],'w').write(json.dumps({'ok': True, 'items': []}))\" \"$args\"\n",
        encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(runner, "_worker_python", lambda: str(script))
    monkeypatch.setattr(runner.media, "ffmpeg_dir", lambda cache: Path(cache))
    got, items = [], []
    r = runner.run_worker("i2v", {"jobs": []}, tmp_path, lambda p, t: got.append((p, t)), on_item=items.append)
    assert r["ok"] and got == [(0.5, "half")] and items == [{"index": 3, "out": "x.mp4", "ok": True}]


def test_video_tab_shows_each_scene_when_ready(tmp_path, monkeypatch):
    """The 🎬 tab streams each finished motion clip, then the whole video; errors become messages."""
    import gradio as gr

    from studio3d import ui

    def fake_generate(path, resolution, progress=None, on_scene=None):
        progress(0.5, "…")
        on_scene(1, "s1.mp4")
        on_scene(2, "s2.mp4")
        return {"video": "all.mp4", "job_dir": str(tmp_path), "seconds": 9.0,
                "scenes": [{"scene": 1, "talking": True}, {"scene": 2, "talking": False, "warning": "no face"}]}

    monkeypatch.setattr(runner, "generate", fake_generate)
    sbf = _storyboard(tmp_path, zipped=False)
    outs = list(ui.video_tool(None, "", None, True, "480p", str(sbf)))
    assert [o[0] for o in outs] == ["s1.mp4", "s2.mp4", "all.mp4"] and "no face" in outs[-1][1]

    def failing(path, resolution, progress=None, on_scene=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "generate", failing)
    with pytest.raises(gr.Error):
        list(ui.video_tool(None, "", None, True, "480p", str(sbf)))
