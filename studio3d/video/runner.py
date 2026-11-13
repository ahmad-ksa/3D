"""Storyboard -> finished video. Runs in the app / MCP process; heavy steps go to the worker."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import config, engine
from . import media
from . import storyboard as sbmod

log = logging.getLogger("studio3d.video")

WAN_FPS = 24
MOTION_MAX_SECONDS = 5.0   # longer voices loop the motion clip (keeps every scene fast)


def wan_frames(seconds: float) -> int:
    """Frames for Wan: 4k+1, between 25 and 121."""
    n = int(round(seconds * WAN_FPS))
    n = int(round((n - 1) / 4)) * 4 + 1
    return max(25, min(121, n))


def _worker_python() -> str:
    return sys.executable


def run_worker(cmd: str, args: dict, job: Path, on_progress: Callable[[float, str], None]) -> dict:
    """Start the worker process, relay its progress, kill it if ⏹ is pressed."""
    args = dict(args, cmd=cmd)
    tag = f"{cmd}_{int(time.time() * 1000)}"
    args_file = job / f"{tag}.json"
    args["result"] = str(job / f"{tag}_result.json")
    args.setdefault("parent_pid", os.getpid())
    args.setdefault("ffmpeg_dir", str(media.ffmpeg_dir(config.MODELS_DIR / "video" / "bin")))
    args_file.write_text(json.dumps(args, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = 0x08000000 if sys.platform == "win32" else 0
    log_file = job / f"{tag}.log"
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([_worker_python(), "-m", "studio3d.video.worker", str(args_file)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", cwd=str(config.PACKAGE_DIR.parent), env=env,
                                creationflags=flags)
        import threading
        stop = threading.Event()

        def killer():
            while not stop.wait(0.5):
                if engine.CANCEL.is_set() and proc.poll() is None:
                    proc.kill()
                    return

        th = threading.Thread(target=killer, daemon=True)
        th.start()
        try:
            for line in proc.stdout:
                lf.write(line)
                if line.startswith("@@P "):
                    parts = line.rstrip("\n").split(" ", 2)
                    try:
                        on_progress(float(parts[1]), parts[2] if len(parts) > 2 else "")
                    except ValueError:
                        pass
            proc.wait()
        finally:
            stop.set()
    engine.check_cancel()
    res_path = Path(args["result"])
    if not res_path.exists():
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-600:]
        code = proc.returncode
        hint = ""
        if "Loading pipeline components" in tail or code in (3221225477, -1073741819, 3221225725, -1073741571):
            hint = ("\nغالباً ذاكرة الجهاز (RAM) ما كفت لتحميل نموذج الفيديو. سكّر البرامج الثانية، "
                    "وتأكد من مدير المهام ما فيه python.exe قديم شغال من توليد سابق (أو أعد تشغيل الجهاز).")
        raise RuntimeError(f"توقف عامل الفيديو بشكل غير متوقع (رمز {code}).{hint}\n{tail}")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    if not res.get("ok"):
        log.error("worker %s failed: %s", cmd, res.get("trace"))
        raise RuntimeError(res.get("error") or "فشل توليد الفيديو")
    return res


def generate(storyboard_path, resolution: str | None = None, seed: int = 0,
             progress: Callable[[float, str], None] = lambda p, d: None,
             worker=run_worker) -> dict:
    """Make the whole video. Returns {"video", "job_dir", "scenes", "seconds"}."""
    engine.reset_cancel()
    job = engine.new_job_dir("video")
    sb = sbmod.load(storyboard_path, job)
    res = resolution if resolution in sbmod.RESOLUTIONS else sb.resolution
    n = len(sb.scenes)
    first = Image.open(sb.scenes[0].image)
    W, H = sbmod.frame_size(first.width, first.height, res)
    clips, info = [], []
    t0 = time.time()
    from ..pipeline import gpu_lock
    for s in sb.scenes:
        engine.check_cancel()
        d = job / f"scene_{s.index:02d}"
        d.mkdir(exist_ok=True)
        lo, hi = (s.index - 1) / n, s.index / n

        def sub(a, b, label):
            return lambda p, txt: progress(lo + (hi - lo) * (a + (b - a) * p), f"المشهد {s.index}/{n}: {txt or label}")

        voice = music = None
        seconds = s.seconds
        if s.audio and s.lipsync:
            voice = media.to_wav16k(s.audio, d / "voice.wav")
            seconds = max(sbmod.MIN_SECONDS, media.duration(voice) or s.seconds)
        elif s.audio:
            music = s.audio
            seconds = max(sbmod.MIN_SECONDS, media.duration(music) or s.seconds)
        motion_secs = min(seconds, MOTION_MAX_SECONDS)
        im = Image.open(s.image)
        w, h = sbmod.frame_size(im.width, im.height, res)
        with gpu_lock(), engine.MANAGER.lock:
            engine.claim_gpu()
            engine.MANAGER.unload()       # the 3D models leave the GPU while the video runs
            raw = d / "motion.mp4"
            worker("i2v", {"image": str(s.image), "prompt": s.prompt, "width": w, "height": h,
                           "frames": wan_frames(motion_secs), "fps": WAN_FPS, "seed": seed + s.index,
                           "out": str(raw)}, d, sub(0.0, 0.6 if voice else 0.9, "توليد الحركة"))
            if voice:
                talk = d / "talk.mp4"
                worker("lipsync", {"video": str(raw), "audio": str(voice), "out": str(talk), "seed": seed},
                       d, sub(0.6, 0.95, "مزامنة الشفايف"))
        sub(0.95, 1.0, "تجهيز")(0, "")
        final = d / "final.mp4"
        if voice:
            media.normalize_clip(talk, final, W, H, seconds=seconds, audio=talk)
        elif music:
            media.normalize_clip(raw, final, W, H, seconds=seconds, audio=music, loop_video=True)
        else:
            media.normalize_clip(raw, final, W, H, seconds=None)
        clips.append(final)
        info.append({"scene": s.index, "seconds": round(media.duration(final), 2), "talking": bool(voice),
                     "prompt": s.prompt})
    progress(0.98, "دمج المشاهد…")
    safe = "".join(ch for ch in sb.title if ch.isalnum() or ch in " _-").strip() or "video"
    out = media.concat(clips, job / f"{safe}.mp4")
    total = round(media.duration(out), 2)
    engine.save_meta(job, {"kind": "video", "title": sb.title, "resolution": res, "size": [W, H],
                           "scenes": info, "seconds": total, "minutes_taken": round((time.time() - t0) / 60, 1),
                           "files": [out.name]})
    progress(1.0, "تم")
    return {"video": str(out), "job_dir": str(job), "scenes": info, "seconds": total}
