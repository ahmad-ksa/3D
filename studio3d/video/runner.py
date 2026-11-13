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


def run_worker(cmd: str, args: dict, job: Path, on_progress: Callable[[float, str], None],
               on_item: Callable[[dict], None] | None = None) -> dict:
    """Start the worker process, relay its progress (and each finished scene), kill it if ⏹ is pressed."""
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
                lf.flush()
                if line.startswith("@@P "):
                    parts = line.rstrip("\n").split(" ", 2)
                    try:
                        on_progress(float(parts[1]), parts[2] if len(parts) > 2 else "")
                    except ValueError:
                        pass
                elif line.startswith("@@S ") and on_item is not None:
                    try:
                        on_item(json.loads(line[4:]))
                    except Exception:  # noqa: BLE001  (a preview must never stop the video)
                        log.exception("scene preview failed")
            proc.wait()
        finally:
            stop.set()
            if proc.poll() is None:      # an error in the app side: never leave the worker running
                proc.kill()
    engine.check_cancel()
    res_path = Path(args["result"])
    if not res_path.exists():
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-1500:]
        code = proc.returncode
        hint = ""
        if "[mem]" in tail or code in (3221225477, -1073741819, 3221225725, -1073741571, 3221226505):
            hint = ("\nغالباً ذاكرة الجهاز (RAM) ما كفت لتحميل نموذج الفيديو. سكّر البرامج الثانية، "
                    "وتأكد من مدير المهام ما فيه python.exe قديم شغال من توليد سابق (أو أعد تشغيل الجهاز).")
        raise RuntimeError(f"توقف عامل الفيديو بشكل غير متوقع (رمز {code}).{hint}\n{tail[-600:]}")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    if not res.get("ok"):
        log.error("worker %s failed: %s", cmd, res.get("trace"))
        raise RuntimeError(res.get("error") or "فشل توليد الفيديو")
    return res


def generate(storyboard_path, resolution: str | None = None, seed: int = 0,
             progress: Callable[[float, str], None] = lambda p, d: None,
             worker=run_worker, on_scene: Callable[[int, str], None] | None = None) -> dict:
    """Make the whole video. Returns {"video", "job_dir", "scenes", "seconds"}.
    The motion of ALL scenes is made by one worker process (the video model loads once), then the
    lip sync of all talking scenes by a second one. on_scene(index, mp4) is called as soon as a
    scene's motion clip is ready (live preview)."""
    engine.reset_cancel()
    job = engine.new_job_dir("video")
    sb = sbmod.load(storyboard_path, job)
    res = resolution if resolution in sbmod.RESOLUTIONS else sb.resolution
    n = len(sb.scenes)
    first = Image.open(sb.scenes[0].image)
    W, H = sbmod.frame_size(first.width, first.height, res)
    t0 = time.time()
    progress(0.0, f"تجهيز {n} مشهد…")
    plans = []
    for s in sb.scenes:
        engine.check_cancel()
        d = job / f"scene_{s.index:02d}"
        d.mkdir(exist_ok=True)
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
        plans.append({"scene": s, "dir": d, "voice": voice, "music": music, "seconds": seconds,
                      "motion_secs": motion_secs, "raw": d / "motion.mp4", "talk": d / "talk.mp4",
                      "width": w, "height": h})
    talking = [p for p in plans if p["voice"]]
    a_end = 0.6 if talking else 0.93
    b_end = 0.95

    def span(a, b):
        return lambda p, txt: progress(a + (b - a) * max(0.0, min(1.0, p)), txt)

    def motion_ready(item):
        if on_scene and item.get("ok") and item.get("out"):
            on_scene(int(item.get("index", 0)), item["out"])

    from ..pipeline import gpu_lock
    with gpu_lock(), engine.MANAGER.lock:
        engine.claim_gpu()
        engine.MANAGER.unload()       # the 3D models leave the GPU while the video runs
        jobs = [{"index": p["scene"].index, "label": f"المشهد {p['scene'].index}/{n}",
                 "image": str(p["scene"].image), "prompt": p["scene"].prompt, "width": p["width"],
                 "height": p["height"], "frames": wan_frames(p["motion_secs"]), "fps": WAN_FPS,
                 "seed": seed + p["scene"].index, "out": str(p["raw"])} for p in plans]
        worker("i2v", {"jobs": jobs}, job, span(0.0, a_end), on_item=motion_ready)
        lip = {}
        if talking:
            jobs = [{"index": p["scene"].index, "label": f"المشهد {p['scene'].index}/{n}",
                     "video": str(p["raw"]), "audio": str(p["voice"]), "out": str(p["talk"]), "seed": seed}
                    for p in talking]
            r = worker("lipsync", {"jobs": jobs}, job, span(a_end, b_end))
            lip = {int(it.get("index", 0)): it for it in (r.get("items") or [])}
    clips, info = [], []
    for k, p in enumerate(plans):
        engine.check_cancel()
        s = p["scene"]
        progress(b_end + (0.98 - b_end) * k / n, f"المشهد {s.index}/{n}: تجهيز")
        final = p["dir"] / "final.mp4"
        talked = bool(p["voice"]) and p["talk"].exists() and lip.get(s.index, {}).get("ok", True)
        warning = None
        if talked:
            # the scene's own audio (full quality), not the 16 kHz copy the lip-sync model listened to
            media.normalize_clip(p["talk"], final, W, H, seconds=p["seconds"], audio=s.audio)
        elif p["voice"] or p["music"]:
            if p["voice"]:
                warning = lip.get(s.index, {}).get("error") or "تعذّرت مزامنة الشفايف"
            media.normalize_clip(p["raw"], final, W, H, seconds=p["seconds"], audio=s.audio, loop_video=True)
        else:
            longer = p["seconds"] > p["motion_secs"] + 0.05
            media.normalize_clip(p["raw"], final, W, H, seconds=p["seconds"] if longer else None,
                                 loop_video=longer)
        clips.append(final)
        info.append({"scene": s.index, "seconds": round(media.duration(final), 2), "talking": talked,
                     "prompt": s.prompt, **({"warning": warning} if warning else {})})
    progress(0.98, "دمج المشاهد…")
    safe = "".join(ch for ch in sb.title if ch.isalnum() or ch in " _-").strip() or "video"
    out = media.concat(clips, job / f"{safe}.mp4")
    total = round(media.duration(out), 2)
    engine.save_meta(job, {"kind": "video", "title": sb.title, "resolution": res, "size": [W, H],
                           "scenes": info, "seconds": total, "minutes_taken": round((time.time() - t0) / 60, 1),
                           "files": [out.name]})
    progress(1.0, "تم")
    return {"video": str(out), "job_dir": str(job), "scenes": info, "seconds": total}
