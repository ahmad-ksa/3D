"""ffmpeg helpers (the ffmpeg build that ships inside the imageio-ffmpeg Python package)."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

FPS = 25          # LatentSync works at 25 fps; the final video too
AUDIO_RATE = 48000


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_dir(cache: Path) -> Path:
    """A folder with the binary named plain 'ffmpeg(.exe)' (LatentSync calls `ffmpeg` by name)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    dst = cache / name
    src = Path(ffmpeg_exe())
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return cache


def _run(args: list, cwd=None):
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *map(str, args)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, creationflags=flags)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr.decode("utf-8", "replace")[-500:])
    return p


def duration(path) -> float:
    """Length of an audio/video file in seconds (0 if unknown)."""
    flags = 0x08000000 if sys.platform == "win32" else 0
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    m = re.search(rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def to_wav16k(src, dst) -> Path:
    """Any audio -> mono 16 kHz wav (what the lip-sync model reads)."""
    _run(["-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst])
    return Path(dst)


def cut_audio(src, dst, start: float, length: float) -> Path:
    """A piece of an audio file (start / length in seconds) as wav."""
    _run(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src, "-ac", "2", "-ar", str(AUDIO_RATE),
          "-c:a", "pcm_s16le", dst])
    return Path(dst)


def normalize_clip(src, dst, width: int, height: int, seconds: float | None = None, audio=None,
                   loop_video: bool = False) -> Path:
    """Re-encode a clip to the common format (size, 25 fps, h264, aac stereo 48k).
    Clips without a voice get a silent track so all scenes can be joined.
    loop_video: repeat the (short) motion clip until the audio ends (music under a scene)."""
    args = (["-stream_loop", "-1"] if loop_video else []) + ["-i", src]
    if audio is None:
        args += ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}"]
    else:
        args += ["-i", audio]
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")
    args += ["-map", "0:v:0", "-map", "1:a:0", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-c:a", "aac", "-ar", str(AUDIO_RATE), "-ac", "2", "-shortest"]
    if seconds:
        args += ["-t", f"{seconds:.3f}"]
    args.append(dst)
    _run(args)
    return Path(dst)


def concat(clips: list, dst) -> Path:
    """Join normalized clips (same format) into one mp4 without re-encoding."""
    dst = Path(dst)
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(c).resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
    _run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", dst])
    lst.unlink(missing_ok=True)
    return dst
