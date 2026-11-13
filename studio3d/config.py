"""Paths, model ids and quality presets."""
import contextlib as _contextlib
import os
import sys
import threading as _threading
from pathlib import Path

APP_NAME = "Studio3D"
APP_TITLE = "استوديو المجسمات ثلاثية الأبعاد"
VERSION = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "vendor"


def _data_root() -> Path:
    env = os.environ.get("STUDIO3D_HOME")
    if env:
        return Path(env)
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME
    return Path.home() / ".studio3d"


DATA_ROOT = _data_root()
OUTPUT_DIR = Path(os.environ.get("STUDIO3D_OUTPUTS", DATA_ROOT / "outputs"))
MODELS_DIR = Path(os.environ.get("STUDIO3D_MODELS", DATA_ROOT / "models"))

# Keep every downloaded model inside our own folder (easy to back up / delete).
os.environ.setdefault("HY3DGEN_MODELS", str(MODELS_DIR / "hy3dgen"))
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR / "rembg"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows
# Slow / unstable connections: wait longer before a read times out (default is 10 s).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Never send our own local UI traffic through a (corporate) proxy.
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    if "127.0.0.1" not in _v:
        os.environ[_k] = ",".join(x for x in (_v, "127.0.0.1", "localhost") if x)

HOST = "127.0.0.1"
PORT = int(os.environ.get("STUDIO3D_PORT", "7860"))

# ---------------------------------------------------------------------------
# Models (Tencent Hunyuan3D 2.0 family, run locally)
# ---------------------------------------------------------------------------
SHAPE_MODELS = {
    # 2-4 photos (front / back / left / right)
    "multiview": {
        "repo": "tencent/Hunyuan3D-2mv",
        "turbo": "hunyuan3d-dit-v2-mv-turbo",
        "full": "hunyuan3d-dit-v2-mv",
    },
    # one photo (also used by text-to-3D)
    "single": {
        "repo": "tencent/Hunyuan3D-2",
        "turbo": "hunyuan3d-dit-v2-0-turbo",
        "full": "hunyuan3d-dit-v2-0",
    },
}
TEXTURE_MODEL = {"repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-paint-v2-0-turbo"}
TEXT2IMAGE_MODEL = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
# pose guide for text-to-image (T-pose / A-pose characters ready for a skeleton), ~3 GB
POSE_CONTROLNET_MODEL = "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose"

# 2D video from a storyboard (runs in a separate worker process, see studio3d/video)
VIDEO_MODEL = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"   # Wan2.2 TI2V-5B, 4-step distilled
VIDEO_TEXT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"              # text encoder if the above lacks it
LIPSYNC_MODEL = "ByteDance/LatentSync-1.5"                         # runs in ~8 GB (1.6 needs 18 GB)
LIPSYNC_VAE = "stabilityai/sd-vae-ft-mse"
# newer diffusers for the video model only (the 3D models keep diffusers 0.32)
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"     # every finished video is copied here (📂 button in the 🎬 tab)
VIDEO_LIBS_DIR = Path(os.environ.get("STUDIO3D_VIDEO_LIBS", DATA_ROOT / "video_libs"))

# ---------------------------------------------------------------------------
# Quality presets (tuned for 8 GB laptop GPUs such as RTX 5050 / 4060)
# ---------------------------------------------------------------------------
PRESETS = {
    "سريع": {"variant": "turbo", "steps": 5, "octree": 256, "guidance": 5.0},
    "متوازن": {"variant": "turbo", "steps": 10, "octree": 384, "guidance": 5.0},
    "عالي الجودة": {"variant": "full", "steps": 30, "octree": 512, "guidance": 5.0},
}
DEFAULT_PRESET = "متوازن"
NUM_CHUNKS = 8000  # lower = less VRAM during mesh extraction

EXPORT_FORMATS = ["glb", "obj", "stl", "ply", "3mf"]


# Files the program never loads, so we never download them:
#  * every Hunyuan3D checkpoint exists twice: .ckpt and .safetensors (~5 GB each) -> we load .safetensors
#  * the paint UNet exists as .bin (loaded by hy3dgen) and .safetensors (unused, ~3.7 GB)
#  * full-quality DiT folders ship model.safetensors next to model.fp16.safetensors -> we load fp16
SKIP_PATTERNS = ["*.ckpt", "hunyuan3d-paint-v2-0*/unet/*.safetensors", "hunyuan3d-dit-*/model.safetensors",
                 "hunyuan3d-vae-*/model.safetensors"]

DOWNLOAD_RETRY_MINUTES = float(os.environ.get("STUDIO3D_RETRY_MINUTES", "2"))
_RETRY_DELAYS = [5, 10, 15, 20]  # seconds, then every 20 s until the 2-minute budget is used


def is_network_error(e: BaseException) -> bool:
    """Connection drops, timeouts, DNS failures (including wrapped/chained ones)."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        name = type(e).__name__
        text = str(e)
        if any(k in name for k in ("Connection", "Timeout", "ChunkedEncoding", "ProtocolError",
                                   "IncompleteRead", "RemoteDisconnected", "NameResolution",
                                   "MaxRetry", "SSLError", "LocalEntryNotFound")):
            return True
        if any(k in text for k in ("getaddrinfo", "timed out", "Connection broken", "Connection reset",
                                   "Temporary failure in name resolution", "No route to host")):
            return True
        e = e.__cause__ or e.__context__
    return False


_LOCAL = _threading.local()


@_contextlib.contextmanager
def local_only(enabled: bool = True):
    """Inside this block every Hugging Face lookup of THIS thread uses the files already on disk,
    with no internet request at all (used when a model is known to be fully downloaded).
    Without it, every model load asks huggingface.co about the latest version first, which costs
    seconds normally and up to a minute on a stalled connection — and can re-download a whole
    model when Tencent publishes a small change."""
    old = getattr(_LOCAL, "on", False)
    _LOCAL.on = bool(enabled) or old
    try:
        yield
    finally:
        _LOCAL.on = old


def is_local_only() -> bool:
    return getattr(_LOCAL, "on", False)


def patch_hf_downloads():
    """Make every model download (ours and Hunyuan3D's internal ones):
    1. skip files we never load (see SKIP_PATTERNS);
    2. survive unstable internet: on a network error wait and retry automatically
       (up to DOWNLOAD_RETRY_MINUTES), resuming from where it stopped."""
    import logging
    import time

    import huggingface_hub as hh

    if getattr(hh.snapshot_download, "_studio3d", False):
        return
    original = hh.snapshot_download
    log = logging.getLogger("studio3d.download")

    def snapshot_download(*args, **kwargs):
        ignore = kwargs.get("ignore_patterns") or []
        if isinstance(ignore, str):
            ignore = [ignore]
        kwargs["ignore_patterns"] = list(ignore) + SKIP_PATTERNS
        if is_local_only():
            kwargs["local_files_only"] = True
        if kwargs.get("local_files_only"):
            return original(*args, **kwargs)
        deadline = time.time() + DOWNLOAD_RETRY_MINUTES * 60
        attempt = 0
        while True:
            try:
                return original(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if not is_network_error(e):
                    raise
                if time.time() >= deadline:
                    print("❌ انقطع الإنترنت — توقف التنزيل. أعد المحاولة لاحقاً وسيكمل من حيث توقف.", flush=True)
                    raise
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                attempt += 1
                msg = (f"انقطع الاتصال أثناء التنزيل — إعادة المحاولة رقم {attempt} بعد {delay} ثانية "
                       f"(يكمل من حيث توقف)")
                log.warning("%s: %s", msg, str(e)[:160])
                print(msg, flush=True)
                time.sleep(delay)

    snapshot_download._studio3d = True
    hh.snapshot_download = snapshot_download


def remove_ckpt_duplicates() -> float:
    """Delete already-downloaded duplicates we never load (see SKIP_PATTERNS). Returns freed GB."""
    freed = 0
    hub = Path(os.environ["HF_HOME"]) / "hub"
    if not hub.exists():
        return 0.0
    dups = [ck for ck in hub.glob("models--*/snapshots/*/**/*.ckpt")
            if ck.with_suffix(".safetensors").exists()]
    dups += [st for st in hub.glob("models--*/snapshots/*/hunyuan3d-paint-v2-0*/unet/*.safetensors")
             if st.with_suffix(".bin").exists()]
    dups += [st for st in hub.glob("models--*/snapshots/*/hunyuan3d-*/model.safetensors")
             if (st.parent / "model.fp16.safetensors").exists()]
    for ck in dups:
        target = ck.resolve()
        try:
            size = target.stat().st_size
            ck.unlink()
            if target != ck and target.exists():
                target.unlink()
            freed += size
        except OSError:
            pass
    return round(freed / 1024 ** 3, 2)
