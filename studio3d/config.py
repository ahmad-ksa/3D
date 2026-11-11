"""Paths, model ids and quality presets."""
import os
import sys
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
