"""AI generation engine: loads one model at a time so everything fits in 8 GB VRAM."""
from __future__ import annotations

import gc
import json
import logging
import random
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import config

log = logging.getLogger("studio3d.engine")

# Our pure-PyTorch rasteriser replaces the compiled CUDA extension (see vendor/).
if str(config.VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(config.VENDOR_DIR))

config.patch_hf_downloads()


# ---------------------------------------------------------------------------
# Cancelling a running generation (the ⏹ button)
# ---------------------------------------------------------------------------
class Cancelled(Exception):
    pass


CANCEL = threading.Event()


def request_cancel():
    CANCEL.set()


def reset_cancel():
    CANCEL.clear()


def check_cancel():
    if CANCEL.is_set():
        raise Cancelled("تم إيقاف التوليد")


_STEP_CB = threading.local()


def set_step_callback(fn):
    """fn(desc, done, total) is called on every diffusion / decoding step of this thread."""
    _STEP_CB.fn = fn


def _step(bar, done):
    fn = getattr(_STEP_CB, "fn", None)
    if fn is None:
        return
    total = getattr(bar, "total", None) or 0
    try:
        fn(str(getattr(bar, "desc", "") or ""), int(done), int(total))
    except Exception:  # noqa: BLE001
        pass


class _CancellableBar:
    """Wraps a tqdm bar: every step checks the stop button (works for loops and .update())
    and reports fine-grained progress."""

    def __init__(self, bar):
        self._bar = bar
        self._done = 0

    def __iter__(self):
        for x in self._bar:
            check_cancel()
            self._done += 1
            _step(self._bar, self._done)
            yield x

    def update(self, n=1, *a, **k):
        check_cancel()
        self._done += n or 0
        _step(self._bar, self._done)
        return self._bar.update(n, *a, **k)

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *exc):
        return self._bar.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._bar, name)


def _patch_progress_bars():
    """Make the diffusion / mesh-decoding loops of Hunyuan3D and diffusers stoppable."""
    import importlib
    for mod_name in ("hy3dgen.shapegen.pipelines", "hy3dgen.shapegen.models.autoencoders.volume_decoders",
                     "diffusers.pipelines.pipeline_utils"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            continue
        orig = getattr(mod, "tqdm", None)
        if orig is None or getattr(orig, "_studio3d", False):
            continue

        def make(*a, _orig=orig, **k):
            return _CancellableBar(_orig(*a, **k))

        make._studio3d = True
        mod.tqdm = make

VIEWS = ("front", "back", "left", "right")


def _torch():
    import torch
    return torch


def device_info() -> dict:
    torch = _torch()
    if not torch.cuda.is_available():
        return {"cuda": False, "name": "CPU", "vram_gb": 0}
    p = torch.cuda.get_device_properties(0)
    return {"cuda": True, "name": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 1)}


# ---------------------------------------------------------------------------
# What is already downloaded? (never start a multi-GB download mid-generation)
# ---------------------------------------------------------------------------
def _local_snapshot(repo: str, patterns: list):
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=repo, allow_patterns=patterns, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def _has_weights(d: Path) -> bool:
    return d.is_dir() and any(f.suffix in (".safetensors", ".bin") for f in d.rglob("*") if f.is_file())


def _diffusers_ready(root: Path) -> bool:
    """A diffusers pipeline folder is complete when every component in model_index.json has weights."""
    idx = root / "model_index.json"
    if not idx.exists():
        return False
    for name, val in json.loads(idx.read_text(encoding="utf-8")).items():
        comp = root / name
        if isinstance(val, list) and comp.is_dir() and (comp / "config.json").exists():
            if not _has_weights(comp):
                return False
    return True


def shape_ready(kind: str, variant: str) -> bool:
    spec = config.SHAPE_MODELS[kind]
    snap = _local_snapshot(spec["repo"], [f"{spec[variant]}/*"])
    return bool(snap) and (snap / spec[variant] / "config.yaml").exists() and \
        any((snap / spec[variant]).glob("*.safetensors"))


def texture_ready() -> bool:
    repo, sub = config.TEXTURE_MODEL["repo"], config.TEXTURE_MODEL["subfolder"]
    snap = _local_snapshot(repo, ["hunyuan3d-delight-v2-0/*", f"{sub}/*"])
    if not snap:
        return False
    # hy3dgen loads the paint UNet from exactly this file
    paint_unet = snap / sub / "unet" / "diffusion_pytorch_model.bin"
    return _diffusers_ready(snap / "hunyuan3d-delight-v2-0") and paint_unet.exists() and \
        _has_weights(snap / sub / "vae") and _has_weights(snap / sub / "text_encoder")


def text_ready() -> bool:
    snap = _local_snapshot(config.TEXT2IMAGE_MODEL, ["*"])
    return bool(snap) and _diffusers_ready(snap)


def pose_ready() -> bool:
    snap = _local_snapshot(config.POSE_CONTROLNET_MODEL, ["config.json", "*.safetensors"])
    return bool(snap) and (snap / "config.json").exists() and \
        (snap / "diffusion_pytorch_model.safetensors").exists()


def _ensure_downloaded(repo: str, ready, label: str, size: str, patterns=None):
    """Download a model BEFORE loading it (with the 2-minute retry), telling the user why it waits."""
    if ready():
        return
    msg = f"⬇️ تنزيل {label} لأول مرة (~{size}) — مرة وحدة بس…"
    log.info(msg)
    fn = getattr(_STEP_CB, "fn", None)
    if fn is not None:
        try:
            fn(msg, 0, 0)
        except Exception:  # noqa: BLE001
            pass
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, allow_patterns=patterns)
    if not ready():
        raise RuntimeError(f"لم يكتمل تنزيل {label}. تأكد من الإنترنت وأعد المحاولة (يكمل من حيث توقف).")


def models_status() -> dict:
    out = {}
    for kind, label in (("multiview", "من 4 صور"), ("single", "من صورة واحدة")):
        for variant, q in (("turbo", "سريع/متوازن"), ("full", "عالي الجودة")):
            out[f"{label} — {q}"] = shape_ready(kind, variant)
    out["التلوين"] = texture_ready()
    out["من نص"] = text_ready()
    return out


def vram_text() -> str:
    """Free / total GPU memory, for the log (overflowing 8 GB makes everything 5-20x slower)."""
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return "cpu"
        free, total = torch.cuda.mem_get_info()
        used_here = torch.cuda.memory_reserved() / 1024 ** 3
        return f"VRAM free {free / 1024 ** 3:.1f}/{total / 1024 ** 3:.1f} GB (this process {used_here:.1f} GB)"
    except Exception:  # noqa: BLE001
        return "?"


# ---------------------------------------------------------------------------
# Sharing the GPU between the desktop app and Claude (two separate processes)
# ---------------------------------------------------------------------------
# Each process keeps its last model in VRAM to reuse it. When the OTHER process starts a job it
# writes a release request; the idle owner unloads within a second, so two sets of models never
# sit in 8 GB together (that overflow into shared memory was the main cause of slow generations).
_OWNER = config.DATA_ROOT / "gpu_owner.json"
_REQUEST = config.DATA_ROOT / "gpu_release.request"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:  # noqa: BLE001
        return False


def _read_pid(path: Path):
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["pid"])
    except Exception:  # noqa: BLE001
        return None


def _write_pid(path: Path):
    import os
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    except OSError:
        pass


def claim_gpu(timeout: float = 30.0):
    """Call while holding the GPU file lock, before loading models: asks another Studio3D process
    that still holds models in VRAM to free them, and waits (max `timeout` s) until it did."""
    import os
    owner = _read_pid(_OWNER)
    if owner is None or owner == os.getpid() or not _pid_alive(owner):
        return
    log.info("asking process %s to free the GPU", owner)
    _write_pid(_REQUEST)
    t = time.time()
    while time.time() - t < timeout:
        cur = _read_pid(_OWNER)
        if cur is None or cur != owner or not _pid_alive(owner):
            log.info("GPU freed by %s after %.1fs", owner, time.time() - t)
            break
        time.sleep(0.3)
    try:
        _REQUEST.unlink()
    except OSError:
        pass


class ModelManager:
    """Keeps at most ONE heavy model on the GPU. Switching unloads the previous one."""

    def __init__(self):
        self._key = None
        self._obj = None
        self._rembg = None
        self.lock = threading.Lock()
        self._watcher = None

    def _free(self):
        import os
        if self._obj is not None:
            log.info("unloading %s", self._key)
        self._obj = None
        self._key = None
        gc.collect()
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _read_pid(_OWNER) == os.getpid():
            try:
                _OWNER.unlink()
            except OSError:
                pass

    def _watch(self):
        """Background: free our models when the other process asks and we are idle."""
        import os
        while True:
            time.sleep(1.0)
            try:
                req = _read_pid(_REQUEST)
                if req is None or req == os.getpid() or self._obj is None:
                    continue
                if self.lock.acquire(blocking=False):
                    try:
                        log.info("process %s needs the GPU — freeing ours", req)
                        self._free()
                    finally:
                        self.lock.release()
            except Exception:  # noqa: BLE001
                pass

    def _start_watcher(self):
        if self._watcher is None:
            self._watcher = threading.Thread(target=self._watch, name="gpu-release", daemon=True)
            self._watcher.start()

    def unload(self):
        self._free()

    def get(self, key: tuple, loader):
        if self._key == key and self._obj is not None:
            return self._obj
        self._free()
        log.info("loading %s (%s)", key, vram_text())
        t = time.time()
        self._obj = loader()
        self._key = key
        _write_pid(_OWNER)
        self._start_watcher()
        log.info("loaded %s in %.1fs (%s)", key, time.time() - t, vram_text())
        return self._obj

    # -- background removal (small, runs on CPU via onnxruntime) --------------
    def rembg(self, image: Image.Image) -> Image.Image:
        if self._rembg is None:
            from hy3dgen.rembg import BackgroundRemover
            self._rembg = BackgroundRemover()
        return self._rembg(image.convert("RGB"))


MANAGER = ModelManager()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
_INIT_FNS = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
             "xavier_uniform_", "xavier_normal_", "constant_", "zeros_", "ones_", "orthogonal_")


@__import__("contextlib").contextmanager
def no_random_init():
    """Build modules WITHOUT random initial weights. Only for parts whose every weight is then
    loaded from the checkpoint (strict load): random init of DINOv2-giant + DiT is ~15-25 s of CPU
    work that is thrown away right after."""
    import torch.nn.init as init
    saved = {n: getattr(init, n) for n in _INIT_FNS if hasattr(init, n)}
    for n in saved:
        setattr(init, n, lambda t, *a, **k: t)
    try:
        from transformers.modeling_utils import no_init_weights
        ctx = no_init_weights()
    except Exception:  # noqa: BLE001
        ctx = __import__("contextlib").nullcontext()
    try:
        with ctx:
            yield
    finally:
        for n, f in saved.items():
            setattr(init, n, f)


def _checkpoint_parts(kind: str, variant: str) -> set:
    """Top-level parts stored in the shape checkpoint (model / vae / conditioner)."""
    from hy3dgen.shapegen.utils import smart_load_model
    spec = config.SHAPE_MODELS[kind]
    try:
        _cfg, ckpt = smart_load_model(spec["repo"], spec[variant], use_safetensors=True, variant="fp16")
        from safetensors import safe_open
        with safe_open(ckpt, framework="pt") as f:
            return {k.split(".", 1)[0] for k in f.keys()}
    except Exception:  # noqa: BLE001
        return set()


def _load_shape(kind: str, variant: str):
    torch = _torch()
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    _patch_progress_bars()
    spec = config.SHAPE_MODELS[kind]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with config.local_only(shape_ready(kind, variant)):
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            spec["repo"], subfolder=spec[variant], use_safetensors=True, device=device,
        )
    # FlashVDM: much faster & lighter mesh decoding.
    pipe.enable_flashvdm(mc_algo="mc")
    if kind == "multiview":
        # one shared scale for all views (T-pose characters keep their proportions)
        from .mv_preprocess import ConsistentMVImageProcessor
        old = pipe.image_processor
        pipe.image_processor = ConsistentMVImageProcessor(getattr(old, "size", 512),
                                                          getattr(old, "border_ratio", None))
    return pipe


def _load_texture():
    torch = _torch()
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    _patch_progress_bars()
    orig_load = torch.load

    def mmap_load(f, *a, **k):
        # read the big .bin straight from disk pages instead of copying it all into RAM first
        if isinstance(f, (str, Path)) and str(f).endswith(".bin") and "mmap" not in k:
            try:
                return orig_load(f, *a, mmap=True, **k)
            except Exception:  # noqa: BLE001  (old file format: fall back to a normal read)
                pass
        return orig_load(f, *a, **k)

    torch.load = mmap_load
    try:
        with config.local_only(texture_ready()):
            pipe = Hunyuan3DPaintPipeline.from_pretrained(config.TEXTURE_MODEL["repo"],
                                                          subfolder=config.TEXTURE_MODEL["subfolder"])
    finally:
        torch.load = orig_load
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()  # needed on 8 GB cards
    return pipe


def _load_t2i():
    torch = _torch()
    from diffusers import AutoPipelineForText2Image
    _patch_progress_bars()

    _ensure_downloaded(config.TEXT2IMAGE_MODEL, text_ready, "نموذج النص ← صورة", "10 GB")
    pipe = AutoPipelineForText2Image.from_pretrained(
        config.TEXT2IMAGE_MODEL, torch_dtype=torch.float16, local_files_only=True,
        enable_pag=True, pag_applied_layers=["blocks.(16|17|18|19)"],
    )
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    return pipe


def _load_t2i_pose():
    torch = _torch()
    from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
    _patch_progress_bars()
    _ensure_downloaded(config.TEXT2IMAGE_MODEL, text_ready, "نموذج النص ← صورة", "10 GB")
    _ensure_downloaded(config.POSE_CONTROLNET_MODEL, pose_ready, "نموذج الوضعيات", "3 GB",
                       ["config.json", "*.safetensors"])
    cn = HunyuanDiT2DControlNetModel.from_pretrained(config.POSE_CONTROLNET_MODEL, torch_dtype=torch.float16,
                                                     local_files_only=True)
    pipe = HunyuanDiTControlNetPipeline.from_pretrained(
        config.TEXT2IMAGE_MODEL, controlnet=cn, torch_dtype=torch.float16, local_files_only=True)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    return pipe


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def new_job_dir(kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = config.OUTPUT_DIR / f"{stamp}_{kind}_{uuid.uuid4().hex[:4]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_meta(job_dir: Path, meta: dict):
    (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    try:  # thumbnail for the library
        from . import library
        (job_dir / "thumb.png").unlink(missing_ok=True)
        library.ensure_thumb(job_dir)
    except Exception:  # noqa: BLE001
        log.warning("thumbnail failed", exc_info=True)


def prepare_image(img: Image.Image, remove_bg: bool) -> Image.Image:
    if img is None:
        return None
    if remove_bg or img.mode != "RGBA":
        img = MANAGER.rembg(img)
    return img


def resolve_seed(seed, randomize: bool) -> int:
    if randomize or seed is None or int(seed) < 0:
        return random.randint(0, 2 ** 31 - 1)
    return int(seed)


def generate_shape(image, kind: str, preset: str, seed: int, octree: int | None = None,
                   steps: int | None = None):
    """image: PIL image (single) or dict view->PIL (multiview). Returns trimesh."""
    torch = _torch()
    from hy3dgen.shapegen.pipelines import export_to_trimesh

    p = config.PRESETS[preset]
    pipe = MANAGER.get(("shape", kind, p["variant"]), lambda: _load_shape(kind, p["variant"]))
    gen = torch.Generator().manual_seed(int(seed))
    out = pipe(
        image=image,
        num_inference_steps=int(steps or p["steps"]),
        guidance_scale=p["guidance"],
        generator=gen,
        octree_resolution=int(octree or p["octree"]),
        num_chunks=config.NUM_CHUNKS,
        output_type="mesh",
    )
    mesh = export_to_trimesh(out)[0]
    if mesh is None:
        raise RuntimeError("فشل استخراج المجسم — جرّب صوراً أوضح أو بذرة (Seed) مختلفة.")
    return mesh


def text_to_image(prompt: str, seed: int, pose: str = "free") -> Image.Image:
    """pose: 'free' | 't' | 'a'. With t/a the image model follows a T/A-pose skeleton
    (pose ControlNet), so the character comes out ready for the auto skeleton."""
    torch = _torch()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pos = ",白色背景,3D风格,最佳质量"
    neg = ("文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态,残缺,多余的手指,"
           "变异,畸形,模糊,糟糕的解剖学,糟糕的比例,多余的肢体")
    if pose in ("t", "a"):
        from . import poses
        pipe = MANAGER.get(("t2i_pose",), _load_t2i_pose)
        # pose words first: the CLIP text encoder only reads the first 77 tokens
        img = pipe(prompt=poses.PROMPT_HINT[pose].lstrip("，") + "，" + prompt[:180] + pos,
                   negative_prompt=neg + ",手臂下垂" + poses.NEGATIVE_HINT,
                   control_image=poses.skeleton_image(pose, 1024), controlnet_conditioning_scale=1.0,
                   num_inference_steps=25, width=1024, height=1024, generator=gen,
                   return_dict=False)[0][0]
        return img
    pipe = MANAGER.get(("t2i",), _load_t2i)
    img = pipe(prompt=prompt[:200] + pos, negative_prompt=neg, num_inference_steps=25,
               pag_scale=1.3, width=1024, height=1024, generator=gen, return_dict=False)[0][0]
    return img


def texture_mesh(mesh, image: Image.Image, max_faces: int = 40000):
    """Paint an AI texture on a mesh using a reference image (front view)."""
    from . import mesh_tools

    mesh = mesh_tools.reduce_faces(mesh, max_faces)  # texturing needs a light mesh
    pipe = MANAGER.get(("texture",), _load_texture)
    return pipe(mesh, image=image)
