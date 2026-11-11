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


class ModelManager:
    """Keeps at most ONE heavy model on the GPU. Switching unloads the previous one."""

    def __init__(self):
        self._key = None
        self._obj = None
        self._rembg = None
        self.lock = threading.Lock()

    def _free(self):
        if self._obj is not None:
            log.info("unloading %s", self._key)
        self._obj = None
        self._key = None
        gc.collect()
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self):
        self._free()

    def get(self, key: tuple, loader):
        if self._key == key and self._obj is not None:
            return self._obj
        self._free()
        log.info("loading %s", key)
        t = time.time()
        self._obj = loader()
        self._key = key
        log.info("loaded %s in %.1fs", key, time.time() - t)
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
def _load_shape(kind: str, variant: str):
    torch = _torch()
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    spec = config.SHAPE_MODELS[kind]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        spec["repo"], subfolder=spec[variant], use_safetensors=True, device=device,
    )
    # FlashVDM: much faster & lighter mesh decoding.
    pipe.enable_flashvdm(mc_algo="mc")
    return pipe


def _load_texture():
    torch = _torch()
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    pipe = Hunyuan3DPaintPipeline.from_pretrained(config.TEXTURE_MODEL["repo"],
                                                  subfolder=config.TEXTURE_MODEL["subfolder"])
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()  # needed on 8 GB cards
    return pipe


def _load_t2i():
    torch = _torch()
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        config.TEXT2IMAGE_MODEL, torch_dtype=torch.float16,
        enable_pag=True, pag_applied_layers=["blocks.(16|17|18|19)"],
    )
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


def text_to_image(prompt: str, seed: int) -> Image.Image:
    torch = _torch()
    pipe = MANAGER.get(("t2i",), _load_t2i)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pos = ",白色背景,3D风格,最佳质量"
    neg = ("文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态,残缺,多余的手指,"
           "变异,畸形,模糊,糟糕的解剖学,糟糕的比例,多余的肢体")
    img = pipe(prompt=prompt[:200] + pos, negative_prompt=neg, num_inference_steps=25,
               pag_scale=1.3, width=1024, height=1024, generator=gen, return_dict=False)[0][0]
    return img


def texture_mesh(mesh, image: Image.Image, max_faces: int = 40000):
    """Paint an AI texture on a mesh using a reference image (front view)."""
    from . import mesh_tools

    mesh = mesh_tools.reduce_faces(mesh, max_faces)  # texturing needs a light mesh
    pipe = MANAGER.get(("texture",), _load_texture)
    return pipe(mesh, image=image)
