"""Video worker: runs in its OWN process (python -m studio3d.video.worker args.json).

Why a separate process:
  * the video model needs a newer diffusers than the 3D models -> loaded from config.VIDEO_LIBS first;
  * every step frees ALL its memory when the process ends (8 GB GPU / 16 GB RAM laptops).

Commands (args["cmd"]), each for ALL scenes of a video in one process (args["jobs"]; models load once):
  i2v      image + description -> short silent clip (Wan2.2 TI2V-5B, FastWan 4-step version)
  lipsync  clip + voice -> clip whose lips follow the voice (LatentSync 1.5)
stdout: "@@P <0..1> <text>" progress of the whole batch, "@@S <json>" a finished scene.
Result json -> args["result"] ({"ok", "items": [...]}).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent


def _setup_paths():
    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
    from studio3d import config
    libs = config.VIDEO_LIBS_DIR
    if libs.exists() and str(libs) not in sys.path:
        sys.path.insert(0, str(libs))          # newer diffusers (Wan2.2) wins over the 3D one
    vendor = config.PACKAGE_DIR / "vendor_video"
    if str(vendor) not in sys.path:
        sys.path.insert(1, str(vendor))         # LatentSync code
    config.patch_hf_downloads()
    return config


def progress(frac: float, text: str = ""):
    print(f"@@P {max(0.0, min(1.0, frac)):.3f} {text}", flush=True)


def _download(config, repo: str, patterns=None, label=""):
    from huggingface_hub import snapshot_download
    try:
        with config.local_only(True):
            p = snapshot_download(repo_id=repo, allow_patterns=patterns)
        if _complete(Path(p), patterns):
            return Path(p)
    except Exception:  # noqa: BLE001
        pass
    progress(0.0, f"⬇️ تنزيل {label} لأول مرة (مرة وحدة بس)…")
    import threading
    stop = threading.Event()
    threading.Thread(target=_download_progress, args=(config, repo, patterns, label, stop), daemon=True).start()
    try:
        return Path(snapshot_download(repo_id=repo, allow_patterns=patterns))
    finally:
        stop.set()


def _download_progress(config, repo, patterns, label, stop):
    """Show "x / y GB" while a model downloads (the hub itself reports nothing to the app)."""
    import fnmatch
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum((f.size or 0) for f in info.siblings
                    if not patterns or any(fnmatch.fnmatch(f.rfilename, p) for p in patterns))
    except Exception:  # noqa: BLE001
        total = 0
    folder = Path(config.MODELS_DIR) / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))
    t0, last = time.time(), None
    while not stop.wait(3.0):
        try:
            # finished files live in snapshots/ (on Windows as copies, the blob may stay too: don't
            # count it twice); files still coming are blobs/*.incomplete
            done = sum(f.stat().st_size for f in (folder / "snapshots").rglob("*")
                       if f.is_file() and not f.is_symlink())
            done += sum(f.stat().st_size for f in (folder / "blobs").glob("*.incomplete"))
        except OSError:
            continue
        el = max(time.time() - t0, 1)
        if last is None:
            last = (done, el)
        speed = max(done - last[0], 0) / max(el - last[1], 1) if el > last[1] + 1 else 0
        gb = done / 1024 ** 3
        parts = sum(1 for _ in folder.rglob("*.incomplete"))
        left_files = f" — {parts} ملفات كبيرة لسه تنزل" if parts else ""
        if total and done < total * 0.97:
            left = max(total - done, 0) / speed / 60 if speed > 0 else 0
            eta = f" — باقي تقريباً {left:.0f} دقيقة" if speed > 0 else ""
            progress(min(done / total, 0.97) * 0.15,
                     f"⬇️ تنزيل {label}: {gb:.1f} / {total / 1024 ** 3:.1f} GB{eta}")
        else:   # the hub's size list can be incomplete: just show what is done and what is left
            progress(0.14, f"⬇️ تنزيل {label}: {gb:.1f} GB تم{left_files}…")


def _complete(root: Path, patterns) -> bool:
    if not root.exists():
        return False
    if patterns:
        import fnmatch
        files = [f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()]
        return all(any(fnmatch.fnmatch(f, pat) for f in files) for pat in patterns)
    idx = root / "model_index.json"
    if not idx.exists():
        return False
    for name, val in json.loads(idx.read_text(encoding="utf-8")).items():
        comp = root / name
        if isinstance(val, list) and val[0] and comp.is_dir() and (comp / "config.json").exists():
            if not any(f.suffix in (".safetensors", ".bin") for f in comp.rglob("*")):
                return False
    return True


# ---------------------------------------------------------------------------
# loading big models with little RAM
# ---------------------------------------------------------------------------
FP8 = "float8_e4m3fn"


def _log_mem(where: str):
    """RAM / VRAM in the log (a crash while loading is then easy to explain)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss / 1024 ** 3
        msg = f"[mem] {where}: worker RAM {rss:.1f} GB, free RAM {vm.available / 1024 ** 3:.1f} GB"
    except Exception:  # noqa: BLE001
        msg = f"[mem] {where}"
    try:
        import torch
        if torch.cuda.is_available():
            msg += f", VRAM {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB"
    except Exception:  # noqa: BLE001
        pass
    print(msg, flush=True)


def _weight_files(folder: Path) -> list:
    """The safetensors files of one model folder (sharded or not, no duplicate variants)."""
    idx = sorted(folder.glob("*.safetensors.index.json"))
    if idx:
        names = sorted(set(json.loads(idx[0].read_text(encoding="utf-8"))["weight_map"].values()))
        return [folder / n for n in names]
    files = sorted(f for f in folder.glob("*.safetensors") if f.name.count(".") == 1)
    if not files:
        raise FileNotFoundError(f"لا توجد أوزان (.safetensors) في {folder}")
    return files


def _empty_model(make):
    """Build a model on the 'meta' device: structure only, no memory for the weights yet."""
    from accelerate import init_empty_weights
    with init_empty_weights():
        return make()


def _set_param_dtypes(model, dtype, keep_fp32=()):
    """Choose the dtype every (still empty) weight will be stored in."""
    import torch
    for name, p in list(model.named_parameters()):
        want = torch.float32 if any(k in name.split(".") for k in keep_fp32) else dtype
        mod_name, _, pname = name.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model
        mod._parameters[pname] = torch.nn.Parameter(torch.empty_like(p, dtype=want, device="meta"),
                                                    requires_grad=False)


def _load_weights(model, folder: Path, on_progress=None):
    """Fill a model built with _empty_model tensor by tensor, straight from the safetensors files.
    Every tensor is converted at once to the dtype its empty weight has (8-bit for the big layers),
    so RAM holds the finished model plus ONE tensor — never the whole 16/32-bit checkpoint
    (that copy, next to a text encoder still in RAM, is the likely cause of the silent crash)."""
    import torch
    from safetensors import safe_open
    files = _weight_files(folder)
    total = sum(f.stat().st_size for f in files) or 1
    done, last = 0, -1.0
    unexpected = []
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                t = sf.get_tensor(key)
                done += t.numel() * t.element_size()
                mod_name, _, name = key.rpartition(".")
                try:
                    mod = model.get_submodule(mod_name) if mod_name else model
                except AttributeError:
                    unexpected.append(key)
                    continue
                if name in mod._parameters and mod._parameters[name] is not None:
                    old = mod._parameters[name]
                    if tuple(old.shape) != tuple(t.shape):
                        raise RuntimeError(f"شكل الوزن {key} لا يطابق النموذج: {tuple(t.shape)} ≠ {tuple(old.shape)}")
                    if str(old.dtype).endswith(FP8):
                        t = t.clamp(-448, 448)       # largest float8_e4m3fn value (no NaN on overflow)
                    mod._parameters[name] = torch.nn.Parameter(t.to(old.dtype), requires_grad=False)
                elif name in mod._buffers and mod._buffers[name] is not None:
                    mod._buffers[name] = t.to(mod._buffers[name].dtype)
                else:
                    unexpected.append(key)
                del t
                if on_progress and done / total - last >= 0.02:
                    last = done / total
                    on_progress(last)
    missing = [n for n, p in model.named_parameters() if p.is_meta]
    if missing:
        raise RuntimeError(f"ملفات النموذج ناقصة ({len(missing)} وزن مفقود، مثل {missing[0]}). "
                           f"احذف المجلد {folder.parent.name} من النماذج وأعد التنزيل.")
    if unexpected:
        print(f"ignored {len(unexpected)} unused weights, e.g. {unexpected[:3]}", flush=True)
    return model


def _gpu() -> bool:
    import torch
    return torch.cuda.is_available()


def _stream_from_ram(module):
    """Keep the weights in RAM and stream them to the GPU block by block (prefetching the next block
    while one computes). Pinned copies are made per block on the fly (low_cpu_mem_usage): pinning the
    whole 5 GB up front doubled the RAM for a moment and locked 5 GB of the 16."""
    import torch
    from diffusers.hooks import apply_group_offloading
    apply_group_offloading(module, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                           offload_type="block_level", num_blocks_per_group=1, use_stream=True,
                           low_cpu_mem_usage=True)


# ---------------------------------------------------------------------------
# image -> video (Wan2.2 TI2V-5B)
# ---------------------------------------------------------------------------
def _load_text_encoder(src: Path, on_progress=None):
    """UMT5-XXL (5.7 B weights, 11 GB in bf16). The big layers are stored in 8-bit (6.7 GB in RAM
    instead of 11) and, with a GPU, run there block by block (seconds instead of minutes on the CPU)."""
    import torch
    from diffusers.hooks.layerwise_casting import apply_layerwise_casting_hook
    from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(str(src), subfolder="tokenizer")
    folder = src / "text_encoder"
    te = _empty_model(lambda: UMT5EncoderModel(AutoConfig.from_pretrained(str(folder))))
    _set_param_dtypes(te, torch.bfloat16)
    # whole attention / feed-forward modules (not single Linear layers): the T5 feed-forward reads
    # wo.weight.dtype BEFORE calling wo and would turn its input into 8-bit
    for name, m in te.named_modules():
        if name.rsplit(".", 1)[-1] in ("SelfAttention", "DenseReluDense"):
            apply_layerwise_casting_hook(m, getattr(torch, FP8), torch.bfloat16, non_blocking=False)
    _load_weights(te, folder, on_progress)
    te.eval()
    if _gpu():
        _stream_from_ram(te.encoder)
    return tok, te


def _encode_prompts(tok, te, prompts: list) -> dict:
    import torch
    from diffusers import WanImageToVideoPipeline

    class _Enc:  # only what _get_t5_prompt_embeds reads
        tokenizer = tok
        text_encoder = te
        _execution_device = torch.device("cpu")

    out = {}
    with torch.no_grad():
        for p in prompts:
            out[p] = WanImageToVideoPipeline._get_t5_prompt_embeds(_Enc(), p, 1, 512, torch.device("cpu"),
                                                                   torch.bfloat16)
    return out


def _prompt_embeds(config, snap: Path, prompts: list, frac=(0.03, 0.08)) -> dict:
    """Text encoder runs once per new text, then is freed; results are cached on disk so the same
    description never has to be encoded again. All texts of a video are encoded in one go."""
    import gc

    import torch
    cache = config.MODELS_DIR / "video" / "prompt_cache"
    cache.mkdir(parents=True, exist_ok=True)

    def path(p):
        return cache / (hashlib.sha1((str(snap.name) + "|" + p).encode("utf-8")).hexdigest()[:20] + ".pt")

    todo = [p for p in dict.fromkeys(prompts) if not path(p).exists()]
    if todo:
        a, b = frac
        progress(a, f"فهم الوصف ({len(todo)} وصف جديد، مرة وحدة لكل وصف)…")
        src = snap
        if not (snap / "text_encoder").is_dir() or not (snap / "tokenizer").is_dir():
            # the distilled repo may ship without the (unchanged) text encoder: take Wan's own
            src = _download(config, config.VIDEO_TEXT_MODEL, ["text_encoder/*", "tokenizer/*"], "مشفّر النص")
        tok, te = _load_text_encoder(src, lambda f: progress(a + (b - a) * 0.8 * f, "تحميل مشفّر النص…"))
        _log_mem("text encoder loaded")
        for p, emb in _encode_prompts(tok, te, todo).items():
            torch.save(emb, path(p))
        # The encoder must be really gone before the 5B video model loads: a reference kept by a
        # helper class (freed only by a later garbage collection) left 11 GB in RAM -> crash.
        del tok, te
        gc.collect()
        if _gpu():
            torch.cuda.empty_cache()
        _log_mem("text encoder freed")
    return {p: torch.load(path(p), map_location="cpu", weights_only=True) for p in prompts}


def _load_i2v(snap: Path, on_progress=None):
    """Wan2.2 TI2V-5B for an 8 GB card and a 16 GB PC:
    * transformer weights read tensor by tensor and stored in 8-bit right away (~5 GB of RAM; the
      bf16 checkpoint is 10 GB), computed in bf16, streamed to the GPU block by block;
    * VAE in bf16 (1.4 GB), on the GPU only while encoding the start image / decoding the frames."""
    import torch
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanImageToVideoPipeline,
        WanTransformer3DModel,
    )
    gpu = _gpu()
    vae = AutoencoderKLWan.from_pretrained(str(snap), subfolder="vae",
                                           torch_dtype=torch.bfloat16 if gpu else torch.float32)
    vae.eval()
    folder = snap / "transformer"
    tr = _empty_model(lambda: WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(str(folder))))
    _set_param_dtypes(tr, torch.bfloat16, tr._keep_in_fp32_modules or ())
    # order matters: 8-bit storage first, then block streaming (the streamed copies are the 8-bit ones)
    tr.enable_layerwise_casting(storage_dtype=getattr(torch, FP8), compute_dtype=torch.bfloat16)
    _load_weights(tr, folder, on_progress)
    tr.eval()
    if gpu:
        _stream_from_ram(tr)
    sched_cfg = json.loads((snap / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    # 4-step FastWan: UniPC on the flow-matching sigmas, shift 8 (whatever scheduler the repo names)
    scheduler = UniPCMultistepScheduler.from_config(sched_cfg, prediction_type="flow_prediction",
                                                    use_flow_sigmas=True, flow_shift=8.0)
    index = json.loads((snap / "model_index.json").read_text(encoding="utf-8"))
    # TI2V-5B conditions on the image through the latents (expand_timesteps), not an image encoder
    expand = index.get("expand_timesteps")
    if expand is None:
        expand = tr.config.in_channels == vae.config.z_dim
    pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler, transformer=tr,
                                   image_encoder=None, image_processor=None, expand_timesteps=bool(expand))
    _vae_on_gpu_only_while_used(pipe.vae)
    _maybe_tiling(pipe.vae)
    return pipe


class _OnGpu:
    """with _OnGpu(vae): ... -> the module visits the GPU and goes back to RAM afterwards."""

    def __init__(self, module):
        self.m = module

    def __enter__(self):
        if _gpu():
            self.m.to("cuda")
        return self.m

    def __exit__(self, *exc):
        if _gpu():
            import torch
            self.m.to("cpu")
            torch.cuda.empty_cache()


def _vae_on_gpu_only_while_used(vae):
    """The pipeline calls vae.encode for the start image: the VAE comes to the GPU only for that."""
    orig = vae.encode

    def encode(x, *a, **k):
        with _OnGpu(vae):
            return orig(x.to(vae.device, vae.dtype), *a, **k)

    vae.encode = encode


def _maybe_tiling(vae) -> bool:
    """Tiled VAE saves memory, but in diffusers 0.35 the tiled path skips the patchify step of the
    Wan2.2 VAE (patch_size=2) -> "expected input ... to have 12 channels, but got 3". Only tile VAEs
    without patch_size (Wan2.1 style); the Wan2.2 VAE decodes frame by frame anyway."""
    if getattr(vae.config, "patch_size", None):
        return False
    try:
        vae.enable_tiling()
        return True
    except Exception:  # noqa: BLE001
        return False


def _open_rgb(path, size):
    """Start image at the video size; transparent PNGs go on white (not on black / garbage)."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB").resize(size, Image.LANCZOS)


def run_i2v(pipe, a: dict, emb, on_progress=lambda f, t: None) -> list:
    """One clip. on_progress(fraction 0..1 of this clip, text)."""
    import torch
    img = _open_rgb(a["image"], (a["width"], a["height"]))
    steps = int(a.get("steps", 4))
    t_start = time.time()

    def on_step(p, i, t, kw):
        mem = ""
        if _gpu():
            mem = f" (VRAM peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB)"
        el = time.time() - t_start
        print(f"step {i + 1}/{steps} after {el:.0f}s{mem}", flush=True)
        on_progress(0.05 + 0.75 * (i + 1) / steps,
                    f"توليد الحركة {i + 1}/{steps} (~{el / (i + 1):.0f} ث لكل خطوة)")
        return {}

    on_progress(0.02, f"توليد الحركة 0/{steps}")
    gen = torch.Generator(device="cpu").manual_seed(int(a.get("seed", 0)))
    dtype = pipe.transformer.dtype if pipe.transformer is not None else torch.bfloat16
    with torch.no_grad():
        latents = pipe(image=img, prompt_embeds=emb.to(dtype), height=a["height"], width=a["width"],
                       num_frames=a["frames"], num_inference_steps=steps, guidance_scale=1.0,
                       generator=gen, callback_on_step_end=on_step, output_type="latent").frames
        on_progress(0.82, "تحويل الحركة إلى صور…")
        return decode_latents(pipe, latents)


def decode_latents(pipe, latents) -> list:
    """VAE decode AFTER the 5B transformer has left the GPU (its blocks are streamed back to RAM after
    each step). Decoding inside the pipeline kept transformer and VAE on the 8 GB card together."""
    import gc

    import torch
    gc.collect()
    if _gpu():
        torch.cuda.empty_cache()
    vae = pipe.vae
    with _OnGpu(vae):
        dev = vae.device
        z = latents.to(dev, vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev, vae.dtype)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev, vae.dtype)
        z = z / inv_std + mean
        video = vae.decode(z, return_dict=False)[0].to("cpu")
        del z
    # to pictures on the CPU (a float32 copy of 121 frames would need another 0.6-1.3 GB of VRAM)
    return pipe.video_processor.postprocess_video(video.float(), output_type="pil")[0]


def _scene_label(j: dict, k: int, n: int) -> str:
    return j.get("label") or f"المشهد {k + 1}/{n}"


def cmd_i2v(a: dict, config) -> dict:
    """All scenes of a video in ONE process: the text encoder and the 5B model load once
    (they used to be reloaded for every scene: ~1-2 minutes and 10+ GB of disk reads each)."""
    jobs = a.get("jobs") or [a]
    n = len(jobs)
    snap = _download(config, config.VIDEO_MODEL, None, "نموذج الفيديو (~22 GB)")
    embs = _prompt_embeds(config, snap, [j["prompt"] for j in jobs])
    progress(0.08, "تحميل نموذج الفيديو…")
    _log_mem("before video model")
    pipe = _load_i2v(snap, lambda f: progress(0.08 + 0.07 * f, f"تحميل نموذج الفيديو {f * 100:.0f}%…"))
    _log_mem("video model loaded")
    from diffusers.utils import export_to_video
    items = []
    for k, j in enumerate(jobs):
        base, span = 0.15 + 0.85 * k / n, 0.85 / n
        label = _scene_label(j, k, n)

        def on_p(f, txt, base=base, span=span, label=label):
            progress(base + span * f, f"{label}: {txt}")

        frames = run_i2v(pipe, j, embs[j["prompt"]], on_p)
        on_p(0.95, "حفظ المقطع…")
        export_to_video(frames, j["out"], fps=int(j.get("fps", 24)))
        item = {"index": j.get("index", k), "out": j["out"], "frames": len(frames), "ok": True}
        items.append(item)
        _emit_item(item)
        del frames
        _log_mem(f"scene {k + 1}/{n} done")
    return {"items": items, "out": items[0]["out"] if n == 1 else None}


def _emit_item(item: dict):
    """A finished scene: the app shows it right away."""
    print("@@S " + json.dumps(item, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# lip sync (LatentSync 1.5)
# ---------------------------------------------------------------------------
def _load_lipsync(config):
    import torch
    from omegaconf import OmegaConf
    ls_snap = _download(config, config.LIPSYNC_MODEL, ["latentsync_unet.pt", "whisper/tiny.pt"],
                        "نموذج مزامنة الشفايف (~5 GB)")
    vae_snap = _download(config, config.LIPSYNC_VAE, ["config.json", "diffusion_pytorch_model.safetensors"],
                         "VAE")
    from studio3d.video import face_mp
    face_mp.ensure_model()

    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    gpu = _gpu()
    dev = "cuda" if gpu else "cpu"
    dtype = torch.float16 if gpu else torch.float32
    vendor = config.PACKAGE_DIR / "vendor_video"
    cfg = OmegaConf.load(vendor / "configs" / "unet" / "stage2.yaml")
    scheduler = DDIMScheduler.from_pretrained(str(vendor / "configs"))
    audio_encoder = Audio2Feature(model_path=str(ls_snap / "whisper" / "tiny.pt"), device=dev,
                                  num_frames=cfg.data.num_frames, audio_feat_length=cfg.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained(str(vae_snap), torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model),
                                                   str(ls_snap / "latentsync_unet.pt"), device="cpu", dtype=dtype)
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to(dev)
    return pipe, cfg, dtype


def _audio_seconds(path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / max(info.samplerate, 1)
    except Exception:  # noqa: BLE001
        return 5.0


def cmd_lipsync(a: dict, config) -> dict:
    """Lip sync for all talking scenes in ONE process (models load once)."""
    import math

    import torch
    jobs = a.get("jobs") or [a]
    n = len(jobs)
    progress(0.02, "تحميل نموذج مزامنة الشفايف…")
    pipe, cfg, dtype = _load_lipsync(config)
    _log_mem("lipsync model loaded")
    items = []
    cwd = os.getcwd()
    for k, j in enumerate(jobs):
        base, span = 0.1 + 0.9 * k / n, 0.9 / n
        label = _scene_label(j, k, n)
        progress(base, f"{label}: مزامنة الشفايف مع الصوت…")
        steps = int(j.get("steps", 20))
        chunks = max(1, math.ceil((_audio_seconds(j["audio"]) * 25 + 2) / cfg.data.num_frames))
        count = [0]

        def cb(step, t, latents, base=base, span=span, label=label, steps=steps, chunks=chunks,
               count=count):
            count[0] += 1
            f = min(count[0] / (steps * chunks), 1.0)
            progress(base + span * (0.15 + 0.75 * f), f"{label}: مزامنة الشفايف {f * 100:.0f}%")

        # LatentSync writes temp files and calls `ffmpeg` with plain names: work inside the scene
        # folder with short relative names (no spaces / Arabic in the shell commands)
        os.chdir(Path(j["out"]).parent)
        try:
            g = torch.Generator(device="cpu").manual_seed(int(j.get("seed", 1247)))
            pipe(video_path=Path(j["video"]).name, audio_path=Path(j["audio"]).name,
                 video_out_path=Path(j["out"]).name, num_frames=cfg.data.num_frames, num_inference_steps=steps,
                 guidance_scale=float(j.get("guidance", 1.5)), weight_dtype=dtype,
                 width=cfg.data.resolution, height=cfg.data.resolution, generator=g,
                 temp_dir="ls_temp", callback=cb, callback_steps=1)
            if not Path(j["out"]).exists():
                raise RuntimeError("فشلت مزامنة الشفايف (لم يُنتج ملف).")
            item = {"index": j.get("index", k), "out": j["out"], "ok": True}
        except RuntimeError as e:
            if "Face not detected" not in str(e):
                raise
            # no face the model can follow (cartoon / back view): this scene keeps its motion + voice
            item = {"index": j.get("index", k), "ok": False, "no_face": True,
                    "error": "ما لقيت وجه واضح في المشهد عشان أحرك الشفايف."}
        finally:
            for tmp in ("temp", "ls_temp"):     # LatentSync's scratch folders
                shutil.rmtree(Path(j["out"]).parent / tmp, ignore_errors=True)
            os.chdir(cwd)
            if _gpu():
                torch.cuda.empty_cache()
        items.append(item)
        _emit_item(item)
    out = {"items": items}
    if n == 1:
        if not items[0]["ok"]:
            raise RuntimeError("Face not detected")
        out["out"] = items[0]["out"]
    return out


COMMANDS = {"i2v": cmd_i2v, "lipsync": cmd_lipsync}


def _watch_parent(pid):
    """If the app is closed, stop at once: a worker left running keeps ~15 GB of RAM and the GPU
    busy, and the next video then crashes while loading."""
    if not pid:
        return
    import threading

    def run():
        try:
            import psutil
        except Exception:  # noqa: BLE001
            return
        while True:
            time.sleep(2)
            if not psutil.pid_exists(int(pid)):
                os._exit(3)

    threading.Thread(target=run, daemon=True).start()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    args_path = Path(argv[0])
    a = json.loads(args_path.read_text(encoding="utf-8"))
    result = Path(a["result"])
    t = time.time()
    import faulthandler
    faulthandler.enable()        # a crash inside native code still leaves a trace in the log
    _watch_parent(a.get("parent_pid"))
    try:
        config = _setup_paths()
        if a.get("ffmpeg_dir"):
            os.environ["PATH"] = a["ffmpeg_dir"] + os.pathsep + os.environ.get("PATH", "")
        out = COMMANDS[a["cmd"]](a, config)
        out.update(ok=True, seconds=round(time.time() - t, 1))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "Face not detected" in msg:
            msg = "ما لقيت وجه واضح في المشهد عشان أحرك الشفايف."
        if "out of memory" in msg.lower():
            msg = "ذاكرة كرت الشاشة ما كفت. جرّب دقة 480p أو مشهد أقصر."
        out = {"ok": False, "error": msg, "trace": traceback.format_exc()[-3000:]}
    result.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
