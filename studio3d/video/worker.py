"""Video worker: runs in its OWN process (python -m studio3d.video.worker args.json).

Why a separate process:
  * the video model needs a newer diffusers than the 3D models -> loaded from config.VIDEO_LIBS first;
  * every step frees ALL its memory when the process ends (8 GB GPU / 16 GB RAM laptops).

Commands (args["cmd"]):
  i2v      image + description -> short silent clip (Wan2.2 TI2V-5B, FastWan 4-step version)
  lipsync  clip + voice -> clip whose lips follow the voice (LatentSync 1.5)
Progress lines on stdout: "@@P <0..1> <text>". Result json -> args["result"].
"""
from __future__ import annotations

import hashlib
import json
import os
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
# image -> video (Wan2.2 TI2V-5B)
# ---------------------------------------------------------------------------
def _prompt_embeds(config, snap: Path, prompt: str):
    """Text encoder (UMT5-XXL, 11 GB) runs once per text on the CPU, then is freed; the result is
    cached on disk so the same description never has to be encoded again."""
    import torch
    cache = config.MODELS_DIR / "video" / "prompt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((str(snap.name) + "|" + prompt).encode("utf-8")).hexdigest()[:20]
    f = cache / f"{key}.pt"
    if f.exists():
        return torch.load(f, map_location="cpu", weights_only=True)
    progress(0.05, "فهم الوصف (مرة وحدة لكل وصف)…")
    from diffusers import WanImageToVideoPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel
    src = snap
    if not (snap / "text_encoder").is_dir() or not (snap / "tokenizer").is_dir():
        # the distilled repo may ship without the (unchanged) text encoder: take Wan's own
        src = _download(config, config.VIDEO_TEXT_MODEL, ["text_encoder/*", "tokenizer/*"], "مشفّر النص")
    tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True)
    te.eval()

    class _Enc:  # only what _get_t5_prompt_embeds reads
        tokenizer = tok
        text_encoder = te
        _execution_device = torch.device("cpu")

    with torch.no_grad():
        emb = WanImageToVideoPipeline._get_t5_prompt_embeds(_Enc(), prompt, 1, 512, torch.device("cpu"),
                                                            torch.bfloat16)
    del te
    import gc
    gc.collect()
    torch.save(emb, f)
    return emb


def _load_i2v(snap: Path):
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
    vae = AutoencoderKLWan.from_pretrained(snap, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(snap, vae=vae, text_encoder=None, tokenizer=None,
                                                   torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    if torch.cuda.is_available():
        # 5B weights stored in 8-bit (computed in bf16): ~5 GB instead of 10 GB on the GPU
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                                  compute_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001
        pass
    return pipe


def run_i2v(pipe, a: dict, emb) -> list:
    import torch
    from PIL import Image
    img = Image.open(a["image"]).convert("RGB").resize((a["width"], a["height"]), Image.LANCZOS)
    steps = int(a.get("steps", 4))

    def on_step(p, i, t, kw):
        progress(0.2 + 0.7 * (i + 1) / steps, f"توليد الحركة {i + 1}/{steps}")
        return {}

    gen = torch.Generator(device="cpu").manual_seed(int(a.get("seed", 0)))
    dtype = pipe.transformer.dtype if pipe.transformer is not None else torch.bfloat16
    with torch.inference_mode():
        return pipe(image=img, prompt_embeds=emb.to(dtype), height=a["height"], width=a["width"],
                    num_frames=a["frames"], num_inference_steps=steps, guidance_scale=1.0,
                    generator=gen, callback_on_step_end=on_step).frames[0]


def cmd_i2v(a: dict, config) -> dict:
    snap = _download(config, config.VIDEO_MODEL, None, "نموذج الفيديو (~22 GB)")
    emb = _prompt_embeds(config, snap, a["prompt"])
    progress(0.15, "تحميل نموذج الفيديو…")
    pipe = _load_i2v(snap)
    frames = run_i2v(pipe, a, emb)
    progress(0.93, "حفظ المقطع…")
    from diffusers.utils import export_to_video
    export_to_video(frames, a["out"], fps=int(a.get("fps", 24)))
    return {"out": a["out"], "frames": len(frames)}


# ---------------------------------------------------------------------------
# lip sync (LatentSync 1.5)
# ---------------------------------------------------------------------------
def cmd_lipsync(a: dict, config) -> dict:
    import torch
    from omegaconf import OmegaConf
    ls_snap = _download(config, config.LIPSYNC_MODEL, ["latentsync_unet.pt", "whisper/tiny.pt"],
                        "نموذج مزامنة الشفايف (~5 GB)")
    vae_snap = _download(config, config.LIPSYNC_VAE, ["config.json", "diffusion_pytorch_model.safetensors"],
                         "VAE")
    from studio3d.video import face_mp
    face_mp.ensure_model()

    progress(0.1, "تحميل نموذج مزامنة الشفايف…")
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    vendor = config.PACKAGE_DIR / "vendor_video"
    cfg = OmegaConf.load(vendor / "configs" / "unet" / "stage2.yaml")
    dtype = torch.float16
    scheduler = DDIMScheduler.from_pretrained(str(vendor / "configs"))
    audio_encoder = Audio2Feature(model_path=str(ls_snap / "whisper" / "tiny.pt"), device="cuda",
                                  num_frames=cfg.data.num_frames, audio_feat_length=cfg.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained(str(vae_snap), torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model),
                                                   str(ls_snap / "latentsync_unet.pt"), device="cpu")
    unet = unet.to(dtype=dtype)
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to("cuda")

    progress(0.2, "مزامنة الشفايف مع الصوت…")
    # LatentSync writes temp files and calls `ffmpeg` with plain names: work inside the scene folder
    os.chdir(Path(a["out"]).parent)
    g = torch.Generator(device="cpu").manual_seed(int(a.get("seed", 1247)))
    pipe(video_path=Path(a["video"]).name, audio_path=Path(a["audio"]).name, video_out_path=Path(a["out"]).name,
         num_frames=cfg.data.num_frames, num_inference_steps=int(a.get("steps", 20)),
         guidance_scale=float(a.get("guidance", 1.5)), weight_dtype=dtype,
         width=cfg.data.resolution, height=cfg.data.resolution, generator=g,
         temp_dir="ls_temp")
    if not Path(a["out"]).exists():
        raise RuntimeError("فشلت مزامنة الشفايف (لم يُنتج ملف).")
    return {"out": a["out"]}


COMMANDS = {"i2v": cmd_i2v, "lipsync": cmd_lipsync}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    args_path = Path(argv[0])
    a = json.loads(args_path.read_text(encoding="utf-8"))
    result = Path(a["result"])
    t = time.time()
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
