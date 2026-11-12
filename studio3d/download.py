"""Pre-download AI models so the first generation starts immediately."""
from __future__ import annotations

from . import config


def download_models(include_text: bool = False):
    config.patch_hf_downloads()
    from huggingface_hub import snapshot_download

    jobs = []
    for spec in config.SHAPE_MODELS.values():
        pats = [f"{spec['turbo']}/*"] + ([f"{spec['full']}/*"] if include_text else [])
        jobs.append((spec["repo"], pats))
    jobs.append(("tencent/Hunyuan3D-2", ["hunyuan3d-vae-v2-0-turbo/*", "hunyuan3d-delight-v2-0/*",
                                         f"{config.TEXTURE_MODEL['subfolder']}/*"]))
    for repo, patterns in jobs:
        print(f"⬇️  {repo}  {patterns}")
        snapshot_download(repo_id=repo, allow_patterns=patterns)

    print("⬇️  نموذج إزالة الخلفية")
    from rembg import new_session
    new_session()

    if include_text:
        print(f"⬇️  {config.TEXT2IMAGE_MODEL} (نص ← صورة)")
        snapshot_download(repo_id=config.TEXT2IMAGE_MODEL)
        print(f"⬇️  {config.POSE_CONTROLNET_MODEL} (وضعية T / A)")
        snapshot_download(repo_id=config.POSE_CONTROLNET_MODEL,
                          allow_patterns=["config.json", "*.safetensors"])
    freed = config.remove_ckpt_duplicates()
    if freed:
        print(f"🧹 حُذفت نسخ مكررة ووفّرنا {freed} GB")
    print("✅ اكتمل تنزيل النماذج.")


def download_video_models():
    """Models of the 2D video feature (~28 GB): Wan2.2 TI2V-5B (FastWan), LatentSync 1.5, face model."""
    config.patch_hf_downloads()
    from huggingface_hub import snapshot_download
    print(f"⬇️  {config.VIDEO_MODEL} (فيديو، ~22 GB)")
    snapshot_download(repo_id=config.VIDEO_MODEL)
    print(f"⬇️  {config.LIPSYNC_MODEL} (مزامنة الشفايف)")
    snapshot_download(repo_id=config.LIPSYNC_MODEL, allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"])
    snapshot_download(repo_id=config.LIPSYNC_VAE, allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
    from .video import face_mp
    face_mp.ensure_model()
    print("✅ اكتمل تنزيل نماذج الفيديو.")
