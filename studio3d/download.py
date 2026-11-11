"""Pre-download AI models so the first generation starts immediately."""
from __future__ import annotations

from . import config


def download_models(include_text: bool = False):
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
    print("✅ اكتمل تنزيل النماذج.")
