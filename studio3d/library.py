"""The model library: every generated / edited model, with names, thumbnails and quick actions.

Each model is one folder in OUTPUT_DIR (its id is the folder name) holding the files,
meta.json and thumb.png.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import config

KIND_LABELS = {"multiview": "من 4 صور", "single": "من صورة", "text": "من نص", "retexture": "تلوين",
               "edit": "تعديل", "quad": "Quad", "rig": "هيكل عظمي", "talk": "يتكلم"}
# best file to show, in order
PREVIEW_ORDER = ["model_talking.glb", "model_rigged.glb", "model_quad.glb", "model_textured.glb", "model.glb", "model_white.glb"]
MODEL_EXT = {".glb", ".obj", ".fbx", ".stl", ".ply", ".3mf", ".zip", ".blend", ".wav", ".mp3", ".ogg", ".flac"}


def _dir(model_id: str) -> Path:
    d = (config.OUTPUT_DIR / model_id).resolve()
    if config.OUTPUT_DIR.resolve() not in d.parents or not d.is_dir():
        raise FileNotFoundError(f"Unknown model id: {model_id}")
    return d


def read_meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_meta(d: Path, meta: dict):
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def preview_file(d: Path) -> Path | None:
    for name in PREVIEW_ORDER:
        if (d / name).exists():
            return d / name
    glbs = sorted(d.glob("*.glb"))
    return glbs[0] if glbs else None


def default_name(d: Path, meta: dict) -> str:
    if meta.get("name"):
        return meta["name"]
    if meta.get("prompt"):
        return meta["prompt"][:40]
    try:
        stamp = datetime.strptime(d.name[:15], "%Y%m%d_%H%M%S").strftime("%m/%d %H:%M")
    except ValueError:
        stamp = d.name[:15]
    return f"{KIND_LABELS.get(meta.get('kind', ''), 'مجسم')} {stamp}"


def ensure_thumb(d: Path, size: int = 256) -> Path | None:
    thumb = d / "thumb.png"
    if thumb.exists():
        return thumb
    src = preview_file(d)
    if src is None:
        return None
    try:
        from . import mesh_tools
        from .preview import render_view
        render_view(mesh_tools.load_mesh(src), azim=30, elev=12, size=size).save(thumb)
        return thumb
    except Exception:  # noqa: BLE001
        return None


def entry(d: Path, with_thumb: bool = True) -> dict:
    meta = read_meta(d)
    prev = preview_file(d)
    files = sorted(str(p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in MODEL_EXT)
    return {
        "id": d.name,
        "name": default_name(d, meta),
        "kind": meta.get("kind", ""),
        "prompt": meta.get("prompt"),
        "stats": meta.get("stats", {}),
        "rigged": (d / "model_rigged.glb").exists() or (d / "model_rigged.fbx").exists(),
        "quad": (d / "model_quad.obj").exists(),
        "textured": (d / "model_textured.glb").exists() or bool(meta.get("texture")),
        "preview": str(prev) if prev else None,
        "thumb": str(ensure_thumb(d)) if with_thumb else None,
        "files": files,
        "folder": str(d),
        "audio": next((str(p) for p in d.glob("voice.*")), None),
    }


def list_models(query: str | None = None, limit: int = 200, with_thumb: bool = True) -> list[dict]:
    if not config.OUTPUT_DIR.exists():
        return []
    q = (query or "").strip().lower()
    out = []
    for d in sorted((x for x in config.OUTPUT_DIR.iterdir() if x.is_dir()), reverse=True):
        if preview_file(d) is None:
            continue
        e = entry(d, with_thumb=False)
        hay = " ".join(str(x) for x in (e["name"], e["prompt"], e["kind"], KIND_LABELS.get(e["kind"], ""),
                                        e["id"])).lower()
        if q and q not in hay:
            continue
        if with_thumb:
            e["thumb"] = str(ensure_thumb(d) or "")
        out.append(e)
        if len(out) >= limit:
            break
    return out


def get(model_id: str) -> dict:
    return entry(_dir(model_id))


def rename(model_id: str, name: str) -> dict:
    d = _dir(model_id)
    meta = read_meta(d)
    meta["name"] = (name or "").strip()[:80]
    write_meta(d, meta)
    return entry(d)


def resolve(ref: str) -> dict:
    """Accept a model id, a folder path or a file path inside the library."""
    p = Path(ref)
    if p.exists():
        d = p if p.is_dir() else p.parent
        return entry(d.resolve())
    return get(ref)


def export_to(model_id: str, destination: str, formats: list[str] | None = None,
              name: str | None = None) -> list[str]:
    """Copy a model's files into another folder (e.g. a Unity Assets folder)."""
    e = get(model_id)
    dest = Path(destination).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    base = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or e["name"])).strip() or e["id"]
    copied = []
    wanted = {f.lower().lstrip(".") for f in formats} if formats else None
    for f in e["files"]:
        src = Path(f)
        ext = src.suffix.lower().lstrip(".")
        if wanted and ext not in wanted:
            continue
        target = dest / f"{base}_{src.stem}{src.suffix}" if len(e["files"]) > 1 else dest / f"{base}{src.suffix}"
        shutil.copy2(src, target)
        copied.append(str(target))
    # textures that belong to OBJ files
    for extra in Path(e["folder"]).glob("*.png"):
        if extra.name != "thumb.png" and not extra.name.startswith(("input_", "clean_")) and \
                (not wanted or "obj" in wanted):
            shutil.copy2(extra, dest / extra.name)
            copied.append(str(dest / extra.name))
    return copied


def open_folder(model_id: str):
    open_path(_dir(model_id))


def open_path(d):
    """Show a folder in the file explorer."""
    d = str(d)
    if sys.platform == "win32":
        os.startfile(d)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", d])
    else:
        subprocess.Popen(["xdg-open", d])


def open_in_blender(model_id: str, which: str | None = None) -> str:
    """Open the model in the Blender window (not hidden) for manual work."""
    from . import blender
    e = get(model_id)
    path = which or next((f for f in e["files"] if f.endswith(("rigged.fbx", "quad.fbx"))), None) or e["preview"]
    blender.open_gui(path)
    return path
