"""Full generation pipeline shared by the desktop UI and the Claude (MCP) server."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from filelock import FileLock

from . import config, engine, mesh_tools


@dataclass
class Options:
    quality: str = config.DEFAULT_PRESET
    texture: bool = False
    remove_bg: bool = True
    clean: bool = True
    holes: bool = False
    smooth: int = 0
    faces: int = 0            # 0 = keep
    size_mm: float = 0.0      # 0 = keep
    seed: int = 1234
    randomize: bool = True
    formats: list = field(default_factory=lambda: ["glb", "stl"])


def gpu_lock() -> FileLock:
    """Cross-process lock: the desktop app and Claude never use the GPU at the same time."""
    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return FileLock(str(config.DATA_ROOT / "gpu.lock"))


def run(kind: str, inputs: dict, prompt: str | None, opt: Options,
        progress: Callable[[float, str], None] = lambda p, d: None) -> dict:
    """kind: 'multiview' | 'single' | 'text'. inputs: view -> PIL image."""
    if opt.quality not in config.PRESETS:
        opt.quality = config.DEFAULT_PRESET
    seed = engine.resolve_seed(opt.seed, opt.randomize)
    job = engine.new_job_dir(kind)
    meta = {"kind": kind, "prompt": prompt, "quality": opt.quality, "seed": seed, "texture": opt.texture,
            "post": {"clean": opt.clean, "holes": opt.holes, "smooth": opt.smooth, "faces": opt.faces,
                     "size_mm": opt.size_mm}}

    with gpu_lock(), engine.MANAGER.lock:
        if kind == "text":
            progress(0.05, "توليد صورة من النص…")
            inputs = {"front": engine.text_to_image(prompt, seed)}

        progress(0.15, "إزالة الخلفية…")
        prepared = {}
        for view, img in inputs.items():
            if img is None:
                continue
            img.save(job / f"input_{view}.png")
            prepared[view] = engine.prepare_image(img, opt.remove_bg)
            prepared[view].save(job / f"clean_{view}.png")
        if "front" not in prepared:
            raise ValueError("الصورة الأمامية مطلوبة.")

        progress(0.25, "توليد الشكل ثلاثي الأبعاد…")
        if kind == "multiview":
            mesh = engine.generate_shape(prepared, "multiview", opt.quality, seed)
        else:
            mesh = engine.generate_shape(prepared["front"], "single", opt.quality, seed)

        # clean-up before texturing so the UVs stay valid
        progress(0.6, "تحسين المجسم…")
        white = mesh_tools.process(mesh, do_clean=opt.clean, target_faces=int(opt.faces or 0),
                                   smooth_iters=int(opt.smooth or 0), holes=opt.holes,
                                   size_mm=0, on_ground=False)
        # Save the shape right away, so a texturing failure never loses it.
        mesh_tools.export(mesh_tools.normalize(white, None, True), job, "model_white", "glb")
        final = white
        warning = None
        if opt.texture and not engine.texture_ready():
            warning = ("تم توليد الشكل بدون تلوين: نموذج التلوين لم يكتمل تنزيله بعد. "
                       "شغّل أمر التنزيل (Studio3D.cmd --download) ثم أعد المحاولة.")
            opt.texture = False
        if opt.texture:
            progress(0.7, "تلوين المجسم (قد يستغرق عدة دقائق)…")
            try:
                final = engine.texture_mesh(white, prepared["front"])
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger("studio3d.pipeline").exception("texturing failed")
                engine.MANAGER.unload()
                warning = ("تم توليد الشكل لكن فشل التلوين: " +
                           ("انقطع الإنترنت أثناء تنزيل نموذج التلوين — أعد المحاولة وسيكمل التنزيل من حيث توقف."
                            if _is_network(e) else str(e)[:300]))
                opt.texture = False

    progress(0.95, "التصدير…")
    size = float(opt.size_mm or 0) or None
    white = mesh_tools.normalize(white, size, True)
    final = mesh_tools.normalize(final, size, True) if opt.texture else white

    preview = mesh_tools.export(final, job, "model_textured" if opt.texture else "model", "glb")
    files = [preview]
    for fmt in opt.formats or []:
        if fmt != "glb":
            files.append(mesh_tools.export(final, job, "model", fmt))
    if opt.texture:
        files.append(mesh_tools.export(white, job, "model_white", "glb"))

    st = mesh_tools.stats(final)
    meta["stats"] = st
    meta["warning"] = warning
    meta["files"] = [Path(f).name for f in files]
    engine.save_meta(job, meta)
    return {"job_dir": str(job), "preview": preview, "files": files, "stats": st, "seed": seed,
            "mesh": final, "warning": warning}


def _is_network(e: Exception) -> bool:
    text = repr(e) + repr(getattr(e, "__cause__", "")) + repr(getattr(e, "__context__", ""))
    keys = ("ConnectionError", "Timeout", "timed out", "getaddrinfo", "NameResolution", "HTTPError",
            "Something wrong while loading")
    return any(k in text for k in keys)


def retexture(mesh_path: str, image, remove_bg: bool = True,
              progress: Callable[[float, str], None] = lambda p, d: None) -> dict:
    job = engine.new_job_dir("retexture")
    mesh = mesh_tools.load_mesh(mesh_path)
    with gpu_lock(), engine.MANAGER.lock:
        progress(0.1, "تجهيز…")
        img = engine.prepare_image(image, remove_bg)
        progress(0.3, "تلوين المجسم…")
        out = engine.texture_mesh(mesh, img)
    path = mesh_tools.export(out, job, "model_textured", "glb")
    obj = mesh_tools.export(out, job, "model_textured", "obj")
    st = mesh_tools.stats(out)
    engine.save_meta(job, {"kind": "retexture", "source": str(mesh_path), "stats": st})
    return {"job_dir": str(job), "preview": path, "files": [path, obj], "stats": st, "mesh": out}


def edit(mesh_path: str, *, clean=True, holes=False, smooth=0, faces=0, size_mm=0.0,
         formats=("glb", "stl")) -> dict:
    job = engine.new_job_dir("edit")
    mesh = mesh_tools.load_mesh(mesh_path)
    before = mesh_tools.stats(mesh)
    mesh = mesh_tools.process(mesh, do_clean=clean, target_faces=int(faces or 0),
                              smooth_iters=int(smooth or 0), holes=holes,
                              size_mm=float(size_mm or 0), on_ground=True)
    after = mesh_tools.stats(mesh)
    preview = mesh_tools.export(mesh, job, "model", "glb")
    files = [preview] + [mesh_tools.export(mesh, job, "model", f) for f in formats or [] if f != "glb"]
    engine.save_meta(job, {"kind": "edit", "source": str(mesh_path), "before": before, "stats": after})
    return {"job_dir": str(job), "preview": preview, "files": files, "before": before, "stats": after,
            "mesh": mesh}
