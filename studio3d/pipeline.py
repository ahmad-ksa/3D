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
    topology: str = "tri"     # "tri" | "quad" (quad = QuadriFlow in Blender)
    rig: bool = False         # humanoid skeleton via Blender


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
                           ("انقطع الإنترنت (حاول البرنامج دقيقتين بدون نجاح) — أعد المحاولة وسيكمل التنزيل من حيث توقف."
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
    warnings = [warning] if warning else []
    if opt.topology == "quad" or opt.rig:
        preview, extra = _blender_steps(preview, job, opt, progress, warnings, meta)
        files += extra
        if "quad" in meta:
            st = dict(st, faces=meta["quad"]["faces"], quads=meta["quad"]["quads"],
                      vertices=meta["quad"]["vertices"])
    warning = " | ".join(warnings) or None
    meta["stats"] = st
    meta["warning"] = warning
    meta["files"] = [Path(f).name for f in files]
    engine.save_meta(job, meta)
    return {"job_dir": str(job), "preview": preview, "files": files, "stats": st, "seed": seed,
            "mesh": final, "warning": warning}


def _blender_steps(model_path, job, opt, progress, warnings, meta):
    """Quad topology and/or skeleton through Blender (runs hidden, then closes)."""
    from . import blender
    files, base = [], model_path
    if opt.topology == "quad":
        progress(0.96, "تحويل إلى مربعات Quad عبر Blender…")
        try:
            q = blender.quad_remesh(base, job, target_faces=int(opt.faces or 10000),
                                    bake_size=2048 if opt.texture else 0)
            files += q["files"]
            base = next(f for f in q["files"] if f.endswith(".glb"))
            meta["quad"] = {k: q[k] for k in ("faces", "quads", "vertices", "method", "textured")}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"فشل تحويل Quad: {str(e)[:200]}")
    if opt.rig:
        progress(0.98, "بناء الهيكل العظمي عبر Blender…")
        try:
            r = blender.auto_rig(base, job)
            files += r["files"]
            meta["rig"] = {k: r.get(k) for k in ("bones", "a_pose", "warnings", "arm_test_moved_fraction")}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"فشل بناء الهيكل العظمي: {str(e)[:200]}")
    return base, files


def quad(model_path: str, target_faces: int = 10000, bake_size: int = 2048) -> dict:
    from . import blender
    job = engine.new_job_dir("quad")
    q = blender.quad_remesh(model_path, job, target_faces=target_faces, bake_size=bake_size)
    preview = next(f for f in q["files"] if f.endswith(".glb"))
    stats = {k: q[k] for k in ("faces", "quads", "vertices")}
    engine.save_meta(job, {"kind": "quad", "source": str(model_path), "stats": stats, "method": q["method"]})
    return {"job_dir": str(job), "preview": preview, "files": q["files"], "stats": stats,
            "mesh": mesh_tools.load_mesh(preview)}


def rig(model_path: str) -> dict:
    from . import blender
    job = engine.new_job_dir("rig")
    r = blender.auto_rig(model_path, job)
    preview = next(f for f in r["files"] if f.endswith(".glb"))
    info = {k: r.get(k) for k in ("bones", "a_pose", "warnings", "arm_test_moved_fraction", "faces",
                                  "vertices")}
    engine.save_meta(job, {"kind": "rig", "source": str(model_path), "rig": info})
    return {"job_dir": str(job), "preview": preview, "files": r["files"], "stats": info,
            "mesh": mesh_tools.load_mesh(preview)}


def lip_sync(model_path: str, audio_path: str, strength: float = 10) -> dict:
    """Talking animation: jaw follows the loudness of an audio file (needs a rigged model)."""
    import shutil
    from . import blender
    model_path = str(model_path)
    if not model_path.lower().endswith((".glb", ".fbx")):
        raise ValueError("استخدم ملف GLB أو FBX فيه هيكل عظمي (من Studio3D).")
    job = engine.new_job_dir("talk")
    audio = job / ("voice" + Path(audio_path).suffix.lower())
    shutil.copy2(audio_path, audio)
    r = blender.lip_sync(model_path, audio, job, max_open_degrees=strength)
    preview = next(f for f in r["files"] if f.endswith(".glb"))
    info = {k: r.get(k) for k in ("frames", "seconds", "fps", "jaw_vertices")}
    engine.save_meta(job, {"kind": "talk", "source": model_path, "audio": audio.name, "talk": info})
    return {"job_dir": str(job), "preview": preview, "files": r["files"] + [str(audio)], "stats": info,
            "audio": str(audio), "mesh": mesh_tools.load_mesh(preview)}


def _is_network(e: Exception) -> bool:
    return config.is_network_error(e) or "Something wrong while loading" in str(e)


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
