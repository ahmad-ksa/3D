"""Arabic (RTL) Gradio interface."""
from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import gradio as gr

from . import config, engine, mesh_tools

log = logging.getLogger("studio3d.ui")

CSS = """
.gradio-container {direction: rtl; font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1400px !important;}
.gradio-container input[type=number], .gradio-container canvas {direction: ltr;}
#title {text-align: center;}
.hint {opacity: .75; font-size: .9em;}
"""

VIEW_LABELS = {"front": "أمام (مطلوبة)", "back": "خلف", "left": "يسار", "right": "يمين"}


# ---------------------------------------------------------------------------
# Shared option widgets
# ---------------------------------------------------------------------------
def _options(show_texture=True, show_quality=True):
    with gr.Accordion("الإعدادات", open=True):
        quality = gr.Radio(list(config.PRESETS), value=config.DEFAULT_PRESET, label="الجودة",
                           visible=show_quality,
                           info="سريع ≈ نصف دقيقة • متوازن ≈ دقيقة • عالي الجودة ≈ عدة دقائق")
        texture = gr.Checkbox(value=False, label="تلوين المجسم بالذكاء الاصطناعي (Texture)",
                              visible=show_texture,
                              info="يستغرق وقتاً إضافياً (عدة دقائق على 8GB)")
        remove_bg = gr.Checkbox(value=True, label="إزالة الخلفية تلقائياً")
    with gr.Accordion("تحسين المجسم", open=False):
        clean = gr.Checkbox(value=True, label="تنظيف (إزالة الشوائب والأوجه التالفة)")
        holes = gr.Checkbox(value=False, label="سد الثقوب")
        smooth = gr.Slider(0, 30, value=0, step=1, label="تنعيم السطح (عدد المرات)")
        faces = gr.Slider(0, 500000, value=0, step=1000, label="عدد المضلعات المستهدف (0 = بدون تقليل)",
                          info="للألعاب والجوال: 5,000 – 30,000")
        size_mm = gr.Number(value=0, label="المقاس الحقيقي لأطول ضلع بالملّيمتر (0 = بدون تغيير)",
                            info="مفيد للطباعة ثلاثية الأبعاد")
    with gr.Accordion("متقدم", open=False):
        seed = gr.Number(value=1234, precision=0, label="البذرة (Seed)")
        rand = gr.Checkbox(value=True, label="بذرة عشوائية كل مرة")
        formats = gr.CheckboxGroup(config.EXPORT_FORMATS, value=["glb", "stl"], label="صيغ التصدير")
    return [quality, texture, remove_bg, clean, holes, smooth, faces, size_mm, seed, rand, formats]


def _outputs():
    model = gr.Model3D(label="المعاينة (اسحب للتدوير)", height=520, clear_color=[0.92, 0.92, 0.95, 1])
    files = gr.File(label="ملفات التحميل", file_count="multiple")
    info = gr.Markdown()
    return model, files, info


# ---------------------------------------------------------------------------
# Core pipeline used by every tab
# ---------------------------------------------------------------------------
def _run(kind, inputs: dict, prompt, quality, texture, remove_bg, clean, holes, smooth, faces,
         size_mm, seed, rand, formats, progress):
    seed = engine.resolve_seed(seed, rand)
    job = engine.new_job_dir(kind)
    meta = {"kind": kind, "prompt": prompt, "quality": quality, "seed": seed, "texture": texture,
            "post": {"clean": clean, "holes": holes, "smooth": smooth, "faces": faces, "size_mm": size_mm}}

    with engine.MANAGER.lock:
        try:
            # 1) text -> image
            if kind == "text":
                progress(0.05, desc="توليد صورة من النص…")
                img = engine.text_to_image(prompt, seed)
                inputs = {"front": img}

            # 2) background removal
            progress(0.15, desc="إزالة الخلفية…")
            prepared = {}
            for view, img in inputs.items():
                if img is None:
                    continue
                img.save(job / f"input_{view}.png")
                prepared[view] = engine.prepare_image(img, remove_bg)
                prepared[view].save(job / f"clean_{view}.png")

            # 3) shape
            progress(0.25, desc="توليد الشكل ثلاثي الأبعاد…")
            if kind == "multiview":
                mesh = engine.generate_shape(prepared, "multiview", quality, seed)
            else:
                mesh = engine.generate_shape(prepared["front"], "single", quality, seed)

            # 4) clean-up / edit (before texturing so the UVs stay valid)
            progress(0.6, desc="تحسين المجسم…")
            mesh = mesh_tools.process(mesh, do_clean=clean, target_faces=int(faces or 0),
                                      smooth_iters=int(smooth or 0), holes=holes,
                                      size_mm=0, on_ground=False)
            white = mesh

            # 5) texture
            final = white
            if texture:
                progress(0.7, desc="تلوين المجسم (قد يستغرق عدة دقائق)…")
                final = engine.texture_mesh(white, prepared["front"])

            # 6) size / ground + export
            progress(0.95, desc="التصدير…")
            white = mesh_tools.normalize(white, float(size_mm or 0) or None, True)
            if final is not white:
                final = mesh_tools.normalize(final, float(size_mm or 0) or None, True)
            else:
                final = white
            preview = mesh_tools.export(final, job, "model_textured" if texture else "model", "glb")
            files = [preview]
            for fmt in formats or []:
                if fmt == "glb":
                    continue
                files.append(mesh_tools.export(final, job, "model", fmt))
            if texture:
                files.append(mesh_tools.export(white, job, "model_white", "glb"))

            st = mesh_tools.stats(final)
            meta["stats"] = st
            engine.save_meta(job, meta)
        except Exception as e:  # noqa: BLE001
            log.error(traceback.format_exc())
            raise gr.Error(_friendly_error(e))

    info = (f"✅ تم — **{st['faces']:,}** مضلع • **{st['vertices']:,}** نقطة • البذرة `{seed}`  \n"
            f"<span class='hint'>محفوظ في: {job}</span>")
    return preview, files, info


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    name = type(e).__name__
    if "OutOfMemory" in name or "out of memory" in msg.lower():
        return "امتلأت ذاكرة كرت الشاشة. اختر جودة أقل، أو أغلق البرامج الأخرى (الألعاب/المتصفح) ثم أعد المحاولة."
    if "Connection" in name or "resolve" in msg.lower() or "HTTPError" in name:
        return "تعذّر تنزيل النموذج. تأكد من الاتصال بالإنترنت (مطلوب أول مرة فقط) ثم أعد المحاولة."
    if "blocked" in msg.lower() or "4551" in msg or "Application Control" in msg:
        return ("حظر Smart App Control أحد ملفات المكتبات. افتح تبويب «الجهاز» واضغط «فحص الحماية» "
                "لمعرفة الملف وطريقة الحل.")
    return f"حدث خطأ: {msg[:400]}"


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------
def gen_multiview(front, back, left, right, *opts, progress=gr.Progress(track_tqdm=True)):
    if front is None:
        raise gr.Error("الصورة الأمامية مطلوبة على الأقل.")
    inputs = {"front": front, "back": back, "left": left, "right": right}
    return _run("multiview", inputs, None, *opts, progress)


def gen_single(image, *opts, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("ارفع صورة أولاً.")
    return _run("single", {"front": image}, None, *opts, progress)


def gen_text(prompt, *opts, progress=gr.Progress(track_tqdm=True)):
    if not prompt or not prompt.strip():
        raise gr.Error("اكتب وصفاً للمجسم.")
    return _run("text", {}, prompt.strip(), *opts, progress)


def retexture(mesh_file, image, remove_bg, progress=gr.Progress(track_tqdm=True)):
    if mesh_file is None or image is None:
        raise gr.Error("ارفع المجسم وصورة مرجعية.")
    job = engine.new_job_dir("retexture")
    with engine.MANAGER.lock:
        try:
            progress(0.1, desc="تجهيز…")
            mesh = mesh_tools.load_mesh(mesh_file)
            img = engine.prepare_image(image, remove_bg)
            progress(0.3, desc="تلوين المجسم…")
            out = engine.texture_mesh(mesh, img)
            path = mesh_tools.export(out, job, "model_textured", "glb")
            obj = mesh_tools.export(out, job, "model_textured", "obj")
            engine.save_meta(job, {"kind": "retexture", "stats": mesh_tools.stats(out)})
        except Exception as e:  # noqa: BLE001
            log.error(traceback.format_exc())
            raise gr.Error(_friendly_error(e))
    return path, [path, obj], f"✅ تم التلوين — محفوظ في {job}"


def edit_mesh(mesh_file, clean, holes, smooth, faces, size_mm, formats):
    if mesh_file is None:
        raise gr.Error("ارفع ملف مجسم (GLB / OBJ / STL / PLY).")
    job = engine.new_job_dir("edit")
    mesh = mesh_tools.load_mesh(mesh_file)
    before = mesh_tools.stats(mesh)
    mesh = mesh_tools.process(mesh, do_clean=clean, target_faces=int(faces or 0),
                              smooth_iters=int(smooth or 0), holes=holes,
                              size_mm=float(size_mm or 0), on_ground=True)
    after = mesh_tools.stats(mesh)
    preview = mesh_tools.export(mesh, job, "model", "glb")
    files = [preview] + [mesh_tools.export(mesh, job, "model", f) for f in formats or [] if f != "glb"]
    info = f"قبل: {before['faces']:,} مضلع ← بعد: **{after['faces']:,}** مضلع • المقاس: {after['size']}"
    return preview, files, info


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def _jobs():
    if not config.OUTPUT_DIR.exists():
        return []
    return sorted([d.name for d in config.OUTPUT_DIR.iterdir() if d.is_dir()], reverse=True)


def refresh_history():
    jobs = _jobs()
    return gr.update(choices=jobs, value=jobs[0] if jobs else None)


def show_job(name):
    if not name:
        return None, None, ""
    d = config.OUTPUT_DIR / name
    files = sorted(str(p) for p in d.iterdir() if p.is_file() and not p.name.startswith(("input_", "clean_")))
    preview = next((f for f in files if f.endswith("model_textured.glb")), None) or \
        next((f for f in files if f.endswith(".glb")), None)
    meta = ""
    if (d / "meta.json").exists():
        meta = "```json\n" + (d / "meta.json").read_text(encoding="utf-8") + "\n```"
    return preview, files, meta


def device_text():
    try:
        info = engine.device_info()
    except Exception as e:  # noqa: BLE001
        return f"تعذّر قراءة كرت الشاشة: {e}"
    if not info["cuda"]:
        return ("⚠️ **لم يتم العثور على كرت NVIDIA يعمل.** التوليد سيكون بطيئاً جداً أو لن يعمل. "
                "حدّث تعريف NVIDIA وأعد تشغيل البرنامج.")
    return f"✅ كرت الشاشة: **{info['name']}** — الذاكرة: **{info['vram_gb']} GB**"


def run_sac_check():
    from . import sac_check
    return sac_check.report_markdown()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")
    with gr.Blocks(title=config.APP_TITLE, css=CSS, theme=theme, analytics_enabled=False) as demo:
        gr.Markdown(f"# 🧊 {config.APP_TITLE}\nتوليد مجسمات ثلاثية الأبعاد بالذكاء الاصطناعي — "
                    "**محلياً على جهازك، مجاناً، وبدون إنترنت** (بعد تنزيل النماذج أول مرة).",
                    elem_id="title")

        with gr.Tabs():
            # ---- multiview ----
            with gr.Tab("📸 من 4 صور"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("صوّر القطعة من **الأمام والخلف واليمين واليسار** على خلفية بسيطة "
                                    "وإضاءة متساوية. صورة الأمام مطلوبة، والبقية تزيد الدقة.",
                                    elem_classes="hint")
                        with gr.Row():
                            mv = {v: gr.Image(type="pil", image_mode="RGBA", label=VIEW_LABELS[v], height=200)
                                  for v in ("front", "back")}
                        with gr.Row():
                            mv.update({v: gr.Image(type="pil", image_mode="RGBA", label=VIEW_LABELS[v],
                                                   height=200) for v in ("left", "right")})
                        opts = _options()
                        btn = gr.Button("✨ توليد المجسم", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        out = _outputs()
                btn.click(gen_multiview, [mv["front"], mv["back"], mv["left"], mv["right"], *opts], list(out))

            # ---- single image ----
            with gr.Tab("🖼️ من صورة واحدة"):
                with gr.Row():
                    with gr.Column(scale=5):
                        img = gr.Image(type="pil", image_mode="RGBA", label="الصورة", height=320)
                        opts1 = _options()
                        btn1 = gr.Button("✨ توليد المجسم", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        out1 = _outputs()
                btn1.click(gen_single, [img, *opts1], list(out1))

            # ---- text ----
            with gr.Tab("✍️ من نص"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(label="وصف المجسم", lines=3,
                                            placeholder="A wooden chair with curved legs",
                                            info="اكتب الوصف بالإنجليزية لأفضل نتيجة. "
                                                 "يتطلب تنزيل نموذج إضافي (~10GB) أول مرة.")
                        opts2 = _options()
                        btn2 = gr.Button("✨ توليد المجسم", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        out2 = _outputs()
                btn2.click(gen_text, [prompt, *opts2], list(out2))

            # ---- retexture ----
            with gr.Tab("🎨 تلوين مجسم"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("ارفع مجسماً جاهزاً (بدون ألوان) وصورة توضح الألوان المطلوبة.",
                                    elem_classes="hint")
                        tmesh = gr.File(label="المجسم (GLB / OBJ / STL / PLY)",
                                        file_types=[".glb", ".obj", ".stl", ".ply"], type="filepath")
                        timg = gr.Image(type="pil", image_mode="RGBA", label="الصورة المرجعية", height=260)
                        trbg = gr.Checkbox(value=True, label="إزالة الخلفية تلقائياً")
                        btn3 = gr.Button("🎨 تلوين", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        out3 = _outputs()
                btn3.click(retexture, [tmesh, timg, trbg], list(out3))

            # ---- mesh tools ----
            with gr.Tab("🛠️ أدوات المجسم"):
                with gr.Row():
                    with gr.Column(scale=5):
                        emesh = gr.File(label="المجسم", file_types=[".glb", ".obj", ".stl", ".ply"],
                                        type="filepath")
                        eclean = gr.Checkbox(value=True, label="تنظيف")
                        eholes = gr.Checkbox(value=False, label="سد الثقوب")
                        esmooth = gr.Slider(0, 30, value=0, step=1, label="تنعيم")
                        efaces = gr.Slider(0, 500000, value=20000, step=1000,
                                           label="عدد المضلعات (Low-poly)")
                        esize = gr.Number(value=0, label="المقاس بالملّيمتر (0 = بدون تغيير)")
                        eformats = gr.CheckboxGroup(config.EXPORT_FORMATS, value=["glb", "stl"],
                                                    label="صيغ التصدير")
                        btn4 = gr.Button("تطبيق", variant="primary")
                    with gr.Column(scale=6):
                        out4 = _outputs()
                btn4.click(edit_mesh, [emesh, eclean, eholes, esmooth, efaces, esize, eformats], list(out4))

            # ---- history ----
            with gr.Tab("🗂️ السجل"):
                with gr.Row():
                    hsel = gr.Dropdown(choices=_jobs(), label="الأعمال السابقة", scale=4)
                    hbtn = gr.Button("تحديث", scale=1)
                with gr.Row():
                    with gr.Column(scale=6):
                        hmodel = gr.Model3D(label="المعاينة", height=480)
                    with gr.Column(scale=4):
                        hfiles = gr.File(label="الملفات", file_count="multiple")
                        hmeta = gr.Markdown()
                hbtn.click(refresh_history, None, hsel)
                hsel.change(show_job, hsel, [hmodel, hfiles, hmeta])

            # ---- device ----
            with gr.Tab("💻 الجهاز"):
                dev = gr.Markdown()
                gr.Markdown(f"مجلد النماذج: `{config.MODELS_DIR}`  \nمجلد النتائج: `{config.OUTPUT_DIR}`")
                sac_btn = gr.Button("🛡️ فحص الحماية (Smart App Control)")
                sac_out = gr.Markdown()
                unload = gr.Button("تفريغ ذاكرة كرت الشاشة")
                unload.click(lambda: (engine.MANAGER.unload(), "تم التفريغ ✅")[1], None, sac_out)
                sac_btn.click(run_sac_check, None, sac_out)
                demo.load(device_text, None, dev)

        gr.Markdown("<span class='hint'>يعمل بنماذج Tencent Hunyuan3D 2.0 مفتوحة المصدر "
                    "(رخصة Tencent Hunyuan 3D 2.0 Community).</span>")
    return demo
