"""Arabic (RTL) Gradio interface."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import gradio as gr

from . import config, engine, mesh_tools, pipeline

log = logging.getLogger("studio3d.ui")

CSS = """
.gradio-container {direction: rtl; font-family: 'Segoe UI', Tahoma, sans-serif;
                   max-width: 100% !important; padding: 6px 14px !important;}
.gradio-container input[type=number], .gradio-container canvas {direction: ltr;}
#title {margin: 0 !important; padding: 0 !important;}
#title h3 {margin: 2px 0 !important;}
.hint {opacity: .7; font-size: .85em; margin: 0 !important;}
.compact {gap: 6px !important;}
.files-out .empty, .files-out .wrap, .files-out button.center {min-height: 56px !important; height: 56px !important;}
.files-out table {max-height: 130px; overflow-y: auto; display: block;}
.img-in .upload-container, .img-in .image-frame {min-height: 0 !important;}
/* elapsed time as mm:ss (see TIMER_JS) */
.progress-text[data-mmss] {font-size: 0 !important;}
.progress-text[data-mmss]::after {content: attr(data-mmss); font-size: var(--text-sm);
                                  font-variant-numeric: tabular-nums;}
"""

# Show Gradio's "12.3/45.6s" timers as "00:12 / 00:46"
TIMER_JS = r"""
<script>
(function () {
  function mmss(v) {
    v = Math.max(0, Math.round(parseFloat(v)));
    var h = Math.floor(v / 3600), m = Math.floor((v % 3600) / 60), s = v % 60;
    var t = String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    return h ? h + ':' + t : t;
  }
  function fmt(text) {
    return text
      .replace(/(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)\s*s\b/g, function (_, a, b) { return mmss(a) + ' / ' + mmss(b); })
      .replace(/(\d+(?:\.\d+)?)\s*s\b/g, function (_, a) { return mmss(a); })
      .replace('processing', 'جارٍ العمل').replace('queue', 'الانتظار');
  }
  function update() {
    document.querySelectorAll('.progress-text').forEach(function (el) {
      var raw = el.textContent || '';
      var out = fmt(raw);
      if (el.getAttribute('data-mmss') !== out) el.setAttribute('data-mmss', out);
    });
  }
  setInterval(update, 250);
})();
</script>
"""

VIEW_LABELS = {"front": "أمام *", "back": "خلف", "left": "يسار", "right": "يمين"}


# ---------------------------------------------------------------------------
# Shared option widgets
# ---------------------------------------------------------------------------
def _quick_options(show_texture=True):
    with gr.Row(equal_height=True, elem_classes="compact"):
        quality = gr.Radio(list(config.PRESETS), value=config.DEFAULT_PRESET, label="الجودة",
                           scale=3, container=True)
        with gr.Column(scale=2, min_width=160, elem_classes="compact"):
            texture = gr.Checkbox(value=False, label="🎨 تلوين (Texture)", visible=show_texture)
            remove_bg = gr.Checkbox(value=True, label="إزالة الخلفية")
    with gr.Row(equal_height=True, elem_classes="compact"):
        topology = gr.Radio([("مثلثات", "tri"), ("مربعات Quad", "quad")], value="tri", label="Topology",
                            scale=3)
        rig = gr.Checkbox(value=False, label="🦴 هيكل عظمي (شخصيات)", scale=2, min_width=160)
    return quality, texture, remove_bg, topology, rig


def _more_options():
    with gr.Accordion("⚙️ إعدادات إضافية", open=False):
        with gr.Row(elem_classes="compact"):
            clean = gr.Checkbox(value=True, label="تنظيف", min_width=90)
            holes = gr.Checkbox(value=False, label="سد الثقوب", min_width=90)
            rand = gr.Checkbox(value=True, label="بذرة عشوائية", min_width=90)
        with gr.Row(elem_classes="compact"):
            smooth = gr.Slider(0, 30, value=0, step=1, label="تنعيم")
            faces = gr.Slider(0, 500000, value=0, step=1000, label="المضلعات (0 = كامل)")
        with gr.Row(elem_classes="compact"):
            size_mm = gr.Number(value=0, label="المقاس بالملّيمتر (0 = بدون)", min_width=120)
            seed = gr.Number(value=1234, precision=0, label="البذرة", min_width=100)
        formats = gr.CheckboxGroup(config.EXPORT_FORMATS, value=["glb", "stl"], label="صيغ التصدير")
    return clean, holes, smooth, faces, size_mm, seed, rand, formats


def _generate_block(show_texture=True, label="✨ توليد المجسم"):
    """Quick settings + button (always visible) + collapsed extra settings."""
    quality, texture, remove_bg, topology, rig = _quick_options(show_texture)
    btn = gr.Button(label, variant="primary")
    clean, holes, smooth, faces, size_mm, seed, rand, formats = _more_options()
    return btn, [quality, texture, remove_bg, clean, holes, smooth, faces, size_mm, seed, rand, formats,
                 topology, rig]


def _outputs():
    model = gr.Model3D(label="المعاينة (اسحب للتدوير)", height=360, clear_color=[0.92, 0.92, 0.95, 1])
    info = gr.Markdown()
    files = gr.File(label="ملفات التحميل", file_count="multiple", elem_classes="files-out")
    return model, files, info


# ---------------------------------------------------------------------------
# Core pipeline used by every tab
# ---------------------------------------------------------------------------
def _run(kind, inputs: dict, prompt, quality, texture, remove_bg, clean, holes, smooth, faces,
         size_mm, seed, rand, formats, topology="tri", rig=False, progress=None):
    if progress is None:
        progress = gr.Progress()
    opt = pipeline.Options(topology=topology or "tri", rig=bool(rig), quality=quality, texture=bool(texture), remove_bg=bool(remove_bg),
                           clean=bool(clean), holes=bool(holes), smooth=int(smooth or 0),
                           faces=int(faces or 0), size_mm=float(size_mm or 0),
                           seed=int(seed if seed is not None else -1), randomize=bool(rand),
                           formats=list(formats or []))
    try:
        res = pipeline.run(kind, inputs, prompt, opt, lambda p, d: progress(p, desc=d))
    except Exception as e:  # noqa: BLE001
        log.error(traceback.format_exc())
        raise gr.Error(_friendly_error(e))
    st = res["stats"]
    if res.get("warning"):
        gr.Warning(res["warning"])
    quads = f" ({st['quads']:,} Quad)" if st.get("quads") else ""
    info = ("" if not res.get("warning") else f"⚠️ {res['warning']}  \n") + (f"✅ تم — **{st['faces']:,}** مضلع{quads} • **{st['vertices']:,}** نقطة • البذرة `{res['seed']}`  \n"
            f"<span class='hint'>محفوظ في: {res['job_dir']}</span>")
    return res["preview"], res["files"], info


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    name = type(e).__name__
    if "OutOfMemory" in name or "out of memory" in msg.lower():
        return "امتلأت ذاكرة كرت الشاشة. اختر جودة أقل، أو أغلق البرامج الأخرى (الألعاب/المتصفح) ثم أعد المحاولة."
    if pipeline._is_network(e):
        return ("انقطع الإنترنت (حاول البرنامج دقيقتين بدون نجاح). تأكد من الاتصال ثم أعد المحاولة — "
                "التنزيل يكمل من حيث توقف (مطلوب أول مرة فقط).")
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
    return _run("multiview", inputs, None, *opts, progress=progress)


def gen_single(image, *opts, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("ارفع صورة أولاً.")
    return _run("single", {"front": image}, None, *opts, progress=progress)


def gen_text(prompt, *opts, progress=gr.Progress(track_tqdm=True)):
    if not prompt or not prompt.strip():
        raise gr.Error("اكتب وصفاً للمجسم.")
    return _run("text", {}, prompt.strip(), *opts, progress=progress)


def retexture(mesh_file, image, remove_bg, progress=gr.Progress(track_tqdm=True)):
    if mesh_file is None or image is None:
        raise gr.Error("ارفع المجسم وصورة مرجعية.")
    try:
        res = pipeline.retexture(mesh_file, image, remove_bg, lambda p, d: progress(p, desc=d))
    except Exception as e:  # noqa: BLE001
        log.error(traceback.format_exc())
        raise gr.Error(_friendly_error(e))
    return res["preview"], res["files"], f"✅ تم التلوين — محفوظ في {res['job_dir']}"


def edit_mesh(mesh_file, clean, holes, smooth, faces, size_mm, formats):
    if mesh_file is None:
        raise gr.Error("ارفع ملف مجسم (GLB / OBJ / STL / PLY).")
    res = pipeline.edit(mesh_file, clean=clean, holes=holes, smooth=smooth, faces=faces,
                        size_mm=size_mm, formats=formats)
    b, a = res["before"], res["stats"]
    info = f"قبل: {b['faces']:,} مضلع ← بعد: **{a['faces']:,}** مضلع • المقاس: {a['size']}"
    return res["preview"], res["files"], info


def blender_tool(mesh_file, do_quad, quad_faces, do_rig, progress=gr.Progress()):
    if mesh_file is None:
        raise gr.Error("ارفع ملف مجسم أو اختر واحداً من السجل.")
    if not (do_quad or do_rig):
        raise gr.Error("اختر Quad أو هيكل عظمي.")
    files, preview, notes = [], None, []
    path = mesh_file
    try:
        if do_quad:
            progress(0.2, desc="تحويل إلى Quad عبر Blender…")
            q = pipeline.quad(path, int(quad_faces or 10000))
            files += q["files"]
            preview = path = q["preview"]
            notes.append(f"Quad: **{q['stats']['faces']:,}** وجه، منها {q['stats']['quads']:,} مربع")
        if do_rig:
            progress(0.6, desc="بناء الهيكل العظمي عبر Blender…")
            r = pipeline.rig(path)
            files += r["files"]
            preview = r["preview"]
            notes.append(f"🦴 هيكل عظمي: **{r['stats']['bones']}** عظمة — جاهز للتصدير إلى Unity/Unreal/Mixamo (FBX)")
    except Exception as e:  # noqa: BLE001
        log.error(traceback.format_exc())
        raise gr.Error(f"Blender: {str(e)[:400]}")
    return preview, files, "✅ " + "  \n".join(notes)


def last_model():
    for name in _jobs():
        d = config.OUTPUT_DIR / name
        for cand in ("model_textured.glb", "model.glb"):
            if (d / cand).exists():
                return str(d / cand)
    return None


def blender_status():
    from . import blender
    exe = blender.find_blender()
    if not exe:
        return "❌ **Blender غير مثبّت** — ثبّته مجاناً من blender.org (4.2 أو أحدث) أو حدّد مساره."
    return f"✅ Blender: `{exe}` — الإصدار {blender.blender_version(exe) or '?'}"


def save_blender_path(path):
    from . import blender
    path = (path or "").strip().strip('"')
    if path and not Path(path).exists():
        return "❌ المسار غير موجود"
    blender.save_setting("blender_path", path)
    return blender_status()


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
    lines = [f"✅ كرت الشاشة: **{info['name']}** — الذاكرة: **{info['vram_gb']} GB**", "", "**النماذج:**"]
    try:
        for name, ok in engine.models_status().items():
            lines.append(f"- {'✅' if ok else '⬇️ لم ينزل بعد'} — {name}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- تعذّر الفحص: {e}")
    return "\n".join(lines)


def connect_claude():
    from . import claude_setup
    done = [claude_setup.update(p) for p in claude_setup.config_paths()]
    return ("✅ تم الربط مع Claude:\n" + "\n".join(f"- `{d}`" for d in done) +
            "\n\nأغلق تطبيق Claude بالكامل (من أيقونة شريط المهام ← Quit) ثم افتحه من جديد.")


def run_sac_check():
    from . import sac_check
    return sac_check.report_markdown()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")
    with gr.Blocks(title=config.APP_TITLE, css=CSS, theme=theme, analytics_enabled=False,
                   head=TIMER_JS, fill_width=True) as demo:
        gr.Markdown(f"### 🧊 {config.APP_TITLE} <span class='hint'>— محلياً على جهازك ومجاناً</span>",
                    elem_id="title")

        with gr.Tabs():
            # ---- multiview ----
            with gr.Tab("📸 من 4 صور"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("صوّر القطعة على خلفية بسيطة. **الأمام مطلوبة** والبقية تزيد الدقة.",
                                    elem_classes="hint")
                        with gr.Row(elem_classes="compact"):
                            mv = {v: gr.Image(type="pil", image_mode="RGBA", label=VIEW_LABELS[v], height=150,
                                              min_width=110, elem_classes="img-in", show_download_button=False)
                                  for v in ("front", "back", "left", "right")}
                        btn, opts = _generate_block()
                    with gr.Column(scale=6):
                        out = _outputs()
                btn.click(gen_multiview, [mv["front"], mv["back"], mv["left"], mv["right"], *opts], list(out))

            # ---- single image ----
            with gr.Tab("🖼️ من صورة واحدة"):
                with gr.Row():
                    with gr.Column(scale=5):
                        img = gr.Image(type="pil", image_mode="RGBA", label="الصورة", height=240)
                        btn1, opts1 = _generate_block()
                    with gr.Column(scale=6):
                        out1 = _outputs()
                btn1.click(gen_single, [img, *opts1], list(out1))

            # ---- text ----
            with gr.Tab("✍️ من نص"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(label="وصف المجسم", lines=2,
                                            placeholder="A wooden chair with curved legs",
                                            info="اكتب الوصف بالإنجليزية لأفضل نتيجة. "
                                                 "يتطلب تنزيل نموذج إضافي (~10GB) أول مرة.")
                        btn2, opts2 = _generate_block()
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
                        timg = gr.Image(type="pil", image_mode="RGBA", label="الصورة المرجعية", height=200)
                        trbg = gr.Checkbox(value=True, label="إزالة الخلفية تلقائياً")
                        btn3 = gr.Button("🎨 تلوين", variant="primary")
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

            # ---- blender: quad + rig ----
            with gr.Tab("🦴 Quad وهيكل عظمي"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("يفتح **Blender** في الخلفية، ينفّذ، يحفظ ويغلق تلقائياً.", elem_classes="hint")
                        bmesh_in = gr.File(label="المجسم", file_types=[".glb", ".obj", ".fbx", ".stl", ".ply"],
                                           type="filepath")
                        use_last = gr.Button("استخدم آخر مجسم ولّدته", size="sm")
                        with gr.Row(elem_classes="compact"):
                            bq = gr.Checkbox(value=True, label="مربعات Quad")
                            bqf = gr.Slider(500, 50000, value=10000, step=500, label="عدد الأوجه")
                        br = gr.Checkbox(value=True, label="🦴 هيكل عظمي (شخصية واقفة T-pose أو A-pose)")
                        bbtn = gr.Button("تنفيذ", variant="primary")
                    with gr.Column(scale=6):
                        bout = _outputs()
                use_last.click(last_model, None, bmesh_in)
                bbtn.click(blender_tool, [bmesh_in, bq, bqf, br], list(bout))

            # ---- history ----
            with gr.Tab("🗂️ السجل"):
                with gr.Row():
                    hsel = gr.Dropdown(choices=_jobs(), label="الأعمال السابقة", scale=4)
                    hbtn = gr.Button("تحديث", scale=1)
                with gr.Row():
                    with gr.Column(scale=6):
                        hmodel = gr.Model3D(label="المعاينة", height=430)
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
                bl_status = gr.Markdown()
                with gr.Row():
                    bl_path = gr.Textbox(label="مسار blender.exe (اختياري — يُكتشف تلقائياً)", scale=4,
                                         placeholder=r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
                    bl_save = gr.Button("حفظ", scale=1)
                bl_save.click(save_blender_path, bl_path, bl_status)
                demo.load(blender_status, None, bl_status)
                claude_btn = gr.Button("🤖 ربط البرنامج مع تطبيق Claude")
                claude_btn.click(connect_claude, None, sac_out)
                unload = gr.Button("تفريغ ذاكرة كرت الشاشة")
                unload.click(lambda: (engine.MANAGER.unload(), "تم التفريغ ✅")[1], None, sac_out)
                sac_btn.click(run_sac_check, None, sac_out)
                demo.load(device_text, None, dev)

        gr.Markdown("<span class='hint'>يعمل بنماذج Tencent Hunyuan3D 2.0 مفتوحة المصدر "
                    "(رخصة Tencent Hunyuan 3D 2.0 Community).</span>")
    return demo
