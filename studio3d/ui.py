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
.audio-in .upload-container, .audio-in .wrap {min-height: 0 !important;}
.audio-in {max-height: 170px;}
.img-in .upload-container, .img-in .image-frame {min-height: 0 !important;}
/* elapsed time as mm:ss (see TIMER_JS) */
.progress-text[data-mmss] {font-size: 0 !important;}
.progress-text[data-mmss]::after {content: attr(data-mmss); font-size: var(--text-sm);
                                  font-variant-numeric: tabular-nums;}
"""

VIEWER_HEAD = """
<script type="importmap">{"imports": {"three": "/s3d-static/three/three.module.min.js"}}</script>
<script type="module" src="/s3d-static/viewer.js"></script>
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


def _served(path) -> str:
    p = Path(path).resolve()
    try:
        rel = p.relative_to(config.OUTPUT_DIR.resolve()).as_posix()
        return f"/s3d-models/{rel}?v={int(p.stat().st_mtime)}"
    except ValueError:
        return f"/gradio_api/file={p.as_posix()}"


def viewer_html(path=None, stats=None, height=360, audio=None) -> str:
    """HTML for our three.js viewer (colours / clay / wireframe / x-ray / lighting / skeleton)."""
    import html
    import json as _json
    src = ""
    if path:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(config.OUTPUT_DIR.resolve()).as_posix()
            src = f"/s3d-models/{rel}?v={int(p.stat().st_mtime)}"
        except ValueError:
            src = f"/gradio_api/file={p.as_posix()}"
    info = html.escape(_json.dumps({"quads": (stats or {}).get("quads", 0)}), quote=True)
    aud = f' audio="{html.escape(_served(audio))}"' if audio else ""
    return f'<s3d-viewer src="{html.escape(src)}" info="{info}"{aud} style="--s3d-h:{height}px"></s3d-viewer>'


def _outputs():
    model = gr.HTML(viewer_html())
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
    return viewer_html(res["preview"], res["stats"]), res["files"], info


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


# ---------------------------------------------------------------------------
# Character sheet -> 4 views
# ---------------------------------------------------------------------------
_BOX_NAMES = ["أمام", "خلف", "يسار", "يمين"]
_BOX_COLORS = [(91, 91, 214), (214, 91, 91), (40, 160, 90), (230, 150, 20)]


def _draw_boxes(img, pts):
    from PIL import ImageDraw
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 300)
    for i in range(0, len(pts) - 1, 2):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        k = i // 2
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=_BOX_COLORS[k], width=lw)
        d.text((min(x0, x1) + 6, min(y0, y1) + 4), str(k + 1), fill=_BOX_COLORS[k])
    if len(pts) % 2 == 1:
        x, y = pts[-1]
        r = lw * 3
        d.ellipse([x - r, y - r, x + r, y + r], fill=_BOX_COLORS[len(pts) // 2])
    return im


def _next_hint(pts):
    if len(pts) >= 8:
        return "✅ حُدّدت المربعات الأربعة — اضغط **✂️ قسّم ووزّع**"
    k = len(pts) // 2
    corner = "الزاوية الأولى" if len(pts) % 2 == 0 else "الزاوية المقابلة"
    return f"اضغط {corner} لمربع **{k + 1} ({_BOX_NAMES[k]})**"


def sheet_loaded(img, mode):
    if img is None:
        return None, [], gr.update(value=None, visible=False), ""
    manual = mode == "boxes"
    return img, [], gr.update(value=img if manual else None, visible=manual), (_next_hint([]) if manual else "")


def sheet_mode_changed(mode, orig):
    manual = mode == "boxes"
    return (gr.update(value=orig if (manual and orig is not None) else None, visible=manual), [],
            _next_hint([]) if manual else "")


def sheet_click(orig, pts, evt: gr.SelectData):
    if orig is None:
        raise gr.Error("ارفع الصورة أولاً.")
    pts = list(pts or [])
    if len(pts) >= 8:
        pts = []
    pts.append(tuple(int(v) for v in evt.index))
    return pts, _draw_boxes(orig, pts), _next_hint(pts)


def sheet_split(orig, mode, pts):
    from . import sheet as sh
    if orig is None:
        raise gr.Error("ارفع الصورة الكاملة أولاً.")
    try:
        if mode == "boxes":
            if len(pts or []) < 8:
                raise ValueError("حدّد 4 مربعات (زاويتين لكل مربع)")
            boxes = []
            for i in range(0, 8, 2):
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                boxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            views = sh.split(orig, "boxes", boxes)
        else:
            views = sh.split(orig, mode)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(str(e))
    return (views["front"], views["back"], views["left"], views["right"],
            "✅ تم التقسيم والتوزيع — راجع الجهات، واستخدم أزرار ⇄ إذا انعكس شيء")


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
    return viewer_html(res["preview"]), res["files"], f"✅ تم التلوين — محفوظ في {res['job_dir']}"


def edit_mesh(mesh_file, clean, holes, smooth, faces, size_mm, formats):
    if mesh_file is None:
        raise gr.Error("ارفع ملف مجسم (GLB / OBJ / STL / PLY).")
    res = pipeline.edit(mesh_file, clean=clean, holes=holes, smooth=smooth, faces=faces,
                        size_mm=size_mm, formats=formats)
    b, a = res["before"], res["stats"]
    info = f"قبل: {b['faces']:,} مضلع ← بعد: **{a['faces']:,}** مضلع • المقاس: {a['size']}"
    return viewer_html(res["preview"], res["stats"]), res["files"], info


def blender_tool(mesh_file, do_quad, quad_faces, do_rig, progress=gr.Progress()):
    if mesh_file is None:
        raise gr.Error("ارفع ملف مجسم أو اختر واحداً من السجل.")
    if not (do_quad or do_rig):
        raise gr.Error("اختر Quad أو هيكل عظمي.")
    files, preview, notes = [], None, []
    last_stats = {}
    path = mesh_file
    try:
        if do_quad:
            progress(0.2, desc="تحويل إلى Quad عبر Blender…")
            q = pipeline.quad(path, int(quad_faces or 10000))
            files += q["files"]
            preview = path = q["preview"]
            last_stats = q["stats"]
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
    return viewer_html(preview, last_stats), files, "✅ " + "  \n".join(notes)


def talk_tool(mesh_file, audio_file, strength, progress=gr.Progress()):
    if mesh_file is None or audio_file is None:
        raise gr.Error("اختر مجسماً فيه هيكل عظمي وملف صوت.")
    try:
        progress(0.3, desc="تحريك الفم مع الصوت عبر Blender…")
        r = pipeline.lip_sync(mesh_file, audio_file, float(strength))
    except Exception as e:  # noqa: BLE001
        log.error(traceback.format_exc())
        raise gr.Error(f"تعذّر: {str(e)[:400]}")
    st = r["stats"]
    return (viewer_html(r["preview"], audio=r["audio"]), r["files"],
            f"✅ {st['seconds']} ثانية • {st['frames']} إطار — اضغط ▶️ تشغيل في العارض")


def last_rigged():
    from . import library
    for e in library.list_models(with_thumb=False):
        for f in e["files"]:
            if f.endswith("model_rigged.glb"):
                return f
    return None


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


def library_gallery(query=""):
    from . import library
    items = library.list_models(query)
    gallery = [(e["thumb"], e["name"]) for e in items if e["thumb"]]
    ids = [e["id"] for e in items if e["thumb"]]
    return gr.update(value=gallery), ids


def library_select(ids, evt: gr.SelectData):
    from . import library
    if not ids or evt.index is None or evt.index >= len(ids):
        return None, viewer_html(), "", None, ""
    e = library.get(ids[evt.index])
    tags = " • ".join(t for t, on in (("🎨 ملوّن", e["textured"]), ("🟦 Quad", e["quad"]),
                                         ("🦴 هيكل عظمي", e["rigged"])) if on)
    st = e["stats"] or {}
    details = (f"**{e['name']}**  \n{tags}  \n"
               f"<span class='hint'>{st.get('faces', '?'):,} وجه • المعرّف `{e['id']}`</span>"
               if isinstance(st.get("faces"), int) else f"**{e['name']}**  \n{tags}")
    return e["id"], viewer_html(e["preview"], st, height=420, audio=e.get("audio")), e["name"], e["files"], details


def library_rename(model_id, name, query):
    from . import library
    if not model_id:
        raise gr.Error("اختر مجسماً أولاً.")
    library.rename(model_id, name)
    gal, ids = library_gallery(query)
    return gal, ids, "✅ تم تغيير الاسم"


def library_open_blender(model_id):
    from . import library
    if not model_id:
        raise gr.Error("اختر مجسماً أولاً.")
    try:
        path = library.open_in_blender(model_id)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(str(e))
    return f"✅ فُتح في Blender: `{Path(path).name}`"


def library_open_folder(model_id):
    from . import library
    if not model_id:
        raise gr.Error("اختر مجسماً أولاً.")
    library.open_folder(model_id)
    return "📂 تم فتح المجلد"


def library_to_tools(model_id):
    from . import library
    if not model_id:
        raise gr.Error("اختر مجسماً أولاً.")
    return library.get(model_id)["preview"], gr.Tabs(selected="tab_blender")


def library_claude_hint(model_id):
    if not model_id:
        raise gr.Error("اختر مجسماً أولاً.")
    return (f"انسخ هذا لـ Claude:  \n`استخدم المجسم {model_id} من مكتبة Studio3D`  \n"
            "<span class='hint'>Claude يقدر يبحث في المكتبة ويستخدم أي مجسم أو ينسخه لمجلد مشروعك.</span>")


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
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan",
                           font=[gr.themes.Font("Segoe UI"), gr.themes.Font("Tahoma"), "sans-serif"],
                           font_mono=[gr.themes.Font("Consolas"), "monospace"])
    with gr.Blocks(title=config.APP_TITLE, css=CSS, theme=theme, analytics_enabled=False,
                   head=VIEWER_HEAD + TIMER_JS, fill_width=True) as demo:
        gr.Markdown(f"### 🧊 {config.APP_TITLE} <span class='hint'>— محلياً على جهازك ومجاناً</span>",
                    elem_id="title")

        with gr.Tabs() as tabs:
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
                        with gr.Row(elem_classes="compact"):
                            sw_fb = gr.Button("⇄ أمام / خلف", size="sm")
                            sw_lr = gr.Button("⇄ يسار / يمين", size="sm")
                        with gr.Accordion("📄 عندك صورة واحدة فيها الجهات الأربع؟ قسّمها تلقائياً", open=False):
                            with gr.Row(elem_classes="compact"):
                                sheet_in = gr.Image(type="pil", label="الصورة الكاملة", height=170, scale=3,
                                                    show_download_button=False)
                                with gr.Column(scale=2, min_width=170):
                                    sheet_mode = gr.Radio([("تلقائي", "auto"), ("صف من 4", "row"),
                                                           ("شبكة 2×2", "grid"), ("تحديد يدوي", "boxes")],
                                                          value="auto", label="طريقة التقسيم")
                                    sheet_btn = gr.Button("✂️ قسّم ووزّع", variant="secondary")
                            sheet_view = gr.Image(type="pil", label="التحديد اليدوي: اضغط زاويتين لكل مربع",
                                                  interactive=False, height=260, visible=False,
                                                  show_download_button=False)
                            sheet_msg = gr.Markdown()
                        sheet_orig = gr.State(None)
                        sheet_pts = gr.State([])
                        btn, opts = _generate_block()
                    with gr.Column(scale=6):
                        out = _outputs()
                btn.click(gen_multiview, [mv["front"], mv["back"], mv["left"], mv["right"], *opts], list(out))
                four = [mv["front"], mv["back"], mv["left"], mv["right"]]
                sw_fb.click(lambda f, b, l, r: (b, f, l, r), four, four)
                sw_lr.click(lambda f, b, l, r: (f, b, r, l), four, four)
                sheet_in.change(sheet_loaded, [sheet_in, sheet_mode], [sheet_orig, sheet_pts, sheet_view, sheet_msg])
                sheet_mode.change(sheet_mode_changed, [sheet_mode, sheet_orig],
                                  [sheet_view, sheet_pts, sheet_msg])
                sheet_view.select(sheet_click, [sheet_orig, sheet_pts], [sheet_pts, sheet_view, sheet_msg])
                sheet_btn.click(sheet_split, [sheet_orig, sheet_mode, sheet_pts], [*four, sheet_msg])

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
            with gr.Tab("🦴 Quad وهيكل عظمي", id="tab_blender"):
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

            # ---- talk ----
            with gr.Tab("👄 كلام"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("الشخصية تحرّك فمها مع الصوت (تحتاج هيكل عظمي من Studio3D).",
                                    elem_classes="hint")
                        tmesh2 = gr.File(label="المجسم بالهيكل العظمي (GLB / FBX)", file_types=[".glb", ".fbx"],
                                         type="filepath", height=110)
                        last_r = gr.Button("استخدم آخر شخصية بهيكل عظمي", size="sm")
                        taudio = gr.Audio(label="الصوت (wav / mp3) — ارفع ملفاً أو سجّل بصوتك", type="filepath",
                                          sources=["upload", "microphone"], elem_classes="audio-in")
                        tstr = gr.Slider(4, 20, value=10, step=1, label="قوة فتح الفم (درجات)")
                        tbtn = gr.Button("👄 تحريك الفم", variant="primary")
                    with gr.Column(scale=6):
                        tout = _outputs()
                last_r.click(last_rigged, None, tmesh2)
                tbtn.click(talk_tool, [tmesh2, taudio, tstr], list(tout))

            # ---- library ----
            with gr.Tab("📚 مكتبتي"):
                lib_ids = gr.State([])
                lib_sel = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row(elem_classes="compact"):
                            lib_q = gr.Textbox(placeholder="بحث بالاسم أو الوصف…", show_label=False, scale=4)
                            lib_refresh = gr.Button("🔄", scale=1, min_width=50)
                        lib_gal = gr.Gallery(columns=4, height=470, show_label=False, allow_preview=False,
                                             object_fit="contain")
                    with gr.Column(scale=6):
                        lib_view = gr.HTML(viewer_html(height=420))
                        lib_details = gr.Markdown()
                        with gr.Row(elem_classes="compact"):
                            lib_name = gr.Textbox(label="الاسم", scale=4)
                            lib_save = gr.Button("حفظ الاسم", scale=1, min_width=90)
                        with gr.Row(elem_classes="compact"):
                            b_blend = gr.Button("🟧 فتح في Blender", size="sm")
                            b_folder = gr.Button("📂 فتح المجلد", size="sm")
                            b_tools = gr.Button("🦴 Quad / هيكل", size="sm")
                            b_claude = gr.Button("🤖 لـ Claude", size="sm")
                        lib_msg = gr.Markdown()
                        lib_files = gr.File(label="الملفات", file_count="multiple", elem_classes="files-out")
                lib_refresh.click(library_gallery, lib_q, [lib_gal, lib_ids])
                lib_q.submit(library_gallery, lib_q, [lib_gal, lib_ids])
                lib_gal.select(library_select, lib_ids, [lib_sel, lib_view, lib_name, lib_files, lib_details])
                lib_save.click(library_rename, [lib_sel, lib_name, lib_q], [lib_gal, lib_ids, lib_msg])
                b_blend.click(library_open_blender, lib_sel, lib_msg)
                b_folder.click(library_open_folder, lib_sel, lib_msg)
                b_claude.click(library_claude_hint, lib_sel, lib_msg)
                b_tools.click(library_to_tools, lib_sel, [bmesh_in, tabs])
                demo.load(library_gallery, lib_q, [lib_gal, lib_ids])

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
