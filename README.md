<div dir="rtl">

# 🧊 استوديو المجسمات ثلاثية الأبعاد (Studio3D)

برنامج ويندوز يولّد مجسمات ثلاثية الأبعاد بالذكاء الاصطناعي **على جهازك مباشرة** — مجاني، بدون اشتراك، وبدون رفع صورك لأي سيرفر.

## المميزات

| الميزة | الوصف |
|---|---|
| 📸 **من 4 صور** | صور أمام / خلف / يمين / يسار ← مجسم دقيق (Hunyuan3D-2mv) |
| 🖼️ **من صورة واحدة** | صورة واحدة ← مجسم كامل (Hunyuan3D-2) |
| ✍️ **من نص** | وصف بالإنجليزية ← صورة ← مجسم |
| 🎨 **تلوين بالذكاء الاصطناعي** | Texture تلقائي للمجسم المولَّد أو لأي مجسم ترفعه |
| ✂️ **إزالة الخلفية** | تلقائياً من الصور |
| 🧹 **تنظيف** | إزالة الشوائب والأوجه التالفة وسد الثقوب |
| 🔻 **Low-poly** | تحديد عدد المضلعات (للألعاب والجوال) |
| 🪶 **تنعيم السطح** | Taubin smoothing |
| 📏 **مقاس حقيقي** | تحديد الطول بالملّيمتر ووضع المجسم على الأرض (للطباعة) |
| 💾 **التصدير** | GLB ، OBJ ، STL ، PLY ، 3MF |
| 👁️ **معاينة 3D** | تدوير وتكبير داخل البرنامج |
| 🗂️ **السجل** | كل الأعمال محفوظة مع إعداداتها |
| ⚡ **3 مستويات جودة** | سريع / متوازن / عالي الجودة |
| 🛡️ **متوافق مع Smart App Control** | لا يحتاج إطفاء الحماية (التفاصيل بالأسفل) |

> غير متوفر حالياً (موجود في Meshy/Tripo السحابية): التحريك (Rigging/Animation) وخامات PBR الكاملة.

## المتطلبات

- ويندوز 10/11 (64-bit)
- كرت **NVIDIA** بذاكرة **8GB** أو أكثر (مُختبر لـ RTX 5050 / 4060) + تعريف حديث (570 أو أحدث لسلسلة RTX 50)
- ذاكرة RAM: ‏16GB (يفضل 32GB للتلوين)
- مساحة: حوالي **25GB** (البرنامج + النماذج)
- إنترنت **أول مرة فقط** لتنزيل النماذج

## التثبيت (أمر واحد)

1. افتح قائمة ابدأ ← اكتب **PowerShell** ← افتحه (بدون صلاحيات مسؤول).
2. الصق هذا الأمر واضغط Enter:

</div>

```powershell
irm https://raw.githubusercontent.com/ahmad-ksa/3D/main/install.ps1 | iex
```

<div dir="rtl">

3. انتظر حتى ينتهي (≈ 10–30 دقيقة حسب سرعة الإنترنت). سيظهر اختصار **Studio3D** على سطح المكتب.

**للتحديث:** شغّل نفس الأمر مرة أخرى.
**لإزالة البرنامج:** احذف المجلد `%LOCALAPPDATA%\Studio3D` والاختصار من سطح المكتب.

## تنزيل النماذج (مرة واحدة)

البرنامج ينزّل كل نموذج تلقائياً عند أول استخدام له. لتنزيلها **كلها مسبقاً** (≈ 15GB):
اضغط **Win + R** ← الصق السطر التالي ← Enter، واترك النافذة حتى تظهر «اكتمل»:

</div>

```
%LOCALAPPDATA%\Studio3D\Studio3D.cmd --download
```

<div dir="rtl">

- لتنزيل نموذج «من نص» أيضاً (≈ 10GB إضافية): استبدل `--download` بـ `--download all`.
- التنزيل **يكمل من حيث توقف** إذا انقطع — أعد تشغيل نفس الأمر.
- الوقت يعتمد على سرعة الإنترنت: بسرعة 2MB/s تقريباً ساعتان.
- مكان النماذج: `%LOCALAPPDATA%\Studio3D\models`

### مصادر النماذج (رسمية — Hugging Face)

| النموذج | الاستخدام | الرابط |
|---|---|---|
| Hunyuan3D-2mv (turbo) | من 4 صور | https://huggingface.co/tencent/Hunyuan3D-2mv |
| Hunyuan3D-2 (turbo + VAE + Paint + Delight) | من صورة واحدة + التلوين | https://huggingface.co/tencent/Hunyuan3D-2 |
| HunyuanDiT v1.1 Distilled | من نص (اختياري) | https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled |
| U²-Net (rembg) | إزالة الخلفية | https://github.com/danielgatis/rembg |
| الكود المصدري لـ Hunyuan3D | مكتبة التوليد | https://github.com/Tencent-Hunyuan/Hunyuan3D-2 |

### روابط التثبيت المستخدمة
| المكوّن | المصدر |
|---|---|
| Python 3.11.9 الرسمي (موقّع) | https://www.nuget.org/packages/python/3.11.9 |
| PyTorch 2.7.1 + CUDA 12.8 | https://download.pytorch.org/whl/cu128 |
| بقية المكتبات | PyPI — بإصدارات مثبّتة في `requirements.txt` |
| تعريف NVIDIA | https://www.nvidia.com/Download/index.aspx |

## الاستخدام

1. افتح **Studio3D** من سطح المكتب — تفتح نافذة سوداء (اتركها مفتوحة) ثم نافذة البرنامج.
2. تبويب **«من 4 صور»** ← ارفع الصور ← اختر الجودة ← **توليد المجسم**.
3. حمّل الملفات من «ملفات التحميل» أو من تبويب «السجل».

### نصائح لأفضل نتيجة
- خلفية بسيطة بلون واحد وإضاءة متساوية بدون ظلال قوية.
- القطعة في وسط الصورة وكاملة (غير مقصوصة).
- نفس المسافة والارتفاع لكل الصور الأربع.
- جرّب «بذرة» مختلفة إذا لم تعجبك النتيجة.

### أوقات تقريبية على RTX 5050 (8GB)
| العملية | الوقت |
|---|---|
| شكل — سريع | ~30 ثانية |
| شكل — متوازن | ~1 دقيقة |
| شكل — عالي الجودة | 3–6 دقائق |
| تلوين | 3–8 دقائق |

(أول تشغيل لكل نوع أبطأ بسبب تنزيل النموذج وتحميله.)

## 🤖 الربط مع Claude (تطبيق الكمبيوتر)

البرنامج يعمل كـ **خادم MCP**، فيقدر Claude يولّد ويعدّل المجسمات بنفسه على جهازك.

1. ثبّت تطبيق Claude للكمبيوتر وسجّل دخولك.
2. الربط يتم تلقائياً أثناء تثبيت Studio3D. إذا ثبّتّ Claude **بعد** Studio3D: افتح Studio3D ← تبويب «الجهاز» ← **«ربط البرنامج مع تطبيق Claude»**.
3. أغلق Claude **بالكامل** (أيقونة شريط المهام ← Quit) ثم افتحه.
4. ضع صورك في مجلد **الصور ← Studio3D** (`Pictures\Studio3D`) واكتب لـ Claude مثلاً:
   - «ولّد مجسم من الصور front.jpg و back.jpg و left.jpg و right.jpg بجودة عالية مع تلوين»
   - «ولّد صندوق كنز Low-poly للعبة بـ 5000 مضلع وصدّره GLB»
   - «قلّل مضلعات آخر مجسم إلى 10000 وصدّره OBJ»

أدوات Claude: `generate_from_images` ، `generate_from_image` ، `generate_from_text` ، `texture_model` ، `edit_model` ، `preview_model` (صورة من 4 زوايا يراها Claude ليحكم على النتيجة) ، `list_recent_models` ، `list_input_images` ، `check_job`.

> البرنامج نفسه لا يحتاج أن يكون مفتوحاً — Claude يشغّله في الخلفية. لا تولّد من البرنامج ومن Claude في نفس اللحظة (سينتظر أحدهما الآخر).

## 🛡️ التحكم الذكي في التطبيقات (Smart App Control)

البرنامج مصمم ليعمل **والحماية مفعّلة**:
- لا يحتوي البرنامج على أي ملف ‎`.exe` أو ‎`.dll` من صنعنا — فقط ملفات بايثون نصية.
- يعمل عبر **Python الرسمي الموقّع رقمياً** ونافذة **Microsoft Edge** الموقّعة من مايكروسوفت.
- المكوّن الوحيد في Hunyuan3D الذي يحتاج تجميع DLL خاص (`custom_rasterizer`) **أعدنا كتابته بالكامل ببايثون/PyTorch** (`studio3d/vendor`).
- التثبيت بأمر PowerShell (وليس ملفاً منزّلاً من المتصفح)، فلا تُضاف «علامة الإنترنت» التي تجعل الحماية تحظر الملفات.

**للتأكد:** داخل البرنامج ← تبويب **«الجهاز»** ← **«فحص الحماية»**. التفاصيل والحلول في [docs/SMART_APP_CONTROL.md](docs/SMART_APP_CONTROL.md).

## الرخصة

- كود Studio3D: ملك الشركة.
- نماذج الذكاء الاصطناعي: **Tencent Hunyuan 3D 2.0 Community License** — يُسمح بالاستخدام التجاري داخل السعودية والخليج، ولا يسري في الاتحاد الأوروبي وبريطانيا وكوريا الجنوبية، ويلزم ترخيص خاص من Tencent إذا تجاوز عدد المستخدمين مليون شهرياً. راجع [NOTICE.md](NOTICE.md).

</div>

---

### Developer notes

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/f8db63096c8282cb27354314d896feba5ba6ff8a.zip
python -m studio3d            # run
python -m pytest -q tests     # tests (CPU only, models mocked)
python -m studio3d.sac_check  # Smart App Control diagnostics
```

Layout: `studio3d/engine.py` (models, one-at-a-time VRAM manager) · `ui.py` (Arabic Gradio UI) ·
`mesh_tools.py` (clean/decimate/smooth/scale/export) · `vendor/custom_rasterizer` (pure PyTorch
replacement for the CUDA extension) · `install.ps1` (SAC-friendly installer).
