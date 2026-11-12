"""Storyboard file: what the user (or Claude) uploads to make a video.

Either a .zip (storyboard.json + the images / voice files), or a storyboard.json whose file
names are relative to the json's folder:

{
  "title": "My clip",
  "resolution": "480p",            # optional: "480p" (default, faster) or "720p"
  "scenes": [
    {"image": "01.png", "prompt": "the girl waves and smiles", "audio": "01.wav"},
    {"image": "02.png", "prompt": "camera slowly zooms in", "seconds": 4}
  ]
}
"prompt" may also be called "description"; "audio" may be "voice". A scene with audio lasts as
long as its audio and the character's lips follow it; without audio "seconds" (default 5) is used.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
RESOLUTIONS = {"480p": 832 * 480, "720p": 1280 * 704}
DEFAULT_SECONDS = 5.0
MIN_SECONDS, MAX_SECONDS = 1.0, 8.0     # one generated motion clip (longer voice -> clip is looped)


class StoryboardError(ValueError):
    pass


@dataclass
class Scene:
    index: int
    image: Path
    prompt: str
    audio: Path | None
    seconds: float
    lipsync: bool = True     # False: the audio is only music/sound under the scene


@dataclass
class Storyboard:
    title: str
    resolution: str
    scenes: list
    root: Path


def _pick(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _resolve(root: Path, name, kinds: set, what: str, n: int) -> Path:
    p = Path(str(name))
    if not p.is_absolute():
        p = (root / p)
    p = p.resolve()
    if not p.exists():
        raise StoryboardError(f"المشهد {n}: ملف {what} غير موجود: {name}")
    if p.suffix.lower() not in kinds:
        raise StoryboardError(f"المشهد {n}: نوع ملف {what} غير مدعوم: {p.name}")
    return p


def _extract(zip_path: Path, dest: Path) -> Path:
    """Unzip safely (no paths outside dest) and return the folder that holds storyboard.json."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise StoryboardError("ملف zip غير صالح.")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    found = sorted(dest.rglob("*.json"), key=lambda p: (p.name.lower() != "storyboard.json", len(p.parts)))
    if not found:
        raise StoryboardError("ملف الـzip لا يحتوي على storyboard.json")
    return found[0]


def load(path, work_dir: Path | None = None) -> Storyboard:
    """Read a storyboard .zip / .json. Zips are extracted into work_dir/storyboard."""
    path = Path(path)
    if not path.exists():
        raise StoryboardError(f"الملف غير موجود: {path}")
    if path.suffix.lower() == ".zip":
        if work_dir is None:
            raise StoryboardError("work_dir مطلوب لفك ملف zip")
        path = _extract(path, Path(work_dir) / "storyboard")
    if path.suffix.lower() != ".json":
        raise StoryboardError("ارفع ملف storyboard.zip أو storyboard.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        raise StoryboardError(f"storyboard.json غير صالح: {e}") from e
    if isinstance(data, list):
        data = {"scenes": data}
    raw = data.get("scenes") or data.get("مشاهد")
    if not isinstance(raw, list) or not raw:
        raise StoryboardError("لا توجد مشاهد (scenes) في الستوري بورد.")
    res = str(data.get("resolution", "480p")).lower()
    if res not in RESOLUTIONS:
        res = "480p"
    root = path.parent
    scenes = []
    for i, s in enumerate(raw, 1):
        if not isinstance(s, dict):
            raise StoryboardError(f"المشهد {i} غير صالح.")
        img = _pick(s, "image", "صورة")
        if not img:
            raise StoryboardError(f"المشهد {i}: الصورة مطلوبة.")
        prompt = str(_pick(s, "prompt", "description", "وصف", default="")).strip()
        audio = _pick(s, "audio", "voice", "صوت")
        try:
            secs = float(_pick(s, "seconds", "duration", "ثواني", default=DEFAULT_SECONDS))
        except (TypeError, ValueError):
            secs = DEFAULT_SECONDS
        scenes.append(Scene(
            index=i,
            image=_resolve(root, img, IMAGE_EXT, "الصورة", i),
            prompt=prompt or "subtle natural movement, gentle breathing, cinematic",
            audio=_resolve(root, audio, AUDIO_EXT, "الصوت", i) if audio else None,
            seconds=min(MAX_SECONDS, max(MIN_SECONDS, secs)),
            lipsync=bool(_pick(s, "lipsync", default=True)),
        ))
    return Storyboard(title=str(data.get("title") or "video"), resolution=res, scenes=scenes, root=root)


def frame_size(image_w: int, image_h: int, resolution: str) -> tuple:
    """Video size with the image's aspect ratio, about the target area, multiples of 32."""
    area = RESOLUTIONS.get(resolution, RESOLUTIONS["480p"])
    ratio = image_w / max(image_h, 1)
    h = (area / ratio) ** 0.5
    w = h * ratio
    w = max(256, int(round(w / 32)) * 32)
    h = max(256, int(round(h / 32)) * 32)
    return w, h


def write_template(dest: Path) -> Path:
    """An example storyboard.zip (2 scenes) the user can copy."""
    from PIL import Image, ImageDraw
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / "_template"
    tmp.mkdir(exist_ok=True)
    for n, color in ((1, (90, 140, 200)), (2, (200, 140, 90))):
        im = Image.new("RGB", (832, 480), color)
        ImageDraw.Draw(im).text((20, 20), f"scene {n} - put your image here", fill=(255, 255, 255))
        im.save(tmp / f"0{n}.png")
    sb = {
        "title": "example",
        "resolution": "480p",
        "scenes": [
            {"image": "01.png", "prompt": "the character waves and smiles", "audio": "01.wav"},
            {"image": "02.png", "prompt": "slow camera zoom in, soft light", "seconds": 4},
        ],
    }
    (tmp / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=2), encoding="utf-8")
    import wave
    with wave.open(str(tmp / "01.wav"), "wb") as w:  # 2 s of silence as a placeholder voice
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 32000)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.iterdir()):
            z.write(f, f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


_SECS = re.compile(r"[|\[(（]\s*(\d+(?:\.\d+)?)\s*(?:s|sec|ث|ثانية|ثواني)?\s*[\])）]?\s*$", re.I)
_NUM = re.compile(r"^\s*(?:(?:scene|المشهد|مشهد)\s*)?\d*\s*[-.:)،]*\s*", re.I)


def parse_story_text(text: str) -> list:
    """One scene per non-empty line: "description | 5" (seconds optional).
    Leading numbering like "1." or "المشهد 2:" is removed."""
    scenes = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        secs = None
        m = _SECS.search(line)
        if m:
            secs = float(m.group(1))
            line = line[:m.start()].strip()
        line = _NUM.sub("", line, count=1).strip() or line
        scenes.append((line, secs))
    return scenes


_WHO = re.compile(r"^\s*[\[【]\s*([^\]】]+?)\s*[\]】]\s*")


def _norm(t: str) -> str:
    return re.sub(r"[\s_\-.]+", "", t).lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")


def _image_for(who, imgs: list, i: int, tagged: bool) -> Path:
    """[name] tag -> image with that name in its file name; untagged stories use the images in order."""
    if who:
        w = _norm(who)
        for p in imgs:
            st = _norm(p.stem)
            if w and (w in st or (st and st in w)):
                return p
        return imgs[0]
    return imgs[0] if tagged else imgs[min(i, len(imgs) - 1)]


def from_inputs(images: list, story_text: str, audio, lipsync: bool, work_dir: Path,
                title: str = "video") -> Path:
    """Build a storyboard.json from the video tab's boxes:
    images  - one per scene, in order (fewer images than scenes: the last image is reused)
    story   - one line per scene (motion description, optional "| seconds")
    audio   - optional song / voice for the WHOLE clip: cut into one piece per scene
    lipsync - the character's lips follow the audio (singing / talking)."""
    from . import media
    imgs = [Path(i) for i in (images or []) if i]
    if not imgs:
        raise StoryboardError("ارفع صورة واحدة على الأقل.")
    lines = parse_story_text(story_text)
    lines += [("", None)] * max(0, len(imgs) - len(lines))   # more images than lines: one scene each
    work_dir = Path(work_dir) / "storyboard"
    work_dir.mkdir(parents=True, exist_ok=True)
    scenes = []
    seg_lens = [s for _, s in lines]
    if audio:
        total = media.duration(audio)
        if total <= 0:
            raise StoryboardError("تعذّر قراءة ملف الصوت.")
        fixed = sum(x for x in seg_lens if x)
        free = [i for i, x in enumerate(seg_lens) if not x]
        rest = max(0.0, total - fixed)
        for i in free:
            seg_lens[i] = rest / len(free) if free else 0
        start = 0.0
    tagged = any(_WHO.match(d) for d, _ in lines)
    for i, (desc, secs) in enumerate(lines):
        m = _WHO.match(desc)
        if m:   # "[ريهاد] prompt": the image whose file name has that name
            desc = desc[m.end():].strip()
        img = _image_for(m.group(1) if m else None, imgs, i, tagged)
        sc = {"image": str(img), "prompt": desc}
        if audio:
            length = seg_lens[i]
            if length < 0.3 or start >= total - 0.05:
                break
            seg = work_dir / f"audio_{i + 1:02d}.wav"
            media.cut_audio(audio, seg, start, length)
            start += length
            sc["audio"] = seg.name
            sc["lipsync"] = bool(lipsync)
        elif secs:
            sc["seconds"] = secs
        scenes.append(sc)
    data = {"title": title, "scenes": scenes}
    path = work_dir / "storyboard.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
