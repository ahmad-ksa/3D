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
