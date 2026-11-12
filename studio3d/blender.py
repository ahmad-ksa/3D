"""Run Blender in the background (no window) for quad remeshing and auto-rigging.

Blender is started, does the job, saves the files and quits by itself.
Blender is signed by the Blender Foundation, so Smart App Control allows it.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config

SCRIPTS = Path(__file__).resolve().parent / "blender_scripts"


class BlenderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Settings (blender path is remembered)
# ---------------------------------------------------------------------------
def _settings_path() -> Path:
    return config.DATA_ROOT / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_setting(key, value):
    s = load_settings()
    s[key] = value
    _settings_path().parent.mkdir(parents=True, exist_ok=True)
    _settings_path().write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Finding Blender
# ---------------------------------------------------------------------------
def _version_key(path: str):
    m = re.search(r"(\d+)\.(\d+)", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def candidates() -> list[str]:
    found = []
    custom = load_settings().get("blender_path") or os.environ.get("STUDIO3D_BLENDER")
    if custom:
        found.append(custom)
    if sys.platform == "win32":
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 os.environ.get("LOCALAPPDATA", "")]
        pats = []
        for r in roots:
            if r:
                pats += [os.path.join(r, "Blender Foundation", "Blender*", "blender.exe"),
                         os.path.join(r, "Programs", "Blender Foundation", "Blender*", "blender.exe"),
                         os.path.join(r, "Steam", "steamapps", "common", "Blender", "blender.exe")]
        for drive in "CDEF":
            pats.append(f"{drive}:\\SteamLibrary\\steamapps\\common\\Blender\\blender.exe")
        hits = []
        for p in pats:
            hits += glob.glob(p)
        found += sorted(hits, key=_version_key, reverse=True)
        store = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", "blender.exe")
        if os.path.exists(store):
            found.append(store)
    elif sys.platform == "darwin":
        found.append("/Applications/Blender.app/Contents/MacOS/Blender")
    which = shutil.which("blender")
    if which:
        found.append(which)
    return [p for p in dict.fromkeys(found) if p and os.path.exists(p)]


def find_blender() -> str | None:
    c = candidates()
    return c[0] if c else None


def _is_python(exe: str) -> bool:
    return Path(exe).stem.lower().startswith("python")


def blender_version(exe: str | None = None) -> str | None:
    exe = exe or find_blender()
    if not exe:
        return None
    if _is_python(exe):
        return "bpy module"
    try:
        r = subprocess.run([exe, "-b", "--factory-startup", "--version"], capture_output=True, text=True,
                           timeout=120, **_no_window())
        m = re.search(r"Blender (\d+\.\d+(?:\.\d+)?)", r.stdout)
        return m.group(1) if m else None
    except Exception:  # noqa: BLE001
        return None


def _no_window() -> dict:
    if sys.platform == "win32":
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


# ---------------------------------------------------------------------------
# Running scripts
# ---------------------------------------------------------------------------
def run_script(name: str, args: dict, log_dir: str | Path | None = None, timeout: int = 1800) -> dict:
    exe = find_blender()
    if not exe:
        raise BlenderError("لم يتم العثور على Blender. ثبّته من blender.org أو حدّد مكانه في تبويب «الجهاز».")
    script = SCRIPTS / f"{name}.py"
    with tempfile.TemporaryDirectory(prefix="studio3d_blender_") as tmp:
        args = dict(args, result=os.path.join(tmp, "result.json"))
        args_path = os.path.join(tmp, "args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False)
        if _is_python(exe):  # standalone bpy module (used by tests)
            cmd = [exe, str(script), "--", args_path]
        else:
            cmd = [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
                   "--", args_path]
        from . import engine
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", **_no_window())
        waited = 0.0
        while True:
            try:
                out, err = proc.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                waited += 1
                if engine.CANCEL.is_set() or waited > timeout:
                    proc.kill()
                    try:
                        proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    if engine.CANCEL.is_set():
                        raise engine.Cancelled("تم إيقاف التوليد")
                    raise BlenderError(f"Blender لم ينتهِ خلال {timeout // 60} دقيقة")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / f"blender_{name}.log").write_text(
                (out or "") + "\n--- stderr ---\n" + (err or ""), encoding="utf-8")
        result_file = Path(args["result"])
        if not result_file.exists():  # the bpy module may crash at interpreter exit AFTER writing
            tail = (err or out or "")[-600:]
            raise BlenderError(f"Blender توقف بدون نتيجة (exit {proc.returncode}): {tail}")
        result = json.loads(result_file.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise BlenderError(result.get("error", "unknown Blender error"))
    return result


def quad_remesh(model_path: str, out_dir: str, target_faces: int = 10000, bake_size: int = 2048,
                name: str = "model_quad", formats=("obj", "fbx", "glb")) -> dict:
    return run_script("quad_remesh", {"input": str(model_path), "outdir": str(out_dir), "name": name,
                                      "target_faces": int(target_faces), "bake_size": int(bake_size),
                                      "formats": list(formats)}, log_dir=out_dir)


def auto_rig(model_path: str, out_dir: str, name: str = "model_rigged", formats=("fbx", "glb"),
             max_faces: int = 60000) -> dict:
    return run_script("auto_rig", {"input": str(model_path), "outdir": str(out_dir), "name": name,
                                   "formats": list(formats), "max_faces": int(max_faces)}, log_dir=out_dir)


def lip_sync(model_path: str, audio_path: str, out_dir: str, name: str = "model_talking",
             max_open_degrees: float = 5, fps: int = 30, mouth: dict | None = None) -> dict:
    args = {"input": str(model_path), "audio": str(audio_path), "outdir": str(out_dir),
            "name": name, "max_open_degrees": float(max_open_degrees), "fps": int(fps)}
    if mouth:
        mouth = dict(mouth)
        if mouth.get("eyes"):
            args["eyes"] = mouth.pop("eyes")
            args["skin"] = mouth.pop("skin", None)
        args["mouth"] = mouth
    return run_script("lip_sync", args, log_dir=out_dir)


def open_gui(model_path: str):
    """Open a model in a normal Blender window (for manual work). Returns immediately."""
    exe = find_blender()
    if not exe or _is_python(exe):
        raise BlenderError("لم يتم العثور على Blender.")
    subprocess.Popen([exe, "--factory-startup", "--python", str(SCRIPTS / "open_model.py"), "--",
                      str(model_path)], close_fds=True)
