"""Smart App Control (SAC) diagnostics.

Studio3D ships no self-compiled .exe/.dll/.pyd — only .py files run by the
official, signed python.exe. The only thing SAC could still block is a binary
inside a third-party library (e.g. a .pyd from PyPI with no reputation yet).
This tool finds out exactly which one, if any.

Run:  python -m studio3d.sac_check
"""
from __future__ import annotations

import subprocess
import sys

MODULES = [
    "numpy", "PIL", "cv2", "torch", "torchvision", "scipy", "trimesh", "pymeshlab", "xatlas",
    "onnxruntime", "rembg", "safetensors", "tokenizers", "transformers", "diffusers",
    "gradio", "uvicorn", "hy3dgen.shapegen", "hy3dgen.texgen",
]

_STATES = {0: "مطفأ (Off)", 1: "مفعّل (On)", 2: "وضع التقييم (Evaluation)"}


def sac_state():
    if sys.platform != "win32":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CI\Policy") as k:
            val, _ = winreg.QueryValueEx(k, "VerifiedAndReputablePolicyState")
            return int(val)
    except OSError:
        return None


def check_module(name: str):
    """Import in a child process so a blocked DLL can't crash us."""
    code = f"import {name}"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return name, False, "timeout"
    if r.returncode == 0:
        return name, True, ""
    err = (r.stderr or "").strip().splitlines()
    return name, False, err[-1] if err else f"exit {r.returncode}"


def recent_blocks(limit: int = 15):
    """Files blocked by Code Integrity / SAC recently (Windows event log)."""
    if sys.platform != "win32":
        return []
    ps = ("Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 300 "
          "-ErrorAction SilentlyContinue | Where-Object { $_.Id -in 3033,3034,3077,3089 } | "
          f"Select-Object -First {limit} | ForEach-Object {{ $_.Properties[1].Value }}")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                           text=True, timeout=60)
        files = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return list(dict.fromkeys(files))
    except Exception:  # noqa: BLE001
        return []


def is_blocked_error(msg: str) -> bool:
    m = msg.lower()
    return "4551" in m or "application control" in m or "blocked" in m


def report_markdown() -> str:
    lines = ["### 🛡️ فحص Smart App Control"]
    st = sac_state()
    if st is None:
        lines.append("- حالة الحماية: غير معروفة")
    else:
        lines.append(f"- حالة الحماية: **{_STATES.get(st, st)}**")

    blocked, other = [], []
    for name in MODULES:
        n, ok, err = check_module(name)
        if ok:
            continue
        (blocked if is_blocked_error(err) else other).append((n, err))

    if not blocked and not other:
        lines.append("- ✅ **كل المكتبات تعمل — لا يوجد أي حظر.**")
    for n, err in blocked:
        lines.append(f"- ⛔ `{n}` محظورة: `{err[:200]}`")
    for n, err in other:
        lines.append(f"- ⚠️ `{n}` لم تُحمّل (ليس بسبب الحماية): `{err[:200]}`")

    files = [f for f in recent_blocks() if "python" in f.lower() or "studio3d" in f.lower()]
    if files:
        lines.append("\n**ملفات حظرتها الحماية مؤخراً:**")
        lines += [f"- `{f}`" for f in files]

    if blocked or files:
        lines.append(
            "\n**الحلول (بالترتيب):**\n"
            "1. أعد التثبيت بالأمر الموجود في README — يعيد تنزيل المكتبات بنسخها الرسمية المعروفة لدى مايكروسوفت.\n"
            "2. حدّث ويندوز ثم أعد المحاولة (تقييم السمعة يتم عبر الإنترنت).\n"
            "3. للشركات: توقيع الملفات بشهادة توقيع كود (Code Signing) خاصة بالشركة — راجع docs/SMART_APP_CONTROL.md.\n"
            "4. كحل أخير: إطفاء Smart App Control من «أمان Windows ← التحكم في التطبيقات والمتصفح». "
            "في تحديثات ويندوز 11 منذ أبريل 2026 يمكن إعادة تشغيله لاحقاً بدون إعادة تثبيت ويندوز."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    out = report_markdown()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(out)
