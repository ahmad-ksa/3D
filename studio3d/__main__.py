"""Entry point:  python -m studio3d  [--no-window] [--port N] [--download [all]]"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from . import config


def _setup_logging():
    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(config.DATA_ROOT / "studio3d.log", encoding="utf-8")],
    )


def _find_edge():
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     r"Microsoft\Edge\Application\msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                     r"Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return shutil.which("msedge")


def open_window(url: str):
    """Open the UI in a clean app-style window (Microsoft Edge, signed by Microsoft)."""
    edge = _find_edge() if sys.platform == "win32" else None
    if edge:
        profile = config.DATA_ROOT / "window"
        subprocess.Popen([edge, f"--app={url}", "--window-size=1400,920",
                          f"--user-data-dir={profile}", "--no-first-run"])
    else:
        webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser(prog="studio3d")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--no-window", action="store_true", help="لا تفتح نافذة تلقائياً")
    ap.add_argument("--download", nargs="?", const="basic", choices=["basic", "all"],
                    help="تنزيل النماذج مسبقاً ثم الخروج")
    args = ap.parse_args()

    _setup_logging()
    log = logging.getLogger("studio3d")

    if args.download:
        from .download import download_models
        download_models(args.download == "all")
        return

    print("=" * 60)
    print(f"  {config.APP_NAME} {config.VERSION}")
    print("  اترك هذه النافذة مفتوحة أثناء استخدام البرنامج.")
    print("=" * 60)

    from . import engine, ui
    info = engine.device_info()
    log.info("GPU: %s", info)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demo = ui.build()
    demo.queue(default_concurrency_limit=1)
    url = f"http://{config.HOST}:{args.port}"
    if not args.no_window:
        threading.Thread(target=lambda: (time.sleep(2.5), open_window(url)), daemon=True).start()
    demo.launch(server_name=config.HOST, server_port=args.port, inbrowser=False, show_api=False,
                allowed_paths=[str(config.OUTPUT_DIR)], favicon_path=None, share=False)


if __name__ == "__main__":
    main()
