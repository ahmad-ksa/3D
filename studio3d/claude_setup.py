"""Register Studio3D in the Claude desktop app (claude_desktop_config.json).

Run:  python -m studio3d.claude_setup          (add / update)
      python -m studio3d.claude_setup --remove (remove)
Then fully quit Claude (system tray -> Quit) and open it again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SERVER_NAME = "studio3d"


def config_paths() -> list[Path]:
    paths = []
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
        local = os.environ.get("LOCALAPPDATA")
        if local:  # Microsoft Store (MSIX) installs keep a virtualised copy
            for pkg in (Path(local) / "Packages").glob("Claude_*"):
                paths.append(pkg / "LocalCache" / "Roaming" / "Claude" / "claude_desktop_config.json")
    elif sys.platform == "darwin":
        paths.append(Path.home() / "Library/Application Support/Claude/claude_desktop_config.json")
    else:
        paths.append(Path.home() / ".config/Claude/claude_desktop_config.json")
    existing = [p for p in paths if p.parent.exists()]
    return existing or paths[:1]


def server_entry() -> dict:
    app_dir = str(Path(__file__).resolve().parent.parent)
    return {
        "command": sys.executable,
        "args": ["-m", "studio3d.mcp_server"],
        "env": {"PYTHONPATH": app_dir, "PYTHONIOENCODING": "utf-8"},
    }


def update(path: Path, remove: bool = False) -> str:
    data = {}
    if path.exists():
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            data = json.loads(text)
            path.with_suffix(".json.bak").write_text(text, encoding="utf-8")
    servers = data.setdefault("mcpServers", {})
    if remove:
        servers.pop(SERVER_NAME, None)
    else:
        servers[SERVER_NAME] = server_entry()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    for p in config_paths():
        try:
            print(("Removed from: " if args.remove else "Claude connected: ") + update(p, args.remove))
        except Exception as e:  # noqa: BLE001
            print(f"Could not update {p}: {e}")
    if not args.remove:
        print("Now fully quit Claude (tray icon -> Quit) and open it again.")


if __name__ == "__main__":
    main()
