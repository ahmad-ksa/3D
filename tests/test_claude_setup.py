import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio3d import claude_setup  # noqa: E402


def test_merge_keeps_other_servers(tmp_path):
    cfg = tmp_path / "claude_desktop_config.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))
    claude_setup.update(cfg)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["theme"] == "dark" and "other" in data["mcpServers"]
    s = data["mcpServers"]["studio3d"]
    assert s["args"] == ["-m", "studio3d.mcp_server"] and s["command"] == sys.executable
    claude_setup.update(cfg, remove=True)
    assert "studio3d" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]


def test_creates_new_file(tmp_path):
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"
    claude_setup.update(cfg)
    assert "studio3d" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
