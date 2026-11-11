"""Studio3D extension entry point for the Claude desktop app.

The heavy program (Python, PyTorch, AI models) is installed by install.ps1 in
%LOCALAPPDATA%\\Studio3D. This tiny launcher runs that installed program's MCP server.
"""
import os
import sys

app = os.environ.get("STUDIO3D_APP") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Studio3D", "app")
if not os.path.isdir(os.path.join(app, "studio3d")):
    sys.stderr.write(f"Studio3D is not installed in {app}. Run the installer from "
                     "https://github.com/ahmad-ksa/3D first.\n")
    sys.exit(1)
sys.path.insert(0, app)
os.chdir(app)

from studio3d.mcp_server import main  # noqa: E402

main()
