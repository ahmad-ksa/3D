"""Talks to the Studio3D MCP server over real stdio, with the AI models mocked."""
import asyncio
import os
import sys
import textwrap
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

LAUNCH = textwrap.dedent("""
    import sys, trimesh
    from studio3d import mcp_server, engine
    print("noise on stdout must not break the protocol")
    def fake_shape(image, kind, preset, seed, **kw):
        print("library print inside a job")
        return trimesh.creation.icosphere(4)
    engine.generate_shape = fake_shape
    engine.prepare_image = lambda img, rb: img.convert("RGBA")
    mcp_server.main()
""")


def test_mcp_roundtrip(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    Image.new("RGB", (64, 64), "white").save(inbox / "front.png")
    env = dict(os.environ, PYTHONPATH=str(ROOT), STUDIO3D_HOME=str(tmp_path / "home"),
               STUDIO3D_INBOX=str(inbox))

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-c", LAUNCH], env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"generate_from_images", "generate_from_text", "check_job", "preview_model",
                        "edit_model", "texture_model", "list_input_images"} <= names
                lst = await s.call_tool("list_input_images", {})
                assert "front.png" in lst.content[0].text
                res = await s.call_tool("generate_from_images",
                                        {"front": "front.png", "texture": False, "quality": "fast",
                                         "target_faces": 1000, "size_mm": 50, "formats": ["glb", "stl"]})
                text = res.content[0].text
                while "still running" in text:
                    job = text.split("job_id=")[1].split()[0]
                    res = await s.call_tool("check_job", {"job_id": job})
                    text = res.content[0].text
                assert "done" in text, text
                assert any(c.type == "image" for c in res.content)
                glb = [ln[2:] for ln in text.splitlines() if ln.endswith(".glb")][0]
                assert Path(glb).exists()
                prev = await s.call_tool("preview_model", {"model": glb})
                assert prev.content[1].type == "image"
                ed = await s.call_tool("edit_model", {"model": glb, "target_faces": 300, "formats": ["obj"]})
                assert "done" in ed.content[0].text

    asyncio.run(go())
