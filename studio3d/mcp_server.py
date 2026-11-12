"""Studio3D as an MCP server, so Claude (desktop app) can generate 3D models locally.

Started by the Claude desktop app via:  python -m studio3d.mcp_server
Set up automatically with:              python -m studio3d.claude_setup
"""
from __future__ import annotations

import os
import sys

# --- Keep the MCP stdio channel clean -----------------------------------------
# The protocol lives on the ORIGINAL stdout. Every other print (ours, the AI
# libraries', even C-level output) is redirected to stderr, which Claude logs.
_PROTO_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import asyncio  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import urllib.request  # noqa: E402
import uuid  # noqa: E402
from io import TextIOWrapper  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal, Optional  # noqa: E402

import anyio  # noqa: E402
from mcp.server.fastmcp import Context, FastMCP, Image  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from . import config  # noqa: E402

log = logging.getLogger("studio3d.mcp")

INBOX = Path(os.environ.get("STUDIO3D_INBOX",
                            Path.home() / "Pictures" / "Studio3D")).expanduser()
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_EXT = {".glb", ".obj", ".stl", ".ply", ".gltf", ".fbx"}
WAIT_SECONDS = 50  # how long one tool call waits before handing back a job id

Quality = Literal["سريع", "متوازن", "عالي الجودة", "fast", "balanced", "high"]
_Q = {"fast": "سريع", "balanced": "متوازن", "high": "عالي الجودة"}
Format = Literal["glb", "obj", "stl", "ply", "3mf"]

mcp = FastMCP(
    "studio3d",
    instructions=(
        "Studio3D generates 3D models LOCALLY on the user's NVIDIA GPU (free, private). "
        "Use it for game assets, animation props and 3D printing. "
        "Input images are file paths (absolute, or a file name inside the inbox folder — call "
        "list_input_images) or http(s) URLs. Generation can take 30 s to 10 min: if a tool returns a "
        "job_id, call check_job(job_id) until it is done. Each result includes a 4-view preview "
        "image so you can judge the shape; regenerate with another seed or higher quality if it is "
        "poor. For game assets use target_faces 3000-30000 and texture=true. Text prompts must be English. "
        "LIBRARY: every model ever generated is kept in the user's Studio3D library. Before generating "
        "something new, search it with list_models; reuse models in your work by reading their file paths "
        "(get_model) or copying them into a project folder with export_model (e.g. a Unity Assets folder, "
        "a game or animation project). open_in_blender opens a model in a visible Blender window."
    ),
)


# ---------------------------------------------------------------------------
# Background jobs (keeps tool calls short; the GPU work continues meanwhile)
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.progress = 0.0
        self.stage = "في الانتظار"
        self.result = None
        self.error = None
        self.done = threading.Event()
        self.started = time.time()


JOBS: dict[str, Job] = {}


def _start(kind: str, fn) -> Job:
    job = Job(kind)
    JOBS[job.id] = job

    def progress(p, desc):
        job.progress, job.stage = float(p), str(desc)

    def worker():
        try:
            job.result = fn(progress)
        except Exception as e:  # noqa: BLE001
            log.error(traceback.format_exc())
            job.error = "cancelled by the user" if type(e).__name__ == "Cancelled" else _friendly(e)
        finally:
            job.done.set()

    threading.Thread(target=worker, daemon=True, name=f"job-{job.id}").start()
    return job


async def _wait(job: Job, ctx: Optional[Context], seconds: float = WAIT_SECONDS):
    end = time.time() + seconds
    while not job.done.is_set() and time.time() < end:
        if ctx is not None:
            try:
                await ctx.report_progress(job.progress, 1.0)
            except Exception:  # noqa: BLE001
                pass
        await asyncio.sleep(1.0)
    return _job_response(job)


def _job_response(job: Job):
    if not job.done.is_set():
        el = int(time.time() - job.started)
        return (f"⏳ job_id={job.id} still running ({int(job.progress * 100)}% — {job.stage}, {el}s). "
                f"Call check_job(job_id='{job.id}') to wait for the result.")
    if job.error:
        return f"❌ job {job.id} failed: {job.error}"
    res = job.result
    lines = [f"✅ job {job.id} done ({int(time.time() - job.started)}s).",
             f"Folder: {res['job_dir']}",
             "Files:"] + [f"- {f}" for f in res["files"]]
    st = res.get("stats", {})
    if st:
        lines.append(f"Faces: {st.get('faces')}, vertices: {st.get('vertices')}, "
                     f"size (x,y,z): {st.get('size')}, watertight: {st.get('watertight')}")
    if res.get("warning"):
        lines.append(f"WARNING: {res['warning']}")
    if "seed" in res:
        lines.append(f"Seed: {res['seed']}")
    out = ["\n".join(lines)]
    try:
        from .preview import png_bytes, render_grid
        out.append(Image(data=png_bytes(render_grid(res["mesh"], 320)), format="png"))
    except Exception as e:  # noqa: BLE001
        log.warning("preview failed: %s", e)
    return out


def _friendly(e: Exception) -> str:
    msg = str(e)
    if "OutOfMemory" in type(e).__name__ or "out of memory" in msg.lower():
        return "GPU out of memory — retry with quality 'fast' or close other GPU apps."
    return msg[:500]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _resolve_file(ref: str, kinds: set) -> Path:
    ref = ref.strip().strip('"')
    if ref.lower().startswith(("http://", "https://")):
        dl = config.DATA_ROOT / "downloads"
        dl.mkdir(parents=True, exist_ok=True)
        suffix = Path(ref.split("?")[0]).suffix.lower()
        if suffix not in kinds:
            suffix = ".png" if kinds is IMAGE_EXT else ".glb"
        target = dl / f"{uuid.uuid4().hex[:8]}{suffix}"
        req = urllib.request.Request(ref, headers={"User-Agent": "Studio3D"})
        with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
            f.write(r.read())
        return target
    p = Path(ref).expanduser()
    if not p.is_absolute():
        for base in (INBOX, config.OUTPUT_DIR):
            cand = base / ref
            if cand.exists():
                return cand
    if not p.exists():
        raise FileNotFoundError(f"File not found: {ref}. Put it in {INBOX} or give a full path.")
    if p.suffix.lower() not in kinds:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return p


def _image(ref: Optional[str]):
    if not ref:
        return None
    return PILImage.open(_resolve_file(ref, IMAGE_EXT)).copy()


def _options(quality, texture, target_faces, size_mm, smooth, seed, formats, remove_background=True,
             topology="tri", rig=False):
    from .pipeline import Options
    return Options(topology=topology, rig=bool(rig), quality=_Q.get(quality, quality), texture=bool(texture), remove_bg=remove_background,
                   clean=True, smooth=int(smooth or 0), faces=int(target_faces or 0),
                   size_mm=float(size_mm or 0), seed=int(seed) if seed is not None else -1,
                   randomize=seed is None, formats=list(formats or ["glb"]))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def studio3d_status() -> str:
    """GPU info, folders and whether a job is running."""
    from . import engine
    info = engine.device_info()
    running = [j.id for j in JOBS.values() if not j.done.is_set()]
    return (f"GPU: {info}\nInbox (put input photos here): {INBOX}\nOutputs: {config.OUTPUT_DIR}\n"
            f"Quality presets: fast, balanced, high\nRunning jobs: {running or 'none'}")


@mcp.tool()
def list_input_images() -> str:
    """List photos in the Studio3D inbox folder (newest first)."""
    INBOX.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in INBOX.rglob("*") if p.suffix.lower() in IMAGE_EXT),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return f"Inbox is empty: {INBOX}"
    return "\n".join(str(p) for p in files[:100])


@mcp.tool()
async def generate_from_images(
    front: str,
    back: Optional[str] = None,
    left: Optional[str] = None,
    right: Optional[str] = None,
    quality: Quality = "balanced",
    texture: bool = True,
    target_faces: int = 0,
    size_mm: float = 0,
    smooth: int = 0,
    seed: Optional[int] = None,
    formats: list[Format] = ["glb"],  # noqa: B006
    topology: Literal["tri", "quad"] = "tri",
    rig: bool = False,
    ctx: Context = None,
) -> list:
    """Build a 3D model from 1-4 photos of the same object (front required; back/left/right
    improve accuracy). target_faces: 0 keeps full detail, 3000-30000 for game assets.
    size_mm: real-world size of the longest side (3D printing).
    topology='quad' gives clean all-quad topology (Blender QuadriFlow, OBJ/FBX keep quads).
    rig=true adds a humanoid skeleton (T/A-pose characters) and exports FBX for Unity/Unreal/Mixamo."""
    imgs = {"front": _image(front), "back": _image(back), "left": _image(left), "right": _image(right)}
    opt = _options(quality, texture, target_faces, size_mm, smooth, seed, formats, topology=topology, rig=rig)
    from . import pipeline
    job = _start("multiview", lambda pr: pipeline.run("multiview", imgs, None, opt, pr))
    return await _wait(job, ctx)


@mcp.tool()
async def generate_from_sheet(
    sheet: str,
    layout: Literal["auto", "row", "grid", "column"] = "auto",
    quality: Quality = "balanced",
    texture: bool = True,
    target_faces: int = 0,
    size_mm: float = 0,
    topology: Literal["tri", "quad"] = "tri",
    rig: bool = False,
    seed: Optional[int] = None,
    formats: list[Format] = ["glb"],  # noqa: B006
    ctx: Context = None,
) -> list:
    """Build a 3D model from ONE character-sheet image that shows the 4 sides (front, back and two
    side views) in a row or a 2x2 grid. The figures are found, cropped, sized equally and assigned to
    front/back/left/right automatically."""
    from . import sheet as sh
    img = _image(sheet)
    views = sh.split(img, layout)
    opt = _options(quality, texture, target_faces, size_mm, 0, seed, formats, topology=topology, rig=rig)
    from . import pipeline
    job = _start("multiview", lambda pr: pipeline.run("multiview", views, None, opt, pr))
    return await _wait(job, ctx)


@mcp.tool()
async def generate_from_image(
    image: str,
    quality: Quality = "balanced",
    texture: bool = True,
    target_faces: int = 0,
    size_mm: float = 0,
    smooth: int = 0,
    seed: Optional[int] = None,
    formats: list[Format] = ["glb"],  # noqa: B006
    topology: Literal["tri", "quad"] = "tri",
    rig: bool = False,
    ctx: Context = None,
) -> list:
    """Build a 3D model from a single photo or drawing (object on a plain background works best)."""
    img = _image(image)
    opt = _options(quality, texture, target_faces, size_mm, smooth, seed, formats, topology=topology, rig=rig)
    from . import pipeline
    job = _start("single", lambda pr: pipeline.run("single", {"front": img}, None, opt, pr))
    return await _wait(job, ctx)


@mcp.tool()
async def generate_from_text(
    prompt: str,
    quality: Quality = "balanced",
    texture: bool = True,
    target_faces: int = 0,
    size_mm: float = 0,
    smooth: int = 0,
    seed: Optional[int] = None,
    formats: list[Format] = ["glb"],  # noqa: B006
    topology: Literal["tri", "quad"] = "tri",
    rig: bool = False,
    pose: Literal["free", "t", "a"] = "free",
    ctx: Context = None,
) -> list:
    """Build a 3D model from an ENGLISH text description of ONE object
    (e.g. 'a low-poly wooden treasure chest, game asset'). First use downloads ~10 GB.
    pose: for characters — "t" forces a T-pose, "a" an A-pose (best with rig=True; first use
    downloads the pose model ~3 GB); "free" lets the image model choose."""
    opt = _options(quality, texture, target_faces, size_mm, smooth, seed, formats, topology=topology, rig=rig)
    opt.pose = pose
    from . import pipeline
    job = _start("text", lambda pr: pipeline.run("text", {}, prompt, opt, pr))
    return await _wait(job, ctx)


@mcp.tool()
async def texture_model(model: str, reference_image: str, ctx: Context = None) -> list:
    """Paint an AI texture onto an existing (untextured) 3D model using a reference image."""
    path = str(_resolve_file(model, MODEL_EXT))
    img = _image(reference_image)
    from . import pipeline
    job = _start("retexture", lambda pr: pipeline.retexture(path, img, True, pr))
    return await _wait(job, ctx)


@mcp.tool()
async def edit_model(
    model: str,
    target_faces: int = 0,
    smooth: int = 0,
    size_mm: float = 0,
    clean: bool = True,
    fill_holes: bool = False,
    formats: list[Format] = ["glb", "stl"],  # noqa: B006
    ctx: Context = None,
) -> list:
    """Clean / decimate (low-poly) / smooth / resize an existing model and export it.
    Note: decimating a textured model removes its texture."""
    path = str(_resolve_file(model, MODEL_EXT))
    from . import pipeline
    job = _start("edit", lambda pr: pipeline.edit(path, clean=clean, holes=fill_holes, smooth=smooth,
                                                  faces=target_faces, size_mm=size_mm, formats=formats))
    return await _wait(job, ctx)


@mcp.tool()
async def quad_remesh_model(model: str, target_faces: int = 10000, bake_texture: bool = True,
                            ctx: Context = None) -> list:
    """Convert a model to clean quad topology with Blender (QuadriFlow) — runs Blender hidden.
    Keeps the texture by re-baking it. Outputs OBJ + FBX (quads) and GLB."""
    path = str(_resolve_file(model, MODEL_EXT))
    from . import pipeline
    job = _start("quad", lambda pr: pipeline.quad(path, target_faces, 2048 if bake_texture else 0))
    return await _wait(job, ctx)


@mcp.tool()
async def rig_model(model: str, ctx: Context = None) -> list:
    """Add an automatic humanoid skeleton (19 bones, Unity/Mixamo names) with skin weights, using
    Blender in the background. Best for characters standing in T-pose or A-pose. Outputs FBX + GLB."""
    path = str(_resolve_file(model, MODEL_EXT | {".fbx"}))
    from . import pipeline
    job = _start("rig", lambda pr: pipeline.rig(path))
    return await _wait(job, ctx)


@mcp.tool()
async def lip_sync_model(model: str, audio: str, strength_degrees: float = 5, ctx: Context = None) -> list:
    """Make a rigged character talk: finds the mouth on the texture, opens a slit there (white
    shows behind it like teeth), and animates jaw + mouth corners from the voice in an audio file
    (open for "aa", wide for "ee", round for "oo"; any language, offline)
    (wav/mp3/ogg/flac; path, inbox file name or URL). The model must have a Studio3D skeleton
    (rig_model first). Outputs GLB + FBX with the animation and a .blend with the sound."""
    path = str(_resolve_file(model, {".glb", ".fbx"}))
    aud = str(_resolve_file(audio, {".wav", ".mp3", ".ogg", ".flac"}))
    from . import pipeline
    job = _start("talk", lambda pr: pipeline.lip_sync(path, aud, strength_degrees))
    return await _wait(job, ctx)


@mcp.tool()
def cancel_job(job_id: str = "") -> str:
    """Stop the running generation (the GPU work stops at the next step)."""
    from . import engine
    engine.request_cancel()
    running = [j.id for j in JOBS.values() if not j.done.is_set()]
    return f"Stop requested. Running jobs: {running or 'none'}"


@mcp.tool()
async def check_job(job_id: str, ctx: Context = None) -> list:
    """Wait for / get the result of a running generation job."""
    job = JOBS.get(job_id)
    if job is None:
        return [f"Unknown job_id {job_id}. Known: {list(JOBS)}"]
    res = await _wait(job, ctx)
    return res if isinstance(res, list) else [res]


def _entry_line(e: dict) -> str:
    tags = [t for t, on in (("textured", e["textured"]), ("quad", e["quad"]), ("rigged", e["rigged"])) if on]
    faces = (e.get("stats") or {}).get("faces")
    return (f"- id={e['id']} | name={e['name']} | kind={e['kind']}"
            + (f" | {faces} faces" if faces else "") + (f" | {', '.join(tags)}" if tags else "")
            + (f" | prompt={e['prompt']}" if e.get("prompt") else "")
            + "\n  files: " + ", ".join(e["files"]))


@mcp.tool()
def list_models(query: str = "", limit: int = 20) -> str:
    """Search the user's library of generated 3D models (newest first). query matches name,
    prompt or type ('rig', 'quad'...). Use this before generating something new."""
    from . import library
    items = library.list_models(query, limit=max(1, min(limit, 100)), with_thumb=False)
    if not items:
        return "No models found." + (f" (query: {query})" if query else "")
    return f"{len(items)} model(s):\n" + "\n".join(_entry_line(e) for e in items)


@mcp.tool()
def get_model(model_id: str) -> list:
    """Details, file paths and a 4-view preview image of one library model (id from list_models,
    or a path to one of its files)."""
    from . import library, mesh_tools
    from .preview import png_bytes, render_grid
    e = library.resolve(model_id)
    out = [_entry_line(e) + f"\n  folder: {e['folder']}"]
    if e["preview"]:
        out.append(Image(data=png_bytes(render_grid(mesh_tools.load_mesh(e["preview"]), 320)), format="png"))
    return out


@mcp.tool()
def export_model(model_id: str, destination: str, formats: Optional[list[str]] = None,
                 name: Optional[str] = None) -> str:
    """Copy a library model into a folder for use in a project (e.g. 'C:/Projects/Game/Assets/Models').
    formats: e.g. ['fbx'] or ['glb','obj']; omit to copy every format available. name: file name."""
    from . import library
    e = library.resolve(model_id)
    files = library.export_to(e["id"], destination, formats, name)
    if not files:
        return f"Nothing copied: model {e['id']} has none of the formats {formats}. Available: {e['files']}"
    return "Copied:\n" + "\n".join(f"- {f}" for f in files)


@mcp.tool()
def open_in_blender(model_id: str) -> str:
    """Open a library model in a normal (visible) Blender window for the user to work on."""
    from . import library
    e = library.resolve(model_id)
    path = library.open_in_blender(e["id"])
    return f"Opened in Blender: {path}"


@mcp.tool()
def rename_model(model_id: str, name: str) -> str:
    """Give a library model a friendly name (shown in the app and searchable)."""
    from . import library
    e = library.rename(library.resolve(model_id)["id"], name)
    return f"Renamed {e['id']} -> {e['name']}"


@mcp.tool()
def preview_model(model: str) -> list:
    """Render a 4-view preview image of any 3D model file so you can inspect it."""
    from . import mesh_tools
    from .preview import png_bytes, render_grid
    path = _resolve_file(model, MODEL_EXT)
    import trimesh
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    st = mesh_tools.stats(mesh)
    return [f"{path}\n{st}", Image(data=png_bytes(render_grid(mesh, 320)), format="png")]


# ---------------------------------------------------------------------------
async def _serve():
    proto_out = anyio.wrap_file(TextIOWrapper(os.fdopen(_PROTO_FD, "wb"), encoding="utf-8"))
    async with stdio_server(stdout=proto_out) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream,
                                  mcp._mcp_server.create_initialization_options())


def main():
    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    INBOX.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr),
                                  logging.FileHandler(config.DATA_ROOT / "mcp.log", encoding="utf-8")])
    log.info("Studio3D MCP server starting (inbox=%s)", INBOX)
    anyio.run(_serve)


if __name__ == "__main__":
    main()
