"""Render a 2x2 preview image (front / right / back / left) of a mesh.

Uses our pure-PyTorch rasteriser, so it works on CPU or GPU with no OpenGL.
Used by the Claude (MCP) server so Claude can *see* what it generated.
"""
from __future__ import annotations

import io
import math
import sys

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

from . import config

if str(config.VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(config.VENDOR_DIR))
import custom_rasterizer as cr  # noqa: E402

BG = np.array([236, 238, 243], dtype=np.float32) / 255.0
LIGHT = np.array([0.35, 0.55, 1.0]) / np.linalg.norm([0.35, 0.55, 1.0])


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        vis = mesh.visual
        if vis.kind == "texture":
            vis = vis.to_color()
        if vis.kind in ("vertex", "face") and vis.vertex_colors is not None:
            c = np.asarray(vis.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            if len(c) == len(mesh.vertices):
                return c
    except Exception:  # noqa: BLE001
        pass
    return np.tile(np.array([[0.78, 0.80, 0.86]], dtype=np.float32), (len(mesh.vertices), 1))


def _rot(azim_deg: float, elev_deg: float) -> np.ndarray:
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return rx @ ry


def render_view(mesh: trimesh.Trimesh, azim: float, elev: float = 15.0, size: int = 384,
                device: str = "cpu") -> Image.Image:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    center = (v.max(0) + v.min(0)) / 2
    radius = np.linalg.norm(v - center, axis=1).max() or 1.0
    v = (v - center) / radius
    R = _rot(azim, elev)
    vr = v @ R.T
    pos = np.concatenate([vr[:, :2] / 1.05, -vr[:, 2:3] / 1.1, np.ones((len(vr), 1))], axis=1)

    faces = np.asarray(mesh.faces, dtype=np.int32)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)[None]
    tri_t = torch.tensor(faces, dtype=torch.int32, device=device)
    findices, bary = cr.rasterize(pos_t, tri_t, (size, size))

    fn = np.asarray(mesh.face_normals, dtype=np.float64) @ R.T
    fn[fn[:, 2] < 0] *= -1  # two-sided lighting
    shade = 0.28 + 0.72 * np.clip(fn @ LIGHT, 0, 1)

    col = torch.tensor(_vertex_colors(mesh), dtype=torch.float32, device=device)[None]
    rgb = cr.interpolate(col, findices, bary, tri_t)[0].cpu().numpy()
    fi = findices.cpu().numpy()
    hit = fi > 0
    img = np.tile(BG, (size, size, 1))
    img[hit] = rgb[hit] * shade[fi[hit] - 1][:, None]
    img = np.flipud(img)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def render_grid(mesh, size: int = 384) -> Image.Image:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    labels = [("أمام / front", 0), ("يمين / right", 90), ("خلف / back", 180), ("يسار / left", 270)]
    grid = Image.new("RGB", (size * 2, size * 2), tuple(int(c * 255) for c in BG))
    for i, (label, az) in enumerate(labels):
        view = render_view(mesh, az, size=size)
        ImageDraw.Draw(view).text((8, 6), label.split(" / ")[1], fill=(90, 90, 110))
        grid.paste(view, ((i % 2) * size, (i // 2) * size))
    return grid


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_textured(mesh: trimesh.Trimesh, azim: float = 0.0, elev: float = 0.0, size: int = 1024,
                    device: str = "cpu"):
    """Sharp render that samples the texture per pixel (not per vertex).
    Returns (RGB image, info) where info maps pixels back to model coordinates:
    model_x = (px/size*2-1)*s + cx ; model_y = ((1-py/size)*2-1)*s + cy   (front view, glTF Y-up)."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    center = (v.max(0) + v.min(0)) / 2
    radius = float(np.linalg.norm(v - center, axis=1).max() or 1.0)
    R = _rot(azim, elev)
    vr = ((v - center) / radius) @ R.T
    pos = np.concatenate([vr[:, :2] / 1.05, -vr[:, 2:3] / 1.1, np.ones((len(vr), 1))], axis=1)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    fi, bc = cr.rasterize(torch.tensor(pos, dtype=torch.float32)[None],
                          torch.tensor(faces, dtype=torch.int32), (size, size))
    fi = fi.numpy()
    bc = bc.numpy()
    img = np.tile(BG, (size, size, 1))
    hit = fi > 0
    vis = mesh.visual
    tex = None
    if getattr(vis, "kind", None) == "texture" and vis.uv is not None:
        try:
            mat = vis.material
            im = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if im is not None:
                tex = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:  # noqa: BLE001
            tex = None
    if tex is not None:
        uv = np.asarray(vis.uv, dtype=np.float64)
        f = faces[fi[hit] - 1]
        b = bc[hit]
        uvp = (uv[f] * b[:, :, None]).sum(1)
        th, tw = tex.shape[:2]
        px = np.clip((uvp[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(int)
        py = np.clip((1 - (uvp[:, 1] % 1.0)) * (th - 1), 0, th - 1).astype(int)
        img[hit] = tex[py, px]
    else:
        cols = _vertex_colors(mesh)
        f = faces[fi[hit] - 1]
        img[hit] = (cols[f] * bc[hit][:, :, None]).sum(1)
    img = np.flipud(img)
    info = {"size": size, "scale": radius * 1.05, "center": center, "rotation": R,
            "mask": np.flipud(hit)}
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), info
