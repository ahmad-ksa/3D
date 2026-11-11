"""Mesh clean-up, editing and export helpers (no GPU / AI needed)."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def stats(mesh: trimesh.Trimesh) -> dict:
    ext = mesh.bounding_box.extents
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "size": [round(float(v), 3) for v in ext],
    }


# ---------------------------------------------------------------------------
# Clean-up (pymeshlab based, same algorithms as Hunyuan3D post-processing)
# ---------------------------------------------------------------------------
def _to_ms(mesh: trimesh.Trimesh):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                               face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    return ms


def _from_ms(ms) -> trimesh.Trimesh:
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


def remove_floaters(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep only the biggest connected part (removes floating noise)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    areas = [p.area for p in parts]
    main = parts[int(np.argmax(areas))]
    # keep other parts that are reasonably large (e.g. separate legs of a table)
    keep = [p for p, a in zip(parts, areas) if a >= 0.05 * max(areas)]
    return trimesh.util.concatenate(keep) if len(keep) > 1 else main


def clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh = remove_floaters(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def reduce_faces(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation (low-poly)."""
    target_faces = int(target_faces)
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    ms = _to_ms(mesh)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservenormal=True,
                                                preserveboundary=True, optimalplacement=True)
    return _from_ms(ms)


def smooth(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    if iterations <= 0:
        return mesh
    mesh = mesh.copy()
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(iterations))
    return mesh


def fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    trimesh.repair.fill_holes(mesh)
    return mesh


def normalize(mesh, target_size: float | None = None, on_ground: bool = True):
    """Center the model; optionally scale its largest side to ``target_size`` (mm)
    and put its lowest point on the ground (Y = 0, glTF is Y-up)."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation(-center)
    if target_size and target_size > 0:
        extent = float(np.max(mesh.bounding_box.extents))
        if extent > 0:
            mesh.apply_scale(float(target_size) / extent)
    if on_ground:
        mesh.apply_translation([0, -mesh.bounds[0][1], 0])
    return mesh


def process(mesh, *, do_clean=True, target_faces=0, smooth_iters=0, holes=False,
            size_mm=0.0, on_ground=True):
    if do_clean:
        mesh = clean(mesh)
    if holes:
        mesh = fill_holes(mesh)
    if smooth_iters:
        mesh = smooth(mesh, smooth_iters)
    if target_faces:
        mesh = reduce_faces(mesh, target_faces)
    mesh = normalize(mesh, size_mm or None, on_ground)
    return mesh


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
_Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def export(mesh, out_dir: str | Path, name: str, fmt: str) -> str:
    """Export to glb / obj / stl / ply / 3mf. Returns the path of the file to
    download (OBJ with texture is zipped together with its .mtl and image)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()

    if fmt in ("stl", "3mf"):
        # 3D-printing formats are Z-up and have no colour/texture.
        m = mesh.copy() if isinstance(mesh, trimesh.Trimesh) else mesh.dump(concatenate=True)
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        m.apply_transform(_Y_UP_TO_Z_UP)
        path = out_dir / f"{name}.{fmt}"
        m.export(str(path))
        return str(path)

    if fmt == "obj":
        obj_dir = out_dir / f"{name}_obj"
        obj_dir.mkdir(exist_ok=True)
        obj_path = obj_dir / f"{name}.obj"
        mesh.export(str(obj_path))
        files = list(obj_dir.iterdir())
        if len(files) == 1:
            final = out_dir / f"{name}.obj"
            shutil.move(str(obj_path), final)
            obj_dir.rmdir()
            return str(final)
        zpath = out_dir / f"{name}_obj.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.name)
        return str(zpath)

    path = out_dir / f"{name}.{fmt}"
    mesh.export(str(path))
    return str(path)
