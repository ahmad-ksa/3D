"""Quad topology for a generated model (runs inside Blender).

1. QuadriFlow remesh -> clean all-quad topology with ~target_faces faces.
   (fallback: voxel remesh + tris-to-quads if QuadriFlow fails)
2. If the source model has a texture: new UVs + bake the colours from the
   original onto the quad mesh (Cycles, selected-to-active).
3. Export OBJ / FBX (quads kept) and GLB.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import bmesh  # noqa: E402

import _common as C  # noqa: E402


def _decimate(obj, max_faces):
    n = len(obj.data.polygons)
    if n <= max_faces:
        return
    C.select_only(obj)
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = max_faces / n
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _weld(obj):
    """glTF/OBJ files split vertices along UV seams; QuadriFlow needs one connected surface."""
    C.select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    diag = max(obj.dimensions) or 1.0
    bpy.ops.mesh.remove_doubles(threshold=1e-5 * diag)
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode="OBJECT")


def _quadriflow(obj, target):
    C.select_only(obj)
    before = len(obj.data.polygons)
    try:
        res = bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=int(target), seed=0,
                                               use_mesh_symmetry=False, use_preserve_sharp=False,
                                               use_preserve_boundary=False, smooth_normals=False)
    except RuntimeError:
        return False
    after = obj.data.polygons
    quads = sum(1 for p in after if len(p.vertices) == 4)
    n = len(after)
    return "FINISHED" in res and n != before and n >= 0.3 * target and quads > 0.9 * n


def _voxel_then_quads(obj, target):
    """Fallback: voxel remesh sized for ~target faces, then pair triangles into quads."""
    import math
    C.select_only(obj)
    area = sum(p.area for p in obj.data.polygons) or 1.0
    voxel = math.sqrt(area / max(target, 100))
    mod = obj.modifiers.new("vox", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=3.1416, angle_shape_threshold=3.1416)
    bm.to_mesh(obj.data)
    bm.free()


def _bake_texture(src, dst, size):
    """Bake base colour from src (textured) to dst (quad mesh) through new UVs."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    C.select_only(dst)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

    img = bpy.data.images.new("model_texture", width=size, height=size, alpha=False)
    mat = bpy.data.materials.new("ModelMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    dst.data.materials.clear()
    dst.data.materials.append(mat)
    nt.nodes.active = tex

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    diag = max(dst.dimensions) or 1.0
    bake = scene.render.bake
    bake.use_pass_direct = False
    bake.use_pass_indirect = False
    bake.use_pass_color = True
    bake.margin = 8
    bpy.ops.object.bake(type="DIFFUSE", use_selected_to_active=True, cage_extrusion=0.02 * diag,
                        max_ray_distance=0.05 * diag)
    img.filepath_raw = os.path.join(bpy.app.tempdir or os.getcwd(), "model_texture.png")
    img.file_format = "PNG"
    img.save()
    img.pack()


def main(args):
    C.reset_scene()
    src = C.import_model(args["input"])
    textured = C.has_image_texture(src) and int(args.get("bake_size", 2048)) > 0
    _decimate(src, int(args.get("max_input_faces", 200000)))

    C.select_only(src)
    bpy.ops.object.duplicate()
    quad = bpy.context.view_layer.objects.active
    quad.name = "Model"
    src.name = "Source"
    quad.data.materials.clear()

    target = int(args.get("target_faces", 10000))
    _weld(quad)
    backup = quad.data.copy()
    method = "quadriflow"
    if not _quadriflow(quad, target):
        quad.data = backup
        method = "voxel"
        _voxel_then_quads(quad, target)

    if textured:
        _bake_texture(src, quad, int(args.get("bake_size", 2048)))
    bpy.data.objects.remove(src, do_unlink=True)

    C.select_only(quad)
    bpy.ops.object.shade_smooth() if args.get("smooth_shading", True) else None
    files = C.export([quad], args["outdir"], args.get("name", "model_quad"),
                     args.get("formats", ["obj", "fbx", "glb"]))
    stats = C.face_stats(quad)
    stats.update({"files": files, "method": method, "textured": textured})
    return stats


if __name__ == "__main__":
    C.run(main)
