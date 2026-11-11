"""Helpers shared by the scripts that Studio3D runs inside Blender (headless).

Every script is started as:
    blender -b --factory-startup --python <script> -- <args.json>
reads its arguments from args.json and writes {"ok": ..., ...} to args["result"].
They also run with the standalone `bpy` Python module (used by the tests).
"""
import json
import os
import sys
import traceback

import bpy


def load_args():
    argv = sys.argv
    path = argv[argv.index("--") + 1] if "--" in argv else argv[-1]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_result(args, data):
    with open(args["result"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def run(main):
    args = load_args()
    try:
        data = main(args) or {}
        data.setdefault("ok", True)
    except Exception as e:  # noqa: BLE001
        data = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    write_result(args, data)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _op(*names):
    """Return the first existing operator among dotted names (API differs across Blender versions)."""
    for name in names:
        mod, op = name.rsplit(".", 1)
        cat = getattr(bpy.ops, mod.split(".")[-1], None)
        if cat is not None and hasattr(cat, op):
            return getattr(cat, op)
    raise RuntimeError(f"Blender operator not available: {names}")


def import_model(path):
    """Import glb/gltf/obj/fbx/stl/ply and return ONE joined mesh object."""
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        _op("wm.obj_import", "import_scene.obj")(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".stl":
        _op("wm.stl_import", "import_mesh.stl")(filepath=path)
    elif ext == ".ply":
        _op("wm.ply_import", "import_mesh.ply")(filepath=path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh found in the file")
    # drop empties/cameras/lights/armatures that came with the file
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.name = "Model"
    return obj


def select_only(*objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def export(objs, outdir, name, formats):
    """Export the given objects. OBJ/FBX keep quads; GLB is triangulated by glTF itself."""
    os.makedirs(outdir, exist_ok=True)
    select_only(*objs)
    files = []
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        if fmt == "fbx":
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, add_leaf_bones=False,
                                     bake_anim=False, path_mode="COPY", embed_textures=True,
                                     mesh_smooth_type="FACE")
        elif fmt == "glb":
            bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True)
        elif fmt == "obj":
            _op("wm.obj_export", "export_scene.obj")(filepath=path, export_selected_objects=True,
                                                      path_mode="COPY") \
                if hasattr(bpy.ops.wm, "obj_export") else \
                bpy.ops.export_scene.obj(filepath=path, use_selection=True, path_mode="COPY")
        else:
            continue
        files.append(path)
    return files


def face_stats(obj):
    me = obj.data
    quads = sum(1 for p in me.polygons if len(p.vertices) == 4)
    return {"faces": len(me.polygons), "quads": quads, "vertices": len(me.vertices)}


def has_image_texture(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image is not None:
                    return True
    return False
