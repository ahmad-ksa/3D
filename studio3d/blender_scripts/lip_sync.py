"""Talking character: jaw bone + mouth animation driven by an audio file (runs inside Blender).

1. Import a rigged model (from auto_rig.py: Head/Neck bones).
2. Add a "Jaw" bone under Head if missing, and give the lower face smooth jaw weights
   (taken from the Head weights, so nothing else changes).
3. Read the audio (wav/mp3/ogg/flac via Blender's aud module, WAV fallback), compute the
   loudness per frame, smooth it and key the jaw opening.
4. Export GLB + FBX with the animation, and a .blend with the sound in the timeline.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import _common as C  # noqa: E402


def import_rigged(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError("Use the rigged GLB or FBX made by Studio3D")
    rig = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == rig]
    if rig is None or not meshes or "Head" not in rig.data.bones:
        raise RuntimeError("This model has no Studio3D skeleton. Add a skeleton first (rig).")
    # drop bone-shape helpers the glTF importer may create
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    return rig, max(meshes, key=lambda o: len(o.data.vertices))


def face_frame(rig, mesh):
    """Head measurements in world space: mouth height, face front, width."""
    head = rig.data.bones["Head"]
    neck = rig.data.bones.get("Neck")
    mw = mesh.matrix_world
    hz = (rig.matrix_world @ head.head_local).z
    cz = (rig.matrix_world @ (neck.head_local if neck else head.head_local)).z
    cx = (rig.matrix_world @ head.head_local).x
    verts = [mw @ v.co for v in mesh.data.vertices]
    top = max(v.z for v in verts)
    hh = max(top - cz, 1e-6)
    # face region: slice around the lower head
    band = [v for v in verts if hz - 0.05 * hh <= v.z <= hz + 0.35 * hh]
    xs = [v.x for v in band] or [cx]
    half_w = max(abs(max(xs) - cx), abs(min(xs) - cx)) or 0.1 * hh
    narrow = [v for v in band if abs(v.x - cx) < 0.35 * half_w]
    front_y = min(v.y for v in narrow) if narrow else min(v.y for v in band)
    back_y = max(v.y for v in narrow) if narrow else max(v.y for v in band)
    mouth_z = hz + 0.10 * hh
    return {"cx": cx, "mouth_z": mouth_z, "chin_z": hz - 0.02 * hh, "hh": hh, "half_w": half_w,
            "front_y": front_y, "back_y": back_y}


def add_jaw(rig, f):
    C.select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    if "Jaw" not in eb:
        inv = rig.matrix_world.inverted()
        depth = f["back_y"] - f["front_y"]
        hinge = Vector((f["cx"], f["front_y"] + 0.55 * depth, f["mouth_z"] + 0.04 * f["hh"]))
        tip = Vector((f["cx"], f["front_y"] + 0.05 * depth, f["mouth_z"] - 0.06 * f["hh"]))
        b = eb.new("Jaw")
        b.head, b.tail = inv @ hinge, inv @ tip
        b.parent = eb["Head"]
        b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")


def add_jaw_at_mouth(rig, f, m):
    """Jaw hinge behind/above the mouth, bone pointing to the mouth."""
    C.select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    if "Jaw" not in eb:
        inv = rig.matrix_world.inverted()
        depth = max(f["back_y"] - m["y"], m["width"])
        hinge = Vector((m["x"], m["y"] + 0.55 * depth, m["z"] + 0.45 * m["width"]))
        tip = Vector((m["x"], m["y"] - 0.02 * depth, m["z"] - 0.15 * m["width"]))
        b = eb.new("Jaw")
        b.head, b.tail = inv @ hinge, inv @ tip
        b.parent = eb["Head"]
        b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")


def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / ((e1 - e0) or 1e-9)))
    return t * t * (3 - 2 * t)


def jaw_weights(mesh, f):
    """Lower face follows the jaw: below the mouth line, front half of the head."""
    mw = mesh.matrix_world
    head_g = mesh.vertex_groups.get("Head")
    jaw_g = mesh.vertex_groups.get("Jaw") or mesh.vertex_groups.new(name="Jaw")
    depth = f["back_y"] - f["front_y"]
    count = 0
    for v in mesh.data.vertices:
        p = mw @ v.co
        w_head = 0.0
        if head_g:
            for g in v.groups:
                if g.group == head_g.index:
                    w_head = g.weight
        if w_head <= 0:
            continue
        down = smoothstep(f["mouth_z"] + 0.02 * f["hh"], f["mouth_z"] - 0.05 * f["hh"], p.z)
        up = smoothstep(f["chin_z"] - 0.12 * f["hh"], f["chin_z"], p.z)  # not into the neck
        front = smoothstep(f["front_y"] + 0.45 * depth, f["front_y"] + 0.2 * depth, p.y)
        side = smoothstep(0.55 * f["half_w"], 0.25 * f["half_w"], abs(p.x - f["cx"]))
        w = down * up * front * side
        if w > 0.01:
            jaw_g.add([v.index], w * w_head, "REPLACE")
            head_g.add([v.index], (1 - w) * w_head, "REPLACE")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Mouth slit (when the mouth line was found on the texture)
# ---------------------------------------------------------------------------
def make_slit(mesh, m):
    """Cut the mesh along the mouth line so the lips can part. m: mouth in Blender coords
    {x, y (front surface), z, width}. Returns the vertex indices of the lower lip edge."""
    import bmesh
    mx, my, mz, w = m["x"], m["y"], m["z"], m["width"]
    me = mesh.data
    mw = mesh.matrix_world
    inv = mw.inverted()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    def wco(v):
        return mw @ v.co

    def in_region(f, sx=0.75, sz=0.35):
        c = mw @ f.calc_center_median()
        return abs(c.x - mx) < sx * w and abs(c.z - mz) < sz * w and c.y < my + 0.35 * w

    region = [f for f in bm.faces if in_region(f)]
    if not region:
        bm.free()
        return []
    # more density around the mouth so the cut follows the lips
    edges = list({e for f in region for e in f.edges})
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=2, use_grid_fill=True)
    region = [f for f in bm.faces if in_region(f)]
    geom = list({x for f in region for x in (f, *f.edges, *f.verts)})
    co = inv @ Vector((mx, my, mz))
    no = (inv.to_3x3() @ Vector((0, 0, 1))).normalized()
    res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=no, dist=1e-6)
    cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    cut = [e for e in cut if all(abs(wco(v).x - mx) < 0.5 * w for v in e.verts)
           and all(wco(v).y < my + 0.3 * w for v in e.verts)]
    if not cut:
        bm.free()
        return []
    bmesh.ops.split_edges(bm, edges=cut)
    # lower-lip copies: on the line, every linked face below it
    lower = []
    for v in bm.verts:
        p = wco(v)
        if abs(p.z - mz) < 1e-4 * max(w, 1e-3) + 1e-6 and abs(p.x - mx) < 0.5 * w and v.link_faces:
            if all((mw @ f.calc_center_median()).z < mz for f in v.link_faces):
                lower.append(v.index)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return lower


def slit_jaw_weights(mesh, m, lower_edge):
    """Lower lip + chin follow the jaw; the corners stay closed."""
    mx, my, mz, w = m["x"], m["y"], m["z"], m["width"]
    mw = mesh.matrix_world
    head_g = mesh.vertex_groups.get("Head")
    jaw_g = mesh.vertex_groups.get("Jaw") or mesh.vertex_groups.new(name="Jaw")
    lower_edge = set(lower_edge)
    n = 0
    for v in mesh.data.vertices:
        p = mw @ v.co
        if p.y > my + 0.6 * w:  # back of the head
            continue
        ax = abs(p.x - mx)
        if v.index in lower_edge:
            # along the lip line: full at the centre, 0 at the corners (corners stay closed)
            wt = 1 - smoothstep(0.2 * w, 0.48 * w, ax)
        elif p.z < mz - 1e-6:
            depth_below = mz - p.z
            half = 0.48 * w + 0.7 * depth_below          # the moving area widens towards the chin
            hx = 1 - smoothstep(0.55 * half, half, ax)
            vz = 1 - smoothstep(0.8 * w, 1.3 * w, depth_below)  # fade out below the chin
            wt = hx * vz
        else:
            continue
        if wt <= 0.01:
            continue
        w_head = 0.0
        if head_g:
            for g in v.groups:
                if g.group == head_g.index:
                    w_head = g.weight
        total = max(w_head, wt)
        jaw_g.add([v.index], wt, "REPLACE")
        if head_g:
            head_g.add([v.index], max(0.0, total - wt), "REPLACE")
        n += 1
    return n


def _mouth_shell(rig, m, tree, name, color, z_from, z_to, depth, roughness, arch=0.0):
    """A thin surface that follows the face from behind, attached to the head."""
    mx, my, mz, w = m["x"], m["y"], m["z"], m["width"]
    nx, nz = 20, 12
    xs = [mx - 0.5 * w + 1.0 * w * i / (nx - 1) for i in range(nx)]
    verts, faces, front = [], [], []
    for j in range(nz):
        for x in xs:
            # arch: the lower edge rises towards the sides (curved row like real teeth)
            u = min(abs(x - mx) / (0.5 * w), 1.0)
            lo = z_from * (1 - arch * u * u)
            z = mz + lo * w + (z_to - lo) * w * j / (nz - 1)
            y, ok = my, tree is None
            if tree is not None:
                hit = tree.ray_cast(Vector((x, my - 10.0, z)), Vector((0, 1, 0)))
                if hit[0] is not None:
                    y = hit[0].y
                    # only where the skin faces the camera (not under the chin)
                    ok = hit[1].y < -0.45 and abs(hit[0].y - my) < 0.6 * w
            # deeper towards the corners so skin sliding sideways never uncovers it
            ax = min(abs(x - mx) / (0.5 * w), 1.0)
            verts.append((x, y + (depth + 0.12 * ax * ax) * w, z))
            front.append(ok)
    for j in range(nz - 1):
        for i in range(nx - 1):
            a0 = j * nx + i
            quad = (a0, a0 + 1, a0 + nx + 1, a0 + nx)
            if all(front[k] for k in quad):
                faces.append(quad)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    t = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(t)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    t.data.materials.append(mat)
    keep = t.matrix_world.copy()
    t.parent = rig
    t.parent_type = "BONE"
    t.parent_bone = "Head"
    bpy.context.view_layer.update()
    t.matrix_world = keep
    return t


def add_teeth(rig, m, face=None):
    """Behind the slit: a white band at the top (teeth) and dark inside under it.

    Closed mouth: both hidden under the skin. Open mouth: white top + dark mouth
    show through the opening and grow with it. Real geometry, nothing painted.
    """
    from mathutils.bvhtree import BVHTree
    tree = None
    if face is not None:
        tree = BVHTree.FromObject(face, bpy.context.evaluated_depsgraph_get())
    teeth = _mouth_shell(rig, m, tree, "Teeth", (0.97, 0.97, 0.95, 1), -0.09, 0.08, 0.035, 0.35,
                         arch=0.75)
    inside = _mouth_shell(rig, m, tree, "MouthInside", (0.035, 0.012, 0.012, 1), -0.5, 0.08, 0.075, 0.9)
    return [teeth, inside]


def add_corners(rig, mesh, m):
    """Two bones on the mouth corners: out = wide ("ي"), in + forward = round ("و")."""
    mx, my, mz, w = m["x"], m["y"], m["z"], m["width"]
    C.select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    inv = rig.matrix_world.inverted()
    for name, sx in (("MouthL", 1), ("MouthR", -1)):
        if name in eb:
            continue
        p = Vector((mx + sx * 0.5 * w, my + 0.08 * w, mz))
        b = eb.new(name)
        b.head, b.tail = inv @ p, inv @ (p + Vector((0, 0, 0.25 * w)))
        b.roll = 0
        b.parent = eb["Head"]
        b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    # weights: soft round area around each corner, taken from the other bones
    mw = mesh.matrix_world
    groups = {g.index: g for g in mesh.vertex_groups}
    n = 0
    for name, sx in (("MouthL", 1), ("MouthR", -1)):
        g = mesh.vertex_groups.get(name) or mesh.vertex_groups.new(name=name)
        cx = mx + sx * 0.5 * w
        for v in mesh.data.vertices:
            p = mw @ v.co
            if p.y > my + 0.5 * w:
                continue
            d = ((p.x - cx) ** 2 + ((p.z - mz) * 1.3) ** 2) ** 0.5
            c = 1 - smoothstep(0.08 * w, 0.42 * w, d)
            if c <= 0.01:
                continue
            for vg in list(v.groups):
                gg = groups.get(vg.group)
                if gg is not None and gg.name != name and vg.weight > 0:
                    gg.add([v.index], vg.weight * (1 - c), "REPLACE")
            g.add([v.index], c, "REPLACE")
            n += 1
    return n


def read_audio(path):
    """(mono samples, rate)."""
    import numpy as np
    try:
        import aud
        snd = aud.Sound(path)
        rate = int(snd.specs[0])
        samples = np.asarray(snd.data(), dtype=np.float32)
    except Exception:  # noqa: BLE001
        import wave
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(-1, ch)
        if width == 1:
            samples -= 128
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, rate


def animate_shapes(rig, shapes, fps, max_deg, w):
    """Jaw + corners keyed from the voice analysis."""
    scene = bpy.context.scene
    scene.render.fps = fps
    n = len(shapes["open"])
    scene.frame_start = 1
    scene.frame_end = n + 1
    C.select_only(rig)
    bpy.ops.object.mode_set(mode="POSE")
    jaw = rig.pose.bones["Jaw"]
    jaw.rotation_mode = "XYZ"
    rot = (rig.matrix_world.to_3x3()).inverted()
    corners = []
    for name, sx in (("MouthL", 1), ("MouthR", -1)):
        pb = rig.pose.bones.get(name)
        if pb is not None:
            to_bone = pb.bone.matrix_local.to_3x3().inverted() @ rot
            corners.append((pb, sx, to_bone))
    for i in range(n + 1):
        o = shapes["open"][i] if i < n else 0.0
        wd = shapes["wide"][i] if i < n else 0.0
        rd = shapes["round"][i] if i < n else 0.0
        jaw.rotation_euler = (math.radians(max_deg) * o, 0, 0)
        jaw.keyframe_insert("rotation_euler", frame=i + 1)
        for pb, sx, to_bone in corners:
            # world move: sideways (out for wide, in for round) + forward for round
            dx = sx * (0.12 * wd - 0.12 * rd) * w
            dy = -0.05 * rd * w
            dz = -0.02 * o * w
            pb.location = to_bone @ Vector((dx, dy, dz))
            pb.keyframe_insert("location", frame=i + 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    if rig.animation_data and rig.animation_data.action:
        rig.animation_data.action.name = "Talk"


def read_loudness(path, fps):
    """Loudness per animation frame, 0..1."""
    import numpy as np
    samples, rate = None, None
    try:
        import aud
        snd = aud.Sound(path)
        rate = int(snd.specs[0])
        samples = np.asarray(snd.data(), dtype=np.float32)
    except Exception:  # noqa: BLE001
        import wave
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(-1, ch)
        if width == 1:
            samples -= 128
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    hop = max(1, int(rate / fps))
    n = len(samples) // hop
    if n == 0:
        raise RuntimeError("The audio file is empty or too short")
    rms = np.sqrt((samples[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    ref = np.percentile(rms, 95) or 1.0
    level = np.clip(rms / ref, 0, 1)
    gate = 0.08  # silence -> closed mouth
    level = np.clip((level - gate) / (1 - gate), 0, 1)
    out, cur = [], 0.0
    for x in level:  # fast open, slower close
        cur = cur + (x - cur) * (0.65 if x > cur else 0.35)
        out.append(float(cur))
    return out, len(samples) / rate


def animate(rig, levels, fps, max_deg):
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = len(levels) + 1
    C.select_only(rig)
    bpy.ops.object.mode_set(mode="POSE")
    pb = rig.pose.bones["Jaw"]
    pb.rotation_mode = "XYZ"
    for i, lv in enumerate(levels):
        pb.rotation_euler = (math.radians(max_deg) * lv, 0, 0)
        pb.keyframe_insert("rotation_euler", frame=i + 1)
    pb.rotation_euler = (0, 0, 0)
    pb.keyframe_insert("rotation_euler", frame=len(levels) + 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    if rig.animation_data and rig.animation_data.action:
        rig.animation_data.action.name = "Talk"


def main(args):
    C.reset_scene()
    rig, mesh = import_rigged(args["input"])
    f = face_frame(rig, mesh)
    mouth = args.get("mouth")
    teeth = None
    if mouth:  # mouth line found on the texture: real slit + white teeth behind it
        f = dict(f, cx=mouth["x"], mouth_z=mouth["z"], front_y=mouth["y"])
        add_jaw_at_mouth(rig, f, mouth)
        lower = make_slit(mesh, mouth)
        n = slit_jaw_weights(mesh, mouth, lower)
        teeth = add_teeth(rig, mouth, mesh)
        add_corners(rig, mesh, mouth)
    else:
        add_jaw(rig, f)
        n = jaw_weights(mesh, f)
    fps = int(args.get("fps", 30))
    max_deg = float(args.get("max_open_degrees", 6))
    shapes = None
    if mouth:
        try:
            import visemes
            samples, rate = read_audio(args["audio"])
            shapes = visemes.analyse(samples, rate, fps)
        except Exception as e:  # noqa: BLE001
            print("visemes failed, loudness only:", e)
    if shapes is not None:
        animate_shapes(rig, shapes, fps, max_deg, mouth["width"])
        levels, seconds = shapes["open"], shapes["seconds"]
    else:
        levels, seconds = read_loudness(args["audio"], fps)
        animate(rig, levels, fps, max_deg)

    # sound in the timeline for preview/render inside Blender
    scene = bpy.context.scene
    try:
        if not scene.sequence_editor:
            scene.sequence_editor_create()
        seqs = getattr(scene.sequence_editor, "strips", None) or scene.sequence_editor.sequences
        seqs.new_sound("voice", args["audio"], 1, 1)
    except Exception:  # noqa: BLE001
        pass

    out, name = args["outdir"], args.get("name", "model_talking")
    os.makedirs(out, exist_ok=True)
    files = []
    C.select_only(rig, mesh)
    mesh.select_set(True)
    for t in teeth or []:
        t.select_set(True)
    glb = os.path.join(out, f"{name}.glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_animations=True)
    files.append(glb)
    fbx = os.path.join(out, f"{name}.fbx")
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, add_leaf_bones=False, bake_anim=True,
                             bake_anim_use_all_actions=False, path_mode="COPY", embed_textures=True)
    files.append(fbx)
    blend = os.path.join(out, f"{name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True)
    files.append(blend)
    return {"files": files, "jaw_vertices": n, "frames": len(levels), "seconds": round(seconds, 2),
            "fps": fps, "slit": bool(mouth), "visemes": shapes is not None}


if __name__ == "__main__":
    C.run(main)
