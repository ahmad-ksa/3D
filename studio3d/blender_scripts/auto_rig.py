"""Automatic humanoid skeleton (runs inside Blender, headless).

Works for characters standing in T-pose or A-pose (what 3D generators produce).
1. Measure the body by slicing the mesh horizontally: crotch height and leg
   positions (where the slice splits in two), shoulder height and hand reach
   (widest slice in the upper body), torso width.
2. Build a standard humanoid armature with Unity/Mixamo-style bone names.
3. Skin it with Blender's automatic (heat) weights; any vertex left without a
   weight is bound to its nearest bone, so nothing stays behind when posing.
4. Optional quick pose test (raises the arms) to verify the skinning.
5. Export FBX (Unity/Unreal/Mixamo) and GLB.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import _common as C  # noqa: E402


# ---------------------------------------------------------------------------
# Body measurement
# ---------------------------------------------------------------------------
def _verts(obj):
    mw = obj.matrix_world
    return [mw @ v.co for v in obj.data.vertices]


def _slice(vs, z, band):
    return [v for v in vs if abs(v.z - z) <= band]


def _clusters_x(points, gap):
    """Split points into groups along X wherever there is an empty gap > gap."""
    xs = sorted(p.x for p in points)
    if not xs:
        return []
    groups, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > gap:
            groups.append(cur)
            cur = [x]
        else:
            cur.append(x)
    groups.append(cur)
    return groups


def _clusters_pts(points, gap):
    """Like _clusters_x but returns the points of each group."""
    pts = sorted(points, key=lambda p: p.x)
    if not pts:
        return []
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p.x - cur[-1].x > gap:
            groups.append(cur)
            cur = [p]
        else:
            cur.append(p)
    groups.append(cur)
    return groups


def _median(vals):
    v = sorted(vals)
    return v[len(v) // 2] if v else 0.0


def _center_y(points, default):
    return sum(p.y for p in points) / len(points) if points else default


def measure(obj):
    vs = _verts(obj)
    xs = [v.x for v in vs]
    ys = [v.y for v in vs]
    zs = [v.z for v in vs]
    zmin, zmax = min(zs), max(zs)
    H = zmax - zmin
    xmin, xmax = min(xs), max(xs)
    cx = (xmin + xmax) / 2
    cy = (min(ys) + max(ys)) / 2
    band = H * 0.012
    gap = H * 0.015

    def z_at(f):
        return zmin + f * H

    # legs: walk UP from the feet while the slice is split into a left and a right leg.
    # The top of that run is the crotch - or the hem of a dress/skirt that hides the thighs.
    def leg_pair(f):
        groups = [g for g in _clusters_pts(_slice(vs, z_at(f), band), gap) if len(g) >= 6]
        left = [g for g in groups if _median([p.x for p in g]) > cx]
        right = [g for g in groups if _median([p.x for p in g]) <= cx]
        if not left or not right:
            return None
        lg = min(left, key=lambda g: abs(_median([p.x for p in g]) - cx))
        rg = min(right, key=lambda g: abs(_median([p.x for p in g]) - cx))
        return lg, rg

    top_f, started, f = None, False, 0.03
    while f <= 0.62:
        pair = leg_pair(f)
        if pair:
            started, top_f = True, f
        elif started:
            break
        f += 0.01
    crotch_f = top_f if top_f is not None else 0.45
    # leg positions measured on the shins (always visible)
    shin = leg_pair(min(0.12, crotch_f * 0.5)) or leg_pair(crotch_f * 0.5)
    if shin:
        legs = tuple((_median([p.x for p in g]), _median([p.y for p in g])) for g in shin)
    else:
        legs = ((cx + 0.09 * H, cy), (cx - 0.09 * H, cy))

    # shoulders: widest slice in the upper body (arms in T/A pose)
    best_w, sh_f = -1, 0.78
    f = 0.55
    while f <= 0.92:
        sl = _slice(vs, z_at(f), band)
        if sl:
            w = max(p.x for p in sl) - min(p.x for p in sl)
            if w > best_w + 1e-6:
                best_w, sh_f = w, f
        f += 0.01
    sh_slice = _slice(vs, z_at(sh_f), band * 2)
    hand_l = max(p.x for p in sh_slice) if sh_slice else xmax
    hand_r = min(p.x for p in sh_slice) if sh_slice else xmin
    arm_span = hand_l - hand_r
    a_pose = arm_span < 0.75 * H  # arms not straight out -> A-pose

    # torso half-width at chest level (central X cluster)
    chest_z = z_at(min(sh_f, 0.8) - 0.06)
    torso = _clusters_pts(_slice(vs, chest_z, band), gap)
    if torso:
        central = min(torso, key=lambda g: abs(_median([p.x for p in g]) - cx))
        xs_c = [p.x for p in central]
        half_torso = max(0.05 * H, (max(xs_c) - min(xs_c)) / 2)
        body_y = _median([p.y for p in central])
    else:
        half_torso, body_y = 0.1 * H, cy

    if a_pose:  # hands are lower: find the outermost points instead
        hand_l, hand_r = xmax, xmin

    return {
        "H": H, "zmin": zmin, "cx": cx, "cy": cy,
        "crotch": z_at(crotch_f), "legs": legs, "shoulder": z_at(sh_f),
        "hand_l": hand_l, "hand_r": hand_r, "half_torso": half_torso, "a_pose": a_pose, "body_y": body_y,
        "vs": vs,
    }


def _hand_z(m, side_x):
    """Height of the hand end of the arm (lower than the shoulder in A-pose)."""
    if not m["a_pose"]:
        return m["shoulder"]
    pts = [v for v in m["vs"] if (v.x - m["cx"]) * (1 if side_x > m["cx"] else -1) >
           0.9 * abs(side_x - m["cx"])]
    return sum(p.z for p in pts) / len(pts) if pts else m["shoulder"] - 0.1 * m["H"]


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------
def build_armature(m):
    H, cx = m["H"], m["cx"]
    vs = m["vs"]

    by = m["body_y"]

    def yc(z):  # spine, neck and head stay in one straight column (hair/ponytails must not bend it)
        return by

    hips_z = m["crotch"] + 0.04 * H
    spine_z = hips_z + 0.10 * H
    neck_z = m["shoulder"] + 0.03 * H
    chest_z = spine_z + 0.6 * (neck_z - spine_z)
    head_z = neck_z + 0.06 * H
    top_z = m["zmin"] + H

    arm = bpy.data.armatures.new("Armature")
    rig = bpy.data.objects.new("Armature", arm)
    bpy.context.scene.collection.objects.link(rig)
    C.select_only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    bone("Hips", (cx, yc(hips_z), hips_z), (cx, yc(spine_z), spine_z))
    bone("Spine", (cx, yc(spine_z), spine_z), (cx, yc(chest_z), chest_z), "Hips", True)
    bone("Chest", (cx, yc(chest_z), chest_z), (cx, yc(neck_z), neck_z), "Spine", True)
    bone("Neck", (cx, yc(neck_z), neck_z), (cx, yc(head_z), head_z), "Chest", True)
    bone("Head", (cx, yc(head_z), head_z), (cx, yc(head_z), top_z), "Neck", True)

    for side, sign, hand_x, leg_x in (("Left", 1, m["hand_l"], m["legs"][0]),
                                      ("Right", -1, m["hand_r"], m["legs"][1])):
        sz = m["shoulder"]
        clav_start = (cx + sign * 0.02 * H, yc(chest_z), chest_z + 0.7 * (neck_z - chest_z))
        sh_x = cx + sign * m["half_torso"] * 1.05
        hz = _hand_z(m, hand_x)
        shoulder = Vector((sh_x, yc(sz), sz))
        hand_end = Vector((hand_x - sign * 0.01 * H, yc(sz), hz))
        wrist = shoulder.lerp(hand_end, 0.82)
        elbow = shoulder.lerp(wrist, 0.5)
        elbow.y += 0.01 * H  # slight bend backwards helps IK solvers
        bone(f"{side}Shoulder", clav_start, shoulder, "Chest")
        bone(f"{side}UpperArm", shoulder, elbow, f"{side}Shoulder", True)
        bone(f"{side}LowerArm", elbow, wrist, f"{side}UpperArm", True)
        bone(f"{side}Hand", wrist, hand_end, f"{side}LowerArm", True)

        lx, ly = leg_x
        hip = Vector((lx, by, m["crotch"]))
        ankle_z = m["zmin"] + 0.06 * H
        knee = Vector((lx, (by + ly) / 2 - 0.01 * H, (m["crotch"] + ankle_z) / 2))
        ankle = Vector((lx, ly, ankle_z))
        toe = Vector((lx, ly - 0.08 * H, m["zmin"] + 0.01 * H))
        bone(f"{side}UpperLeg", hip, knee, "Hips")
        bone(f"{side}LowerLeg", knee, ankle, f"{side}UpperLeg", True)
        bone(f"{side}Foot", ankle, toe, f"{side}LowerLeg", True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


# ---------------------------------------------------------------------------
# Skinning
# ---------------------------------------------------------------------------
def _dist_point_segment(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / (ab.length_squared or 1e-12)))
    return (a + ab * t - p).length


def skin(obj, rig):
    C.select_only(obj, rig)
    obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    warnings = []
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as e:
        warnings.append(f"heat weights failed: {e}")
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")

    # bind any unweighted vertex to its nearest bone
    bones = [(b.name, rig.matrix_world @ b.head_local, rig.matrix_world @ b.tail_local)
             for b in rig.data.bones]
    groups = {g.name: g for g in obj.vertex_groups}
    for name, _, _ in bones:
        if name not in groups:
            groups[name] = obj.vertex_groups.new(name=name)
    mw = obj.matrix_world
    fixed = 0
    for v in obj.data.vertices:
        if sum(g.weight for g in v.groups) > 1e-4:
            continue
        p = mw @ v.co
        name = min(bones, key=lambda b: _dist_point_segment(p, b[1], b[2]))[0]
        groups[name].add([v.index], 1.0, "REPLACE")
        fixed += 1
    if fixed:
        warnings.append(f"{fixed} vertices bound to nearest bone")
    return warnings


def pose_test(rig, obj):
    """Raise both arms and check the mesh actually moves with them."""
    import numpy as np  # bundled with Blender

    def coords():
        dg = bpy.context.evaluated_depsgraph_get()
        me = obj.evaluated_get(dg).to_mesh()
        arr = np.array([v.co[:] for v in me.vertices])
        obj.evaluated_get(dg).to_mesh_clear()
        return arr

    rest = coords()
    for side, sign in (("Left", 1), ("Right", -1)):
        pb = rig.pose.bones[f"{side}UpperArm"]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, sign * math.radians(-40), 0)
    bpy.context.view_layer.update()
    moved = np.linalg.norm(coords() - rest, axis=1)
    for pb in rig.pose.bones:
        pb.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    return float((moved > 1e-4).mean())


def main(args):
    C.reset_scene()
    obj = C.import_model(args["input"])
    max_faces = int(args.get("max_faces", 60000))
    n = len(obj.data.polygons)
    if n > max_faces:  # heat weighting is slow/unstable on very dense meshes
        C.select_only(obj)
        mod = obj.modifiers.new("dec", "DECIMATE")
        mod.ratio = max_faces / n
        bpy.ops.object.modifier_apply(modifier=mod.name)
    # merge duplicate vertices so heat weights can flow across the surface
    C.select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.object.mode_set(mode="OBJECT")

    m = measure(obj)
    rig = build_armature(m)
    warnings = skin(obj, rig)
    moved = pose_test(rig, obj)
    files = C.export([rig, obj], args["outdir"], args.get("name", "model_rigged"),
                     args.get("formats", ["fbx", "glb"]))
    return {"files": files, "bones": len(rig.data.bones), "warnings": warnings,
            "a_pose": m["a_pose"], "arm_test_moved_fraction": round(moved, 3),
            **C.face_stats(obj)}


if __name__ == "__main__":
    C.run(main)
