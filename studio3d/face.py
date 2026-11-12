"""Find the mouth of a generated character from its texture (front view).

1. Render the front view with the real texture (sharp, per-pixel).
2. Skin colour = the dominant non-hair colour in the upper body.
3. Face = skin pixels; its holes are the eyes, brows and mouth.
4. Eyes = the two biggest holes side by side; mouth = the hole below them, centred.
Returns the mouth line in model coordinates (glTF Y-up): centre x, y and width.
"""
from __future__ import annotations

import numpy as np


def _components(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.where(lab[sl] == i)
        out.append({"area": len(ys), "y0": sl[0].start, "y1": sl[0].stop, "x0": sl[1].start, "x1": sl[1].stop,
                    "cy": sl[0].start + ys.mean(), "cx": sl[1].start + xs.mean()})
    return out


def find_mouth(mesh, size: int = 1024, debug: dict | None = None) -> dict | None:
    from scipy import ndimage
    from .preview import render_textured

    img, info = render_textured(mesh, 0, 0, size)
    a = np.asarray(img).astype(np.float32) / 255.0
    body = info["mask"]
    rows = np.where(body.any(axis=1))[0]
    if len(rows) < 20:
        return None
    top, bottom = rows.min(), rows.max()
    H = bottom - top
    head = np.zeros_like(body)
    head[top: top + int(0.45 * H)] = body[top: top + int(0.45 * H)]

    lum = a.mean(axis=2)
    cand = head & (lum > 0.35) & (lum < 0.97)
    if cand.sum() < 100:
        return None
    # skin = most common colour (quantised) among bright-enough head pixels
    q = (a[cand] * 15).astype(int)
    keys, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
    k = keys[np.argmax(counts)]
    skin = np.array([k // 256, (k // 16) % 16, k % 16], dtype=np.float32) / 15 + 1 / 30
    dist = np.abs(a - skin).max(axis=2)
    skin_mask = head & (dist < 0.13)
    skin_mask = ndimage.binary_opening(skin_mask, iterations=1)
    # the face = biggest skin blob in the head
    lab, n = ndimage.label(skin_mask)
    if n == 0:
        return None
    sizes = ndimage.sum(skin_mask, lab, range(1, n + 1))
    face = lab == (1 + int(np.argmax(sizes)))
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(face, iterations=3))
    holes = filled & ~face
    holes = ndimage.binary_opening(holes, iterations=1)
    comps = [c for c in _components(holes) if c["area"] > 15]
    fy, fx = np.where(face)
    face_box = (fx.min(), fy.min(), fx.max(), fy.max())
    fw = face_box[2] - face_box[0]
    if not comps or fw < 20:
        return None

    # eyes: two big holes at a similar height, one each side of the face centre
    cx_face = (face_box[0] + face_box[2]) / 2
    eyes = None
    big = sorted(comps, key=lambda c: -c["area"])[:6]
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            a1, a2 = big[i], big[j]
            if abs(a1["cy"] - a2["cy"]) < 0.08 * fw and (a1["cx"] - cx_face) * (a2["cx"] - cx_face) < 0 \
                    and 0.15 * fw < abs(a1["cx"] - a2["cx"]) < 0.8 * fw:
                eyes = (a1, a2)
                break
        if eyes:
            break
    if eyes:
        eye_y = (eyes[0]["cy"] + eyes[1]["cy"]) / 2
        eye_d = abs(eyes[0]["cx"] - eyes[1]["cx"])
        mid_x = (eyes[0]["cx"] + eyes[1]["cx"]) / 2
    else:  # no eyes found: assume the upper half of the face
        eye_y = face_box[1] + 0.45 * (face_box[3] - face_box[1])
        eye_d = 0.4 * fw
        mid_x = cx_face

    # mouth: hole below the eyes, near the centre line (the nose is skin-coloured, so it is not a hole)
    mouth = None
    below = [c for c in comps if c not in (eyes or ()) and eye_y + 0.35 * eye_d < c["cy"] < eye_y + 1.6 * eye_d
             and abs(c["cx"] - mid_x) < 0.35 * eye_d]
    if below:
        mouth = max(below, key=lambda c: c["area"] / (1 + abs(c["cx"] - mid_x)))
    if mouth:
        # the lip line = darkest row inside the mouth area; its dark span = mouth width
        y0 = max(0, int(mouth["y0"] - 0.1 * eye_d))
        y1 = min(size, int(mouth["y1"] + 0.1 * eye_d))
        x0 = max(0, int(mouth["x0"] - 0.1 * eye_d))
        x1 = min(size, int(mouth["x1"] + 0.1 * eye_d))
        skin_lum = float(lum[face].mean())
        dark = (lum[y0:y1, x0:x1] < skin_lum - 0.05) & (face | holes)[y0:y1, x0:x1]
        widths = dark.sum(axis=1).astype(float)
        widths = np.convolve(widths, np.ones(3) / 3, "same")
        ly = int(np.argmax(widths + np.arange(len(widths)) * 1e-3))  # widest dark line (lower wins ties)
        my = y0 + ly
        cols = np.where(dark[max(0, ly - 1): ly + 2].any(axis=0))[0]
        if len(cols) >= 3:
            mx = x0 + (cols.min() + cols.max()) / 2
            mw = max(cols.max() - cols.min(), 0.25 * eye_d)
        else:
            mx, mw = mouth["cx"], max(mouth["x1"] - mouth["x0"], 0.3 * eye_d)
        found = True
    else:  # fall back to typical proportions
        my, mx, mw = eye_y + 0.8 * eye_d, mid_x, 0.45 * eye_d
        found = False

    s, c = info["scale"], info["center"]
    to_x = lambda px: (px / size * 2 - 1) * s + c[0]  # noqa: E731
    to_y = lambda py: ((1 - py / size) * 2 - 1) * s + c[1]  # noqa: E731
    # front surface depth at the mouth: max z of vertices near the mouth
    mxm, mym, half = to_x(mx), to_y(my), mw / size * s
    v = np.asarray(mesh.vertices)
    near = v[(np.abs(v[:, 0] - mxm) < half) & (np.abs(v[:, 1] - mym) < half)]
    mz = float(near[:, 2].max()) if len(near) else float(v[:, 2].max())
    if debug is not None:
        debug.update({"image": img, "px": (mx, my, mw), "eyes": eyes, "face": face_box})
    return {"x": float(mxm), "y": float(mym), "z": mz, "width": float(mw / size * 2 * s),
            "eye_distance": float(eye_d / size * 2 * s), "found": found}
