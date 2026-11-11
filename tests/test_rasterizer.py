"""Checks the pure-PyTorch rasteriser against a direct (slow) port of the CUDA kernel."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio3d" / "vendor"))
import custom_rasterizer as cr  # noqa: E402
from custom_rasterizer import render  # noqa: E402

MAXINT = 2147483647


def reference(V, F, H, W):
    V = V.numpy().astype(np.float32)
    F = F.numpy()
    zbuf = np.full(H * W, MAXINT * MAXINT + MAXINT - 1, dtype=np.uint64)

    def scr(v):
        return np.float32([(v[0] / v[3] * 0.5 + 0.5) * (W - 1) + 0.5,
                           (0.5 + 0.5 * v[1] / v[3]) * (H - 1) + 0.5,
                           v[2] / v[3] * np.float32(0.49999) + 0.5])

    def area(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])

    def bary(a, b, c, p):
        ar = area(a, b, c)
        if ar == 0:
            return [-1, -1, -1]
        be = area(a, p, c) / ar
        ga = area(a, b, p) / ar
        return [1 - be - ga, be, ga]

    for i, f in enumerate(F):
        a, b, c = scr(V[f[0]]), scr(V[f[1]]), scr(V[f[2]])
        xmin, xmax = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
        ymin, ymax = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
        px = int(xmin)
        while px < xmax + 1:
            if 0 <= px < W:
                py = int(ymin)
                while py < ymax + 1:
                    if 0 <= py < H:
                        bc = bary(a, b, c, np.float32([px + 0.5, py + 0.5]))
                        if all(0 <= t <= 1 for t in bc):
                            d = bc[0] * a[2] + bc[1] * b[2] + bc[2] * c[2]
                            zq = int(d * (2 << 17))
                            if zq >= 0:
                                tok = np.uint64(zq * MAXINT + i + 1)
                                pix = py * W + px
                                zbuf[pix] = min(zbuf[pix], tok)
                    py += 1
            px += 1
    f = (zbuf % MAXINT).astype(np.int64)
    return np.where(f == MAXINT - 1, 0, f).reshape(H, W)


def _random_mesh(n_faces, seed):
    g = torch.Generator().manual_seed(seed)
    V = torch.rand(n_faces * 3, 4, generator=g) * 2 - 1
    V[:, 2] = V[:, 2] * 0.9
    V[:, 3] = 1.0 + torch.rand(n_faces * 3, generator=g) * 0.5
    F = torch.arange(n_faces * 3, dtype=torch.int32).view(n_faces, 3)
    return V, F


def test_matches_reference():
    for seed in range(3):
        V, F = _random_mesh(25, seed)
        H, W = 40, 56
        fi, bc = cr.rasterize(V[None], F, (H, W))
        ref = reference(V, F, H, W)
        mismatch = (fi.numpy() != ref).mean()
        assert mismatch < 0.002, mismatch
        hit = fi.numpy() > 0
        assert hit.any()
        s = bc.numpy()[hit].sum(-1)
        assert np.allclose(s, 1.0, atol=1e-4)
        assert (bc.numpy()[~hit] == 0).all()


def test_chunking_gives_same_result():
    V, F = _random_mesh(60, 7)
    a = cr.rasterize(V[None], F, (64, 64))[0]
    old = render.CHUNK_CANDIDATES
    render.CHUNK_CANDIDATES = 50
    try:
        b = cr.rasterize(V[None], F, (64, 64))[0]
    finally:
        render.CHUNK_CANDIDATES = old
    assert torch.equal(a, b)


def test_interpolate_shape():
    V, F = _random_mesh(10, 1)
    fi, bc = cr.rasterize(V[None], F, (16, 16))
    col = torch.rand(1, V.shape[0], 2)
    out = cr.interpolate(col, fi, bc, F)
    assert out.shape == (1, 16, 16, 2)
