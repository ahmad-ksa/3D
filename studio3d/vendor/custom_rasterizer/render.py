"""Vectorised z-buffer triangle rasteriser written with PyTorch ops only.

It mirrors the behaviour of Hunyuan3D-2's CUDA kernel
(hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer_gpu.cu):

* vertices arrive in clip space (x, y, z, w);
* pixel centres are sampled at (px + 0.5, py + 0.5);
* the nearest triangle wins through a 64-bit token ``depth_q * MAXINT + face_id + 1``;
* returned face indices are 1-based (0 = empty) and barycentrics are
  perspective-corrected.
"""
import torch

MAXINT = 2147483647
_EMPTY_TOKEN = MAXINT * MAXINT + (MAXINT - 1)
_DEPTH_SCALE = 2 << 17
# Max number of candidate (face, pixel) pairs evaluated at once. Keeps memory
# bounded (~300 MB peak) on 8 GB GPUs.
CHUNK_CANDIDATES = 4_000_000


def _to_screen(V, width, height):
    w = V[:, 3]
    sx = (V[:, 0] / w * 0.5 + 0.5) * (width - 1) + 0.5
    sy = (0.5 + 0.5 * V[:, 1] / w) * (height - 1) + 0.5
    sz = V[:, 2] / w * 0.49999 + 0.5
    return sx, sy, sz, w


def _area2(ax, ay, bx, by, cx, cy):
    return (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)


def _barycentric(x, y, px, py):
    """x, y: (..., 3) triangle screen coords; px, py: (...) sample points."""
    ax, bx, cx = x[..., 0], x[..., 1], x[..., 2]
    ay, by, cy = y[..., 0], y[..., 1], y[..., 2]
    area = _area2(ax, ay, bx, by, cx, cy)
    valid = area != 0
    inv = 1.0 / torch.where(valid, area, torch.ones_like(area))
    beta = _area2(ax, ay, px, py, cx, cy) * inv
    gamma = _area2(ax, ay, bx, by, px, py) * inv
    alpha = 1.0 - beta - gamma
    return alpha, beta, gamma, valid


def rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0):
    """Same signature/outputs as ``custom_rasterizer.rasterize``.

    pos: (1, N, 4) clip-space vertices. tri: (F, 3) int. resolution: (H, W).
    Returns findices (H, W) int32 (1-based, 0 = background) and
    barycentric (H, W, 3) float32.
    """
    assert pos.device == tri.device
    V = pos[0].float()
    F = tri.long()
    height, width = int(resolution[0]), int(resolution[1])
    device = V.device

    sx, sy, sz, w = _to_screen(V, width, height)
    fx = sx[F]  # (F, 3)
    fy = sy[F]
    fz = sz[F]

    finite = torch.isfinite(fx).all(1) & torch.isfinite(fy).all(1) & torch.isfinite(fz).all(1)
    fx_s = torch.where(finite[:, None], fx, torch.zeros_like(fx))
    fy_s = torch.where(finite[:, None], fy, torch.zeros_like(fy))

    # Same pixel loop bounds as the CUDA kernel: px from trunc(min) while px < max + 1.
    x0 = torch.trunc(fx_s.min(1).values).clamp(min=0).long()
    x1 = (torch.ceil(fx_s.max(1).values + 1) - 1).clamp(max=width - 1).long()
    y0 = torch.trunc(fy_s.min(1).values).clamp(min=0).long()
    y1 = (torch.ceil(fy_s.max(1).values + 1) - 1).clamp(max=height - 1).long()
    bw = (x1 - x0 + 1).clamp(min=0)
    bh = (y1 - y0 + 1).clamp(min=0)
    counts = torch.where(finite, bw * bh, torch.zeros_like(bw))

    zbuf = torch.full((height * width,), _EMPTY_TOKEN, dtype=torch.int64, device=device)

    face_ids = torch.nonzero(counts > 0).flatten()
    if face_ids.numel() > 0:
        c_all = counts[face_ids]
        csum = torch.cumsum(c_all, 0)
        csum_cpu = csum.cpu()
        n_faces = face_ids.numel()
        start = 0
        base = 0
        while start < n_faces:
            # largest end such that csum[end-1] - base <= budget (at least one face)
            end = int(torch.searchsorted(csum_cpu, base + CHUNK_CANDIDATES, right=True))
            end = max(end, start + 1)
            fid = face_ids[start:end]
            c = c_all[start:end]
            total = int(c.sum())
            local_face = torch.repeat_interleave(torch.arange(fid.numel(), device=device), c)
            offsets = torch.cumsum(c, 0) - c
            local = torch.arange(total, device=device) - torch.repeat_interleave(offsets, c)
            gf = fid[local_face]
            w_box = bw[gf]
            px = x0[gf] + local % w_box
            py = y0[gf] + torch.div(local, w_box, rounding_mode='floor')

            tx = fx[gf]
            ty = fy[gf]
            alpha, beta, gamma, valid = _barycentric(tx, ty, px.float() + 0.5, py.float() + 0.5)
            inside = valid & (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1) & (gamma >= 0) & (gamma <= 1)

            tz = fz[gf]
            depth = alpha * tz[:, 0] + beta * tz[:, 1] + gamma * tz[:, 2]
            zq = torch.trunc(depth * _DEPTH_SCALE)
            # In the CUDA kernel tokens are unsigned: negative depths wrap to huge values and never win.
            inside = inside & (zq >= 0) & torch.isfinite(zq)
            if inside.any():
                token = zq[inside].long() * MAXINT + (gf[inside] + 1)
                pix = py[inside] * width + px[inside]
                zbuf.scatter_reduce_(0, pix, token, reduce='amin')

            base = int(csum_cpu[end - 1])
            start = end

    f = zbuf % MAXINT
    hit = f != (MAXINT - 1)
    findices = torch.where(hit, f, torch.zeros_like(f)).to(torch.int32)

    # Perspective-correct barycentrics at pixel centres for the winning faces.
    bary = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    hit_idx = torch.nonzero(hit).flatten()
    if hit_idx.numel() > 0:
        face = (f[hit_idx] - 1).long()
        tri_v = F[face]
        px = (hit_idx % width).float() + 0.5
        py = torch.div(hit_idx, width, rounding_mode='floor').float() + 0.5
        a, b, g, _ = _barycentric(sx[tri_v], sy[tri_v], px, py)
        wv = w[tri_v]
        a = a / wv[:, 0]
        b = b / wv[:, 1]
        g = g / wv[:, 2]
        norm = 1.0 / (a + b + g)
        bary[hit_idx] = torch.stack([a * norm, b * norm, g * norm], dim=1)

    return findices.view(height, width), bary.view(height, width, 3)


def interpolate(col, findices, barycentric, tri):
    """Identical to the original implementation."""
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
