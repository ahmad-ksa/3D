"""Mouth shapes from a voice recording, without any AI model (works offline, any language).

For every animation frame we estimate:
  open  - how far the jaw opens (loudness, more for open vowels like "آ")
  wide  - lips stretched sideways ("ي", "ee", "س")
  round - lips pushed together/forward ("و", "oo", "o")
from the loudness and the first two formants (F1/F2) of the voice, found with LPC.
Pure numpy, so it runs inside Blender too.
"""
import numpy as np


def _resample(x, rate, target=11025):
    if rate <= target:
        return x, rate
    # anti-alias low-pass (windowed sinc) before taking fewer samples
    cut = 0.45 * target / rate
    k = np.arange(-64, 65)
    h = np.sinc(2 * cut * k) * np.hamming(len(k))
    x = np.convolve(x, h / h.sum(), mode="same")
    n = int(len(x) * target / rate)
    t = np.linspace(0, len(x) - 1, n)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32), target


def _lpc(frame, order):
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 1e-9:
        return None
    r = r.copy()
    r[0] *= 1.0001
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):  # Levinson-Durbin
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / err
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        err *= (1 - k * k)
        if err <= 0:
            return None
    return a


def formants(frame, rate):
    """First two formant frequencies (Hz) of a voiced frame, or (None, None)."""
    if len(frame) < 64:
        return None, None
    x = np.append(frame[0], frame[1:] - 0.63 * frame[:-1])  # pre-emphasis
    x = x * np.hamming(len(x))
    a = _lpc(x, 2 + rate // 1000)
    if a is None:
        return None, None
    roots = [z for z in np.roots(a) if np.imag(z) > 0]
    found = []
    for z in roots:
        f = np.angle(z) * rate / (2 * np.pi)
        bw = -0.5 * rate / (2 * np.pi) * np.log(max(abs(z), 1e-9))
        if 200 < f < 4000 and bw < 300:
            found.append(f)
    found.sort()
    if len(found) < 2:
        return None, None
    return found[0], found[1]


def _smooth(v, up, down):
    out, cur = [], 0.0
    for x in v:
        cur = cur + (x - cur) * (up if x > cur else down)
        out.append(cur)
    return np.array(out, dtype=np.float32)


def analyse(samples, rate, fps):
    """-> dict of per-frame lists: open, wide, round (0..1), seconds."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    seconds = len(x) / float(rate)
    x, rate = _resample(x, int(rate))
    hop = max(1, int(rate / fps))
    win = max(hop, int(0.03 * rate))
    n = max(1, int(np.ceil(len(x) / hop)))
    rms = np.zeros(n, np.float32)
    f1 = np.full(n, np.nan)
    f2 = np.full(n, np.nan)
    for i in range(n):
        c = i * hop + hop // 2
        seg = x[max(0, c - win // 2):c + win // 2]
        if len(seg) == 0:
            continue
        rms[i] = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.percentile(rms, 95)) if rms.max() > 0 else 0.0
    level = np.clip(rms / (peak or 1.0), 0, 1)
    gate = 0.08  # silence -> closed mouth (م / ب / pauses)
    level = np.clip((level - gate) / (1 - gate), 0, 1)
    for i in range(n):
        if level[i] < 0.12:
            continue
        c = i * hop + hop // 2
        seg = x[max(0, c - win // 2):c + win // 2]
        a, b = formants(seg, rate)
        if a is not None:
            f1[i], f2[i] = a, b
    # fill gaps (consonants / unclear frames) with the neighbours
    idx = np.arange(n)
    ok = ~np.isnan(f1)
    if ok.sum() >= 2:
        f1 = np.interp(idx, idx[ok], f1[ok])
        f2 = np.interp(idx, idx[ok], f2[ok])
    else:
        f1 = np.full(n, 550.0)
        f2 = np.full(n, 1500.0)
    openness = np.clip((f1 - 300) / 450, 0, 1)        # آ high F1, ي/و low F1
    wide = np.clip((f2 - 1500) / 800, 0, 1)            # ي: high F2
    rnd = np.clip((1250 - f2) / 450, 0, 1)             # و: low F2
    voiced = level > 0.05
    open_ = level * (0.55 + 0.45 * openness)
    wide = np.where(voiced, wide, 0) * np.clip(level * 2, 0, 1)
    rnd = np.where(voiced, rnd, 0) * np.clip(level * 2, 0, 1)
    return {
        "open": _smooth(open_, 0.65, 0.35).tolist(),
        "wide": _smooth(wide, 0.5, 0.3).tolist(),
        "round": _smooth(rnd, 0.5, 0.3).tolist(),
        "seconds": seconds,
    }
