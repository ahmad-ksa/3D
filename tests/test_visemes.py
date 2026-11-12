import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio3d" / "blender_scripts"))
import visemes  # noqa: E402


def _vowel(rate, formants, dur=0.5, f0=200):
    from scipy.signal import lfilter
    n = int(dur * rate)
    y = np.zeros(n)
    y[::rate // f0] = 1
    for f, bw in zip(formants, [80, 100, 120]):
        r = np.exp(-np.pi * bw / rate)
        th = 2 * np.pi * f / rate
        y = lfilter([1], [1, -2 * r * np.cos(th), r * r], y)
    return y / np.abs(y).max()


def test_vowels_give_open_wide_round():
    for rate in (16000, 44100):
        z = np.zeros(rate // 4)
        x = np.concatenate([_vowel(rate, [800, 1250, 2600]), z, _vowel(rate, [300, 2300, 3000]), z,
                            _vowel(rate, [320, 800, 2400])])
        r = visemes.analyse(x + np.random.randn(len(x)) * 0.01, rate, 30)
        o, w, ro = (np.array(r[k]) for k in ("open", "wide", "round"))
        a, i, u = slice(5, 14), slice(25, 36), slice(47, 58)
        assert o[a].mean() > 0.8 and o[a].mean() > o[i].mean() + 0.3      # آ open
        assert w[i].mean() > 0.5 and ro[i].mean() < 0.2                    # ي wide
        assert ro[u].mean() > 0.6 and w[u].mean() < 0.2                    # و round


def test_silence_is_closed():
    r = visemes.analyse(np.zeros(16000), 16000, 30)
    assert max(r["open"]) == 0 and max(r["wide"]) == 0 and max(r["round"]) == 0
