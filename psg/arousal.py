"""EEG arousal detection (AASM section IV).

An arousal is an abrupt shift of EEG frequency (alpha, theta and/or > 16 Hz, but not
spindles) lasting at least 3 s, preceded by at least 10 s of stable sleep.  During REM
a concurrent rise in chin EMG of at least 1 s is also required.

Implementation: the ratio of fast (alpha 8-12 Hz + beta 16-30 Hz; sigma excluded so
spindles do not trigger) to slow (delta 0.5-4 Hz) EEG power is tracked on a 0.25-s grid,
smoothed over 2 s, and compared with its own median over the preceding 2-12 s
("background").  Using a fast/slow ratio makes the detector insensitive to overall
amplitude changes (movement, electrode impedance).  A run where it exceeds the threshold
for >= 3 s is an arousal.  The chin EMG is tracked the same way for the REM rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import preprocess as pp
from .io import Recording, STAGE_W, STAGE_R, EPOCH_S


@dataclass
class ArousalParams:
    min_duration: float = 3.0
    max_duration: float = 15.0      # longer shifts are wake; still returned but capped
    ratio_thr: float = 4.0          # (fast/slow power) vs its background
    emg_ratio_thr: float = 2.0      # for the REM rule
    background_s: tuple[float, float] = (2.0, 12.0)  # window before t used as background
    grid_s: float = 0.25
    refractory_s: float = 10.0      # AASM: 10 s of sleep must precede an arousal


def _power_track(x: np.ndarray, fs: float, grid_s: float) -> np.ndarray:
    """Mean squared amplitude on a coarse grid."""
    step = max(int(round(grid_s * fs)), 1)
    n = len(x) // step
    return np.square(x[:n * step]).reshape(n, step).mean(axis=1)


def _background_ratio(track: np.ndarray, grid_s: float, window: tuple[float, float]) -> np.ndarray:
    """track / median(track over [t - window[1], t - window[0]])."""
    a, b = int(window[0] / grid_s), int(window[1] / grid_s)
    size = max(b - a, 2)
    med = ndimage.median_filter(track, size=size, mode="nearest")
    # median_filter is centered; shift so the window ends `a` grid steps before t.
    shift = a + size // 2
    bg = np.empty_like(med)
    bg[shift:] = med[:len(med) - shift]
    bg[:shift] = med[0]
    return track / np.maximum(bg, 1e-12)


def detect_arousals(rec: Recording, stages: np.ndarray | None = None,
                    params: ArousalParams | None = None) -> list[tuple[float, float]]:
    p = params or ArousalParams()
    eeg = rec.get("eeg")
    if eeg is None:
        return []
    fs = eeg.fs
    x = pp.filt(eeg.data, fs, 0.3, 35.0)
    fast = pp.filt(x, fs, 8.0, 12.0) + pp.filt(x, fs, 16.0, 30.0)
    slow = pp.filt(x, fs, 0.5, 4.0)
    smooth = int(2.0 / p.grid_s)
    fast_t = pp.moving_mean(_power_track(fast, fs, p.grid_s), smooth)
    slow_t = pp.moving_mean(_power_track(slow, fs, p.grid_s), smooth)
    ratio = _background_ratio(fast_t / np.maximum(slow_t, 1e-12), p.grid_s, p.background_s)

    emg_ratio = None
    emg = rec.get("emg")
    if emg is not None and emg.fs >= 40:
        e = pp.filt(emg.data, emg.fs, 10.0, min(30.0, emg.fs / 2 * 0.9))
        et = _power_track(e, emg.fs, p.grid_s)
        er = _background_ratio(et, p.grid_s, p.background_s)
        n = min(len(er), len(ratio))
        emg_ratio = np.zeros(len(ratio))
        emg_ratio[:n] = er[:n]

    grid = p.grid_s
    above = ratio > p.ratio_thr
    # Fill 0.5-s dips so one arousal is not split into several.
    above = ndimage.binary_closing(above, structure=np.ones(int(0.5 / grid) + 1, dtype=bool))
    min_n = int(p.min_duration / grid)

    out: list[tuple[float, float]] = []
    last_end = -1e9
    for s, e in pp.runs(above):
        if e - s < min_n:
            continue
        onset = s * grid
        dur = min((e - s) * grid, p.max_duration)
        if onset - last_end < p.refractory_s:
            continue
        if stages is not None and int(onset // EPOCH_S) < len(stages):
            k0, k1 = int(max(onset - p.refractory_s, 0) // EPOCH_S), int(onset // EPOCH_S)
            if stages[k1] == STAGE_W or stages[k0] == STAGE_W:
                continue
            if stages[k1] == STAGE_R and emg_ratio is not None:
                seg = emg_ratio[s:e]
                if seg.size and np.max(np.convolve(seg > p.emg_ratio_thr, np.ones(int(1 / grid)), "same")) < int(1 / grid):
                    continue
        out.append((onset, dur))
        last_end = onset + dur
    return out


def arousal_index(arousals: list[tuple[float, float]], tst_hours: float) -> float:
    return len(arousals) / tst_hours if tst_hours > 0 else float("nan")
