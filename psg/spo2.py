"""Oxygen desaturation scoring from the cleaned SpO2 trace.

A desaturation is a fall of at least `drop` percentage points from a preceding
local maximum to a nadir, reached within `max_fall_s`.  Counting them per hour of
sleep gives the ODI; T90 is the time spent below 90 %.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Desaturation:
    onset: float       # s, where the fall starts
    nadir_time: float  # s
    baseline: float    # % before the fall
    nadir: float       # % lowest value
    duration: float    # s, onset -> recovery (or nadir if no recovery)

    @property
    def drop(self) -> float:
        return self.baseline - self.nadir

    @property
    def end(self) -> float:
        return self.onset + self.duration


def detect_desaturations(spo2: np.ndarray, fs: float, drop: float = 3.0,
                         max_fall_s: float = 120.0, min_gap_s: float = 5.0) -> list[Desaturation]:
    """Classic peak-to-nadir scan on a 1-Hz version of the trace (NaN = untrusted)."""
    step = max(int(round(fs)), 1)
    x = np.asarray(spo2, dtype=np.float64)[::step]  # 1 Hz
    n = len(x)
    out: list[Desaturation] = []
    i = 0
    while i < n - 1:
        if not np.isfinite(x[i]):
            i += 1
            continue
        # Walk forward from a local maximum and look for the nadir before the trace climbs back.
        base = x[i]
        j = i + 1
        nadir_val, nadir_idx = base, i
        while j < n and j - i <= max_fall_s:
            v = x[j]
            if not np.isfinite(v):
                break
            if v < nadir_val:
                nadir_val, nadir_idx = v, j
            elif v >= base:          # back at baseline: fall is over
                break
            elif v > nadir_val + 1.0 and base - nadir_val >= drop:  # clear recovery after a valid drop
                break
            j += 1
        if base - nadir_val >= drop and nadir_idx > i:
            out.append(Desaturation(onset=float(i), nadir_time=float(nadir_idx), baseline=float(base),
                                    nadir=float(nadir_val), duration=float(max(j, nadir_idx) - i)))
            i = max(nadir_idx + 1, i + int(min_gap_s))
        else:
            i += 1
    return out


def oxygen_summary(spo2: np.ndarray, fs: float, sleep_mask_1hz: np.ndarray | None = None) -> dict:
    step = max(int(round(fs)), 1)
    x = np.asarray(spo2, dtype=np.float64)[::step]
    if sleep_mask_1hz is not None:
        m = sleep_mask_1hz[:len(x)]
        x = x[:len(m)][m]
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return {"mean": None, "nadir": None, "t90_min": 0.0, "t88_min": 0.0, "valid_fraction": 0.0}
    return {
        "mean": float(valid.mean()),
        "nadir": float(valid.min()),
        "t90_min": float((valid < 90).sum() / 60.0),
        "t88_min": float((valid < 88).sum() / 60.0),
        "valid_fraction": float(valid.size / max(len(x), 1)),
    }
