"""Signal cleaning and filtering.

Everything here is plain NumPy/SciPy: Butterworth filters applied forwards and
backwards (zero phase), rolling statistics, artifact masks, and gap handling.
A "bad mask" is a boolean array the same length as the signal where True means
the sample should not be trusted (sensor dropout, flat line, impossible values).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import ndimage, signal


# ----------------------------------------------------------------------------- basics

def fill_nans(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaNs. Returns (filled copy, nan mask)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.isnan(x)
    if not m.any():
        return x.copy(), m
    if m.all():
        return np.zeros_like(x), m
    idx = np.arange(len(x))
    y = x.copy()
    y[m] = np.interp(idx[m], idx[~m], x[~m])
    return y, m


def _sos(fs: float, lo: float | None, hi: float | None, order: int):
    nyq = fs / 2.0
    if hi is not None:
        hi = min(hi, 0.95 * nyq)
    if lo is not None and hi is not None:
        return signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    if lo is not None:
        return signal.butter(order, lo / nyq, btype="highpass", output="sos")
    if hi is not None:
        return signal.butter(order, hi / nyq, btype="lowpass", output="sos")
    return None


def filt(x: np.ndarray, fs: float, lo: float | None = None, hi: float | None = None,
         order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band/low/high-pass. NaNs are interpolated before filtering."""
    y, _ = fill_nans(x)
    sos = _sos(fs, lo, hi, order)
    if sos is None or len(y) < 3 * (2 * order + 1) + 1:
        return y
    return signal.sosfiltfilt(sos, y)


def notch(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    y, _ = fill_nans(x)
    if f0 >= fs / 2:
        return y
    b, a = signal.iirnotch(f0, q, fs)
    return signal.filtfilt(b, a, y)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    y, _ = fill_nans(x)
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(y, fr.numerator, fr.denominator)


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    n = max(int(n), 1)
    return ndimage.uniform_filter1d(np.asarray(x, dtype=np.float64), n, mode="nearest")


def moving_rms(x: np.ndarray, n: int) -> np.ndarray:
    return np.sqrt(np.maximum(moving_mean(np.square(x), n), 0.0))


def moving_std(x: np.ndarray, n: int) -> np.ndarray:
    m = moving_mean(x, n)
    return np.sqrt(np.maximum(moving_mean(np.square(x), n) - m * m, 0.0))


def peak_to_peak(x: np.ndarray, n: int) -> np.ndarray:
    """Max minus min over a centered sliding window of n samples."""
    n = max(int(n), 1)
    return (ndimage.maximum_filter1d(x, n, mode="nearest")
            - ndimage.minimum_filter1d(x, n, mode="nearest"))


def robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD, a standard-deviation estimate that ignores outliers."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start (inclusive) / end (exclusive) indices of each True run."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=np.ones(2 * n + 1, dtype=bool))


# ----------------------------------------------------------------------------- artifact masks

def flatline_mask(x: np.ndarray, fs: float, win_s: float = 8.0, rel: float = 0.02) -> np.ndarray:
    """True where the signal is essentially constant: sensor off, disconnected lead, saturation."""
    y, nanmask = fill_nans(x)
    n = max(int(win_s * fs), 3)
    sd = moving_std(y, n)
    scale = robust_scale(np.diff(y)) or (np.std(y) or 1.0)
    return (sd < rel * scale) | nanmask


def clipping_mask(x: np.ndarray, fs: float, min_run_s: float = 0.4, tol: float = 0.995) -> np.ndarray:
    """True where the signal sits at its recorded extreme for at least min_run_s (ADC saturation)."""
    y, _ = fill_nans(x)
    hi, lo = np.max(y), np.min(y)
    at = (y >= lo + tol * (hi - lo)) | (y <= lo + (1 - tol) * (hi - lo))
    out = np.zeros_like(at)
    min_run = max(int(min_run_s * fs), 2)
    for s, e in runs(at):
        if e - s >= min_run:
            out[s:e] = True
    return out


# ----------------------------------------------------------------------------- per-signal cleaners

def clean_spo2(spo2: np.ndarray, fs: float, lo: float = 50.0, hi: float = 100.0,
               max_slope: float = 4.0, max_gap_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Clean a pulse-oximetry trace.

    Rejects physiologically impossible values (<50 % or >100 %), jumps faster than
    max_slope %/s (probe movement, dropout edges), and one second around each bad
    run.  Gaps up to max_gap_s are linearly interpolated; longer gaps stay NaN and
    are reported in the bad mask so no event is scored on them.
    """
    x = np.asarray(spo2, dtype=np.float64).copy()
    bad = ~np.isfinite(x) | (x < lo) | (x > hi)
    x[bad] = np.nan
    d = np.abs(np.diff(x)) * fs
    jump = np.zeros_like(bad)
    jump[1:] |= d > max_slope
    jump[:-1] |= d > max_slope
    bad |= jump
    bad = dilate(bad, int(round(fs)))
    x[bad] = np.nan

    # Fill short gaps only.
    filled, _ = fill_nans(x)
    long_gap = np.zeros_like(bad)
    max_gap = int(max_gap_s * fs)
    for s, e in runs(bad):
        if e - s > max_gap:
            long_gap[s:e] = True
    filled[long_gap] = np.nan
    # A light median filter removes single-sample glitches without blurring desaturations.
    k = int(fs) | 1
    if k >= 3:
        tmp, m = fill_nans(filled)
        tmp = signal.medfilt(tmp, k)
        tmp[long_gap] = np.nan
        filled = tmp
    return filled, long_gap


def clean_respiratory(x: np.ndarray, fs: float, lo: float = 0.05, hi: float = 2.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Band-limit a flow/effort signal to the breathing band and flag dead/clipped segments."""
    y = filt(x, fs, lo, hi, order=2)
    bad = flatline_mask(x, fs) | clipping_mask(x, fs)
    return y, bad


def breath_amplitude(x: np.ndarray, fs: float, win_s: float = 3.5, smooth_s: float = 0.5) -> np.ndarray:
    """Per-sample breath amplitude: peak-to-trough excursion over a sliding window that is
    long enough to hold at least one inspiration or expiration at normal breathing rates."""
    env = peak_to_peak(np.asarray(x, dtype=np.float64), int(win_s * fs))
    return moving_mean(env, int(smooth_s * fs))


def causal_baseline(env: np.ndarray, fs: float, win_s: float = 120.0, exclude: np.ndarray | None = None,
                    step_s: float = 1.0, stable_frac: float = 0.75) -> np.ndarray:
    """AASM breathing baseline from the preceding 2 minutes of breath amplitude.

    AASM: use the mean amplitude of *stable* breathing; when there is no stable baseline
    (periodic breathing, clusters of events) use the mean of the three largest breaths.
    Breathing counts as stable when the median amplitude is at least `stable_frac` of the
    95th percentile; otherwise the mean of the top decile of breaths is taken.  Samples in
    `exclude` (already-detected events) are ignored.
    """
    step = max(int(step_s * fs), 1)
    e = np.asarray(env, dtype=np.float64).copy()
    if exclude is not None:
        e[exclude] = np.nan
    coarse = e[::step]
    w = max(int(win_s / step_s), 2)
    n = len(coarse)
    base = np.full(n, np.nan)
    for i in range(n):
        seg = coarse[max(0, i - w):i]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 5:
            med = np.median(seg)
            p90, p95 = np.percentile(seg, [90, 95])
            base[i] = med if med >= stable_frac * p95 else seg[seg >= p90].mean()
    base, _ = fill_nans(base)
    if not np.isfinite(base).any() or base.max() == 0:
        base[:] = np.nanmedian(e) if np.isfinite(e).any() else 1.0
    # Back to full rate.
    full = np.interp(np.arange(len(e)), np.arange(n) * step, base)
    return np.maximum(full, 1e-9)


def noise_floor(x: np.ndarray, fs: float, env: np.ndarray, bad: np.ndarray | None = None) -> float:
    """Amplitude the breath envelope keeps when no breathing is present (cardiogenic wobble,
    sensor noise).  Estimated as the smaller of the typical amplitude of the 0.7-2 Hz
    residual and the 1st percentile of the envelope itself, so that a night without any
    apneas is not over-corrected."""
    good = ~bad if bad is not None else np.ones(len(env), dtype=bool)
    resid = filt(x, fs, 0.7, 2.0, order=2)
    resid_amp = breath_amplitude(resid, fs)
    if good.sum() < 10:
        return 0.0
    return float(min(np.median(resid_amp[good]), np.percentile(env[good], 1)))
