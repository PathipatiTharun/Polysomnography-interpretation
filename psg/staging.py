"""Rule-based sleep staging on 30-s epochs (AASM stages W, N1, N2, N3, R).

Per-epoch features follow the AASM definitions:
  * relative EEG band powers (delta 0.5-4, theta 4-8, alpha 8-12, sigma 12-15, beta 16-30 Hz)
  * slow-wave fraction: share of the epoch covered by 0.5-2 Hz waves of large amplitude
    (the 75 uV rule, with the amplitude threshold calibrated per recording because many
    files do not carry calibrated microvolt units)
  * sleep-spindle density from the sigma-band envelope
  * chin EMG level
  * rapid-eye-movement density from the EOG slope

The rules are turned into soft stage memberships, and a small hidden-Markov smoothing
(Viterbi) enforces plausible stage continuity, which is how the AASM "continue N2 until..."
rules behave in practice.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import preprocess as pp
from .io import Recording, STAGE_W, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R, EPOCH_S

STAGES = [STAGE_W, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R]


@dataclass
class StagingParams:
    """Thresholds are on unit-free quantities: relative band powers (fraction of 0.5-30 Hz
    power), per-night z-scores of the EMG/EOG level, and ratios to the night's median."""
    emg_w_z: float = 1.4           # chin EMG z-score above this suggests wake
    alpha_w_ratio: float = 1.8     # alpha share relative to the night's median (eyes-closed wake)
    beta_w_ratio: float = 1.6      # beta share relative to the night's median
    eog_w_z: float = 1.8           # eye-movement energy z-score (blinks, saccades)
    delta_n3: float = 0.80         # relative delta for N3
    sw_fraction_n3: float = 0.30   # share of the epoch covered by large slow waves
    sw_amp_frac: float = 0.40      # slow-wave amplitude threshold as a fraction of the night's p99
    delta_n2: float = 0.62         # relative delta above which NREM is at least N2
    delta_light: float = 0.72      # relative delta below which the EEG is "mixed frequency" (N1 / R)
    emg_r_z: float = -0.3          # chin EMG must be below this z-score for REM (atonia)
    rem_density_r: float = 2.0     # eye-movement bursts per epoch that support REM
    self_transition: float = 0.85  # HMM stickiness
    emg_strong_atonia_z: float = 1.5   # chin EMG z-score below minus this counts as clear atonia
    n2_atonia_penalty: float = 0.8     # how much clear atonia argues against N2 (0 = ignore tone for N2)
    max_night_h: float = 12.0      # recordings longer than this get lights-off/on detection


@dataclass
class EpochFeatures:
    n: int
    delta: np.ndarray
    theta: np.ndarray
    alpha: np.ndarray
    sigma: np.ndarray
    beta: np.ndarray
    sw_fraction: np.ndarray
    spindles: np.ndarray
    emg: np.ndarray            # log RMS
    emg_z: np.ndarray
    rem_density: np.ndarray
    eog_activity: np.ndarray   # log slope energy, for wake/blinks
    eog_z: np.ndarray
    eog_corr: np.ndarray       # left/right EOG correlation (conjugate eye movements)
    artifact: np.ndarray       # bool
    rest: tuple[int, int] | None = None  # main rest period (first, last+1) epoch, set by stage_sleep
    beta_abs: np.ndarray | None = None   # log absolute 16-30 Hz EEG power (forehead muscle / wake activity)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("delta", "theta", "alpha", "sigma", "beta", "sw_fraction",
                                              "spindles", "emg_z", "rem_density", "eog_z", "eog_corr",
                                              "artifact")}


def _zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    sd = pp.robust_scale(x) or (np.nanstd(x) or 1.0)
    return (x - med) / sd


def _band(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f < hi)
    return pxx[:, m].sum(axis=1)


def compute_features(rec: Recording, params: StagingParams | None = None) -> EpochFeatures:
    p = params or StagingParams()
    n_ep = rec.n_epochs
    eeg = rec.get("eeg")
    if eeg is None:
        raise ValueError("No EEG channel: cannot stage sleep")
    fs = eeg.fs
    x = pp.filt(eeg.data, fs, 0.3, 35.0)
    ep_len = int(EPOCH_S * fs)
    X = x[:n_ep * ep_len].reshape(n_ep, ep_len)

    # --- spectral bands (Welch, 4-s segments) ------------------------------------------------
    f, pxx = signal.welch(X, fs=fs, nperseg=int(4 * fs), axis=1)
    total = _band(f, pxx, 0.5, 30.0) + 1e-12
    delta = _band(f, pxx, 0.5, 4.0) / total
    theta = _band(f, pxx, 4.0, 8.0) / total
    alpha = _band(f, pxx, 8.0, 12.0) / total
    sigma = _band(f, pxx, 12.0, 15.0) / total
    beta = _band(f, pxx, 16.0, 30.0) / total

    # --- slow waves --------------------------------------------------------------------------
    sw = pp.filt(eeg.data, fs, 0.5, 2.0, order=2)[:n_ep * ep_len]
    sec = int(fs)
    sw_p2p = pp.peak_to_peak(sw, sec)[::sec][:n_ep * 30].reshape(n_ep, 30)
    p2p_epoch = pp.peak_to_peak(x, ep_len)[::ep_len][:n_ep]
    artifact = (p2p_epoch > 10 * np.median(p2p_epoch)) | (p2p_epoch < 0.02 * np.median(p2p_epoch))
    ref = np.percentile(sw_p2p[~artifact], 99) if (~artifact).any() else np.percentile(sw_p2p, 99)
    sw_fraction = (sw_p2p > p.sw_amp_frac * ref).mean(axis=1)

    # --- spindles ----------------------------------------------------------------------------
    sig_env = pp.moving_rms(pp.filt(eeg.data, fs, 11.0, 16.0), int(0.3 * fs))[:n_ep * ep_len]
    thr = 2.5 * np.median(sig_env)
    spindles = np.zeros(n_ep)
    for k, (s, e) in enumerate(pp.runs(sig_env > thr)):
        pass
    runs = pp.runs(sig_env > thr)
    for s, e in runs:
        d = (e - s) / fs
        if 0.5 <= d <= 2.5:
            spindles[min(s // ep_len, n_ep - 1)] += 1

    # --- chin EMG ----------------------------------------------------------------------------
    emg_ch = rec.get("emg")
    if emg_ch is not None:
        el = max(int(EPOCH_S * emg_ch.fs), 1)
        if emg_ch.fs >= 40:
            e = pp.filt(emg_ch.data, emg_ch.fs, 10.0, min(30.0, 0.9 * emg_ch.fs / 2))
            E = e[:n_ep * el].reshape(n_ep, el)
            emg = np.log(np.sqrt((E * E).mean(axis=1)) + 1e-9)
        else:
            # Low-rate channel: already an amplitude envelope (e.g. Sleep-EDF 1-Hz RMS EMG).
            E = np.abs(np.asarray(emg_ch.data[:n_ep * el], dtype=np.float64)).reshape(n_ep, el)
            emg = np.log(E.mean(axis=1) + 1e-9)
    else:
        emg = np.zeros(n_ep)
    emg_z = _zscore(emg)

    # --- EOG ---------------------------------------------------------------------------------
    l, r = rec.get("eog_l"), rec.get("eog_r")
    rem_density = np.zeros(n_ep)
    eog_activity = np.zeros(n_ep)
    eog_corr = np.zeros(n_ep)
    if l is not None:
        efs = l.fs
        el = int(EPOCH_S * efs)
        Lf = pp.filt(l.data, efs, 0.3, 5.0)
        L = np.diff(Lf, prepend=0.0) * efs
        chans = [L]
        filt_chans = [Lf]
        if r is not None and abs(r.fs - efs) < 1e-6:
            Rf = pp.filt(r.data, efs, 0.3, 5.0)
            chans.append(np.diff(Rf, prepend=0.0) * efs)
            filt_chans.append(Rf)
        n = min(len(c) for c in chans)
        chans = [c[:n] for c in chans]
        # A rapid eye movement is a steep deflection (3 robust SDs of the slope); with two
        # channels both must deflect within 0.1 s of each other (conjugate movement).
        fast = np.abs(chans[0]) > 3.0 * (pp.robust_scale(chans[0]) or 1.0)
        if len(chans) > 1:
            other = np.abs(chans[1]) > 3.0 * (pp.robust_scale(chans[1]) or 1.0)
            fast &= pp.dilate(other, int(0.1 * efs))
        for s, e in pp.runs(fast):
            if e - s >= 0.05 * efs:
                rem_density[min(s // el, n_ep - 1)] += 1
        A = chans[0][:n_ep * el].reshape(n_ep, el)
        eog_activity = np.log((A * A).mean(axis=1) + 1e-9)
        if len(filt_chans) > 1:
            m = min(len(filt_chans[0]), len(filt_chans[1]), n_ep * el)
            X1 = filt_chans[0][:m // el * el].reshape(-1, el)
            X2 = filt_chans[1][:m // el * el].reshape(-1, el)
            X1 = X1 - X1.mean(axis=1, keepdims=True)
            X2 = X2 - X2.mean(axis=1, keepdims=True)
            den = np.sqrt((X1 * X1).sum(axis=1) * (X2 * X2).sum(axis=1)) + 1e-12
            c = (X1 * X2).sum(axis=1) / den
            eog_corr[:len(c)] = c
    eog_z = _zscore(eog_activity)

    return EpochFeatures(n_ep, delta, theta, alpha, sigma, beta, sw_fraction, spindles, emg, emg_z,
                         rem_density, eog_activity, eog_z, eog_corr, artifact,
                         beta_abs=np.log(_band(f, pxx, 16.0, 30.0) + 1e-12))


def _sig(x: np.ndarray, thr: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - thr) / max(width, 1e-6)))


def memberships(F: EpochFeatures, p: StagingParams) -> np.ndarray:
    """Soft evidence for each stage, shape (n_epochs, 5) in STAGES order."""
    alpha_ratio = F.alpha / (np.median(F.alpha) + 1e-9)
    beta_ratio = F.beta / (np.median(F.beta) + 1e-9)
    tonic = _sig(F.emg_z, 0.0, 0.3)  # chin tone at least average: not atonic
    # Wake: high chin tone, or eye blinks/saccades, or a dominant alpha/beta EEG with tone.
    wake = np.maximum.reduce([
        _sig(F.emg_z, p.emg_w_z, 0.25),
        _sig(F.eog_z, p.eog_w_z, 0.3) * tonic,
        _sig(alpha_ratio, p.alpha_w_ratio, 0.15) * tonic,
        _sig(beta_ratio, p.beta_w_ratio, 0.2) * tonic,
    ])
    # N3: delta-dominated EEG with large slow waves covering >= the required share of the epoch.
    n3 = _sig(F.delta, p.delta_n3, 0.03) * _sig(F.sw_fraction, p.sw_fraction_n3, 0.05)
    # REM: chin atonia with a low-amplitude mixed-frequency EEG; eye-movement bursts add support.
    atonia = _sig(-F.emg_z, -p.emg_r_z, 0.2)
    mixed = _sig(-F.delta, -p.delta_light, 0.04)
    rem = atonia * mixed * (1 - n3) * (0.5 + 0.5 * _sig(F.rem_density, p.rem_density_r, 1.0))
    # N2: moderately delta-rich NREM without the N3 slow-wave load.
    # Clear chin atonia is the hallmark of REM, so it argues against N2 (tone in N2 is reduced, not absent).
    strong_atonia = _sig(-F.emg_z, p.emg_strong_atonia_z, 0.3)
    n2 = (_sig(F.delta, p.delta_n2, 0.04) * (1 - n3) * (1 - wake) * (1 - 0.7 * rem)
          * (1 - p.n2_atonia_penalty * strong_atonia))
    # N1: mixed-frequency EEG with chin tone still present.
    n1 = (1 - wake) * (1 - n3) * mixed * (1 - rem) * _sig(F.emg_z, -0.5, 0.3)
    M = np.stack([wake, n1, n2, n3, rem], axis=1)
    return np.clip(M, 1e-3, 1.0)


def _viterbi(logE: np.ndarray, logT: np.ndarray, logP0: np.ndarray) -> np.ndarray:
    n, k = logE.shape
    score = np.zeros((n, k))
    back = np.zeros((n, k), dtype=int)
    score[0] = logP0 + logE[0]
    for t in range(1, n):
        cand = score[t - 1][:, None] + logT
        back[t] = cand.argmax(axis=0)
        score[t] = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(score[-1].argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transition_matrix(stick: float) -> np.ndarray:
    # Rows/cols: W, N1, N2, N3, R.  Off-diagonal weights encode physiologically common moves.
    T = np.array([
        [0, 6, 2, 0.2, 0.5],   # from W
        [4, 0, 6, 0.2, 1.0],   # from N1
        [1, 2, 0, 4, 2.0],     # from N2
        [0.5, 0.5, 5, 0, 0.3], # from N3
        [2, 2, 2, 0.1, 0],     # from R
    ], dtype=float)
    T = T / T.sum(axis=1, keepdims=True) * (1 - stick)
    T[np.diag_indices(5)] = stick
    return T


def _otsu(x: np.ndarray, bins: int = 64) -> float:
    """Threshold that best splits a bimodal distribution (Otsu's method)."""
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[int(np.argmax(between))])


def rest_period(F: EpochFeatures, smooth_epochs: int = 20, min_active_epochs: int = 120,
                bridge_epochs: int = 60, margin_epochs: int = 30) -> tuple[int, int]:
    """Main rest period ("lights off" to "lights on") as (first, last+1) epoch.

    In-lab PSGs are recorded from lights-off to lights-on, so normally the whole file is the
    rest period.  Ambulatory or 24-h recordings (e.g. Sleep-EDF) add hours of daytime wake,
    which would distort the per-night normalisation.  We therefore trim only a continuous
    *active* stretch (chin EMG + eye movements above the quiet/active split, 10-min
    smoothing, quiet gaps < 30 min bridged) of at least one hour at the very start or end.
    """
    from scipy import ndimage
    n = F.n
    # Each channel is z-scored and capped at +-4 so that one channel with a tiny spread (e.g. a
    # rectified EMG envelope that is flat through 16 h of daytime wake) cannot swamp the others.
    def zc(v):
        return np.clip((v - np.median(v)) / (pp.robust_scale(v) or 1.0), -4.0, 4.0)
    act = zc(F.emg)
    if np.any(F.eog_activity):
        act = act + zc(F.eog_activity)
    if F.beta_abs is not None:
        # Absolute fast EEG power (scalp/forehead muscle, active wake) drops sharply at sleep onset.
        act = act + zc(F.beta_abs)
    act = pp.moving_mean(act, smooth_epochs)
    active = act >= _otsu(act)
    active = ndimage.binary_closing(np.pad(active, bridge_epochs, constant_values=True),
                                    structure=np.ones(bridge_epochs, dtype=bool))[bridge_epochs:-bridge_epochs]
    runs = pp.runs(active)
    a, b = 0, n
    if runs and runs[0][0] == 0 and runs[0][1] >= min_active_epochs:
        a = max(runs[0][1] - margin_epochs, 0)
    if runs and runs[-1][1] == n and runs[-1][1] - runs[-1][0] >= min_active_epochs:
        b = min(runs[-1][0] + margin_epochs, n)
    if b - a < 60:  # implausible: keep everything
        return 0, n
    return a, b


def _slice_features(F: EpochFeatures, a: int, b: int) -> EpochFeatures:
    kw = {f.name: (getattr(F, f.name)[a:b] if isinstance(getattr(F, f.name), np.ndarray) else getattr(F, f.name))
          for f in dataclasses.fields(F)}
    kw["n"] = b - a
    sub = EpochFeatures(**kw)
    # Re-normalise within the rest period so wake outside it does not set the reference.
    sub.emg_z = _zscore(sub.emg)
    sub.eog_z = _zscore(sub.eog_activity)
    return sub


def stage_sleep(rec: Recording, params: StagingParams | None = None
                ) -> tuple[np.ndarray, EpochFeatures, np.ndarray]:
    """Returns (stage codes per epoch, features, memberships).  Epochs outside the detected
    rest period are scored Wake; `features.rest` holds the (first, last+1) epoch range."""
    p = params or StagingParams()
    F = compute_features(rec, p)
    # A lab PSG spans lights-off to lights-on (AASM TRT): never trim it.  Only recordings much
    # longer than a night (ambulatory / 24-h) get a detected rest period.
    a, b = rest_period(F) if F.n * EPOCH_S > p.max_night_h * 3600 else (0, F.n)
    sub = _slice_features(F, a, b) if (a, b) != (0, F.n) else F
    M_sub = memberships(sub, p)
    logE = np.log(M_sub / M_sub.sum(axis=1, keepdims=True))
    logT = np.log(transition_matrix(p.self_transition))
    logP0 = np.log(np.array([0.8, 0.1, 0.05, 0.025, 0.025]))
    path = _viterbi(logE, logT, logP0)
    stages = np.full(F.n, STAGE_W, dtype=int)
    stages[a:b] = [STAGES[i] for i in path]
    M = np.zeros((F.n, 5))
    M[:, 0] = 1.0
    M[a:b] = M_sub
    F.rest = (a, b)
    return stages, F, M


def sleep_summary(stages: np.ndarray, period: tuple[int, int] | None = None) -> dict:
    """Standard sleep-architecture measures.  `period` restricts them to the rest period
    (lights off -> lights on) when the recording extends beyond it."""
    if period is not None:
        stages = stages[period[0]:period[1]]
    n = len(stages)
    sleep = np.isin(stages, [STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R])
    tst_min = sleep.sum() * EPOCH_S / 60
    trt_min = n * EPOCH_S / 60
    first = int(np.argmax(sleep)) if sleep.any() else n
    last = n - int(np.argmax(sleep[::-1])) if sleep.any() else n
    latency = first * EPOCH_S / 60
    rem_idx = np.flatnonzero(stages == STAGE_R)
    rem_latency = (rem_idx[0] - first) * EPOCH_S / 60 if rem_idx.size else None
    waso = ((~sleep)[first:last]).sum() * EPOCH_S / 60 if sleep.any() else 0.0
    def pct(code: int) -> float:
        return float((stages == code).sum() / sleep.sum() * 100) if sleep.any() else 0.0
    return {
        "trt_min": trt_min, "tst_min": float(tst_min), "sleep_efficiency": float(tst_min / trt_min * 100) if trt_min else 0.0,
        "sleep_latency_min": latency, "rem_latency_min": rem_latency, "waso_min": float(waso),
        "pct_n1": pct(STAGE_N1), "pct_n2": pct(STAGE_N2), "pct_n3": pct(STAGE_N3), "pct_rem": pct(STAGE_R),
        "n_epochs": n,
    }
