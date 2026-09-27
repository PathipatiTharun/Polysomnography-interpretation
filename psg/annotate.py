"""Automatic sleep annotations beyond breathing: sleep stages, estimated lights off/on, sleep
spindles, slow waves, rapid eye movements, EEG arousals and artifact epochs.

Detectors (all on the first EEG / EOG channel found):
  * Spindles   - 11-16 Hz band, Hilbert envelope; a burst above 3x the NREM median envelope,
                 edges where it falls below 1.5x, 0.5-3 s long, 11-16 Hz by zero crossings,
                 in N2/N3 epochs (AASM: "train of distinct waves 11-16 Hz, >= 0.5 s").
  * Slow waves - 0.3-2 Hz band; negative half-wave 0.25-1.0 s, negative peak <= -40 uV and
                 peak-to-peak >= 75 uV (the AASM 75 uV amplitude, Massimini-style detection),
                 in N2/N3 epochs.  Needs calibrated microvolts; skipped otherwise.
  * Rapid eye movements - steep EOG deflections (slope > max(3 robust SD, 300 uV/s), >= 40 uV
                 excursion, <= 1 s) in REM epochs.
  * Arousals   - psg.arousal (abrupt EEG frequency shift >= 3 s after >= 10 s of sleep).
Stages come from psg.staging (or are given), lights off/on from its rest-period detection.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import preprocess as pp
from .arousal import detect_arousals
from .io import EPOCH_S, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R, STAGE_UNK, STAGE_W, Recording
from .staging import sleep_summary, stage_sleep

STAGE_LABEL = {STAGE_W: "Sleep stage W", STAGE_N1: "Sleep stage N1", STAGE_N2: "Sleep stage N2",
               STAGE_N3: "Sleep stage N3", STAGE_R: "Sleep stage R", STAGE_UNK: "Sleep stage ?"}


@dataclass
class Annotation:
    onset: float
    duration: float
    label: str
    channel: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.onset + self.duration


@dataclass
class AnnotationResult:
    rec: Recording
    stages: np.ndarray
    rest: tuple[int, int] | None
    annotations: list[Annotation]
    summary: dict

    def of(self, label: str) -> list[Annotation]:
        return [a for a in self.annotations if a.label == label]


def _epoch_mask(stages: np.ndarray, codes, n: int, fs: float) -> np.ndarray:
    ep = np.isin(stages, codes)
    m = np.repeat(ep, int(EPOCH_S * fs))
    out = np.zeros(n, dtype=bool)
    out[:min(n, len(m))] = m[:n]
    return out


def _stage_at(stages: np.ndarray, t: float) -> int:
    k = int(t // EPOCH_S)
    return int(stages[k]) if 0 <= k < len(stages) else STAGE_UNK


# ----------------------------------------------------------------------------- detectors

def detect_spindles(x: np.ndarray, fs: float, stages: np.ndarray, channel: str = "",
                    lo: float = 11.0, hi: float = 16.0, thr_hi: float = 3.0, thr_lo: float = 1.5,
                    min_dur: float = 0.5, max_dur: float = 3.0) -> list[Annotation]:
    if fs < 2.5 * hi:
        return []
    band = pp.filt(x, fs, lo, hi, order=4)
    env = pp.moving_mean(np.abs(signal.hilbert(band)), int(0.1 * fs))
    nrem = _epoch_mask(stages, [STAGE_N2, STAGE_N3], len(env), fs)
    if nrem.sum() < 60 * fs:
        return []
    base = float(np.median(env[nrem]))
    out = []
    for s, e in pp.runs(env > thr_lo * base):
        dur = (e - s) / fs
        if not (min_dur <= dur <= max_dur) or env[s:e].max() < thr_hi * base:
            continue
        mid = (s + e) / 2 / fs
        if _stage_at(stages, mid) not in (STAGE_N2, STAGE_N3):
            continue
        seg = band[s:e]
        crossings = np.count_nonzero(np.diff(np.signbit(seg)))
        freq = crossings / 2 / dur
        if not (lo <= freq <= hi):
            continue
        out.append(Annotation(s / fs, dur, "Spindle", channel,
                              {"freq_hz": round(freq, 1), "amp_uv": round(float(seg.max() - seg.min()), 1)}))
    return out


def detect_slow_waves(x: np.ndarray, fs: float, stages: np.ndarray, channel: str = "",
                      neg_thr: float = -40.0, p2p_thr: float = 75.0,
                      min_neg: float = 0.25, max_neg: float = 1.0) -> list[Annotation]:
    y = pp.filt(x, fs, 0.3, 2.0, order=2)
    neg = np.signbit(y)
    down = np.flatnonzero(~neg[:-1] & neg[1:]) + 1      # positive -> negative
    up = np.flatnonzero(neg[:-1] & ~neg[1:]) + 1        # negative -> positive
    if down.size < 2 or up.size < 1:
        return []
    out = []
    j = np.searchsorted(up, down)
    for i, d in enumerate(down[:-1]):
        if j[i] >= up.size:
            break
        u = up[j[i]]
        d2 = down[i + 1]
        if u >= d2:
            continue
        neg_dur = (u - d) / fs
        if not (min_neg <= neg_dur <= max_neg):
            continue
        trough = float(y[d:u].min())
        peak = float(y[u:d2].max())
        if trough > neg_thr or peak - trough < p2p_thr:
            continue
        if _stage_at(stages, d / fs) not in (STAGE_N2, STAGE_N3):
            continue
        out.append(Annotation(d / fs, min((d2 - d) / fs, 2.5), "Slow wave", channel,
                              {"neg_uv": round(trough, 1), "p2p_uv": round(peak - trough, 1)}))
    return out


def detect_rems(eog: np.ndarray, fs: float, stages: np.ndarray, channel: str = "",
                min_slope: float = 300.0, min_amp: float = 40.0) -> list[Annotation]:
    y = pp.filt(eog, fs, 0.3, 5.0, order=2)
    d = np.gradient(y) * fs                              # uV/s
    rem = _epoch_mask(stages, [STAGE_R], len(y), fs)
    if rem.sum() < 30 * fs:
        return []
    thr = max(3.0 * pp.robust_scale(d[rem]), min_slope)
    fast = (np.abs(d) > thr) & rem
    fast = pp.dilate(fast, int(0.1 * fs))
    out = []
    half = int(0.25 * fs)
    for s, e in pp.runs(fast):
        dur = (e - s) / fs
        if dur > 1.0:
            continue
        seg = y[max(s - half, 0):min(e + half, len(y))]
        amp = float(seg.max() - seg.min())
        if amp < min_amp:
            continue
        out.append(Annotation(s / fs, dur, "Rapid eye movement", channel, {"amp_uv": round(amp, 1)}))
    return out


def _stage_runs(stages: np.ndarray) -> list[Annotation]:
    out, k = [], 0
    while k < len(stages):
        j = k
        while j + 1 < len(stages) and stages[j + 1] == stages[k]:
            j += 1
        out.append(Annotation(k * EPOCH_S, (j - k + 1) * EPOCH_S, STAGE_LABEL.get(int(stages[k]), "Sleep stage ?")))
        k = j + 1
    return out


# ----------------------------------------------------------------------------- main entry

def annotate_recording(rec: Recording, stages: np.ndarray | None = None) -> AnnotationResult:
    rest = None
    features = None
    if stages is None:
        stages, features, _ = stage_sleep(rec)
        rest = features.rest if features.rest != (0, len(stages)) else None
    stages = np.asarray(stages, dtype=int)
    anns: list[Annotation] = _stage_runs(stages)
    if rest is not None:
        anns.append(Annotation(rest[0] * EPOCH_S, 0.0, "Lights off (estimated)"))
        anns.append(Annotation(rest[1] * EPOCH_S, 0.0, "Lights on (estimated)"))
    if features is not None:
        for k in np.flatnonzero(features.artifact):
            anns.append(Annotation(k * EPOCH_S, EPOCH_S, "Artifact"))

    eeg, eog = rec.get("eeg"), rec.get("eog_l")
    sp = sw = rems = []
    if eeg is not None:
        sp = detect_spindles(eeg.data, eeg.fs, stages, eeg.label)
        if eeg.unit.lower() in ("uv", "µv"):
            sw = detect_slow_waves(eeg.data, eeg.fs, stages, eeg.label)
    if eog is not None and eog.fs >= 20:
        rems = detect_rems(eog.data, eog.fs, stages, eog.label)
    ar = [Annotation(a, d, "Arousal", eeg.label if eeg else "") for a, d in detect_arousals(rec, stages)]
    anns += sp + sw + rems + ar
    anns.sort(key=lambda a: (a.onset, a.label))

    ss = sleep_summary(stages, rest)
    n2_min = float((stages == STAGE_N2).sum() * EPOCH_S / 60)
    n3_min = float((stages == STAGE_N3).sum() * EPOCH_S / 60)
    nrem_min = n2_min + n3_min
    r_min = float((stages == STAGE_R).sum() * EPOCH_S / 60)
    tst_h = ss["tst_min"] / 60
    summary = {
        **ss,
        "rest_period_h": None if rest is None else (rest[0] * EPOCH_S / 3600, rest[1] * EPOCH_S / 3600),
        "n_spindles": len(sp), "spindle_density_per_min_nrem": len(sp) / nrem_min if nrem_min else None,
        "spindle_freq_hz": float(np.median([a.detail["freq_hz"] for a in sp])) if sp else None,
        "n_slow_waves": len(sw), "slow_waves_per_min_n3": len(sw) / n3_min if n3_min else None,
        "n_rems": len(rems), "rem_density_per_min_r": len(rems) / r_min if r_min else None,
        "n_arousals": len(ar), "arousal_index": len(ar) / tst_h if tst_h else None,
        "n_artifact_epochs": int(features.artifact.sum()) if features is not None else 0,
        "slow_waves_skipped": eeg is not None and eeg.unit.lower() not in ("uv", "µv"),
    }
    return AnnotationResult(rec, stages, rest, anns, summary)


# ----------------------------------------------------------------------------- export

def write_csv(res: AnnotationResult, path) -> None:
    import csv
    start = res.rec.start
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["onset_s", "clock_time", "duration_s", "annotation", "channel", "details"])
        for a in res.annotations:
            clock = (start + dt.timedelta(seconds=a.onset)).strftime("%H:%M:%S") if start else ""
            w.writerow([f"{a.onset:.2f}", clock, f"{a.duration:.2f}", a.label, a.channel,
                        "; ".join(f"{k}={v}" for k, v in a.detail.items())])


def write_edfplus(res: AnnotationResult, path) -> None:
    from .edfplus import write_annotations_edf
    write_annotations_edf(path, [(a.onset, a.duration, a.label) for a in res.annotations], res.rec.start,
                          patient=res.rec.patient or "X X X X")
