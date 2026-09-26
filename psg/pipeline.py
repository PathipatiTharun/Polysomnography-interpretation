"""End-to-end analysis of one recording: staging -> arousals -> respiratory events -> indices -> diagnosis."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import preprocess as pp
from .arousal import ArousalParams, detect_arousals
from .io import (EPOCH_S, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R, STAGE_UNK, STAGE_W, Recording)
from .respiratory import RespEvent, ScoringParams, ScoringResult, empty_result, score_respiratory
from .spo2 import Desaturation, detect_desaturations, oxygen_summary
from .staging import EpochFeatures, StagingParams, sleep_summary, stage_sleep

SLEEP_CODES = [STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R]


@dataclass
class AnalysisOptions:
    staging_source: str = "auto"      # 'auto' (algorithm) | 'expert' (use annotation file when present)
    detect_arousals: bool = True
    scoring: ScoringParams = field(default_factory=ScoringParams)
    staging: StagingParams = field(default_factory=StagingParams)
    arousal: ArousalParams = field(default_factory=ArousalParams)


@dataclass
class AnalysisResult:
    rec: Recording
    options: AnalysisOptions
    stages: np.ndarray
    stage_source: str
    features: EpochFeatures | None
    arousals: list[tuple[float, float]]
    resp: ScoringResult
    desaturations: list[Desaturation]
    summary: dict
    diagnosis: dict
    runtime_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def events(self) -> list[RespEvent]:
        return self.resp.events


def ahi_severity(ahi: float) -> str:
    """AASM / ICSD-3 adult severity bands."""
    if ahi < 5:
        return "Normal"
    if ahi < 15:
        return "Mild"
    if ahi < 30:
        return "Moderate"
    return "Severe"


def _sleep_mask_1hz(stages: np.ndarray, n_sec: int) -> np.ndarray:
    m = np.repeat(np.isin(stages, SLEEP_CODES), int(EPOCH_S))
    out = np.zeros(n_sec, dtype=bool)
    out[:min(len(m), n_sec)] = m[:n_sec]
    return out


def _signal_quality(rec: Recording, resp: ScoringResult) -> dict[str, float]:
    """Fraction of usable signal per key channel (not flat, not clipped, SpO2 in range)."""
    q: dict[str, float] = {}
    for role in ("eeg", "eog_l", "emg", "flow", "thorax", "abdomen"):
        ch = rec.get(role)
        if ch is None:
            continue
        bad = pp.flatline_mask(ch.data, ch.fs) | pp.clipping_mask(ch.data, ch.fs)
        q[ch.label] = float(1.0 - bad.mean())
    if resp.spo2_bad is not None:
        q[rec.roles["spo2"]] = float(1.0 - resp.spo2_bad.mean())
    return q


def compute_indices(res_events: list[RespEvent], stages: np.ndarray, desats: list[Desaturation],
                    arousals: list[tuple[float, float]], spo2: np.ndarray | None, spo2_fs: float | None,
                    position: np.ndarray | None = None, pos_fs: float | None = None,
                    period: tuple[int, int] | None = None) -> dict:
    ss = sleep_summary(stages, period)
    tst_h = ss["tst_min"] / 60.0
    n = len(res_events)
    by = lambda kind, sub=None: [e for e in res_events if e.kind == kind and (sub is None or e.subtype == sub)]
    oa, ca, ma = by("apnea", "obstructive"), by("apnea", "central"), by("apnea", "mixed")
    hyp = by("hypopnea")
    oh, ch = by("hypopnea", "obstructive"), by("hypopnea", "central")
    per_h = lambda k: (k / tst_h) if tst_h > 0 else float("nan")

    # Sleep-only desaturations for the ODI.
    sleep1 = _sleep_mask_1hz(stages, int(len(stages) * EPOCH_S))
    desat_sleep = [d for d in desats if int(d.nadir_time) < len(sleep1) and sleep1[int(d.nadir_time)]]

    # REM vs NREM AHI.
    rem_h = (stages == STAGE_R).sum() * EPOCH_S / 3600
    nrem_h = np.isin(stages, [STAGE_N1, STAGE_N2, STAGE_N3]).sum() * EPOCH_S / 3600
    rem_ev = [e for e in res_events if e.stage == "R"]
    nrem_ev = [e for e in res_events if e.stage in ("N1", "N2", "N3")]

    # Supine vs non-supine AHI (UCDDB: 1 = supine; other labs differ, so this is best-effort).
    supine = None
    if position is not None and pos_fs is not None:
        pos_epoch = np.array([np.median(position[int(k * EPOCH_S * pos_fs):int((k + 1) * EPOCH_S * pos_fs)])
                              if int(k * EPOCH_S * pos_fs) < len(position) else np.nan for k in range(len(stages))])
        sup = np.isin(stages, SLEEP_CODES) & (np.round(pos_epoch) == 1)
        sup_h = sup.sum() * EPOCH_S / 3600
        sup_ev = [e for e in res_events if e.position == 1]
        supine = {"supine_hours": sup_h, "supine_ahi": len(sup_ev) / sup_h if sup_h > 0.1 else None,
                  "nonsupine_ahi": (n - len(sup_ev)) / (tst_h - sup_h) if tst_h - sup_h > 0.1 else None}

    ox = oxygen_summary(spo2, spo2_fs, sleep1) if spo2 is not None else {}
    durations = [e.duration for e in res_events]
    return {
        **ss,
        "tst_h": tst_h,
        "n_events": n,
        "n_obstructive_apnea": len(oa), "n_central_apnea": len(ca), "n_mixed_apnea": len(ma),
        "n_hypopnea": len(hyp), "n_obstructive_hypopnea": len(oh), "n_central_hypopnea": len(ch),
        "ahi": per_h(n),
        "ai": per_h(len(oa) + len(ca) + len(ma)),
        "hi": per_h(len(hyp)),
        "oahi": per_h(len(oa) + len(ma) + len(oh)),       # obstructive AHI (mixed counted as obstructive)
        "cahi": per_h(len(ca) + len(ch)),                  # central AHI
        "central_apnea_index": per_h(len(ca)),
        "rem_ahi": len(rem_ev) / rem_h if rem_h > 0.1 else None,
        "nrem_ahi": len(nrem_ev) / nrem_h if nrem_h > 0.1 else None,
        "odi": per_h(len(desat_sleep)),
        "n_desaturations": len(desat_sleep),
        "arousal_index": per_h(len(arousals)),
        "n_arousals": len(arousals),
        "mean_event_duration": float(np.mean(durations)) if durations else None,
        "max_event_duration": float(np.max(durations)) if durations else None,
        "spo2": ox,
        "position": supine,
    }


def make_diagnosis(ix: dict) -> dict:
    """Plain-language interpretation of the indices, following AASM/ICSD-3 adult criteria.
    This is decision support for a physician, not a diagnosis on its own."""
    ahi = ix["ahi"]
    findings: list[str] = []
    if not ix.get("respiratory_scored", True):
        findings.append(f"Sleep efficiency {ix['sleep_efficiency']:.0f} %, arousal index {ix['arousal_index']:.1f}/h.")
        return {"primary": "Respiratory events not scored (no usable airflow/effort signal)",
                "severity": "Unknown", "findings": findings,
                "flags": ["Sleep staging only. An AHI requires airflow and effort sensors sampled at >= 4 Hz."]}
    if not np.isfinite(ahi):
        return {"primary": "Insufficient sleep scored to compute AHI", "severity": "Unknown",
                "findings": findings, "flags": []}
    severity = ahi_severity(ahi)
    total_ap = ix["n_obstructive_apnea"] + ix["n_central_apnea"] + ix["n_mixed_apnea"]
    central_share = (ix["n_central_apnea"] + ix["n_central_hypopnea"]) / ix["n_events"] if ix["n_events"] else 0.0
    flags: list[str] = []

    # Central sleep apnea (ICSD-3): central apneas + central hypopneas >= 5/h and > 50 % of all
    # events.  Hypopnea subtyping from belts is the least reliable part of scoring, so we also
    # require the central apnea index itself to reach 5/h before calling CSA predominant.
    if ahi < 5:
        primary = "No obstructive sleep apnea (AHI < 5/h)"
    elif ix["cahi"] >= 5 and central_share > 0.5 and ix["central_apnea_index"] >= 5:
        primary = f"Central sleep apnea predominant - {severity.lower()} (AHI {ahi:.1f}/h, central AHI {ix['cahi']:.1f}/h)"
        flags.append("Central events are > 50 % of all events: consider cardiac (e.g. heart failure), "
                     "opioid, high-altitude or treatment-emergent causes; look for Cheyne-Stokes breathing.")
    else:
        primary = f"Obstructive sleep apnea - {severity.lower()} (AHI {ahi:.1f}/h)"
        if ix["central_apnea_index"] >= 5:
            flags.append(f"Central component present (central apnea index {ix['central_apnea_index']:.1f}/h, "
                         f"central AHI {ix['cahi']:.1f}/h).")
        elif central_share > 0.5:
            flags.append(f"Many hypopneas lack obstructive features ({central_share * 100:.0f} % of events "
                         f"classified central) - review hypopnea classification manually.")

    findings.append(f"{ix['n_events']} respiratory events in {ix['tst_h']:.1f} h of sleep: "
                    f"{ix['n_obstructive_apnea']} obstructive, {ix['n_central_apnea']} central and "
                    f"{ix['n_mixed_apnea']} mixed apneas; {ix['n_hypopnea']} hypopneas "
                    f"({ix['n_obstructive_hypopnea']} obstructive, {ix['n_central_hypopnea']} central).")
    if ix.get("rem_ahi") is not None and ix.get("nrem_ahi") is not None and ix["nrem_ahi"] > 0:
        if ix["rem_ahi"] >= 2 * ix["nrem_ahi"] and ix["rem_ahi"] >= 5:
            flags.append(f"REM-related pattern: REM AHI {ix['rem_ahi']:.1f}/h vs NREM {ix['nrem_ahi']:.1f}/h.")
    pos = ix.get("position")
    if pos and pos.get("supine_ahi") is not None and pos.get("nonsupine_ahi") is not None:
        if pos["supine_ahi"] >= 2 * max(pos["nonsupine_ahi"], 0.1) and ahi >= 5:
            flags.append(f"Positional pattern: supine AHI {pos['supine_ahi']:.1f}/h vs non-supine "
                         f"{pos['nonsupine_ahi']:.1f}/h - positional therapy may help.")
    ox = ix.get("spo2") or {}
    if ox.get("nadir") is not None:
        findings.append(f"SpO2 mean {ox['mean']:.1f} %, nadir {ox['nadir']:.0f} %, "
                        f"time below 90 %: {ox['t90_min']:.1f} min; ODI {ix['odi']:.1f}/h.")
        if ox["t90_min"] >= 0.1 * ix["tst_min"] and ox["t90_min"] > 5:
            flags.append("Sustained hypoxaemia (>= 10 % of sleep below 90 %): assess for "
                         "sleep-related hypoventilation / underlying lung disease.")
    findings.append(f"Sleep efficiency {ix['sleep_efficiency']:.0f} %, arousal index {ix['arousal_index']:.1f}/h.")
    if ix["sleep_efficiency"] < 70:
        flags.append("Low sleep efficiency - results may underestimate disease; consider repeat study.")
    return {"primary": primary, "severity": severity, "findings": findings, "flags": flags}


def analyze(rec: Recording, options: AnalysisOptions | None = None,
            progress: Callable[[int, str], None] | None = None) -> AnalysisResult:
    opt = options or AnalysisOptions()
    say = progress or (lambda pct, msg: None)
    t0 = time.time()
    warnings: list[str] = []

    say(5, "Staging sleep")
    features = None
    if opt.staging_source == "expert" and rec.expert_stages is not None:
        stages = rec.expert_stages.copy()
        source = "expert annotation"
    else:
        if opt.staging_source == "expert":
            warnings.append("No expert hypnogram found: using automatic staging.")
        try:
            stages, features, _ = stage_sleep(rec, opt.staging)
            source = "automatic (rule-based)"
        except ValueError as e:
            warnings.append(f"Staging unavailable ({e}); treating the whole recording as sleep.")
            stages = np.full(rec.n_epochs, STAGE_N2, dtype=int)
            source = "none"
    # Pad/trim to the recording length.
    if len(stages) < rec.n_epochs:
        stages = np.concatenate([stages, np.full(rec.n_epochs - len(stages), STAGE_UNK)])
    stages = stages[:rec.n_epochs]

    say(30, "Detecting EEG arousals")
    arousals = detect_arousals(rec, stages, opt.arousal) if opt.detect_arousals else []

    say(50, "Scoring apneas and hypopneas (AASM)")
    try:
        resp = score_respiratory(rec, opt.scoring, stages=stages, arousals=arousals)
    except ValueError as e:
        warnings.append(str(e))
        resp = empty_result(opt.scoring, str(e))
        spo2 = rec.get("spo2")
        if spo2 is not None:  # still report oximetry
            resp.spo2_clean, resp.spo2_bad = pp.clean_spo2(spo2.data, spo2.fs)
            resp.spo2_fs = spo2.fs

    say(80, "Scoring oxygen desaturations")
    desats = []
    if resp.spo2_clean is not None:
        desats = detect_desaturations(resp.spo2_clean, resp.spo2_fs, drop=opt.scoring.hypopnea_desat)
    else:
        warnings.append("No SpO2 channel: hypopneas can only be confirmed by arousals.")

    say(90, "Computing indices")
    pos = rec.get("position")
    period = features.rest if features is not None and features.rest != (0, rec.n_epochs) else None
    if period is not None:
        warnings.append(f"Recording extends beyond the rest period: sleep measures use "
                        f"{period[0] * EPOCH_S / 3600:.1f} h - {period[1] * EPOCH_S / 3600:.1f} h (auto-detected lights off/on).")
    ix = compute_indices(resp.events, stages, desats, arousals, resp.spo2_clean, resp.spo2_fs,
                         pos.data if pos is not None else None, pos.fs if pos is not None else None, period)
    ix["respiratory_scored"] = resp.fs > 1.0 or bool(resp.events)
    ix["signal_quality"] = _signal_quality(rec, resp)
    for ch, q in ix["signal_quality"].items():
        if q < 0.8:
            warnings.append(f"{ch}: only {q * 100:.0f} % usable signal.")
    diag = make_diagnosis(ix)
    say(100, "Done")
    return AnalysisResult(rec, opt, stages, source, features, arousals, resp, desats, ix, diag,
                          time.time() - t0, warnings)
