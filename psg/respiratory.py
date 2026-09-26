"""AASM respiratory event scoring (Manual for the Scoring of Sleep and Associated Events,
section VIII, adult rules).

Rules implemented
-----------------
Apnea:  peak flow excursion drops >= 90 % of pre-event baseline for >= 10 s.
    obstructive - inspiratory effort continues/increases throughout the absent airflow
    central     - effort absent throughout
    mixed       - effort absent at first, then resumes in the latter part of the event
Hypopnea (AASM recommended rule): flow drops >= 30 % of baseline for >= 10 s AND is followed
    by a >= 3 % oxygen desaturation OR an EEG arousal (a 4 % desaturation-only variant is
    available as `hypopnea_desat=4.0, hypopnea_arousal=False` - the AASM
    'acceptable' rule required by CMS/Medicare).
    Optional hypopnea classification: obstructive if snoring, inspiratory flow
    flattening, or thoraco-abdominal paradox is present during the event; central otherwise.
Baseline: breath amplitude over the preceding 2 minutes (see preprocess.causal_baseline).
Effort is judged on the thoracic and abdominal belts individually (the RIP sum cancels out
during obstruction, when the two belts move paradoxically).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import preprocess as pp
from .io import Recording, STAGE_W, STAGE_UNK, STAGE_NAMES, EPOCH_S

MIN_RESP_FS = 4.0  # Hz


@dataclass
class ScoringParams:
    min_duration: float = 10.0        # s, both apneas and hypopneas
    apnea_drop: float = 0.90          # fraction of baseline
    hypopnea_drop: float = 0.30
    hypopnea_desat: float = 3.0       # % points
    hypopnea_arousal: bool = True     # allow arousal as the alternative criterion
    noise_floor: bool = True          # subtract the envelope's no-breathing floor
    desat_pre_s: float = 45.0         # how far before onset to look for the pre-event SpO2 level
    desat_post_s: float = 60.0        # how long after event end the nadir may occur
    desat_baseline: str = "max"       # 'max' | 'walkback' | 'p90' (max matched expert scoring best)
    effort_absent: float = 0.18       # belt amplitude ratio below which effort counts as absent
    effort_resume: float = 0.30       # ratio the belts must regain for a mixed apnea
    hyp_effort_margin: float = 0.10   # effort drop this much smaller than flow drop -> obstructive
    merge_gap_s: float = 2.0          # join reduced-breathing runs separated by less than this
    max_duration: float = 180.0       # longer runs are treated as sensor problems
    bad_fraction_reject: float = 0.3  # discard event if this much of it lies on bad flow data
    sleep_only: bool = True           # ignore events whose midpoint is in a Wake epoch


@dataclass
class RespEvent:
    onset: float
    duration: float
    kind: str                       # 'apnea' | 'hypopnea'
    subtype: str                    # 'obstructive' | 'central' | 'mixed'
    flow_drop: float                # 0..1 fraction of baseline lost
    effort_drop: float | None = None
    desat: float | None = None      # % points, None if none found
    desat_nadir: float | None = None
    arousal: bool = False
    snoring: bool = False
    paradox: bool = False
    stage: str | None = None
    position: int | None = None
    notes: str = ""

    @property
    def end(self) -> float:
        return self.onset + self.duration

    @property
    def label(self) -> str:
        return f"{self.subtype.capitalize()} {self.kind}"

    @property
    def code(self) -> str:
        return {"apnea": "A", "hypopnea": "H"}[self.kind] + self.subtype[0].upper()


@dataclass
class ScoringResult:
    events: list[RespEvent]
    params: ScoringParams
    fs: float                              # rate of the derived traces below
    flow_env: np.ndarray
    flow_baseline: np.ndarray
    flow_ratio: np.ndarray
    flow_bad: np.ndarray
    effort_ratio: np.ndarray | None        # max of the belts, at `fs`
    spo2_clean: np.ndarray | None
    spo2_fs: float | None
    spo2_bad: np.ndarray | None
    arousals: list[tuple[float, float]] = field(default_factory=list)  # (onset, duration)
    flow_source: str = ""
    effort_source: str = ""
    noise_floor: float = 0.0
    rejected: dict[str, int] = field(default_factory=dict)  # reason -> count of dropped candidates

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for e in self.events:
            c[e.label] = c.get(e.label, 0) + 1
        return c


# ----------------------------------------------------------------------------- helpers

def _amplitude_ratio(x: np.ndarray, fs: float, floor: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(envelope, baseline, ratio) with a two-pass baseline that ignores detected reductions."""
    env = np.maximum(pp.breath_amplitude(x, fs) - floor, 0.0)
    exclude = None
    for _ in range(2):
        base = pp.causal_baseline(env, fs, exclude=exclude)
        ratio = env / base
        exclude = ratio <= 0.7
    return env, base, ratio


def _effort_ratio(rec: Recording, fs_out: float) -> tuple[np.ndarray | None, str]:
    """Per-belt amplitude ratios, combined as the element-wise maximum (effort is present if
    either belt keeps moving).  Falls back to the RIP sum when no belts are present."""
    ratios = []
    names = []
    for role in ("thorax", "abdomen"):
        ch = rec.get(role)
        if ch is None or ch.fs < MIN_RESP_FS:
            continue
        y, _ = pp.clean_respiratory(ch.data.astype(np.float64), ch.fs)
        _, _, r = _amplitude_ratio(y, ch.fs)
        if abs(ch.fs - fs_out) > 1e-6:
            r = pp.resample(r, ch.fs, fs_out)
        ratios.append(r)
        names.append(ch.label)
    if not ratios:
        ch = rec.get("effort_sum")
        if ch is None or ch.fs < MIN_RESP_FS:
            return None, ""
        y, _ = pp.clean_respiratory(ch.data.astype(np.float64), ch.fs)
        _, _, r = _amplitude_ratio(y, ch.fs)
        if abs(ch.fs - fs_out) > 1e-6:
            r = pp.resample(r, ch.fs, fs_out)
        return r, ch.label
    n = min(len(r) for r in ratios)
    return np.maximum.reduce([r[:n] for r in ratios]), "+".join(names)


def _sustained(ratio: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """Running median of the amplitude ratio: the level held over `win_s`, which ignores a
    single stray breath or noise spike inside an event."""
    from scipy import ndimage
    step = max(int(fs / 2), 1)             # evaluate at 2 Hz, then hold
    coarse = ratio[::step]
    med = ndimage.median_filter(coarse, size=max(int(win_s * fs / step), 1), mode="nearest")
    return np.repeat(med, step)[:len(ratio)]


def _candidate_runs(ratio: np.ndarray, fs: float, p: ScoringParams) -> list[tuple[int, int]]:
    reduced = _sustained(ratio, fs, 3.0) <= (1.0 - p.hypopnea_drop)
    merged: list[list[int]] = []
    gap = int(p.merge_gap_s * fs)
    for s, e in pp.runs(reduced):
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    min_n = int(p.min_duration * fs)
    return [(s, e) for s, e in merged if e - s >= min_n]


def _classify_apnea(eff: np.ndarray | None, p: ScoringParams) -> tuple[str, float | None]:
    """Obstructive / central / mixed from the belt amplitude ratio during the event."""
    if eff is None or len(eff) < 3:
        return "obstructive", None
    drop = float(1.0 - np.nanmean(eff))
    third = max(len(eff) // 3, 1)
    if np.nanmedian(eff) < p.effort_absent:
        return "central", drop
    if np.nanmedian(eff[:third]) < p.effort_absent and np.nanmedian(eff[-third:]) >= p.effort_resume:
        return "mixed", drop
    return "obstructive", drop


def _link_desaturation(spo2: np.ndarray | None, fs: float | None, onset: float, end: float,
                       p: ScoringParams) -> tuple[float | None, float | None]:
    """Fall from the pre-event SpO2 level to the nadir reached after the event onset.

    'walkback' (default): from the nadir, walk back in time to the peak immediately before
    the fall, i.e. the level the trace left when it started dropping - what a scorer reads
    off the screen.  'max' / 'p90' summarise the `desat_pre_s` seconds before onset.
    """
    if spo2 is None or fs is None:
        return None, None
    step = max(int(round(fs)), 1)
    x = spo2[::step]                       # 1 Hz
    i_on, i_end = int(onset), int(min(end + p.desat_post_s, len(x) - 1))
    i_lo = int(max(onset - p.desat_pre_s, 0))
    post = x[i_on:i_end + 1]
    if post.size == 0 or not np.isfinite(post).any():
        return None, None
    nadir_rel = int(np.nanargmin(post))
    nadir = float(post[nadir_rel])
    i_nadir = i_on + nadir_rel
    if p.desat_baseline == "walkback":
        run_max = nadir
        i = i_nadir
        while i > i_lo:
            i -= 1
            v = x[i]
            if not np.isfinite(v):
                break
            if v > run_max:
                run_max = v
            elif v < run_max - 1.0:  # we crossed the preceding peak and are descending again
                break
        base = run_max
    else:
        pre = x[i_lo:max(i_on, i_lo + 1)]
        pre = pre[np.isfinite(pre)]
        if pre.size == 0:
            return None, None
        base = float(np.max(pre)) if p.desat_baseline == "max" else float(np.percentile(pre, 90))
    return base - nadir, nadir


def _paradox(thorax: np.ndarray | None, abdomen: np.ndarray | None, fs: float | None,
             onset: float, end: float) -> bool:
    """Thoraco-abdominal paradox: belts move out of phase during the event but in phase before it."""
    if thorax is None or abdomen is None or fs is None:
        return False
    def corr(a: float, b: float) -> float:
        i, j = int(a * fs), int(b * fs)
        x, y = thorax[i:j], abdomen[i:j]
        n = min(len(x), len(y))
        if n < 3 * fs:
            return 1.0
        x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / d) if d > 0 else 1.0
    return corr(onset, end) < 0.0 and corr(max(onset - 60, 0), onset) > 0.3


def _flattening(flow: np.ndarray, fs: float, onset: float, end: float, min_breaths: int = 2) -> bool:
    """Inspiratory flow limitation: breaths with a plateau instead of a rounded peak.

    For each inspiration (positive half-cycle) the flatness is mean/max of the flow over
    that half-cycle; a sinusoid gives 2/pi = 0.64, a plateau approaches 1.  The event counts
    as flow-limited when the median flatness of its breaths exceeds 0.75.  Needs at least a
    few samples per breath, so it is skipped for very low sampling rates.
    """
    if fs < 8:
        return False
    x = flow[int(onset * fs):int(end * fs)]
    if x.size < 3 * fs:
        return False
    x = x - np.median(x)
    flat = []
    for a, b in pp.runs(x > 0):
        if b - a >= max(int(0.5 * fs), 4):
            seg = x[a:b]
            m = seg.max()
            if m > 0:
                flat.append(seg.mean() / m)
    return len(flat) >= min_breaths and float(np.median(flat)) > 0.75


def _snoring(snore_env: np.ndarray | None, fs: float | None, onset: float, end: float, ref: float) -> bool:
    if snore_env is None or fs is None or ref <= 0:
        return False
    seg = snore_env[int(onset * fs):int(end * fs)]
    return seg.size > 0 and float(np.percentile(seg, 75)) > 1.5 * ref


def _stage_at(stages: np.ndarray | None, t: float) -> int:
    if stages is None:
        return STAGE_UNK
    k = int(t // EPOCH_S)
    return int(stages[k]) if 0 <= k < len(stages) else STAGE_UNK


def empty_result(params: ScoringParams | None = None, reason: str = "") -> ScoringResult:
    """Result object for recordings where respiratory scoring is impossible."""
    z = np.zeros(1)
    return ScoringResult(events=[], params=params or ScoringParams(), fs=1.0, flow_env=z, flow_baseline=z,
                         flow_ratio=z, flow_bad=np.zeros(1, dtype=bool), effort_ratio=None, spo2_clean=None,
                         spo2_fs=None, spo2_bad=None, flow_source=reason)


# ----------------------------------------------------------------------------- main entry

def score_respiratory(rec: Recording, params: ScoringParams | None = None,
                      stages: np.ndarray | None = None,
                      arousals: list[tuple[float, float]] | None = None) -> ScoringResult:
    """Detect and classify apneas and hypopneas for one recording.

    `stages` (canonical per-epoch codes) and `arousals` ((onset, duration) list) are optional;
    they come from the staging/arousal modules or from expert files.
    """
    p = params or ScoringParams()
    rejected: dict[str, int] = {}
    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    # --- airflow (or an effort surrogate when no usable flow sensor exists) ------------------
    # Breath-by-breath scoring needs a few samples per breath; 1-Hz summary channels are unusable.
    usable = lambda ch: ch is not None and ch.fs >= MIN_RESP_FS
    flow_ch = rec.get("flow")
    if usable(flow_ch):
        flow_raw, fs, source = flow_ch.data.astype(np.float64), flow_ch.fs, flow_ch.label
    else:
        ch = next((c for c in (rec.get("effort_sum"), rec.get("thorax"), rec.get("abdomen")) if usable(c)), None)
        if ch is None:
            why = (f"airflow channel '{flow_ch.label}' is sampled at {flow_ch.fs:g} Hz (need >= {MIN_RESP_FS} Hz)"
                   if flow_ch is not None else "no airflow or effort channel")
            raise ValueError(f"Respiratory scoring not possible: {why}")
        flow_raw, fs, source = ch.data.astype(np.float64), ch.fs, f"{ch.label} (effort used as flow surrogate)"
    flow, flow_bad = pp.clean_respiratory(flow_raw, fs)
    raw_env = pp.breath_amplitude(flow, fs)
    floor = pp.noise_floor(flow, fs, raw_env, flow_bad) if p.noise_floor else 0.0
    flow_env, baseline, ratio = _amplitude_ratio(flow, fs, floor)
    cands = _candidate_runs(ratio, fs, p)

    # --- effort, belts, SpO2, snore ----------------------------------------------------------
    eff_ratio, eff_source = _effort_ratio(rec, fs)
    thor, abdo = rec.get("thorax"), rec.get("abdomen")
    if thor is not None and abdo is not None and thor.fs >= MIN_RESP_FS and abs(thor.fs - abdo.fs) < 1e-6:
        thor_f, abdo_f, belt_fs = pp.filt(thor.data, thor.fs, 0.1, 1.0), pp.filt(abdo.data, abdo.fs, 0.1, 1.0), thor.fs
    else:
        thor_f = abdo_f = belt_fs = None

    spo2_ch = rec.get("spo2")
    spo2 = spo2_bad = spo2_fs = None
    if spo2_ch is not None:
        spo2, spo2_bad = pp.clean_spo2(spo2_ch.data, spo2_ch.fs)
        spo2_fs = spo2_ch.fs

    snore_ch = rec.get("snore")
    snore_env = snore_fs = None
    snore_ref = 0.0
    if snore_ch is not None:
        snore_env = pp.moving_rms(snore_ch.data - np.median(snore_ch.data), int(0.5 * snore_ch.fs))
        snore_fs = snore_ch.fs
        snore_ref = float(np.median(snore_env))

    pos_ch = rec.get("position")
    arousals = arousals or []
    ar_onsets = np.array([a[0] for a in arousals]) if arousals else np.empty(0)

    # --- build events -----------------------------------------------------------------------
    events: list[RespEvent] = []
    apnea_thr = 1.0 - p.apnea_drop
    min_n = int(p.min_duration * fs)
    sustained10 = _sustained(ratio, fs, p.min_duration)
    for s, e in cands:
        onset, end = s / fs, e / fs
        dur = end - onset
        if dur > p.max_duration:
            reject("too long (sensor?)")
            continue
        if flow_bad[s:e].mean() > p.bad_fraction_reject:
            reject("bad flow signal")
            continue
        seg = ratio[s:e]
        # A run of reduced breathing is one event, typed by its most severe part: it is an
        # apnea when some 10-s stretch inside it holds a >= 90 % drop.
        deepest = float(np.min(sustained10[s:e])) if dur >= p.min_duration else 1.0
        if deepest <= apnea_thr:
            kind = "apnea"
            flow_drop = 1.0 - deepest
        else:
            kind = "hypopnea"
            flow_drop = float(1.0 - np.median(seg))

        desat, nadir = _link_desaturation(spo2, spo2_fs, onset, end, p)
        has_arousal = bool(((ar_onsets >= onset - 2.0) & (ar_onsets <= end + 5.0)).any()) if ar_onsets.size else False

        stage_code = _stage_at(stages, onset + dur / 2)
        if kind == "hypopnea":
            ok_desat = desat is not None and desat >= p.hypopnea_desat
            ok_arousal = p.hypopnea_arousal and has_arousal
            if not (ok_desat or ok_arousal):
                reject("hypopnea without desaturation/arousal")
                continue
        if p.sleep_only and stages is not None and stage_code == STAGE_W:
            reject("during wake")
            continue

        snoring = _snoring(snore_env, snore_fs, onset, end, snore_ref)
        paradox = _paradox(thor_f, abdo_f, belt_fs, onset, end)
        flat = _flattening(flow, fs, onset, end)
        er = eff_ratio[s:e] if eff_ratio is not None else None
        notes = []
        if kind == "apnea":
            subtype, effort_drop = _classify_apnea(er, p)
        else:
            # AASM optional hypopnea classification: obstructive if snoring, inspiratory flattening or paradox; central
            # only when none is present.  We also count effort that is clearly better
            # preserved than airflow as evidence of upper-airway resistance.
            effort_drop = float(1.0 - np.nanmean(er)) if er is not None and len(er) else None
            preserved = effort_drop is not None and effort_drop < flow_drop - p.hyp_effort_margin
            subtype = "obstructive" if (snoring or paradox or flat or preserved) else "central"
            if flat:
                notes.append("flattening")
            if preserved:
                notes.append("effort preserved")

        position = None
        if pos_ch is not None:
            k = int((onset + dur / 2) * pos_ch.fs)
            if 0 <= k < len(pos_ch.data):
                position = int(round(float(pos_ch.data[k])))

        events.append(RespEvent(
            onset=onset, duration=dur, kind=kind, subtype=subtype, flow_drop=flow_drop,
            effort_drop=None if effort_drop is None or np.isnan(effort_drop) else effort_drop,
            desat=None if desat is None else round(desat, 1),
            desat_nadir=None if nadir is None else round(nadir, 1),
            arousal=has_arousal, snoring=snoring, paradox=paradox,
            stage=STAGE_NAMES.get(stage_code) if stages is not None else None,
            position=position, notes=", ".join(notes),
        ))

    return ScoringResult(
        events=events, params=p, fs=fs, flow_env=flow_env, flow_baseline=baseline, flow_ratio=ratio,
        flow_bad=flow_bad, effort_ratio=eff_ratio, spo2_clean=spo2, spo2_fs=spo2_fs, spo2_bad=spo2_bad,
        arousals=arousals, flow_source=source, effort_source=eff_source, noise_floor=floor,
        rejected=rejected,
    )
