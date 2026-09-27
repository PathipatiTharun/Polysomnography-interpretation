"""AASM respiratory event scoring (Manual for the Scoring of Sleep and Associated Events,
adult rules), parameterised so that every published rule version in `psg.rules` can be run.

Default rule (AASM v2.0 rule 1A, recommended, unchanged in later versions)
-------------------------------------------------------------------------
Apnea:     peak flow excursion drops >= 90 % of pre-event baseline for >= 10 s.
    obstructive - inspiratory effort continues/increases throughout the absent airflow
    central     - effort absent throughout
    mixed       - effort absent at first, then resumes in the latter part of the event
Hypopnea:  flow drops >= 30 % of baseline for >= 10 s AND a >= 3 % desaturation or an arousal
    (rule 1B / CMS: >= 4 % desaturation, no arousal).  Obstructive if snoring, inspiratory
    flattening or thoraco-abdominal paradox is present during the event; central otherwise.
Sleep:     an event counts if it begins or ends in a sleep epoch (v2.0 note 3).
Baseline:  breath amplitude over the preceding 2 minutes (see preprocess.causal_baseline).
Effort is judged on the thoracic and abdominal belts individually (the RIP sum cancels out
during obstruction, when the two belts move paradoxically).

Every scored event - and every candidate that was rejected - carries the list of criteria it
was tested against (`RespEvent.checks`), a confidence (how far it clears the thresholds) and
a plain-language explanation, so the physician can see why it was or was not counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import preprocess as pp
from .io import Recording, STAGE_W, STAGE_UNK, STAGE_NAMES, EPOCH_S

MIN_RESP_FS = 4.0  # Hz
SLEEP_CODES = (1, 2, 3, 4)


@dataclass
class ScoringParams:
    # --- rule thresholds (set from psg.rules.RuleVersion, adjustable with sliders) ----------
    rule_id: str = "aasm2012_rec"
    min_duration: float = 10.0        # s, both apneas and hypopneas
    apnea_drop: float = 0.90          # fraction of baseline
    hypopnea_drop: float = 0.30
    hypopnea_desat: float = 3.0       # % points
    hypopnea_arousal: bool = True     # allow arousal as the alternative criterion
    hypopnea_drop_alone: float | None = None  # Chicago 1999: this drop counts without desat/arousal
    amplitude_fraction: float = 0.0   # 2007 manual: share of the event that must meet the drop
    classify_hypopneas: bool = True   # obstructive / central hypopneas (AASM 2012 rules 2-3)
    sleep_rule: str = "begins_or_ends"  # 'begins_or_ends' (AASM v2.0 note 3) | 'midpoint' | 'off'
    # --- engineering settings --------------------------------------------------------------
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
    arousal_confidence: float = 0.75  # confidence granted by the (heuristic) arousal detector
    sleep_only: bool = True           # False disables the sleep rule entirely


@dataclass
class Check:
    """One criterion the event was tested against."""
    step: str            # flowchart step id (psg.rules.flowchart)
    label: str
    value: float | None
    threshold: float | None
    unit: str
    passed: bool
    score: float         # 0..1, how comfortably the criterion is met (0.5 = exactly at threshold)
    detail: str = ""
    counts: bool = True  # part of the rule (affects confidence) vs. informational / type classification

    def text(self) -> str:
        v = "—" if self.value is None else (f"{self.value:.0f}" if self.unit in ("%", "s") else f"{self.value:.2f}")
        t = "" if self.threshold is None else f" (rule: ≥{self.threshold:g}{self.unit})"
        return f"{self.label}: {v}{self.unit if self.value is not None else ''}{t}" + (f" — {self.detail}" if self.detail else "")


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
    confidence: float = 1.0         # that this is a scorable event under the chosen rule
    type_confidence: float = 1.0    # that the obstructive/central/mixed label is right
    checks: list[Check] = field(default_factory=list)
    accepted: bool = True
    reject_reason: str = ""

    @property
    def end(self) -> float:
        return self.onset + self.duration

    @property
    def label(self) -> str:
        return f"{self.subtype.capitalize()} {self.kind}"

    @property
    def code(self) -> str:
        return {"apnea": "A", "hypopnea": "H"}[self.kind] + self.subtype[0].upper()

    @property
    def rule_checks(self) -> list[Check]:
        return [c for c in self.checks if c.counts]

    @property
    def limiting(self) -> Check | None:
        """The rule criterion with the smallest margin - what the confidence is limited by."""
        rc = self.rule_checks
        return min(rc, key=lambda c: c.score) if rc else None

    @property
    def path(self) -> list[str]:
        """Flowchart step ids this event passed through (for highlighting)."""
        ids = ["start", "sleep", "signal", "apnea"]
        if self.kind == "hypopnea":
            ids += ["hypopnea", "confirm"]
        if self.accepted:
            ids += (["effort"] if self.kind == "apnea" else ["hyp_type"]) + [self.code]
        else:
            failed = [c.step for c in self.checks if not c.passed]
            ids.append({"sleep": "reject_wake", "signal": "reject_signal", "confirm": "reject_confirm",
                        "hypopnea": "reject_drop"}.get(failed[0] if failed else "", "reject_drop"))
        return ids

    def explanation(self) -> str:
        why = "; ".join(c.text() for c in self.checks)
        if not self.accepted:
            return f"Not scored — {self.reject_reason}. {why}."
        lim = self.limiting
        lim_txt = f" Weakest criterion: {lim.label.lower()}." if lim is not None and lim.score < 0.9 else ""
        stage = f" during {self.stage}" if self.stage else ""
        return (f"Scored as {self.label.lower()}{stage} with {self.confidence * 100:.0f}% confidence "
                f"(type {self.type_confidence * 100:.0f}%).{lim_txt} {why}.")


@dataclass
class RespSignals:
    """Everything derived from the raw channels that the rule thresholds do not depend on.
    Computing this is the slow part (~10 s); re-scoring with new thresholds is then ~1 s."""
    fs: float
    flow: np.ndarray            # band-limited airflow
    raw_env: np.ndarray         # breath amplitude before floor subtraction
    flow_bad: np.ndarray
    floor: float
    flow_source: str
    effort_ratio: np.ndarray | None
    effort_source: str
    thor_f: np.ndarray | None
    abdo_f: np.ndarray | None
    belt_fs: float | None
    spo2: np.ndarray | None
    spo2_fs: float | None
    spo2_bad: np.ndarray | None
    snore_env: np.ndarray | None
    snore_fs: float | None
    snore_ref: float
    position: np.ndarray | None
    position_fs: float | None
    _ratio_cache: dict = field(default_factory=dict, repr=False)

    def ratio(self, use_floor: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(envelope, baseline, ratio) for the chosen floor setting, cached."""
        if use_floor not in self._ratio_cache:
            self._ratio_cache[use_floor] = _amplitude_ratio(self.raw_env, self.fs, self.floor if use_floor else 0.0)
        return self._ratio_cache[use_floor]


@dataclass
class ScoringResult:
    events: list[RespEvent]                # accepted events
    candidates: list[RespEvent]            # every candidate, accepted or not, in time order
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
    signals: RespSignals | None = None
    stages: np.ndarray | None = None

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for e in self.events:
            c[e.label] = c.get(e.label, 0) + 1
        return c

    @property
    def rejected_events(self) -> list[RespEvent]:
        return [c for c in self.candidates if not c.accepted]

    def events_at_confidence(self, min_conf: float) -> list[RespEvent]:
        return [e for e in self.events if e.confidence >= min_conf]


# ----------------------------------------------------------------------------- helpers

def _sig(x: float, width: float) -> float:
    """Soft margin: 0.5 exactly at the threshold, ~0.95 three widths past it."""
    return float(1.0 / (1.0 + math.exp(-x / max(width, 1e-9))))


def _amplitude_ratio(env_raw: np.ndarray, fs: float, floor: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(envelope, baseline, ratio) with a two-pass baseline that ignores detected reductions."""
    env = np.maximum(env_raw - floor, 0.0)
    exclude = None
    for _ in range(2):
        base = pp.causal_baseline(env, fs, exclude=exclude)
        ratio = env / base
        exclude = ratio <= 0.7
    return env, base, ratio


def _effort_ratio(rec: Recording, fs_out: float) -> tuple[np.ndarray | None, str]:
    """Per-belt amplitude ratios, combined as the element-wise maximum (effort is present if
    either belt keeps moving).  Falls back to the RIP sum when no belts are present."""
    ratios, names = [], []
    for role in ("thorax", "abdomen"):
        ch = rec.get(role)
        if ch is None or ch.fs < MIN_RESP_FS:
            continue
        y, _ = pp.clean_respiratory(ch.data.astype(np.float64), ch.fs)
        _, _, r = _amplitude_ratio(pp.breath_amplitude(y, ch.fs), ch.fs)
        if abs(ch.fs - fs_out) > 1e-6:
            r = pp.resample(r, ch.fs, fs_out)
        ratios.append(r)
        names.append(ch.label)
    if not ratios:
        ch = rec.get("effort_sum")
        if ch is None or ch.fs < MIN_RESP_FS:
            return None, ""
        y, _ = pp.clean_respiratory(ch.data.astype(np.float64), ch.fs)
        _, _, r = _amplitude_ratio(pp.breath_amplitude(y, ch.fs), ch.fs)
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


def _classify_apnea(eff: np.ndarray | None, p: ScoringParams) -> tuple[str, float | None, float, str]:
    """(subtype, effort_drop, type_confidence, detail) from the belt amplitude ratio."""
    if eff is None or len(eff) < 3:
        return "obstructive", None, 0.3, "no effort belts: obstructive assumed"
    drop = float(1.0 - np.nanmean(eff))
    third = max(len(eff) // 3, 1)
    med, first, last = float(np.nanmedian(eff)), float(np.nanmedian(eff[:third])), float(np.nanmedian(eff[-third:]))
    if med < p.effort_absent:
        conf = _sig(p.effort_absent - med, 0.06)
        return "central", drop, conf, f"belts at {med * 100:.0f}% of baseline throughout (absent < {p.effort_absent * 100:.0f}%)"
    if first < p.effort_absent and last >= p.effort_resume:
        conf = min(_sig(p.effort_absent - first, 0.06), _sig(last - p.effort_resume, 0.1))
        return "mixed", drop, conf, f"belts {first * 100:.0f}% at start, {last * 100:.0f}% at end"
    conf = _sig(med - p.effort_absent, 0.08)
    return "obstructive", drop, conf, f"belts kept {med * 100:.0f}% of baseline movement"


def _link_desaturation(spo2: np.ndarray | None, fs: float | None, onset: float, end: float,
                       p: ScoringParams) -> tuple[float | None, float | None]:
    """Fall from the pre-event SpO2 level to the nadir reached after the event onset."""
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
        run_max, i = nadir, i_nadir
        while i > i_lo:
            i -= 1
            v = x[i]
            if not np.isfinite(v):
                break
            if v > run_max:
                run_max = v
            elif v < run_max - 1.0:
                break
        base = run_max
    else:
        pre = x[i_lo:max(i_on, i_lo + 1)]
        pre = pre[np.isfinite(pre)]
        if pre.size == 0:
            return None, None
        base = float(np.max(pre)) if p.desat_baseline == "max" else float(np.percentile(pre, 90))
    return base - nadir, nadir


def _paradox(thorax, abdomen, fs, onset: float, end: float) -> bool:
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
    """Inspiratory flow limitation: breaths with a plateau instead of a rounded peak (a sinusoid
    has mean/max = 0.64 over a half-cycle; a plateau approaches 1)."""
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


def _snoring(snore_env, fs, onset: float, end: float, ref: float) -> bool:
    if snore_env is None or fs is None or ref <= 0:
        return False
    seg = snore_env[int(onset * fs):int(end * fs)]
    return seg.size > 0 and float(np.percentile(seg, 75)) > 1.5 * ref


def _stage_at(stages: np.ndarray | None, t: float) -> int:
    if stages is None:
        return STAGE_UNK
    k = int(t // EPOCH_S)
    return int(stages[k]) if 0 <= k < len(stages) else STAGE_UNK


def _sleep_check(stages: np.ndarray | None, onset: float, end: float, p: ScoringParams) -> Check:
    """AASM v2.0 note 3: score the event if it begins or ends in a sleep epoch; do not score it
    if it lies entirely within wake.  `score` is the share of the event spent in sleep."""
    if stages is None or not p.sleep_only or p.sleep_rule == "off":
        return Check("sleep", "Sleep", None, None, "", True, 1.0, "no hypnogram: sleep assumed")
    k0, k1 = int(onset // EPOCH_S), int(end // EPOCH_S)
    codes = [int(stages[k]) for k in range(k0, k1 + 1) if 0 <= k < len(stages)]
    if not codes:
        return Check("sleep", "Sleep", None, None, "", True, 1.0, "outside hypnogram")
    in_sleep = [c in SLEEP_CODES for c in codes]
    frac = float(np.mean(in_sleep))
    if p.sleep_rule == "midpoint":
        ok = _stage_at(stages, (onset + end) / 2) in SLEEP_CODES
    else:
        ok = in_sleep[0] or in_sleep[-1]
    names = "/".join(dict.fromkeys(STAGE_NAMES.get(c, "?") for c in codes))
    return Check("sleep", "Sleep", frac * 100, None, "%", ok, max(frac, 0.3) if ok else 0.0,
                 f"epochs {names}" + ("" if ok else " (entirely wake)"))


# ----------------------------------------------------------------------------- signal preparation

def prepare_signals(rec: Recording, params: ScoringParams | None = None) -> RespSignals:
    """The threshold-independent part of scoring: cleaned traces, envelopes, effort, SpO2, snore."""
    p = params or ScoringParams()
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
    floor = pp.noise_floor(flow, fs, raw_env, flow_bad)

    eff_ratio, eff_source = _effort_ratio(rec, fs)
    thor, abdo = rec.get("thorax"), rec.get("abdomen")
    if thor is not None and abdo is not None and thor.fs >= MIN_RESP_FS and abs(thor.fs - abdo.fs) < 1e-6:
        thor_f, abdo_f, belt_fs = pp.filt(thor.data, thor.fs, 0.1, 1.0), pp.filt(abdo.data, abdo.fs, 0.1, 1.0), thor.fs
    else:
        thor_f = abdo_f = belt_fs = None

    spo2 = spo2_bad = spo2_fs = None
    spo2_ch = rec.get("spo2")
    if spo2_ch is not None:
        spo2, spo2_bad = pp.clean_spo2(spo2_ch.data, spo2_ch.fs)
        spo2_fs = spo2_ch.fs

    snore_env = snore_fs = None
    snore_ref = 0.0
    snore_ch = rec.get("snore")
    if snore_ch is not None:
        snore_env = pp.moving_rms(snore_ch.data - np.median(snore_ch.data), int(0.5 * snore_ch.fs))
        snore_fs, snore_ref = snore_ch.fs, float(np.median(snore_env))

    pos_ch = rec.get("position")
    return RespSignals(
        fs=fs, flow=flow, raw_env=raw_env, flow_bad=flow_bad, floor=floor, flow_source=source,
        effort_ratio=eff_ratio, effort_source=eff_source, thor_f=thor_f, abdo_f=abdo_f, belt_fs=belt_fs,
        spo2=spo2, spo2_fs=spo2_fs, spo2_bad=spo2_bad, snore_env=snore_env, snore_fs=snore_fs,
        snore_ref=snore_ref, position=None if pos_ch is None else pos_ch.data, position_fs=None if pos_ch is None else pos_ch.fs,
    )


# ----------------------------------------------------------------------------- scoring

def score_from_signals(sg: RespSignals, params: ScoringParams | None = None,
                       stages: np.ndarray | None = None,
                       arousals: list[tuple[float, float]] | None = None) -> ScoringResult:
    """Apply the rule thresholds in `params` to prepared signals."""
    p = params or ScoringParams()
    fs = sg.fs
    flow_env, baseline, ratio = sg.ratio(p.noise_floor)
    cands = _candidate_runs(ratio, fs, p)
    arousals = arousals or []
    ar_onsets = np.array([a[0] for a in arousals]) if arousals else np.empty(0)
    rejected: dict[str, int] = {}
    apnea_thr = 1.0 - p.apnea_drop
    hyp_thr = 1.0 - p.hypopnea_drop
    sustained_min = _sustained(ratio, fs, p.min_duration)

    events: list[RespEvent] = []
    candidates: list[RespEvent] = []
    for s, e in cands:
        onset, end = s / fs, e / fs
        dur = end - onset
        if dur > p.max_duration:
            rejected["too long (sensor?)"] = rejected.get("too long (sensor?)", 0) + 1
            continue
        seg = ratio[s:e]
        checks: list[Check] = []
        reasons: list[str] = []

        # -- sleep (AASM v2.0 note 3) ----------------------------------------------------------
        sleep = _sleep_check(stages, onset, end, p)
        checks.append(sleep)
        if not sleep.passed:
            reasons.append("occurs entirely during wake")

        # -- signal quality -------------------------------------------------------------------
        bad = float(sg.flow_bad[s:e].mean())
        checks.append(Check("signal", "Usable airflow signal", (1 - bad) * 100, (1 - p.bad_fraction_reject) * 100, "%",
                            bad <= p.bad_fraction_reject, _sig(p.bad_fraction_reject - bad, 0.1)))
        if bad > p.bad_fraction_reject:
            reasons.append("airflow signal unreliable during the event")

        # -- apnea or hypopnea -----------------------------------------------------------------
        deepest = float(np.min(sustained_min[s:e])) if dur >= p.min_duration else 1.0
        if deepest <= apnea_thr:
            kind, flow_drop = "apnea", 1.0 - deepest
            checks.append(Check("apnea", "Airflow drop", flow_drop * 100, p.apnea_drop * 100, "%", True,
                                _sig(flow_drop - p.apnea_drop, 0.03), f"held ≥{p.min_duration:.0f} s"))
        else:
            kind, flow_drop = "hypopnea", float(1.0 - np.median(seg))
            checks.append(Check("hypopnea", "Airflow drop", flow_drop * 100, p.hypopnea_drop * 100, "%",
                                flow_drop >= p.hypopnea_drop, _sig(flow_drop - p.hypopnea_drop, 0.05)))
        checks.append(Check("hypopnea" if kind == "hypopnea" else "apnea", "Duration", dur, p.min_duration, "s",
                            dur >= p.min_duration, _sig(dur - p.min_duration, 1.5)))
        if p.amplitude_fraction:
            frac = float(np.mean(seg <= hyp_thr))
            checks.append(Check("hypopnea", "Share of event below threshold", frac * 100, p.amplitude_fraction * 100, "%",
                                frac >= p.amplitude_fraction, _sig(frac - p.amplitude_fraction, 0.04)))
            if frac < p.amplitude_fraction:
                reasons.append(f"only {frac * 100:.0f}% of the event meets the amplitude criterion")

        # -- desaturation / arousal confirmation ---------------------------------------------
        desat, nadir = _link_desaturation(sg.spo2, sg.spo2_fs, onset, end, p)
        has_arousal = bool(((ar_onsets >= onset - 2.0) & (ar_onsets <= end + 5.0)).any()) if ar_onsets.size else False
        if kind == "hypopnea":
            d_ok = desat is not None and desat >= p.hypopnea_desat
            a_ok = p.hypopnea_arousal and has_arousal
            alone_ok = p.hypopnea_drop_alone is not None and flow_drop >= p.hypopnea_drop_alone
            d_score = _sig(desat - p.hypopnea_desat, 0.5) if desat is not None else 0.0
            score = max(d_score, p.arousal_confidence if a_ok else 0.0,
                        _sig(flow_drop - p.hypopnea_drop_alone, 0.05) if p.hypopnea_drop_alone else 0.0)
            detail = []
            if a_ok:
                detail.append("EEG arousal at event end")
            if alone_ok:
                detail.append(f"drop ≥{p.hypopnea_drop_alone * 100:.0f}% counts alone")
            if not (d_ok or a_ok or alone_ok):
                detail.append("no arousal detected" if p.hypopnea_arousal else "arousals not accepted by this rule")
            checks.append(Check("confirm", "SpO2 desaturation", desat, p.hypopnea_desat, "%", d_ok or a_ok or alone_ok,
                                score, ", ".join(detail)))
            if not (d_ok or a_ok or alone_ok):
                reasons.append(f"SpO2 fell {desat:.1f}% (rule needs ≥{p.hypopnea_desat:g}%)" if desat is not None
                               else "no SpO2 drop measurable")
        elif desat is not None:
            checks.append(Check("apnea", "SpO2 desaturation", desat, None, "%", True, 1.0, "not required for apneas",
                                counts=False))

        # -- classification -------------------------------------------------------------------
        snoring = _snoring(sg.snore_env, sg.snore_fs, onset, end, sg.snore_ref)
        paradox = _paradox(sg.thor_f, sg.abdo_f, sg.belt_fs, onset, end)
        flat = _flattening(sg.flow, fs, onset, end)
        er = sg.effort_ratio[s:e] if sg.effort_ratio is not None else None
        notes: list[str] = []
        if kind == "apnea":
            subtype, effort_drop, type_conf, detail = _classify_apnea(er, p)
            checks.append(Check("effort", "Breathing effort", None if effort_drop is None else (1 - effort_drop) * 100,
                                None, "%", True, type_conf, detail, counts=False))
        else:
            effort_drop = float(1.0 - np.nanmean(er)) if er is not None and len(er) else None
            preserved = effort_drop is not None and effort_drop < flow_drop - p.hyp_effort_margin
            cues = [n for n, f in (("snoring", snoring), ("flow flattening", flat), ("paradox", paradox),
                                   ("effort preserved", preserved)) if f]
            if p.classify_hypopneas:
                subtype = "obstructive" if cues else "central"
                type_conf = (0.85 if len(cues) >= 2 else 0.65) if cues else (0.55 if er is not None else 0.3)
                checks.append(Check("hyp_type", "Obstructive features", float(len(cues)), None, "", True, type_conf,
                                    ", ".join(cues) if cues else "none found → central", counts=False))
            else:
                subtype, type_conf = "obstructive", 0.5
            notes = [c for c in cues if c in ("flow flattening", "effort preserved")]

        position = None
        if sg.position is not None:
            k = int((onset + dur / 2) * sg.position_fs)
            if 0 <= k < len(sg.position):
                position = int(round(float(sg.position[k])))

        confidence = min((c.score for c in checks if c.counts), default=1.0)
        accepted = not reasons
        ev = RespEvent(
            onset=onset, duration=dur, kind=kind, subtype=subtype, flow_drop=flow_drop,
            effort_drop=None if effort_drop is None or np.isnan(effort_drop) else effort_drop,
            desat=None if desat is None else round(desat, 1), desat_nadir=None if nadir is None else round(nadir, 1),
            arousal=has_arousal, snoring=snoring, paradox=paradox,
            stage=STAGE_NAMES.get(_stage_at(stages, onset + dur / 2)) if stages is not None else None,
            position=position, notes=", ".join(notes), confidence=round(confidence, 3),
            type_confidence=round(type_conf, 3), checks=checks, accepted=accepted,
            reject_reason="; ".join(reasons),
        )
        candidates.append(ev)
        if accepted:
            events.append(ev)
        else:
            key = reasons[0] if "wake" in reasons[0] else ("bad flow signal" if "unreliable" in reasons[0]
                                                            else "hypopnea without desaturation/arousal")
            rejected[key] = rejected.get(key, 0) + 1

    return ScoringResult(
        events=events, candidates=candidates, params=p, fs=fs, flow_env=flow_env, flow_baseline=baseline,
        flow_ratio=ratio, flow_bad=sg.flow_bad, effort_ratio=sg.effort_ratio, spo2_clean=sg.spo2, spo2_fs=sg.spo2_fs,
        spo2_bad=sg.spo2_bad, arousals=arousals, flow_source=sg.flow_source, effort_source=sg.effort_source,
        noise_floor=sg.floor if p.noise_floor else 0.0, rejected=rejected, signals=sg, stages=stages,
    )


def score_respiratory(rec: Recording, params: ScoringParams | None = None,
                      stages: np.ndarray | None = None,
                      arousals: list[tuple[float, float]] | None = None) -> ScoringResult:
    """Detect and classify apneas and hypopneas for one recording (prepare + score)."""
    return score_from_signals(prepare_signals(rec, params), params, stages, arousals)


def rescore(result: ScoringResult, params: ScoringParams, stages: np.ndarray | None = None,
            arousals: list[tuple[float, float]] | None = None) -> ScoringResult:
    """Re-run the rule thresholds on an existing result's cached signals (fast: ~1 s)."""
    if result.signals is None:
        raise ValueError("Result has no cached signals; run score_respiratory first")
    return score_from_signals(result.signals, params,
                              result.stages if stages is None else stages,
                              result.arousals if arousals is None else arousals)


def empty_result(params: ScoringParams | None = None, reason: str = "") -> ScoringResult:
    """Result object for recordings where respiratory scoring is impossible."""
    z = np.zeros(1)
    return ScoringResult(events=[], candidates=[], params=params or ScoringParams(), fs=1.0, flow_env=z,
                         flow_baseline=z, flow_ratio=z, flow_bad=np.zeros(1, dtype=bool), effort_ratio=None,
                         spo2_clean=None, spo2_fs=None, spo2_bad=None, flow_source=reason)
