"""Synthetic-signal checks of the AASM rules: known apneas/hypopneas are inserted into clean
breathing and must be found and classified correctly.  Run:  python -m pytest tests -q
(or simply  python tests/test_scoring.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psg.io import Channel, Recording, STAGE_N2  # noqa: E402
from psg.respiratory import ScoringParams, score_respiratory  # noqa: E402
from psg.spo2 import detect_desaturations  # noqa: E402
from psg import preprocess as pp  # noqa: E402

FS = 16.0
DUR = 1800.0  # 30 min
# (onset, duration, kind): kind in OA (no flow, effort continues), CA (no flow, no effort),
# MA (no effort first half, effort second half), H (flow -50 %, effort -50 %, with desaturation)
EVENTS = [(300, 20, "OA"), (600, 25, "CA"), (900, 30, "MA"), (1200, 20, "H"), (1500, 18, "H")]


def _synthetic() -> Recording:
    rng = np.random.default_rng(0)
    t = np.arange(int(DUR * FS)) / FS
    breath = np.sin(2 * np.pi * 0.25 * t)            # 15 breaths/min
    flow_amp = np.ones_like(t)
    thor_amp = np.ones_like(t)
    abdo_amp = np.ones_like(t)
    spo2 = np.full(int(DUR), 96.0)
    for on, dur, kind in EVENTS:
        m = (t >= on) & (t < on + dur)
        if kind in ("OA", "CA", "MA"):
            flow_amp[m] = 0.02
        if kind == "CA":
            thor_amp[m] = abdo_amp[m] = 0.03
        if kind == "MA":
            first = (t >= on) & (t < on + dur / 2)
            thor_amp[first] = abdo_amp[first] = 0.03
        if kind == "H":
            flow_amp[m] = 0.45
            thor_amp[m] = abdo_amp[m] = 0.5
        # Desaturation reaching its nadir ~15 s after the event ends, then recovery.
        k = np.arange(int(on + 5), int(on + dur + 15))
        spo2[k] = 96 - 5 * (k - k[0]) / len(k)
        r = np.arange(int(on + dur + 15), int(on + dur + 30))
        spo2[r] = 91 + 5 * (r - r[0]) / len(r)
    noise = lambda: 0.02 * rng.standard_normal(len(t))
    ch = {
        "Flow": Channel("Flow", "flow", (flow_amp * breath + noise()).astype(np.float32), FS),
        "Thor": Channel("Thor", "thorax", (thor_amp * breath + noise()).astype(np.float32), FS),
        "Abdo": Channel("Abdo", "abdomen", (abdo_amp * breath + noise()).astype(np.float32), FS),
        "SpO2": Channel("SpO2", "spo2", np.repeat(spo2, 1).astype(np.float32), 1.0),
    }
    roles = {c.role: c.label for c in ch.values()}
    return Recording(Path("synthetic.edf"), None, DUR, ch, roles)


def test_detects_and_classifies_events():
    rec = _synthetic()
    stages = np.full(int(DUR // 30), STAGE_N2)
    res = score_respiratory(rec, ScoringParams(), stages=stages)
    found = {(round(e.onset / 100) * 100): e for e in res.events}
    expected = {"OA": ("apnea", "obstructive"), "CA": ("apnea", "central"),
                "MA": ("apnea", "mixed"), "H": ("hypopnea", None)}
    assert len(res.events) == len(EVENTS), [(e.onset, e.code) for e in res.events]
    for on, dur, kind in EVENTS:
        e = min(res.events, key=lambda x: abs(x.onset - on))
        assert abs(e.onset - on) < 5, (kind, e.onset)
        assert abs(e.duration - dur) < 6, (kind, e.duration)
        k, sub = expected[kind]
        assert e.kind == k, (kind, e.kind)
        if sub:
            assert e.subtype == sub, (kind, e.subtype)
        assert e.desat is not None and e.desat >= 3


def test_hypopnea_requires_desaturation():
    rec = _synthetic()
    rec.channels["SpO2"].data[:] = 96.0          # no desaturations at all
    res = score_respiratory(rec, ScoringParams(hypopnea_arousal=False))
    assert all(e.kind == "apnea" for e in res.events)
    assert len(res.events) == 3


def test_spo2_cleaning_and_desaturation():
    x = np.full(600, 95.0)
    x[100:103] = 0.0                  # probe dropout -> must be removed, not scored
    x[300:330] = np.linspace(95, 89, 30)
    x[330:360] = np.linspace(89, 95, 30)
    clean, bad = pp.clean_spo2(x, 1.0)
    d = detect_desaturations(clean, 1.0, drop=3)
    assert len(d) == 1 and 5.5 <= d[0].drop <= 6.5
    assert np.nanmin(clean[95:110]) > 90  # dropout interpolated


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
