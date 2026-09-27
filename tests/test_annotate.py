"""EDF+ annotation writer round trip and the spindle / slow-wave / eye-movement detectors on
synthetic signals.  Run: python tests/test_annotate.py"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psg.annotate import detect_rems, detect_slow_waves, detect_spindles  # noqa: E402
from psg.edf import EdfFile  # noqa: E402
from psg.edfplus import write_annotations_edf  # noqa: E402
from psg.io import STAGE_N2, STAGE_N3, STAGE_R  # noqa: E402

FS = 100.0


def test_edfplus_roundtrip():
    anns = [(0.0, 30630.0, "Sleep stage W"), (30630.0, 120.0, "Sleep stage N1"), (31000.25, 0.8, "Spindle"),
            (40000.0, 0.0, "Lights off (estimated)")]
    with tempfile.TemporaryDirectory() as d:
        p = write_annotations_edf(Path(d) / "a.edf", anns, dt.datetime(1989, 4, 24, 16, 13), "X F X Test")
        back = EdfFile(p).annotations()
        hdr = EdfFile(p).header
    assert hdr.reserved.startswith("EDF+C") and hdr.labels() == ["EDF Annotations"]
    assert hdr.start == dt.datetime(1989, 4, 24, 16, 13)
    assert [(round(a, 2), round(b, 2), t) for a, b, t in back] == [(round(a, 2), round(b, 2), t) for a, b, t in anns]


def test_detectors_on_synthetic_eeg():
    rng = np.random.default_rng(0)
    n_ep = 20
    t = np.arange(int(n_ep * 30 * FS)) / FS
    eeg = 8 * rng.standard_normal(len(t))
    stages = np.full(n_ep, STAGE_N2)
    stages[10:15] = STAGE_N3
    stages[15:] = STAGE_R
    # Spindles: 1-s 13 Hz bursts, 40 uV, at known times in N2.
    spindle_at = [35.0, 95.0, 155.0, 215.0]
    for s in spindle_at:
        m = (t >= s) & (t < s + 1.0)
        eeg[m] += 40 * np.sin(2 * np.pi * 13 * t[m]) * np.hanning(m.sum())
    # Slow waves: 1-Hz 150 uV p2p waves through N3 epochs.
    n3 = (t >= 300) & (t < 450)
    eeg[n3] += 75 * np.sin(2 * np.pi * 1.0 * t[n3])
    sp = detect_spindles(eeg, FS, stages)
    found = [a.onset for a in sp]
    assert all(any(abs(f - s) < 0.6 for f in found) for s in spindle_at), found
    assert all(11 <= a.detail["freq_hz"] <= 16 for a in sp)
    sw = detect_slow_waves(eeg, FS, stages)
    assert 100 <= len(sw) <= 160 and all(300 <= a.onset < 451 for a in sw)
    # Eye movements: steep 150 uV steps in REM only.
    eog = 5 * rng.standard_normal(len(t))
    for s in (460.0, 470.0, 480.0, 500.0):
        eog[t >= s] += 150 * (1 if int(s) % 20 == 0 else -1)
    rems = detect_rems(eog, FS, stages)
    assert len(rems) >= 3 and all(a.onset >= 450 for a in rems)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
