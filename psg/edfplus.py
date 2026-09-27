"""Write EDF+ annotation files (the format of Sleep-EDF "*-Hypnogram.edf" files).

The file holds a single "EDF Annotations" signal in one data record of duration 0. Each
annotation is a Time-stamped Annotation List (TAL): "+onset\x15duration\x14text\x14\x00". The
first TAL in the record is the mandatory time-keeping TAL "+0\x14\x14\x00". Any EDF+ viewer
(EDFbrowser, MNE, Polyman...) can open it next to the recording.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path


def _field(text: str, width: int) -> bytes:
    b = text.encode("ascii", errors="replace")[:width]
    return b + b" " * (width - len(b))


def _num(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def write_annotations_edf(path: str | Path, annotations: list[tuple[float, float, str]],
                          start: dt.datetime | None, patient: str = "X X X X",
                          recording: str | None = None) -> Path:
    """annotations: (onset_s, duration_s, text), onset relative to the recording start."""
    start = start or dt.datetime(1985, 1, 1)
    if recording is None:
        recording = f"Startdate {start.strftime('%d-%b-%Y').upper()} X X X"
    tals = [b"+0\x14\x14\x00"]
    for onset, dur, text in sorted(annotations, key=lambda a: a[0]):
        t = str(text).replace("\x14", " ").replace("\x15", " ").replace("\x00", " ")
        dur_part = b"\x15" + _num(dur).encode() if dur and dur > 0 else b""
        tals.append(b"+" + _num(onset).encode() + dur_part + b"\x14" + t.encode("latin-1", errors="replace") + b"\x14\x00")
    data = b"".join(tals)
    n_samples = math.ceil(len(data) / 2)
    data += b"\x00" * (2 * n_samples - len(data))

    ns = 1
    hdr = b"".join([
        _field("0", 8), _field(patient, 80), _field(recording, 80),
        _field(start.strftime("%d.%m.%y"), 8), _field(start.strftime("%H.%M.%S"), 8),
        _field(str(256 + 256 * ns), 8), _field("EDF+C", 44), _field("1", 8), _field("0", 8), _field(str(ns), 4),
    ])
    sig = b"".join([
        _field("EDF Annotations", 16), _field("", 80), _field("", 8),
        _field("-1", 8), _field("1", 8), _field("-32768", 8), _field("32767", 8),
        _field("", 80), _field(str(n_samples), 8), _field("", 32),
    ])
    path = Path(path)
    path.write_bytes(hdr + sig + data)
    return path
