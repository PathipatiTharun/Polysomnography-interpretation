"""Minimal EDF / EDF+ reader (pure NumPy).

Reads the header and per-signal data of European Data Format files. Works on
files with a non-standard extension such as PhysioNet's ``.rec``. Each signal
can have its own sampling rate, which is the normal case for PSG recordings.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER_BYTES = 256  # fixed part; per-signal part is another 256 bytes each


@dataclass
class EdfSignal:
    index: int
    label: str
    transducer: str
    unit: str
    phys_min: float
    phys_max: float
    dig_min: int
    dig_max: int
    prefilter: str
    n_samples_per_record: int

    @property
    def gain(self) -> float:
        span = self.dig_max - self.dig_min
        return (self.phys_max - self.phys_min) / span if span else 1.0

    @property
    def offset(self) -> float:
        return self.phys_min - self.dig_min * self.gain


@dataclass
class EdfHeader:
    path: Path
    version: str
    patient: str
    recording: str
    start: dt.datetime | None
    header_bytes: int
    reserved: str
    n_records: int
    record_duration: float
    signals: list[EdfSignal] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.n_records * self.record_duration

    def sfreq(self, idx: int) -> float:
        return self.signals[idx].n_samples_per_record / self.record_duration

    def labels(self) -> list[str]:
        return [s.label for s in self.signals]


def _ascii(b: bytes) -> str:
    return b.decode("ascii", errors="replace").strip()


def _parse_start(date_s: str, time_s: str) -> dt.datetime | None:
    try:
        d, m, y = (int(x) for x in date_s.split("."))
        hh, mm, ss = (int(x) for x in time_s.split("."))
        y += 2000 if y < 85 else 1900  # EDF spec: 85-99 -> 1985-1999
        return dt.datetime(y, m, d, hh, mm, ss)
    except ValueError:
        return None


def read_header(path: str | Path) -> EdfHeader:
    path = Path(path)
    with open(path, "rb") as f:
        h = f.read(HEADER_BYTES)
        if len(h) < HEADER_BYTES:
            raise ValueError(f"{path}: file too short to be EDF")
        version = _ascii(h[0:8])
        patient = _ascii(h[8:88])
        recording = _ascii(h[88:168])
        start = _parse_start(_ascii(h[168:176]), _ascii(h[176:184]))
        header_bytes = int(_ascii(h[184:192]))
        reserved = _ascii(h[192:236])
        n_records = int(_ascii(h[236:244]))
        record_duration = float(_ascii(h[244:252]))
        ns = int(_ascii(h[252:256]))

        sig = f.read(ns * 256)
        # Fields are stored "all labels, then all transducers, ..." not per signal.
        def col(offset: int, width: int) -> list[str]:
            base = offset * ns
            return [_ascii(sig[base + i * width: base + (i + 1) * width]) for i in range(ns)]

        labels = col(0, 16)
        transducers = col(16, 80)
        units = col(96, 8)
        pmins = col(104, 8)
        pmaxs = col(112, 8)
        dmins = col(120, 8)
        dmaxs = col(128, 8)
        prefilters = col(136, 80)
        nsamps = col(216, 8)

    signals = []
    for i in range(ns):
        signals.append(EdfSignal(
            index=i, label=labels[i], transducer=transducers[i], unit=units[i],
            phys_min=float(pmins[i]), phys_max=float(pmaxs[i]),
            dig_min=int(float(dmins[i])), dig_max=int(float(dmaxs[i])),
            prefilter=prefilters[i], n_samples_per_record=int(nsamps[i]),
        ))

    if n_records < 0:  # unknown length: derive from file size
        rec_bytes = 2 * sum(s.n_samples_per_record for s in signals)
        n_records = (path.stat().st_size - header_bytes) // rec_bytes

    return EdfHeader(path, version, patient, recording, start, header_bytes,
                     reserved, n_records, record_duration, signals)


class EdfFile:
    """Lazy accessor over an EDF file. Signals are converted to physical units."""

    def __init__(self, path: str | Path):
        self.header = read_header(path)
        samples_per_record = sum(s.n_samples_per_record for s in self.header.signals)
        # Guard against truncated files: never map past the end.
        available = (self.header.path.stat().st_size - self.header.header_bytes) // (2 * samples_per_record)
        if available < self.header.n_records:
            self.header.n_records = int(available)
        shape = (self.header.n_records, samples_per_record)
        try:
            self._mm = np.memmap(self.header.path, dtype="<i2", mode="r",
                                 offset=self.header.header_bytes, shape=shape)
        except (OSError, ValueError, NotImplementedError):
            # Environments without mmap (e.g. the browser build under Pyodide): read into memory.
            with open(self.header.path, "rb") as f:
                f.seek(self.header.header_bytes)
                buf = np.fromfile(f, dtype="<i2", count=shape[0] * shape[1])
            self._mm = buf.reshape(shape)
        starts = np.cumsum([0] + [s.n_samples_per_record for s in self.header.signals])
        self._slices = [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(self.header.signals))]

    @property
    def labels(self) -> list[str]:
        return self.header.labels()

    def sfreq(self, idx: int) -> float:
        return self.header.sfreq(idx)

    def find(self, *candidates: str) -> int | None:
        """Return the index of the first signal whose label matches any candidate (case-insensitive substring)."""
        low = [l.lower() for l in self.labels]
        for cand in candidates:
            c = cand.lower()
            for i, l in enumerate(low):
                if l == c:
                    return i
        for cand in candidates:
            c = cand.lower()
            for i, l in enumerate(low):
                if c in l:
                    return i
        return None

    def read(self, idx: int) -> np.ndarray:
        """Whole signal as float32 in physical units."""
        s = self.header.signals[idx]
        dig = np.asarray(self._mm[:, self._slices[idx]]).reshape(-1).astype(np.float32)
        return dig * np.float32(s.gain) + np.float32(s.offset)

    def read_all(self) -> dict[str, tuple[np.ndarray, float]]:
        return {s.label: (self.read(i), self.sfreq(i)) for i, s in enumerate(self.header.signals)}

    def annotations(self) -> list[tuple[float, float, str]]:
        """EDF+ annotations as (onset_s, duration_s, text).

        EDF+ stores them in a pseudo-signal named 'EDF Annotations' as Time-stamped
        Annotation Lists: '+onset[\\x15duration]\\x14text\\x14...\\x00'.
        """
        out: list[tuple[float, float, str]] = []
        for i, s in enumerate(self.header.signals):
            if not s.label.lower().startswith("edf annotations"):
                continue
            raw = np.asarray(self._mm[:, self._slices[i]]).astype("<i2").tobytes()
            for tal in raw.split(b"\x00"):
                if not tal or tal[:1] not in (b"+", b"-"):
                    continue
                parts = tal.split(b"\x14")
                head = parts[0].split(b"\x15")
                try:
                    onset = float(head[0])
                    dur = float(head[1]) if len(head) > 1 and head[1] else 0.0
                except ValueError:
                    continue
                for txt in parts[1:]:
                    t = txt.decode("latin-1").strip()
                    if t:
                        out.append((onset, dur, t))
        return out
