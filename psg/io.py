"""Load a PSG recording into canonical channel roles, plus expert annotations when available.

Canonical roles are what the analysis modules ask for ("flow", "spo2", ...), so
the scorer does not care what a particular lab named its channels.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .edf import EdfFile

# Canonical role -> label candidates (exact match tried first, then substring, case-insensitive).
ROLE_CANDIDATES: dict[str, list[str]] = {
    "eeg": ["C3A2", "C3-A2", "C3-M2", "EEG C3-A2", "EEG Fpz-Cz", "C4A1", "C4-A1", "C4-M1", "EEG"],
    "eeg2": ["C4A1", "C4-A1", "C4-M1", "EEG Pz-Oz", "O1A2", "O2A1"],
    "eog_l": ["Lefteye", "EOG(L)", "EOG L", "LOC", "E1", "EOG horizontal", "EOG"],
    "eog_r": ["RightEye", "EOG(R)", "EOG R", "ROC", "E2"],
    "emg": ["EMG", "EMG submental", "Chin", "EMG Chin"],
    "ecg": ["ECG", "EKG"],
    "spo2": ["SpO2", "SaO2", "Sat"],
    "flow": ["Flow", "Airflow", "Nasal", "NasalPres", "Thermistor", "Therm", "Pres"],
    "thorax": ["ribcage", "Thor", "Chest", "RIP Thor", "THO"],
    "abdomen": ["abdo", "Abdomen", "ABD", "RIP Abd"],
    "effort_sum": ["Sum", "RIP Sum"],
    "snore": ["Snore", "Sound", "Mic"],
    "position": ["BodyPos", "Position", "Pos"],
    "pulse": ["Pulse", "HR", "Heart rate"],
}

# Canonical stage codes used throughout the project.
STAGE_W, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R, STAGE_UNK = 0, 1, 2, 3, 4, -1
STAGE_NAMES = {STAGE_W: "W", STAGE_N1: "N1", STAGE_N2: "N2", STAGE_N3: "N3", STAGE_R: "R", STAGE_UNK: "?"}
EPOCH_S = 30.0

# UCDDB stage file codes: 0 W, 1 REM, 2 S1, 3 S2, 4 S3, 5 S4, 6 artifact, 7 indeterminate.
_UCDDB_STAGE_MAP = {0: STAGE_W, 1: STAGE_R, 2: STAGE_N1, 3: STAGE_N2, 4: STAGE_N3, 5: STAGE_N3,
                    6: STAGE_UNK, 7: STAGE_UNK}


@dataclass
class Channel:
    label: str
    role: str | None
    data: np.ndarray  # physical units, float32
    fs: float
    unit: str = ""

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs

    def times(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.fs


@dataclass
class ExpertEvent:
    """A respiratory event from a lab's annotation file (ground truth for evaluation)."""
    onset: float
    duration: float
    kind: str        # 'apnea' | 'hypopnea'
    subtype: str     # 'obstructive' | 'central' | 'mixed' | 'unknown'
    desat_drop: float | None = None
    desat_low: float | None = None
    arousal: bool | None = None
    snore: bool | None = None

    @property
    def end(self) -> float:
        return self.onset + self.duration


@dataclass
class Recording:
    path: Path
    start: dt.datetime | None
    duration: float
    channels: dict[str, Channel]                 # by original label
    roles: dict[str, str] = field(default_factory=dict)  # role -> label
    expert_stages: np.ndarray | None = None      # canonical codes per 30-s epoch
    expert_events: list[ExpertEvent] | None = None
    patient: str = ""

    def get(self, role: str) -> Channel | None:
        label = self.roles.get(role)
        return self.channels[label] if label else None

    def has(self, *roles: str) -> bool:
        return all(r in self.roles for r in roles)

    @property
    def n_epochs(self) -> int:
        return int(self.duration // EPOCH_S)

    @property
    def name(self) -> str:
        return self.path.stem


def _assign_roles(labels: list[str]) -> dict[str, str]:
    low = {l.lower(): l for l in labels}
    roles: dict[str, str] = {}
    used: set[str] = set()
    for role, cands in ROLE_CANDIDATES.items():
        found = None
        for c in cands:  # exact first
            if c.lower() in low and low[c.lower()] not in used:
                found = low[c.lower()]
                break
        if found is None:
            for c in cands:  # then substring
                for l in labels:
                    if c.lower() in l.lower() and l not in used:
                        found = l
                        break
                if found:
                    break
        if found:
            roles[role] = found
            used.add(found)
    return roles


def load_recording(path: str | Path, load_annotations: bool = True) -> Recording:
    path = Path(path)
    edf = EdfFile(path)
    hdr = edf.header
    channels: dict[str, Channel] = {}
    for i, sig in enumerate(hdr.signals):
        if sig.label.lower().startswith("edf annotations"):
            continue
        channels[sig.label] = Channel(sig.label, None, edf.read(i), edf.sfreq(i), sig.unit)
    roles = _assign_roles(list(channels))
    for role, label in roles.items():
        channels[label].role = role
    rec = Recording(path, hdr.start, hdr.duration, channels, roles, patient=hdr.patient)

    if load_annotations:
        stage_file = path.with_name(path.stem + "_stage.txt")
        resp_file = path.with_name(path.stem + "_respevt.txt")
        if stage_file.exists():
            rec.expert_stages = load_ucddb_stages(stage_file)
        if resp_file.exists() and hdr.start is not None:
            rec.expert_events = load_ucddb_respevt(resp_file, hdr.start)
        hyp = find_sleepedf_hypnogram(path)
        if hyp is not None and rec.expert_stages is None:
            rec.expert_stages = load_edfplus_hypnogram(hyp, rec.n_epochs)
    return rec


# ----------------------------------------------------------------------------- Sleep-EDF / EDF+ hypnograms

_EDFPLUS_STAGE = {"sleep stage w": STAGE_W, "sleep stage 1": STAGE_N1, "sleep stage 2": STAGE_N2,
                  "sleep stage 3": STAGE_N3, "sleep stage 4": STAGE_N3, "sleep stage r": STAGE_R,
                  "sleep stage n1": STAGE_N1, "sleep stage n2": STAGE_N2, "sleep stage n3": STAGE_N3}


def find_sleepedf_hypnogram(psg_path: Path) -> Path | None:
    """Sleep-EDF pairs 'SC4001E0-PSG.edf' with 'SC4001EC-Hypnogram.edf' (scorer letter differs)."""
    stem = psg_path.stem
    if not stem.endswith("-PSG"):
        return None
    prefix = stem[:6]
    for cand in sorted(psg_path.parent.glob(f"{prefix}*-Hypnogram.edf")):
        return cand
    return None


def load_edfplus_hypnogram(path: str | Path, n_epochs: int) -> np.ndarray:
    ann = EdfFile(path).annotations()
    stages = np.full(n_epochs, STAGE_UNK, dtype=int)
    for onset, dur, text in ann:
        code = _EDFPLUS_STAGE.get(text.lower())
        if code is None:
            continue
        k0 = int(round(onset / EPOCH_S))
        k1 = int(round((onset + dur) / EPOCH_S))
        stages[max(k0, 0):min(k1, n_epochs)] = code
    return stages


def load_ucddb_stages(path: str | Path) -> np.ndarray:
    codes = [int(l.strip()) for l in Path(path).read_text().splitlines() if l.strip()]
    return np.array([_UCDDB_STAGE_MAP.get(c, STAGE_UNK) for c in codes], dtype=int)


_RESP_LINE = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+(APNEA|HYP)-([OCMB])\b")
_KIND = {"APNEA": "apnea", "HYP": "hypopnea"}
_SUB = {"O": "obstructive", "C": "central", "M": "mixed", "B": "unknown"}


def _float_or_none(s: str) -> float | None:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def load_ucddb_respevt(path: str | Path, rec_start: dt.datetime) -> list[ExpertEvent]:
    """Parse the fixed-width UCDDB respiratory event list. Times are wall-clock, so they
    are converted to seconds from the recording start (wrapping past midnight)."""
    events: list[ExpertEvent] = []
    start_s = rec_start.hour * 3600 + rec_start.minute * 60 + rec_start.second
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _RESP_LINE.match(line)
        if not m:
            continue
        hh, mm, ss = (int(x) for x in m.group(1).split(":"))
        onset = (hh * 3600 + mm * 60 + ss - start_s) % 86400
        # Columns after the type: optional PB/CS flag, duration, desat low, %drop, snore, arousal.
        rest = line[m.end():]
        tokens = rest.split()
        if tokens and tokens[0] in ("PB", "CS"):
            tokens = tokens[1:]
        duration = float(tokens[0]) if tokens else 0.0
        low = _float_or_none(line[35:44])
        drop = _float_or_none(line[44:51])
        snore_c, arousal_c = line[51:56].strip(), line[56:62].strip()
        events.append(ExpertEvent(
            onset=float(onset), duration=duration, kind=_KIND[m.group(2)], subtype=_SUB[m.group(3)],
            desat_drop=drop, desat_low=low,
            snore=(snore_c == "+") if snore_c in ("+", "-") else None,
            arousal=(arousal_c == "+") if arousal_c in ("+", "-") else None,
        ))
    return events
