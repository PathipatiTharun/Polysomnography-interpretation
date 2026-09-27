"""One-call study processing: validate an uploaded recording, analyse it, save the report,
and keep an index of processed studies.  Used by the desktop app and the CLI."""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .edf import read_header
from .io import Recording, load_recording
from .pipeline import AnalysisOptions, AnalysisResult, analyze
from .report import write_events_csv, write_html, write_json

ACCEPTED_SUFFIXES = {".edf", ".rec", ".bdf"}
INDEX_NAME = "studies.json"
MIN_DURATION_S = 10 * 60


class StudyError(Exception):
    """A problem with the uploaded file, worded for the clinician."""


@dataclass
class Study:
    rec: Recording
    result: AnalysisResult
    folder: Path

    @property
    def report_path(self) -> Path:
        return self.folder / f"{self.rec.name}_report.html"


def validate_file(path: str | Path) -> None:
    """Fail early, in plain language, before spending time on analysis."""
    p = Path(path)
    if not p.exists():
        raise StudyError(f"File not found: {p}")
    if p.suffix.lower() not in ACCEPTED_SUFFIXES:
        raise StudyError(f"'{p.name}' is not a PSG recording. Please upload an EDF file "
                         f"(.edf, .rec or .bdf) exported from the sleep system.")
    if p.name.lower().endswith("-hypnogram.edf"):
        raise StudyError(f"'{p.name}' is a hypnogram (scoring) file, not the recording itself. "
                         f"Please upload the matching PSG file.")
    try:
        hdr = read_header(p)
    except Exception as e:
        raise StudyError(f"'{p.name}' could not be read as EDF ({e}). The file may be damaged or "
                         f"in a proprietary format - export it as EDF from the acquisition software.")
    if hdr.duration < MIN_DURATION_S:
        raise StudyError(f"'{p.name}' is only {hdr.duration / 60:.0f} minutes long - too short for a sleep study.")


def validate_channels(rec: Recording) -> list[str]:
    """Return clinician-facing notes about missing sensors; raise if nothing can be analysed."""
    if not any(rec.has(r) for r in ("eeg", "flow", "thorax", "abdomen", "effort_sum", "spo2")):
        raise StudyError("No EEG, airflow, effort or SpO2 channel was recognised in this file. "
                         f"Channels found: {', '.join(rec.channels) or 'none'}.")
    notes = []
    if not rec.has("eeg"):
        notes.append("No EEG: sleep cannot be staged, so indices use recording time instead of sleep time.")
    if not rec.has("flow"):
        notes.append("No airflow channel: breathing effort is used as a surrogate for airflow.")
    if not rec.has("spo2"):
        notes.append("No SpO2: hypopneas can only be confirmed by arousals; ODI not available.")
    if not (rec.has("thorax") or rec.has("abdomen") or rec.has("effort_sum")):
        notes.append("No effort belts: apneas cannot be classified as obstructive / central / mixed.")
    return notes


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "study"


def save_reports(result: AnalysisResult, out_root: str | Path, validation: dict | None = None) -> Path:
    rec = result.rec
    folder = Path(out_root) / _safe(rec.name)
    folder.mkdir(parents=True, exist_ok=True)
    write_html(result, folder / f"{rec.name}_report.html", validation)
    write_events_csv(result, folder / f"{rec.name}_events.csv")
    write_json(result, folder / f"{rec.name}_summary.json")
    return folder


def _update_index(out_root: Path, study: Study) -> None:
    idx_path = out_root / INDEX_NAME
    entries = list_studies(out_root)
    r, ix = study.result, study.result.summary
    ahi = ix.get("ahi")
    entry = {
        "name": study.rec.name,
        "source": str(study.rec.path.resolve()),
        "report": str(study.report_path.resolve()),
        "processed": dt.datetime.now().isoformat(timespec="seconds"),
        "recorded": str(study.rec.start) if study.rec.start else "",
        "diagnosis": r.diagnosis["primary"],
        "severity": r.diagnosis["severity"],
        "ahi": None if ahi is None or not np.isfinite(ahi) else round(float(ahi), 1),
        "rule": r.rule_name,
        "min_confidence": r.options.min_confidence,
    }
    entries = [e for e in entries if e.get("source") != entry["source"]]
    entries.insert(0, entry)
    idx_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def list_studies(out_root: str | Path) -> list[dict]:
    p = Path(out_root) / INDEX_NAME
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def process_study(path: str | Path, options: AnalysisOptions | None = None,
                  out_root: str | Path = "reports",
                  progress: Callable[[int, str], None] | None = None,
                  validation_fn: Callable[[AnalysisResult], dict] | None = None) -> Study:
    """Upload -> validated, analysed, reported.  Progress: 0-10 load, 10-90 analysis, 90-100 report."""
    say = progress or (lambda pct, msg: None)
    say(1, "Checking file")
    validate_file(path)
    say(4, "Reading signals")
    rec = load_recording(path)
    notes = validate_channels(rec)
    say(10, "Cleaning signals")
    result = analyze(rec, options, progress=lambda pct, msg: say(10 + int(pct * 0.8), msg))
    result.warnings[:0] = notes
    say(92, "Writing report")
    val = None
    if validation_fn is not None:
        val = {k: v for k, v in validation_fn(result).items() if not k.startswith("_")}
    folder = save_reports(result, out_root, val)
    study = Study(rec, result, folder)
    _update_index(Path(out_root), study)
    say(100, "Done")
    return study
