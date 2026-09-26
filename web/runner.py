"""Browser entry point (runs inside Pyodide in a Web Worker).

Same processing as a desktop upload: validate -> load -> analyse -> report.  Progress is
reported through the JavaScript function `psgProgress(pct, message)` defined by the worker.
"""
import json
import math
import os
import traceback
from pathlib import Path

from js import psgProgress

from psg.io import load_recording
from psg.pipeline import AnalysisOptions, analyze
from psg.report import write_events_csv, write_html, write_json
from psg.respiratory import ScoringParams
from psg.study import StudyError, validate_channels, validate_file


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, 1)


def run_study(path: str, rule: str) -> str:
    try:
        psgProgress(1, "Checking file")
        validate_file(path)
        psgProgress(4, "Reading signals")
        rec = load_recording(path, load_annotations=False)
        notes = validate_channels(rec)
        psgProgress(10, "Cleaning signals")
        scoring = ScoringParams() if rule == "aasm3" else ScoringParams(hypopnea_desat=4.0, hypopnea_arousal=False)
        res = analyze(rec, AnalysisOptions(scoring=scoring),
                      progress=lambda p, m: psgProgress(10 + int(p * 0.8), m))
        res.warnings[:0] = notes
        psgProgress(92, "Writing report")
        out = Path("/work/out")
        out.mkdir(parents=True, exist_ok=True)
        write_html(res, out / "report.html")
        write_events_csv(res, out / "events.csv")
        write_json(res, out / "summary.json")
        ix = res.summary
        ox = ix.get("spo2") or {}
        payload = {
            "ok": True,
            "name": rec.name,
            "hours": _num(rec.duration / 3600),
            "channels": len(rec.channels),
            "diagnosis": res.diagnosis,
            "warnings": res.warnings,
            "metrics": {
                "ahi": _num(ix.get("ahi")), "oahi": _num(ix.get("oahi")), "cahi": _num(ix.get("cahi")),
                "odi": _num(ix.get("odi")), "nadir": _num(ox.get("nadir")), "t90": _num(ox.get("t90_min")),
                "tst": _num(ix.get("tst_min")), "se": _num(ix.get("sleep_efficiency")),
                "events": ix.get("n_events", 0),
                "oa": ix.get("n_obstructive_apnea", 0), "ca": ix.get("n_central_apnea", 0),
                "ma": ix.get("n_mixed_apnea", 0), "hyp": ix.get("n_hypopnea", 0),
            },
            "runtime": _num(res.runtime_s),
            "html": (out / "report.html").read_text(encoding="utf-8"),
            "csv": (out / "events.csv").read_text(encoding="utf-8"),
            "json": (out / "summary.json").read_text(encoding="utf-8"),
        }
        psgProgress(100, "Done")
        return json.dumps(payload)
    except StudyError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except MemoryError:
        return json.dumps({"ok": False, "error": "The browser ran out of memory for this recording. "
                                                 "Try the desktop app for very large files."})
    except Exception:
        return json.dumps({"ok": False, "error": "Unexpected error while processing the study.",
                           "detail": traceback.format_exc()})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
