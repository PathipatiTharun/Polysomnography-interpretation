"""Headless batch analysis and validation.

    python analyze.py data/ucddb/ucddb002.rec                 # analyse one night
    python analyze.py data/ucddb/*.rec --out reports          # several nights, write reports
    python analyze.py data/ucddb/*.rec --expert-stages        # score events on the expert hypnogram

For every recording that ships with expert annotations (UCDDB *_respevt.txt / *_stage.txt,
Sleep-EDF hypnograms) the automatic result is compared with the expert scoring.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from psg.evaluate import match_events, staging_agreement
from psg.io import load_recording
from psg.pipeline import AnalysisOptions, ahi_severity, analyze
from psg.rules import DEFAULT_RULE, RULES, get_rule
from psg.study import StudyError, process_study


def validation_summary(res) -> dict:
    rec = res.rec
    out: dict = {}
    if rec.expert_stages is not None and res.stage_source.startswith("automatic"):
        agr = staging_agreement(res.stages, rec.expert_stages)
        out["Staging accuracy / Cohen's kappa"] = f"{agr['accuracy'] * 100:.0f} % / {agr['kappa']:.2f} ({agr['n']} epochs)"
    if rec.expert_events is not None and rec.expert_stages is not None:
        tst_h = np.isin(rec.expert_stages, [1, 2, 3, 4]).sum() * 30 / 3600
        ex_ahi = len(rec.expert_events) / tst_h if tst_h else float("nan")
        m = match_events(res.events, rec.expert_events)
        out["Expert AHI / automatic AHI"] = f"{ex_ahi:.1f} ({ahi_severity(ex_ahi)}) / {res.summary['ahi']:.1f} ({res.diagnosis['severity']})"
        out["Event sensitivity / precision / F1"] = f"{m.sensitivity * 100:.0f} % / {m.precision * 100:.0f} % / {m.f1:.2f}"
        out["Apnea-vs-hypopnea agreement"] = f"{m.kind_agreement() * 100:.0f} %"
        out["Obstructive/central/mixed agreement"] = f"{m.subtype_agreement() * 100:.0f} %"
        out["_match"] = m
        out["_expert_ahi"] = ex_ahi
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="reports", help="output folder for HTML/CSV/JSON")
    ap.add_argument("--expert-stages", action="store_true", help="use the expert hypnogram when available")
    ap.add_argument("--rule", choices=list(RULES) + ["aasm3", "aasm4"], default=DEFAULT_RULE,
                    help="scoring rule version (see docs/rule_versions.md); aasm3/aasm4 are shorthands for "
                         "aasm2012_rec / aasm2013_acc")
    ap.add_argument("--min-confidence", type=float, default=0.0, metavar="PCT",
                    help="only count events with at least this confidence (0-100)")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    files = [f for pat in args.files for f in (glob.glob(pat) or [pat])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rule_id = {"aasm3": "aasm2012_rec", "aasm4": "aasm2013_acc"}.get(args.rule, args.rule)
    scoring = get_rule(rule_id).params()
    opts = AnalysisOptions(staging_source="expert" if args.expert_stages else "auto", scoring=scoring,
                           min_confidence=args.min_confidence / 100.0)
    print(f"Scoring rule: {get_rule(rule_id).name}" + (f"; confidence ≥ {args.min_confidence:.0f} %" if args.min_confidence else ""))

    rows = []
    for f in files:
        try:
            if args.no_report:
                rec = load_recording(f)
                res = analyze(rec, opts)
            else:  # same path as a doctor's upload in the app: report saved + study index updated
                study = process_study(f, opts, out, validation_fn=validation_summary)
                rec, res = study.rec, study.result
        except StudyError as e:
            print(f"\n=== {Path(f).name}: skipped - {e}")
            continue
        val = validation_summary(res)
        ix = res.summary
        print(f"\n=== {rec.name}  ({rec.duration / 3600:.1f} h, {len(rec.channels)} channels, staging: {res.stage_source})")
        print(f"  {res.diagnosis['primary']}")
        print(f"  AHI {ix['ahi']:.1f}/h | OA {ix['n_obstructive_apnea']} CA {ix['n_central_apnea']} MA {ix['n_mixed_apnea']} "
              f"| OH {ix['n_obstructive_hypopnea']} CH {ix['n_central_hypopnea']} | ODI {ix['odi']:.1f}/h "
              f"| TST {ix['tst_min']:.0f} min SE {ix['sleep_efficiency']:.0f}% | {res.runtime_s:.1f}s")
        for k, v in val.items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        for w in res.warnings:
            print(f"  ! {w}")
        rows.append((rec.name, res, val))

    scored = [(n, r, v) for n, r, v in rows if "_match" in v]
    if len(scored) > 1:
        ms = [v["_match"] for _, _, v in scored]
        tp = sum(len(m.matched) for m in ms)
        ne = sum(m.n_expert for m in ms)
        nd = sum(m.n_detected for m in ms)
        sev_ok = sum(ahi_severity(v["_expert_ahi"]) == r.diagnosis["severity"] for _, r, v in scored)
        print(f"\nPOOLED over {len(scored)} nights: sensitivity {tp / ne * 100:.0f} %, precision {tp / nd * 100:.0f} %, "
              f"severity class agreement {sev_ok}/{len(scored)}")
        ea = [v["_expert_ahi"] for _, _, v in scored]
        aa = [r.summary["ahi"] for _, r, _ in scored]
        if len(scored) > 2:
            print(f"  AHI correlation r = {np.corrcoef(ea, aa)[0, 1]:.3f}")
    if not args.no_report:
        print(f"\nReports written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
