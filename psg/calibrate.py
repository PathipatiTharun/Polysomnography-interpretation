"""Learned event probability: logistic regression on the per-event criteria margins.

The rule-based scorer gives every candidate a *confidence* (the smallest margin by which it
clears the rule thresholds).  This module learns, from technician-scored nights, the
probability that a technician would have scored a candidate, using the same measurements as
features:

    flow_drop, log(duration), desaturation, arousal, sleep fraction, usable-signal fraction,
    effort kept, is_apnea

It is a plain L2-regularised logistic regression fitted with Newton's method (NumPy only).
Validation is leave-one-night-out, and the report compares the learned probability with the
hand-made margin confidence on the same task (AUC, Brier score, calibration by decile).

    python -m psg.calibrate data/ucddb/ucddb002.rec ... --pair test_files/unseen_night_A.edf=test_files/answer_key/ucddb010

writes psg/models/event_lr.json, which the scorer then loads to fill `RespEvent.probability`.
The caveat is important: with a handful of nights from one laboratory, the model learns that
laboratory's scoring habits.  Leave-one-night-out is the honest estimate of how it transfers
to a new night from the same lab, not to a different lab.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from .io import Recording, load_recording, load_ucddb_respevt, load_ucddb_stages
from .respiratory import RespEvent

MODEL_PATH = Path(__file__).resolve().parent / "models" / "event_lr.json"
FEATURES = ["flow_drop", "log_duration", "desat", "arousal", "sleep_frac", "signal_frac", "effort_kept", "is_apnea"]


# ----------------------------------------------------------------------------- features

def event_features(e: RespEvent) -> list[float]:
    """Feature vector in FEATURES order, from the checks the scorer already recorded."""
    by = {c.label: c for c in e.checks}
    sleep = by.get("Sleep")
    sleep_frac = (sleep.value / 100.0) if sleep is not None and sleep.value is not None else 1.0
    sig = by.get("Usable airflow signal")
    signal_frac = (sig.value / 100.0) if sig is not None and sig.value is not None else 1.0
    effort_kept = 1.0 - e.effort_drop if e.effort_drop is not None else 0.5
    return [
        float(e.flow_drop),
        math.log(max(e.duration, 1.0)),
        float(e.desat) if e.desat is not None else 0.0,
        1.0 if e.arousal else 0.0,
        float(sleep_frac),
        float(signal_frac),
        float(min(max(effort_kept, 0.0), 1.5)),
        1.0 if e.kind == "apnea" else 0.0,
    ]


# ----------------------------------------------------------------------------- model

class LogisticModel:
    def __init__(self, mean, std, weights, bias, meta=None):
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)
        self.w, self.b = np.asarray(weights, float), float(bias)
        self.meta = meta or {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def predict_event(self, e: RespEvent) -> float:
        return float(self.predict(np.array([event_features(e)]))[0])

    def to_json(self) -> dict:
        return {"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.w.tolist(), "bias": self.b, **self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "LogisticModel":
        meta = {k: v for k, v in d.items() if k not in ("features", "mean", "std", "weights", "bias")}
        return cls(d["mean"], d["std"], d["weights"], d["bias"], meta)


_MODEL: LogisticModel | None | bool = False   # False = not loaded yet


def load_model(path: Path = MODEL_PATH) -> LogisticModel | None:
    global _MODEL
    if _MODEL is False:
        try:
            _MODEL = LogisticModel.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            _MODEL = None
    return _MODEL


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> LogisticModel:
    """Newton-Raphson on standardised features with an L2 penalty (not on the bias)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    n, d = Zb.shape
    theta = np.zeros(d)
    reg = np.full(d, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Zb @ theta)))
        grad = Zb.T @ (p - y) + reg * theta
        W = p * (1 - p)
        H = (Zb * W[:, None]).T @ Zb + np.diag(reg)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return LogisticModel(mean, std, theta[:-1], theta[-1])


# ----------------------------------------------------------------------------- metrics

def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (ties averaged)."""
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y, float)) ** 2))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 5) -> list[tuple[float, float, float, int]]:
    """(bin low, mean predicted, observed fraction, n) per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            out.append((float(lo), float(p[m].mean()), float(np.mean(y[m])), int(m.sum())))
    return out


# ----------------------------------------------------------------------------- dataset

def _label(cands: list[RespEvent], expert, slack: float = 5.0) -> np.ndarray:
    """1 when a candidate overlaps (within `slack` s) a technician-scored event."""
    ex = sorted(((e.onset, e.end) for e in expert), key=lambda t: t[0])
    starts = np.array([a for a, _ in ex])
    ends = np.array([b for _, b in ex])
    y = np.zeros(len(cands), dtype=float)
    for i, c in enumerate(cands):
        y[i] = float(np.any((starts <= c.end + slack) & (ends >= c.onset - slack)))
    return y


def build_dataset(nights: list[tuple[Recording, list, np.ndarray | None]], progress=None):
    """Run the standard analysis on each night and collect (features, label, night, confidence)."""
    from .pipeline import AnalysisOptions, analyze
    rows, labels, groups, conf = [], [], [], []
    for k, (rec, expert, _stages) in enumerate(nights):
        if progress:
            progress(f"analysing {rec.name} ({k + 1}/{len(nights)})")
        res = analyze(rec, AnalysisOptions())
        cands = res.resp.candidates
        y = _label(cands, expert)
        for c, yy in zip(cands, y):
            rows.append(event_features(c))
            labels.append(yy)
            groups.append(rec.name)
            # Margin confidence as a competing predictor: rejected candidates get their failing margin.
            conf.append(c.confidence if c.accepted else min((ch.score for ch in c.rule_checks), default=0.0))
    return np.array(rows), np.array(labels), np.array(groups), np.array(conf)


def leave_one_night_out(X, y, groups, conf, l2: float = 1.0) -> dict:
    report = {"nights": {}, "pooled": {}}
    p_all = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        model = fit_logistic(X[tr], y[tr], l2)
        p = model.predict(X[te])
        p_all[te] = p
        report["nights"][str(g)] = {
            "n": int(te.sum()), "positives": int(y[te].sum()),
            "auc_lr": auc(p, y[te]), "auc_margin": auc(conf[te], y[te]),
            "brier_lr": brier(p, y[te]), "brier_margin": brier(conf[te], y[te]),
        }
    report["pooled"] = {
        "n": int(len(y)), "positives": int(y.sum()),
        "auc_lr": auc(p_all, y), "auc_margin": auc(conf, y),
        "brier_lr": brier(p_all, y), "brier_margin": brier(conf, y),
        "calibration_lr": calibration_table(p_all, y),
        "calibration_margin": calibration_table(conf, y),
    }
    return report


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="recordings whose expert files sit next to them (UCDDB layout)")
    ap.add_argument("--pair", action="append", default=[], metavar="EDF=ANSWER_PREFIX",
                    help="recording with its answer key elsewhere, e.g. test_files/unseen_night_A.edf=test_files/answer_key/ucddb010")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", default=str(MODEL_PATH))
    args = ap.parse_args(argv)

    nights = []
    for f in args.files:
        rec = load_recording(f)
        if not rec.expert_events:
            print(f"skipping {rec.name}: no expert events next to it")
            continue
        nights.append((rec, rec.expert_events, rec.expert_stages))
    for pair in args.pair:
        f, prefix = pair.split("=", 1)
        rec = load_recording(f)
        ex = load_ucddb_respevt(prefix + "_respevt.txt", rec.start)
        st = load_ucddb_stages(prefix + "_stage.txt")
        nights.append((rec, ex, st))
    if len(nights) < 2:
        print("need at least two scored nights")
        return 1

    X, y, groups, conf = build_dataset(nights, progress=lambda m: print("  " + m, flush=True))
    rep = leave_one_night_out(X, y, groups, conf, args.l2)
    print(f"\nCandidates: {len(y)}  technician-confirmed: {int(y.sum())} ({y.mean() * 100:.0f} %)")
    print(f"{'night':18s} {'n':>5s} {'pos':>5s}  {'AUC lr':>7s} {'AUC margin':>10s}  {'Brier lr':>8s} {'Brier margin':>12s}")
    for g, r in rep["nights"].items():
        print(f"{g:18s} {r['n']:5d} {r['positives']:5d}  {r['auc_lr']:7.3f} {r['auc_margin']:10.3f}  {r['brier_lr']:8.3f} {r['brier_margin']:12.3f}")
    p = rep["pooled"]
    print(f"{'POOLED (LONO)':18s} {p['n']:5d} {p['positives']:5d}  {p['auc_lr']:7.3f} {p['auc_margin']:10.3f}  {p['brier_lr']:8.3f} {p['brier_margin']:12.3f}")
    print("\nCalibration (learned probability): predicted -> observed share confirmed")
    for lo, pm, obs, n in p["calibration_lr"]:
        print(f"  {lo:.1f}-{lo + 0.2:.1f}: predicted {pm:.2f}  observed {obs:.2f}  (n={n})")

    final = fit_logistic(X, y, args.l2)
    final.meta = {
        "trained": dt.datetime.now().isoformat(timespec="seconds"),
        "nights": sorted(set(groups.tolist())), "n_candidates": int(len(y)), "n_positive": int(y.sum()),
        "l2": args.l2, "validation": "leave-one-night-out",
        "auc_lr": p["auc_lr"], "auc_margin": p["auc_margin"], "brier_lr": p["brier_lr"], "brier_margin": p["brier_margin"],
        "per_night": rep["nights"], "calibration_lr": p["calibration_lr"],
        "note": "Trained on technician scoring from one laboratory (UCDDB); probabilities reflect that lab's habits.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final.to_json(), indent=2), encoding="utf-8")
    print("\nWeights (standardised features):")
    for name, w in zip(FEATURES, final.w):
        print(f"  {name:14s} {w:+.3f}")
    print(f"  bias           {final.b:+.3f}")
    print(f"\nModel written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
