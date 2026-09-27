"""Rule-version library, confidence and fast re-scoring.  Run: python tests/test_rules.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scoring import _synthetic, EVENTS, DUR  # noqa: E402
from psg.io import STAGE_N2, STAGE_W  # noqa: E402
from psg.respiratory import ScoringParams, rescore, score_respiratory  # noqa: E402
from psg.rules import DEFAULT_RULE, RULES, flowchart, get_rule  # noqa: E402


def test_rule_library():
    assert len(RULES) >= 5 and DEFAULT_RULE in RULES
    for r in RULES.values():
        p = r.params()
        assert (p.apnea_drop, p.hypopnea_drop, p.min_duration, p.hypopnea_desat, p.hypopnea_arousal) == \
               (r.apnea_drop, r.hypopnea_drop, r.min_duration, r.desat, r.arousal)
        assert p.rule_id == r.id and r.source and r.summary
        steps = {s.id: s for s in flowchart(p)}
        assert {"AO", "AC", "AM", "reject_confirm"} <= set(steps)
        assert ("HO" in steps) == r.classify_hypopneas or "H" in steps
    try:
        get_rule("nope")
        assert False, "unknown rule must raise"
    except KeyError:
        pass


def test_confidence_and_rescore():
    rec = _synthetic()
    stages = np.full(int(DUR // 30), STAGE_N2)
    res = score_respiratory(rec, ScoringParams(), stages=stages)
    assert len(res.events) == len(EVENTS)
    for e in res.events:
        assert 0 <= e.confidence <= 1 and e.checks and e.accepted
        assert e.explanation().startswith("Scored as")
        lim = e.limiting
        assert lim is not None and lim.counts
    hyp = [e for e in res.events if e.kind == "hypopnea"]
    assert all(e.confidence > 0.85 for e in hyp), [e.confidence for e in hyp]   # 5 % desat vs 3 % rule: clear

    # Same parameters through the fast path must reproduce the same events.
    again = rescore(res, ScoringParams())
    assert [(round(e.onset), e.code) for e in again.events] == [(round(e.onset), e.code) for e in res.events]

    # Move the desaturation threshold onto the synthetic 5 % drops: borderline -> ~50 % confidence.
    edge = rescore(res, ScoringParams(hypopnea_desat=5.0, hypopnea_arousal=False))
    hyp_edge = [e for e in edge.events if e.kind == "hypopnea"]
    assert len(hyp_edge) == len(hyp)
    assert all(0.3 <= e.confidence <= 0.7 for e in hyp_edge), [e.confidence for e in hyp_edge]
    assert all(e.limiting.label == "SpO2 desaturation" for e in hyp_edge)
    assert len(edge.events_at_confidence(0.9)) == len(res.events) - len(hyp)

    # Beyond the drop: hypopneas rejected with a stated reason, apneas unaffected.
    strict = rescore(res, ScoringParams(hypopnea_desat=5.6, hypopnea_arousal=False))
    assert all(e.kind == "apnea" for e in strict.events)
    rej = strict.rejected_events
    assert len(rej) == len(hyp) and all("SpO2 fell" in r.reject_reason for r in rej)
    assert all(not r.accepted and r.path[-1] == "reject_confirm" for r in rej)

    # Rule versions: the 2007 alternative rule (>= 50 % drop) drops the -50 % synthetic hypopneas
    # (they sit at the threshold, some fall under it), the 4 % rules keep them.
    for rid in ("aasm2013_acc", "shhs", "chicago1999", "aasm_current"):
        rr = rescore(res, get_rule(rid).params())
        assert len(rr.events) == len(EVENTS), (rid, len(rr.events))


def test_sleep_rule_begins_or_ends():
    rec = _synthetic()
    n = int(DUR // 30)
    # Event at 300-320 s lies in epoch 10.  Wake everywhere except that epoch -> still scored.
    stages = np.full(n, STAGE_W)
    stages[10] = STAGE_N2
    res = score_respiratory(rec, ScoringParams(), stages=stages)
    assert any(abs(e.onset - 300) < 5 for e in res.events)
    # Epoch 10 wake, everything else sleep -> that event is entirely in wake -> rejected, others kept.
    stages = np.full(n, STAGE_N2)
    stages[10] = STAGE_W
    res = score_respiratory(rec, ScoringParams(), stages=stages)
    assert not any(abs(e.onset - 300) < 5 for e in res.events)
    rej = [r for r in res.rejected_events if abs(r.onset - 300) < 5]
    assert rej and "wake" in rej[0].reject_reason and rej[0].path[-1] == "reject_wake"
    assert len(res.events) == len(EVENTS) - 1


def test_logistic_calibration():
    from psg.calibrate import FEATURES, auc, brier, event_features, fit_logistic
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, len(FEATURES)))
    true_w = np.array([1.0, -0.5, 2.0, 0.3, 1.0, 0.0, -1.0, 0.5])
    y = (1 / (1 + np.exp(-(X @ true_w))) > rng.random(400)).astype(float)
    m = fit_logistic(X, y, l2=0.5)
    p = m.predict(X)
    assert auc(p, y) > 0.9 and brier(p, y) < 0.15
    assert np.sign(m.w[2]) > 0 and np.sign(m.w[6]) < 0          # strongest effects recovered
    rec = _synthetic()
    res = score_respiratory(rec, ScoringParams(), stages=np.full(int(DUR // 30), STAGE_N2))
    f = event_features(res.events[0])
    assert len(f) == len(FEATURES) and all(np.isfinite(f))
    assert all(e.probability is None or 0.0 <= e.probability <= 1.0 for e in res.events)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
