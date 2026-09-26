"""Compare automatic scoring with expert annotations (event matching, staging agreement)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io import ExpertEvent, STAGE_W, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R, STAGE_NAMES
from .respiratory import RespEvent


@dataclass
class EventMatch:
    n_expert: int
    n_detected: int
    matched: list[tuple[RespEvent, ExpertEvent]] = field(default_factory=list)
    missed: list[ExpertEvent] = field(default_factory=list)
    false_alarm: list[RespEvent] = field(default_factory=list)

    @property
    def sensitivity(self) -> float:
        return len(self.matched) / self.n_expert if self.n_expert else float("nan")

    @property
    def precision(self) -> float:
        return len(self.matched) / self.n_detected if self.n_detected else float("nan")

    @property
    def f1(self) -> float:
        s, p = self.sensitivity, self.precision
        return 2 * s * p / (s + p) if (s + p) > 0 else 0.0

    def kind_agreement(self) -> float:
        if not self.matched:
            return float("nan")
        return float(np.mean([d.kind == e.kind for d, e in self.matched]))

    def subtype_agreement(self) -> float:
        pairs = [(d, e) for d, e in self.matched if e.subtype != "unknown"]
        if not pairs:
            return float("nan")
        return float(np.mean([d.subtype == e.subtype for d, e in pairs]))

    def confusion(self) -> dict[tuple[str, str], int]:
        c: dict[tuple[str, str], int] = {}
        for d, e in self.matched:
            k = (e.kind[0] + "-" + e.subtype[0], d.kind[0] + "-" + d.subtype[0])
            c[k] = c.get(k, 0) + 1
        return c


def match_events(detected: list[RespEvent], expert: list[ExpertEvent], slack_s: float = 5.0) -> EventMatch:
    """Greedy one-to-one matching: a pair matches when the intervals overlap (allowing `slack_s`)."""
    res = EventMatch(n_expert=len(expert), n_detected=len(detected))
    used = np.zeros(len(detected), dtype=bool)
    det_sorted = sorted(range(len(detected)), key=lambda i: detected[i].onset)
    for e in sorted(expert, key=lambda x: x.onset):
        best, best_ov = None, -1.0
        for i in det_sorted:
            if used[i]:
                continue
            d = detected[i]
            if d.onset > e.end + slack_s:
                break
            ov = min(d.end, e.end) - max(d.onset, e.onset)
            if ov >= -slack_s and ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            res.missed.append(e)
        else:
            used[best] = True
            res.matched.append((detected[best], e))
    res.false_alarm = [detected[i] for i in range(len(detected)) if not used[i]]
    return res


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == l)) * float(np.mean(b == l)) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def staging_agreement(pred: np.ndarray, expert: np.ndarray) -> dict:
    n = min(len(pred), len(expert))
    p, e = np.asarray(pred[:n]), np.asarray(expert[:n])
    keep = (e >= 0) & (p >= 0)
    p, e = p[keep], e[keep]
    stages = [STAGE_W, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R]
    conf = np.zeros((5, 5), dtype=int)
    for i, s in enumerate(stages):
        for j, t in enumerate(stages):
            conf[i, j] = int(np.sum((e == s) & (p == t)))
    per_stage = {}
    for i, s in enumerate(stages):
        tp = conf[i, i]
        fn = conf[i].sum() - tp
        fp = conf[:, i].sum() - tp
        per_stage[STAGE_NAMES[s]] = {
            "n": int(conf[i].sum()),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
        }
    return {"n": int(len(e)), "accuracy": float(np.mean(p == e)) if len(e) else float("nan"),
            "kappa": cohen_kappa(p, e), "confusion": conf, "per_stage": per_stage,
            "labels": [STAGE_NAMES[s] for s in stages]}
