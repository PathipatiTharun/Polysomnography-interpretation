"""Figures and an HTML page for a recording and its automatic annotations.

  overview   - whole recording: technician vs automatic hypnogram, EEG spectrogram, event
               rasters (spindles, slow waves, eye movements, arousals), EMG level,
               temperature / breathing when present; estimated lights-off period shaded
  examples   - one representative 30-s epoch per sleep stage with EEG / EOG traces and the
               detected events shaded
  agreement  - confusion matrix and stage minutes, automatic vs technician
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import signal

from . import preprocess as pp
from .annotate import AnnotationResult
from .evaluate import staging_agreement
from .io import EPOCH_S, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R, STAGE_UNK, STAGE_W

STAGE_COLORS = {STAGE_W: "#cfd6de", STAGE_N1: "#a6cbe3", STAGE_N2: "#4a90c9", STAGE_N3: "#1b3f73",
                STAGE_R: "#e2574c", STAGE_UNK: "#eef1f4"}
STAGE_Y = {STAGE_W: 4, STAGE_R: 3, STAGE_N1: 2, STAGE_N2: 1, STAGE_N3: 0}
STAGE_NAME = {STAGE_W: "W", STAGE_N1: "N1", STAGE_N2: "N2", STAGE_N3: "N3", STAGE_R: "R", STAGE_UNK: "?"}
EVENT_COLORS = {"Spindle": "#7b2cbf", "Slow wave": "#1b3f73", "Rapid eye movement": "#e2574c", "Arousal": "#ef6c00"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b8c2cc",
                     "axes.labelcolor": "#3c4650", "xtick.color": "#5b6b7c", "ytick.color": "#5b6b7c"})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _times(res: AnnotationResult, n: int, step_s: float):
    start = res.rec.start or dt.datetime(2000, 1, 1)
    return [start + dt.timedelta(seconds=i * step_s) for i in range(n)]


def _hypno(ax, res, stages, title):
    t = _times(res, len(stages) + 1, EPOCH_S)
    x = mdates.date2num(t)
    for k, s in enumerate(stages):
        s = int(s)
        if s in STAGE_Y:
            ax.add_patch(Rectangle((x[k], STAGE_Y[s] - 0.38), x[k + 1] - x[k], 0.76, color=STAGE_COLORS[s], lw=0))
    y = [STAGE_Y.get(int(s), np.nan) for s in stages]
    ax.step(x[:-1], y, where="post", color="#8a96a3", lw=0.5)
    ax.set_yticks([4, 3, 2, 1, 0], ["W", "R", "N1", "N2", "N3"])
    ax.set_ylim(-0.6, 4.6)
    ax.set_title(title)


def overview_figure(res: AnnotationResult) -> bytes:
    rec = res.rec
    has_expert = rec.expert_stages is not None
    rows = [("expert", 1.0)] if has_expert else []
    rows += [("auto", 1.0), ("spec", 1.6), ("events", 1.0), ("emg", 0.6)]
    temp = next((c for c in rec.channels.values() if "temp" in c.label.lower()), None)
    resp = rec.get("flow")
    if temp is not None:
        rows.append(("temp", 0.7))
    if resp is not None:
        rows.append(("resp", 0.6))
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.35 * sum(h for _, h in rows) + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": [h for _, h in rows], "hspace": 0.45})
    ax = dict(zip([r for r, _ in rows], axes))
    n = len(res.stages)
    t_ep = mdates.date2num(_times(res, n + 1, EPOCH_S))

    if has_expert:
        ex = np.asarray(rec.expert_stages)[:n]
        _hypno(ax["expert"], res, ex, "Hypnogram – technician")
    _hypno(ax["auto"], res, res.stages, "Hypnogram – automatic")

    # EEG spectrogram per 30-s epoch (Welch, log power), 0.5-25 Hz.
    eeg = rec.get("eeg")
    if eeg is not None:
        el = int(EPOCH_S * eeg.fs)
        X = eeg.data[:n * el].reshape(n, el)
        f, pxx = signal.welch(X, fs=eeg.fs, nperseg=int(4 * eeg.fs), axis=1)
        m = (f >= 0.5) & (f <= 25)
        P = 10 * np.log10(pxx[:, m].T + 1e-12)
        lo, hi = np.percentile(P, [5, 99])
        ax["spec"].imshow(P, aspect="auto", origin="lower", cmap="magma", vmin=lo, vmax=hi,
                          extent=[t_ep[0], t_ep[-1], f[m][0], f[m][-1]], interpolation="nearest")
        for fb, name in ((4, "δ | θ"), (8, "θ | α"), (12, "α | σ"), (16, "σ | β")):
            ax["spec"].axhline(fb, color="white", lw=0.4, alpha=0.5)
            ax["spec"].text(t_ep[-1], fb, f" {name}", color="#555", fontsize=6, va="center")
        ax["spec"].set_ylabel("Hz")
        ax["spec"].set_title(f"EEG spectrogram – {eeg.label} (power per 30-s epoch, dB)")

    # Event rasters.
    labels = ["Spindle", "Slow wave", "Rapid eye movement", "Arousal"]
    ea = ax["events"]
    start = rec.start or dt.datetime(2000, 1, 1)
    for i, lab in enumerate(labels):
        evs = res.of(lab)
        if evs:
            xs = mdates.date2num([start + dt.timedelta(seconds=a.onset) for a in evs])
            ea.vlines(xs, i + 0.12, i + 0.88, color=EVENT_COLORS[lab], lw=0.5, alpha=0.7)
    ea.set_yticks(np.arange(len(labels)) + 0.5, [f"{l}s ({len(res.of(l))})" if l != "Rapid eye movement"
                                                 else f"REMs ({len(res.of(l))})" for l in labels])
    ea.set_ylim(0, len(labels))
    ea.set_title("Detected events")

    emg = rec.get("emg")
    if emg is not None:
        step = max(int(emg.fs * EPOCH_S), 1)
        v = np.abs(emg.data[:n * step]).reshape(-1, step).mean(axis=1) if emg.fs < 40 else \
            pp.moving_rms(pp.filt(emg.data, emg.fs, 10, min(30, 0.45 * emg.fs)), step)[::step][:n]
        ax["emg"].plot(t_ep[:len(v)], v, color="#6f42c1", lw=0.7)
        ax["emg"].set_title(f"Chin muscle tone – {emg.label} (per epoch)")
    if temp is not None:
        ts = mdates.date2num(_times(res, len(temp.data), 1 / temp.fs))
        ax["temp"].plot(ts, temp.data, color="#c2410c", lw=0.8)
        ax["temp"].set_ylabel("°C")
        ax["temp"].set_title(f"{temp.label} – body temperature falls during the night (circadian rhythm)")
    if resp is not None:
        step = max(int(resp.fs * EPOCH_S), 1)
        r = resp.data[:n * step].reshape(-1, step).std(axis=1)
        ax["resp"].plot(t_ep[:len(r)], r, color="#1c4e80", lw=0.7)
        ax["resp"].set_title(f"{resp.label} – breathing amplitude per epoch (1 Hz channel: too coarse for apnea scoring)")

    if res.rest is not None:
        a, b = mdates.date2num([start + dt.timedelta(seconds=res.rest[0] * EPOCH_S),
                                start + dt.timedelta(seconds=res.rest[1] * EPOCH_S)])
        for axx in axes:
            axx.axvspan(a, b, color="#1f6fb2", alpha=0.05, lw=0)
        axes[0].annotate("estimated lights off → on", xy=(a, 4.6), fontsize=7, color="#1f6fb2", va="bottom")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].set_xlim(t_ep[0], t_ep[-1])
    return _png(fig)


def _pick_epoch(res: AnnotationResult, code: int, prefer: str | None) -> int | None:
    st = res.stages
    ex = res.rec.expert_stages
    cands = [k for k in range(1, len(st) - 1) if st[k] == code and st[k - 1] == code and st[k + 1] == code
             and (ex is None or (k < len(ex) and ex[k] == code))]
    if not cands:
        cands = [k for k in range(len(st)) if st[k] == code]
    if not cands:
        return None
    if prefer:
        counts = {k: 0 for k in cands}
        for a in res.of(prefer):
            k = int(a.onset // EPOCH_S)
            if k in counts:
                counts[k] += 1
        best = max(counts.values())
        if best > 0:
            cands = [k for k, c in counts.items() if c >= max(1, int(0.6 * best))]
    return cands[len(cands) // 2]


def examples_figure(res: AnnotationResult) -> bytes:
    rec = res.rec
    chans = [c for c in (rec.get("eeg"), rec.get("eeg2"), rec.get("eog_l")) if c is not None and c.fs >= 20]
    plan = [(STAGE_W, None, "Wake – fast activity, blinks and eye movements on the EOG (alpha 8–12 Hz when the eyes close)"),
            (STAGE_N1, None, "N1 – slower theta (4–8 Hz), alpha disappears, slow rolling eye movements"),
            (STAGE_N2, "Spindle", "N2 – sleep spindles (11–16 Hz bursts, purple) and K-complexes"),
            (STAGE_N3, "Slow wave", "N3 – deep sleep: large slow waves ≥ 75 µV (blue) fill the epoch"),
            (STAGE_R, "Rapid eye movement", "REM – mixed-frequency EEG with rapid eye movements (red) on the EOG")]
    plan = [(c, p, t, _pick_epoch(res, c, p)) for c, p, t in plan]
    plan = [x for x in plan if x[3] is not None]
    fig, axes = plt.subplots(len(plan), 1, figsize=(13, 2.1 * len(plan)), squeeze=False)
    start = rec.start or dt.datetime(2000, 1, 1)
    for ax, (code, prefer, title, k) in zip(axes[:, 0], plan):
        t0, t1 = k * EPOCH_S, (k + 1) * EPOCH_S
        offsets = []
        for i, ch in enumerate(chans):
            i0, i1 = int(t0 * ch.fs), int(t1 * ch.fs)
            y = pp.filt(ch.data[max(i0 - 500, 0):i1 + 500], ch.fs, 0.3, 35.0, order=2)[i0 - max(i0 - 500, 0):][:i1 - i0]
            off = -i * 160.0
            offsets.append(off)
            tt = np.arange(len(y)) / ch.fs
            col = "#0b7285" if ch.role == "eog_l" else "#2b2f36"
            ax.plot(tt, y + off, color=col, lw=0.6)
            ax.text(-0.3, off, ch.label, ha="right", va="center", fontsize=7, color=col)
        for lab, colr in EVENT_COLORS.items():
            for a in res.annotations:
                if a.label == lab and a.end > t0 and a.onset < t1:
                    ax.axvspan(max(a.onset, t0) - t0, min(a.end, t1) - t0, color=colr, alpha=0.18, lw=0)
        ax.set_xlim(0, EPOCH_S)
        ax.set_ylim(offsets[-1] - 120, 120)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        clock = (start + dt.timedelta(seconds=t0)).strftime("%H:%M:%S")
        agree = ""
        if rec.expert_stages is not None and k < len(rec.expert_stages):
            agree = f"  ·  technician: {STAGE_NAME.get(int(rec.expert_stages[k]), '?')}"
        ax.set_title(f"{title}   (epoch {k + 1}, {clock}{agree})")
        ax.plot([EPOCH_S - 1.2, EPOCH_S - 0.2], [offsets[-1] - 90] * 2, color="k", lw=1)
        ax.plot([EPOCH_S - 0.2] * 2, [offsets[-1] - 90, offsets[-1] - 40], color="k", lw=1)
        ax.text(EPOCH_S - 0.25, offsets[-1] - 100, "1 s / 50 µV", ha="right", va="top", fontsize=6)
    axes[-1, 0].set_xlabel("seconds within the 30-s epoch")
    fig.tight_layout()
    return _png(fig)


def agreement_figure(res: AnnotationResult) -> tuple[bytes | None, dict | None]:
    ex = res.rec.expert_stages
    if ex is None:
        return None, None
    a, b = res.rest if res.rest is not None else (0, len(res.stages))
    agr = staging_agreement(res.stages[a:b], np.asarray(ex)[a:b])
    agr_all = staging_agreement(res.stages, ex)
    C = agr["confusion"].astype(float)
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.35})
    a1.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            a1.text(j, i, f"{int(C[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if Cn[i, j] > 0.55 else "#223")
    a1.set_xticks(range(5), agr["labels"])
    a1.set_yticks(range(5), agr["labels"])
    a1.set_xlabel("automatic")
    a1.set_ylabel("technician")
    a1.set_title(f"Epochs, sleep period: {agr['accuracy'] * 100:.0f} % agree, κ {agr['kappa']:.2f}")
    codes = [STAGE_W, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_R]
    exm = [float((np.asarray(ex)[a:b] == c).sum() * 0.5) for c in codes]
    aum = [float((res.stages[a:b] == c).sum() * 0.5) for c in codes]
    xs = np.arange(5)
    a2.bar(xs - 0.2, exm, 0.4, label="technician", color="#9aa6b2")
    a2.bar(xs + 0.2, aum, 0.4, label="automatic", color=[STAGE_COLORS[c] for c in codes])
    a2.set_xticks(xs, [STAGE_NAME[c] for c in codes])
    a2.set_ylabel("minutes")
    a2.set_title("Minutes per stage (sleep period)")
    a2.legend(frameon=False)
    fig.tight_layout()
    return _png(fig), {"period": agr, "whole": agr_all}


def write_page(res: AnnotationResult, out_dir: Path, files: dict[str, str]) -> Path:
    out_dir = Path(out_dir)
    name = res.rec.name
    imgs = {"overview": overview_figure(res), "examples": examples_figure(res)}
    agr_png, agr = agreement_figure(res)
    for k, v in imgs.items():
        (out_dir / f"{name}_{k}.png").write_bytes(v)
    if agr_png:
        (out_dir / f"{name}_agreement.png").write_bytes(agr_png)
    b64 = lambda b: base64.b64encode(b).decode()
    s = res.summary
    fmt = lambda v, f="{:.1f}": "—" if v is None else f.format(v)
    rows = [
        ("Recording", f"{name} · {res.rec.duration / 3600:.1f} h · {len(res.rec.channels)} channels · start {res.rec.start}"),
        ("Estimated lights off → on", "whole file" if s["rest_period_h"] is None else
         f"{s['rest_period_h'][0]:.1f} h → {s['rest_period_h'][1]:.1f} h after start"),
        ("Total sleep time / efficiency", f"{s['tst_min']:.0f} min / {s['sleep_efficiency']:.0f} %"),
        ("Sleep latency / REM latency", f"{fmt(s['sleep_latency_min'], '{:.0f}')} / {fmt(s['rem_latency_min'], '{:.0f}')} min"),
        ("N1 / N2 / N3 / REM", f"{s['pct_n1']:.0f} / {s['pct_n2']:.0f} / {s['pct_n3']:.0f} / {s['pct_rem']:.0f} % of sleep"),
        ("Sleep spindles", f"{s['n_spindles']} · {fmt(s['spindle_density_per_min_nrem'], '{:.2f}')} per NREM minute · median {fmt(s['spindle_freq_hz'])} Hz"),
        ("Slow waves", "skipped (EEG not in µV)" if s["slow_waves_skipped"] else
         f"{s['n_slow_waves']} · {fmt(s['slow_waves_per_min_n3'])} per N3 minute"),
        ("Rapid eye movements", f"{s['n_rems']} · {fmt(s['rem_density_per_min_r'])} per REM minute"),
        ("EEG arousals", f"{s['n_arousals']} · {fmt(s['arousal_index'])} per hour of sleep"),
    ]
    if agr:
        p, w = agr["period"], agr["whole"]
        rows.append(("Agreement with technician", f"sleep period: {p['accuracy'] * 100:.0f} %, κ {p['kappa']:.2f} "
                                                  f"({p['n']} epochs) · whole file: {w['accuracy'] * 100:.0f} %, κ {w['kappa']:.2f}"))
    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows)
    counts: dict[str, int] = {}
    for a in res.annotations:
        counts[a.label] = counts.get(a.label, 0) + 1
    count_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(counts.items()))
    links = "".join(f"<li><a href='{html.escape(f)}'>{html.escape(f)}</a> — {html.escape(d)}</li>" for f, d in files.items())
    agr_html = f"<h2>Agreement with the technician</h2><img src='data:image/png;base64,{b64(agr_png)}'>" if agr_png else ""
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Annotations – {html.escape(name)}</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1b2a3a;max-width:1180px;margin:24px auto;padding:0 16px;background:#f4f6f9}}
.card{{background:#fff;border:1px solid #dde3ea;border-radius:10px;padding:14px 18px;margin:14px 0}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:0 0 8px}} img{{max-width:100%}}
table{{border-collapse:collapse}} th,td{{text-align:left;padding:3px 12px 3px 0;border-bottom:1px solid #eef2f6;vertical-align:top}}
th{{color:#5b6b7c;font-weight:500}} .muted{{color:#5b6b7c;font-size:12px}}</style></head><body>
<h1>Automatic annotations – {html.escape(name)}</h1>
<div class="muted">Generated {dt.datetime.now():%Y-%m-%d %H:%M} by PSG Interpreter · automated, to be reviewed by a qualified scorer</div>
<div class="card"><h2>Summary</h2><table>{table}</table></div>
<div class="card"><h2>The whole recording</h2><img src='data:image/png;base64,{b64(imgs['overview'])}'></div>
<div class="card"><h2>What each sleep stage looks like in this recording</h2><img src='data:image/png;base64,{b64(imgs['examples'])}'></div>
<div class="card">{agr_html}</div>
<div class="card"><h2>Annotation files</h2><ul>{links}</ul><table>{count_rows}</table>
<p class="muted">Stages use AASM names (N1–N3, R). The technician scored with the older Rechtschaffen &amp; Kales rules
(stages 1–4); for comparison stages 3 and 4 are merged into N3.</p></div>
</body></html>"""
    path = out_dir / f"{name}_annotations.html"
    path.write_text(doc, encoding="utf-8")
    return path
