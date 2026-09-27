"""Export analysis results: HTML clinical-style report, CSV event list, JSON indices."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

from .io import STAGE_NAMES, EPOCH_S
from .pipeline import AnalysisResult


def _fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _clock(res: AnalysisResult, sec: float) -> str:
    if res.rec.start is None:
        return _fmt_time(sec)
    import datetime as dt
    return (res.rec.start + dt.timedelta(seconds=sec)).strftime("%H:%M:%S")


def _num(v, fmt="{:.1f}", none="—") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return none
    return fmt.format(v)


def write_events_csv(res: AnalysisResult, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onset_s", "clock", "duration_s", "type", "subtype", "flow_drop_pct", "effort_drop_pct",
                    "desat_pct", "spo2_nadir", "arousal", "snoring", "paradox", "stage", "position", "notes"])
        for e in res.events:
            w.writerow([f"{e.onset:.1f}", _clock(res, e.onset), f"{e.duration:.1f}", e.kind, e.subtype,
                        f"{e.flow_drop * 100:.0f}",
                        "" if e.effort_drop is None else f"{e.effort_drop * 100:.0f}",
                        "" if e.desat is None else e.desat, "" if e.desat_nadir is None else e.desat_nadir,
                        int(e.arousal), int(e.snoring), int(e.paradox), e.stage or "",
                        "" if e.position is None else e.position, e.notes])


def write_json(res: AnalysisResult, path: str | Path) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    payload = {"recording": res.rec.name, "start": str(res.rec.start), "stage_source": res.stage_source,
               "diagnosis": res.diagnosis, "indices": res.summary, "warnings": res.warnings,
               "hypnogram": [STAGE_NAMES[int(s)] for s in res.stages]}
    Path(path).write_text(json.dumps(clean(payload), indent=2))


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _overview_figure(res: AnalysisResult):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = np.arange(len(res.stages)) * EPOCH_S / 3600
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1.2]})
    # Hypnogram (clinical order: W at top, N3 at bottom, R drawn at its own level).
    ypos = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0, -1: np.nan}
    y = np.array([ypos[int(s)] for s in res.stages], dtype=float)
    ax = axes[0]
    ax.step(hours, y, where="post", color="#333", lw=1)
    rem = res.stages == 4
    ax.fill_between(hours, 2.8, 3.2, where=rem, step="post", color="#d62728", alpha=0.8, lw=0)
    ax.set_yticks([4, 3, 2, 1, 0], ["W", "R", "N1", "N2", "N3"])
    ax.set_ylim(-0.5, 4.5)
    ax.set_title("Hypnogram", loc="left", fontsize=9)
    # Events.
    ax = axes[1]
    colors = {"AO": "#1f77b4", "AC": "#2ca02c", "AM": "#9467bd", "HO": "#ff7f0e", "HC": "#8c564b"}
    rows = {"AO": 4, "AM": 3, "AC": 2, "HO": 1, "HC": 0}
    for e in res.events:
        ax.barh(rows[e.code], e.duration / 3600, left=e.onset / 3600, height=0.7, color=colors[e.code], lw=0)
    ax.set_yticks(list(rows.values()), ["Obstr. apnea", "Mixed apnea", "Central apnea", "Obstr. hypopnea", "Central hypopnea"], fontsize=7)
    ax.set_ylim(-0.6, 4.6)
    ax.set_title("Respiratory events", loc="left", fontsize=9)
    # SpO2.
    ax = axes[2]
    if res.resp.spo2_clean is not None:
        fs = res.resp.spo2_fs
        step = max(int(fs), 1)
        s = res.resp.spo2_clean[::step]
        ax.plot(np.arange(len(s)) / 3600, s, color="#1f77b4", lw=0.6)
        ax.axhline(90, color="#d62728", lw=0.6, ls="--")
        lo = np.nanmin(s) if np.isfinite(s).any() else 80
        ax.set_ylim(max(min(lo - 2, 88), 50), 100)
    ax.set_ylabel("SpO2 %", fontsize=8)
    ax.set_xlabel("Hours from recording start", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    b64 = _png_b64(fig)
    plt.close(fig)
    return b64


def write_html(res: AnalysisResult, path: str | Path, validation: dict | None = None) -> None:
    ix, dg = res.summary, res.diagnosis
    ox = ix.get("spo2") or {}
    sev_color = {"Normal": "#2e7d32", "Mild": "#f9a825", "Moderate": "#ef6c00", "Severe": "#c62828"}.get(dg["severity"], "#555")
    p = res.options.scoring
    from .rules import RULES
    rv = RULES.get(p.rule_id)
    rule = (f"{rv.name}: {rv.short()}" if rv else
            f"Apnea ≥{p.apnea_drop * 100:.0f}% drop ≥{p.min_duration:.0f} s · Hypopnea ≥{p.hypopnea_drop * 100:.0f}% "
            f"drop ≥{p.min_duration:.0f} s + ≥{p.hypopnea_desat:g}% desaturation" + (" or arousal" if p.hypopnea_arousal else ""))
    cc = ix.get("confidence_counts", {})
    conf_txt = " / ".join(f"≥{int(k * 100)}%: {v}" for k, v in cc.items()) if cc else "—"
    min_conf = ix.get("min_confidence", 0.0)

    def row(k, v):
        return f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>"

    resp_rows = "".join([
        row("Scoring rule", html.escape(rule)),
        row("Events by confidence", f"{conf_txt} (of {ix.get('n_scored_all', ix['n_events'])} scored; "
                                    f"{ix.get('n_rejected', 0)} candidates rejected)"
                                    + (f"; indices use ≥{min_conf * 100:.0f}%" if min_conf else "")),
        row("AHI (apnea-hypopnea index)", f"<b>{_num(ix['ahi'])}</b> /h"),
        row("Obstructive AHI / Central AHI", f"{_num(ix['oahi'])} / {_num(ix['cahi'])} /h"),
        row("Apnea index / Hypopnea index", f"{_num(ix['ai'])} / {_num(ix['hi'])} /h"),
        row("REM AHI / NREM AHI", f"{_num(ix['rem_ahi'])} / {_num(ix['nrem_ahi'])} /h"),
        row("Obstructive apneas", ix["n_obstructive_apnea"]),
        row("Central apneas", ix["n_central_apnea"]),
        row("Mixed apneas", ix["n_mixed_apnea"]),
        row("Obstructive hypopneas", ix["n_obstructive_hypopnea"]),
        row("Central hypopneas", ix["n_central_hypopnea"]),
        row("Mean / longest event", f"{_num(ix['mean_event_duration'])} / {_num(ix['max_event_duration'])} s"),
        row("Arousal index", f"{_num(ix['arousal_index'])} /h"),
    ])
    pos = ix.get("position") or {}
    if pos:
        resp_rows += row("Supine AHI / non-supine AHI", f"{_num(pos.get('supine_ahi'))} / {_num(pos.get('nonsupine_ahi'))} /h")
    ox_rows = "".join([
        row("ODI (≥3 % desaturations)", f"{_num(ix['odi'])} /h"),
        row("Mean SpO2 (sleep)", f"{_num(ox.get('mean'))} %"),
        row("Nadir SpO2", f"{_num(ox.get('nadir'), '{:.0f}')} %"),
        row("Time < 90 % / < 88 %", f"{_num(ox.get('t90_min'))} / {_num(ox.get('t88_min'))} min"),
    ])
    sleep_rows = "".join([
        row("Total recording time", f"{_num(ix['trt_min'], '{:.0f}')} min"),
        row("Total sleep time", f"{_num(ix['tst_min'], '{:.0f}')} min"),
        row("Sleep efficiency", f"{_num(ix['sleep_efficiency'], '{:.0f}')} %"),
        row("Sleep latency", f"{_num(ix['sleep_latency_min'], '{:.0f}')} min"),
        row("REM latency", f"{_num(ix['rem_latency_min'], '{:.0f}')} min"),
        row("WASO", f"{_num(ix['waso_min'], '{:.0f}')} min"),
        row("N1 / N2 / N3 / REM", f"{ix['pct_n1']:.0f} / {ix['pct_n2']:.0f} / {ix['pct_n3']:.0f} / {ix['pct_rem']:.0f} % of TST"),
    ])
    q_rows = "".join(row(k, f"{v * 100:.0f} %") for k, v in ix.get("signal_quality", {}).items())
    def conf_cell(e):
        c = e.confidence * 100
        col = "#2e7d32" if c >= 90 else ("#f9a825" if c >= 75 else "#c62828")
        lim = e.limiting
        tip = html.escape(f"limited by {lim.label.lower()}: {lim.text()}" if lim else "")
        return f"<td style='color:{col}' title='{tip}'>{c:.0f}</td>"
    ev_rows = "".join(
        f"<tr><td>{_clock(res, e.onset)}</td><td>{e.duration:.0f}</td><td>{html.escape(e.label)}</td>{conf_cell(e)}"
        f"<td>{e.flow_drop * 100:.0f}</td><td>{_num(e.desat)}</td><td>{_num(e.desat_nadir, '{:.0f}')}</td>"
        f"<td>{e.stage or ''}</td><td>{'✓' if e.arousal else ''}</td><td>{html.escape(e.notes)}</td></tr>"
        for e in res.events)
    flags = "".join(f"<li>{html.escape(f)}</li>" for f in dg["flags"]) or "<li>None</li>"
    findings = "".join(f"<li>{html.escape(f)}</li>" for f in dg["findings"])
    warns = "".join(f"<li>{html.escape(w)}</li>" for w in res.warnings)
    val_html = ""
    if validation:
        val_html = "<h2>Agreement with expert scoring</h2><table>" + "".join(
            row(k, html.escape(str(v))) for k, v in validation.items()) + "</table>"
    img = _overview_figure(res)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>PSG report {html.escape(res.rec.name)}</title>
<style>
body{{font:14px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#222;max-width:1000px;margin:24px auto;padding:0 16px}}
h1{{font-size:22px;margin:0}} h2{{font-size:16px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:28px}}
.meta{{color:#666;font-size:12px}} .dx{{border-left:6px solid {sev_color};background:#fafafa;padding:10px 14px;margin:16px 0}}
.dx b{{font-size:17px}} .sev{{display:inline-block;background:{sev_color};color:#fff;border-radius:3px;padding:1px 8px;font-size:12px;margin-left:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:3px 6px;border-bottom:1px solid #eee}}
th{{font-weight:500;color:#555}} .ev th{{background:#f4f4f4}} .ev td,.ev th{{font-size:12px}}
.note{{font-size:11px;color:#777}} img{{max-width:100%}}
</style></head><body>
<h1>Polysomnography interpretation</h1>
<div class="meta">Recording <b>{html.escape(res.rec.name)}</b> · start {html.escape(str(res.rec.start))} ·
staging: {html.escape(res.stage_source)} · generated automatically in {res.runtime_s:.0f} s</div>
<div class="dx"><b>{html.escape(dg['primary'])}</b><span class="sev">{html.escape(dg['severity'])}</span>
<ul>{findings}</ul><div><b style="font-size:13px">Clinical flags</b><ul>{flags}</ul></div></div>
<img src="data:image/png;base64,{img}" alt="Hypnogram, events and SpO2 overview">
<div class="grid">
<div><h2>Respiratory</h2><table>{resp_rows}</table></div>
<div><h2>Oxygenation</h2><table>{ox_rows}</table><h2>Sleep architecture</h2><table>{sleep_rows}</table></div>
</div>
<h2>Signal quality</h2><table>{q_rows}</table>
{('<h2>Warnings</h2><ul>' + warns + '</ul>') if warns else ''}
{val_html}
<h2>Event list ({len(res.events)})</h2>
<table class="ev"><tr><th>Time</th><th>Dur s</th><th>Type</th><th>Conf %</th><th>Flow ↓%</th><th>Desat %</th><th>Nadir</th><th>Stage</th><th>Arousal</th><th>Notes</th></tr>{ev_rows}</table>
<p class="note">Scoring rule: {html.escape(rule)}. Obstructive / central / mixed apneas by thoraco-abdominal effort;
hypopnea obstructive if snoring, flow flattening, paradox or preserved effort.
Confidence = smallest margin by which the event clears the rule's thresholds (50 % = exactly at a threshold).
Severity: AHI &lt;5 normal, 5–15 mild, 15–30 moderate, ≥30 severe.
This is automated decision support and must be reviewed by a qualified sleep physician.</p>
</body></html>"""
    Path(path).write_text(doc, encoding="utf-8")
