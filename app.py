"""PSG Interpreter - PyQt5 desktop application.

    python app.py [recording.edf ...]

Workflow: the doctor uploads a recording (drag-and-drop anywhere, or the Upload button) and
the system does the rest - file check, cleaning, sleep staging, arousals, AASM respiratory
scoring, oximetry, indices, diagnosis - then saves the report automatically and opens the
results.  Several files can be dropped at once; they are processed one after another.

Screens
  * Home          drop zone + list of previously processed studies
  * Processing    live checklist of analysis steps
  * Results       overview strip, signal viewer with scored events, event table, report tab
Keys: Left/Right (or PgUp/PgDn) page, N/P next/previous event, +/- amplitude, Ctrl+O upload.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from psg import preprocess as pp
from psg.io import EPOCH_S, STAGE_NAMES, Recording, load_recording
from psg.pipeline import AnalysisOptions, AnalysisResult, analyze, rescore_analysis
from psg.report import write_events_csv, write_html, write_json
from psg.respiratory import ScoringParams
from psg.study import ACCEPTED_SUFFIXES, StudyError, list_studies, process_study
from widgets import AlgorithmPanel, ExplanationPane, FlowchartView

APP_NAME = "PSG Interpreter"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
SEVERITY_COLORS = {"Normal": "#2e7d32", "Mild": "#f9a825", "Moderate": "#ef6c00", "Severe": "#c62828"}

# Display colours per event code (fill with alpha).
EVENT_COLORS = {
    "AO": (31, 119, 180), "AC": (44, 160, 44), "AM": (148, 103, 189),
    "HO": (255, 127, 14), "HC": (140, 86, 75),
}
EVENT_NAMES = {"AO": "Obstructive apnea", "AC": "Central apnea", "AM": "Mixed apnea",
               "HO": "Obstructive hypopnea", "HC": "Central hypopnea"}
STAGE_Y = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0, -1: np.nan}  # clinical hypnogram order: W R N1 N2 N3

# AASM-recommended display filters per role (low, high) in Hz; None = raw.
DISPLAY_FILTERS = {
    "eeg": (0.3, 35.0), "eeg2": (0.3, 35.0), "eog_l": (0.3, 35.0), "eog_r": (0.3, 35.0),
    "emg": (10.0, 100.0), "ecg": (0.3, 70.0), "snore": (10.0, 100.0),
    "flow": (None, 15.0), "thorax": (0.1, 15.0), "abdomen": (0.1, 15.0), "effort_sum": (0.1, 15.0),
}
DEFAULT_ORDER = ["eog_l", "eog_r", "eeg", "eeg2", "emg", "ecg", "flow", "thorax", "abdomen",
                 "effort_sum", "snore", "spo2", "pulse", "position"]
RESP_ROLES = {"flow", "thorax", "abdomen", "effort_sum", "spo2"}


def clock_str(rec: Recording | None, sec: float) -> str:
    import datetime as dt
    if rec is not None and rec.start is not None:
        return (rec.start + dt.timedelta(seconds=float(sec))).strftime("%H:%M:%S")
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ============================================================================ background work

class Worker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.finished.emit(self.fn(*self.args, progress=self.progress.emit))
        except StudyError as e:          # problem with the uploaded file: plain-language message
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit("Unexpected error while processing the study.\n\n" + traceback.format_exc())


def _process(path, options, progress=None):
    """Worker-thread job for one upload: full analysis + saved report + display copies."""
    from analyze import validation_summary
    study = process_study(path, options, REPORTS_DIR, progress, validation_fn=validation_summary)
    return study, _display_copies(study.rec)


def _display_copies(rec: Recording) -> dict:
    """Pre-filter display copies once (AASM display filters) so paging is instant."""
    display = {}
    for label, ch in rec.channels.items():
        lohi = DISPLAY_FILTERS.get(ch.role or "")
        if lohi and ch.fs > 2 * (lohi[0] or 0) + 1:
            lo, hi = lohi
            hi = min(hi, 0.45 * ch.fs) if hi else None
            try:
                display[label] = pp.filt(ch.data, ch.fs, lo, hi, order=2).astype(np.float32)
            except Exception:
                display[label] = ch.data
        else:
            display[label] = ch.data
    return display


# ============================================================================ widgets

class TimeAxis(pg.AxisItem):
    """X axis that shows wall-clock time."""
    def __init__(self, get_rec, **kw):
        super().__init__(orientation="bottom", **kw)
        self.get_rec = get_rec

    def tickStrings(self, values, scale, spacing):
        return [clock_str(self.get_rec(), v) for v in values]


class Overview(pg.GraphicsLayoutWidget):
    """Whole-night navigation strip: hypnogram, events, SpO2."""
    seek = QtCore.pyqtSignal(float)  # centre time requested

    def __init__(self, get_rec):
        super().__init__()
        self.setBackground("w")
        self.setFixedHeight(250)
        self.hyp = self.addPlot(row=0, col=0, axisItems={"bottom": TimeAxis(get_rec)})
        self.hyp.getAxis("left").setTicks([[(4, "W"), (3, "R"), (2, "N1"), (1, "N2"), (0, "N3")]])
        self.hyp.setYRange(-0.5, 4.5, padding=0)
        self.hyp.setMouseEnabled(x=False, y=False)
        self.hyp.hideButtons()
        self.hyp.getAxis("left").setWidth(48)
        self.ev = self.addPlot(row=1, col=0)
        self.ev.setXLink(self.hyp)
        self.ev.hideAxis("bottom")
        self.ev.getAxis("left").setTicks([[(1.5, "Auto"), (0.5, "Expert")]])
        self.ev.getAxis("left").setWidth(48)
        self.ev.setYRange(0, 2, padding=0)
        self.ev.setMouseEnabled(x=False, y=False)
        self.ev.hideButtons()
        self.sp = self.addPlot(row=2, col=0)
        self.sp.setXLink(self.hyp)
        self.sp.hideAxis("bottom")
        self.sp.getAxis("left").setWidth(48)
        self.sp.setLabel("left", "SpO2")
        self.sp.setMouseEnabled(x=False, y=False)
        self.sp.hideButtons()
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.ci.layout.setRowStretchFactor(2, 2)
        self.region = pg.LinearRegionItem(brush=(255, 200, 0, 70), movable=True)
        self.region.setZValue(10)
        self.hyp.addItem(self.region)
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        for p in (self.hyp, self.ev, self.sp):
            p.scene().sigMouseClicked.connect(self._clicked)
        self._items = []
        self._block = False

    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        for p in (self.hyp, self.ev, self.sp):
            if p.sceneBoundingRect().contains(ev.scenePos()):
                x = p.vb.mapSceneToView(ev.scenePos()).x()
                self.seek.emit(float(x))
                return

    def _region_moved(self):
        if self._block:
            return
        a, b = self.region.getRegion()
        self.seek.emit((a + b) / 2)

    def set_window(self, t0, t1):
        self._block = True
        self.region.setRegion((t0, t1))
        self._block = False

    def clear_all(self):
        for p, it in self._items:
            p.removeItem(it)
        self._items = []

    def _add(self, plot, item):
        plot.addItem(item)
        self._items.append((plot, item))

    def show_recording(self, rec: Recording, stages=None, spo2=None, spo2_fs=None):
        self.clear_all()
        dur = rec.duration
        self.hyp.setXRange(0, dur, padding=0)
        st = stages if stages is not None else rec.expert_stages
        if st is not None:
            t = np.arange(len(st) + 1) * EPOCH_S
            y = np.array([STAGE_Y[int(s)] for s in st] + [STAGE_Y[int(st[-1])]], dtype=float)
            self._add(self.hyp, pg.PlotDataItem(t, y, stepMode="left",
                                                pen=pg.mkPen("#333", width=1)))
            rem = np.flatnonzero(np.asarray(st) == 4)
            if rem.size:
                xs = np.repeat(rem * EPOCH_S, 2) + np.tile([0, EPOCH_S], rem.size)
                ys = np.full(xs.size, 3.0)
                conn = np.tile([1, 0], rem.size).astype(bool)
                self._add(self.hyp, pg.PlotDataItem(xs, ys, connect=conn, pen=pg.mkPen("#d62728", width=4)))
        if spo2 is None:
            ch = rec.get("spo2")
            if ch is not None:
                spo2, spo2_fs = ch.data, ch.fs
        if spo2 is not None:
            step = max(int(spo2_fs), 1)
            s = np.asarray(spo2[::step], dtype=float)
            s = np.where((s > 50) & (s <= 100), s, np.nan)
            self._add(self.sp, pg.PlotDataItem(np.arange(len(s)), s, pen=pg.mkPen("#1f77b4", width=1), connect="finite"))
            lo = np.nanpercentile(s, 0.5) if np.isfinite(s).any() else 80
            self.sp.setYRange(max(min(lo - 2, 88), 50), 100, padding=0)
        self.set_window(0, 30)

    def show_events(self, events, expert=None):
        for row, evs, getcode in ((1, events, lambda e: e.code),
                                  (0, expert or [], lambda e: ("A" if e.kind == "apnea" else "H") + (e.subtype[0].upper() if e.subtype != "unknown" else "O"))):
            by: dict[str, list] = {}
            for e in evs:
                by.setdefault(getcode(e), []).append(e)
            for code, lst in by.items():
                xs = np.array([[e.onset, e.onset] for e in lst]).ravel()
                ys = np.tile([row + 0.1, row + 0.9], len(lst))
                conn = np.tile([1, 0], len(lst)).astype(bool)
                self._add(self.ev, pg.PlotDataItem(xs, ys, connect=conn, pen=pg.mkPen(EVENT_COLORS.get(code, (0, 0, 0)), width=1)))


class SignalView(QtWidgets.QScrollArea):
    """Stacked per-channel plots sharing a time axis."""
    def __init__(self, get_rec):
        super().__init__()
        self.setWidgetResizable(True)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.setWidget(self.glw)
        self.get_rec = get_rec
        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.overlays: list[tuple[pg.PlotItem, object]] = []
        self.gain = 1.0
        self.row_height = 70

    def build(self, rec: Recording, labels: list[str]):
        self.glw.clear()
        self.plots.clear()
        self.curves.clear()
        self.overlays.clear()
        first = None
        for i, label in enumerate(labels):
            ch = rec.channels[label]
            last = i == len(labels) - 1
            axis = {"bottom": TimeAxis(self.get_rec)} if last else {}
            p = self.glw.addPlot(row=i, col=0, axisItems=axis)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setMouseEnabled(x=False, y=True)
            p.showGrid(x=True, y=False, alpha=0.3)
            if not last:
                p.hideAxis("bottom")
            ax = p.getAxis("left")
            ax.setWidth(80)
            ax.setStyle(showValues=False)
            ax.enableAutoSIPrefix(False)
            p.setLabel("left", f"<span style='font-size:8pt'>{label}</span>")
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            c = p.plot(pen=pg.mkPen("#222", width=1))
            self.plots[label] = p
            self.curves[label] = c
            if first is None:
                first = p
            else:
                p.setXLink(first)
        self.glw.setMinimumHeight(self.row_height * max(len(labels), 1) + 30)

    def draw(self, rec: Recording, display: dict, t0: float, t1: float, result: AnalysisResult | None,
             show_expert: bool, show_derived: bool, show_rejected: bool = False):
        for p, it in self.overlays:
            p.removeItem(it)
        self.overlays.clear()
        for label, p in self.plots.items():
            ch = rec.channels[label]
            i0, i1 = max(int(t0 * ch.fs), 0), min(int(t1 * ch.fs) + 1, len(ch.data))
            y = display[label][i0:i1]
            x = np.arange(i0, i1) / ch.fs
            self.curves[label].setData(x, y)
            p.setXRange(t0, t1, padding=0)
            if y.size:
                if ch.role == "spo2":
                    fin = y[np.isfinite(y) & (y > 50)]
                    lo = min(float(fin.min()) - 2, 88) if fin.size else 80
                    p.setYRange(max(lo, 50), 100, padding=0)
                elif ch.role == "position":
                    p.setYRange(0, 5, padding=0)
                else:
                    med = float(np.nanmedian(y))
                    spread = float(np.nanpercentile(np.abs(y - med), 99)) or 1.0
                    spread /= self.gain
                    p.setYRange(med - spread, med + spread, padding=0.05)
        if result is None and not (show_expert and rec.expert_events):
            return
        # Event shading on respiratory channels, arousal marks on EEG.
        resp_labels = [l for l, p in self.plots.items() if rec.channels[l].role in RESP_ROLES]
        eeg_labels = [l for l, p in self.plots.items() if rec.channels[l].role in ("eeg", "eeg2")]
        if result is not None:
            for e in result.events:
                if e.end < t0 or e.onset > t1:
                    continue
                col = EVENT_COLORS[e.code]
                for l in resp_labels:
                    r = pg.LinearRegionItem((e.onset, e.end), movable=False, brush=(*col, 60), pen=pg.mkPen((*col, 160)))
                    r.setZValue(-10)
                    self._add(self.plots[l], r)
                if resp_labels:
                    p = self.plots[resp_labels[0]]
                    txt = pg.TextItem(f"{e.code}  {e.duration:.0f}s" + (f"  -{e.desat:.0f}%" if e.desat else "")
                                      + f"  ·{e.confidence * 100:.0f}%", color=col, anchor=(0, 0))
                    txt.setPos(max(e.onset, t0), p.vb.viewRange()[1][1])
                    self._add(p, txt)
            if show_rejected:
                # Candidates that failed the rule (or fell under the confidence cut-off): grey, with the reason.
                shown = set(map(id, result.events))
                for e in result.resp.candidates:
                    if id(e) in shown or e.end < t0 or e.onset > t1:
                        continue
                    for l in resp_labels:
                        r = pg.LinearRegionItem((e.onset, e.end), movable=False, brush=(120, 120, 120, 35),
                                                pen=pg.mkPen((120, 120, 120, 140), style=QtCore.Qt.DashLine))
                        r.setZValue(-10)
                        self._add(self.plots[l], r)
                    if resp_labels:
                        p = self.plots[resp_labels[0]]
                        why = e.reject_reason if not e.accepted else f"below confidence cut-off ({e.confidence * 100:.0f}%)"
                        txt = pg.TextItem(f"not scored: {why}", color=(110, 110, 110), anchor=(0, 0))
                        txt.setPos(max(e.onset, t0), p.vb.viewRange()[1][1])
                        self._add(p, txt)
            for (a, d) in result.arousals:
                if a + d < t0 or a > t1:
                    continue
                for l in eeg_labels:
                    r = pg.LinearRegionItem((a, a + d), movable=False, brush=(214, 39, 40, 40), pen=pg.mkPen((214, 39, 40, 120)))
                    r.setZValue(-10)
                    self._add(self.plots[l], r)
            if show_derived and rec.roles.get("flow") in self.plots and result.resp.fs > 1:
                # Breath amplitude envelope and the 30 % / 90 % reduction thresholds.
                p = self.plots[rec.roles["flow"]]
                fs = result.resp.fs
                i0, i1 = max(int(t0 * fs), 0), min(int(t1 * fs) + 1, len(result.resp.flow_env))
                x = np.arange(i0, i1) / fs
                base = result.resp.flow_baseline[i0:i1] / 2
                env = (result.resp.flow_env[i0:i1] + result.resp.noise_floor) / 2
                for yy, pen in ((env, pg.mkPen((0, 150, 0), width=1.5)),
                                (base, pg.mkPen((0, 0, 200), width=1, style=QtCore.Qt.DashLine)),
                                (base * 0.7, pg.mkPen((255, 127, 14), width=1, style=QtCore.Qt.DotLine)),
                                (base * 0.1, pg.mkPen((200, 0, 0), width=1, style=QtCore.Qt.DotLine))):
                    self._add(p, pg.PlotDataItem(x, yy, pen=pen))
        if show_expert and rec.expert_events and resp_labels:
            p = self.plots[resp_labels[0]]
            ymin, ymax = p.vb.viewRange()[1]
            for e in rec.expert_events:
                if e.end < t0 or e.onset > t1:
                    continue
                code = ("A" if e.kind == "apnea" else "H") + (e.subtype[0].upper() if e.subtype != "unknown" else "O")
                col = EVENT_COLORS.get(code, (0, 0, 0))
                y = ymin + 0.04 * (ymax - ymin)
                bar = pg.PlotDataItem([e.onset, e.end], [y, y], pen=pg.mkPen(col, width=5))
                self._add(p, bar)
                t = pg.TextItem(f"expert {code}", color=col, anchor=(0, 1))
                t.setPos(e.onset, y)
                self._add(p, t)

    def _add(self, plot, item):
        plot.addItem(item)
        self.overlays.append((plot, item))


# ============================================================================ upload screens

def _local_files(mime: QtCore.QMimeData) -> list[str]:
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()] if mime.hasUrls() else []


class DropZone(QtWidgets.QFrame):
    """Large drag-and-drop target that also opens a file dialog when clicked."""
    files_dropped = QtCore.pyqtSignal(list)
    clicked = QtCore.pyqtSignal()

    _STYLE = ("DropZone{border:2px dashed %s;border-radius:14px;background:%s}"
              "QLabel{background:transparent;border:none}")

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setMinimumHeight(230)
        v = QtWidgets.QVBoxLayout(self)
        v.setAlignment(QtCore.Qt.AlignCenter)
        icon = QtWidgets.QLabel("⬆")
        icon.setAlignment(QtCore.Qt.AlignCenter)
        icon.setStyleSheet("font-size:44px;color:#1f77b4")
        title = QtWidgets.QLabel("Drop the PSG recording here")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:20px;font-weight:600;color:#222")
        sub = QtWidgets.QLabel("or click to choose a file  ·  EDF / EDF+ / .rec  ·  several files at once are fine")
        sub.setAlignment(QtCore.Qt.AlignCenter)
        sub.setStyleSheet("font-size:12px;color:#666")
        btn = QtWidgets.QPushButton("  Upload recording…  ")
        btn.setStyleSheet("font-size:14px;padding:8px 18px;background:#1f77b4;color:white;border-radius:6px")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.clicked.connect(self.clicked.emit)
        for w in (icon, title, sub):
            v.addWidget(w)
        v.addSpacing(10)
        v.addWidget(btn, 0, QtCore.Qt.AlignCenter)
        self._highlight(False)

    def _highlight(self, on: bool):
        self.setStyleSheet(self._STYLE % (("#1f77b4", "#eaf3fb") if on else ("#9bb8d3", "#f7fafd")))

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, ev):
        if _local_files(ev.mimeData()):
            ev.acceptProposedAction()
            self._highlight(True)

    def dragLeaveEvent(self, ev):
        self._highlight(False)

    def dropEvent(self, ev):
        self._highlight(False)
        files = _local_files(ev.mimeData())
        if files:
            ev.acceptProposedAction()
            self.files_dropped.emit(files)


class HomePage(QtWidgets.QWidget):
    """Start screen: upload + previously processed studies."""
    open_report = QtCore.pyqtSignal(str)
    reopen = QtCore.pyqtSignal(str)

    COLS = ["Recording", "Recorded", "Processed", "Diagnosis", "AHI /h", "Severity", "Scoring rule"]

    def __init__(self):
        super().__init__()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(40, 24, 40, 24)
        title = QtWidgets.QLabel("PSG Interpreter")
        title.setStyleSheet("font-size:26px;font-weight:700;color:#1b2a3a")
        sub = QtWidgets.QLabel("Upload an overnight polysomnography recording. Signal checks, cleaning, sleep staging, "
                               "AASM respiratory scoring, oximetry and the report are done automatically.")
        sub.setWordWrap(True)
        sub.setStyleSheet("font-size:13px;color:#555")
        outer.addWidget(title)
        outer.addWidget(sub)
        outer.addSpacing(12)
        self.drop = DropZone()
        outer.addWidget(self.drop)
        self.rule_note = QtWidgets.QLabel()
        self.rule_note.setStyleSheet("color:#666;font-size:12px")
        outer.addWidget(self.rule_note)
        outer.addSpacing(14)
        hdr = QtWidgets.QHBoxLayout()
        h = QtWidgets.QLabel("Previous studies")
        h.setStyleSheet("font-size:16px;font-weight:600")
        hdr.addWidget(h)
        hdr.addStretch(1)
        self.btn_report = QtWidgets.QPushButton("Open report")
        self.btn_review = QtWidgets.QPushButton("Review signals")
        self.btn_review.setToolTip("Re-opens the recording in the viewer (re-analyses it, about 30 s)")
        self.btn_folder = QtWidgets.QPushButton("Reports folder")
        for b in (self.btn_report, self.btn_review, self.btn_folder):
            hdr.addWidget(b)
        outer.addLayout(hdr)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.table.doubleClicked.connect(lambda *_: self._emit(self.open_report, "report"))
        outer.addWidget(self.table, 1)
        self.empty = QtWidgets.QLabel("No studies yet - upload a recording to start.")
        self.empty.setStyleSheet("color:#888;padding:8px")
        outer.addWidget(self.empty)
        self.btn_report.clicked.connect(lambda: self._emit(self.open_report, "report"))
        self.btn_review.clicked.connect(lambda: self._emit(self.reopen, "source"))
        self.btn_folder.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(REPORTS_DIR))))
        self.entries: list[dict] = []

    def _emit(self, sig, key):
        r = self.table.currentRow()
        if 0 <= r < len(self.entries):
            sig.emit(self.entries[r][key])

    def refresh(self):
        self.entries = list_studies(REPORTS_DIR)
        self.table.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            vals = [e["name"], e.get("recorded", "")[:16], e.get("processed", "").replace("T", " ")[:16],
                    e.get("diagnosis", ""), "" if e.get("ahi") is None else f"{e['ahi']:.1f}", e.get("severity", ""),
                    e.get("rule", e.get("hypopnea_rule", ""))]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 5 and v in SEVERITY_COLORS:
                    it.setForeground(QtGui.QColor(SEVERITY_COLORS[v]))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        has = bool(self.entries)
        self.empty.setVisible(not has)
        for b in (self.btn_report, self.btn_review):
            b.setEnabled(has)
        if has:
            self.table.selectRow(0)


class ProcessingPage(QtWidgets.QWidget):
    """Live checklist while a study is analysed."""
    STEPS = [
        ("Checking the file", ("checking", "reading")),
        ("Cleaning signals (artifacts, gaps, filters)", ("cleaning",)),
        ("Sleep staging (30-s epochs)", ("staging",)),
        ("EEG arousal detection", ("arousal",)),
        ("Apneas and hypopneas (AASM rules)", ("apnea", "hypopnea")),
        ("Oxygen desaturations and indices", ("desaturation", "indices")),
        ("Writing the report", ("report", "done")),
    ]

    def __init__(self):
        super().__init__()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setAlignment(QtCore.Qt.AlignCenter)
        box = QtWidgets.QFrame()
        box.setMaximumWidth(620)
        box.setStyleSheet("QFrame{background:white;border:1px solid #dde3ea;border-radius:12px}"
                          "QLabel{border:none;background:transparent}")
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(30, 24, 30, 24)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size:18px;font-weight:600")
        self.subtitle = QtWidgets.QLabel()
        self.subtitle.setStyleSheet("color:#666")
        v.addWidget(self.title)
        v.addWidget(self.subtitle)
        v.addSpacing(10)
        self.rows = []
        for name, _ in self.STEPS:
            lab = QtWidgets.QLabel()
            lab.setStyleSheet("font-size:14px;padding:3px")
            v.addWidget(lab)
            self.rows.append((name, lab))
        v.addSpacing(10)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(True)
        v.addWidget(self.bar)
        self.foot = QtWidgets.QLabel()
        self.foot.setStyleSheet("color:#888")
        v.addWidget(self.foot)
        outer.addWidget(box, 0, QtCore.Qt.AlignCenter)
        self._t = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._queued = 0

    def start(self, path: str, queued: int):
        self.title.setText(f"Analysing {Path(path).name}")
        self.subtitle.setText("You can drop more recordings - they will be processed next.")
        self._queued = queued
        self._current = 0
        self.bar.setValue(0)
        self._render()
        self._t.start()
        self._timer.start(500)
        self._tick()

    def stop(self):
        self._timer.stop()

    def set_queued(self, n: int):
        self._queued = n
        self._tick()

    def _tick(self):
        q = f"  ·  {self._queued} more in queue" if self._queued else ""
        self.foot.setText(f"Elapsed {self._t.elapsed() / 1000:.0f} s{q}")

    def set_progress(self, pct: int, msg: str):
        m = msg.lower()
        for i, (_, keys) in enumerate(self.STEPS):
            if any(k in m for k in keys):
                self._current = max(self._current, i)
                break
        if m.startswith("done"):
            self._current = len(self.STEPS)
        self.bar.setValue(pct)
        self._render()

    def _render(self):
        for i, (name, lab) in enumerate(self.rows):
            if i < self._current:
                lab.setText(f"<span style='color:#2e7d32'>✔</span>&nbsp;&nbsp;{name}")
            elif i == self._current:
                lab.setText(f"<span style='color:#1f77b4'>●</span>&nbsp;&nbsp;<b>{name}…</b>")
            else:
                lab.setText(f"<span style='color:#bbb'>○</span>&nbsp;&nbsp;<span style='color:#999'>{name}</span>")


# ============================================================================ main window

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)
        self.rec: Recording | None = None
        self.display: dict = {}
        self.result: AnalysisResult | None = None
        self.t0 = 0.0
        self.page = 30.0
        self._thread = None
        self.study = None
        self._queue: list[str] = []
        self.setAcceptDrops(True)
        self._build_ui()
        self.show_home()
        self._update_actions()

    # --------------------------------------------------------------------- UI construction
    def _build_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        st = self.style()
        self.act_home = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DirHomeIcon), "Home", self.show_home)
        self.act_open = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton), "Upload…", self.open_dialog)
        self.act_open.setShortcut("Ctrl+O")
        self.act_report = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView), "Open report",
                                       self.open_saved_report)
        self.act_analyze = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_BrowserReload), "Re-analyze", self.run_analysis)
        self.act_analyze.setToolTip("Run the full analysis again (needed after changing the staging source; "
                                    "rule and threshold changes re-score instantly)")
        self.act_analyze.setShortcut("F5")
        self.act_export = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Save copy…", self.export)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("  Staging: "))
        self.staging = QtWidgets.QComboBox()
        self.staging.addItems(["Automatic", "Expert file (if present)"])
        self.staging.setToolTip("Source of the hypnogram. Changing it needs a full re-analysis (F5).")
        tb.addWidget(self.staging)
        self.staging.currentIndexChanged.connect(self._update_rule_note)
        tb.addSeparator()
        self.page_label = QtWidgets.QLabel(" Page: ")
        tb.addWidget(self.page_label)
        self.page_box = QtWidgets.QComboBox()
        for s in ["10 s", "30 s", "60 s", "120 s", "5 min", "10 min"]:
            self.page_box.addItem(s)
        self.page_box.setCurrentIndex(1)
        self.page_box.currentIndexChanged.connect(self._page_changed)
        tb.addWidget(self.page_box)
        self.act_prev = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward), "", lambda: self.step(-1))
        self.act_next = tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_MediaSeekForward), "", lambda: self.step(1))
        self.chk_expert = QtWidgets.QCheckBox("Show expert events")
        self.chk_expert.setChecked(True)
        self.chk_expert.toggled.connect(self.redraw)
        tb.addWidget(self.chk_expert)
        self.chk_derived = QtWidgets.QCheckBox("Show flow envelope")
        self.chk_derived.setToolTip("Breath amplitude (green), pre-event baseline (blue), 30 % hypopnea and 90 % apnea thresholds")
        self.chk_derived.toggled.connect(self.redraw)
        tb.addWidget(self.chk_derived)

        # Central: stacked screens - home (upload), processing, results viewer.
        self.stack = QtWidgets.QStackedWidget()
        self.home = HomePage()
        self.home.drop.files_dropped.connect(self.enqueue)
        self.home.drop.clicked.connect(self.open_dialog)
        self.home.open_report.connect(self._open_report_path)
        self.home.reopen.connect(lambda p: self.enqueue([p]))
        self.processing = ProcessingPage()
        viewer = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(viewer)
        v.setContentsMargins(4, 4, 4, 4)
        self.overview = Overview(lambda: self.rec)
        self.overview.seek.connect(self.center_on)
        v.addWidget(self.overview)
        self.pos_label = QtWidgets.QLabel("")
        self.pos_label.setStyleSheet("color:#444;padding:2px")
        v.addWidget(self.pos_label)
        self.view = SignalView(lambda: self.rec)
        v.addWidget(self.view, 1)
        for w in (self.home, self.processing, viewer):
            self.stack.addWidget(w)
        self.viewer = viewer
        self.setCentralWidget(self.stack)

        # Right: tabs.
        dock = QtWidgets.QDockWidget("Analysis", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.tabs = QtWidgets.QTabWidget()
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(460)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.dock = dock

        # Left: algorithm (published rule version + threshold sliders).
        self.algo = AlgorithmPanel()
        self.algo.params_changed.connect(self._params_changed)
        algo_dock = QtWidgets.QDockWidget("Algorithm", self)
        algo_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        algo_dock.setWidget(self.algo)
        algo_dock.setMinimumWidth(340)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, algo_dock)
        self.algo_dock = algo_dock

        # Channels tab.
        w = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(w)
        self.ch_list = QtWidgets.QListWidget()
        self.ch_list.itemChanged.connect(self._channels_changed)
        lv.addWidget(QtWidgets.QLabel("Tick channels to display (role detected automatically):"))
        lv.addWidget(self.ch_list, 1)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Defaults", self._select_default), ("All", lambda: self._select_all(True)),
                         ("None", lambda: self._select_all(False)), ("Respiratory", self._select_resp)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lv.addLayout(row)
        self.tabs.addTab(w, "Channels")

        # Events tab: confidence cut-off, filter, table, explanation of the selected event.
        w = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(w)
        conf_row = QtWidgets.QHBoxLayout()
        conf_row.addWidget(QtWidgets.QLabel("Confidence ≥"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(0, 99)
        self.conf_slider.setValue(0)
        self.conf_slider.setToolTip("Only events at or above this confidence are counted in the AHI and shown")
        self.conf_slider.valueChanged.connect(self._conf_changed)
        conf_row.addWidget(self.conf_slider, 1)
        self.conf_label = QtWidgets.QLabel("0 %")
        self.conf_label.setMinimumWidth(150)
        conf_row.addWidget(self.conf_label)
        lv.addLayout(conf_row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Show:"))
        self.ev_filter = QtWidgets.QComboBox()
        self.ev_filter.addItems(["All events", "Apneas", "Hypopneas"] + list(EVENT_NAMES.values()))
        self.ev_filter.currentIndexChanged.connect(self._fill_events)
        row.addWidget(self.ev_filter, 1)
        self.chk_rejected = QtWidgets.QCheckBox("Rejected candidates")
        self.chk_rejected.setToolTip("Also list breathing reductions that did NOT meet the rule, with the reason")
        self.chk_rejected.toggled.connect(self._fill_events)
        self.chk_rejected.toggled.connect(self.redraw)
        row.addWidget(self.chk_rejected)
        lv.addLayout(row)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.ev_table = QtWidgets.QTableWidget(0, 9)
        self.ev_table.setHorizontalHeaderLabels(["Time", "Dur s", "Type", "Conf %", "Flow ↓%", "Desat %", "Nadir", "Stage", "Notes"])
        self.ev_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ev_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ev_table.verticalHeader().setVisible(False)
        self.ev_table.horizontalHeader().setStretchLastSection(True)
        self.ev_table.setWordWrap(False)
        self.ev_table.setSortingEnabled(False)
        self.ev_table.cellDoubleClicked.connect(self._event_clicked)
        self.ev_table.cellClicked.connect(self._event_clicked)
        split.addWidget(self.ev_table)
        self.explain = ExplanationPane()
        split.addWidget(self.explain)
        split.setSizes([420, 260])
        lv.addWidget(split, 1)
        legend = QtWidgets.QLabel(" ".join(
            f"<span style='background:rgb{c};color:white;padding:1px 4px'>{k}</span> {EVENT_NAMES[k]}&nbsp;&nbsp;"
            for k, c in EVENT_COLORS.items()))
        legend.setWordWrap(True)
        lv.addWidget(legend)
        self.tabs.addTab(w, "Events")

        # Report tab.
        self.report = QtWidgets.QTextBrowser()
        self.report.setOpenExternalLinks(True)
        self.report.setHtml("<p style='color:#666'>Run <b>Analyze</b> to see the interpretation.</p>")
        self.tabs.addTab(self.report, "Report")

        # Flowchart tab: the decision steps under the current parameters; the selected event's path is highlighted.
        self.flow = FlowchartView()
        self.flow.set_params(self.algo.params())
        self.tabs.addTab(self.flow, "Flowchart")

        # Status bar.
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        for key, fn in (("Right", lambda: self.step(1)), ("Left", lambda: self.step(-1)),
                        ("PgDown", lambda: self.step(1)), ("PgUp", lambda: self.step(-1)),
                        ("N", lambda: self.jump_event(1)), ("P", lambda: self.jump_event(-1)),
                        ("+", lambda: self._gain(1.25)), ("=", lambda: self._gain(1.25)), ("-", lambda: self._gain(0.8))):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(fn)

    def _update_actions(self):
        has = self.rec is not None
        busy = self._thread is not None
        in_viewer = self.stack.currentWidget() is self.viewer
        self.act_analyze.setEnabled(has and not busy)
        self.act_export.setEnabled(self.result is not None and not busy)
        self.act_report.setEnabled(self.study is not None)
        self.act_home.setEnabled(not busy)
        for w in (self.page_box, self.chk_expert, self.chk_derived, self.page_label):
            w.setEnabled(in_viewer)
        self.act_prev.setEnabled(in_viewer)
        self.act_next.setEnabled(in_viewer)

    # --------------------------------------------------------------------- screens
    def show_home(self):
        self.home.refresh()
        self._update_rule_note()
        self.stack.setCurrentWidget(self.home)
        self.dock.hide()
        self.algo_dock.show()
        self._update_actions()

    def show_viewer(self):
        self.stack.setCurrentWidget(self.viewer)
        self.dock.show()
        self.algo_dock.show()
        self._update_actions()

    def _update_rule_note(self):
        r = self.algo.rule()
        staging = "automatic sleep staging" if self.staging.currentIndex() == 0 else "technician hypnogram when available"
        self.home.rule_note.setText(f"Scoring rule: {r.name} — {r.short()}; {staging}. "
                                    f"Choose the rule in the Algorithm panel; thresholds can be changed after the analysis too.")

    # --------------------------------------------------------------------- live re-scoring
    def _params_changed(self, params):
        """Rule version or a slider changed: re-score the loaded night with the new thresholds."""
        self.flow.set_params(params)
        self._update_rule_note()
        if self.result is None or self._thread is not None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.result = rescore_analysis(self.result, params, self.conf_slider.value() / 100.0)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_results(f"Re-scored with {self.result.rule_name} in {self.result.runtime_s:.1f} s")

    def _conf_changed(self, value):
        self.conf_label.setText(f"{value} %")
        if self.result is None or self._thread is not None:
            return
        self.result = rescore_analysis(self.result, None, value / 100.0)
        self._refresh_results(f"Confidence cut-off {value} %")

    def _refresh_results(self, status: str = ""):
        res = self.result
        self.overview.show_events(res.events, self.rec.expert_events)
        self._fill_events()
        self.report.setHtml(self._report_html(res))
        cc = res.summary.get("confidence_counts", {})
        self.conf_label.setText(f"{self.conf_slider.value()} %  ·  {len(res.events)} of {res.summary.get('n_scored_all', 0)} events")
        self.algo.set_status(f"{res.diagnosis['primary']}\nEvents at ≥90 % / ≥75 % / ≥50 % confidence: "
                             f"{cc.get(0.9, 0)} / {cc.get(0.75, 0)} / {cc.get(0.5, 0)}; "
                             f"{res.summary.get('n_rejected', 0)} candidates rejected.")
        self.redraw()
        if status:
            self.statusBar().showMessage(f"{res.diagnosis['primary']}   —   {status}")

    # --------------------------------------------------------------------- upload queue
    def dragEnterEvent(self, ev):
        if _local_files(ev.mimeData()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        files = _local_files(ev.mimeData())
        if files:
            ev.acceptProposedAction()
            self.enqueue(files)

    def enqueue(self, paths):
        good, bad = [], []
        for p in paths:
            (good if Path(p).suffix.lower() in ACCEPTED_SUFFIXES else bad).append(p)
        if bad:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME, "These files are not PSG recordings and were skipped:\n\n"
                + "\n".join(Path(b).name for b in bad) + "\n\nPlease upload EDF files (.edf, .rec, .bdf).")
        # Hypnogram files sometimes get dropped together with the recording: skip them quietly.
        good = [p for p in good if not Path(p).name.lower().endswith("-hypnogram.edf")]
        self._queue.extend(good)
        if self._thread is None:
            self._next_in_queue()
        else:
            self.processing.set_queued(len(self._queue))

    def _next_in_queue(self):
        if not self._queue:
            return
        path = self._queue.pop(0)
        self.processing.start(path, len(self._queue))
        self.stack.setCurrentWidget(self.processing)
        self.dock.hide()
        self._run(_process, (path, self._options()), self._study_done, f"Analysing {Path(path).name}…",
                  on_fail=self._study_failed)

    def _study_done(self, obj):
        self.processing.stop()
        study, display = obj
        self.study = study
        self._show_recording(study.rec, display)
        self._analyzed(study.result)
        self.home.refresh()
        self.show_viewer()
        if study.result.events:  # open on the first scored event, with respiratory channels, 60-s page
            self._select_resp()
            self.page_box.setCurrentIndex(2)
            first = study.result.events[0]
            self.center_on(first.onset + first.duration / 2)
            self._select_row_for(first)
        self.statusBar().showMessage(f"{study.result.diagnosis['primary']}   —   report saved to {study.report_path}")
        if self._queue:
            QtCore.QTimer.singleShot(0, self._next_in_queue)

    def _study_failed(self, msg):
        self.processing.stop()
        if self._queue:
            QtCore.QTimer.singleShot(0, self._next_in_queue)
        elif self.rec is not None:
            self.show_viewer()
        else:
            self.show_home()

    def open_saved_report(self):
        if self.study is not None:
            self._open_report_path(str(self.study.report_path))

    def _open_report_path(self, path):
        if Path(path).exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        else:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Report not found:\n{path}")

    def closeEvent(self, ev):
        if self._thread is not None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "An analysis is still running - please wait for it to finish.")
            ev.ignore()
            return
        super().closeEvent(ev)

    # --------------------------------------------------------------------- threading helper
    def _run(self, fn, args, on_done, label, on_fail=None):
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(label)
        th = QtCore.QThread(self)
        wk = Worker(fn, *args)
        wk.moveToThread(th)
        th.started.connect(wk.run)
        wk.progress.connect(lambda pct, msg: (self.progress.setValue(pct), self.statusBar().showMessage(msg),
                                              self.processing.set_progress(pct, msg)))

        def done(obj):
            th.quit()
            th.wait()
            self._thread = None
            self.progress.setVisible(False)
            on_done(obj)
            self._update_actions()

        def fail(msg):
            th.quit()
            th.wait()
            self._thread = None
            self.progress.setVisible(False)
            self._update_actions()
            self.statusBar().showMessage("Could not process the recording")
            QtWidgets.QMessageBox.critical(self, APP_NAME, msg[-3000:])
            if on_fail is not None:
                on_fail(msg)

        wk.finished.connect(done)
        wk.failed.connect(fail)
        self._thread = (th, wk)
        self._update_actions()
        th.start()

    # --------------------------------------------------------------------- open
    def open_dialog(self):
        start = str(Path(__file__).parent / "data")
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Upload PSG recording(s)", start,
                                                          "PSG files (*.edf *.rec *.EDF *.bdf);;All files (*)")
        if paths:
            self.enqueue(paths)

    def open_file(self, path):
        self.enqueue([path])

    def _show_recording(self, rec: Recording, display: dict):
        self.rec, self.display = rec, display
        self.result = None
        self.setWindowTitle(f"{APP_NAME} — {rec.name}")
        self.ch_list.blockSignals(True)
        self.ch_list.clear()
        role_of = {v: k for k, v in rec.roles.items()}
        for label, ch in rec.channels.items():
            role = role_of.get(label)
            it = QtWidgets.QListWidgetItem(f"{label}   ({ch.fs:g} Hz{', ' + role if role else ''})")
            it.setData(QtCore.Qt.UserRole, label)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self.ch_list.addItem(it)
        self.ch_list.blockSignals(False)
        self._select_default()
        self.ev_table.setRowCount(0)
        self.t0 = 0.0

    # --------------------------------------------------------------------- channels
    def _checked_labels(self):
        out = []
        for i in range(self.ch_list.count()):
            it = self.ch_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                out.append(it.data(QtCore.Qt.UserRole))
        # Order by clinical montage order, then original order.
        order = {self.rec.roles[r]: k for k, r in enumerate(DEFAULT_ORDER) if r in self.rec.roles}
        return sorted(out, key=lambda l: (order.get(l, 100), list(self.rec.channels).index(l)))

    def _set_checked(self, labels):
        self.ch_list.blockSignals(True)
        for i in range(self.ch_list.count()):
            it = self.ch_list.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.data(QtCore.Qt.UserRole) in labels else QtCore.Qt.Unchecked)
        self.ch_list.blockSignals(False)
        self._channels_changed()

    def _select_default(self):
        if self.rec:
            labels = {self.rec.roles[r] for r in DEFAULT_ORDER if r in self.rec.roles}
            self._set_checked(labels or set(list(self.rec.channels)[:12]))

    def _select_resp(self):
        if self.rec:
            self._set_checked({self.rec.roles[r] for r in ("eeg", "flow", "thorax", "abdomen", "effort_sum", "snore", "spo2")
                               if r in self.rec.roles})

    def _select_all(self, on):
        if self.rec:
            self._set_checked(set(self.rec.channels) if on else set())

    def _channels_changed(self, *_):
        if not self.rec:
            return
        self.view.build(self.rec, self._checked_labels())
        self.redraw()

    # --------------------------------------------------------------------- navigation
    def _page_changed(self):
        self.page = [10, 30, 60, 120, 300, 600][self.page_box.currentIndex()]
        self.redraw()

    def step(self, direction):
        if self.rec:
            self.t0 = float(np.clip(self.t0 + direction * self.page, 0, max(self.rec.duration - self.page, 0)))
            self.redraw()

    def center_on(self, t):
        if self.rec:
            # Snap to 30-s epochs for pages >= 30 s, as scorers do.
            t0 = t - self.page / 2
            if self.page >= EPOCH_S:
                t0 = np.floor(t0 / EPOCH_S) * EPOCH_S
            self.t0 = float(np.clip(t0, 0, max(self.rec.duration - self.page, 0)))
            self.redraw()

    def jump_event(self, direction):
        if not self.result or not self.result.events:
            return
        mid = self.t0 + self.page / 2
        ons = np.array([e.onset + e.duration / 2 for e in self.result.events])
        idx = np.flatnonzero(ons > mid + 1) if direction > 0 else np.flatnonzero(ons < mid - 1)
        if idx.size:
            k = idx[0] if direction > 0 else idx[-1]
            self.center_on(float(ons[k]))
            self._select_row_for(self.result.events[k])

    def _gain(self, f):
        self.view.gain *= f
        self.redraw()

    def redraw(self):
        if not self.rec:
            return
        t1 = self.t0 + self.page
        self.view.draw(self.rec, self.display, self.t0, t1, self.result,
                       self.chk_expert.isChecked(), self.chk_derived.isChecked(), self.chk_rejected.isChecked())
        self.overview.set_window(self.t0, t1)
        ep = int(self.t0 // EPOCH_S) + 1
        stage = ""
        st = self.result.stages if self.result is not None else self.rec.expert_stages
        if st is not None and ep - 1 < len(st):
            stage = f" | stage {STAGE_NAMES[int(st[ep - 1])]}"
            if self.result is not None and self.rec.expert_stages is not None and ep - 1 < len(self.rec.expert_stages):
                stage += f" (expert {STAGE_NAMES[int(self.rec.expert_stages[ep - 1])]})"
        self.pos_label.setText(f"{clock_str(self.rec, self.t0)} – {clock_str(self.rec, t1)}   |   epoch {ep} / "
                               f"{self.rec.n_epochs}{stage}   |   ←/→ page, N/P next/prev event, +/- amplitude")

    # --------------------------------------------------------------------- analysis
    def _options(self):
        return AnalysisOptions(staging_source="auto" if self.staging.currentIndex() == 0 else "expert",
                               scoring=self.algo.params(), min_confidence=self.conf_slider.value() / 100.0)

    def run_analysis(self):
        if self.rec:
            self.enqueue([str(self.rec.path)])

    def _analyzed(self, res: AnalysisResult):
        self.result = res
        self.overview.show_recording(self.rec, res.stages, res.resp.spo2_clean, res.resp.spo2_fs)
        self.flow.set_params(res.options.scoring)
        self.flow.highlight(None)
        self.explain.clear_event()
        self._refresh_results()
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage(f"{res.diagnosis['primary']}  —  {len(res.events)} events, analysis {res.runtime_s:.0f} s")

    def _filtered_events(self):
        """(event, accepted) rows: scored events above the confidence cut-off, plus rejected
        candidates when requested, in time order."""
        if not self.result:
            return []
        f = self.ev_filter.currentText()
        shown = set(map(id, self.result.events))
        out = []
        for e in self.result.resp.candidates:
            ok = id(e) in shown
            if not ok and not (self.chk_rejected.isChecked() and not e.accepted):
                continue
            if f == "All events" or (f == "Apneas" and e.kind == "apnea") or (f == "Hypopneas" and e.kind == "hypopnea") \
                    or EVENT_NAMES.get(e.code) == f:
                out.append((e, ok))
        return out

    def _fill_events(self):
        rows = self._filtered_events()
        self.ev_table.setRowCount(len(rows))
        grey = QtGui.QColor("#9aa3ad")
        for r, (e, ok) in enumerate(rows):
            vals = [clock_str(self.rec, e.onset), f"{e.duration:.0f}", EVENT_NAMES[e.code] if ok else "not scored",
                    f"{e.confidence * 100:.0f}" if ok else "", f"{e.flow_drop * 100:.0f}",
                    "" if e.desat is None else f"{e.desat:.1f}", "" if e.desat_nadir is None else f"{e.desat_nadir:.0f}",
                    e.stage or "", (e.notes + (" arousal" if e.arousal else "")) if ok else e.reject_reason]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 0:
                    it.setData(QtCore.Qt.UserRole, e)
                if not ok:
                    it.setForeground(grey)
                elif c == 2:
                    it.setForeground(QtGui.QColor(*EVENT_COLORS[e.code]))
                elif c == 3:
                    conf = e.confidence * 100
                    it.setForeground(QtGui.QColor("#2e7d32" if conf >= 90 else "#f9a825" if conf >= 75 else "#c62828"))
                    lim = e.limiting
                    if lim is not None:
                        it.setToolTip(f"Limited by {lim.label.lower()}: {lim.text()}")
                self.ev_table.setItem(r, c, it)
        self.ev_table.resizeColumnsToContents()
        n = len(self.result.events) if self.result else 0
        self.tabs.setTabText(1, f"Events ({n})")

    def _show_event_details(self, e):
        self.explain.show_event(e, clock_str(self.rec, e.onset))
        self.flow.highlight(e.path)

    def _event_clicked(self, row, _col):
        it = self.ev_table.item(row, 0)
        if it is None or not self.result:
            return
        e = it.data(QtCore.Qt.UserRole)
        if self.page < 60:
            self.page_box.setCurrentIndex(2)  # 60-s page shows an event with context
        self.center_on(e.onset + e.duration / 2)
        self._show_event_details(e)

    def _select_row_for(self, event):
        for r in range(self.ev_table.rowCount()):
            it = self.ev_table.item(r, 0)
            if it and it.data(QtCore.Qt.UserRole) is event:
                self.ev_table.selectRow(r)
                self.ev_table.scrollToItem(it)
                self._show_event_details(event)
                return

    def _report_html(self, res: AnalysisResult) -> str:
        from analyze import validation_summary
        from html import escape as esc
        ix = res.summary
        dg = {k: (esc(v) if isinstance(v, str) else [esc(x) for x in v]) for k, v in res.diagnosis.items()}
        ox = ix.get("spo2") or {}
        sev_col = {"Normal": "#2e7d32", "Mild": "#f9a825", "Moderate": "#ef6c00", "Severe": "#c62828"}.get(dg["severity"], "#555")
        n = lambda v, f="{:.1f}": "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)
        rows = lambda pairs: "".join(f"<tr><td style='color:#555;padding-right:14px'>{k}</td><td><b>{v}</b></td></tr>" for k, v in pairs)
        val = validation_summary(res)
        val_rows = rows([(k, v) for k, v in val.items() if not k.startswith("_")])
        html = f"""
<h2 style='margin:0'>{dg['primary']}</h2>
<p><span style='background:{sev_col};color:white;padding:2px 8px'>&nbsp;{dg['severity']}&nbsp;</span>
&nbsp; staging: {res.stage_source}</p>
<ul>{''.join(f'<li>{f}</li>' for f in dg['findings'])}</ul>
<p><b>Clinical flags</b></p><ul>{''.join(f'<li>{f}</li>' for f in dg['flags']) or '<li>None</li>'}</ul>
<h3>Respiratory</h3><table>{rows([
    ("Scoring rule", esc(res.rule_name)),
    ("Events by confidence ≥90 / ≥75 / ≥50 %", " / ".join(str(v) for v in ix.get('confidence_counts', {}).values())
        + f" of {ix.get('n_scored_all', 0)} scored; {ix.get('n_rejected', 0)} candidates rejected"
        + (f"; indices use ≥{ix.get('min_confidence', 0) * 100:.0f} %" if ix.get('min_confidence') else "")),
    ("AHI", n(ix['ahi']) + " /h"), ("Obstructive AHI / Central AHI", f"{n(ix['oahi'])} / {n(ix['cahi'])} /h"),
    ("REM AHI / NREM AHI", f"{n(ix['rem_ahi'])} / {n(ix['nrem_ahi'])} /h"),
    ("Obstructive / central / mixed apneas", f"{ix['n_obstructive_apnea']} / {ix['n_central_apnea']} / {ix['n_mixed_apnea']}"),
    ("Obstructive / central hypopneas", f"{ix['n_obstructive_hypopnea']} / {ix['n_central_hypopnea']}"),
    ("Arousal index", n(ix['arousal_index']) + " /h")])}</table>
<h3>Oxygenation</h3><table>{rows([
    ("ODI", n(ix['odi']) + " /h"), ("Mean / nadir SpO2", f"{n(ox.get('mean'))} / {n(ox.get('nadir'), '{:.0f}')} %"),
    ("Time &lt; 90 %", n(ox.get('t90_min')) + " min")])}</table>
<h3>Sleep</h3><table>{rows([
    ("TST / TRT", f"{ix['tst_min']:.0f} / {ix['trt_min']:.0f} min"), ("Sleep efficiency", f"{ix['sleep_efficiency']:.0f} %"),
    ("Sleep latency / REM latency", f"{n(ix['sleep_latency_min'], '{:.0f}')} / {n(ix['rem_latency_min'], '{:.0f}')} min"),
    ("N1 / N2 / N3 / REM", f"{ix['pct_n1']:.0f} / {ix['pct_n2']:.0f} / {ix['pct_n3']:.0f} / {ix['pct_rem']:.0f} %")])}</table>
<h3>Signal quality</h3><table>{rows([(k, f"{v * 100:.0f} % usable") for k, v in ix.get('signal_quality', {}).items()])}</table>
{('<h3>Warnings</h3><ul>' + ''.join(f'<li>{esc(w)}</li>' for w in res.warnings) + '</ul>') if res.warnings else ''}
{('<h3>Agreement with expert scoring</h3><table>' + val_rows + '</table>') if val_rows else ''}
<p style='color:#888;font-size:small'>Rules: AASM Scoring Manual (adult). Apnea ≥90 % flow drop ≥10 s; hypopnea
≥30 % drop ≥10 s with {'≥3 % desaturation or arousal' if res.options.scoring.hypopnea_arousal else '≥4 % desaturation'}.
Automated decision support — to be reviewed by a sleep physician.</p>"""
        return html

    def export(self):
        if not self.result:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save a copy of the report to folder", str(Path.home()))
        if not folder:
            return
        from analyze import validation_summary
        name = self.rec.name
        out = Path(folder)
        val = {k: v for k, v in validation_summary(self.result).items() if not k.startswith("_")}
        write_html(self.result, out / f"{name}_report.html", val)
        write_events_csv(self.result, out / f"{name}_events.csv")
        write_json(self.result, out / f"{name}_summary.json")
        self.statusBar().showMessage(f"Exported to {out}")
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out / f"{name}_report.html")))


def main():
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.enqueue(sys.argv[1:])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
