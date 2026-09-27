"""Qt widgets for the algorithm panel (rule version + threshold sliders), the decision flowchart
with per-event path highlighting, and the per-event explanation pane."""
from __future__ import annotations

import dataclasses

from PyQt5 import QtCore, QtGui, QtWidgets

from psg.respiratory import RespEvent, ScoringParams
from psg.rules import DEFAULT_RULE, RULES, RuleVersion, flowchart, get_rule

EVENT_COLORS = {"AO": (31, 119, 180), "AC": (44, 160, 44), "AM": (148, 103, 189),
                "HO": (255, 127, 14), "HC": (140, 86, 75)}


# ============================================================================ algorithm panel

class _Slider(QtWidgets.QWidget):
    """Labelled horizontal slider with a value readout; works in real units via a scale."""
    changed = QtCore.pyqtSignal()

    def __init__(self, title: str, lo: float, hi: float, step: float, unit: str, fmt: str = "{:g}"):
        super().__init__()
        self.scale, self.unit, self.fmt = 1.0 / step, unit, fmt
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.title = QtWidgets.QLabel(title)
        self.value_lbl = QtWidgets.QLabel()
        self.value_lbl.setStyleSheet("font-weight:600")
        self.value_lbl.setAlignment(QtCore.Qt.AlignRight)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(round(lo * self.scale)), int(round(hi * self.scale)))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._on_change)
        lay.addWidget(self.title, 0, 0)
        lay.addWidget(self.value_lbl, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)

    def value(self) -> float:
        return self.slider.value() / self.scale

    def set_value(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self.scale)))
        self.slider.blockSignals(False)
        self._refresh()

    def _refresh(self):
        self.value_lbl.setText(self.fmt.format(self.value()) + self.unit)

    def _on_change(self):
        self._refresh()
        self.changed.emit()


class AlgorithmPanel(QtWidgets.QWidget):
    """Pick a published rule version, then fine-tune its thresholds with sliders.
    Emits `params_changed(ScoringParams)` (debounced) whenever the effective parameters change."""
    params_changed = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._base = ScoringParams()
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        head = QtWidgets.QLabel("Scoring algorithm")
        head.setStyleSheet("font-size:14px;font-weight:600")
        v.addWidget(head)
        v.addWidget(QtWidgets.QLabel("Rule version"))
        self.combo = QtWidgets.QComboBox()
        for r in RULES.values():
            self.combo.addItem(f"{r.year}  ·  {r.name}", r.id)
        self.combo.setCurrentIndex(list(RULES).index(DEFAULT_RULE))
        self.combo.currentIndexChanged.connect(self._rule_selected)
        v.addWidget(self.combo)
        self.desc = QtWidgets.QLabel()
        self.desc.setWordWrap(True)
        self.desc.setOpenExternalLinks(True)
        self.desc.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.desc.setStyleSheet("color:#444;font-size:12px;background:#f7f9fb;border:1px solid #e1e6ec;"
                                "border-radius:6px;padding:8px")
        v.addWidget(self.desc)

        box = QtWidgets.QGroupBox("Thresholds (drag to explore; the night is re-scored in ~1 s)")
        bl = QtWidgets.QVBoxLayout(box)
        self.s_apnea = _Slider("Apnea: airflow drop ≥", 70, 99, 1, " %")
        self.s_hyp = _Slider("Hypopnea: airflow drop ≥", 10, 70, 1, " %")
        self.s_dur = _Slider("Minimum duration ≥", 5, 20, 1, " s")
        self.s_desat = _Slider("Desaturation ≥", 1.0, 6.0, 0.5, " %", "{:.1f}")
        for s in (self.s_apnea, self.s_hyp, self.s_dur, self.s_desat):
            s.changed.connect(self._schedule)
            bl.addWidget(s)
        self.c_arousal = QtWidgets.QCheckBox("Arousal confirms a hypopnea")
        self.c_classify = QtWidgets.QCheckBox("Classify hypopneas obstructive / central")
        for c in (self.c_arousal, self.c_classify):
            c.toggled.connect(self._schedule)
            bl.addWidget(c)
        self.extra = QtWidgets.QLabel()
        self.extra.setStyleSheet("color:#666;font-size:11px")
        self.extra.setWordWrap(True)
        bl.addWidget(self.extra)
        row = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset to rule values")
        self.btn_reset.clicked.connect(self._rule_selected)
        row.addWidget(self.btn_reset)
        row.addStretch(1)
        bl.addLayout(row)
        v.addWidget(box)
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color:#666;font-size:11px")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(lambda: self.params_changed.emit(self.params()))
        self._rule_selected()

    # -- rule / params -------------------------------------------------------------------------
    def rule(self) -> RuleVersion:
        return get_rule(self.combo.currentData())

    def _rule_selected(self, *_):
        r = self.rule()
        self._base = r.params(self._base)
        for w in (self.s_apnea, self.s_hyp, self.s_dur, self.s_desat, self.c_arousal, self.c_classify):
            w.blockSignals(True)
        self.s_apnea.set_value(r.apnea_drop * 100)
        self.s_hyp.set_value(r.hypopnea_drop * 100)
        self.s_dur.set_value(r.min_duration)
        self.s_desat.set_value(r.desat)
        self.c_arousal.setChecked(r.arousal)
        self.c_classify.setChecked(r.classify_hypopneas)
        for w in (self.s_apnea, self.s_hyp, self.s_dur, self.s_desat, self.c_arousal, self.c_classify):
            w.blockSignals(False)
        extras = []
        if r.hypopnea_drop_alone:
            extras.append(f"A drop ≥{r.hypopnea_drop_alone * 100:.0f}% counts without desaturation/arousal.")
        if r.amplitude_fraction:
            extras.append(f"≥{r.amplitude_fraction * 100:.0f}% of the event must meet the amplitude drop.")
        self.extra.setText(" ".join(extras))
        self.desc.setText(
            f"<b>{r.name}</b><br>{r.summary}<br><span style='color:#777'>{r.notes}</span><br>"
            f"<span style='color:#777'>Source: {r.source}</span>" + (f" · <a href='{r.url}'>link</a>" if r.url else ""))
        self._schedule()

    def params(self) -> ScoringParams:
        p = dataclasses.replace(
            self._base, apnea_drop=self.s_apnea.value() / 100, hypopnea_drop=self.s_hyp.value() / 100,
            min_duration=self.s_dur.value(), hypopnea_desat=self.s_desat.value(),
            hypopnea_arousal=self.c_arousal.isChecked(), classify_hypopneas=self.c_classify.isChecked())
        r = self.rule()
        if (p.apnea_drop, p.hypopnea_drop, p.min_duration, p.hypopnea_desat, p.hypopnea_arousal, p.classify_hypopneas) != \
                (r.apnea_drop, r.hypopnea_drop, r.min_duration, r.desat, r.arousal, r.classify_hypopneas):
            p = dataclasses.replace(p, rule_id=f"{r.id}+custom")
        return p

    def set_params(self, p: ScoringParams):
        """Show an existing parameter set (e.g. from a loaded study)."""
        rid = p.rule_id.split("+")[0]
        if rid in RULES:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(list(RULES).index(rid))
            self.combo.blockSignals(False)
            self._rule_selected()
        self._base = p
        self.s_apnea.set_value(p.apnea_drop * 100)
        self.s_hyp.set_value(p.hypopnea_drop * 100)
        self.s_dur.set_value(p.min_duration)
        self.s_desat.set_value(p.hypopnea_desat)
        self.c_arousal.setChecked(p.hypopnea_arousal)
        self.c_classify.setChecked(p.classify_hypopneas)

    def _schedule(self, *_):
        self._timer.start()

    def set_status(self, text: str):
        self.status.setText(text)


# ============================================================================ flowchart

class FlowchartView(QtWidgets.QGraphicsView):
    """Draws the scorer's decision steps for the current parameters and highlights the path of
    one event.  Click a node to get its step id via `step_clicked`."""
    step_clicked = QtCore.pyqtSignal(str)

    COL_W, ROW_H, NODE_W, NODE_H = 205, 96, 170, 62
    MIN_SCALE = 0.72
    POS = {  # step id -> (column, row); compact three-column layout so it stays legible in a dock
        "start": (0, 0), "sleep": (0, 1), "reject_wake": (1.15, 1), "signal": (0, 2), "reject_signal": (1.15, 2),
        "apnea": (0, 3), "effort": (-1.3, 4), "AO": (-1.3, 5), "AC": (-1.3, 5.8), "AM": (-1.3, 6.6),
        "hypopnea": (0, 4), "reject_drop": (1.15, 4), "confirm": (0, 5), "reject_confirm": (1.15, 5),
        "hyp_type": (0, 6), "HO": (-0.3, 7.1), "HC": (0.6, 7.1), "H": (0, 7.1),
    }

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setBackgroundBrush(QtGui.QColor("#fbfcfd"))
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self._nodes: dict[str, QtWidgets.QGraphicsRectItem] = {}
        self._edges: dict[tuple[str, str], list] = {}
        self._steps = {}
        self._path: list[str] = []

    def set_params(self, p: ScoringParams):
        self.scene().clear()
        self._nodes.clear()
        self._edges.clear()
        self._steps = {s.id: s for s in flowchart(p)}
        font = QtGui.QFont()
        font.setPointSize(9)
        for sid, s in self._steps.items():
            if sid not in self.POS:
                continue
            c, r = self.POS[sid]
            x, y = c * self.COL_W, r * self.ROW_H
            rect = QtCore.QRectF(x - self.NODE_W / 2, y - self.NODE_H / 2, self.NODE_W, self.NODE_H)
            if s.kind == "decision":
                item = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([
                    QtCore.QPointF(rect.center().x(), rect.top() - 6), QtCore.QPointF(rect.right() + 22, rect.center().y()),
                    QtCore.QPointF(rect.center().x(), rect.bottom() + 6), QtCore.QPointF(rect.left() - 22, rect.center().y())]))
            else:
                item = QtWidgets.QGraphicsRectItem(rect)
            item.setData(0, sid)
            item.setPen(QtGui.QPen(QtGui.QColor("#7a8794"), 1.2))
            self.scene().addItem(item)
            txt = QtWidgets.QGraphicsSimpleTextItem(s.text)
            txt.setFont(font)
            b = txt.boundingRect()
            txt.setPos(x - b.width() / 2, y - b.height() / 2)
            txt.setZValue(2)
            self.scene().addItem(txt)
            self._nodes[sid] = item
        for sid, s in self._steps.items():
            if sid not in self._nodes:
                continue
            links = []
            if s.yes:
                links.append((s.yes, "yes" if s.kind == "decision" else ""))
            if s.no:
                links.append((s.no, "no"))
            links += [(t, lab) for lab, t in s.branches.items()]
            for k, (target, label) in enumerate(links):
                if target in self._nodes:
                    # Multi-way branches beyond the first are routed around the left side so they
                    # do not run through the boxes stacked below the decision.
                    detour = 28 * k if s.branches and k > 0 else 0
                    self._edges[(sid, target)] = self._draw_edge(sid, target, label, detour)
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-30, -30, 30, 30))
        self._style()
        self.fit()

    def _anchor(self, sid: str, other: str, out: bool) -> QtCore.QPointF:
        c1, r1 = self.POS[sid]
        c2, r2 = self.POS[other]
        x, y = c1 * self.COL_W, r1 * self.ROW_H
        is_decision = self._steps[sid].kind == "decision"
        if abs(r2 - r1) < 0.5:                      # horizontal move
            dx = self.NODE_W / 2 + (22 if is_decision else 0)
            return QtCore.QPointF(x + dx * (1 if c2 > c1 else -1), y)
        dy = self.NODE_H / 2 + (6 if is_decision else 0)
        return QtCore.QPointF(x, y + dy * (1 if out else -1))

    def _draw_edge(self, a: str, b: str, label: str, detour: float = 0.0):
        pen = QtGui.QPen(QtGui.QColor("#9aa6b2"), 1.2)
        if detour:
            # Out of the source's left side, down a lane to the left of the column, into the target's left side.
            c1, r1 = self.POS[a]
            c2, r2 = self.POS[b]
            lane = c1 * self.COL_W - self.NODE_W / 2 - 22 - detour
            p1 = QtCore.QPointF(c1 * self.COL_W - self.NODE_W / 2 - 22, r1 * self.ROW_H)
            p2 = QtCore.QPointF(c2 * self.COL_W - self.NODE_W / 2, r2 * self.ROW_H)
            path = QtGui.QPainterPath(p1)
            path.lineTo(QtCore.QPointF(lane, p1.y()))
            path.lineTo(QtCore.QPointF(lane, p2.y()))
            path.lineTo(p2)
            arrow_dir = "right"
            label_pos = QtCore.QPointF(lane + 4, (p1.y() + p2.y()) / 2 - 8)
        else:
            p1, p2 = self._anchor(a, b, True), self._anchor(b, a, False)
            path = QtGui.QPainterPath(p1)
            if abs(p1.x() - p2.x()) > 5 and abs(p1.y() - p2.y()) > 5:   # elbow
                mid = QtCore.QPointF(p1.x(), p2.y() - 18)
                path.lineTo(mid)
                path.lineTo(QtCore.QPointF(p2.x(), mid.y()))
            path.lineTo(p2)
            arrow_dir = "down" if p2.y() > p1.y() + 5 else ("right" if p2.x() > p1.x() else "left")
            pos = path.pointAtPercent(0.35)
            label_pos = QtCore.QPointF(pos.x() + 4, pos.y() - 14)
        line = self.scene().addPath(path, pen)
        tips = {"down": [(-5, -8), (5, -8)], "right": [(-8, -5), (-8, 5)], "left": [(8, -5), (8, 5)]}[arrow_dir]
        head = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([p2] + [p2 + QtCore.QPointF(*d) for d in tips]))
        head.setBrush(QtGui.QColor("#9aa6b2"))
        head.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.scene().addItem(head)
        items = [line, head]
        if label:
            t = QtWidgets.QGraphicsSimpleTextItem(label)
            f = QtGui.QFont()
            f.setPointSize(7)
            f.setItalic(True)
            t.setFont(f)
            t.setBrush(QtGui.QColor("#556"))
            t.setPos(label_pos)
            self.scene().addItem(t)
            items.append(t)
        return items

    def highlight(self, path_ids: list[str] | None):
        self._path = list(path_ids or [])
        self._style()

    def _style(self):
        on = set(self._path)
        for sid, item in self._nodes.items():
            s = self._steps[sid]
            if s.kind == "terminal":
                col = QtGui.QColor(*EVENT_COLORS[s.outcome]) if s.outcome in EVENT_COLORS else QtGui.QColor("#b0b8c0")
                if sid in on:
                    item.setBrush(col)
                    item.setPen(QtGui.QPen(col.darker(130), 2))
                else:
                    light = QtGui.QColor(col)
                    light.setAlpha(45)
                    item.setBrush(light)
                    item.setPen(QtGui.QPen(QtGui.QColor("#aab4be"), 1))
            else:
                item.setBrush(QtGui.QColor("#dbe9f7") if sid in on else QtGui.QColor("white"))
                item.setPen(QtGui.QPen(QtGui.QColor("#1f6fb2") if sid in on else QtGui.QColor("#7a8794"), 2.2 if sid in on else 1.2))
        for (a, b), items in self._edges.items():
            active = a in on and b in on and self._path.index(b) == self._path.index(a) + 1 if (a in on and b in on) else False
            pen = QtGui.QPen(QtGui.QColor("#1f6fb2") if active else QtGui.QColor("#9aa6b2"), 2.6 if active else 1.2)
            items[0].setPen(pen)
            items[1].setBrush(QtGui.QColor("#1f6fb2") if active else QtGui.QColor("#9aa6b2"))

    def fit(self):
        """Fit to the viewport width but never shrink below MIN_SCALE (scrollbars take over)."""
        rect = self.scene().sceneRect()
        if rect.width() <= 0:
            return
        vw = max(self.viewport().width() - 4, 50)
        scale = max(min(vw / rect.width(), 1.0), self.MIN_SCALE)
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(rect.center().x(), rect.top() + self.viewport().height() / (2 * scale))

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
            self.scale(f, f)
        else:
            super().wheelEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fit()

    def mousePressEvent(self, ev):
        item = self.itemAt(ev.pos())
        while item is not None and item.data(0) is None:
            item = item.parentItem()
        if item is not None:
            self.step_clicked.emit(item.data(0))
        super().mousePressEvent(ev)


# ============================================================================ explanation pane

class ExplanationPane(QtWidgets.QTextBrowser):
    """Plain-language 'why' for one event, with every criterion, its value and the rule threshold."""

    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(False)
        self.setStyleSheet("QTextBrowser{background:#fbfcfd;border:1px solid #e1e6ec;border-radius:6px}")
        self.clear_event()

    def clear_event(self):
        self.setHtml("<p style='color:#777'>Select an event to see why it was (or was not) scored.</p>")

    def show_event(self, e: RespEvent, clock: str = ""):
        col = "#c62828" if not e.accepted else ("#2e7d32" if e.confidence >= 0.9 else "#ef6c00")
        title = (f"{e.label} at {clock} · {e.duration:.0f} s" if e.accepted
                 else f"Candidate at {clock} · {e.duration:.0f} s · <span style='color:{col}'>not scored</span>")
        conf = (f"<span style='background:{col};color:white;padding:1px 8px;border-radius:3px'>confidence "
                f"{e.confidence * 100:.0f}%</span> &nbsp;type {e.type_confidence * 100:.0f}%" if e.accepted else
                f"<span style='color:{col}'><b>{e.reject_reason}</b></span>")
        rows = []
        for c in e.checks:
            mark = "✔" if c.passed else "✘"
            mcol = "#2e7d32" if c.passed else "#c62828"
            # Qt's rich-text engine ignores sized <div>s, so the margin bar is a 2-cell table.
            bar_w = max(1, int(max(0.0, min(1.0, c.score)) * 90))
            fill = "#1f6fb2" if c.counts else "#b0b8c0"
            bar = (f"<table cellspacing='0' cellpadding='0'><tr><td bgcolor='{fill}' width='{bar_w}' height='8'></td>"
                   f"<td bgcolor='#e8edf2' width='{90 - bar_w}' height='8'></td></tr></table>"
                   if c.counts or c.step in ("effort", "hyp_type") else "")
            val = "—" if c.value is None else (f"{c.value:.0f}{c.unit}" if c.unit in ("%", "s") else f"{c.value:.2f}")
            thr = "" if c.threshold is None else f"≥ {c.threshold:g}{c.unit}"
            rows.append(f"<tr><td style='color:{mcol};font-weight:700'>{mark}</td><td>{c.label}</td>"
                        f"<td><b>{val}</b></td><td style='color:#666'>{thr}</td><td>{bar}</td>"
                        f"<td style='color:#666'>{c.detail}</td></tr>")
        lim = e.limiting
        lim_txt = (f"<p style='color:#555'>Confidence is limited by <b>{lim.label.lower()}</b>: {lim.text()}.</p>"
                   if e.accepted and lim is not None and lim.score < 0.9 else "")
        self.setHtml(f"""
<h3 style='margin:0 0 4px'>{title}</h3><p style='margin:0 0 8px'>{conf}</p>{lim_txt}
<table cellspacing='0' cellpadding='3' style='font-size:12px'>
<tr style='color:#666'><th></th><th align='left'>Criterion</th><th align='left'>Measured</th><th align='left'>Rule</th><th align='left'>Margin</th><th align='left'>Detail</th></tr>
{''.join(rows)}</table>
<p style='color:#777;font-size:11px;margin-top:8px'>Margin bar: how far the value clears the threshold (half = exactly at the threshold). Grey bars are informational or type classification and do not affect the confidence.</p>""")
