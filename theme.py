"""Visual theme for the desktop app: palette, fonts, colour maps and the Qt stylesheet.

One accent colour, neutral greys, white cards on a light background, 8-px rhythm. Colours for
sleep stages, event types and severity are defined once here and reused by every widget so the
overview, the traces, the tables and the report agree.
"""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

QtCore_Qt = QtCore.Qt
QtCore_QPointF = QtCore.QPointF

ACCENT = "#1f6fb2"
ACCENT_SOFT = "#eaf3fb"
INK = "#1b2a3a"
MUTED = "#5b6b7c"
LINE = "#dde3ea"
BG = "#f4f6f9"
CARD = "#ffffff"

SEVERITY_COLORS = {"Normal": "#2e7d32", "Mild": "#f9a825", "Moderate": "#ef6c00", "Severe": "#c62828",
                   "Unknown": "#6b7785"}

# Canonical stage code -> colour (W grey, N1 light, N2 mid, N3 dark blue, REM red).
STAGE_COLORS = {0: "#cfd6de", 1: "#a6cbe3", 2: "#4a90c9", 3: "#1b3f73", 4: "#e2574c", -1: "#eef1f4"}
STAGE_TEXT = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "?"}

EVENT_COLORS = {"AO": (31, 119, 180), "AC": (44, 160, 44), "AM": (148, 103, 189),
                "HO": (255, 127, 14), "HC": (140, 86, 75)}
EVENT_NAMES = {"AO": "Obstructive apnea", "AC": "Central apnea", "AM": "Mixed apnea",
               "HO": "Obstructive hypopnea", "HC": "Central hypopnea"}

# Trace colours by channel role: neuro in dark tones, respiratory in blues, oximetry accent.
CHANNEL_COLORS = {
    "eeg": "#2b2f36", "eeg2": "#2b2f36", "eog_l": "#0b7285", "eog_r": "#0b7285", "emg": "#6f42c1",
    "ecg": "#a61e4d", "flow": "#1c4e80", "thorax": "#3b6ea5", "abdomen": "#3b6ea5", "effort_sum": "#3b6ea5",
    "snore": "#6b7280", "spo2": ACCENT, "pulse": "#c2410c", "position": "#4b5563",
}
DEFAULT_TRACE = "#33393f"


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


QSS = f"""
QToolBar {{ background: {CARD}; border: none; border-bottom: 1px solid {LINE}; padding: 4px 8px; spacing: 4px; }}
QToolBar::separator {{ background: {LINE}; width: 1px; margin: 6px 8px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; color: {INK}; }}
QToolButton:hover {{ background: {ACCENT_SOFT}; border-color: #cfe1f2; }}
QToolButton:pressed {{ background: #d8e8f7; }}
QToolButton:disabled {{ color: #a3adb8; }}

QDockWidget {{ font-weight: 600; color: {MUTED}; }}
QDockWidget::title {{ background: {BG}; padding: 7px 12px; text-align: left; border-bottom: 1px solid {LINE}; }}

QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: {CARD}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 7px 14px; margin-right: 2px; color: {MUTED};
                border: 1px solid transparent; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: {CARD}; border-color: {LINE}; color: {INK}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ background: {ACCENT_SOFT}; }}

QTableWidget, QTableView {{ background: {CARD}; alternate-background-color: #f8fafc; gridline-color: #eef2f6;
                            border: none; selection-background-color: #d8e8f7; selection-color: {INK}; }}
QHeaderView::section {{ background: #f7f9fb; color: {MUTED}; padding: 6px 6px; border: none;
                        border-bottom: 1px solid {LINE}; font-weight: 600; }}
QListWidget {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 6px; padding: 2px; }}
QListWidget::item {{ padding: 3px 2px; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {INK}; }}

QComboBox {{ background: {CARD}; border: 1px solid #cfd8e3; border-radius: 6px; padding: 4px 8px; min-height: 20px; }}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {CARD}; selection-background-color: {ACCENT_SOFT}; selection-color: {INK};
                               border: 1px solid {LINE}; }}

QPushButton {{ background: {CARD}; border: 1px solid #cfd8e3; border-radius: 6px; padding: 6px 14px; color: {INK}; }}
QPushButton:hover {{ border-color: {ACCENT}; background: {ACCENT_SOFT}; }}
QPushButton:pressed {{ background: #d8e8f7; }}
QPushButton:disabled {{ color: #a3adb8; border-color: #e1e6ec; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: white; border-color: {ACCENT}; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: #185c95; }}

QGroupBox {{ border: 1px solid {LINE}; border-radius: 8px; margin-top: 14px; padding: 12px 8px 8px 8px; background: {CARD}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; font-weight: 600; }}

QSlider::groove:horizontal {{ height: 5px; background: #e1e6ec; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {CARD}; border: 2px solid {ACCENT}; width: 14px; height: 14px; margin: -6px 0; border-radius: 9px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT_SOFT}; }}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid #b7c2cf; border-radius: 4px; background: {CARD}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url(CHECK_ICON); }}

QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab:first {{ margin-left: 6px; }}

QProgressBar {{ border: none; background: #e1e6ec; border-radius: 5px; max-height: 10px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QStatusBar {{ background: {CARD}; border-top: 1px solid {LINE}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #c9d3de; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #aebbc9; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #c9d3de; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: #aebbc9; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: #e6ebf1; }}
QSplitter::handle:vertical {{ height: 5px; }}
QTextBrowser {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px; }}
QToolTip {{ background: {INK}; color: white; border: none; padding: 6px 8px; border-radius: 4px; }}
"""


def _check_icon_path() -> str:
    """A white tick for checked boxes, rendered once to the temp folder (QSS needs a file URL)."""
    import tempfile
    from pathlib import Path
    path = Path(tempfile.gettempdir()) / "psg_check.png"
    if not path.exists():
        pm = QtGui.QPixmap(24, 24)
        pm.fill(QtGui.QColor(0, 0, 0, 0))
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor("white"), 3.2, QtCore_Qt.SolidLine, QtCore_Qt.RoundCap, QtCore_Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline(QtGui.QPolygonF([QtCore_QPointF(5, 12.5), QtCore_QPointF(10, 17.5), QtCore_QPointF(19, 7)]))
        p.end()
        pm.save(str(path), "PNG")
    return path.as_posix()


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BG))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(INK))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(CARD))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f8fafc"))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(INK))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(CARD))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(INK))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(ACCENT))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("white"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(INK))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("white"))
    pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(MUTED))
    app.setPalette(pal)
    font = QtGui.QFont("Segoe UI", 9)
    if not QtGui.QFontInfo(font).family().lower().startswith("segoe"):
        font = QtGui.QFont()
        font.setPointSize(9)
    app.setFont(font)
    app.setStyleSheet(QSS.replace("CHECK_ICON", _check_icon_path()))
