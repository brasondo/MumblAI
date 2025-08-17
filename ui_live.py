import os
import re
import sys
import time
import queue
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from PySide6 import QtCore, QtGui, QtWidgets

# ─────────────────────────────────────────────
# External modules from your project
# ─────────────────────────────────────────────
from transcriber import AudioTranscriber
try:
    from mumble_extender import MumbleExtender
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

# ─────────────────────────────────────────────
# Tuning (kept from live_main.py)
# ─────────────────────────────────────────────
CHANNELS = 1
CHUNK_SEC = 1.2          # snappier ticks
SLEEP = CHUNK_SEC / 2
WINDOW = 40              # max tokens kept
RATE_CANDIDATES = [16000, 48000, 44100]
FILLERS = {"uh", "umm", "um", "er", "uhh", "you know", "like", "okay", "ok"}
STYLE = os.getenv("STYLE", "clean, clever, punchy")

# ─────────────────────────────────────────────
# Helpers (kept from live_main.py)
# ─────────────────────────────────────────────
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")

def _color(tok: Dict[str, str]) -> str:
    lab = tok["label"]
    txt = tok["text"]
    if lab == "CLEAR":     return f"\033[92m{txt}\033[0m"
    if lab == "UNCERTAIN": return f"\033[93m{txt}\033[0m"
    return f"\033[90m{txt}\033[0m"  # MUMBLE


def _record_cb(q: "queue.Queue[np.ndarray]"):
    def cb(indata, frames, t, status):
        if status:
            print("Audio status:", status, file=sys.stderr)
        q.put(indata.copy())
    return cb


def _flush_numpy(q: "queue.Queue[np.ndarray]") -> Optional[np.ndarray]:
    bufs = []
    try:
        while True:
            bufs.append(q.get_nowait())
    except queue.Empty:
        pass
    if not bufs:
        return None
    return np.concatenate(bufs, axis=0)


def _write_wav(np_audio: np.ndarray, path: str, rate: int):
    # ensure mono int16 and (N, 1) for soundfile
    if np_audio.ndim == 2 and np_audio.shape[1] == 1:
        data = np_audio
    elif np_audio.ndim == 1:
        data = np_audio.reshape(-1, 1)
    else:
        # collapse to mono if needed
        data = np_audio.astype(np.float32).mean(axis=1, keepdims=True)
        data = np.clip(data, -32768, 32767).astype(np.int16)
    if data.dtype != np.int16:
        data = np.clip(data, -32768, 32767).astype(np.int16)
    sf.write(path, data, rate, subtype="PCM_16")


def _wasapi_index():
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api.get("name", "").lower():
                return i
    except Exception:
        pass
    return None


def _device_hostapi_index(dev_idx):
    try:
        return sd.query_devices(dev_idx).get("hostapi", None)
    except Exception:
        return None


def candidate_devices(explicit_first=1):
    """
    Yield candidate input device indices, prioritizing WASAPI devices:
    1) explicit index (15 from your list)
    2) system default input
    3) all devices with input channels > 0 (WASAPI first)
    """
    wasapi = _wasapi_index()

    def _good(i):
        try:
            return sd.query_devices(i).get("max_input_channels", 0) > 0
        except Exception:
            return False

    # 1) explicit
    if _good(explicit_first):
        yield explicit_first
    # 2) default input
    try:
        di = sd.default.device[0]
        if di is not None and _good(di):
            yield di
    except Exception:
        pass
    # 3) all inputs
    try:
        idxs = list(range(len(sd.query_devices())))
        if wasapi is not None:
            was = [i for i in idxs if _good(i) and _device_hostapi_index(i) == wasapi]
            oth = [i for i in idxs if _good(i) and _device_hostapi_index(i) != wasapi]
            for i in was: yield i
            for i in oth: yield i
        else:
            for i in idxs:
                if _good(i):
                    yield i
    except Exception:
        pass


def open_working_stream(q: "queue.Queue[np.ndarray]"):
    """
    Try (device, samplerate) combos until one opens.
    Returns (stream, rate, dev_idx, hostapi_name).
    """
    for dev in candidate_devices():
        name = ""
        hostapi_name = "Unknown"
        try:
            info = sd.query_devices(dev)
            name = info.get("name", "")
            ha = info.get("hostapi", None)
            if ha is not None:
                hostapi_name = sd.query_hostapis()[ha]["name"]
        except Exception:
            pass
        for rate in RATE_CANDIDATES:
            try:
                sd.check_input_settings(device=dev, samplerate=rate, channels=CHANNELS, dtype="int16")
                stream = sd.InputStream(
                    device=dev,
                    samplerate=rate,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=int(rate * CHUNK_SEC),
                    callback=_record_cb(q),
                )
                stream.start()
                print(f"🎧 Using {hostapi_name} | device {dev}: {name} | {rate} Hz")
                return stream, rate, dev, hostapi_name
            except Exception:
                continue
    raise RuntimeError("No input device could be opened. Check permissions/mic index.")


def _is_filler(s: str) -> bool:
    w = s.lower().strip(".,?!:;")
    return w in FILLERS

# ─────────────────────────────────────────────
# UI Components (adapted from your UI-only file)
# ─────────────────────────────────────────────
class FrostCard(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FrostCard")
        self.setStyleSheet(
            "#FrostCard{background:rgba(20,22,26,0.72); "
            "border:1px solid rgba(255,255,255,0.08); border-radius:18px}"
        )
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 18)
        shadow.setColor(QtGui.QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)


class RecorderButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(96, 96)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        # micro-motion
        self._pulse = QtCore.QPropertyAnimation(self, b"maximumWidth")
        self._pulse.setDuration(1200)
        self._pulse.setStartValue(96)
        self._pulse.setEndValue(104)
        self._pulse.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)
        self._update_style()
        self.toggled.connect(self._update_style)

    def _update_style(self):
        if self.isChecked():
            self.setStyleSheet(
                """
                QPushButton {
                    background: qradialgradient(cx:.5,cy:.5,radius:.7,fx:.5,fy:.5,
                        stop:0 #ff4a3f, stop:1 #c41e17);
                    border-radius:48px; border:none;
                }
                """
            )
            if QtCore.QCoreApplication.instance() is not None and \
               QtCore.QCoreApplication.instance().property("REDUCED_MOTION") != "1":
                self._pulse.start()
        else:
            self.setStyleSheet(
                """
                QPushButton { background:transparent; border-radius:48px; border:2px solid #ff4a3f; }
                QPushButton:hover { border-color:#ff6159; }
                """
            )
            self._pulse.stop()


class LevelMeter(QtWidgets.QWidget):
    """
    Voice meter: 7 bars reacting to provided 'rms' level (0..1).
    Call set_level(value) from the audio thread via signal.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._n = 7
        self._rms = 0.0          # current level (0..1)
        self.setMinimumHeight(32)
        self.setMaximumHeight(38)

    @QtCore.Slot(float)
    def set_level(self, value: float):
        value = 0.0 if value is None else max(0.0, min(1.0, float(value)))
        # ease a bit for smoothness
        self._rms += 0.35 * (value - self._rms)
        if abs(self._rms - value) < 0.01:
            self._rms = value
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)
        r = self.rect()
        gap = 6
        w = (r.width() - gap*(self._n-1)) / self._n
        rms = max(0.0, min(self._rms, 1.0))
        for i in range(self._n):
            t = (i+1)/self._n
            h = r.height() * (0.25 + 0.75 * rms * t)
            x = int(i*(w+gap))
            y = int(r.center().y() - h/2)
            bar = QtCore.QRectF(x, y, w, h)
            p.setBrush(QtGui.QColor(255,255,255, int(60 + 120*t)))
            p.setPen(QtGui.QPen(QtGui.QColor(255,255,255,50), 1))
            p.drawRoundedRect(bar, 6, 6)


class VoiceDock(QtWidgets.QFrame):
    """Glassy bottom dock: meter + central mic + tiny status text."""
    def __init__(self, button: RecorderButton, parent=None):
        super().__init__(parent)
        self.button = button
        self.setObjectName("VoiceDock")
        self.setStyleSheet("""
            #VoiceDock {
                background: rgba(22,24,28,0.66);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
            }
            QLabel { color:#C2C7D0; font-size:12px; }
        """)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        self.meter = LevelMeter(self)
        lay.addWidget(self.meter)

        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        row.addWidget(self.button)
        row.addStretch()
        lay.addLayout(row)

        self.status = QtWidgets.QLabel("Tap to speak")
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.status)

    def set_recording(self, on: bool):
        self.status.setText("Listening…" if on else "Tap to speak")


class TranscriptView(QtWidgets.QTextBrowser):
    """
    Transcript view with colored chips.
    Call append_word(text, label) to add tokens.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.document().setDefaultStyleSheet(
        """
        body { font-family:-apple-system, SF Pro Text, Inter, Segoe UI; font-size:17px; color:#E6E8EC; line-height:1.52; letter-spacing:0.1px; }
        .chip { padding:4px 6px; border-radius:8px; margin:2px 2px 2px 0; display:inline-block; border:none; background:transparent }
        .clear { color:#3FB950; }        /* GREEN */
        .uncertain { color:#FFD666; }    /* YELLOW */
        .mumble { color:#FF6159; }       /* RED  */
        """)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet("QTextBrowser{background:transparent;border:none}")
        self._html = "<body></body>"

    @QtCore.Slot(str, str)
    def append_word(self, text: str, label: str):
        cls = "clear" if label == "CLEAR" else "uncertain" if label == "UNCERTAIN" else "mumble"
        safe = (text or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        snippet = f'<span class="chip {cls}">{safe}</span>'
        self._html = self._html[:-7] + snippet + "</body>"
        self.setHtml(self._html)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    @QtCore.Slot()
    def reset(self):
        self._html = "<body></body>"
        self.setHtml(self._html)


class Background(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        rect = self.rect()
        p.fillRect(rect, QtGui.QColor("#0b0c0f"))
        grad1 = QtGui.QRadialGradient(rect.width()*0.1, rect.height()*-0.1, rect.width()*0.8)
        grad1.setColorAt(0.0, QtGui.QColor(10,132,255,26))
        grad1.setColorAt(1.0, QtCore.Qt.transparent)
        p.fillRect(rect, QtGui.QBrush(grad1))
        grad2 = QtGui.QRadialGradient(rect.width()*0.9, 0, rect.width()*0.7)
        grad2.setColorAt(0.0, QtGui.QColor(88,86,214,18))
        grad2.setColorAt(1.0, QtCore.Qt.transparent)
        p.fillRect(rect, QtGui.QBrush(grad2))

# ─────────────────────────────────────────────
# Worker Thread: connects live_main functionality to UI
# ─────────────────────────────────────────────
class LiveWorker(QtCore.QThread):
    token_signal = QtCore.Signal(str, str)          # text, label
    rms_signal = QtCore.Signal(float)               # 0..1
    status_signal = QtCore.Signal(str)              # human text
    arrow_signal = QtCore.Signal(str)               # colored ansi-less → line (for debug)
    check_signal = QtCore.Signal(str)               # completed ✓ line
    started_ok = QtCore.Signal(str)                 # device description
    error_signal = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._RATE = None
        self._rolling: List[Dict[str, str]] = []
        self._last_arrow = ""
        self._last_check = ""
        self.transcriber = AudioTranscriber(model_size="distil-large-v3")
        self.extender = MumbleExtender(model=os.getenv("OLLAMA_MODEL", "llama3")) if HAS_OLLAMA else None

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            self._stream, self._RATE, dev_idx, hostapi_name = open_working_stream(self._q)
            self.started_ok.emit(f"{hostapi_name} @ {self._RATE} Hz")
        except Exception as e:
            self.error_signal.emit(str(e))
            return

        try:
            while self._running:
                time.sleep(SLEEP)
                audio_np = _flush_numpy(self._q)
                if audio_np is None or len(audio_np) < int(self._RATE * CHUNK_SEC / 2):
                    continue

                # RMS level for UI meter
                rms = float(np.sqrt(np.mean((audio_np.astype(np.float32) / 32768.0) ** 2)))
                self.rms_signal.emit(min(1.0, rms * 12.0))  # scale a bit for visibility

                # light VAD gate
                if rms < 0.007:
                    continue

                # Windows-safe temp write
                tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = tf.name
                tf.close()
                try:
                    _write_wav(audio_np, temp_path, self._RATE)
                    _, segments = self.transcriber.transcribe(temp_path)
                finally:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                new_tokens = [{"text": s.get("text", "").strip(), "label": s.get("label", "MUMBLE")} for s in segments if s.get("text", "").strip()]
                new_tokens = [t for t in new_tokens if not _is_filler(t["text"])]
                if not new_tokens:
                    continue

                tail = self._rolling[-len(new_tokens):] if len(self._rolling) >= len(new_tokens) else []
                if tail and [t["text"] for t in tail] == [t["text"] for t in new_tokens]:
                    continue

                self._rolling.extend(new_tokens)
                self._rolling = self._rolling[-WINDOW:]

                # collapse trivial repeats
                dedup: List[Dict[str, str]] = []
                for t in self._rolling[-30:]:
                    if not dedup or dedup[-1]["text"] != t["text"]:
                        dedup.append(t)
                self._rolling = dedup

                # emit tokens to UI
                for t in new_tokens:
                    self.token_signal.emit(t["text"], t["label"])

                # scaffolding for extender
                scaff = " ".join([t["text"] if t["label"] == "CLEAR" else "[…]" for t in self._rolling])
                clear_words = [t["text"] for t in self._rolling if t["label"] == "CLEAR"]

                completed_line: Optional[str] = None
                if self.extender and any(t["label"] != "CLEAR" for t in self._rolling):
                    try:
                        completed_line = self.extender.fill_scaffold(
                            scaffolding=scaff,
                            clear_words=clear_words,
                            style=STYLE,
                        )
                    except Exception as e:
                        self.status_signal.emit(f"Extender error: {e}")

                arrow = " ".join(_color(t) for t in self._rolling)
                raw_line = " ".join(t["text"] for t in self._rolling)
                if completed_line is None or not (completed_line or "").strip():
                    completed_line = raw_line

                if arrow != self._last_arrow:
                    # also print to console for devs
                    print("→", arrow)
                    self.arrow_signal.emit(_strip_ansi(arrow))
                    self._last_arrow = arrow

                arrow_plain = " ".join(_strip_ansi(arrow).split())
                check_plain = " ".join((completed_line or "").split())
                if check_plain != arrow_plain and completed_line != self._last_check:
                    print("✓", completed_line)
                    self.check_signal.emit(completed_line)
                    self._last_check = completed_line
        finally:
            try:
                if self._stream:
                    self._stream.stop(); self._stream.close()
            except Exception:
                pass

# ─────────────────────────────────────────────
# Main Window – wires UI to LiveWorker
# ─────────────────────────────────────────────
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MumblAI — Live")
        self.resize(980, 660)
        self.setStyleSheet("QWidget{background:transparent}")

        self._worker: Optional[LiveWorker] = None

        # background
        self.bg = Background(self)
        self.bg.lower()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        card = FrostCard()
        card_l = QtWidgets.QVBoxLayout(card)
        card_l.setContentsMargins(20, 18, 18, 18)
        card_l.setSpacing(12)

        # header
        header = QtWidgets.QHBoxLayout()
        avatar = QtWidgets.QLabel(); avatar.setFixedSize(40, 40)
        avatar.setStyleSheet(
            "border-radius:20px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, "
            "stop:0 #2a4157, stop:1 #20343f); border:1px solid rgba(255,255,255,.08)"
        )
        title_box = QtWidgets.QVBoxLayout(); title_box.setSpacing(2); title_box.setContentsMargins(0,0,0,0)
        title = QtWidgets.QLabel("MumblAI")
        title.setStyleSheet("font-weight:700; font-size:18px; color:#f2f4f8; letter-spacing:.2px")
        self.sub = QtWidgets.QLabel("Live transcript")
        self.sub.setStyleSheet("color:#c2c7d0; font-size:12px")
        title_box.addWidget(title); title_box.addWidget(self.sub)
        header.addWidget(avatar); header.addLayout(title_box); header.addStretch()

        self.dot = QtWidgets.QLabel(); self.dot.setFixedSize(10,10)
        self.dot.setStyleSheet("border-radius:5px; background:#3fb950")
        header.addWidget(self.dot)
        card_l.addLayout(header)

        # transcript area
        transcript_wrap = QtWidgets.QFrame()
        transcript_wrap.setStyleSheet("background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:14px")
        tw_l = QtWidgets.QVBoxLayout(transcript_wrap)
        tw_l.setContentsMargins(14,14,14,14)
        self.transcript = TranscriptView(); tw_l.addWidget(self.transcript)
        card_l.addWidget(transcript_wrap, 1)

        # voice dock
        self.button = RecorderButton()
        self.voice = VoiceDock(self.button)
        card_l.addWidget(self.voice)

        # footer
        footer = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("color:#c2c7d0; font-size:12px")
        hint = QtWidgets.QLabel("Space to toggle • Esc to stop")
        hint.setStyleSheet("color:#8b90a0; font-size:12px")
        footer.addWidget(self.status); footer.addStretch(); footer.addWidget(hint)
        card_l.addLayout(footer)

        root.addWidget(card, 1)

        # wiring
        self.button.toggled.connect(self.on_toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.button.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, activated=lambda: (self.button.setChecked(False), self.on_toggle(False)))

    # resize background to window
    def resizeEvent(self, ev):
        self.bg.resize(self.size())
        return super().resizeEvent(ev)

    # start/stop
    @QtCore.Slot(bool)
    def on_toggle(self, checked: bool):
        if checked:
            self.start_worker()
        else:
            self.stop_worker()

    def start_worker(self):
        if self._worker and self._worker.isRunning():
            return
        self.transcript.reset()
        self.status.setText("Starting…")
        self.sub.setText("Live transcript")
        self.dot.setStyleSheet("border-radius:5px; background:#ff6159")
        self.voice.set_recording(True)

        self._worker = LiveWorker()
        # connect signals → slots
        self._worker.token_signal.connect(self.transcript.append_word)
        self._worker.rms_signal.connect(self.voice.meter.set_level)
        self._worker.status_signal.connect(self.status.setText)
        self._worker.started_ok.connect(lambda s: self.status.setText(f"Recording ({s})"))
        self._worker.error_signal.connect(self.on_error)
        self._worker.finished.connect(self.on_finished)
        # optional debug lines
        # self._worker.arrow_signal.connect(lambda s: print("UI ← →", s))
        # self._worker.check_signal.connect(lambda s: print("UI ← ✓", s))
        self._worker.start()

    def stop_worker(self):
        self.status.setText("Stopping…")
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None
        self.status.setText("Stopped")
        self.dot.setStyleSheet("border-radius:5px; background:#3fb950")
        self.voice.set_recording(False)

    @QtCore.Slot()
    def on_finished(self):
        self.dot.setStyleSheet("border-radius:5px; background:#3fb950")
        self.voice.set_recording(False)
        if self.button.isChecked():
            self.button.setChecked(False)
        self.status.setText("Stopped")

    @QtCore.Slot(str)
    def on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Audio Error", msg)
        self.on_finished()

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    # HiDPI polish
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MumblAI — Live")
    # app.setProperty("REDUCED_MOTION", "1")  # optional
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
