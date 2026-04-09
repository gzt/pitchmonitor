#!/usr/bin/env python3
"""
Vocal Pitch Monitor
A real-time pitch detection and visualization tool.
Requires: numpy, scipy, sounddevice, matplotlib
Install: pip install numpy scipy sounddevice matplotlib
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import scipy.fft as fft
import threading
import queue
import time
import math
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

# ── Tuning & Music Theory ────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F",
              "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]

# Major scale intervals (semitones from root)
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
# Natural minor intervals
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
# Chromatic = all 12
CHROMATIC_INTERVALS = list(range(12))

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_TO_SEMITONE = {k: i for i, k in enumerate(KEYS)}

def freq_to_midi(freq, a4=440.0):
    if freq <= 0:
        return None
    return 69 + 12 * math.log2(freq / a4)

def midi_to_freq(midi, a4=440.0):
    return a4 * 2 ** ((midi - 69) / 12)

def freq_to_note_info(freq, a4=440.0):
    """Return (note_name, octave, nearest_freq, cents_offset)."""
    if freq <= 0:
        return None
    midi = freq_to_midi(freq, a4)
    nearest_midi = round(midi)
    cents = (midi - nearest_midi) * 100
    note_idx = nearest_midi % 12
    octave = (nearest_midi // 12) - 1
    note_name = NOTE_NAMES[note_idx]
    nearest_freq = midi_to_freq(nearest_midi, a4)
    return note_name, octave, nearest_freq, cents

def get_scale_freqs(key, scale_type, a4=440.0, octave_range=(2, 6)):
    """Return list of frequencies for the given key/scale across octave range."""
    root_semitone = KEY_TO_SEMITONE[key]
    intervals = {"Major": MAJOR_INTERVALS,
                 "Minor": MINOR_INTERVALS,
                 "Chromatic": CHROMATIC_INTERVALS}[scale_type]
    freqs = []
    for oct in range(octave_range[0], octave_range[1] + 1):
        for interval in intervals:
            semitone = root_semitone + interval
            midi = (oct + 1) * 12 + semitone
            freqs.append(midi_to_freq(midi, a4))
    return freqs

def get_scale_note_names(key, scale_type):
    """Return list of note names in the scale."""
    root_semitone = KEY_TO_SEMITONE[key]
    intervals = {"Major": MAJOR_INTERVALS,
                 "Minor": MINOR_INTERVALS,
                 "Chromatic": CHROMATIC_INTERVALS}[scale_type]
    return [NOTE_NAMES[(root_semitone + i) % 12] for i in intervals]

# ── Pitch Detection ──────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
CHUNK = 2048          # ~46ms per chunk
HISTORY_SECONDS = 4
HISTORY_LEN = int(HISTORY_SECONDS * SAMPLE_RATE / CHUNK)

MIN_FREQ = 60.0       # Hz — below this we ignore
MAX_FREQ = 1200.0     # Hz

def detect_pitch_yin(audio, sr, threshold=0.15):
    """YIN algorithm for robust pitch detection."""
    N = len(audio)
    tau_min = int(sr / MAX_FREQ)
    tau_max = int(sr / MIN_FREQ)
    tau_max = min(tau_max, N // 2)

    # Step 1: difference function
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff[tau] = np.sum((audio[:N - tau] - audio[tau:N]) ** 2)

    # Step 2: cumulative mean normalized difference
    cmnd = np.zeros(tau_max)
    cmnd[0] = 1
    running_sum = 0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmnd[tau] = diff[tau] * tau / running_sum if running_sum > 0 else 1

    # Step 3: absolute threshold
    tau_est = tau_min
    found = False
    for tau in range(tau_min, tau_max - 1):
        if cmnd[tau] < threshold:
            # Step 4: local minimum
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            found = True
            break

    if not found:
        return 0.0, 0.0

    # Step 5: parabolic interpolation
    if 0 < tau_est < tau_max - 1:
        s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        tau_interp = tau_est + (s2 - s0) / (2 * (2 * s1 - s2 - s0) + 1e-10)
    else:
        tau_interp = tau_est

    freq = sr / tau_interp if tau_interp > 0 else 0
    confidence = 1.0 - cmnd[tau_est]
    return freq, confidence


# ── Main Application ─────────────────────────────────────────────────────────

class VocalPitchMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Vocal Pitch Monitor")
        self.root.configure(bg="#0d0f14")
        self.root.minsize(900, 600)

        # State
        self.a4 = tk.DoubleVar(value=440.0)
        self.key = tk.StringVar(value="C")
        self.scale_type = tk.StringVar(value="Chromatic")
        self.running = False
        self.audio_queue = queue.Queue(maxsize=20)
        self.freq_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.conf_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.current_freq = 0.0
        self.current_conf = 0.0
        self.stream = None
        self.devices = self._get_input_devices()
        self.selected_device = tk.IntVar(value=self._default_device())
        self.conf_threshold = tk.DoubleVar(value=0.7)

        self._build_ui()
        self._start_audio()
        self._schedule_update()

    # ── Device helpers ────────────────────────────────────────────────────

    def _get_input_devices(self):
        devs = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                devs.append((i, d['name']))
        return devs

    def _default_device(self):
        try:
            return sd.default.device[0]
        except Exception:
            return self.devices[0][0] if self.devices else 0

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        BG = "#0d0f14"
        PANEL = "#13161e"
        ACCENT = "#00e5a0"
        DIM = "#2a2f3d"
        TEXT = "#e8eaf0"
        MUTED = "#6b7280"

        self.colors = dict(bg=BG, panel=PANEL, accent=ACCENT, dim=DIM,
                           text=TEXT, muted=MUTED)

        # ── Top info bar ──────────────────────────────────────────────────
        info_frame = tk.Frame(self.root, bg=PANEL, pady=10)
        info_frame.pack(fill="x", padx=0, pady=(0, 2))

        self.freq_label = tk.Label(info_frame, text="— Hz",
            font=("Courier New", 38, "bold"), fg=ACCENT, bg=PANEL)
        self.freq_label.pack(side="left", padx=24)

        note_block = tk.Frame(info_frame, bg=PANEL)
        note_block.pack(side="left", padx=16)
        self.note_label = tk.Label(note_block, text="—",
            font=("Courier New", 28, "bold"), fg=TEXT, bg=PANEL)
        self.note_label.pack(anchor="w")
        self.octave_label = tk.Label(note_block, text="octave —",
            font=("Courier New", 11), fg=MUTED, bg=PANEL)
        self.octave_label.pack(anchor="w")

        divider = tk.Frame(info_frame, bg=DIM, width=1)
        divider.pack(side="left", fill="y", padx=16, pady=4)

        detail_block = tk.Frame(info_frame, bg=PANEL)
        detail_block.pack(side="left")
        self.nearest_label = tk.Label(detail_block, text="nearest: — Hz",
            font=("Courier New", 13), fg=MUTED, bg=PANEL)
        self.nearest_label.pack(anchor="w")
        self.cents_label = tk.Label(detail_block, text="cents: ±0",
            font=("Courier New", 13), fg=MUTED, bg=PANEL)
        self.cents_label.pack(anchor="w")

        # Cents bar (visual tuner meter)
        self.cents_canvas = tk.Canvas(info_frame, width=220, height=48,
            bg=PANEL, highlightthickness=0)
        self.cents_canvas.pack(side="left", padx=24)
        self._draw_cents_bar(0)

        # Confidence indicator
        conf_block = tk.Frame(info_frame, bg=PANEL)
        conf_block.pack(side="right", padx=24)
        tk.Label(conf_block, text="CONFIDENCE", font=("Courier New", 9),
            fg=MUTED, bg=PANEL).pack()
        self.conf_bar = tk.Canvas(conf_block, width=80, height=14,
            bg=DIM, highlightthickness=0)
        self.conf_bar.pack(pady=2)
        self.conf_pct_label = tk.Label(conf_block, text="0%",
            font=("Courier New", 10), fg=MUTED, bg=PANEL)
        self.conf_pct_label.pack()

        # ── Controls bar ──────────────────────────────────────────────────
        ctrl_frame = tk.Frame(self.root, bg=BG, pady=4)
        ctrl_frame.pack(fill="x", padx=12)

        def lbl(parent, text):
            return tk.Label(parent, text=text,
                font=("Courier New", 10), fg=MUTED, bg=BG)

        lbl(ctrl_frame, "KEY").pack(side="left", padx=(0, 4))
        key_combo = ttk.Combobox(ctrl_frame, textvariable=self.key,
            values=KEYS, width=4, state="readonly", font=("Courier New", 10))
        key_combo.pack(side="left", padx=(0, 12))
        key_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        lbl(ctrl_frame, "SCALE").pack(side="left", padx=(0, 4))
        scale_combo = ttk.Combobox(ctrl_frame, textvariable=self.scale_type,
            values=["Chromatic", "Major", "Minor"], width=9,
            state="readonly", font=("Courier New", 10))
        scale_combo.pack(side="left", padx=(0, 12))
        scale_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        lbl(ctrl_frame, "A4 =").pack(side="left", padx=(0, 4))
        a4_spin = tk.Spinbox(ctrl_frame, from_=400, to=480,
            textvariable=self.a4, width=5, font=("Courier New", 10),
            bg=PANEL, fg=TEXT, buttonbackground=DIM,
            command=self._refresh_plot)
        a4_spin.pack(side="left", padx=(0, 4))
        a4_spin.bind("<Return>", lambda e: self._refresh_plot())
        lbl(ctrl_frame, "Hz").pack(side="left", padx=(0, 20))

        lbl(ctrl_frame, "DEVICE").pack(side="left", padx=(0, 4))
        device_names = [f"{i}: {n[:28]}" for i, n in self.devices]
        self.device_combo = ttk.Combobox(ctrl_frame, values=device_names,
            width=28, state="readonly", font=("Courier New", 9))
        # Try to select default
        for idx, (dev_id, _) in enumerate(self.devices):
            if dev_id == self.selected_device.get():
                self.device_combo.current(idx)
                break
        self.device_combo.pack(side="left", padx=(0, 12))
        self.device_combo.bind("<<ComboboxSelected>>", self._change_device)

        self.start_btn = tk.Button(ctrl_frame, text="■ STOP",
            command=self._toggle_audio,
            font=("Courier New", 10, "bold"), fg="#ff4f6e", bg=PANEL,
            relief="flat", padx=10, cursor="hand2")
        self.start_btn.pack(side="right", padx=4)

        # Style comboboxes
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL,
            background=PANEL, foreground=TEXT, selectbackground=DIM,
            arrowcolor=ACCENT)

        # ── Matplotlib plot ───────────────────────────────────────────────
        self.fig = Figure(figsize=(10, 5), facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
            padx=0, pady=(2, 0))

        self._init_plot_elements()
        self._refresh_plot()

    def _style_axes(self):
        BG = self.colors['bg']
        DIM = self.colors['dim']
        MUTED = self.colors['muted']
        ax = self.ax
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8, length=2)
        for spine in ax.spines.values():
            spine.set_color(DIM)
        ax.set_xlabel("time (s)", color=MUTED, fontsize=8,
            fontfamily="monospace")
        ax.set_ylabel("frequency (Hz)", color=MUTED, fontsize=8,
            fontfamily="monospace")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    def _init_plot_elements(self):
        """Create the persistent line and scatter objects."""
        ACCENT = self.colors['accent']
        MUTED = self.colors['muted']
        t = np.linspace(-HISTORY_SECONDS, 0, HISTORY_LEN)
        freqs = np.zeros(HISTORY_LEN)

        self.pitch_line, = self.ax.plot(
            t, np.where(freqs > 0, freqs, np.nan),
            color=ACCENT, linewidth=1.6, alpha=0.9, zorder=5,
            solid_capstyle="round")

        # Current position dot
        self.current_dot, = self.ax.plot(
            [0], [0], 'o', color=ACCENT, markersize=7,
            markeredgecolor="#ffffff30", zorder=6)

        self.scale_lines = []   # horizontal pitch guide lines
        self.scale_labels = []

    def _refresh_plot(self, *args):
        """Redraw scale lines and reset axis limits."""
        a4 = float(self.a4.get())
        key = self.key.get()
        scale_type = self.scale_type.get()

        # Remove old scale lines / labels
        for ln in self.scale_lines:
            ln.remove()
        for lb in self.scale_labels:
            lb.remove()
        self.scale_lines.clear()
        self.scale_labels.clear()

        # Y axis: log-ish range for vocal/instrument range
        y_min, y_max = 60, 1100
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_yscale("log")
        self.ax.set_xlim(-HISTORY_SECONDS, 0.1)

        # Freq → note label helper for y-axis
        self.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(self._freq_formatter))

        scale_freqs = get_scale_freqs(key, scale_type, a4, (2, 6))
        scale_note_names = get_scale_note_names(key, scale_type)

        DIM = self.colors['dim']
        MUTED = self.colors['muted']
        ACCENT = self.colors['accent']

        is_chromatic = scale_type == "Chromatic"

        for freq in scale_freqs:
            if not (y_min <= freq <= y_max):
                continue
            # Color: root note gets accent highlight
            info = freq_to_note_info(freq, a4)
            if info:
                is_root = (info[0].split("/")[0] == key.replace("#", "♯"))
                color = ACCENT if is_root else DIM
                alpha = 0.9 if is_root else (0.35 if is_chromatic else 0.55)
                lw = 1.0 if is_root else (0.4 if is_chromatic else 0.6)
            else:
                color = DIM; alpha = 0.35; lw = 0.5

            ln = self.ax.axhline(y=freq, color=color, linewidth=lw,
                alpha=alpha, zorder=1, linestyle="--" if is_root else "-")
            self.scale_lines.append(ln)

            # Label rightmost edge (only for non-chromatic, or roots)
            if not is_chromatic or (info and is_root):
                if info:
                    note_str = f"{info[0]}{info[1]}"
                    lb = self.ax.text(0.08, freq, note_str,
                        transform=self.ax.get_yaxis_transform(),
                        fontsize=6.5, color=color, alpha=min(alpha + 0.2, 1),
                        va="center", fontfamily="monospace", zorder=7)
                    self.scale_labels.append(lb)

        self.canvas.draw_idle()

    def _freq_formatter(self, val, pos):
        info = freq_to_note_info(val, float(self.a4.get()))
        if info:
            return f"{info[0]}{info[1]}"
        return f"{val:.0f}"

    # ── Cents bar ─────────────────────────────────────────────────────────

    def _draw_cents_bar(self, cents):
        c = self.cents_canvas
        c.delete("all")
        W, H = 220, 48
        mid = W // 2
        ACCENT = self.colors['accent']
        DIM = self.colors['dim']
        MUTED = self.colors['muted']

        # Background track
        c.create_rectangle(10, H//2 - 3, W - 10, H//2 + 3,
            fill=DIM, outline="")
        # Center mark
        c.create_rectangle(mid - 1, H//2 - 8, mid + 1, H//2 + 8,
            fill="#ffffff40", outline="")

        # Marker position: ±50 cents maps to ±95px from center
        clamped = max(-50, min(50, cents))
        px = mid + int(clamped / 50 * 95)

        # Color: green when in tune (<5c), yellow (<15c), red otherwise
        abs_c = abs(clamped)
        color = ACCENT if abs_c < 8 else ("#f0c040" if abs_c < 20 else "#ff4f6e")

        # Bar from center to marker
        if px != mid:
            c.create_rectangle(min(mid, px), H//2 - 4,
                max(mid, px), H//2 + 4, fill=color, outline="")

        # Marker
        c.create_oval(px - 7, H//2 - 7, px + 7, H//2 + 7,
            fill=color, outline="#ffffff20")

        # Labels
        c.create_text(10, H - 6, text="-50¢", fill=MUTED,
            font=("Courier New", 7), anchor="w")
        c.create_text(W // 2, H - 6, text="0", fill=MUTED,
            font=("Courier New", 7), anchor="center")
        c.create_text(W - 10, H - 6, text="+50¢", fill=MUTED,
            font=("Courier New", 7), anchor="e")

    # ── Audio I/O ─────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        if not self.audio_queue.full():
            self.audio_queue.put_nowait(mono)

    def _start_audio(self):
        dev_id = self.selected_device.get()
        try:
            self.stream = sd.InputStream(
                device=dev_id,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK,
                callback=self._audio_callback)
            self.stream.start()
            self.running = True
        except Exception as e:
            print(f"Audio error: {e}")
            self.running = False

    def _stop_audio(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _toggle_audio(self):
        if self.running:
            self._stop_audio()
            self.start_btn.config(text="▶ START", fg=self.colors['accent'])
        else:
            self._start_audio()
            self.start_btn.config(text="■ STOP", fg="#ff4f6e")

    def _change_device(self, event):
        sel = self.device_combo.current()
        if 0 <= sel < len(self.devices):
            dev_id = self.devices[sel][0]
            self.selected_device.set(dev_id)
            was_running = self.running
            self._stop_audio()
            if was_running:
                self._start_audio()

    # ── Update loop ───────────────────────────────────────────────────────

    def _process_audio(self):
        """Drain queue, detect pitch, update history."""
        chunks_processed = 0
        while not self.audio_queue.empty() and chunks_processed < 4:
            audio = self.audio_queue.get_nowait()
            chunks_processed += 1

            # Hann window
            windowed = audio * np.hanning(len(audio))
            rms = np.sqrt(np.mean(windowed ** 2))

            if rms < 0.003:  # silence threshold
                freq, conf = 0.0, 0.0
            else:
                freq, conf = detect_pitch_yin(windowed, SAMPLE_RATE)
                if not (MIN_FREQ <= freq <= MAX_FREQ):
                    freq, conf = 0.0, 0.0

            self.freq_history.append(freq)
            self.conf_history.append(conf)
            self.current_freq = freq
            self.current_conf = conf

    def _update_ui(self):
        freq = self.current_freq
        conf = self.current_conf
        a4 = float(self.a4.get())
        ACCENT = self.colors['accent']
        MUTED = self.colors['muted']
        TEXT = self.colors['text']

        active = freq > 0 and conf >= float(self.conf_threshold.get())

        if active:
            info = freq_to_note_info(freq, a4)
            self.freq_label.config(
                text=f"{freq:.1f} Hz", fg=ACCENT)
            if info:
                note_name, octave, nearest_freq, cents = info
                self.note_label.config(text=note_name, fg=TEXT)
                self.octave_label.config(text=f"octave {octave}")
                self.nearest_label.config(
                    text=f"nearest: {nearest_freq:.2f} Hz  ({note_name}{octave})",
                    fg=MUTED)
                sign = "+" if cents >= 0 else ""
                cents_color = (ACCENT if abs(cents) < 8
                               else ("#f0c040" if abs(cents) < 20 else "#ff4f6e"))
                self.cents_label.config(
                    text=f"cents: {sign}{cents:.1f}¢",
                    fg=cents_color)
                self._draw_cents_bar(cents)
        else:
            self.freq_label.config(text="— Hz", fg=ACCENT)
            self.note_label.config(text="—", fg="#3a3f52")
            self.octave_label.config(text="octave —", fg=MUTED)
            self.nearest_label.config(text="nearest: — Hz", fg=MUTED)
            self.cents_label.config(text="cents: ±0", fg=MUTED)
            self._draw_cents_bar(0)

        # Confidence bar
        self.conf_bar.delete("all")
        pct = conf
        W = int(80 * min(pct, 1.0))
        color = ACCENT if pct > 0.75 else ("#f0c040" if pct > 0.5 else "#ff4f6e")
        if W > 0:
            self.conf_bar.create_rectangle(0, 0, W, 14, fill=color, outline="")
        self.conf_pct_label.config(text=f"{int(pct * 100)}%",
            fg=MUTED if pct < 0.3 else TEXT)

    def _update_plot(self):
        freqs = np.array(self.freq_history)
        confs = np.array(self.conf_history)
        t = np.linspace(-HISTORY_SECONDS, 0, HISTORY_LEN)

        conf_thresh = float(self.conf_threshold.get())
        display = np.where((freqs > 0) & (confs >= conf_thresh),
                           freqs, np.nan)

        self.pitch_line.set_data(t, display)

        # Current dot
        if self.current_freq > 0 and self.current_conf >= conf_thresh:
            self.current_dot.set_data([0], [self.current_freq])
            self.current_dot.set_visible(True)
        else:
            self.current_dot.set_visible(False)

        self.canvas.draw_idle()

    def _schedule_update(self):
        self._process_audio()
        self._update_ui()
        self._update_plot()
        self.root.after(50, self._schedule_update)  # ~20 fps

    def on_close(self):
        self._stop_audio()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = VocalPitchMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
