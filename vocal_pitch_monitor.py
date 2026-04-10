#!/usr/bin/env python3
"""
Vocal Pitch Monitor
Real-time pitch detection and visualization.
Requires: numpy, scipy, sounddevice, matplotlib
Install:  pip install numpy scipy sounddevice matplotlib
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import threading
import queue
import math
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker

# ── Music Theory ─────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F",
              "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]

MAJOR_INTERVALS     = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS     = [0, 2, 3, 5, 7, 8, 10]
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
    """Return (note_name, octave, nearest_freq, cents_offset) or None."""
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

def get_scale_freqs(key, scale_type, a4=440.0, octave_range=(3, 5)):
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

# ── Pitch Detection ───────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
CHUNK       = 4096      # ~93 ms per chunk

MIN_FREQ = 60.0
MAX_FREQ = 1200.0

def detect_pitch(audio, sr):
    """
    FFT autocorrelation pitch detection + parabolic interpolation.
    Returns (freq_hz, confidence 0..1). Fully vectorised.
    """
    N = len(audio)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return 0.0, 0.0
    audio = audio / peak

    X   = np.fft.rfft(audio, n=N * 2)
    acf = np.fft.irfft(X * np.conj(X))[:N]
    z   = acf[0]
    if z < 1e-10:
        return 0.0, 0.0
    acf /= z

    lag_min = max(1, int(sr / MAX_FREQ))
    lag_max = min(N - 1, int(sr / MIN_FREQ))
    if lag_min >= lag_max:
        return 0.0, 0.0

    search     = acf[lag_min:lag_max + 1]
    best_idx   = int(np.argmax(search))
    confidence = float(np.clip(search[best_idx], 0.0, 1.0))

    if confidence < 0.2:
        return 0.0, confidence

    abs_idx = best_idx + lag_min
    if 0 < abs_idx < N - 1:
        y0, y1, y2 = acf[abs_idx - 1], acf[abs_idx], acf[abs_idx + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        offset = (y2 - y0) / denom if abs(denom) > 1e-10 else 0.0
        lag = abs_idx + offset
    else:
        lag = float(abs_idx)

    freq = sr / lag if lag > 0 else 0.0
    return freq, confidence

# ── Smoothing ─────────────────────────────────────────────────────────────────

class PitchSmoother:
    """
    EMA on the MIDI scale (perceptually uniform) with gap-aware hold.
    alpha=1.0  -> no smoothing (raw values)
    alpha=0.08 -> very heavy smoothing
    """
    def __init__(self, alpha=0.25, hold_frames=3):
        self.alpha        = alpha
        self.hold_frames  = hold_frames
        self._midi_smooth = None
        self._silent_n    = 0

    def update(self, freq, conf, conf_thresh):
        active = freq > 0 and conf >= conf_thresh
        if active:
            self._silent_n = 0
            midi_raw = freq_to_midi(freq)
            if self._midi_smooth is None:
                self._midi_smooth = midi_raw
            else:
                self._midi_smooth += self.alpha * (midi_raw - self._midi_smooth)
            return midi_to_freq(self._midi_smooth), conf
        else:
            self._silent_n += 1
            if self._silent_n >= self.hold_frames:
                self._midi_smooth = None
            return 0.0, conf

# ── Main Application ──────────────────────────────────────────────────────────

DEFAULT_HISTORY_SEC = 20
DEFAULT_OCT_LO      = 3
DEFAULT_OCT_HI      = 5

class VocalPitchMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Vocal Pitch Monitor")
        self.root.configure(bg="#0d0f14")
        self.root.minsize(900, 600)

        self.a4          = tk.DoubleVar(value=440.0)
        self.key         = tk.StringVar(value="C")
        self.scale_type  = tk.StringVar(value="Chromatic")
        self.conf_thresh = tk.DoubleVar(value=0.30)
        self.history_sec = tk.IntVar(value=DEFAULT_HISTORY_SEC)
        self.oct_lo      = tk.IntVar(value=DEFAULT_OCT_LO)
        self.oct_hi      = tk.IntVar(value=DEFAULT_OCT_HI)

        self.running         = False
        self._worker_running = False
        self._worker_thread  = None
        self.stream          = None
        self.audio_queue     = queue.Queue(maxsize=40)
        self.devices         = self._get_input_devices()
        self.selected_device = tk.IntVar(value=self._default_device())

        self._history_len = self._calc_history_len()
        self.freq_history = deque([0.0] * self._history_len,
                                  maxlen=self._history_len)
        self.conf_history = deque([0.0] * self._history_len,
                                  maxlen=self._history_len)
        self.smoother     = PitchSmoother(alpha=0.25)
        self.current_freq = 0.0
        self.current_conf = 0.0

        self._build_ui()
        self._start_audio()
        self._schedule_update()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _calc_history_len(self):
        return max(10, int(self.history_sec.get() * SAMPLE_RATE / CHUNK))

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

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        BG     = "#0d0f14"
        PANEL  = "#13161e"
        ACCENT = "#00e5a0"
        DIM    = "#2a2f3d"
        TEXT   = "#e8eaf0"
        MUTED  = "#6b7280"
        self.colors = dict(bg=BG, panel=PANEL, accent=ACCENT,
                           dim=DIM, text=TEXT, muted=MUTED)

        # Info bar
        info = tk.Frame(self.root, bg=PANEL, pady=10)
        info.pack(fill="x", pady=(0, 2))

        self.freq_label = tk.Label(info, text="--- Hz",
            font=("Courier New", 38, "bold"), fg=ACCENT, bg=PANEL)
        self.freq_label.pack(side="left", padx=24)

        nb = tk.Frame(info, bg=PANEL)
        nb.pack(side="left", padx=16)
        self.note_label = tk.Label(nb, text="--",
            font=("Courier New", 28, "bold"), fg=TEXT, bg=PANEL)
        self.note_label.pack(anchor="w")
        self.octave_label = tk.Label(nb, text="octave --",
            font=("Courier New", 11), fg=MUTED, bg=PANEL)
        self.octave_label.pack(anchor="w")

        tk.Frame(info, bg=DIM, width=1).pack(
            side="left", fill="y", padx=16, pady=4)

        db = tk.Frame(info, bg=PANEL)
        db.pack(side="left")
        self.nearest_label = tk.Label(db, text="nearest: --- Hz",
            font=("Courier New", 13), fg=MUTED, bg=PANEL)
        self.nearest_label.pack(anchor="w")
        self.cents_label = tk.Label(db, text="cents: +/-0",
            font=("Courier New", 13), fg=MUTED, bg=PANEL)
        self.cents_label.pack(anchor="w")

        self.cents_canvas = tk.Canvas(info, width=220, height=48,
            bg=PANEL, highlightthickness=0)
        self.cents_canvas.pack(side="left", padx=24)
        self._draw_cents_bar(0)

        cb = tk.Frame(info, bg=PANEL)
        cb.pack(side="right", padx=24)
        tk.Label(cb, text="CONFIDENCE",
            font=("Courier New", 9), fg=MUTED, bg=PANEL).pack()
        self.conf_bar = tk.Canvas(cb, width=80, height=14,
            bg=DIM, highlightthickness=0)
        self.conf_bar.pack(pady=2)
        self.conf_pct_label = tk.Label(cb, text="0%",
            font=("Courier New", 10), fg=MUTED, bg=PANEL)
        self.conf_pct_label.pack()

        # Controls
        ctrl = tk.Frame(self.root, bg=BG, pady=4)
        ctrl.pack(fill="x", padx=12)

        def lbl(text, pad=(0, 4)):
            tk.Label(ctrl, text=text, font=("Courier New", 10),
                     fg=MUTED, bg=BG).pack(side="left", padx=pad)

        lbl("KEY")
        kc = ttk.Combobox(ctrl, textvariable=self.key, values=KEYS,
                          width=4, state="readonly",
                          font=("Courier New", 10))
        kc.pack(side="left", padx=(0, 10))
        kc.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        lbl("SCALE")
        sc = ttk.Combobox(ctrl, textvariable=self.scale_type,
                          values=["Chromatic", "Major", "Minor"],
                          width=9, state="readonly",
                          font=("Courier New", 10))
        sc.pack(side="left", padx=(0, 10))
        sc.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        lbl("A4 =")
        tk.Spinbox(ctrl, from_=400, to=480, textvariable=self.a4,
                   width=5, font=("Courier New", 10),
                   bg=PANEL, fg=TEXT, buttonbackground=DIM,
                   command=self._refresh_plot).pack(side="left")
        lbl("Hz", pad=(2, 14))

        lbl("SMOOTH")
        self.smooth_cb = ttk.Combobox(ctrl,
            values=["0.08", "0.15", "0.25", "0.40", "0.65", "1.0"],
            width=5, font=("Courier New", 10))
        self.smooth_cb.set("0.25")
        self.smooth_cb.pack(side="left", padx=(0, 14))
        self.smooth_cb.bind("<<ComboboxSelected>>",
            lambda e: self._set_smooth(self.smooth_cb.get()))
        self.smooth_cb.bind("<Return>",
            lambda e: self._set_smooth(self.smooth_cb.get()))

        lbl("HISTORY")
        self.hist_cb = ttk.Combobox(ctrl,
            values=["10", "15", "20", "30", "60"],
            width=4, font=("Courier New", 10))
        self.hist_cb.set(str(DEFAULT_HISTORY_SEC))
        self.hist_cb.pack(side="left")
        lbl("s", pad=(2, 14))
        self.hist_cb.bind("<<ComboboxSelected>>",
            lambda e: self._set_history(self.hist_cb.get()))
        self.hist_cb.bind("<Return>",
            lambda e: self._set_history(self.hist_cb.get()))

        lbl("OCT")
        tk.Spinbox(ctrl, from_=1, to=7, textvariable=self.oct_lo,
                   width=2, font=("Courier New", 10),
                   bg=PANEL, fg=TEXT, buttonbackground=DIM,
                   command=self._refresh_plot).pack(side="left")
        lbl("-", pad=(2, 2))
        tk.Spinbox(ctrl, from_=1, to=7, textvariable=self.oct_hi,
                   width=2, font=("Courier New", 10),
                   bg=PANEL, fg=TEXT, buttonbackground=DIM,
                   command=self._refresh_plot).pack(side="left", padx=(0, 14))

        lbl("DEVICE")
        device_names = [f"{i}: {n[:28]}" for i, n in self.devices]
        self.device_combo = ttk.Combobox(ctrl, values=device_names,
            width=26, state="readonly", font=("Courier New", 9))
        default_id = self.selected_device.get()
        for idx, (dev_id, _) in enumerate(self.devices):
            if dev_id == default_id:
                self.device_combo.current(idx)
                break
        else:
            if self.devices:
                self.device_combo.current(0)
        self.device_combo.pack(side="left", padx=(0, 10))
        self.device_combo.bind("<<ComboboxSelected>>", self._change_device)

        self.start_btn = tk.Button(ctrl, text="STOP",
            command=self._toggle_audio,
            font=("Courier New", 10, "bold"), fg="#ff4f6e", bg=PANEL,
            relief="flat", padx=10, cursor="hand2")
        self.start_btn.pack(side="right")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
            fieldbackground=PANEL, background=PANEL,
            foreground=TEXT, selectbackground=DIM, arrowcolor=ACCENT)

        # Plot
        self.fig = Figure(figsize=(10, 5), facecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._init_plot_elements()
        self._refresh_plot()

    # ── Control callbacks ─────────────────────────────────────────────────

    def _set_smooth(self, val):
        try:
            self.smoother.alpha = max(0.01, min(1.0, float(val)))
        except ValueError:
            pass

    def _set_history(self, val):
        try:
            secs    = max(5, int(val))
            new_len = max(10, int(secs * SAMPLE_RATE / CHUNK))
            self.history_sec.set(secs)
            self._history_len = new_len
            self.freq_history = deque([0.0] * new_len, maxlen=new_len)
            self.conf_history = deque([0.0] * new_len, maxlen=new_len)
            self.ax.set_xlim(-secs, 0.1)
            self.canvas.draw_idle()
        except ValueError:
            pass

    # ── Plot helpers ──────────────────────────────────────────────────────

    def _style_axes(self):
        BG    = self.colors['bg']
        DIM   = self.colors['dim']
        MUTED = self.colors['muted']
        ax = self.ax
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8, length=2)
        for spine in ax.spines.values():
            spine.set_color(DIM)
        ax.set_xlabel("time (s)", color=MUTED, fontsize=8,
                      fontfamily="monospace")
        ax.set_ylabel("pitch", color=MUTED, fontsize=8,
                      fontfamily="monospace")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    def _init_plot_elements(self):
        ACCENT = self.colors['accent']
        n = self._history_len
        t = np.linspace(-self.history_sec.get(), 0, n)
        self.pitch_line, = self.ax.plot(
            t, np.full(n, np.nan),
            color=ACCENT, linewidth=1.8, alpha=0.9, zorder=5,
            solid_capstyle="round")
        self.current_dot, = self.ax.plot(
            [0], [200], 'o', color=ACCENT, markersize=7,
            markeredgecolor="#ffffff30", zorder=6, visible=False)
        self.scale_lines  = []
        self.scale_labels = []

    def _octave_freq_range(self):
        a4 = float(self.a4.get())
        lo = self.oct_lo.get()
        hi = self.oct_hi.get()
        if lo > hi:
            lo, hi = hi, lo
        y_min = midi_to_freq((lo + 1) * 12, a4) * 0.92
        y_max = midi_to_freq((hi + 1) * 12 + 11, a4) * 1.08
        return y_min, y_max

    def _refresh_plot(self, *_):
        a4         = float(self.a4.get())
        key        = self.key.get()
        scale_type = self.scale_type.get()
        lo         = self.oct_lo.get()
        hi         = self.oct_hi.get()
        if lo > hi:
            lo, hi = hi, lo

        for ln in self.scale_lines:  ln.remove()
        for lb in self.scale_labels: lb.remove()
        self.scale_lines.clear()
        self.scale_labels.clear()

        y_min, y_max = self._octave_freq_range()
        secs         = self.history_sec.get()

        self.ax.set_yscale("log")
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(-secs, 0.1)
        self.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(self._freq_formatter))

        scale_freqs  = get_scale_freqs(key, scale_type, a4, (lo - 1, hi + 1))
        is_chromatic = scale_type == "Chromatic"
        DIM    = self.colors['dim']
        ACCENT = self.colors['accent']

        for freq in scale_freqs:
            if not (y_min * 0.9 <= freq <= y_max * 1.1):
                continue
            info = freq_to_note_info(freq, a4)
            if info:
                is_root = info[0].split("/")[0] == key.replace("#", "#")
                color   = ACCENT if is_root else DIM
                alpha   = 0.9  if is_root else (0.28 if is_chromatic else 0.50)
                lw      = 1.0  if is_root else (0.35 if is_chromatic else 0.55)
                ls      = "--" if is_root else "-"
            else:
                color = DIM; alpha = 0.28; lw = 0.4; ls = "-"

            ln = self.ax.axhline(y=freq, color=color, linewidth=lw,
                                 alpha=alpha, zorder=1, linestyle=ls)
            self.scale_lines.append(ln)

            if not is_chromatic or (info and is_root):
                if info:
                    lb = self.ax.text(
                        0.08, freq, f"{info[0]}{info[1]}",
                        transform=self.ax.get_yaxis_transform(),
                        fontsize=6.5, color=color,
                        alpha=min(alpha + 0.2, 1.0),
                        va="center", fontfamily="monospace", zorder=7)
                    self.scale_labels.append(lb)

        self.canvas.draw_idle()

    def _freq_formatter(self, val, pos):
        info = freq_to_note_info(val, float(self.a4.get()))
        return f"{info[0]}{info[1]}" if info else f"{val:.0f}"

    # ── Cents bar ─────────────────────────────────────────────────────────

    def _draw_cents_bar(self, cents):
        c = self.cents_canvas
        c.delete("all")
        W, H   = 220, 48
        mid    = W // 2
        ACCENT = self.colors['accent']
        DIM    = self.colors['dim']
        MUTED  = self.colors['muted']

        c.create_rectangle(10, H//2 - 3, W - 10, H//2 + 3,
                           fill=DIM, outline="")
        c.create_rectangle(mid - 1, H//2 - 8, mid + 1, H//2 + 8,
                           fill="#ffffff", outline="")

        clamped = max(-50, min(50, cents))
        px      = mid + int(clamped / 50 * 95)
        abs_c   = abs(clamped)
        color   = (ACCENT if abs_c < 8 else
                   "#f0c040" if abs_c < 20 else "#ff4f6e")

        if px != mid:
            c.create_rectangle(min(mid, px), H//2 - 4,
                               max(mid, px), H//2 + 4,
                               fill=color, outline="")
        c.create_oval(px - 7, H//2 - 7, px + 7, H//2 + 7,
                      fill=color, outline="#ffffff")
        c.create_text(10,     H - 6, text="-50c", fill=MUTED,
                      font=("Courier New", 7), anchor="w")
        c.create_text(W // 2, H - 6, text="0",    fill=MUTED,
                      font=("Courier New", 7), anchor="center")
        c.create_text(W - 10, H - 6, text="+50c", fill=MUTED,
                      font=("Courier New", 7), anchor="e")

    # ── Audio ─────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        if not self.audio_queue.full():
            self.audio_queue.put_nowait(mono)

    def _pitch_worker(self):
        while self._worker_running:
            try:
                audio = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                freq, conf = 0.0, 0.0
            else:
                freq, conf = detect_pitch(audio, SAMPLE_RATE)
                if not (MIN_FREQ <= freq <= MAX_FREQ):
                    freq, conf = 0.0, 0.0

            freq_s, _ = self.smoother.update(
                freq, conf, self.conf_thresh.get())

            self.freq_history.append(freq_s)
            self.conf_history.append(conf)
            self.current_freq = freq_s
            self.current_conf = conf

    def _start_audio(self):
        sel = self.device_combo.current() if hasattr(self, 'device_combo') else -1
        if 0 <= sel < len(self.devices):
            dev_id = self.devices[sel][0]
        else:
            dev_id = self.selected_device.get()

        try:
            self.stream = sd.InputStream(
                device=dev_id,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK,
                callback=self._audio_callback)
            self.stream.start()
            self.running         = True
            self._worker_running = True
            self._worker_thread  = threading.Thread(
                target=self._pitch_worker, daemon=True)
            self._worker_thread.start()
            print(f"Audio started on device {dev_id}")
        except Exception as e:
            print(f"Audio error: {e}")
            self.running = False

    def _stop_audio(self):
        self.running         = False
        self._worker_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _toggle_audio(self):
        if self.running:
            self._stop_audio()
            self.start_btn.config(text="START", fg=self.colors['accent'])
        else:
            self._start_audio()
            self.start_btn.config(text="STOP", fg="#ff4f6e")

    def _change_device(self, event):
        sel = self.device_combo.current()
        if 0 <= sel < len(self.devices):
            self.selected_device.set(self.devices[sel][0])
            was_running = self.running
            self._stop_audio()
            if was_running:
                self._start_audio()

    # ── UI update loop ────────────────────────────────────────────────────

    def _update_ui(self):
        freq   = self.current_freq
        conf   = self.current_conf
        a4     = float(self.a4.get())
        ACCENT = self.colors['accent']
        MUTED  = self.colors['muted']
        TEXT   = self.colors['text']
        active = freq > 0 and conf >= self.conf_thresh.get()

        if active:
            info = freq_to_note_info(freq, a4)
            self.freq_label.config(text=f"{freq:.1f} Hz", fg=ACCENT)
            if info:
                note_name, octave, nearest_freq, cents = info
                self.note_label.config(text=note_name, fg=TEXT)
                self.octave_label.config(text=f"octave {octave}")
                self.nearest_label.config(
                    text=f"nearest: {nearest_freq:.2f} Hz  ({note_name}{octave})",
                    fg=MUTED)
                sign        = "+" if cents >= 0 else ""
                cents_color = (ACCENT    if abs(cents) < 8  else
                               "#f0c040" if abs(cents) < 20 else
                               "#ff4f6e")
                self.cents_label.config(
                    text=f"cents: {sign}{cents:.1f}", fg=cents_color)
                self._draw_cents_bar(cents)
        else:
            self.freq_label.config(text="--- Hz", fg=ACCENT)
            self.note_label.config(text="--", fg="#3a3f52")
            self.octave_label.config(text="octave --", fg=MUTED)
            self.nearest_label.config(text="nearest: --- Hz", fg=MUTED)
            self.cents_label.config(text="cents: +/-0", fg=MUTED)
            self._draw_cents_bar(0)

        self.conf_bar.delete("all")
        pct   = conf
        W     = int(80 * min(pct, 1.0))
        color = (ACCENT if pct > 0.75 else
                 "#f0c040" if pct > 0.5 else "#ff4f6e")
        if W > 0:
            self.conf_bar.create_rectangle(0, 0, W, 14,
                                           fill=color, outline="")
        self.conf_pct_label.config(
            text=f"{int(pct * 100)}%",
            fg=MUTED if pct < 0.3 else TEXT)

    def _update_plot(self):
        freqs   = np.array(self.freq_history)
        secs    = self.history_sec.get()
        n       = len(freqs)
        t       = np.linspace(-secs, 0, n)
        display = np.where(freqs > 0, freqs, np.nan)
        self.pitch_line.set_data(t, display)

        if self.current_freq > 0 and self.current_conf >= self.conf_thresh.get():
            self.current_dot.set_data([0], [self.current_freq])
            self.current_dot.set_visible(True)
        else:
            self.current_dot.set_visible(False)

        self.canvas.draw_idle()

    def _schedule_update(self):
        self._update_ui()
        self._update_plot()
        self.root.after(50, self._schedule_update)

    def on_close(self):
        self._stop_audio()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app  = VocalPitchMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
