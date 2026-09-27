"""
app.py
WaveCraft - interactive DSP audio workbench.

UI layer: Canvas-drawn rounded buttons, elevated cards, vector icons, a
responsive flow layout, custom EQ sliders, an animated splash screen and a
theme-matched matplotlib waveform.  No third-party UI libraries are needed:
everything below uses tkinter / ttk / matplotlib / numpy only.
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import time
import os
import threading
import math
import random
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import audio_ops as ao

try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except OSError:
    PLAYBACK_AVAILABLE = False


# ================================================================ Theme ====

APP_NAME = "WaveCraft"
FONT_FAMILY = "Segoe UI" if os.name == "nt" else "Helvetica"
FONT_MONO = "Consolas" if os.name == "nt" else "Courier"

# Same structure as before (light / dark dicts).  A few extra keys were added
# ("shadow", "shadow_a") for card elevation; every original key is still here.
THEMES = {
    "light": {
        "bg": "#eceef5",              # app canvas (cool grey-lavender)
        "bg_alt": "#f8f9fc",          # header / status bar surface
        "fg": "#1a1d2e",              # ink (not pure black)
        "fg_muted": "#5d667d",        # secondary text (>=4.5:1 on cards)
        "accent": "#5b45e0",          # single brand accent: indigo-violet
        "accent_hover": "#4b36cc",
        "accent_fg": "#ffffff",
        "border": "#dcdfeb",
        "sidebar_bg": "#f8f9fc",
        "sidebar_active": "#5b45e0",
        "entry_bg": "#f4f5fa",
        "plot_bg": "#f6f7fc",
        "plot_line": "#5b45e0",
        "plot_grid": "#dfe2ee",
        "seek_line": "#e5354f",       # playhead: coral-red
        "danger": "#d42a48",
        "card_bg": "#ffffff",
        "wave_sub": "#0a9db0",        # secondary signal colour: teal
        "shadow": "#39377a",
        "shadow_a": 0.55,
    },
    "dark": {
        "bg": "#0c0e13",
        "bg_alt": "#12151c",
        "fg": "#e7eaf3",
        "fg_muted": "#8c94a9",
        "accent": "#6d55f5",
        "accent_hover": "#8069ff",
        "accent_fg": "#ffffff",
        "border": "#262c3b",
        "sidebar_bg": "#10131a",
        "sidebar_active": "#7b63ff",
        "entry_bg": "#0e1118",
        "plot_bg": "#0e1118",
        "plot_line": "#9f8fff",
        "plot_grid": "#222839",
        "seek_line": "#ff6b81",
        "danger": "#ff5d73",
        "card_bg": "#171b25",
        "wave_sub": "#35d0dd",
        "shadow": "#000000",
        "shadow_a": 0.9,
    },
}

# (Key, Icon name, Label, Icon tint).  Tints share saturation/lightness so the
# sidebar reads as one family instead of a rainbow.
PAGES = [
    ("edit", "scissors", "Trim & Edit", "#ff6b81"),
    ("convolution", "pulse", "Convolution Effects", "#c084fc"),
    ("space", "layers", "Space Simulator", "#a78bfa"),
    # ("sampling", "wave", "Sampling & Aliasing", "#34d399"),  # commented out
    ("analysis", "wave", "FFT & Analysis", "#34d399"),
    ("decompose", "layers", "Stem Separator", "#fbbf24"),
    ("equalizer", "sliders", "Parametric EQ", "#38bdf8"),
    ("noise", "target", "Spectral Subtraction", "#f472b6"),
    ("convert", "bolt", "Format Converter", "#facc15"),
    # ("youtube", "video", "Internet Song Search", "#f87171"),  # disabled for now (Jamendo signup pending)
    ("voice", "mic", "Voice & Speaker ID", "#818cf8"),
    ("doppler", "doppler", "Doppler Speed Shift", "#22d3ee"),
    ("bpm", "pulse", "BPM Pulse Detector", "#fb923c"),
    ("stress", "volume", "Voice Stress Test", "#f43f5e"),
    ("sonar", "search", "Pulse-Echo Ranging", "#0ea5e9"),
]

PAGE_DESCRIPTIONS = {
    "edit": "Trim, scale, fade or reverse the clip in your workspace.",
    "convolution": "Smooth the signal or add echoes by convolving with an impulse response.",
    "space": "Convolve with a real recorded room/hall impulse response (OpenAIR) to simulate that space.",
    # "sampling": "Downsample deliberately to see/hear aliasing, and compare against a properly anti-aliased downsample.",  # commented out
    "analysis": "Inspect spectrum and pitch, or apply a simple low / high-pass filter.",
    "decompose": "Mute or isolate a frequency band, or split stems with the Demucs neural network.",
    "equalizer": "Shape the tone with a multi-band gain curve (in dB).",
    "noise": "Capture a noise-only passage as a profile, then subtract it from the track.",
    "convert": "Write the current audio to another container format.",
    "youtube": "Search a song name and pull openly-licensed audio from Internet Archive into the workspace.",
    "voice": "Register speaker voiceprints, then identify who is talking in the clip.",
    "doppler": "Estimate how fast a sound source moves from its frequency shift.",
    "bpm": "Estimate the tempo of a song, or the pulse rate of a heartbeat recording.",
    "stress": "Track pitch and amplitude stability frame-by-frame to estimate vocal tension (jitter/shimmer).",
    "sonar": "Emit a pulse, record its echo, and measure distance from the round-trip delay - like sonar/radar.",
}


def F(size=9, weight="normal"):
    return (FONT_FAMILY, size, weight)


# ============================================================ Colour utils ==

def _rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(a, b, t):
    """Blend colour a toward b by t (0..1); returns #rrggbb."""
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * t), round(ga + (gb - ga) * t), round(ba + (bb - ba) * t))


def luminance(h):
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(h)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def ink_on(bg):
    return "#0b1020" if luminance(bg) > 0.30 else "#ffffff"


_FONTS = {}


def get_font(size, weight="normal"):
    key = (size, weight)
    if key not in _FONTS:
        _FONTS[key] = tkfont.Font(family=FONT_FAMILY, size=size, weight=weight)
    return _FONTS[key]


def rr_points(x1, y1, x2, y2, r):
    """Point list for a rounded rectangle (use with create_polygon(smooth=True))."""
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    return [x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r,
            x2, y2 - r, x2, y2 - r, x2, y2, x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1]


# ================================================================ Icons ====
# Tiny vector icon set drawn straight onto Canvas (works on every OS, no emoji
# fonts needed, always crisp and recolourable).  Unit box is [-1, 1].

def _arc(cx, cy, r, a0, a1, n=14):
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        out.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return out


def _crescent():
    R, cx, cy, r = 0.85, 0.42, -0.30, 0.72
    base = math.atan2(cy, cx) + math.pi
    outer, inner = [], []
    for i in range(61):
        a = base - math.pi + 2 * math.pi * i / 60
        x, y = R * math.cos(a), R * math.sin(a)
        if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
            outer.append((x, y))
    for i in range(61):
        a = base - math.pi + 2 * math.pi * i / 60
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        if x * x + y * y < R * R:
            inner.append((x, y))
    if outer and inner:
        d_fwd = math.dist(outer[-1], inner[0])
        d_rev = math.dist(outer[-1], inner[-1])
        if d_rev < d_fwd:
            inner.reverse()
    return outer + inner


def _star(n=5, ro=0.95, ri=0.4):
    pts = []
    for i in range(n * 2):
        a = -math.pi / 2 + i * math.pi / n
        r = ro if i % 2 == 0 else ri
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


_SUN_RAYS = [("L", [(0.62 * math.cos(math.radians(a)), 0.62 * math.sin(math.radians(a))),
                    (0.92 * math.cos(math.radians(a)), 0.92 * math.sin(math.radians(a)))])
             for a in range(0, 360, 45)]

ICONS = {
    "play": [("F", [(-0.5, -0.8), (-0.5, 0.8), (0.85, 0)])],
    "stop": [("F", _rect(-0.62, -0.62, 0.62, 0.62))],
    "rec": [("O", 0, 0, 0.62, True)],
    "arrow_left": [("L", [(0.85, 0), (-0.85, 0)]), ("L", [(-0.2, -0.65), (-0.85, 0), (-0.2, 0.65)])],
    "folder": [("L", [(-0.9, -0.55), (-0.3, -0.55), (-0.05, -0.25), (0.9, -0.25), (0.9, 0.65), (-0.9, 0.65)], True)],
    "plus": [("L", [(0, -0.8), (0, 0.8)]), ("L", [(-0.8, 0), (0.8, 0)])],
    "export": [("L", [(0, -0.85), (0, 0.35)]), ("L", [(-0.5, -0.15), (0, 0.4), (0.5, -0.15)]),
               ("L", [(-0.85, 0.3), (-0.85, 0.85), (0.85, 0.85), (0.85, 0.3)])],
    "undo": [("L", [(0.8, 0.6), (0.8, 0.1), (0.55, -0.25), (0.1, -0.4), (-0.4, -0.4)]),
             ("L", [(-0.1, -0.85), (-0.7, -0.4), (-0.1, 0.05)])],
    "trash": [("L", [(-0.85, -0.5), (0.85, -0.5)]), ("L", [(-0.3, -0.5), (-0.3, -0.85), (0.3, -0.85), (0.3, -0.5)]),
              ("L", [(-0.6, -0.3), (-0.5, 0.9), (0.5, 0.9), (0.6, -0.3)]),
              ("L", [(-0.2, -0.1), (-0.2, 0.6)]), ("L", [(0.2, -0.1), (0.2, 0.6)])],
    "sun": [("O", 0, 0, 0.36, False)] + _SUN_RAYS,
    "moon": [("F", _crescent())],
    "scissors": [("O", -0.55, 0.55, 0.3, False), ("O", 0.55, 0.55, 0.3, False),
                 ("L", [(-0.4, 0.3), (0.75, -0.85)]), ("L", [(0.4, 0.3), (-0.75, -0.85)])],
    "wave": [("L", [(-0.95 + i * 0.095, -0.6 * math.sin(i * 0.095 * math.pi * 1.6)) for i in range(21)])],
    "sparkle": [("F", [(0, -0.95), (0.22, -0.22), (0.95, 0), (0.22, 0.22), (0, 0.95),
                       (-0.22, 0.22), (-0.95, 0), (-0.22, -0.22)])],
    "layers": [("L", [(-0.9, -0.1), (0, -0.6), (0.9, -0.1), (0, 0.4)], True),
               ("L", [(-0.9, 0.3), (0, 0.8), (0.9, 0.3)])],
    "sliders": [("L", [(-0.55, -0.9), (-0.55, 0.9)]), ("L", [(0, -0.9), (0, 0.9)]), ("L", [(0.55, -0.9), (0.55, 0.9)]),
                ("F", _rect(-0.8, 0.15, -0.3, 0.5)), ("F", _rect(-0.25, -0.5, 0.25, -0.15)),
                ("F", _rect(0.3, -0.1, 0.8, 0.25))],
    "target": [("O", 0, 0, 0.62, False), ("O", 0, 0, 0.13, True),
               ("L", [(0, -0.95), (0, -0.5)]), ("L", [(0, 0.5), (0, 0.95)]),
               ("L", [(-0.95, 0), (-0.5, 0)]), ("L", [(0.5, 0), (0.95, 0)])],
    "bolt": [("F", [(0.2, -0.95), (-0.6, 0.15), (-0.05, 0.15), (-0.25, 0.95), (0.6, -0.2), (0.05, -0.2)])],
    "video": [("L", _rect(-0.9, -0.6, 0.9, 0.6), True), ("F", [(-0.2, -0.3), (-0.2, 0.3), (0.35, 0)])],
    "mic": [("L", [(-0.3, -0.7), (-0.15, -0.9), (0.15, -0.9), (0.3, -0.7), (0.3, 0.05), (0.15, 0.3),
                   (-0.15, 0.3), (-0.3, 0.05)], True),
            ("L", _arc(0, 0.05, 0.62, 180, 360)), ("L", [(0, 0.67), (0, 0.95)])],
    "doppler": [("L", [(-0.95, 0), (0.05, 0)]), ("L", [(-0.25, -0.35), (0.15, 0), (-0.25, 0.35)]),
                ("L", _arc(0.15, 0, 0.5, -55, 55)), ("L", _arc(0.15, 0, 0.9, -55, 55))],
    "echo": [("O", -0.6, 0, 0.16, True), ("L", _arc(-0.6, 0, 0.55, -55, 55)), ("L", _arc(-0.6, 0, 1.0, -55, 55))],
    "pulse": [("L", [(-0.95, 0.1), (-0.5, 0.1), (-0.3, -0.7), (0, 0.8), (0.25, -0.3), (0.4, 0.1), (0.95, 0.1)])],
    "reverse": [("L", [(0.9, -0.4), (-0.7, -0.4)]), ("L", [(-0.3, -0.8), (-0.75, -0.4), (-0.3, 0)]),
                ("L", [(-0.9, 0.45), (0.7, 0.45)]), ("L", [(0.3, 0.05), (0.75, 0.45), (0.3, 0.85)])],
    "volume": [("F", [(-0.9, -0.3), (-0.5, -0.3), (-0.05, -0.75), (-0.05, 0.75), (-0.5, 0.3), (-0.9, 0.3)]),
               ("L", _arc(0.05, 0, 0.5, -50, 50)), ("L", _arc(0.05, 0, 0.9, -50, 50))],
    "ramp": [("F", [(-0.9, 0.7), (0.9, 0.7), (0.9, -0.7)])],
    "spectrum": [("F", _rect(-0.85, 0.1, -0.5, 0.85)), ("F", _rect(-0.3, -0.5, 0.05, 0.85)),
                 ("F", _rect(0.2, -0.85, 0.55, 0.85)), ("F", _rect(0.7, -0.1, 0.95, 0.85))],
    "grid": [("F", _rect(-0.9 + 0.65 * i, -0.9 + 0.65 * j, -0.9 + 0.65 * i + 0.5, -0.9 + 0.65 * j + 0.5))
             for i in range(3) for j in range(3)],
    "note": [("O", -0.35, 0.55, 0.3, True), ("L", [(-0.05, 0.55), (-0.05, -0.85)]),
             ("L", [(-0.05, -0.85), (0.6, -0.5), (0.6, -0.15)])],
    "piano": [("L", _rect(-0.9, -0.7, 0.9, 0.7), True), ("L", [(-0.3, 0.0), (-0.3, 0.7)]), ("L", [(0.3, 0.0), (0.3, 0.7)]),
              ("F", _rect(-0.5, -0.7, -0.1, 0.1)), ("F", _rect(0.1, -0.7, 0.5, 0.1))],
    "lowpass": [("L", [(-0.95, -0.5), (0.0, -0.5), (0.55, 0.5), (0.95, 0.5)])],
    "highpass": [("L", [(-0.95, 0.5), (-0.55, 0.5), (0.0, -0.5), (0.95, -0.5)])],
    "ban": [("O", 0, 0, 0.75, False), ("L", [(-0.55, -0.55), (0.55, 0.55)])],
    "filter": [("F", [(-0.9, -0.75), (0.9, -0.75), (0.2, 0.05), (0.2, 0.8), (-0.2, 0.55), (-0.2, 0.05)])],
    "reset": [("L", _arc(0, 0, 0.65, 40, 330)), ("F", [(0.35, -0.85), (0.95, -0.55), (0.3, -0.25)])],
    "search": [("O", -0.2, -0.2, 0.55, False), ("L", [(0.2, 0.2), (0.85, 0.85)])],
    "save": [("L", [(-0.8, -0.8), (0.5, -0.8), (0.8, -0.5), (0.8, 0.8), (-0.8, 0.8)], True),
             ("L", [(-0.45, -0.8), (-0.45, -0.3), (0.3, -0.3), (0.3, -0.8)]),
             ("L", [(-0.4, 0.8), (-0.4, 0.2), (0.4, 0.2), (0.4, 0.8)])],
    "calc": [("L", [(-0.7, -0.35), (0.7, -0.35)]), ("L", [(-0.7, 0.35), (0.7, 0.35)]),
             ("L", [(-0.4, -0.8), (0.4, 0.8)])],
    "star": [("F", _star())],
    "info": [("O", 0, 0, 0.8, False), ("L", [(0, -0.05), (0, 0.5)]), ("O", 0, -0.42, 0.06, True)],
    "logo": [("F", _rect(-0.92, -0.25, -0.62, 0.25)), ("F", _rect(-0.5, -0.6, -0.2, 0.6)),
             ("F", _rect(-0.08, -0.95, 0.22, 0.95)), ("F", _rect(0.34, -0.5, 0.64, 0.5)),
             ("F", _rect(0.76, -0.2, 1.06, 0.2))],
}


def draw_icon(cv, name, cx, cy, size, color, tag="icon", width=None):
    """Draw icon `name` centred at (cx, cy).  Unknown names fall back to text."""
    spec = ICONS.get(name)
    if spec is None:
        return cv.create_text(cx, cy, text=name, fill=color, font=F(max(7, int(size * 0.75))), tags=tag)
    s = size / 2.0
    w = width or max(1.4, size / 9.0)
    for op in spec:
        kind = op[0]
        if kind in ("L", "F"):
            flat = []
            for x, y in op[1]:
                flat += [cx + x * s, cy + y * s]
            if kind == "L":
                if len(op) > 2 and op[2]:
                    flat += flat[:2]
                cv.create_line(*flat, fill=color, width=w, capstyle="round", joinstyle="round", tags=tag)
            else:
                cv.create_polygon(*flat, fill=color, outline=color, width=1, joinstyle="round", tags=tag)
        else:
            _, ox, oy, r, filled = op
            cv.create_oval(cx + (ox - r) * s, cy + (oy - r) * s, cx + (ox + r) * s, cy + (oy + r) * s,
                           outline=color, width=w, fill=color if filled else "", tags=tag)


# ============================================================== Tween mixin ==

class _Tween:
    """Cheap per-widget animation: eases named channels (0..1) toward targets
    with `after` ticks.  Only the widget being animated redraws."""

    def _tw_init(self, *channels):
        self._tv = {c: 0.0 for c in channels}
        self._tt = {c: 0.0 for c in channels}
        self._tw_job = None
        self.bind("<Destroy>", self._tw_destroy, add="+")

    def _tw_destroy(self, event=None):
        if event is not None and event.widget is not self:
            return
        if self._tw_job is not None:
            try:
                self.after_cancel(self._tw_job)
            except tk.TclError:
                pass
            self._tw_job = None

    def _tw_set(self, ch, target, instant=False):
        self._tt[ch] = float(target)
        if instant:
            self._tv[ch] = float(target)
        if self._tw_job is None and self._tv[ch] != self._tt[ch]:
            self._tw_job = self.after(16, self._tw_step)

    def _tw_step(self):
        self._tw_job = None
        moving = False
        for ch, tgt in self._tt.items():
            d = tgt - self._tv[ch]
            if abs(d) < 0.05:
                self._tv[ch] = tgt
            else:
                self._tv[ch] += d * 0.5
                moving = True
        try:
            self._render()
        except tk.TclError:
            return
        if moving:
            self._tw_job = self.after(16, self._tw_step)


# ============================================================ RoundedButton ==

class RoundedButton(tk.Canvas, _Tween):
    """Canvas-drawn rounded button with hover glow, pressed state, focus ring,
    optional vector icon, and an icon-only compact mode.  Supports the ttk-ish
    .config(state=..., text=..., command=...) calls the app relies on."""

    def __init__(self, parent, colors, text="", icon=None, command=None, kind="secondary",
                 bg_key="card_bg", size=9, height=34, padx=14, radius=9, state="normal"):
        tk.Canvas.__init__(self, parent, highlightthickness=0, bd=0, takefocus=1)
        self.colors = colors
        self.text = text
        self.icon = icon
        self.command = command
        self.kind = kind
        self.bg_key = bg_key
        self._size = size
        self._bh = height
        self._padx = padx
        self._radius = radius
        self._state = state
        self._pressed = False
        self._focus = False
        self.compact = False
        self._tw_init("h")
        self._measure()
        self.configure_cursor()

        self.bind("<Configure>", self._on_resize)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<space>", self._on_key)
        self.bind("<Return>", self._on_key)
        self._render()

    # -- geometry --------------------------------------------------------
    def _weight(self):
        return "bold" if self.kind == "primary" else "normal"

    def _measure(self):
        f = get_font(self._size, self._weight())
        tw = f.measure(self.text) if self.text else 0
        iw = 14 if self.icon else 0
        gap = 8 if (self.text and self.icon) else 0
        self.full_width = self._padx * 2 + iw + gap + tw
        self.compact_width = self._bh + 2
        show_text = bool(self.text) and not self.compact
        w = self.full_width if show_text else max(self.compact_width, self._padx * 2 + iw)
        tk.Canvas.configure(self, width=w, height=self._bh + 3)

    def set_compact(self, flag):
        if flag != self.compact:
            self.compact = flag
            self._measure()
            self._render()

    def configure_cursor(self):
        tk.Canvas.configure(self, cursor="arrow" if self._state == "disabled" else "hand2")

    # -- ttk-style option handling ----------------------------------------
    def configure(self, cnf=None, **kw):
        if isinstance(cnf, dict):
            kw = {**cnf, **kw}
        elif cnf is not None:
            return tk.Canvas.configure(self, cnf)
        if not kw:
            return tk.Canvas.configure(self)
        redraw = False
        if "state" in kw:
            self._state = str(kw.pop("state"))
            self._pressed = False
            self.configure_cursor()
            redraw = True
        if "text" in kw:
            self.text = kw.pop("text")
            self._measure()
            redraw = True
        if "icon" in kw:
            self.icon = kw.pop("icon")
            self._measure()
            redraw = True
        if "command" in kw:
            self.command = kw.pop("command")
        if kw:
            tk.Canvas.configure(self, **kw)
        if redraw:
            self._render()

    config = configure

    def cget(self, key):
        if key == "text":
            return self.text
        if key == "state":
            return self._state
        return tk.Canvas.cget(self, key)

    def set_theme(self, colors):
        self.colors = colors
        self._render()

    # -- events ------------------------------------------------------------
    def _enabled(self):
        return self._state != "disabled"

    def _cur_width(self):
        w = self.winfo_width()
        return w if w > 1 else int(tk.Canvas.cget(self, "width"))

    def _on_resize(self, event):
        if event.width != getattr(self, "_drawn_w", None):
            self._render()

    def _on_enter(self, event):
        if self._enabled():
            self._tw_set("h", 1.0)

    def _on_leave(self, event):
        self._pressed = False
        self._tw_set("h", 0.0)
        self._render()

    def _on_press(self, event):
        if self._enabled():
            self._pressed = True
            self._render()

    def _on_release(self, event):
        was = self._pressed
        self._pressed = False
        self._render()
        inside = 0 <= event.x <= self._cur_width() and 0 <= event.y <= self._bh + 3
        if was and inside and self._enabled() and self.command:
            self.command()

    def _on_focus_in(self, event):
        self._focus = True
        self._render()

    def _on_focus_out(self, event):
        self._focus = False
        self._render()

    def _on_key(self, event):
        if self._enabled() and self.command:
            self.command()

    # -- drawing ------------------------------------------------------------
    def _palette(self):
        c = self.colors
        bgc = c[self.bg_key]
        k = self.kind
        if k == "primary":
            return dict(fill=c["accent"], hov=c["accent_hover"], press=mix(c["accent"], "#000000", 0.22),
                        fg=c["accent_fg"], fgh=c["accent_fg"], bd=c["accent"], bdh=c["accent_hover"], glow=c["accent"])
        if k == "danger":
            return dict(fill=mix(bgc, c["danger"], 0.10), hov=c["danger"], press=mix(c["danger"], "#000000", 0.22),
                        fg=c["danger"], fgh="#ffffff", bd=mix(bgc, c["danger"], 0.38), bdh=c["danger"], glow=c["danger"])
        if k == "ghost":
            return dict(fill=bgc, hov=mix(bgc, c["fg"], 0.08), press=mix(bgc, c["fg"], 0.15),
                        fg=c["fg_muted"], fgh=c["fg"], bd=bgc, bdh=mix(bgc, c["fg"], 0.08), glow=None)
        return dict(fill=mix(bgc, c["fg"], 0.045), hov=mix(bgc, c["fg"], 0.10), press=mix(bgc, c["fg"], 0.17),
                    fg=c["fg"], fgh=c["fg"], bd=c["border"], bdh=mix(c["border"], c["accent"], 0.65), glow=None)

    def _render(self):
        try:
            self.delete("all")
            c = self.colors
            bgc = c[self.bg_key]
            tk.Canvas.configure(self, bg=bgc)
        except tk.TclError:
            return
        w = self._cur_width()
        self._drawn_w = w
        h = self._bh
        r = self._radius
        p = self._palette()
        enabled = self._enabled()
        hv = self._tv["h"] if enabled else 0.0
        pressed = self._pressed and enabled

        if not enabled:
            fill = mix(bgc, c["fg"], 0.03)
            bd = mix(bgc, c["fg"], 0.07) if self.kind != "ghost" else bgc
            fg = mix(bgc, c["fg"], 0.32)
            dy = 0
        elif pressed:
            fill, bd, fg, dy = p["press"], p["press"], p["fgh"], 1
        else:
            fill = mix(p["fill"], p["hov"], hv)
            bd = mix(p["bd"], p["bdh"], hv)
            fg = mix(p["fg"], p["fgh"], hv)
            dy = 0

        # elevation: soft underline / glow that grows on hover, vanishes on press
        if enabled and not pressed and hv > 0.02 and self.kind != "ghost":
            if p["glow"]:
                sc = mix(bgc, p["glow"], 0.42 * hv)
            else:
                sc = mix(bgc, c["shadow"], c["shadow_a"] * 0.32 * hv)
            self.create_polygon(rr_points(1, 3, w - 1, h + 2, r), smooth=True, fill=sc, outline="")
        self.create_polygon(rr_points(1, 1 + dy, w - 1, h - 1 + dy, r), smooth=True, fill=fill, outline=bd, width=1)
        if self._focus and enabled:
            self.create_polygon(rr_points(0, 0, w, h, r + 1), smooth=True, fill="", outline=c["accent"], width=2)

        cy = h / 2.0 + dy
        show_text = bool(self.text) and not self.compact
        iw = 14 if self.icon else 0
        if show_text:
            f = get_font(self._size, self._weight())
            tw = f.measure(self.text)
            gap = 8 if self.icon else 0
            x0 = (w - (iw + gap + tw)) / 2.0
            if self.icon:
                draw_icon(self, self.icon, x0 + iw / 2.0, cy, 14, fg)
            self.create_text(x0 + iw + gap, cy, text=self.text, anchor="w", fill=fg, font=f)
        elif self.icon:
            draw_icon(self, self.icon, w / 2.0, cy, 15, fg)


# ===================================================================== Card ==

class Card(tk.Canvas):
    """Rounded panel with a layered soft shadow (light) / hairline glow (dark).
    Put content in `.inner` (a ttk.Frame).  Redraws only when resized."""

    def __init__(self, parent, colors, bg_key="bg", fill_key="card_bg", pad=16, radius=14,
                 elevated=True, fit_height=True, fit_width=False, style="Card.TFrame"):
        tk.Canvas.__init__(self, parent, highlightthickness=0, bd=0)
        self.colors = colors
        self.bg_key = bg_key
        self.fill_key = fill_key
        self.pad = pad
        self.radius = radius
        self.elevated = elevated
        self.fit_height = fit_height
        self.fit_width = fit_width
        self.mx, self.mt, self.mb = (4, 3, 7) if elevated else (1, 1, 1)
        self.inner = ttk.Frame(self, style=style)
        self._win = self.create_window(self.mx + pad, self.mt + pad, window=self.inner, anchor="nw")
        self.configure(bg=colors[bg_key])
        self.bind("<Configure>", self._on_configure)
        self.inner.bind("<Configure>", self._on_inner_configure)

    def content_size(self):
        p = self.pad
        return (self.inner.winfo_reqwidth() + 2 * (p + self.mx), self.inner.winfo_reqheight() + 2 * p + self.mt + self.mb)

    def _on_inner_configure(self, event):
        cw, ch = self.content_size()
        if self.fit_height and int(self.cget("height")) != ch:
            self.configure(height=ch)
        if self.fit_width and int(self.cget("width")) != cw:
            self.configure(width=cw)

    def _on_configure(self, event):
        p = self.pad
        iw = max(10, event.width - 2 * (p + self.mx))
        if self.fit_height:
            self.itemconfigure(self._win, width=iw)
        else:
            ih = max(10, event.height - 2 * p - self.mt - self.mb)
            self.itemconfigure(self._win, width=iw, height=ih)
        self._draw(event.width, event.height)

    def set_theme(self, colors):
        self.colors = colors
        self.configure(bg=colors[self.bg_key])
        self._draw(self.winfo_width(), self.winfo_height())

    def _draw(self, w, h):
        if w < 4 or h < 4:
            return
        c = self.colors
        self.delete("card")
        bgc = c[self.bg_key]
        x1, y1, x2, y2 = self.mx, self.mt, w - self.mx, h - self.mb
        if self.elevated:
            a = c["shadow_a"]
            for k, f in ((4, 0.10), (3, 0.18), (2, 0.30), (1, 0.48)):
                self.create_polygon(rr_points(x1 - k, y1 - k + 2, x2 + k, y2 + k + 2, self.radius + k), smooth=True,
                                    fill=mix(bgc, c["shadow"], a * f * 0.55), outline="", tags="card")
        border = c["border"]
        self.create_polygon(rr_points(x1, y1, x2, y2, self.radius), smooth=True, fill=c[self.fill_key],
                            outline=border, width=1, tags="card")
        self.tag_lower("card")


# ================================================================ FlowFrame ==

class FlowFrame(ttk.Frame):
    """Wraps its children onto new rows when the available width shrinks
    (children are positioned with place(); height is set to fit)."""

    def __init__(self, parent, hgap=14, vgap=12, style="Card.TFrame"):
        super().__init__(parent, style=style)
        self.hgap, self.vgap = hgap, vgap
        self._items = []
        self._last_w = -1
        self._retry = None
        self.bind("<Configure>", self._on_configure)

    def add(self, widget, anchor="s"):
        self._items.append((widget, anchor))
        self._last_w = -1
        self.after_idle(self.relayout)

    def _on_configure(self, event):
        if event.width != self._last_w:
            self.relayout()

    @staticmethod
    def _req(w):
        if hasattr(w, "content_size"):
            return w.content_size()
        return w.winfo_reqwidth(), w.winfo_reqheight()

    def relayout(self):
        avail = self.winfo_width()
        if avail <= 20 or not self._items:
            return
        sizes = [self._req(w) for w, _ in self._items]
        if any(s[0] <= 1 for s in sizes):
            if self._retry is None:
                self._retry = self.after(30, self._retry_layout)
            return
        self._last_w = avail
        rows, cur, x = [], [], 0
        for i, (wdg, anc) in enumerate(self._items):
            wd = sizes[i][0]
            if cur and x + wd > avail:
                rows.append(cur)
                cur, x = [], 0
            cur.append(i)
            x += wd + self.hgap
        rows.append(cur)
        y = 0
        for ri, row in enumerate(rows):
            rh = max(sizes[i][1] for i in row)
            x = 0
            # Stretch cards so every row fills the full width (no dead space at
            # the right edge) and cards in a row share the same height.
            cards = [i for i in row if hasattr(self._items[i][0], "content_size")]
            used = sum(sizes[i][0] for i in row) + self.hgap * (len(row) - 1)
            extra = max(0, avail - used)
            share, rem = (divmod(extra, len(cards)) if cards else (0, 0))
            for i in row:
                wdg, anc = self._items[i]
                wd, ht = sizes[i]
                oy = rh - ht if anc == "s" else (0 if anc == "n" else (rh - ht) // 2)
                if i in cards:
                    k = cards.index(i)
                    wd += share + (1 if k < rem else 0)
                    ht, oy = rh, 0
                    wdg.place(x=x, y=y, width=wd, height=ht)
                elif hasattr(wdg, "content_size"):
                    wdg.place(x=x, y=y + oy, width=wd, height=ht)
                else:
                    wdg.place(x=x, y=y + oy)
                x += wd + self.hgap
            y += rh + (self.vgap if ri < len(rows) - 1 else 0)
        if int(self.cget("height")) != y:
            self.configure(height=y)

    def _retry_layout(self):
        self._retry = None
        self.relayout()


# ================================================================ EQSlider ==

class EQSlider(tk.Canvas, _Tween):
    """Vertical dB slider bound to a tk.DoubleVar (drag, wheel, double-click resets)."""

    def __init__(self, parent, variable, colors, bg_key="card_bg", lo=-12.0, hi=12.0, on_change=None):
        tk.Canvas.__init__(self, parent, width=52, height=110, highlightthickness=0, bd=0, cursor="hand2")
        self.var, self.colors, self.bg_key, self.lo, self.hi = variable, colors, bg_key, lo, hi
        self._drag = False
        self._tw_init("h")
        self.var.trace_add("write", lambda *a: self._render())
        self.bind("<Configure>", lambda e: self._render())
        self.bind("<Enter>", lambda e: self._tw_set("h", 1.0))
        self.bind("<Leave>", lambda e: self._tw_set("h", 0.0))
        self.bind("<ButtonPress-1>", self._set_from_event)
        self.bind("<B1-Motion>", self._set_from_event)
        self.bind("<Double-Button-1>", lambda e: self.var.set(0.0))
        self.bind("<MouseWheel>", lambda e: self._nudge(1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda e: self._nudge(1))
        self.bind("<Button-5>", lambda e: self._nudge(-1))

    def set_theme(self, colors):
        self.colors = colors
        self._render()

    def _track(self):
        h = max(60, self.winfo_height())
        return 22, h - 10

    def _nudge(self, d):
        self.var.set(max(self.lo, min(self.hi, self.var.get() + d)))

    def _set_from_event(self, event):
        y0, y1 = self._track()
        frac = (event.y - y0) / float(y1 - y0)
        v = self.hi - frac * (self.hi - self.lo)
        self.var.set(max(self.lo, min(self.hi, v)))

    def _render(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        c = self.colors
        bgc = c[self.bg_key]
        tk.Canvas.configure(self, bg=bgc)
        w = max(30, self.winfo_width())
        y0, y1 = self._track()
        cx = w / 2.0
        v = float(self.var.get())
        hv = self._tv["h"]

        def v2y(val):
            return y0 + (self.hi - val) / (self.hi - self.lo) * (y1 - y0)

        self.create_line(cx, y0, cx, y1, fill=mix(bgc, c["fg"], 0.14), width=5, capstyle="round")
        zy = v2y(0.0)
        self.create_line(cx - 12, zy, cx + 12, zy, fill=mix(bgc, c["fg"], 0.25), width=1)
        vy = v2y(v)
        if abs(vy - zy) > 0.5:
            self.create_line(cx, zy, cx, vy, fill=c["accent"], width=5, capstyle="round")
        rr = 7 + 1.5 * hv
        if hv > 0.05:
            g = rr + 4 * hv
            self.create_oval(cx - g, vy - g, cx + g, vy + g, fill=mix(bgc, c["accent"], 0.22 * hv), outline="")
        self.create_oval(cx - rr, vy - rr, cx + rr, vy + rr, fill="#ffffff", outline=c["accent"], width=2)
        txt = "0 dB" if round(v) == 0 else f"{round(v):+d} dB"
        self.create_text(cx, 8, text=txt, fill=c["accent"] if round(v) != 0 else c["fg_muted"], font=F(8, "bold"))


# ====================================================== Sidebar nav button ==

class ColorfulNavButton(tk.Canvas, _Tween):
    """Sidebar item: rounded active pill + accent bar, tinted icon badge, eased hover."""

    def __init__(self, parent, icon_symbol, label_text, color, command=None, colors=None, **kwargs):
        tk.Canvas.__init__(self, parent, height=42, highlightthickness=0, bd=0, cursor="hand2", takefocus=1, **kwargs)
        self.command = command
        self.vivid_color = color
        self.icon_symbol = icon_symbol
        self.label_text = label_text
        self.is_active = False
        self.compact = False
        self.colors = colors or THEMES["dark"]
        self._tw_init("h", "a")
        self.bind("<ButtonRelease-1>", self._on_click)
        self.bind("<Enter>", lambda e: self._tw_set("h", 1.0))
        self.bind("<Leave>", lambda e: self._tw_set("h", 0.0))
        self.bind("<Configure>", lambda e: self._render())
        self.bind("<Return>", self._on_click)
        self.bind("<space>", self._on_click)
        self.bind("<FocusIn>", lambda e: self._render())
        self.bind("<FocusOut>", lambda e: self._render())

    def set_theme(self, colors, active_key, my_key):
        self.colors = colors
        self.is_active = (active_key == my_key)
        self._tw_set("a", 1.0 if self.is_active else 0.0)
        self._render()

    def set_compact(self, flag):
        if flag != self.compact:
            self.compact = flag
            self._render()

    def redraw(self, hover=False):
        self._render()

    def _on_click(self, event=None):
        if self.command:
            self.command()

    def _render(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        c = self.colors
        sb = c["sidebar_bg"]
        tk.Canvas.configure(self, bg=sb)
        w = self.winfo_width()
        if w <= 1:
            w = 240
        h = 42
        hv, av = self._tv["h"], self._tv["a"]
        pill = mix(mix(sb, c["fg"], 0.06 * hv), mix(sb, c["sidebar_active"], 0.20), av)
        pill_bd = mix(pill, c["sidebar_active"], 0.45 * av)
        self.create_polygon(rr_points(6, 3, w - 6, h - 3, 10), smooth=True, fill=pill, outline=pill_bd, width=1)
        if av > 0.05:
            bh = 20 * av
            self.create_polygon(rr_points(6, h / 2 - bh / 2, 9.5, h / 2 + bh / 2, 1.5), smooth=True,
                                fill=c["sidebar_active"], outline="")

        bx = w / 2.0 if self.compact else 27
        tint = mix(sb, self.vivid_color, 0.25)
        badge = mix(mix(mix(sb, c["fg"], 0.07), tint, hv), self.vivid_color, av)
        icon_col = mix(mix(c["fg_muted"], self.vivid_color, hv), ink_on(self.vivid_color), av)
        self.create_polygon(rr_points(bx - 13, h / 2 - 13, bx + 13, h / 2 + 13, 8), smooth=True, fill=badge, outline="")
        draw_icon(self, self.icon_symbol, bx, h / 2, 15, icon_col)
        if not self.compact:
            fnt = get_font(9, "bold" if av > 0.5 else "normal")
            self.create_text(48, h / 2, text=self.label_text, anchor="w", font=fnt,
                             fill=mix(mix(c["fg_muted"], c["fg"], 0.55 * hv), c["fg"], av))
        if self.focus_get() is self:
            self.create_polygon(rr_points(5, 2, w - 5, h - 2, 11), smooth=True, fill="", outline=c["accent"], width=2)


# ================================================================== Header ==

class AnimatedHeaderCanvas(tk.Canvas):
    """Brand bar: logo, title, a gently flowing dual-wave ribbon that reacts to
    the cursor, and a level meter that only animates while audio plays.
    All animation uses coords() on pre-built items (no per-frame rebuilds)."""
    HEIGHT = 58

    def __init__(self, parent, colors, is_active=None, **kwargs):
        super().__init__(parent, height=self.HEIGHT, bg=colors["bg_alt"], highlightthickness=0, **kwargs)
        self.colors = colors
        self.is_active = is_active or (lambda: False)
        self.phase = 0.0
        self.mouse_x = -1000
        self.ripple = 0.0
        self.target = 0.0
        self._bars = []
        self._state_txt = None
        self._wave_x0, self._wave_x1 = 360, 600
        self._last_playing = None
        self.bind("<Configure>", self._build)
        self.bind("<Motion>", self._on_move)
        self.bind("<Leave>", lambda e: setattr(self, "target", 0.0))
        self._tick()

    def set_colors(self, colors):
        self.colors = colors
        self.configure(bg=colors["bg_alt"])
        self._build()

    def _on_move(self, event):
        self.mouse_x = event.x
        self.target = 1.0

    def _build(self, event=None):
        c = self.colors
        self.delete("all")
        w = max(self.winfo_width(), 200)
        h = self.HEIGHT
        bg = c["bg_alt"]
        self._wave_x0 = 372
        self._wave_x1 = max(self._wave_x0 + 40, w - 150)
        self._w2 = self.create_line(0, 0, 1, 1, fill=mix(bg, c["wave_sub"], 0.45), width=1.5)
        self._w1 = self.create_line(0, 0, 1, 1, fill=mix(bg, c["accent"], 0.75), width=2.2)
        # brand
        self.create_polygon(rr_points(20, h / 2 - 16, 52, h / 2 + 16, 9), smooth=True, fill=c["accent"], outline="")
        draw_icon(self, "logo", 36, h / 2, 17, c["accent_fg"], width=1)
        self.create_text(64, h / 2 - 1, text=APP_NAME, anchor="w", fill=c["fg"], font=F(16, "bold"))
        self.create_text(178, h / 2 + 2, text="Audio DSP workbench", anchor="w", fill=c["fg_muted"], font=F(9))
        self.create_line(0, h - 1, w, h - 1, fill=c["border"])
        # level meter (right)
        self._bars = []
        for i in range(5):
            x = w - 44 + i * 7
            self._bars.append(self.create_rectangle(x, h / 2 - 2, x + 4, h / 2 + 2, fill=c["border"], outline=""))
        self._state_txt = self.create_text(w - 56, h / 2, text="Idle", anchor="e", fill=c["fg_muted"], font=F(9))
        self._last_playing = None
        self._update(force=True)

    def _tick(self):
        try:
            if self.winfo_ismapped():
                self.phase += 0.07
                self.ripple += (self.target - self.ripple) * 0.12
                self._update()
        except tk.TclError:
            return
        self.after(34, self._tick)

    def _update(self, force=False):
        if not hasattr(self, "_w1"):
            return
        h = self.HEIGHT
        x0, x1 = self._wave_x0, self._wave_x1
        pts1, pts2 = [], []
        for x in range(x0, x1 + 1, 7):
            d = abs(x - self.mouse_x)
            rip = 0.0
            if d < 140 and self.ripple > 0.01:
                rip = math.exp(-((d / 38.0) ** 2)) * self.ripple * math.sin(d * 0.18 - self.phase * 3.5) * 14
            edge = min(1.0, (x - x0) / 40.0, (x1 - x) / 40.0)
            a = edge * (9 * math.cos(self.phase * 0.5) + 2)
            pts1 += [x, h / 2 + math.sin(x * 0.02 + self.phase) * a + rip]
            pts2 += [x, h / 2 + math.cos(x * 0.017 - self.phase) * 6 * edge - rip * 0.6]
        if len(pts1) >= 4:
            self.coords(self._w1, *pts1)
            self.coords(self._w2, *pts2)
        playing = bool(self.is_active())
        c = self.colors
        for i, b in enumerate(self._bars):
            bx = self.coords(b)[0]
            if playing:
                hh = 3 + (h / 2 - 12) * (0.5 + 0.5 * math.sin(self.phase * 3.2 + i * 1.3))
                col = c["accent"] if i % 2 == 0 else c["wave_sub"]
            else:
                hh = 2
                col = c["border"]
            self.coords(b, bx, h / 2 - hh, bx + 4, h / 2 + hh)
            if playing != self._last_playing or force:
                self.itemconfigure(b, fill=col)
            elif playing:
                self.itemconfigure(b, fill=col)
        if playing != self._last_playing or force:
            self.itemconfigure(self._state_txt, text="Playing" if playing else "Idle",
                               fill=c["fg"] if playing else c["fg_muted"])
            self._last_playing = playing


# ================================================================== Splash ==

class InteractiveWelcomeCanvas(tk.Canvas):
    """Splash screen.  Static layers (gradient, grid, card) are drawn once per
    resize; the waves / particles / cursor halo are moved with coords() each
    frame, so the animation stays smooth and cheap."""

    BG_TOP, BG_BOT = "#0b1020", "#05070d"
    CARD, CARD_BD = "#0f1526", "#2a3160"
    WAVES = [("#06b6d4", 3.0, 35, 0.012, 1.0), ("#8b5cf6", 2.0, 25, 0.018, -0.8), ("#f43f5e", 1.5, 45, 0.008, 1.4)]

    def __init__(self, parent, colors, on_enter, on_exit, **kwargs):
        super().__init__(parent, bg="#070a12", highlightthickness=0, **kwargs)
        self.colors = colors
        self.on_enter = on_enter
        self.on_exit = on_exit

        self.phase = 0.0
        self.animating = True
        self.mouse_x = -1000
        self.mouse_y = -1000
        self._cw = self._ch = 0
        self._rebuild_job = None
        self._wave_items = []
        self._particle_items = []
        self._halo = []

        self.particles = []
        for _ in range(45):
            self.particles.append({
                "x": random.randint(0, 1400), "y": random.randint(0, 900),
                "r": random.uniform(1.5, 3.2), "speed": random.uniform(0.3, 1.2),
                "color": random.choice(["#06b6d4", "#8b5cf6", "#f43f5e", "#34d399", "#fbbf24"]),
            })

        # splash-local palette so the buttons blend into the card
        self.btn_colors = dict(THEMES["dark"])
        self.btn_colors["splash_card"] = self.CARD

        self.bind("<Configure>", self._on_configure)
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Leave>", self._on_mouse_leave)

        self._build_widgets()
        self._animate()

    def _on_mouse_move(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def _on_mouse_leave(self, event):
        self.mouse_x = self.mouse_y = -1000

    def _build_widgets(self):
        self.btn_enter = RoundedButton(self, self.btn_colors, text="Enter Workbench", icon="bolt", kind="primary",
                                       bg_key="splash_card", size=10, height=42, padx=24, radius=12,
                                       command=self.on_enter)
        self.btn_exit = RoundedButton(self, self.btn_colors, text="Exit", icon="arrow_left", kind="secondary",
                                      bg_key="splash_card", size=10, height=42, padx=22, radius=12,
                                      command=self.on_exit)

    def _on_configure(self, event):
        self._cw, self._ch = event.width, event.height
        if self._rebuild_job is None:
            self._rebuild_job = self.after(30, self._rebuild)

    def _rebuild(self):
        self._rebuild_job = None
        w, h = self._cw, self._ch
        if w < 50 or h < 50:
            return
        self.delete("all")
        # vertical gradient
        bands = 28
        for i in range(bands):
            self.create_rectangle(0, h * i / bands, w, h * (i + 1) / bands + 1,
                                  fill=mix(self.BG_TOP, self.BG_BOT, i / (bands - 1)), outline="")
        # soft spotlight behind the card
        for i, s in enumerate((0.95, 0.75, 0.55, 0.38)):
            rx, ry = w * 0.5 * s, h * 0.55 * s
            self.create_oval(w / 2 - rx, h / 2 - ry, w / 2 + rx, h / 2 + ry,
                             fill=mix(self.BG_TOP, "#1a2350", 0.10 + 0.08 * i), outline="")
        # faint dotted grid
        for x in range(0, w, 48):
            self.create_line(x, 0, x, h, fill="#141b33", dash=(1, 6))
        for y in range(0, h, 48):
            self.create_line(0, y, w, y, fill="#141b33", dash=(1, 6))

        # waves (glow layer + crisp layer)
        self._wave_items = []
        for color, width, amp, freq, sm in self.WAVES:
            glow = self.create_line(0, 0, 1, 1, fill=mix(self.BG_TOP, color, 0.22), width=width * 4, capstyle="round")
            main = self.create_line(0, 0, 1, 1, fill=color, width=width, capstyle="round")
            self._wave_items.append((glow, main, amp, freq, sm))
        # particles
        self._particle_items = []
        for p in self.particles:
            halo = self.create_oval(0, 0, 1, 1, fill=mix(self.BG_TOP, p["color"], 0.18), outline="")
            core = self.create_oval(0, 0, 1, 1, fill=p["color"], outline="")
            self._particle_items.append((halo, core))
        self._halo = [self.create_oval(0, 0, 1, 1, outline="#06b6d4", width=1.5, state="hidden"),
                      self.create_oval(0, 0, 1, 1, outline="#f43f5e", width=1.5, state="hidden")]

        # centre card
        cx, cy = w / 2, h / 2
        cw, ch = 540, 340
        x1, y1, x2, y2 = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
        for k, f in ((10, 0.05), (7, 0.08), (4, 0.14), (2, 0.22)):
            self.create_polygon(rr_points(x1 - k, y1 - k + 6, x2 + k, y2 + k + 6, 22 + k), smooth=True,
                                fill=mix(self.BG_TOP, "#000000", 0.35 + f * 2), outline="")
        self.create_polygon(rr_points(x1, y1, x2, y2, 22), smooth=True, fill=self.CARD, outline=self.CARD_BD, width=1)
        self.create_polygon(rr_points(x1 + 28, y1 + 1, x2 - 28, y1 + 3, 1), smooth=True, fill="#6d55f5", outline="")
        # logo mark
        self.create_polygon(rr_points(cx - 32, y1 + 34, cx + 32, y1 + 98, 18), smooth=True, fill="#6d55f5", outline="")
        draw_icon(self, "logo", cx, y1 + 66, 36, "#ffffff", width=1)
        self.create_text(cx, y1 + 132, text=APP_NAME, fill="#eef1ff", font=F(30, "bold"))
        self.create_text(cx, y1 + 168, text="Interactive Digital Signal Processing & AI Workbench",
                         fill="#9aa3c7", font=F(10))
        self.create_text(cx, y1 + 190, text="Edit, filter, analyze and separate audio in one place.",
                         fill="#5f6890", font=F(9))
        self.create_window(cx - 92, y2 - 58, window=self.btn_exit, anchor="center")
        self.create_window(cx + 86, y2 - 58, window=self.btn_enter, anchor="center")
        self.create_text(cx, h - 22, text="Move your cursor across the screen - the signal follows it.",
                         fill="#3b4468", font=F(9))
        self._update_frame()

    def _update_frame(self):
        w, h = self._cw, self._ch
        if not self._wave_items:
            return
        mx, my = self.mouse_x, self.mouse_y
        for glow, main, amp, freq, sm in self._wave_items:
            pts = []
            for x in range(0, w + 8, 6):
                dx = abs(x - mx)
                force = math.exp(-((dx / 110.0) ** 2)) if dx < 250 else 0.0
                off = (my - h / 2) * 0.25 * force
                y = h / 2 + math.sin(x * freq + self.phase * sm) * (amp + force * 40.0) + off
                pts += [x, y]
            self.coords(glow, *pts)
            self.coords(main, *pts)
        for p, (halo, core) in zip(self.particles, self._particle_items):
            p["y"] -= p["speed"]
            if p["y"] < 0:
                p["y"] = h + 10
                p["x"] = random.randint(0, max(w, 10))
            r = p["r"] * (2.6 if math.hypot(p["x"] - mx, p["y"] - my) < 120 else 1.0)
            self.coords(core, p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r)
            self.coords(halo, p["x"] - r * 2.2, p["y"] - r * 2.2, p["x"] + r * 2.2, p["y"] + r * 2.2)
        if 0 <= mx <= w and 0 <= my <= h:
            r1 = 35 + math.sin(self.phase * 4) * 5
            r2 = 18 + math.cos(self.phase * 4) * 3
            self.coords(self._halo[0], mx - r1, my - r1, mx + r1, my + r1)
            self.coords(self._halo[1], mx - r2, my - r2, mx + r2, my + r2)
            for it in self._halo:
                self.itemconfigure(it, state="normal")
        else:
            for it in self._halo:
                self.itemconfigure(it, state="hidden")

    def _animate(self):
        if not self.animating:
            return
        try:
            if self.winfo_ismapped():
                self.phase += 0.05
                self._update_frame()
            self.after(30, self._animate)
        except tk.TclError:
            pass

    def stop_animation(self):
        self.animating = False

    def start_animation(self):
        if not self.animating:
            self.animating = True
            self._animate()


# ============================================================ Application ====



TIP_DEFAULT = "Hover over controls to view parameter descriptions."


class AudioEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} - Interactive DSP Suite")
        self.root.geometry("1320x860")
        self.root.minsize(900, 680)

        self.sr = None
        self.data = None
        self.history = []
        self.filepath = None
        self.seek_time = 0.0
        self._seek_line = None
        self._is_playing = False
        self._playback_start_walltime = None
        self._playback_start_seek = None
        self._last_detected_freq = None
        self._yt_download_thread = None
        self._archive_download_thread = None
        self._jamendo_download_thread = None
        self._ir_download_thread = None
        self.ir_sr = None
        self.ir_data = None
        self.ir_name = None
        self._last_stress_result = None
        self.chirp_sr = None
        self.chirp_data = None
        self.sonar_recorder = None
        self.sonar_recorded_sr = None
        self.sonar_recorded_data = None
        self._last_echo_result = None
        self._voiceprints_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voiceprints.json")
        self.enrolled_speakers = ao.load_voiceprints(self._voiceprints_path)
        self.now_playing_var = tk.StringVar(value="")

        self.theme_name = "dark"
        self._theme_colors = THEMES[self.theme_name]
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.nav_buttons = {}
        self.pages = {}
        self._active_page = "edit"
        self._themed = []          # widgets that repaint themselves via set_theme(colors)
        self._combos = []
        self._tb_buttons = []
        self._sidebar_compact = False
        self._toolbar_compact = False

        # Dropdown lists (Combobox popdowns) use the Tk option database
        self.root.option_add("*TCombobox*Listbox.font", (FONT_FAMILY, 10))
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.root.option_add("*TCombobox*Listbox.highlightThickness", 0)

        # Initialize Main Workbench UI container Frame
        self.workbench_frame = ttk.Frame(self.root)

        # Initialize Welcome Interactive Page
        self.welcome_canvas = InteractiveWelcomeCanvas(
            self.root, self._theme_colors,
            on_enter=self.enter_workbench,
            on_exit=self.exit_app
        )
        self.welcome_canvas.pack(fill=tk.BOTH, expand=True)

        self._build_ui()
        self._apply_theme(self.theme_name)
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        self.root.after(50, self._playhead_heartbeat)

    # ------------------------------------------------------------ screens ---
    def enter_workbench(self):
        """Switches display from Welcome Canvas to Main Workbench UI."""
        if self.welcome_canvas:
            self.welcome_canvas.stop_animation()
            self.welcome_canvas.pack_forget()
        self.workbench_frame.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()

        # Re-trigger page selection & theme update after mapping elements to screen
        self._select_page("edit")
        self._apply_theme(self.theme_name)
        for btn in self.nav_buttons.values():
            btn.redraw()
        self._on_root_configure(None, force=True)

    def show_welcome_page(self):
        """Hides Main Workbench UI and brings back Welcome Canvas."""
        self.stop()
        self.workbench_frame.pack_forget()
        self.welcome_canvas.pack(fill=tk.BOTH, expand=True)
        self.welcome_canvas.start_animation()

    def exit_app(self):
        """Closes the entire application."""
        self.stop()
        self.root.destroy()

    # ------------------------------------------------------ small helpers ---
    def _reg(self, widget):
        self._themed.append(widget)
        return widget

    def _btn(self, parent, text, icon, command, tip=None, kind="secondary", bg_key="card_bg", **kw):
        b = RoundedButton(parent, self._theme_colors, text=text, icon=icon, command=command, kind=kind,
                          bg_key=bg_key, **kw)
        self._reg(b)
        if tip:
            self._tip(b, tip)
        return b

    def _card(self, parent, **kw):
        return self._reg(Card(parent, self._theme_colors, **kw))

    def _tile(self, parent, title):
        """Outlined rounded group used inside module pages."""
        t = self._card(parent, bg_key="card_bg", pad=12, radius=10, elevated=False, fit_width=True)
        ttk.Label(t.inner, text=title, style="TileTitle.TLabel").pack(anchor="w", pady=(0, 8))
        return t, t.inner

    def _field(self, parent, caption, width=8, default=None, tip=None):
        f = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(f, text=caption, style="Caption.TLabel").pack(anchor="w")
        e = ttk.Entry(f, width=width, font=F(10))
        if default is not None:
            e.insert(0, default)
        e.pack(anchor="w", pady=(4, 3))
        if tip:
            self._tip(e, tip)
        return f, e

    def _combo(self, parent, caption, var=None, values=(), width=12, tip=None):
        f = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(f, text=caption, style="Caption.TLabel").pack(anchor="w")
        cb = ttk.Combobox(f, textvariable=var, values=list(values), width=width, state="readonly", font=F(10))
        cb.pack(anchor="w", pady=(4, 3))
        self._combos.append(cb)
        if tip:
            self._tip(cb, tip)
        return f, cb

    def _status_label(self, parent, var, style="Status.TLabel"):
        lb = ttk.Label(parent, textvariable=var, style=style, anchor="w", justify="left")
        lb.bind("<Configure>", lambda e, l=lb: l.configure(wraplength=max(120, e.width - 4)))
        return lb

    def _tip(self, widget, text):
        widget.bind("<Enter>", lambda e: self.tooltip_var.set(text), add="+")
        widget.bind("<Leave>", lambda e: self.tooltip_var.set(TIP_DEFAULT), add="+")

    def _center_on_root(self, win, w, h):
        self.root.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - h) // 3
        win.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

    def _grab(self, win):
        win.update_idletasks()
        try:
            win.grab_set()
        except tk.TclError:
            def retry():
                try:
                    win.grab_set()
                except tk.TclError:
                    pass
            win.after(80, retry)

    # ---------------------------------------------------------- dialogs -----
    def load_audio_options(self):
        self._show_source_dialog(
            title="Load Audio Source",
            file_cmd=self.load_file,
            rec_cmd=lambda: self.open_recorder_window(mode="load")
        )

    def join_audio_options(self):
        if not self._require_audio():
            return
        self._show_source_dialog(
            title="Join Audio Source",
            file_cmd=self.join_file,
            rec_cmd=lambda: self.open_recorder_window(mode="join")
        )

    def _show_source_dialog(self, title, file_cmd, rec_cmd):
        c = self._theme_colors
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=c["bg"])
        win.resizable(False, False)
        win.transient(self.root)
        self._center_on_root(win, 460, 220)
        self._grab(win)

        card = Card(win, c, bg_key="bg", pad=20, radius=14, fit_height=False)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        inner = card.inner
        ttk.Label(inner, text="Select Source Input Stream", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(inner, text="Choose where the audio should come from.", style="Status.TLabel").pack(anchor="w", pady=(4, 20))
        row = ttk.Frame(inner, style="Card.TFrame")
        row.pack(anchor="w")
        RoundedButton(row, c, text="Open Audio File", icon="folder", kind="primary", height=38,
                      command=lambda: [win.destroy(), file_cmd()]).pack(side=tk.LEFT, padx=(0, 10))
        RoundedButton(row, c, text="Capture Live Mic", icon="mic", kind="secondary", height=38,
                      command=lambda: [win.destroy(), rec_cmd()]).pack(side=tk.LEFT)

    def open_recorder_window(self, mode="load"):
        if not PLAYBACK_AVAILABLE:
            messagebox.showerror("Error", "PortAudio/sounddevice driver missing.")
            return

        c = self._theme_colors
        win = tk.Toplevel(self.root)
        win.title("Microphone Recorder Stream")
        win.configure(bg=c["bg"])
        win.resizable(False, False)
        win.transient(self.root)
        self._center_on_root(win, 480, 270)
        self._grab(win)

        target_sr = self.sr if (mode == "join" and self.sr) else 44100
        recorder = ao.AudioRecorder(sample_rate=target_sr)
        recorded_audio = [None, None]

        card = Card(win, c, bg_key="bg", pad=20, radius=14, fit_height=False)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        inner = card.inner
        ttk.Label(inner, text="Microphone Recorder", style="CardTitle.TLabel").pack(anchor="w")
        status_lbl = ttk.Label(inner, text="Ready to record audio input stream.", style="Status.TLabel")
        status_lbl.pack(anchor="w", pady=(6, 18))

        def start_recording():
            try:
                recorder.start()
                status_lbl.config(text="Recording Audio Stream...", foreground=self._theme_colors["danger"])
                btn_start.config(state=tk.DISABLED)
                btn_stop.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Microphone Error", str(e))

        def stop_recording():
            sr, data = recorder.stop()
            recorded_audio[0] = sr
            recorded_audio[1] = data
            duration = len(data) / sr if len(data) > 0 else 0
            status_lbl.config(text=f"Captured {duration:.2f} seconds.", foreground=self._theme_colors["fg"])
            btn_start.config(state=tk.NORMAL)
            btn_stop.config(state=tk.DISABLED)
            btn_save_load.config(state=tk.NORMAL)

        def process_recording():
            rec_sr, rec_data = recorded_audio[0], recorded_audio[1]
            if rec_data is None or len(rec_data) == 0:
                messagebox.showwarning("Warning", "No audio recorded.")
                return

            path = filedialog.asksaveasfilename(
                title="Save Audio Recording",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")]
            )
            if path:
                ao.save_wav(path, rec_sr, rec_data)

            if mode == "load":
                self.sr = rec_sr
                self.data = rec_data
                self.filepath = path if path else "Recorded Stream"
                self.history.clear()
            elif mode == "join":
                self._snapshot()
                self.data = ao.join(self.data, rec_data)

            self._refresh_plot()
            win.destroy()

        btn_frame = ttk.Frame(inner, style="Card.TFrame")
        btn_frame.pack(anchor="w")

        btn_start = RoundedButton(btn_frame, c, text="Start Record", icon="rec", kind="danger", height=38,
                                  command=start_recording)
        btn_start.pack(side=tk.LEFT, padx=(0, 10))

        btn_stop = RoundedButton(btn_frame, c, text="Stop", icon="stop", kind="secondary", height=38,
                                 command=stop_recording, state=tk.DISABLED)
        btn_stop.pack(side=tk.LEFT)

        btn_save_load = RoundedButton(inner, c, text="Save & Process Audio Stream", icon="save", kind="primary",
                                      height=38, command=process_recording, state=tk.DISABLED)
        btn_save_load.pack(anchor="w", pady=(14, 0))

    # ------------------------------------------------------------- layout ---
    def _build_ui(self):
        c = self._theme_colors
        wf = self.workbench_frame

        self.header_canvas = AnimatedHeaderCanvas(wf, c, is_active=lambda: self._is_playing)
        self.header_canvas.pack(side=tk.TOP, fill=tk.X)

        # Tooltip / status bar - packed BEFORE the body so it can never be squeezed off-screen
        bar = ttk.Frame(wf, style="Status.TFrame")
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(bar, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X)
        tip_row = ttk.Frame(bar, style="Status.TFrame")
        tip_row.pack(fill=tk.X, padx=16, pady=7)
        ttk.Label(tip_row, text="TIP", style="TipTag.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        self.tooltip_var = tk.StringVar(value=TIP_DEFAULT)
        ttk.Label(tip_row, textvariable=self.tooltip_var, style="Tooltip.TLabel", anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(tip_row, text=("Audio output ready" if PLAYBACK_AVAILABLE else "Audio output unavailable (PortAudio missing)"),
                  style="Tooltip.TLabel").pack(side=tk.RIGHT)

        body = ttk.Frame(wf)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---- sidebar -------------------------------------------------------
        self.sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=244)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        ttk.Separator(body, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        self.sidebar_title = ttk.Label(self.sidebar, text="Modules", style="SidebarTitle.TLabel")
        self.sidebar_title.pack(anchor="w", padx=(22, 0), pady=(18, 8))

        self.nav_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.nav_frame.pack(fill=tk.X)
        self.nav_buttons = {}
        for key, icon, label, color in PAGES:
            btn = ColorfulNavButton(
                self.nav_frame, icon_symbol=icon, label_text=label, color=color, colors=c,
                command=lambda k=key: self._select_page(k)
            )
            btn.pack(fill=tk.X, pady=1)
            self._tip(btn, PAGE_DESCRIPTIONS[key])
            self.nav_buttons[key] = btn

        side_foot = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        side_foot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=12)
        self.theme_toggle_btn = self._btn(side_foot, "Light mode", "sun", self.toggle_theme,
                                          tip="Switch between the dark and light interface theme.",
                                          kind="ghost", bg_key="sidebar_bg")
        self.theme_toggle_btn.pack(fill=tk.X)

        # ---- main column ---------------------------------------------------
        main_content = ttk.Frame(body)
        main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._main_content = main_content
        self._build_toolbar(main_content)
        self._build_module_card(main_content)      # packed to the bottom first ...
        self._build_waveform_card(main_content)    # ... so the waveform takes all remaining space
        self._build_pages()

    def _build_toolbar(self, parent):
        card = self._card(parent, pad=10, radius=12)
        self.toolbar_card = card
        card.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 4))
        row = ttk.Frame(card.inner, style="Card.TFrame")
        row.pack(fill=tk.X)
        self._tb_buttons = []

        def add(text, icon, cmd, tip, kind="secondary", side=tk.LEFT, padx=(0, 6)):
            b = self._btn(row, text, icon, cmd, tip, kind=kind)
            b.pack(side=side, padx=padx)
            self._tb_buttons.append(b)
            return b

        def sep():
            ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(2, 8), pady=4)

        add("Back", "arrow_left", self.show_welcome_page, "Return back to the welcome screen.", kind="ghost")
        sep()
        add("Open File", "folder", self.load_audio_options, "Load audio file or open live microphone stream.")
        add("Join Audio", "plus", self.join_audio_options, "Append external audio track onto end of stream.")
        add("Export", "export", self.save_file,
            "Export current audio buffer into audio formats (WAV, MP3, FLAC, OGG, M4A).")
        sep()
        add("Undo", "undo", self.undo, "Revert back to the snapshot state prior to the last applied modification.")
        add("Clear Frame", "trash", self.delete_audio, "Flush active audio buffers and reset workspace.", kind="danger")
        add("Stop", "stop", self.stop, "Halt current playback output.", side=tk.RIGHT, padx=(6, 0))
        add("Play Stream", "play", self.play, "Initiate audio playback starting from playhead position.",
            kind="primary", side=tk.RIGHT, padx=(6, 0))
        row.bind("<Configure>", self._on_toolbar_configure)

    def _on_toolbar_configure(self, event):
        need = sum(b.full_width for b in self._tb_buttons) + 6 * len(self._tb_buttons) + 2 * 14 + 34
        compact = self._toolbar_compact
        if not compact and event.width < need:
            compact = True
        elif compact and event.width > need + 24:
            compact = False
        if compact != self._toolbar_compact:
            self._toolbar_compact = compact
            for b in self._tb_buttons:
                b.set_compact(compact)

    def _build_waveform_card(self, parent):
        card = self._card(parent, pad=14, radius=14, fit_height=False)
        card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=4)
        inner = card.inner

        head = ttk.Frame(inner, style="Card.TFrame")
        head.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(head, text="Waveform", style="CardTitle.TLabel").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Status: Idle (No Audio Loaded)")
        ttk.Label(head, textvariable=self.status_var, style="Mono.TLabel").pack(side=tk.RIGHT)

        now_playing_row = ttk.Frame(inner, style="Card.TFrame")
        now_playing_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(now_playing_row, textvariable=self.now_playing_var, style="Result.TLabel").pack(side=tk.LEFT, pady=(2, 0))

        self.seek_status_var = tk.StringVar(value="Click interactive waveform canvas to place playhead position.")
        foot = ttk.Frame(inner, style="Card.TFrame")
        foot.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(foot, textvariable=self.seek_status_var, style="Muted.TLabel").pack(side=tk.LEFT, pady=(6, 0))

        fig = Figure(figsize=(9, 2.7), dpi=100)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=inner)
        widget = self.canvas.get_tk_widget()
        widget.configure(highlightthickness=0, bd=0)
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        self.canvas.mpl_connect("button_press_event", self._on_waveform_click)
        self.canvas.mpl_connect("resize_event", self._on_fig_resize)
        self._tip(widget, "Click anywhere on the waveform to place the playhead and start playback from that point.")
        self._draw_empty_state(self._theme_colors)

    def _build_module_card(self, parent):
        card = self._card(parent, pad=16, radius=14)
        card.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 12))
        inner = card.inner

        hdr = ttk.Frame(inner, style="Card.TFrame")
        hdr.pack(fill=tk.X)
        self.page_badge = tk.Canvas(hdr, width=38, height=38, highlightthickness=0, bd=0)
        self.page_badge.pack(side=tk.LEFT, padx=(0, 12))
        txt = ttk.Frame(hdr, style="Card.TFrame")
        txt.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.page_title_var = tk.StringVar()
        self.page_desc_var = tk.StringVar()
        ttk.Label(txt, textvariable=self.page_title_var, style="CardTitle.TLabel").pack(anchor="w")
        self._status_label(txt, self.page_desc_var, "Caption2.TLabel").pack(anchor="w", fill=tk.X)
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 12))

        # The page area lives in a small scroll host: when the window is short the
        # waveform keeps a guaranteed minimum height and the controls scroll
        # instead of being clipped or crushing the plot.
        self.module_card = card
        host = ttk.Frame(inner, style="Card.TFrame")
        host.pack(fill=tk.BOTH, expand=True)
        self.mod_canvas = tk.Canvas(host, highlightthickness=0, bd=0, height=10, bg=self._theme_colors["card_bg"])
        self.mod_vsb = ttk.Scrollbar(host, orient=tk.VERTICAL, command=self.mod_canvas.yview)
        self.mod_canvas.configure(yscrollcommand=self.mod_vsb.set)
        self.mod_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pages_outer = ttk.Frame(self.mod_canvas, style="Card.TFrame")
        self._mod_win = self.mod_canvas.create_window(0, 0, window=self.pages_outer, anchor="nw")
        self.pages_outer.grid_rowconfigure(0, weight=1)
        self.pages_outer.grid_columnconfigure(0, weight=1)
        self.pages_outer.bind("<Configure>", self._fit_module)
        self.mod_canvas.bind("<Configure>", lambda e: (self.mod_canvas.itemconfigure(self._mod_win, width=e.width),
                                                        self._fit_module()))
        self._main_content.bind("<Configure>", self._fit_module, add="+")
        self._vsb_shown = False

        def wheel(e):
            if self._vsb_shown:
                d = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
                self.mod_canvas.yview_scroll(d * 2, "units")
        host.bind("<Enter>", lambda e: (self.root.bind_all("<MouseWheel>", wheel),
                                        self.root.bind_all("<Button-4>", wheel),
                                        self.root.bind_all("<Button-5>", wheel)))
        host.bind("<Leave>", lambda e: (self.root.unbind_all("<MouseWheel>"),
                                        self.root.unbind_all("<Button-4>"),
                                        self.root.unbind_all("<Button-5>")))

    def _fit_module(self, event=None):
        try:
            req = self.pages_outer.winfo_reqheight()
            main_h = self._main_content.winfo_height()
            if main_h < 50 or req < 5:
                return
            cur = int(self.mod_canvas.cget("height"))
            chrome = self.module_card.content_size()[1] - cur
            tb = self.toolbar_card.winfo_height() + 20
            allowed = main_h - tb - 190 - chrome - 16      # 190px = waveform minimum
            h = max(110, min(req, allowed))
            if h != cur:
                self.mod_canvas.configure(height=h)
            self.mod_canvas.configure(scrollregion=(0, 0, 1, req))
            need = req > h + 1
            if need and not self._vsb_shown:
                self.mod_vsb.pack(side=tk.RIGHT, fill=tk.Y, before=self.mod_canvas)
                self._vsb_shown = True
            elif not need and self._vsb_shown:
                self.mod_vsb.pack_forget()
                self.mod_canvas.yview_moveto(0)
                self._vsb_shown = False
        except tk.TclError:
            pass

    def _build_pages(self):
        def add_page(key):
            frame = ttk.Frame(self.pages_outer, style="Card.TFrame")
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = frame
            return frame

        left = add_page("edit")
        right = add_page("convolution")
        space_sim = add_page("space")
        # sampling_page = add_page("sampling")  # commented out
        analysis = add_page("analysis")
        decompose = add_page("decompose")
        equalizer = add_page("equalizer")
        noise = add_page("noise")
        convert = add_page("convert")
        yt_capture = add_page("youtube")
        voice = add_page("voice")
        doppler = add_page("doppler")
        bpm_tab = add_page("bpm")
        stress_tab = add_page("stress")
        sonar_tab = add_page("sonar")

        def flow_for(page):
            fl = FlowFrame(page)
            fl.pack(fill=tk.X)
            return fl

        def hrow(parent):
            r = ttk.Frame(parent, style="Card.TFrame")
            r.pack(anchor="w")
            return r

        # --- Sub-Module 1: Trim & Edit Controls ---
        fl = flow_for(left)
        t, inn = self._tile(fl, "Trim")
        row = hrow(inn)
        f, self.trim_start = self._field(row, "Start time (s)", 8, tip="Start of the region to keep, in seconds.")
        f.pack(side=tk.LEFT, padx=(0, 10))
        f, self.trim_end = self._field(row, "End time (s)", 8, tip="End of the region to keep, in seconds.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Apply Cut/Trim", "scissors", self.do_trim,
                      "Truncate signal data outside specified start and end timestamps.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "Gain")
        row = hrow(inn)
        f, self.scale_factor = self._field(row, "Multiplier (1.0 = unchanged)", 10, "1.0",
                                           tip="Linear amplitude factor: 0.5 halves the level, 2.0 doubles it.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Scale Volume Gain", "volume", self.do_scale,
                      "Multiply linear signal amplitude values by numeric factor.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "Fade")
        row = hrow(inn)
        f, self.fade_in = self._field(row, "Fade in (s)", 6, "0", tip="Length of the volume ramp-up at the start, in seconds.")
        f.pack(side=tk.LEFT, padx=(0, 10))
        f, self.fade_out = self._field(row, "Fade out (s)", 6, "0", tip="Length of the volume ramp-down at the end, in seconds.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Apply Ramp Envelope", "ramp", self.do_fade,
                      "Apply linear ramp volume envelopes to boundary ends of track.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "Reverse")
        row = hrow(inn)
        b = self._btn(row, "Reverse Audio Stream", "reverse", self.do_reverse,
                      "Flip temporal sequence of audio array backward in time.")
        b.pack(side=tk.LEFT, pady=(0, 3))
        fl.add(t, "n")

        # --- Sub-Module 2: Convolution Effects ---
        fl = flow_for(right)
        t, inn = self._tile(fl, "Smoothing")
        row = hrow(inn)
        f, self.smooth_size = self._field(row, "Moving average kernel size", 8, "9",
                                          tip="Number of samples averaged together; larger values smooth more.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Apply Moving Avg", "wave", self.do_smooth,
                      "Convolve signal with uniform boxcar impulse response.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "Echo")
        row = hrow(inn)
        f, self.echo_delay = self._field(row, "Delay (s)", 6, "0.3", tip="Time between the original sound and its echo.")
        f.pack(side=tk.LEFT, padx=(0, 10))
        f, self.echo_decay = self._field(row, "Decay factor", 6, "0.5", tip="How much quieter the echo is (0 - 1).")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Synthesize Echo Delay", "echo", self.do_echo,
                      "Convolve audio with decaying periodic impulse train.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        # --- Sub-Module 2b: Space Simulator (real-IR convolution) ---
        fl = flow_for(space_sim)
        t, inn = self._tile(fl, "Get an impulse response")
        row = hrow(inn)
        f, self.ir_url_entry = self._field(
            row, "Direct IR file URL (e.g. from OpenAIR)", 40,
            tip="Copy a file's direct download link from a recording page on "
                "openair.hosted.york.ac.uk (or any other direct audio URL).")
        f.pack(side=tk.LEFT, padx=(0, 12))
        self.ir_download_btn = self._btn(row, "Download IR", "video", self.do_download_ir,
                                         "Download the impulse response at this URL into memory.", kind="primary")
        self.ir_download_btn.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        b = self._btn(row, "Load IR From Disk", "video", self.do_load_ir_file,
                      "Load a local impulse response WAV/AIFF/FLAC file instead.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "Simulate the space")
        row = hrow(inn)
        f, self.ir_wet = self._field(row, "Wet mix (0-1)", 6, "1.0",
                                     tip="1.0 = fully the convolved/reverberant signal, lower blends in the dry original.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Apply Space Convolution", "sparkle", self.do_apply_ir_convolution,
                      "Convolve the loaded track with the currently loaded impulse response.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")
        self.ir_status_var = tk.StringVar(value="No impulse response loaded.")
        self._status_label(space_sim, self.ir_status_var).pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 2c: Sampling & Aliasing (commented out) ---
        # fl = flow_for(sampling_page)
        # t, inn = self._tile(fl, "Downsample the track")
        # row = hrow(inn)
        # f, self.sampling_rate_entry = self._field(row, "Target sample rate (Hz)", 10, "8000",
        #                                            tip="Must be below the track's current sample rate. Try 8000, then 2000, to hear the Nyquist limit being crossed.")
        # f.pack(side=tk.LEFT, padx=(0, 12))
        # b = self._btn(row, "Naive Downsample (aliased)", "wave", self.do_naive_resample,
        #               "Drops samples with NO anti-aliasing filter first - frequencies above the new Nyquist limit fold back "
        #               "into the wrong lower frequencies instead of being removed. This is the audible/visible aliasing artifact.")
        # b.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        # b = self._btn(row, "Proper Downsample (anti-aliased)", "wave", self.do_proper_resample,
        #               "Band-limits the signal first (FFT-based), so high frequencies are cleanly removed rather than folded back. "
        #               "Lower fidelity, but no aliasing distortion.")
        # b.pack(side=tk.LEFT, anchor="s")
        # fl.add(t, "n")
        # self.sampling_status_var = tk.StringVar(value="Track is at its original sample rate.")
        # self._status_label(sampling_page, self.sampling_status_var).pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 3: FFT & Frequency Analysis ---
        fl = flow_for(analysis)
        t, inn = self._tile(fl, "Visualise")
        row = hrow(inn)
        b = self._btn(row, "Spectrum Plot", "spectrum", self.do_show_spectrum,
                      "Plot frequency domain breakdown using Fast Fourier Transform.")
        b.pack(side=tk.LEFT, padx=(0, 8), pady=(0, 3))
        b = self._btn(row, "Spectrogram Map", "grid", self.do_show_spectrogram,
                      "Compute and plot Short-Time Fourier Transform (STFT) intensity.")
        b.pack(side=tk.LEFT, padx=(0, 8), pady=(0, 3))
        fl.add(t, "n")

        t, inn = self._tile(fl, "Pitch")
        row = hrow(inn)
        b = self._btn(row, "Fundamental Pitch Peak", "bolt", self.do_detect_pitch,
                      "Identify peak dominant fundamental frequency and musical note.")
        b.pack(side=tk.LEFT, padx=(0, 8), pady=(0, 3))
        b = self._btn(row, "Dynamic Piano Roll", "piano", self.do_track_pitch,
                      "Track fundamental note frequencies across time dynamically.")
        b.pack(side=tk.LEFT, pady=(0, 3))
        fl.add(t, "n")

        t, inn = self._tile(fl, "Low-pass")
        row = hrow(inn)
        f, self.lowpass_cutoff = self._field(row, "Cutoff (Hz)", 7, "3000",
                                             tip="Frequencies above this are attenuated.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Lowpass Filter", "lowpass", self.do_lowpass,
                      "Remove content above the cutoff frequency (keeps the lows).")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "High-pass")
        row = hrow(inn)
        f, self.highpass_cutoff = self._field(row, "Cutoff (Hz)", 7, "300",
                                              tip="Frequencies below this are attenuated.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Highpass Filter", "highpass", self.do_highpass,
                      "Remove content below the cutoff frequency (keeps the highs).")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        # --- Sub-Module 4: AI Stem Separator ---
        fl = flow_for(decompose)
        t, inn = self._tile(fl, "Standard bandpass preset filtering")
        row = hrow(inn)
        self.decompose_preset = tk.StringVar(value="Vocals")
        f, preset_menu = self._combo(row, "Target preset", self.decompose_preset, list(ao.DECOMPOSE_PRESETS.keys()), 14,
                                     tip="Pick an instrument preset to fill in a typical frequency band.")
        f.pack(side=tk.LEFT, padx=(0, 10))
        preset_menu.bind("<<ComboboxSelected>>", self.on_preset_selected)
        f, self.decompose_low = self._field(row, "Low (Hz)", 7, tip="Lower edge of the band.")
        f.pack(side=tk.LEFT, padx=(0, 10))
        f, self.decompose_high = self._field(row, "High (Hz)", 7, tip="Upper edge of the band.")
        f.pack(side=tk.LEFT)
        self.on_preset_selected()

        row2 = hrow(inn)
        row2.pack_configure(pady=(6, 0))
        b = self._btn(row2, "Bandstop (Mute Target)", "ban", self.do_decompose_remove,
                      "Cut the selected band out of the audio.")
        b.pack(side=tk.LEFT, padx=(0, 8))
        b = self._btn(row2, "Bandpass (Isolate Target)", "target", self.do_decompose_keep,
                      "Keep only the selected band and remove everything else.")
        b.pack(side=tk.LEFT)
        fl.add(t, "n")

        t, inn = self._tile(fl, "Demucs neural AI stem extractor")
        row3 = hrow(inn)
        f, self.ml_stem = self._combo(row3, "Target stem", None, ["vocals", "drums", "bass", "other"], 12,
                                      tip="Which stem the neural network should isolate.")
        self.ml_stem.current(0)
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row3, "Isolate Stem (Demucs AI)", "sparkle", self.do_demucs_decompose,
                      "Separate the chosen stem with the Demucs neural network (can take a while).", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        # --- Sub-Module 5: Parametric Multi-Band Equalizer ---
        top = ttk.Frame(equalizer, style="Card.TFrame")
        top.pack(fill=tk.X)
        b = self._btn(top, "Reset Sliders", "reset", self.reset_eq_sliders, "Return every band to 0 dB.")
        b.pack(side=tk.RIGHT)
        b = self._btn(top, "Apply Equalizer Gains", "sliders", self.do_equalize,
                      "Apply the current band gains to the audio.", kind="primary")
        b.pack(side=tk.RIGHT, padx=(0, 8))
        eq_hint = tk.StringVar(value="Parametric Equalizer Gain Curves (dB) - drag a band, scroll to nudge, double-click to reset.")
        self._status_label(top, eq_hint).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))

        eq_sliders_row = ttk.Frame(equalizer, style="Card.TFrame")
        eq_sliders_row.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.eq_vars = []
        for name, lo, hi in ao.EQ_BANDS:
            col = ttk.Frame(eq_sliders_row, style="Card.TFrame")
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            var = tk.DoubleVar(value=0.0)
            self.eq_vars.append(var)
            slider = self._reg(EQSlider(col, var, self._theme_colors))
            slider.pack(fill=tk.BOTH, expand=True)
            self._tip(slider, f"Gain for the {str(name).replace(chr(10), ' ')} band (-12 to +12 dB). Double-click to reset.")
            ttk.Label(col, text=name, justify=tk.CENTER, style="Caption.TLabel").pack(pady=(2, 0))

        # --- Sub-Module 6: Spectral Subtraction Noise Removal ---
        fl = flow_for(noise)
        t, inn = self._tile(fl, "Noise sample interval")
        row = hrow(inn)
        f, self.noise_start = self._field(row, "Start (s)", 6, "0", tip="Start of a section containing only background noise.")
        f.pack(side=tk.LEFT, padx=(0, 10))
        f, self.noise_end = self._field(row, "End (s)", 6, "1", tip="End of the noise-only section.")
        f.pack(side=tk.LEFT)
        fl.add(t, "n")

        t, inn = self._tile(fl, "Subtraction")
        row = hrow(inn)
        f, self.noise_strength = self._field(row, "Multiplier", 6, "1.5",
                                             tip="How aggressively the noise profile is subtracted (higher = stronger).")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Execute Spectral Subtraction", "target", self.do_remove_noise,
                      "Learn the noise profile from the interval and subtract it from the whole track.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")
        self.noise_profile_status = tk.StringVar(value="")
        self._status_label(noise, self.noise_profile_status).pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 7: Format Conversion ---
        row = hrow(convert)
        self.convert_format = tk.StringVar(value="mp3")
        f, _cb = self._combo(row, "Target container format", self.convert_format, ["wav", "mp3", "flac", "ogg", "m4a"], 12,
                             tip="File format the audio will be written as.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Convert & Save Audio Track", "export", self.do_convert_save,
                      "Choose a destination and write the audio in the selected format.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")

        # --- Sub-Module 8: Internet Song Search (Internet Archive) ---
        row = hrow(yt_capture)
        f, self.yt_search_entry = self._field(row, "Search query (song name)", 36,
                                              tip="Type a song name; the best matching openly-licensed match is fetched.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        self.archive_download_btn = self._btn(
            row, "Fetch Audio Track (Internet Archive)", "video", self.do_archive_download,
            "Search archive.org and download openly-licensed audio straight into the workspace - "
            "explicitly allowed for free reuse, unlike stream-ripping YouTube.", kind="primary",
        )
        self.archive_download_btn.pack(side=tk.LEFT, anchor="s")

        row2 = hrow(yt_capture)
        f, self.jamendo_client_id_entry = self._field(row2, "Jamendo client_id (free, devportal.jamendo.com)", 26,
                                                       tip="One-time free API key, needed only for the Jamendo search below.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        self.jamendo_download_btn = self._btn(
            row2, "Fetch from Jamendo (CC-licensed)", "video", self.do_jamendo_download,
            "Search Jamendo's licensed catalog (uses the song name above) and download the top match.",
        )
        self.jamendo_download_btn.pack(side=tk.LEFT, anchor="s")

        self.yt_status_var = tk.StringVar(value="Idle")
        self._status_label(yt_capture, self.yt_status_var).pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 9: Speaker Voice Identification ---
        fl = flow_for(voice)
        t, inn = self._tile(fl, "Register a voiceprint")
        row = hrow(inn)
        f, self.speaker_name_entry = self._field(row, "Speaker name", 16, tip="Label to store the current clip's voiceprint under.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Register Voiceprint", "mic", self.do_enroll_speaker,
                      "Extract MFCC features from the current audio and enrol them under this name.")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        b = self._btn(row, "Clear Voiceprints", "trash", self.do_clear_speakers, "Forget every registered voiceprint.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "Identify")
        row = hrow(inn)
        f, self.speaker_threshold = self._field(row, "Distance limit", 6, "40",
                                                tip="Maximum voiceprint distance still counted as a match.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Identify Audio Speaker", "search", self.do_identify_speaker,
                      "Compare the current audio against every registered voiceprint.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")
        self.enrolled_status_var = tk.StringVar(value="No speakers registered.")
        self._status_label(voice, self.enrolled_status_var).pack(fill=tk.X, pady=(10, 0))
        self.speaker_result_var = tk.StringVar(value="")
        self._status_label(voice, self.speaker_result_var, "Result.TLabel").pack(fill=tk.X, pady=(4, 0))

        # --- Sub-Module 10: Doppler Speed Calculator ---
        row = hrow(doppler)
        b = self._btn(row, "Auto-Detect from Clip", "target", self.do_doppler_autodetect,
                      "Automatically split the loaded clip at its loudest (closest pass-by) point and "
                      "detect the approaching/departing pitch on each half.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 20))

        row = hrow(doppler)
        f, self.doppler_approach = self._field(row, "Approaching pitch (Hz)", 10,
                                               tip="Frequency heard while the source was moving toward you.")
        f.pack(side=tk.LEFT, padx=(0, 4))
        b = self._btn(row, "Autofill", "target", self.do_doppler_autofill_approach,
                      "Use the pitch found by 'Fundamental Pitch Peak' as the approaching frequency.")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 12))
        f, self.doppler_depart = self._field(row, "Departing pitch (Hz)", 10,
                                             tip="Frequency heard while the source was moving away from you.")
        f.pack(side=tk.LEFT, padx=(0, 4))
        b = self._btn(row, "Autofill", "target", self.do_doppler_autofill_depart,
                      "Use the pitch found by 'Fundamental Pitch Peak' as the departing frequency.")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 12))
        b = self._btn(row, "Calculate Velocity Shift", "calc", self.do_doppler_calc,
                      "Compute the source speed from the two observed frequencies - no assumed source frequency needed.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        self.doppler_result_var = tk.StringVar(value="")
        self._status_label(doppler, self.doppler_result_var, "Result.TLabel").pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 11: BPM & Tempo ---
        row = hrow(bpm_tab)
        self.bpm_mode = tk.StringVar(value="Song")
        f, mode_menu = self._combo(row, "Detection profile", self.bpm_mode, ["Song", "Heartbeat"], 12,
                                   tip="'Song' looks for music tempo; 'Heartbeat' looks for slower pulses (20-150 per minute).")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Compute Pulse BPM", "pulse", self.do_detect_bpm,
                      "Estimate beats per minute from the amplitude envelope.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        self.bpm_result_var = tk.StringVar(value="")
        self._status_label(bpm_tab, self.bpm_result_var, "Result.TLabel").pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 12: Voice Stress Test (jitter/shimmer) ---
        fl = flow_for(stress_tab)
        t, inn = self._tile(fl, "Analyze pitch & amplitude stability")
        row = hrow(inn)
        f, self.stress_frame_ms = self._field(row, "Frame size (ms)", 8, "30",
                                              tip="Analysis window length. 20-50ms is standard for voice.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        f, self.stress_hop_ms = self._field(row, "Hop size (ms)", 8, "15",
                                            tip="Step between frames. Smaller = smoother track, slower to compute.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Analyze Voice Stress", "volume", self.do_analyze_stress,
                      "Tracks pitch frame-by-frame; frame-to-frame pitch variation (jitter) and amplitude "
                      "variation (shimmer) above a calm baseline suggest vocal tension.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        b = self._btn(row, "Plot Pitch Stability", "spectrum", self.do_plot_stress,
                      "Shows the tracked pitch (and where it's unstable) over the clip's timeline.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")
        self.stress_result_var = tk.StringVar(
            value="Load a clip of continuous speech/vowel sound, then analyze.")
        self._status_label(stress_tab, self.stress_result_var, "Result.TLabel").pack(fill=tk.X, pady=(10, 0))

        # --- Sub-Module 13: Pulse-Echo Distance Measurer (audio sonar) ---
        fl = flow_for(sonar_tab)
        t, inn = self._tile(fl, "1. Generate & play a test pulse")
        row = hrow(inn)
        f, self.sonar_pulse_ms = self._field(row, "Pulse duration (ms)", 8, "20",
                                             tip="A short chirp sweep. Its distinctive shape correlates far more "
                                                 "sharply than a plain click or tone.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Generate Chirp Pulse", "bolt", self.do_generate_chirp,
                      "Creates the outgoing pulse (2kHz-9kHz sweep) used for ranging.")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        self.sonar_play_btn = self._btn(row, "Play Pulse", "play", self.do_play_chirp,
                                        "Plays the generated pulse out loud through your speakers.")
        self.sonar_play_btn.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "2. Record the echo (real mic test)")
        row = hrow(inn)
        self.sonar_record_btn = self._btn(row, "Start Recording", "mic", self.do_sonar_start_record,
                                          "Start capturing mic input, then click 'Play Pulse' while it records, "
                                          "then Stop once the echo has had time to return.", kind="primary")
        self.sonar_record_btn.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        self.sonar_stop_btn = self._btn(row, "Stop Recording", "stop", self.do_sonar_stop_record,
                                        "Stop capturing and keep the recording for measurement.")
        self.sonar_stop_btn.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        self.sonar_stop_btn.config(state=tk.DISABLED)
        b = self._btn(row, "Measure Distance", "search", self.do_measure_echo,
                      "Cross-correlates the recording against the pulse to find the echo delay, "
                      "then converts delay to distance using the speed of sound.", kind="primary")
        b.pack(side=tk.LEFT, anchor="s", padx=(0, 8))
        b = self._btn(row, "Plot Correlation", "spectrum", self.do_plot_correlation,
                      "Shows the cross-correlation curve with the direct-path and echo peaks marked.")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        t, inn = self._tile(fl, "3. Verify the math (no mic needed)")
        row = hrow(inn)
        f, self.sonar_test_distance = self._field(row, "Simulated distance (m)", 8, "2.0",
                                                   tip="Builds a fake recording with a known reflection distance, "
                                                       "then checks the measured result against it.")
        f.pack(side=tk.LEFT, padx=(0, 12))
        b = self._btn(row, "Run Synthetic Self-Test", "target", self.do_sonar_self_test,
                      "Generates a pulse + a delayed, quieter, noisy copy simulating a fake echo at the distance "
                      "above, then measures it - confirming the math works before trying a real room/mic.",
                      kind="primary")
        b.pack(side=tk.LEFT, anchor="s")
        fl.add(t, "n")

        self.sonar_status_var = tk.StringVar(value="Generate a pulse to begin.")
        self._status_label(sonar_tab, self.sonar_status_var, "Result.TLabel").pack(fill=tk.X, pady=(10, 0))

    # --------------------------------------------------------- responsive ---
    def _on_root_configure(self, event, force=False):
        if not force and (event is None or event.widget is not self.root):
            return
        if not self.workbench_frame.winfo_ismapped() and not force:
            return
        w = self.root.winfo_width()
        compact = self._sidebar_compact
        if not compact and w < 1140:
            compact = True
        elif compact and w > 1170:
            compact = False
        if compact != self._sidebar_compact:
            self._set_sidebar_compact(compact)

    def _set_sidebar_compact(self, compact):
        self._sidebar_compact = compact
        self.sidebar.configure(width=78 if compact else 244)
        if compact:
            self.sidebar_title.pack_forget()
        else:
            self.sidebar_title.pack(anchor="w", padx=(22, 0), pady=(18, 8), before=self.nav_frame)
        for b in self.nav_buttons.values():
            b.set_compact(compact)
        self.theme_toggle_btn.set_compact(compact)

    # ------------------------------------------------------------- theme ----
    def _apply_theme(self, name):
        self.theme_name = name
        c = THEMES[name]
        self._theme_colors = c

        self.root.configure(bg=c["bg"])
        style = self.style
        card = c["card_bg"]

        style.configure(".", background=c["bg"], foreground=c["fg"], fieldbackground=c["entry_bg"],
                        bordercolor=c["border"], font=F(9))
        style.configure("TFrame", background=c["bg"])
        style.configure("TLabel", background=c["bg"], foreground=c["fg"], font=F(9))
        style.configure("TSeparator", background=c["border"])

        style.configure("Card.TFrame", background=card)
        style.configure("Card.TLabel", background=card, foreground=c["fg"], font=F(9))
        style.configure("CardTitle.TLabel", background=card, foreground=c["fg"], font=F(12, "bold"))
        style.configure("TileTitle.TLabel", background=card, foreground=c["fg"], font=F(9, "bold"))
        style.configure("Caption.TLabel", background=card, foreground=c["fg_muted"], font=F(8))
        style.configure("Caption2.TLabel", background=card, foreground=c["fg_muted"], font=F(9))
        style.configure("Status.TLabel", background=card, foreground=c["fg_muted"], font=F(9))
        style.configure("Result.TLabel", background=card, foreground=c["accent"], font=F(10, "bold"))
        style.configure("Mono.TLabel", background=card, foreground=c["fg_muted"], font=(FONT_MONO, 9))
        style.configure("Muted.TLabel", background=card, foreground=c["fg_muted"], font=F(8))

        style.configure("Sidebar.TFrame", background=c["sidebar_bg"])
        style.configure("SidebarTitle.TLabel", background=c["sidebar_bg"], foreground=c["fg_muted"], font=F(9, "bold"))
        style.configure("Status.TFrame", background=c["bg_alt"])
        style.configure("Tooltip.TLabel", background=c["bg_alt"], foreground=c["fg_muted"], font=F(9))
        style.configure("TipTag.TLabel", background=c["bg_alt"], foreground=c["accent"], font=F(8, "bold"))

        # inputs: flat, 1px border that lights up with the accent on focus
        style.configure("TEntry", fieldbackground=c["entry_bg"], foreground=c["fg"], insertcolor=c["fg"],
                        bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                        selectbackground=c["accent"], selectforeground=c["accent_fg"], padding=(8, 7), relief="flat")
        style.map("TEntry",
                  bordercolor=[("focus", c["accent"]), ("hover", mix(c["border"], c["accent"], 0.5))],
                  lightcolor=[("focus", c["accent"]), ("hover", mix(c["border"], c["accent"], 0.5))],
                  darkcolor=[("focus", c["accent"]), ("hover", mix(c["border"], c["accent"], 0.5))])
        style.configure("TCombobox", fieldbackground=c["entry_bg"], foreground=c["fg"], background=c["entry_bg"],
                        arrowcolor=c["fg_muted"], bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                        selectbackground=c["entry_bg"], selectforeground=c["fg"], padding=(8, 7), arrowsize=14,
                        relief="flat")
        style.map("TCombobox",
                  fieldbackground=[("readonly", c["entry_bg"])], foreground=[("readonly", c["fg"])],
                  background=[("active", c["entry_bg"]), ("readonly", c["entry_bg"])],
                  selectbackground=[("readonly", c["entry_bg"])], selectforeground=[("readonly", c["fg"])],
                  arrowcolor=[("active", c["accent"]), ("focus", c["accent"])],
                  bordercolor=[("focus", c["accent"]), ("hover", mix(c["border"], c["accent"], 0.5))],
                  lightcolor=[("focus", c["accent"])], darkcolor=[("focus", c["accent"])])
        for sb in ("Horizontal.TScrollbar", "Vertical.TScrollbar"):
            style.configure(sb, background=mix(c["card_bg"], c["fg"], 0.18), troughcolor=c["card_bg"],
                            bordercolor=c["card_bg"], lightcolor=mix(c["card_bg"], c["fg"], 0.18),
                            darkcolor=mix(c["card_bg"], c["fg"], 0.18), arrowcolor=c["card_bg"],
                            relief="flat", arrowsize=8, gripcount=0, width=8)
            style.map(sb, background=[("active", c["accent"])])

        for w in self._themed:
            try:
                w.set_theme(c)
            except tk.TclError:
                pass
        for pname, btn in self.nav_buttons.items():
            btn.set_theme(c, self._active_page, pname)

        if hasattr(self, "header_canvas"):
            self.header_canvas.set_colors(c)
        if hasattr(self, "theme_toggle_btn"):
            dark = (name == "dark")
            self.theme_toggle_btn.config(text=("Light mode" if dark else "Dark mode"), icon=("sun" if dark else "moon"))
        if hasattr(self, "page_badge"):
            self._update_page_header()
        self._style_combo_popdowns(c)
        if hasattr(self, "mod_canvas"):
            self.mod_canvas.configure(bg=c["card_bg"])

        if hasattr(self, "fig"):
            self.canvas.get_tk_widget().configure(bg=c["card_bg"])
            if self.data is not None:
                self._refresh_plot()
            else:
                self._draw_empty_state(c)

    def _style_combo_popdowns(self, c):
        self.root.option_add("*TCombobox*Listbox.background", c["entry_bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", c["accent_fg"])
        for cb in self._combos:
            try:
                pop = str(self.root.tk.call("ttk::combobox::PopdownWindow", str(cb)))
                self.root.tk.call(pop + ".f.l", "configure", "-background", c["entry_bg"], "-foreground", c["fg"],
                                  "-selectbackground", c["accent"], "-selectforeground", c["accent_fg"])
            except tk.TclError:
                pass

    def toggle_theme(self):
        self._apply_theme("dark" if self.theme_name == "light" else "light")

    def _select_page(self, key):
        self._active_page = key
        self.pages[key].tkraise()
        for pname, btn in self.nav_buttons.items():
            btn.set_theme(self._theme_colors, key, pname)
        self._update_page_header()

    def _update_page_header(self):
        info = next(p for p in PAGES if p[0] == self._active_page)
        c = self._theme_colors
        cv = self.page_badge
        cv.delete("all")
        cv.configure(bg=c["card_bg"])
        cv.create_polygon(rr_points(1, 1, 37, 37, 11), smooth=True, fill=info[3], outline="")
        draw_icon(cv, info[1], 19, 19, 20, ink_on(info[3]))
        self.page_title_var.set(info[2])
        self.page_desc_var.set(PAGE_DESCRIPTIONS[self._active_page])

    # ------------------------------------------------------------- plotting -
    def _snapshot(self):
        if self.data is not None:
            self.history.append(self.data.copy())

    def _theme_axes(self, fig, ax, c, face="card_bg"):
        fig.patch.set_facecolor(c[face])
        ax.set_facecolor(c["plot_bg"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(c["border"])
        ax.tick_params(colors=c["fg_muted"], labelsize=8, length=0, pad=5)
        ax.xaxis.label.set_color(c["fg_muted"])
        ax.yaxis.label.set_color(c["fg_muted"])
        ax.title.set_color(c["fg"])
        # softened, dotted guides instead of a hard grid
        ax.grid(True, color=c["plot_grid"], linewidth=0.7, alpha=0.75, linestyle=(0, (1, 3)))
        ax.set_axisbelow(True)

    def _on_fig_resize(self, event):
        w, h = event.width, event.height
        if w < 120 or h < 90:
            return
        self.fig.subplots_adjust(left=min(0.25, 62 / w), right=1 - 12 / w, bottom=min(0.45, 40 / h), top=1 - 10 / h)

    def _draw_empty_state(self, c):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Time (seconds)", fontfamily=FONT_FAMILY, fontsize=9)
        ax.set_ylabel("Amplitude", fontfamily=FONT_FAMILY, fontsize=9)
        ax.axhline(0, color=c["plot_grid"], linewidth=1.0, zorder=1)
        ax.text(0.5, 0.56, "No audio loaded", transform=ax.transAxes, ha="center", va="center",
                color=c["fg"], fontsize=12, fontweight="bold", fontfamily=FONT_FAMILY)
        ax.text(0.5, 0.40, "Use Open File to load a clip, record from your microphone, or fetch a track from YouTube.",
                transform=ax.transAxes, ha="center", va="center", color=c["fg_muted"], fontsize=9,
                fontfamily=FONT_FAMILY)
        self._theme_axes(self.fig, ax, c)
        self.canvas.draw_idle()

    def _draw_waveform(self, c):
        ax = self.ax
        y = self.data if self.data.ndim == 1 else self.data.mean(axis=1)
        n = len(y)
        dur = n / self.sr
        bins = 2400
        if n > bins * 2:
            # min/max envelope per pixel column: fast for any length, and peaks are never lost
            edges = np.linspace(0, n, bins + 1).astype(int)
            lo = np.minimum.reduceat(y, edges[:-1])
            hi = np.maximum.reduceat(y, edges[:-1])
            t = (edges[:-1] + edges[1:]) / 2.0 / self.sr
        else:
            t = np.arange(n) / self.sr
            lo, hi = np.minimum(y, 0), np.maximum(y, 0)
        peak = max(float(np.max(np.abs(y))), 1e-6)
        lim = peak * 1.12

        # gradient fill: translucent near the centre line, stronger toward the peaks
        poly = ax.fill_between(t, lo, hi, facecolor="none", edgecolor="none", linewidth=0)
        ramp = np.linspace(1.0, -1.0, 256)
        img = np.zeros((256, 1, 4))
        img[:, 0, :3] = to_rgb(c["plot_line"])
        img[:, 0, 3] = 0.12 + 0.55 * np.abs(ramp) ** 1.2
        im = ax.imshow(img, extent=[0, dur, -lim, lim], origin="upper", aspect="auto", interpolation="bilinear", zorder=2)
        im.set_clip_path(poly.get_paths()[0], ax.transData)

        ax.axhline(0, color=c["plot_grid"], linewidth=0.8, zorder=1)
        ax.plot(t, hi, linewidth=0.8, color=c["plot_line"], alpha=0.95, zorder=3)
        if lo is not hi:
            ax.plot(t, lo, linewidth=0.8, color=c["plot_line"], alpha=0.95, zorder=3)
        ax.set_xlim(0, dur)
        ax.set_ylim(-lim, lim)

    def _refresh_plot(self):
        c = self._theme_colors
        self.ax.clear()
        self.ax.set_xlabel("Time (seconds)", fontfamily=FONT_FAMILY, fontsize=9)
        self.ax.set_ylabel("Amplitude", fontfamily=FONT_FAMILY, fontsize=9)
        self.seek_time = 0.0
        self._seek_line = None
        self._is_playing = False
        self.now_playing_var.set("")
        if self.data is not None:
            self._draw_waveform(c)
            dur = len(self.data) / self.sr
            self.status_var.set(f"Loaded: {dur:.2f}s  |  sr={self.sr} Hz  |  {len(self.data)} frames")
            self._draw_seek_marker()
            self._theme_axes(self.fig, self.ax, c)
            self.canvas.draw()
        else:
            self._draw_empty_state(c)

    def _draw_seek_marker(self):
        if self._seek_line is not None:
            try:
                self._seek_line.remove()
            except (ValueError, NotImplementedError):
                pass
        self._seek_line = self.ax.axvline(self.seek_time, color=self._theme_colors["seek_line"], linewidth=1.6, zorder=6)
        self.seek_status_var.set(f"Seek Head Position: {self.seek_time:.2f}s (Click waveform to jump position)")
        self.canvas.draw_idle()

    def _on_waveform_click(self, event):
        if self.data is None or event.inaxes != self.ax or event.xdata is None:
            return
        duration = len(self.data) / self.sr
        self.seek_time = max(0.0, min(event.xdata, duration))
        self._draw_seek_marker()
        self._play_from_seek()

    def _play_from_seek(self):
        if not PLAYBACK_AVAILABLE or self.data is None:
            return
        sd.stop()
        start_sample = int(self.seek_time * self.sr)
        start_sample = max(0, min(start_sample, len(self.data) - 1))
        sd.play(self.data[start_sample:], self.sr)

        self._playback_start_walltime = time.time()
        self._playback_start_seek = start_sample / self.sr
        self._is_playing = True
        title = os.path.basename(str(self.filepath)) if self.filepath else "Untitled Clip"
        self.now_playing_var.set(f"▶ Now Playing: {title}")

    def _playhead_heartbeat(self):
        try:
            if self._is_playing and self.data is not None and self._playback_start_walltime is not None:
                elapsed = time.time() - self._playback_start_walltime
                current_pos = self._playback_start_seek + elapsed
                duration = len(self.data) / self.sr

                if current_pos >= duration:
                    self._is_playing = False
                    self.now_playing_var.set("")
                    self.seek_time = duration
                else:
                    self.seek_time = current_pos

                # Move the existing line instead of doing a full
                # _draw_seek_marker() (which remove()s + recreates the
                # Line2D) - cheaper at 20fps and avoids flicker.
                if self._seek_line is not None:
                    self._seek_line.set_xdata([self.seek_time, self.seek_time])
                    self.seek_status_var.set(
                        f"Seek Head Position: {self.seek_time:.2f}s (Click waveform to jump position)"
                    )
                    self.canvas.draw_idle()
        except Exception:
            pass
        finally:
            self.root.after(50, self._playhead_heartbeat)

    def _require_audio(self):
        if self.data is None:
            messagebox.showwarning("No Audio Track", "Load or record an audio clip first.")
            return False
        return True

    def delete_audio(self):
        if self.data is None:
            messagebox.showinfo("Clear Buffer", "No active audio track loaded.")
            return

        if not messagebox.askyesno("Confirm Clear", "Clear active workspace audio buffer?"):
            return

        self.stop()
        self.sr = None
        self.data = None
        self.history.clear()
        self.filepath = None
        self.seek_time = 0.0
        self._seek_line = None
        self._is_playing = False
        self._playback_start_walltime = None
        self._playback_start_seek = None
        self._last_detected_freq = None

        self.status_var.set("Status: Idle (No Audio Loaded)")
        self.seek_status_var.set("Click interactive waveform canvas to place playhead position.")
        self._refresh_plot()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3"), ("WAV files", "*.wav"), ("MP3 files", "*.mp3")])
        if not path:
            return
        try:
            self.sr, self.data = ao.load_audio(path)
            self.filepath = path
            self.history.clear()
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def join_file(self):
        if not self._require_audio():
            return
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")])
        if not path:
            return
        try:
            sr2, data2 = ao.load_audio(path)
            if sr2 != self.sr:
                messagebox.showerror("Sample Rate Error", f"Sample rate mismatch ({self.sr} Hz vs {sr2} Hz).")
                return
            self._snapshot()
            self.data = ao.join(self.data, data2)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Join Audio Error", str(e))

    def save_file(self):
        if not self._require_audio():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("FLAC files", "*.flac"), ("OGG files", "*.ogg"), ("M4A files", "*.m4a")],
        )
        if not path:
            return
        try:
            ao.save_audio(path, self.sr, self.data)
            messagebox.showinfo("Export Success", f"Audio track written to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def do_convert_save(self):
        if not self._require_audio():
            return
        fmt = self.convert_format.get()
        path = filedialog.asksaveasfilename(
            title="Save Converted File", defaultextension=f".{fmt}", filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")]
        )
        if not path:
            return
        try:
            ao.save_audio(path, self.sr, self.data, fmt=fmt)
            messagebox.showinfo("Conversion Success", f"Converted track written as {fmt.upper()}:\n{path}")
        except Exception as e:
            messagebox.showerror("Conversion Error", str(e))

    def undo(self):
        if not self.history:
            messagebox.showinfo("Undo Engine", "Undo history buffer empty.")
            return
        self.data = self.history.pop()
        self._refresh_plot()

    def play(self):
        if not self._require_audio():
            return
        if not PLAYBACK_AVAILABLE:
            messagebox.showwarning("Playback Error", "PortAudio sound driver missing.")
            return
        self._play_from_seek()

    def stop(self):
        if PLAYBACK_AVAILABLE:
            sd.stop()
        self._is_playing = False
        self.now_playing_var.set("")

    def do_trim(self):
        if not self._require_audio():
            return
        try:
            start = float(self.trim_start.get())
            end = float(self.trim_end.get())
            self._snapshot()
            self.data = ao.trim(self.data, self.sr, start, end)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Trim Error", str(e))

    def do_reverse(self):
        if not self._require_audio():
            return
        self._snapshot()
        self.data = ao.reverse(self.data)
        self._refresh_plot()

    def do_scale(self):
        if not self._require_audio():
            return
        try:
            factor = float(self.scale_factor.get())
            self._snapshot()
            self.data = ao.scale(self.data, factor)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Gain Scale Error", str(e))

    def do_fade(self):
        if not self._require_audio():
            return
        try:
            fin = float(self.fade_in.get())
            fout = float(self.fade_out.get())
            self._snapshot()
            self.data = ao.fade(self.data, self.sr, fin, fout)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Ramp Envelope Error", str(e))

    def do_smooth(self):
        if not self._require_audio():
            return
        try:
            size = int(self.smooth_size.get())
            self._snapshot()
            self.data = ao.smooth(self.data, size)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Convolution Error", str(e))

    def do_echo(self):
        if not self._require_audio():
            return
        try:
            delay = float(self.echo_delay.get())
            decay = float(self.echo_decay.get())
            self._snapshot()
            self.data = ao.echo(self.data, self.sr, delay, decay)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Echo Synthesis Error", str(e))

    def _show_plot_window(self, title, plot_fn):
        c = self._theme_colors
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=c["bg"])
        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        plot_fn(ax)
        self._theme_axes(fig, ax, c)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def do_show_spectrum(self):
        if not self._require_audio():
            return
        freqs, mag = ao.compute_spectrum(self.data, self.sr)
        pitch = ao.detect_pitch(self.data, self.sr)

        def plot(ax):
            ax.plot(freqs, mag, linewidth=0.8, color=self._theme_colors["plot_line"])
            ax.set_xlabel("Frequency Bin (Hz)", fontfamily=FONT_FAMILY)
            ax.set_ylabel("FFT Magnitude Spectrum", fontfamily=FONT_FAMILY)
            ax.set_title("Frequency Domain Spectrum Peak View", fontfamily=FONT_FAMILY, fontweight="bold")
            ax.set_xlim(0, self.sr / 2)

            if pitch and not pitch["is_silent"]:
                note = pitch["note_info"]["full_note"]
                freq = pitch["frequency"]
                max_mag = np.max(mag)
                ax.annotate(
                    f"Peak Pitch: {note} ({freq} Hz)",
                    xy=(freq, max_mag),
                    xytext=(freq, max_mag * 0.85),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                    fontfamily=FONT_FAMILY
                )

        self._show_plot_window("Frequency Domain Spectrum", plot)

    def do_show_spectrogram(self):
        if not self._require_audio():
            return
        f, t, Sxx = ao.compute_spectrogram(self.data, self.sr)

        def plot(ax):
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
            ax.set_xlabel("Time (s)", fontfamily=FONT_FAMILY)
            ax.set_ylabel("Frequency (Hz)", fontfamily=FONT_FAMILY)
            ax.set_title("STFT Intensity Spectrogram", fontfamily=FONT_FAMILY, fontweight="bold")

        self._show_plot_window("Short-Time Fourier Transform Spectrogram", plot)

    def do_detect_pitch(self):
        if not self._require_audio():
            return
        pitch_data = ao.detect_pitch(self.data, self.sr)
        if not pitch_data or pitch_data["is_silent"]:
            messagebox.showinfo("Pitch Peak Detector", "No dominant fundamental frequency detected.")
            return

        info = pitch_data["note_info"]
        freq = pitch_data["frequency"]
        self._last_detected_freq = freq
        cents = info["cents"]
        cents_str = f"+{cents}" if cents > 0 else f"{cents}"

        msg = (
            f"🎵 Peak Dominant Pitch: {freq} Hz\n"
            f"🎼 Nearest Musical Note: {info['full_note']}\n"
            f"🎯 Fine Tuning Offset: {cents_str} cents"
        )
        messagebox.showinfo("Fundamental Pitch Detector", msg)

    def do_lowpass(self):
        if not self._require_audio():
            return
        try:
            cutoff = float(self.lowpass_cutoff.get())
            self._snapshot()
            self.data = ao.lowpass_filter(self.data, self.sr, cutoff)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Lowpass Filter Error", str(e))

    def do_highpass(self):
        if not self._require_audio():
            return
        try:
            cutoff = float(self.highpass_cutoff.get())
            self._snapshot()
            self.data = ao.highpass_filter(self.data, self.sr, cutoff)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Highpass Filter Error", str(e))

    def do_track_pitch(self):
        if not self._require_audio():
            return

        times, midi_notes = ao.track_pitch_over_time(self.data, self.sr, hop_sec=0.25)
        valid_midi = midi_notes[~np.isnan(midi_notes)]
        if len(times) == 0 or len(valid_midi) == 0:
            messagebox.showinfo("Piano Roll Tracker", "No clear musical note sequence detected.")
            return

        c = self._theme_colors
        win = tk.Toplevel(self.root)
        win.title("Dynamic Piano Roll Visualizer")
        win.geometry("1100x550")
        win.configure(bg=c["bg"])

        canvas_container = tk.Canvas(win, bg=c["bg"], highlightthickness=0)
        hbar = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=canvas_container.xview)
        canvas_container.configure(xscrollcommand=hbar.set)

        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scroll_frame = ttk.Frame(canvas_container)
        canvas_container.create_window((0, 0), window=scroll_frame, anchor="nw")

        duration = times[-1] if len(times) > 0 else 10
        fig_width = min(35.0, max(12.0, duration * 0.15))
        fig_height = 4.8

        fig = Figure(figsize=(fig_width, fig_height), dpi=100)
        ax = fig.add_subplot(111)

        ax.scatter(times, midi_notes, s=24, color=c["plot_line"], alpha=0.85)
        ax.set_xlabel("Time (seconds)", fontfamily=FONT_FAMILY)
        ax.set_ylabel("Musical Note Pitch", fontfamily=FONT_FAMILY)
        ax.set_title("Dynamic Fundamental Pitch MIDI Visualizer", fontfamily=FONT_FAMILY, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        min_m = int(np.floor(np.min(valid_midi)))
        max_m = int(np.ceil(np.max(valid_midi)))

        step = 1 if (max_m - min_m) <= 24 else 2
        yticks = np.arange(min_m, max_m + 1, step)

        yticklabels = []
        for m in yticks:
            freq = 440.0 * (2 ** ((m - 69) / 12))
            note_info = ao.pd.frequency_to_note(freq)
            yticklabels.append(note_info["full_note"] if note_info else "")

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=8, fontfamily=FONT_FAMILY)
        ax.set_ylim(min_m - 1, max_m + 1)
        self._theme_axes(fig, ax, c)
        fig.tight_layout()

        plot_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        win.update_idletasks()
        canvas_container.config(scrollregion=canvas_container.bbox("all"))

    def on_preset_selected(self, event=None):
        low, high = ao.DECOMPOSE_PRESETS[self.decompose_preset.get()]
        self.decompose_low.delete(0, tk.END)
        self.decompose_low.insert(0, str(low))
        self.decompose_high.delete(0, tk.END)
        self.decompose_high.insert(0, str(high))

    def do_decompose_remove(self):
        if not self._require_audio():
            return
        try:
            low = float(self.decompose_low.get())
            high = float(self.decompose_high.get())
            self._snapshot()
            self.data = ao.bandstop_filter(self.data, self.sr, low, high)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Bandstop Filter Error", str(e))

    def do_decompose_keep(self):
        if not self._require_audio():
            return
        try:
            low = float(self.decompose_low.get())
            high = float(self.decompose_high.get())
            self._snapshot()
            self.data = ao.bandpass_filter(self.data, self.sr, low, high)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Bandpass Filter Error", str(e))

    def do_demucs_decompose(self):
        if not self._require_audio():
            return
        stem = self.ml_stem.get()
        try:
            messagebox.showinfo("Neural AI Engine", f"Extracting '{stem}' stem using Demucs AI...")
            isolated_data, new_sr = ao.separate_audio_demucs(self.data, self.sr, stem=stem)
            self._snapshot()
            self.data = isolated_data
            self.sr = new_sr
            self._refresh_plot()
            messagebox.showinfo("Success", f"Isolated '{stem}' audio stem successfully.")
        except Exception as e:
            messagebox.showerror("Demucs AI Error", str(e))

    def reset_eq_sliders(self):
        for var in self.eq_vars:
            var.set(0.0)

    def do_equalize(self):
        if not self._require_audio():
            return
        try:
            gains_db = [var.get() for var in self.eq_vars]
            self._snapshot()
            self.data = ao.equalize(self.data, self.sr, gains_db)
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Equalizer Error", str(e))

    def do_remove_noise(self):
        if not self._require_audio():
            return
        try:
            start = float(self.noise_start.get())
            end = float(self.noise_end.get())
            strength = float(self.noise_strength.get())
            profile = ao.estimate_noise_profile(self.data, self.sr, start, end)
            self._snapshot()
            self.data = ao.remove_noise(self.data, self.sr, profile, strength=strength)
            self.noise_profile_status.set(f"Spectral noise profile captured [{start:.2f}s-{end:.2f}s] - Subtraction complete.")
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Spectral Noise Subtraction Error", str(e))

    def do_archive_download(self):
        query = self.yt_search_entry.get().strip()
        if not query:
            messagebox.showwarning("Internet Archive", "Type a search query first.")
            return
        if self._archive_download_thread is not None and self._archive_download_thread.is_alive():
            return

        self.archive_download_btn.config(state=tk.DISABLED)
        self.yt_status_var.set(f"Searching Internet Archive for '{query}'...")

        def progress_hook(status):
            st = status.get("status")
            if st == "searching":
                self.root.after(0, self.yt_status_var.set, f"Searching Internet Archive for '{query}'...")
            elif st == "downloading":
                title = status.get("title", query)
                self.root.after(0, self.yt_status_var.set, f"Downloading '{title}' from Internet Archive...")
            elif st == "converting":
                self.root.after(0, self.yt_status_var.set, "Download complete, converting to WAV...")

        def worker():
            try:
                sr, data, title = ao.fetch_from_archive_org(query, progress_hook=progress_hook)
                self.root.after(0, self._archive_download_done, sr, data, title, None)
            except Exception as e:
                self.root.after(0, self._archive_download_done, None, None, None, e)

        self._archive_download_thread = threading.Thread(target=worker, daemon=True)
        self._archive_download_thread.start()

    def _archive_download_done(self, sr, data, title, error):
        self.archive_download_btn.config(state=tk.NORMAL)

        if error is not None:
            messagebox.showerror("Internet Archive Error", str(error))
            self.yt_status_var.set("Internet Archive fetch failed.")
            return

        if data is None or len(data) == 0:
            self.yt_status_var.set("Empty audio returned from Internet Archive.")
            return

        self.sr = sr
        self.data = data
        self.filepath = f"Internet Archive: {title}" if title else "Internet Archive Download"
        self.history.clear()
        self._refresh_plot()
        duration = len(data) / sr
        self.yt_status_var.set(f"Fetched '{title}' ({duration:.2f}s) from Internet Archive - safe to use, no ToS issue.")

    def do_download_ir(self):
        url = self.ir_url_entry.get().strip()
        if not url:
            messagebox.showwarning("Space Simulator", "Paste a direct impulse-response file URL first.")
            return
        if self._ir_download_thread is not None and self._ir_download_thread.is_alive():
            return

        self.ir_download_btn.config(state=tk.DISABLED)
        self.ir_status_var.set(f"Downloading impulse response from '{url}'...")

        def progress_hook(status):
            if status.get("status") == "converting":
                self.root.after(0, self.ir_status_var.set, "Download complete, decoding impulse response...")

        def worker():
            try:
                sr, data, name = ao.fetch_impulse_response(url, progress_hook=progress_hook)
                self.root.after(0, self._ir_download_done, sr, data, name, None)
            except Exception as e:
                self.root.after(0, self._ir_download_done, None, None, None, e)

        self._ir_download_thread = threading.Thread(target=worker, daemon=True)
        self._ir_download_thread.start()

    def _ir_download_done(self, sr, data, name, error):
        self.ir_download_btn.config(state=tk.NORMAL)
        if error is not None:
            messagebox.showerror("Impulse Response Download Error", str(error))
            self.ir_status_var.set("Impulse response download failed.")
            return
        self.ir_sr, self.ir_data, self.ir_name = sr, data, name
        duration = len(data) / sr
        self.ir_status_var.set(f"Loaded impulse response '{name}' ({duration:.2f}s, sr={sr} Hz) - ready to convolve.")

    def do_load_ir_file(self):
        path = filedialog.askopenfilename(
            title="Load Impulse Response",
            filetypes=[("Audio files", "*.wav *.aiff *.aif *.flac *.ogg *.mp3"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            sr, data = ao.load_audio(path)
            self.ir_sr, self.ir_data, self.ir_name = sr, data, os.path.basename(path)
            duration = len(data) / sr
            self.ir_status_var.set(f"Loaded impulse response '{self.ir_name}' ({duration:.2f}s, sr={sr} Hz) - ready to convolve.")
        except Exception as e:
            messagebox.showerror("Impulse Response Load Error", str(e))

    def do_apply_ir_convolution(self):
        if not self._require_audio():
            return
        if self.ir_data is None:
            messagebox.showinfo("Space Simulator", "Download or load an impulse response first.")
            return
        try:
            wet = float(self.ir_wet.get())
            self._snapshot()
            self.data = ao.convolve_with_ir(self.data, self.sr, self.ir_data, self.ir_sr, wet=wet)
            self._refresh_plot()
            self.ir_status_var.set(f"Applied '{self.ir_name}' convolution (wet={wet:.2f}).")
        except Exception as e:
            messagebox.showerror("Space Convolution Error", str(e))

    def do_jamendo_download(self):
        query = self.yt_search_entry.get().strip()
        client_id = self.jamendo_client_id_entry.get().strip()
        if not query:
            messagebox.showwarning("Jamendo", "Type a search query first.")
            return
        if self._jamendo_download_thread is not None and self._jamendo_download_thread.is_alive():
            return

        self.jamendo_download_btn.config(state=tk.DISABLED)
        self.yt_status_var.set(f"Searching Jamendo for '{query}'...")

        def worker():
            try:
                sr, data, title = ao.fetch_from_jamendo(query, client_id)
                self.root.after(0, self._jamendo_download_done, sr, data, title, None)
            except Exception as e:
                self.root.after(0, self._jamendo_download_done, None, None, None, e)

        self._jamendo_download_thread = threading.Thread(target=worker, daemon=True)
        self._jamendo_download_thread.start()

    def _jamendo_download_done(self, sr, data, title, error):
        self.jamendo_download_btn.config(state=tk.NORMAL)
        if error is not None:
            messagebox.showerror("Jamendo Error", str(error))
            self.yt_status_var.set("Jamendo fetch failed.")
            return
        if data is None or len(data) == 0:
            self.yt_status_var.set("Empty audio returned from Jamendo.")
            return
        self.sr = sr
        self.data = data
        self.filepath = f"Jamendo: {title}" if title else "Jamendo Download"
        self.history.clear()
        self._refresh_plot()
        duration = len(data) / sr
        self.yt_status_var.set(f"Fetched '{title}' ({duration:.2f}s) from Jamendo - CC-licensed.")

    # def do_naive_resample(self):
    #     if not self._require_audio():
    #         return
    #     try:
    #         target_sr = int(float(self.sampling_rate_entry.get()))
    #         self._snapshot()
    #         down_data, down_sr = ao.resample_naive(self.data, self.sr, target_sr)
    #         # Upsample back to the original rate purely for playback/plotting -
    #         # the aliasing damage is already baked in and stays audible.
    #         self.data, _ = ao.upsample_to(down_data, down_sr, self.sr)
    #         self._refresh_plot()
    #         self.sampling_status_var.set(
    #             f"Naively downsampled to {target_sr} Hz (Nyquist limit = {target_sr / 2:.0f} Hz) - "
    #             f"listen for harsh/robotic aliasing artifacts."
    #         )
    #     except Exception as e:
    #         messagebox.showerror("Sampling Error", str(e))

    # def do_proper_resample(self):
    #     if not self._require_audio():
    #         return
    #     try:
    #         target_sr = int(float(self.sampling_rate_entry.get()))
    #         self._snapshot()
    #         down_data, down_sr = ao.resample_proper(self.data, self.sr, target_sr)
    #         self.data, _ = ao.upsample_to(down_data, down_sr, self.sr)
    #         self._refresh_plot()
    #         self.sampling_status_var.set(
    #             f"Properly (anti-aliased) downsampled to {target_sr} Hz - "
    #             f"high frequencies above {target_sr / 2:.0f} Hz were cleanly removed, no aliasing distortion."
    #         )
    #     except Exception as e:
    #         messagebox.showerror("Sampling Error", str(e))

    def _refresh_enrolled_status(self):
        if not self.enrolled_speakers:
            self.enrolled_status_var.set("No enrolled voiceprints.")
        else:
            names = ", ".join(self.enrolled_speakers.keys())
            self.enrolled_status_var.set(f"Enrolled Voiceprints ({len(self.enrolled_speakers)}): {names}")

    def do_enroll_speaker(self):
        if not self._require_audio():
            return
        name = self.speaker_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Voice Registration", "Enter speaker label first.")
            return
        try:
            threshold = float(self.speaker_threshold.get())
            vec = ao.extract_mfcc(self.data, self.sr)

            # Check for an existing match before registering, so the same
            # person doesn't end up enrolled twice under different names.
            if self.enrolled_speakers:
                existing_name, dist = ao.identify_speaker(self.data, self.sr, self.enrolled_speakers, threshold=threshold)
                if existing_name != "Unknown":
                    self.speaker_result_var.set(
                        f"Already registered as '{existing_name}' (distance: {dist:.1f}) - not re-registering."
                    )
                    return

            self.enrolled_speakers[name] = vec
            ao.save_voiceprints(self._voiceprints_path, self.enrolled_speakers)
            self._refresh_enrolled_status()
            self.speaker_result_var.set(f"Registered new voiceprint for '{name}' (saved to disk).")
        except Exception as e:
            messagebox.showerror("Registration Error", str(e))

    def do_clear_speakers(self):
        self.enrolled_speakers.clear()
        ao.save_voiceprints(self._voiceprints_path, self.enrolled_speakers)
        self._refresh_enrolled_status()
        self.speaker_result_var.set("")

    def do_identify_speaker(self):
        if not self._require_audio():
            return
        if not self.enrolled_speakers:
            messagebox.showinfo("Speaker ID", "Enroll voiceprints first.")
            return
        try:
            threshold = float(self.speaker_threshold.get())
            name, dist = ao.identify_speaker(self.data, self.sr, self.enrolled_speakers, threshold=threshold)
            if name == "Unknown":
                self.speaker_result_var.set(f"None matched (Minimum distance: {dist:.1f}, Threshold limit: {threshold}).")
            else:
                self.speaker_result_var.set(f"Identified Match: '{name}' (Euclidean Vector Distance: {dist:.1f})")
        except Exception as e:
            messagebox.showerror("Voice Identification Error", str(e))


    def do_doppler_autodetect(self):
        if not self._require_audio():
            return
        try:
            f_approach, f_depart = ao.detect_doppler_shift(self.data, self.sr)
            self.doppler_approach.delete(0, tk.END)
            self.doppler_approach.insert(0, str(f_approach))
            self.doppler_depart.delete(0, tk.END)
            self.doppler_depart.insert(0, str(f_depart))
            self.doppler_result_var.set(
                f"Auto-detected: approaching {f_approach} Hz, departing {f_depart} Hz. Click Calculate."
            )
        except Exception as e:
            messagebox.showerror("Auto-Detect Failed", str(e))

    def do_doppler_autofill_approach(self):
        if self._last_detected_freq is None:
            messagebox.showinfo("Doppler Shift", "Run pitch detection on analysis panel first.")
            return
        self.doppler_approach.delete(0, tk.END)
        self.doppler_approach.insert(0, str(self._last_detected_freq))

    def do_doppler_autofill_depart(self):
        if self._last_detected_freq is None:
            messagebox.showinfo("Doppler Shift", "Run pitch detection on analysis panel first.")
            return
        self.doppler_depart.delete(0, tk.END)
        self.doppler_depart.insert(0, str(self._last_detected_freq))

    def do_doppler_calc(self):
        try:
            f_approach = float(self.doppler_approach.get())
            f_depart = float(self.doppler_depart.get())
            speed = ao.doppler_speed(f_approach, f_depart)
            self.doppler_result_var.set(
                f"Source speed: {speed:.2f} m/s ({speed * 3.6:.1f} km/h)"
            )
        except Exception as e:
            messagebox.showerror("Doppler Shift Error", str(e))

    def do_detect_bpm(self):
        if not self._require_audio():
            return
        try:
            band = (20, 150) if self.bpm_mode.get() == "Heartbeat" else None
            bpm = ao.compute_bpm(self.data, self.sr, band=band)
            if bpm is None:
                self.bpm_result_var.set("No repeating pulse envelope detected.")
            else:
                label = "pulses/min" if self.bpm_mode.get() == "Heartbeat" else "BPM"
                self.bpm_result_var.set(f"Estimated Track Tempo: {bpm} {label}")
        except Exception as e:
            messagebox.showerror("Tempo Detection Error", str(e))

    # ----------------------------------------------------- voice stress test -
    def do_analyze_stress(self):
        if not self._require_audio():
            return
        try:
            frame_ms = float(self.stress_frame_ms.get())
            hop_ms = float(self.stress_hop_ms.get())
            result = ao.analyze_voice_stress(self.data, self.sr, frame_ms=frame_ms, hop_ms=hop_ms)
            self._last_stress_result = result
            self.stress_result_var.set(
                f"Jitter: {result['jitter_percent']}%   |   Shimmer: {result['shimmer_percent']}%   |   "
                f"Stress score: {result['stress_score']}/100 - {result['stress_label']}"
            )
        except Exception as e:
            messagebox.showerror("Voice Stress Analysis Error", str(e))

    def do_plot_stress(self):
        if self._last_stress_result is None:
            messagebox.showinfo("Voice Stress Test", "Run 'Analyze Voice Stress' first.")
            return
        result = self._last_stress_result

        def plot(ax):
            ax.plot(result["times"], result["f0_track"], linewidth=1.2,
                    color=self._theme_colors["plot_line"], marker="o", markersize=2)
            ax.set_xlabel("Time (seconds)", fontfamily=FONT_FAMILY)
            ax.set_ylabel("Estimated Pitch (Hz)", fontfamily=FONT_FAMILY)
            ax.set_title(
                f"Pitch Stability Over Time  -  Jitter {result['jitter_percent']}%, "
                f"Shimmer {result['shimmer_percent']}% ({result['stress_label']})",
                fontfamily=FONT_FAMILY, fontweight="bold", fontsize=10,
            )

        self._show_plot_window("Voice Stress - Pitch Stability", plot)

    # ------------------------------------------------------ pulse-echo sonar -
    def do_generate_chirp(self):
        try:
            duration_sec = float(self.sonar_pulse_ms.get()) / 1000.0
            sr = self.sr if self.sr else 44100
            self.chirp_sr = sr
            self.chirp_data = ao.generate_chirp(sr=sr, duration_sec=duration_sec)
            self.sonar_status_var.set(
                f"Pulse generated: {duration_sec * 1000:.0f}ms chirp @ {sr} Hz. Ready to play/record."
            )
        except Exception as e:
            messagebox.showerror("Pulse Generation Error", str(e))

    def do_play_chirp(self):
        if not PLAYBACK_AVAILABLE:
            messagebox.showerror("Error", "PortAudio/sounddevice driver missing.")
            return
        if self.chirp_data is None:
            messagebox.showinfo("Pulse-Echo Ranging", "Generate a pulse first.")
            return
        sd.play(self.chirp_data, self.chirp_sr)

    def do_sonar_start_record(self):
        if not PLAYBACK_AVAILABLE:
            messagebox.showerror("Error", "PortAudio/sounddevice driver missing.")
            return
        if self.chirp_data is None:
            messagebox.showinfo("Pulse-Echo Ranging", "Generate a pulse first, so the recording uses a matching sample rate.")
            return
        try:
            self.sonar_recorder = ao.AudioRecorder(sample_rate=self.chirp_sr)
            self.sonar_recorder.start()
            self.sonar_status_var.set("Recording... click 'Play Pulse' now, then Stop once the echo has returned.")
            self.sonar_record_btn.config(state=tk.DISABLED)
            self.sonar_stop_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Microphone Error", str(e))

    def do_sonar_stop_record(self):
        if self.sonar_recorder is None:
            return
        sr, data = self.sonar_recorder.stop()
        self.sonar_recorded_sr, self.sonar_recorded_data = sr, data
        duration = len(data) / sr if len(data) > 0 else 0
        self.sonar_status_var.set(f"Captured {duration:.2f}s of audio. Click 'Measure Distance' to analyze it.")
        self.sonar_record_btn.config(state=tk.NORMAL)
        self.sonar_stop_btn.config(state=tk.DISABLED)

    def do_measure_echo(self):
        if self.chirp_data is None:
            messagebox.showinfo("Pulse-Echo Ranging", "Generate a pulse first.")
            return
        if self.sonar_recorded_data is None or len(self.sonar_recorded_data) == 0:
            messagebox.showinfo("Pulse-Echo Ranging", "Record a mic clip first (or run the synthetic self-test).")
            return
        try:
            result = ao.measure_echo_delay(self.chirp_data, self.sonar_recorded_data, self.sonar_recorded_sr)
            self._last_echo_result = result
            self.sonar_status_var.set(
                f"Echo delay: {result['delay_sec'] * 1000:.2f} ms  ->  Estimated distance: "
                f"{result['distance_m']:.2f} m (speed of sound = {ao.SPEED_OF_SOUND_M_S} m/s)"
            )
        except Exception as e:
            messagebox.showerror("Ranging Error", str(e))

    def do_plot_correlation(self):
        if self._last_echo_result is None:
            messagebox.showinfo("Pulse-Echo Ranging", "Run 'Measure Distance' (or the self-test) first.")
            return
        result = self._last_echo_result

        def plot(ax):
            ax.plot(result["lags_sec"] * 1000, result["correlation"], linewidth=0.9,
                    color=self._theme_colors["plot_line"])
            ax.axvline(result["direct_lag_sec"] * 1000, color="#22c55e", linestyle="--", linewidth=1.2,
                       label="Direct path")
            ax.axvline(result["echo_lag_sec"] * 1000, color="#ef4444", linestyle="--", linewidth=1.2,
                       label="Echo")
            ax.set_xlabel("Lag (milliseconds)", fontfamily=FONT_FAMILY)
            ax.set_ylabel("Cross-Correlation", fontfamily=FONT_FAMILY)
            ax.set_title(f"Pulse-Echo Correlation - Distance {result['distance_m']:.2f} m",
                        fontfamily=FONT_FAMILY, fontweight="bold", fontsize=10)
            ax.legend(fontsize=8)

        self._show_plot_window("Pulse-Echo Ranging - Correlation", plot)

    def do_sonar_self_test(self):
        if self.chirp_data is None:
            messagebox.showinfo("Pulse-Echo Ranging", "Generate a pulse first.")
            return
        try:
            true_distance = float(self.sonar_test_distance.get())
            sr = self.chirp_sr
            synthetic_received = ao.simulate_echo(self.chirp_data, sr, true_distance)
            result = ao.measure_echo_delay(self.chirp_data, synthetic_received, sr)
            self._last_echo_result = result
            error_m = abs(result["distance_m"] - true_distance)
            self.sonar_status_var.set(
                f"Self-test: simulated {true_distance:.2f} m  ->  measured {result['distance_m']:.2f} m "
                f"(error {error_m:.3f} m). Math verified - now try a real mic recording."
            )
        except Exception as e:
            messagebox.showerror("Self-Test Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEditorApp(root)
    root.mainloop()