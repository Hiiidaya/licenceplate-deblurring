"""
gui_dashboard.py  —  PlateReveal Studio
=========================================
Light-theme, animated, experience-first interface.

Controls
--------
  Circular dial knob     — angle (drag to rotate, ticks, pointer sweeps live)
  Visual stretch slider  — blur length, kernel heatmap grows as you drag
  Reveal slider          — before/after drag-to-compare
  Live kernel canvas     — PSF heatmap updates instantly on every param change
  Step cards             — pipeline steps appear one-by-one as they complete
  Typewriter OCR         — characters type in, coloured green/red vs GT
  Count-up metrics       — numbers animate to their final value

Launch
------
    python gui_dashboard.py
    python gui_dashboard.py --image input/img1.png
"""

import sys, os, math, threading, queue, time, datetime, argparse
from pathlib import Path
from io import BytesIO

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageDraw, ImageFilter

HERE     = Path(__file__).parent.resolve()
PIPELINE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPELINE))

INPUT_DIR   = HERE / "input"
OUTPUT_DIR  = HERE / "output"
NOT_BLURRED = PIPELINE / "BEST_TEST_IM" / "not_blurred"

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#f8f7ff"
CARD    = "#ffffff"
SURFACE = "#f3f0ff"
ACCENT  = "#7c3aed"
A_DARK  = "#6d28d9"
A_LIGHT = "#ede9fe"
ROSE    = "#ec4899"
GREEN   = "#10b981"
AMBER   = "#f59e0b"
RED     = "#ef4444"
INK     = "#1e1b4b"
GRAY    = "#6b7280"
LGRAY   = "#d1d5db"
BORDER  = "#e5e7eb"

# ── Known ground-truth plates for predefined examples ─────────────────────────
# Keys are substrings matched against the image filename (case-insensitive).
# Add your own plate/filename pairs here — the sidebar auto-fills on image select.
_KNOWN_GT = {
    "img1": "GJW115A1138", "img2": "TN52U1580",  "img3": "WB06F9209",
    "img4": "DL10CG4693",  "img5": "TN45BA1065", "img6": "MH14EP4660",
    "img7": "TN21AU7234",  "img8": "HR26DA0471",
}

def _gt_for_path(path: str) -> str:
    """Return known GT plate text for a given image path, or ''."""
    stem = Path(path).stem.lower()
    for k, v in _KNOWN_GT.items():
        if k.lower() in stem:
            return v
    return ""

# ── Fonts ──────────────────────────────────────────────────────────────────────
FH1   = ("Segoe UI", 15, "bold")
FH2   = ("Segoe UI", 12, "bold")
FH3   = ("Segoe UI", 10, "bold")
FBODY = ("Segoe UI", 10)
FSM   = ("Segoe UI", 9)
FTINY = ("Segoe UI", 8)
FMONO = ("Consolas", 10)
FMONO_SM = ("Consolas", 9)


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _lerp_color(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
    r2, g2, b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CUSTOM WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class Spinner(tk.Canvas):
    """Rotating arc loading spinner."""
    def __init__(self, parent, size=38, color=ACCENT, bg=CARD, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=bg, highlightthickness=0, **kw)
        self._s   = size
        self._col = color
        self._ang = 0
        self._job = None

    def start(self):
        if not self._job:
            self._spin()

    def stop(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        self.delete("all")

    def _spin(self):
        s = self._s; p = 5
        self.delete("all")
        self.create_arc(p, p, s-p, s-p,
                        start=self._ang, extent=260,
                        outline=self._col, width=3, style="arc")
        self._ang = (self._ang + 14) % 360
        self._job = self.after(35, self._spin)


class DialKnob(tk.Canvas):
    """
    Circular angle selector 0-180°.
    Drag anywhere to set angle. Callback called on every change.
    """
    SZ = 164

    def __init__(self, parent, value=0, callback=None, **kw):
        s = self.SZ
        super().__init__(parent, width=s, height=s,
                         bg=BG, highlightthickness=0, **kw)
        self.value    = float(value) % 180
        self.callback = callback
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self._draw()

    def set(self, v):
        self.value = float(v) % 180
        self._draw()

    def _angle_from_event(self, e):
        cx = cy = self.SZ / 2
        a = math.degrees(math.atan2(-(e.y - cy), e.x - cx)) % 180
        return a

    def _press(self, e):  self.value = self._angle_from_event(e); self._draw(); self._fire()
    def _drag(self, e):   self.value = self._angle_from_event(e); self._draw(); self._fire()
    def _release(self, e): pass
    def _fire(self):
        if self.callback: self.callback(self.value)

    def _draw(self):
        s = self.SZ; cx = cy = s / 2; r = s/2 - 10
        self.delete("all")

        # Outer track
        self.create_oval(10, 10, s-10, s-10, outline=BORDER, width=2, fill=CARD)

        # Filled arc (shows how far from 0 the angle is)
        self.create_arc(10, 10, s-10, s-10,
                        start=180, extent=-self.value,
                        fill=A_LIGHT, outline="", style="pie")
        self.create_oval(10, 10, s-10, s-10, outline=ACCENT, width=2, fill="")

        # Degree tick marks
        for deg in range(0, 181, 15):
            a = math.radians(180 - deg)
            long_tick = deg % 45 == 0
            r1 = r if long_tick else r + 3
            r2 = r - (9 if long_tick else 5)
            x1, y1 = cx + r1*math.cos(a), cy - r1*math.sin(a)
            x2, y2 = cx + r2*math.cos(a), cy - r2*math.sin(a)
            col = ACCENT if abs(deg - self.value) < 8 else LGRAY
            self.create_line(x1, y1, x2, y2, fill=col,
                             width=2 if long_tick else 1)
            if long_tick:
                lx = cx + (r2-10)*math.cos(a)
                ly = cy - (r2-10)*math.sin(a)
                self.create_text(lx, ly, text=f"{deg}",
                                 fill=GRAY, font=("Segoe UI", 7))

        # Pointer
        a = math.radians(180 - self.value)
        ri = r - 18
        self.create_line(cx, cy,
                         cx + ri*math.cos(a), cy - ri*math.sin(a),
                         fill=ACCENT, width=3, capstyle="round")

        # Blur direction arrow head at pointer tip
        tip_x = cx + ri*math.cos(a); tip_y = cy - ri*math.sin(a)
        self.create_oval(tip_x-5, tip_y-5, tip_x+5, tip_y+5,
                         fill=ACCENT, outline="")

        # Centre hub
        self.create_oval(cx-6, cy-6, cx+6, cy+6, fill=ACCENT, outline=CARD, width=2)

        # Value label
        self.create_text(cx, cy+26, text=f"{self.value:.1f}°",
                         fill=INK, font=FH2)
        self.create_text(cx, s-8, text="BLUR ANGLE",
                         fill=GRAY, font=("Segoe UI", 7))


class KernelCanvas(tk.Canvas):
    """Live PSF kernel heatmap — updates instantly on param change."""
    SZ = 130

    def __init__(self, parent, **kw):
        s = self.SZ
        super().__init__(parent, width=s, height=s, bg=CARD,
                         highlightthickness=1, highlightbackground=BORDER, **kw)
        self._photo = None
        self._empty()

    def _empty(self):
        self.delete("all")
        self.create_text(self.SZ//2, self.SZ//2,
                         text="kernel\npreview", fill=LGRAY,
                         font=FSM, justify="center")

    def update(self, length: int, angle: float):
        try:
            from deblurring_input import make_motion_kernel
        except ImportError:
            return
        k = make_motion_kernel(max(2, int(length)), float(angle))
        k = k / (k.max() + 1e-10)
        # Render with 'hot' colormap
        rgba = cm.hot(k)[:, :, :3]
        arr  = (rgba * 255).astype(np.uint8)
        img  = Image.fromarray(arr).resize((self.SZ, self.SZ), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)
        # Direction overlay
        s = self.SZ; cx = cy = s//2
        a = math.radians(angle)
        hl = min(s//2-6, int(length * s/60))
        self.create_line(cx - hl*math.cos(a), cy + hl*math.sin(a),
                         cx + hl*math.cos(a), cy - hl*math.sin(a),
                         fill="#00ffcc", width=1, dash=(3, 2))
        self.create_text(cx, s-9, text=f"{angle:.0f}° / {length}px",
                         fill="white", font=FTINY)


class BlurSlider(tk.Frame):
    """Custom slider: drag track, numeric display, live callback."""
    def __init__(self, parent, label, from_, to_, value,
                 callback=None, width=220, integer=True, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._from    = float(from_)
        self._to      = float(to_)
        self.value    = float(value)
        self.callback = callback
        self.integer  = integer
        self._W       = width

        tk.Label(self, text=label.upper(), bg=BG, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x")
        self._c = tk.Canvas(row, width=width, height=30,
                            bg=BG, highlightthickness=0)
        self._c.pack(side="left")
        self._vl = tk.Label(row, text=self._fmt(self.value),
                            bg=BG, fg=INK, font=FH3, width=6)
        self._vl.pack(side="left", padx=(6,0))
        self._c.bind("<ButtonPress-1>", self._hit)
        self._c.bind("<B1-Motion>",     self._hit)
        self._draw()

    def set(self, v):
        self.value = float(max(self._from, min(self._to, v)))
        self._vl.config(text=self._fmt(self.value))
        self._draw()

    def _fmt(self, v): return str(int(v)) if self.integer else f"{v:.3f}"

    def _px(self):
        frac = (self.value - self._from) / max(self._to - self._from, 1)
        return 12 + int(frac * (self._W - 24))

    def _hit(self, e):
        frac = (e.x - 12) / max(self._W - 24, 1)
        v = self._from + frac * (self._to - self._from)
        self.value = max(self._from, min(self._to, v))
        if self.integer: self.value = round(self.value)
        self._vl.config(text=self._fmt(self.value))
        self._draw()
        if self.callback: self.callback(self.value)

    def _draw(self):
        c = self._c; W = self._W; py = 15
        c.delete("all")
        c.create_line(12, py, W-12, py, fill=LGRAY, width=4, capstyle="round")
        px = self._px()
        c.create_line(12, py, px, py, fill=ACCENT, width=4, capstyle="round")
        c.create_oval(px-10, py-10, px+10, py+10,
                      fill=ACCENT, outline=CARD, width=2)


class RevealSlider(tk.Canvas):
    """Before/after drag-to-compare. Drag the vertical divider."""
    def __init__(self, parent, width=420, height=200, **kw):
        super().__init__(parent, width=width, height=height, bg=SURFACE,
                         highlightthickness=1, highlightbackground=BORDER, **kw)
        self._W = width; self._H = height
        self._split  = width // 2
        self._before = None; self._after = None
        self._photo  = None
        self.bind("<ButtonPress-1>", self._move)
        self.bind("<B1-Motion>",     self._move)
        self._draw_empty()

    def load(self, before_arr: np.ndarray, after_arr: np.ndarray):
        def _fit(arr):
            img = Image.fromarray(arr.astype(np.uint8))
            img.thumbnail((self._W, self._H), Image.LANCZOS)
            canvas = Image.new("RGB", (self._W, self._H), (243, 240, 255))
            ox = (self._W - img.width)//2; oy = (self._H - img.height)//2
            canvas.paste(img, (ox, oy))
            return np.array(canvas)
        self._before = _fit(before_arr)
        self._after  = _fit(after_arr)
        self._redraw()

    def _move(self, e):
        self._split = max(10, min(self._W-10, e.x))
        self._redraw()

    def _draw_empty(self):
        self.delete("all")
        self.create_text(self._W//2, self._H//2,
                         text="Run pipeline →\nthen drag to compare",
                         fill=LGRAY, font=FBODY, justify="center")

    def _redraw(self):
        if self._before is None: return
        s = self._split
        comp = np.zeros((self._H, self._W, 3), dtype=np.uint8)
        comp[:, :s] = self._before[:, :s]
        comp[:, s:] = self._after[:, s:]
        self._photo = ImageTk.PhotoImage(Image.fromarray(comp))
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)
        # Labels
        if s > 60:
            self.create_text(s//2, 14, text="BLURRED",
                             fill="white", font=("Segoe UI", 8, "bold"))
        if self._W - s > 70:
            self.create_text(s + (self._W-s)//2, 14, text="DEBLURRED",
                             fill="white", font=("Segoe UI", 8, "bold"))
        # Divider
        self.create_line(s, 0, s, self._H, fill="white", width=2)
        # Handle
        cy = self._H // 2; r = 17
        self.create_oval(s-r, cy-r, s+r, cy+r,
                         fill="white", outline=ACCENT, width=2)
        self.create_text(s, cy, text="◄►", fill=ACCENT,
                         font=("Segoe UI", 10, "bold"))


class StepCard(tk.Frame):
    """One pipeline step: idle → spinning → done (image fades in)."""
    CW, CH = 175, 138

    def __init__(self, parent, label, step_id, on_zoom=None, **kw):
        super().__init__(parent, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        self._id     = step_id
        self._photo  = None
        self._arr    = None
        self._spinner = None
        self._zoom_cb = on_zoom

        # ── Header ──
        hdr = tk.Frame(self, bg=SURFACE, height=26)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        self._dot = tk.Label(hdr, text="●", bg=SURFACE, fg=LGRAY, font=("Segoe UI",8))
        self._dot.pack(side="left", padx=6)
        tk.Label(hdr, text=label, bg=SURFACE, fg=INK,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self._step_lbl = tk.Label(hdr, text="", bg=SURFACE, fg=GRAY, font=FTINY)
        self._step_lbl.pack(side="right", padx=6)

        # ── Image area ──
        self._area = tk.Frame(self, bg="#f0eff8",
                              width=self.CW, height=self.CH)
        self._area.pack(fill="both", expand=True, padx=2, pady=2)
        self._area.pack_propagate(False)

        self._ph_lbl = tk.Label(self._area, text="waiting…",
                                 bg="#f0eff8", fg=LGRAY, font=FSM)
        self._ph_lbl.place(relx=.5, rely=.5, anchor="center")

        self._img_lbl = tk.Label(self._area, bg="#f0eff8", cursor="hand2")

        for w in (self, self._area, self._img_lbl):
            w.bind("<Button-1>", self._click)

    def set_idle(self):
        self._dot.config(fg=LGRAY)
        self._ph_lbl.config(text="waiting…", fg=LGRAY)
        self._ph_lbl.place(relx=.5, rely=.5, anchor="center")
        self._img_lbl.place_forget()
        if self._spinner: self._spinner.stop(); self._spinner.place_forget()
        self._step_lbl.config(text="")

    def set_running(self, label=""):
        self._dot.config(fg=AMBER)
        self._ph_lbl.place_forget()
        self._img_lbl.place_forget()
        self._step_lbl.config(text=label)
        if not self._spinner:
            self._spinner = Spinner(self._area, size=36, bg="#f0eff8")
        self._spinner.place(relx=.5, rely=.5, anchor="center")
        self._spinner.start()

    def set_done(self, arr: np.ndarray, label="✓"):
        self._dot.config(fg=GREEN)
        self._step_lbl.config(text=label, fg=GREEN)
        if self._spinner: self._spinner.stop(); self._spinner.place_forget()
        self._ph_lbl.place_forget()
        self._arr = arr
        img = Image.fromarray(arr.astype(np.uint8))
        img.thumbnail((self.CW-8, self.CH-8), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._img_lbl.config(image=self._photo)
        self._img_lbl.place(relx=.5, rely=.5, anchor="center")
        # Subtle fade-in via background colour lerp
        self._fade_in(0)

    def _fade_in(self, step):
        if step > 8: return
        t = step / 8
        col = _lerp_color("#f0eff8", CARD, t)
        self._area.config(bg=col)
        self._img_lbl.config(bg=col)
        self.after(25, self._fade_in, step+1)

    def _click(self, _):
        if self._zoom_cb and self._arr is not None:
            self._zoom_cb(self._id, self._arr)


class OCRDisplay(tk.Frame):
    """Characters animate in one-by-one. Green=correct, Red=wrong, Violet=unknown."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, **kw)
        self._boxes = []

    def clear(self):
        for b in self._boxes: b.destroy()
        self._boxes = []

    def show(self, text: str, gt: str = None, ms: int = 65):
        self.clear()
        if not text:
            lbl = tk.Label(self, text="(no result)", bg=CARD, fg=LGRAY, font=FMONO)
            lbl.pack(side="left")
            self._boxes.append(lbl)
            return
        for i, ch in enumerate(text):
            correct = None
            if gt:
                if i < len(gt): correct = (ch == gt[i])
            self.after(i * ms, self._add, ch, correct)

    def _add(self, ch, correct):
        if correct is True:   bg, fg = GREEN, "white"
        elif correct is False: bg, fg = RED, "white"
        else:                  bg, fg = A_LIGHT, ACCENT
        box = tk.Frame(self, bg=bg, padx=5, pady=3,
                       highlightbackground=BORDER, highlightthickness=1)
        box.pack(side="left", padx=1)
        tk.Label(box, text=ch, bg=bg, fg=fg, font=FMONO).pack()
        self._boxes.append(box)


class MetricBadge(tk.Frame):
    """Animated metric: value counts up from 0 on .set()."""
    def __init__(self, parent, title, unit="", color=INK, **kw):
        super().__init__(parent, bg=CARD, padx=14, pady=10,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        tk.Label(self, text=title.upper(), bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack()
        self._v = tk.Label(self, text="—", bg=CARD, fg=color,
                           font=("Segoe UI", 20, "bold"))
        self._v.pack()
        self._u = tk.Label(self, text=unit, bg=CARD, fg=GRAY, font=FSM)
        self._u.pack()

    def reset(self): self._v.config(text="—", fg=GRAY)

    def set(self, value, unit=None, color=None):
        if unit:  self._u.config(text=unit)
        if color: self._v.config(fg=color)
        if value is None:
            self._v.config(text="N/A", fg=LGRAY); return
        self._v.config(fg=color or INK)
        self._count(0.0, float(value), 22)

    def _count(self, cur, tgt, steps):
        if steps <= 0:
            self._v.config(text=f"{tgt:.2f}"); return
        cur += (tgt - cur) * 0.28
        self._v.config(text=f"{cur:.2f}")
        self.after(28, self._count, cur, tgt, steps-1)


class PulseButton(tk.Canvas):
    """Big action button with pulsing outer ring when ready."""
    W, H = 220, 52

    def __init__(self, parent, text, command=None, **kw):
        super().__init__(parent, width=self.W, height=self.H,
                         bg=BG, highlightthickness=0, **kw)
        self._text    = text
        self._command = command
        self._pulse_r = 0
        self._pulse_job = None
        self._hover   = False
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self._draw(ACCENT)

    def start_pulse(self):
        self._pulse_loop()

    def stop_pulse(self):
        if self._pulse_job: self.after_cancel(self._pulse_job); self._pulse_job = None
        self._draw(ACCENT)

    def _pulse_loop(self):
        self._pulse_r = (getattr(self, "_pulse_r", 0) + 1) % 24
        t = self._pulse_r / 24
        # Outer ring fades from ACCENT toward BG
        ring = _lerp_color(ACCENT, BG, t)
        self._draw(A_DARK if self._hover else ACCENT, ring_color=ring,
                   ring_r=6 + int(t * 14))
        self._pulse_job = self.after(45, self._pulse_loop)

    def _draw(self, fill, ring_color=None, ring_r=0):
        self.delete("all")
        W, H = self.W, self.H
        r = 10  # corner radius
        # Outer pulse ring
        if ring_color and ring_r > 0:
            self.create_oval(W//2-ring_r-W//2+r, H//2-ring_r-H//2+r,
                             W//2+ring_r+W//2-r, H//2+ring_r+H//2-r,
                             outline=ring_color, width=2, fill="")
        # Rounded rect (approx with oval + rectangles)
        self.create_rectangle(r, 0, W-r, H, fill=fill, outline="")
        self.create_rectangle(0, r, W, H-r, fill=fill, outline="")
        self.create_oval(0, 0, 2*r, 2*r, fill=fill, outline="")
        self.create_oval(W-2*r, 0, W, 2*r, fill=fill, outline="")
        self.create_oval(0, H-2*r, 2*r, H, fill=fill, outline="")
        self.create_oval(W-2*r, H-2*r, W, H, fill=fill, outline="")
        self.create_text(W//2, H//2, text=self._text,
                         fill="white", font=FH2)

    def _on_enter(self, _): self._hover = True;  self._draw(A_DARK)
    def _on_leave(self, _): self._hover = False; self._draw(ACCENT)
    def _on_click(self, _):
        if self._command: self._command()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

class SideBar(tk.Frame):
    def __init__(self, parent, on_run, **kw):
        super().__init__(parent, bg=CARD, width=270,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        self.pack_propagate(False)
        self._on_run = on_run
        self._img_path = None
        self._photo    = None
        self._mode     = tk.StringVar(value="blurred")  # or "unblurred"
        self._build()

    def _build(self):
        # ── Logo ──────────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=ACCENT, height=52)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="◉  PlateReveal",
                 bg=ACCENT, fg="white", font=FH1).pack(side="left", padx=14)

        pad = tk.Frame(self, bg=CARD)
        pad.pack(fill="both", expand=True, padx=12, pady=8)

        # ── Drop zone / thumbnail ─────────────────────────────────────────────
        # ── Quick example picker ──────────────────────────────────────────────
        tk.Label(pad, text="QUICK SELECT", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 2))
        self._example_row = tk.Frame(pad, bg=CARD)
        self._example_row.pack(fill="x", pady=(0, 4))
        self.after(200, self._populate_examples)   # populate after window exists

        tk.Frame(pad, bg=BORDER, height=1).pack(fill="x", pady=(0, 6))

        zone_lbl = tk.Label(pad, text="OR BROWSE", bg=CARD, fg=GRAY,
                            font=("Segoe UI", 8, "bold"))
        zone_lbl.pack(anchor="w", pady=(0, 2))

        self._zone = tk.Label(pad, text="click to browse\nor drag & drop",
                              bg=SURFACE, fg=LGRAY, font=FSM,
                              width=26, height=7, relief="flat",
                              cursor="hand2", justify="center")
        self._zone.pack(fill="x", pady=(0,4))
        self._zone.bind("<Button-1>", lambda _: self._browse())

        self._img_name = tk.Label(pad, text="No image selected",
                                  bg=CARD, fg=GRAY, font=FSM,
                                  wraplength=220, justify="left")
        self._img_name.pack(anchor="w", pady=(0,8))

        tk.Frame(pad, bg=BORDER, height=1).pack(fill="x", pady=4)

        # ── Mode selector ─────────────────────────────────────────────────────
        tk.Label(pad, text="MODE", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4,2))

        for val, lbl in (("blurred",   "Already blurred  →  auto-estimate kernel"),
                         ("unblurred", "Sharp image  →  apply known blur then deblur")):
            rb = tk.Radiobutton(pad, text=lbl, variable=self._mode,
                                value=val, bg=CARD, fg=INK, font=FSM,
                                activebackground=CARD, selectcolor=CARD,
                                command=self._mode_changed)
            rb.pack(anchor="w")

        # ── Unblurred-mode angle/length ───────────────────────────────────────
        self._unblur_frame = tk.Frame(pad, bg=SURFACE,
                                       highlightbackground=BORDER, highlightthickness=1)

        inner = tk.Frame(self._unblur_frame, bg=SURFACE)
        inner.pack(padx=8, pady=6, fill="x")

        tk.Label(inner, text="Angle to apply (°)", bg=SURFACE, fg=INK,
                 font=FSM).grid(row=0, column=0, sticky="w")
        self._ub_angle = tk.Spinbox(inner, from_=0, to=179, width=6,
                                     font=FSM, bg=CARD, fg=INK)
        self._ub_angle.grid(row=0, column=1, padx=6, pady=2)

        tk.Label(inner, text="Length to apply (px)", bg=SURFACE, fg=INK,
                 font=FSM).grid(row=1, column=0, sticky="w")
        self._ub_length = tk.Spinbox(inner, from_=3, to=60, width=6,
                                      font=FSM, bg=CARD, fg=INK)
        self._ub_length.grid(row=1, column=1, padx=6, pady=2)
        try: self._ub_angle.delete(0,"end"); self._ub_angle.insert(0,"45")
        except: pass
        try: self._ub_length.delete(0,"end"); self._ub_length.insert(0,"20")
        except: pass

        tk.Frame(pad, bg=BORDER, height=1).pack(fill="x", pady=(10,4))

        # ── Quick params ──────────────────────────────────────────────────────
        tk.Label(pad, text="QUICK PARAMS", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4,4))

        qp = tk.Frame(pad, bg=CARD)
        qp.pack(fill="x")

        def _stepper(parent, label, default, lo, hi):
            f = tk.Frame(parent, bg=CARD)
            tk.Label(f, text=label, bg=CARD, fg=INK, font=FSM, width=10,
                     anchor="w").pack(side="left")
            var = tk.IntVar(value=default)
            tk.Button(f, text="−", bg=SURFACE, fg=ACCENT, relief="flat",
                      font=FH3, width=2,
                      command=lambda: var.set(max(lo, var.get()-1))).pack(side="left")
            tk.Label(f, textvariable=var, bg=CARD, fg=INK,
                     font=FH3, width=3).pack(side="left")
            tk.Button(f, text="+", bg=SURFACE, fg=ACCENT, relief="flat",
                      font=FH3, width=2,
                      command=lambda: var.set(min(hi, var.get()+1))).pack(side="left")
            f.pack(fill="x", pady=1)
            return var

        self._n_outer = _stepper(qp, "Outer iter",  8,  1, 20)
        self._n_tv    = _stepper(qp, "TV iter",    40,  5, 100)

        tk.Frame(pad, bg=BORDER, height=1).pack(fill="x", pady=(8,10))

        # ── License plate text (GT for OCR comparison) ────────────────────────
        tk.Label(pad, text="LICENSE PLATE TEXT", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(8,2))
        plate_row = tk.Frame(pad, bg=CARD)
        plate_row.pack(fill="x", pady=(0,2))
        self._plate_var = tk.StringVar()
        self._plate_entry = tk.Entry(plate_row, textvariable=self._plate_var,
                                      font=("Consolas", 11, "bold"),
                                      bg=SURFACE, fg=INK, insertbackground=ACCENT,
                                      relief="flat", width=12)
        self._plate_entry.pack(side="left", ipady=4, padx=(0,4))
        tk.Button(plate_row, text="Auto", bg=A_LIGHT, fg=ACCENT, relief="flat",
                  font=FTINY, cursor="hand2",
                  command=self._autofill_plate).pack(side="left")
        self._plate_hint = tk.Label(pad, text="Enter the license plate (e.g. AB123CD)",
                                     bg=CARD, fg=LGRAY, font=FTINY)
        self._plate_hint.pack(anchor="w", pady=(0,6))

        tk.Frame(pad, bg=BORDER, height=1).pack(fill="x", pady=(0,6))

        # ── GT reference image (optional, for PSNR/SSIM) ──────────────────────
        gt_row = tk.Frame(pad, bg=CARD)
        gt_row.pack(fill="x", pady=(0,4))
        tk.Label(gt_row, text="GT image (opt.):", bg=CARD, fg=GRAY,
                 font=FSM).pack(side="left")
        self._gt_btn = tk.Button(gt_row, text="Browse", bg=SURFACE,
                                  fg=ACCENT, relief="flat", font=FSM, cursor="hand2",
                                  command=self._browse_gt)
        self._gt_btn.pack(side="right")
        self._gt_path = None
        self._gt_lbl = tk.Label(pad, text="None", bg=CARD, fg=LGRAY,
                                 font=FTINY, wraplength=220)
        self._gt_lbl.pack(anchor="w", pady=(0,8))

        # ── REVEAL button ─────────────────────────────────────────────────────
        self._btn = PulseButton(pad, "⚡  REVEAL", command=self._run)
        self._btn.pack(pady=(0,4))
        self._btn.start_pulse()

        # Status text
        self._status = tk.Label(pad, text="Ready", bg=CARD, fg=GRAY, font=FSM)
        self._status.pack()

    def _populate_examples(self):
        """Fill the quick-select strip with thumbnails from input dirs."""
        candidates = []
        for d in (INPUT_DIR, NOT_BLURRED,
                  HERE / "input", PIPELINE / "BEST_TEST_IM" / "not_blurred"):
            if d and d.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    candidates += list(d.glob(ext))
        # deduplicate, take up to 8
        seen = set()
        unique = []
        for p in candidates:
            if str(p) not in seen:
                seen.add(str(p)); unique.append(p)
            if len(unique) >= 8:
                break

        for widget in self._example_row.winfo_children():
            widget.destroy()

        if not unique:
            tk.Label(self._example_row, text="No examples found in input/ folders",
                     bg=CARD, fg=LGRAY, font=FTINY).pack(side="left")
            return

        for img_path in unique:
            try:
                thumb = Image.open(img_path).convert("RGB")
                thumb.thumbnail((52, 32), Image.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
            except Exception:
                continue
            frame = tk.Frame(self._example_row, bg=SURFACE, cursor="hand2",
                             highlightbackground=BORDER, highlightthickness=1)
            frame.pack(side="left", padx=2, pady=1)
            lbl = tk.Label(frame, image=photo, bg=SURFACE, cursor="hand2")
            lbl.image = photo   # keep reference
            lbl.pack()
            name_short = img_path.stem[:8]
            tk.Label(frame, text=name_short, bg=SURFACE, fg=GRAY,
                     font=FTINY).pack()
            path_str = str(img_path)
            lbl.bind("<Button-1>", lambda e, p=path_str: self._set_image(p))
            frame.bind("<Button-1>", lambda e, p=path_str: self._set_image(p))

    def _mode_changed(self):
        if self._mode.get() == "unblurred":
            self._unblur_frame.pack(fill="x", pady=(4,0))
        else:
            self._unblur_frame.pack_forget()

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All", "*.*")],
            initialdir=str(INPUT_DIR) if INPUT_DIR.exists() else str(HERE))
        if p: self._set_image(p)

    def _browse_gt(self):
        p = filedialog.askopenfilename(
            title="Select ground-truth sharp image",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")],
            initialdir=str(NOT_BLURRED) if NOT_BLURRED.exists() else str(HERE))
        if p:
            self._gt_path = p
            self._gt_lbl.config(text=Path(p).name, fg=GREEN)

    def set_image(self, path: str):
        self._set_image(path)

    def _autofill_plate(self):
        if self._img_path:
            gt = _gt_for_path(self._img_path)
            if gt:
                self._plate_var.set(gt)
                self._plate_entry.config(state="readonly", bg=A_LIGHT)
                self._plate_hint.config(text=f"Auto-filled  ·  known example image", fg=GREEN)
            else:
                self._plate_hint.config(
                    text="No known plate for this image — type it above.", fg=ROSE)

    def _set_image(self, path: str):
        self._img_path = path
        name = Path(path).name
        self._img_name.config(text=name, fg=INK)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((220, 120), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self._zone.config(image=self._photo, text="", bg=CARD)
        except Exception:
            self._zone.config(image="", text=name, bg=SURFACE)
        # Auto-fill plate if known; lock entry for example images, unlock for custom
        gt = _gt_for_path(path)
        if gt:
            self._plate_var.set(gt)
            self._plate_entry.config(state="readonly", bg=A_LIGHT,
                                      disabledforeground=ACCENT)
            self._plate_hint.config(text="Auto-filled  ·  known example image", fg=GREEN)
        else:
            self._plate_var.set("")
            self._plate_entry.config(state="normal", bg=SURFACE)
            self._plate_hint.config(
                text="Enter the license plate (e.g. AB123CD)", fg=LGRAY)

    def _run(self):
        if not self._img_path:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        params = {
            "image_path":  self._img_path,
            "gt_path":     self._gt_path,
            "gt_text":     self._plate_var.get().strip().upper() or None,
            "synthetic":   self._mode.get() == "unblurred",
            "angle_hint":  float(self._ub_angle.get()) if self._mode.get()=="unblurred" else None,
            "length_hint": int(self._ub_length.get())  if self._mode.get()=="unblurred" else None,
            "n_outer":     self._n_outer.get(),
            "n_tv":        self._n_tv.get(),
        }
        self._on_run(params)

    def set_status(self, msg: str, color=GRAY):
        self._status.config(text=msg, fg=color)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2B — CINEMA CANVAS (animated step visualiser)
# ══════════════════════════════════════════════════════════════════════════════

class CinemaCanvas(tk.Canvas):
    """Large animated display area for the currently active pipeline step."""
    W, H = 520, 220

    def __init__(self, parent, **kw):
        super().__init__(parent, width=self.W, height=self.H,
                         bg="#0a0010", highlightthickness=1,
                         highlightbackground=BORDER, **kw)
        self._anim_id  = None
        self._photo    = None
        self._mode     = None
        self._t        = 0
        # Per-mode stored data
        self._arr      = None
        self._cep_angle  = 0
        self._cep_length = 0
        self._ker_angle  = 0
        self._ker_length = 0
        self._kernel     = None
        self._hqs_n      = 0
        self._hqs_tv     = 0
        self._hqs_start  = 0.0
        self._draw_placeholder()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _stop(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    def _draw_placeholder(self):
        self.delete("all")
        self.create_text(self.W // 2, self.H // 2 - 12,
                         text="No step selected", fill="#333", font=FH2)
        self.create_text(self.W // 2, self.H // 2 + 14,
                         text="Click any step above after running the pipeline",
                         fill="#222", font=FTINY)

    def _render_array(self, arr: np.ndarray) -> "ImageTk.PhotoImage | None":
        if arr is None: return None
        h, w = arr.shape[:2]
        scale = min(self.W / max(w, 1), (self.H - 34) / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        x0 = (self.W - nw) // 2
        y0 = (self.H - 34 - nh) // 2
        self.create_image(x0, y0, anchor="nw", image=self._photo)
        return self._photo

    def _caption(self, text: str, color="#aaaaaa"):
        self.create_text(self.W // 2, self.H - 12, text=text,
                         fill=color, font=FSM, tags="cap")

    # ── Public API ────────────────────────────────────────────────────────────

    def show_image(self, arr: np.ndarray, caption=""):
        self._stop(); self._mode = "image"; self._arr = arr; self._t = 0
        self._animate_image(caption)

    def _animate_image(self, caption):
        self.delete("all")
        if self._arr is not None:
            # Gentle brightness pulse
            fac = 0.92 + 0.08 * math.sin(self._t * 0.06)
            boosted = np.clip(self._arr * fac, 0, 255).astype(np.uint8)
            self._render_array(boosted)
        self._caption(caption, "#aaaaaa")
        self._t += 1
        self._anim_id = self.after(60, lambda: self._animate_image(caption))

    def show_fft(self, fft_arr: np.ndarray):
        self._stop(); self._mode = "fft"; self._arr = fft_arr; self._t = 0
        self._animate_fft()

    def _animate_fft(self):
        self.delete("all")
        if self._arr is not None:
            t = self._t
            # Hue-rotate the FFT image to show "frequency energy flowing"
            fac = 0.75 + 0.25 * abs(math.sin(t * 0.08))
            boosted = np.clip(self._arr.astype(float) * fac, 0, 255).astype(np.uint8)
            self._render_array(boosted)
            # Overlay radial "frequency lines" pulsing outward
            cx, cy = self.W // 2, (self.H - 34) // 2
            for i in range(6):
                ang = math.radians(i * 30 + t * 2)
                r = 30 + 20 * abs(math.sin(t * 0.1 + i))
                x2 = cx + r * math.cos(ang)
                y2 = cy + r * math.sin(ang)
                alpha = int(180 * abs(math.sin(t * 0.1 + i)))
                color = f"#{alpha:02x}{alpha//2:02x}ff"
                self.create_line(cx, cy, x2, y2, fill=color, width=1)
        self._caption("🔵  FFT Frequency Domain — peaks reveal motion blur direction", "#8888ff")
        self._t += 1
        self._anim_id = self.after(45, self._animate_fft)

    def show_cepstrum(self, cep_arr, angle: float, length: int):
        self._stop(); self._mode = "cepstrum"
        self._arr = cep_arr; self._cep_angle = angle; self._cep_length = length
        self._t = 0
        self._animate_cepstrum()

    def _animate_cepstrum(self):
        self.delete("all")
        self.configure(bg="#020008")
        t = self._t
        W, H = self.W, self.H
        content_h = H - 34

        if self._arr is not None:
            self._render_array(self._arr)
        else:
            # No cepstrum image available (length was hint-provided) — synthesize one
            # Draw a radial heatmap that looks like a real cepstrum
            cx2, cy2 = (W - 120) // 2, content_h // 2
            L = self._cep_length or 20
            ang_rad = math.radians(self._cep_angle or 0)
            # Background gradient rings
            for ring in range(60, 0, -4):
                bright = max(0, int(80 - ring * 1.1) + int(20 * abs(math.sin(t * 0.03))))
                self.create_oval(cx2 - ring, cy2 - ring, cx2 + ring, cy2 + ring,
                                  outline=f"#{bright:02x}{bright:02x}{bright:02x}", width=1)
            # Two peaks along blur direction (cepstrum shows symmetric peaks at ±L)
            pulse = 0.6 + 0.4 * abs(math.sin(t * 0.12))
            for sign in (+1, -1):
                px = cx2 + int(sign * L * math.cos(ang_rad) * 2.2)
                py = cy2 - int(sign * L * math.sin(ang_rad) * 2.2)
                bright2 = int(pulse * 220)
                r2 = max(1, int(pulse * 5))
                self.create_oval(px - r2, py - r2, px + r2, py + r2,
                                  fill=f"#{bright2//2:02x}{bright2:02x}{bright2//3:02x}",
                                  outline="")
            # Centre peak
            cb = int(pulse * 255)
            self.create_oval(cx2-5, cy2-5, cx2+5, cy2+5,
                              fill=f"#{cb:02x}{cb:02x}{cb:02x}", outline="")

        # Overlay: right-side info + peak ring
        cx, cy = self.W // 2, content_h // 2
        alpha = 0.45 + 0.55 * abs(math.sin(t * 0.15))   # min 0.45 so L text never goes black
        r = 12 + int(alpha * 5)
        gclr = f"#{int(alpha*200):02x}{int(alpha*255):02x}{int(alpha*60):02x}"
        self.create_text(W - 120, 28, text="Cepstrum", fill="#aaffaa", font=FH3, anchor="w")
        self.create_text(W - 120, 50, text=f"θ = {self._cep_angle:.0f}°",
                         fill="#aaffaa", font=FMONO, anchor="w")
        self.create_text(W - 120, 68, text=f"L = {self._cep_length} px",
                         fill=gclr, font=("Consolas", 11, "bold"), anchor="w")
        self.create_text(W - 120, 90,
                         text=f"Peak → distance {self._cep_length}px\nfrom centre",
                         fill="#668866", font=FTINY, anchor="w", justify="left")
        self._caption("📊  Cepstrum — blur length encoded as spectral peak distance", "#aaffaa")
        self._t += 1
        self._anim_id = self.after(60, self._animate_cepstrum)

    def show_kernel(self, kernel: np.ndarray, angle: float, length: int):
        self._stop(); self._mode = "kernel"
        self._kernel = kernel; self._ker_angle = angle; self._ker_length = length
        self._t = 0
        self._animate_kernel()

    def _animate_kernel(self):
        self.delete("all")
        self.configure(bg="#0a0010")
        t = self._t
        k = self._kernel
        # Left: kernel heatmap
        if k is not None:
            k_norm = k / (k.max() + 1e-10)
            rgba = (cm.hot(k_norm)[:, :, :3] * 255).astype(np.uint8)
            kh, kw = rgba.shape[:2]
            scale = min(160 / max(kw, 1), 120 / max(kh, 1))
            nw, nh = max(1, int(kw * scale)), max(1, int(kh * scale))
            img = Image.fromarray(rgba).resize((nw, nh), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            self.create_image(20, (self.H - 34 - nh) // 2, anchor="nw", image=self._photo)
        # Right: animated direction arrow
        ax, ay = int(self.W * 0.68), (self.H - 34) // 2
        r = 60
        # Sweeping ghost arc
        for i in range(4):
            sweep = math.radians((self._ker_angle - 10 + i * 5) % 180)
            ex = ax + r * math.cos(sweep); ey = ay - r * math.sin(sweep)
            fade = int(40 + 40 * i)
            r_ = min(255, fade); g_ = min(255, fade // 2); b_ = min(255, fade * 2)
            self.create_line(ax - r*math.cos(sweep), ay + r*math.sin(sweep),
                             ex, ey, fill=f"#{r_:02x}{g_:02x}{b_:02x}", width=1)
        # Main arrow (static actual angle)
        rad = math.radians(self._ker_angle)
        ex = ax + r * math.cos(rad); ey = ay - r * math.sin(rad)
        sx = ax - r * math.cos(rad); sy = ay + r * math.sin(rad)
        self.create_line(sx, sy, ex, ey, fill=ACCENT, width=3, arrow="last",
                         arrowshape=(10, 12, 4))
        # Pulsing circle at centre
        pa = 0.5 + 0.5 * math.sin(t * 0.18)
        pr = int(4 + pa * 3)
        self.create_oval(ax - pr, ay - pr, ax + pr, ay + pr, fill=ACCENT, outline="")
        self.create_text(ax, ay + r + 16, text=f"θ = {self._ker_angle:.0f}°",
                         fill=ACCENT, font=FH3)
        self.create_text(ax, ay + r + 34, text=f"L = {self._ker_length} px",
                         fill=ROSE, font=FMONO)
        self._caption("🔥  PSF Motion Kernel — heatmap left, direction arrow right", "#ffaa66")
        self._t += 1
        self._anim_id = self.after(35, self._animate_kernel)

    def show_hqs_start(self, n_outer: int, n_tv: int):
        self._stop(); self._mode = "hqs_running"
        self._hqs_n = n_outer; self._hqs_tv = n_tv
        self._hqs_start = time.time(); self._t = 0
        self._animate_hqs()

    def _animate_hqs(self):
        if self._mode != "hqs_running": return
        self.delete("all")
        self.configure(bg="#020010")
        elapsed = time.time() - self._hqs_start
        # Estimate iterations based on elapsed time (rough proxy)
        est = min(self._hqs_n - 1, int(elapsed * 1.8))
        prog = est / max(self._hqs_n, 1)
        t = self._t
        # Title
        self.create_text(self.W // 2, 28, text="HQS Solver Running",
                         fill="#aaaaff", font=FH2)
        # Outer iteration counter (big)
        self.create_text(self.W // 2, 70,
                         text=f"Outer  {est} / {self._hqs_n}",
                         fill="white", font=("Segoe UI", 20, "bold"))
        self.create_text(self.W // 2, 100,
                         text=f"TV inner  {self._hqs_tv} per outer",
                         fill=LGRAY, font=FH3)
        # Progress bar
        bx, by, bw = self.W // 2 - 140, 120, 280
        self.create_rectangle(bx, by, bx + bw, by + 14, fill="#1a1030", outline="#333")
        self.create_rectangle(bx, by, bx + int(bw * prog), by + 14, fill=ACCENT, outline="")
        # Scrolling math text at bottom
        eqs = ["min_u  λ‖∇u‖₁ + ½‖Au − y‖²",
               "u-step : FFT solve  (A^H A + γI)u = A^H y + γ(v − d)",
               "v-step : soft threshold   shrink(u + d, λ/γ)",
               "d-step : dual ascent       d ← d + u − v"]
        line = eqs[int(elapsed) % len(eqs)]
        fade = 0.5 + 0.5 * abs(math.sin(t * 0.1))
        color = f"#{int(fade*100):02x}{int(fade*120):02x}{int(fade*200):02x}"
        self.create_text(self.W // 2, 155, text=line, fill=color, font=FMONO_SM)
        # Dots animation
        dots = "●" * (1 + (int(elapsed * 3)) % 4)
        self.create_text(self.W // 2, 178, text=dots, fill=ACCENT, font=FH2)
        self._caption("⚡  Half-Quadratic Splitting + Total Variation denoising", "#aaaaff")
        self._t += 1
        self._anim_id = self.after(80, self._animate_hqs)

    def show_hqs_done(self, arr: np.ndarray):
        self._stop(); self._mode = "hqs_done"; self._arr = arr; self._t = 0
        self._animate_hqs_done()

    def _animate_hqs_done(self):
        if self._mode != "hqs_done": return
        self.delete("all")
        self.configure(bg="#0a0010")
        if self._arr is not None:
            fac = 0.95 + 0.05 * math.sin(self._t * 0.05)
            boosted = np.clip(self._arr * fac, 0, 255).astype(np.uint8)
            self._render_array(boosted)
        # Pulsing checkmark in corner
        pulse = 0.6 + 0.4 * abs(math.sin(self._t * 0.08))
        col = f"#{int(pulse*16):02x}{int(pulse*185):02x}{int(pulse*129):02x}"
        self.create_text(self.W - 10, 12, text="✓ DONE", fill=col,
                         font=("Segoe UI", 9, "bold"), anchor="ne")
        self._caption("✓  HQS complete — final deblurred output", col)
        self._t += 1
        self._anim_id = self.after(80, self._animate_hqs_done)

    def stop(self):
        self._stop()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PIPELINE TAB
# ══════════════════════════════════════════════════════════════════════════════

class PipelineTab(tk.Frame):
    # (id, icon, label, short description shown in timeline)
    STEPS = [
        ("blurred",  "📷", "Input",    "Raw blurred image"),
        ("fft",      "🔵", "FFT",      "Frequency domain"),
        ("cepstrum", "📊", "Cepstrum", "Blur length peak"),
        ("kernel",   "🔥", "Kernel",   "PSF estimation"),
        ("wiener",   "🌊", "Wiener",   "Rough deblur pass"),
        ("hqs",      "⚡", "HQS",      "Final TV solver"),
    ]
    _IDLE    = (SURFACE,  BORDER,  GRAY)
    _RUNNING = ("#fffbe6", AMBER,   AMBER)
    _DONE    = ("#f0fff4", GREEN,   GREEN)

    def __init__(self, parent, on_zoom, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._zoom       = on_zoom
        self._step_btns  = {}   # sid → {"frame","icon_lbl","state_lbl"}
        self._step_data  = {}   # sid → dict with arr/kernel/etc.
        self._gt_text    = ""
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(hdr, text="Pipeline Deep Dive", bg=BG, fg=INK, font=FH1
                 ).pack(side="left")
        tk.Label(hdr, text="click any step to explore its animation",
                 bg=BG, fg=GRAY, font=FSM).pack(side="left", padx=12)

        # ── Step timeline ─────────────────────────────────────────────────────
        tl = tk.Frame(self, bg=BG)
        tl.pack(fill="x", padx=16, pady=(0, 6))

        for i, (sid, icon, label, desc) in enumerate(self.STEPS):
            cell = tk.Frame(tl, bg=SURFACE, cursor="hand2",
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=10, pady=6)
            cell.pack(side="left", padx=2)
            icon_lbl  = tk.Label(cell, text=icon, bg=SURFACE, font=("Segoe UI", 14))
            icon_lbl.pack()
            name_lbl  = tk.Label(cell, text=label, bg=SURFACE, fg=INK, font=FH3)
            name_lbl.pack()
            state_lbl = tk.Label(cell, text="○  idle", bg=SURFACE, fg=LGRAY, font=FTINY)
            state_lbl.pack()
            desc_lbl  = tk.Label(cell, text=desc, bg=SURFACE, fg=GRAY, font=FTINY,
                                  wraplength=80)
            desc_lbl.pack()

            for w in (cell, icon_lbl, name_lbl, state_lbl, desc_lbl):
                w.bind("<Button-1>", lambda e, s=sid: self._select_step(s))
            self._step_btns[sid] = dict(frame=cell, icon=icon_lbl,
                                         state=state_lbl, desc=desc_lbl)
            if i < len(self.STEPS) - 1:
                tk.Label(tl, text="→", bg=BG, fg=LGRAY,
                         font=("Segoe UI", 14)).pack(side="left")

        # ── Main split: cinema (left) + data log (right) ──────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self._cinema = CinemaCanvas(left)
        self._cinema.pack(pady=(0, 6))

        # ── OCR row ───────────────────────────────────────────────────────────
        ocr_card = tk.Frame(left, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1)
        ocr_card.pack(fill="x", pady=(0, 6))
        oi = tk.Frame(ocr_card, bg=CARD)
        oi.pack(padx=12, pady=8, fill="x")
        tk.Label(oi, text="OCR RESULTS", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))
        for engine, attr in (("Tesseract", "_tess_row"), ("EasyOCR", "_easy_row")):
            row = tk.Frame(oi, bg=CARD); row.pack(anchor="w", pady=2, fill="x")
            tk.Label(row, text=f"{engine}:", bg=CARD, fg=GRAY,
                     font=FSM, width=10, anchor="w").pack(side="left")
            disp = OCRDisplay(row); disp.pack(side="left")
            setattr(self, attr, disp)

        # ── Metric badges ─────────────────────────────────────────────────────
        br = tk.Frame(left, bg=BG)
        br.pack(fill="x")
        self._b_angle  = MetricBadge(br, "Angle",   "°",  ACCENT)
        self._b_length = MetricBadge(br, "Length",  "px", ROSE)
        self._b_sharp  = MetricBadge(br, "Sharp Δ", "",   GREEN)
        self._b_psnr   = MetricBadge(br, "PSNR",    "dB", AMBER)
        self._b_ssim   = MetricBadge(br, "SSIM",    "",   INK)
        for b in (self._b_angle, self._b_length, self._b_sharp,
                  self._b_psnr, self._b_ssim):
            b.pack(side="left", padx=4)

        # ── Live data log (right panel) ───────────────────────────────────────
        right = tk.Frame(body, bg=CARD, width=230,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        tk.Label(right, text="LIVE DATA STREAM", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        import tkinter.scrolledtext as _st
        self._log = _st.ScrolledText(right, font=FMONO_SM, bg="#020008",
                                      fg="#00ff88", relief="flat",
                                      state="disabled", wrap="word",
                                      insertbackground="#00ff88")
        self._log.pack(fill="both", expand=True, padx=6, pady=(0, 8))

    # ── Timeline helpers ──────────────────────────────────────────────────────

    def _set_step_state(self, sid, state: str):
        """state: 'idle' | 'running' | 'done'"""
        if sid not in self._step_btns: return
        w = self._step_btns[sid]
        bg, border, fg = (self._IDLE if state == "idle" else
                          self._RUNNING if state == "running" else self._DONE)
        icon_txt = {"idle": "○  idle", "running": "⟳  running", "done": "✓  done"}[state]
        w["frame"].config(bg=bg, highlightbackground=border)
        w["icon"].config(bg=bg)
        w["state"].config(bg=bg, text=icon_txt, fg=fg)
        w["desc"].config(bg=bg)

    def _select_step(self, sid: str):
        data = self._step_data.get(sid)
        if not data:
            return
        # Highlight selected
        for s, w in self._step_btns.items():
            hl = ACCENT if s == sid else (
                GREEN  if self._step_data.get(s) else BORDER)
            w["frame"].config(highlightbackground=hl)
        # Drive cinema
        if sid == "blurred":
            self._cinema.show_image(data["arr"], "📷  Input — raw blurred license plate")
        elif sid == "fft":
            self._cinema.show_fft(data["arr"])
        elif sid == "cepstrum":
            img = data.get("cep_image")
            self._cinema.show_cepstrum(img, data.get("angle", 0), data.get("length", 0))
        elif sid == "kernel":
            self._cinema.show_kernel(data["kernel"], data.get("angle", 0),
                                     data.get("length", 0))
        elif sid == "wiener":
            self._cinema.show_image(data["arr"], "🌊  Wiener rough deblur pass")
        elif sid == "hqs":
            self._cinema.show_hqs_done(data["arr"])

    def _log_write(self, line: str):
        self._log.config(state="normal")
        self._log.insert("end", line + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    # ── Public interface (called by PlateReveal._dispatch) ────────────────────

    def reset(self):
        self._step_data.clear()
        for sid in self._step_btns: self._set_step_state(sid, "idle")
        self._cinema.stop()
        self._cinema.delete("all")
        self._cinema._draw_placeholder()
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")
        self._tess_row.clear(); self._easy_row.clear()
        for b in (self._b_angle, self._b_length, self._b_sharp,
                  self._b_psnr, self._b_ssim):
            b.reset()

    def step_running(self, step_id: str):
        self._set_step_state(step_id, "running")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_write(f"[{ts}]  ⟳  {step_id}  …")

    def step_done(self, step_id: str, arr: np.ndarray = None, label="✓",
                  **extra):
        self._set_step_state(step_id, "done")
        data = {"arr": arr, **extra}
        self._step_data[step_id] = data
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_write(f"[{ts}]  ✓  {step_id}")
        # Auto-drive the cinema to the latest done step
        self._select_step(step_id)

    def show_hqs_start(self, n_outer: int, n_tv: int, delay_ms: int = 0):
        self._set_step_state("hqs", "running")
        self._log_write(f"  ⚡  HQS  n_outer={n_outer}  n_tv={n_tv}")
        if delay_ms > 0:
            # Delay switching the cinema so the Wiener result stays visible briefly
            self.after(delay_ms, lambda: self._cinema.show_hqs_start(n_outer, n_tv))
        else:
            self._cinema.show_hqs_start(n_outer, n_tv)

    def show_results(self, results: dict):
        p = results["params"]; m = results["metrics"]
        gt = results.get("gt_text") or ""
        self._gt_text = gt
        self._b_angle.set(p["angle"],            "°",  ACCENT)
        self._b_length.set(p["length"],           "px", ROSE)
        self._b_sharp.set(m["sharpness_gain"],    "",
                          GREEN if m["sharpness_gain"] > 0 else RED)
        self._b_psnr.set(m.get("psnr"),           "dB", AMBER)
        self._b_ssim.set(m.get("ssim"),           "",   INK)
        ocr = results["ocr"]
        self._tess_row.show(ocr["tesseract"] or "", gt)
        self._easy_row.show(ocr["easyocr"]   or "", gt)
        self._log_write(f"  Tess: {ocr['tesseract'] or '—'}   Easy: {ocr['easyocr'] or '—'}")
        if gt:
            self._log_write(f"  GT:   {gt}")
        self._log_write(f"  PSNR: {m.get('psnr') or '—'}  SSIM: {m.get('ssim') or '—'}"
                        f"  ΔSharp: {m['sharpness_gain']:+.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LAB TAB
# ══════════════════════════════════════════════════════════════════════════════

class LabTab(tk.Frame):
    def __init__(self, parent, on_fast, on_full, on_bench=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._on_fast  = on_fast
        self._on_full  = on_full
        self._on_bench = on_bench
        self._results = None
        self._blurred_arr = None
        self._build()

    def _build(self):
        tk.Label(self, text="Parameter Lab", bg=BG, fg=INK,
                 font=FH1).pack(anchor="w", padx=16, pady=(14,8))

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16)

        # ── LEFT: Controls ────────────────────────────────────────────────────
        left = tk.Frame(main, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0,10), pady=4)

        ctrl = tk.Frame(left, bg=CARD)
        ctrl.pack(padx=14, pady=12, fill="x")

        tk.Label(ctrl, text="ANGLE", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        # Circular dial
        self._dial = DialKnob(ctrl, value=45, callback=self._params_changed)
        self._dial.pack(pady=4)

        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Sliders
        self._sl_len = BlurSlider(ctrl, "Blur Length (px)", 3, 60, 15,
                                   callback=self._params_changed, width=200)
        self._sl_len.pack(pady=(4, 0))
        tk.Label(ctrl, text="length of the motion blur line in the kernel",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w", pady=(0, 4))

        self._sl_lam = BlurSlider(ctrl, "TV Strength  λ  (edge smoothing)", 0.001, 0.2, 0.02,
                                   callback=self._params_changed,
                                   width=200, integer=False)
        self._sl_lam.pack(pady=(4, 0))
        tk.Label(ctrl, text="↑ smoother result   ↓ sharper but noisier  (try 0.01–0.05)",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w", pady=(0, 4))

        self._sl_gam = BlurSlider(ctrl, "Hough Geometry  γ  (streak removal)", 0.0, 0.1, 0.01,
                                   callback=self._params_changed,
                                   width=200, integer=False)
        self._sl_gam.pack(pady=(4, 0))
        tk.Label(ctrl, text="penalises gradients ⊥ to blur — removes directional streaks",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w", pady=(0, 4))

        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Presets
        tk.Label(ctrl, text="PRESETS  (sets all parameters at once)", bg=CARD, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0,4))
        preset_row = tk.Frame(ctrl, bg=CARD)
        preset_row.pack(fill="x")

        PRESETS = {
            "Fast":     dict(n_outer=2,  n_tv=10,  lam=0.02, gamma=0.01),
            "Balanced": dict(n_outer=8,  n_tv=40,  lam=0.02, gamma=0.01),
            "Sharp":    dict(n_outer=12, n_tv=60,  lam=0.01, gamma=0.02),
        }
        for name, vals in PRESETS.items():
            def _apply(v=vals, n=name):
                self._sl_lam.set(v["lam"])
                self._sl_gam.set(v["gamma"])
                self._n_outer_var.set(v["n_outer"])
                self._n_tv_var.set(v["n_tv"])
                self._preset_lbl.config(text=f"Preset: {n}", fg=ACCENT)
            btn = tk.Button(preset_row, text=name, command=_apply,
                            bg=SURFACE, fg=ACCENT, relief="flat",
                            font=FSM, padx=8, pady=3, cursor="hand2")
            btn.pack(side="left", padx=2)

        self._preset_lbl = tk.Label(ctrl, text="", bg=CARD, fg=ACCENT, font=FSM)
        self._preset_lbl.pack(anchor="w", pady=(4,0))

        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Iteration steppers
        self._n_outer_var = tk.IntVar(value=8)
        self._n_tv_var    = tk.IntVar(value=40)

        def _stpr(parent, lbl, var, lo, hi):
            f = tk.Frame(parent, bg=CARD)
            tk.Label(f, text=lbl, bg=CARD, fg=INK, font=FSM,
                     width=11, anchor="w").pack(side="left")
            tk.Button(f, text="−", bg=SURFACE, fg=ACCENT, relief="flat",
                      font=FH3, width=2,
                      command=lambda: var.set(max(lo, var.get()-1))).pack(side="left")
            tk.Label(f, textvariable=var, bg=CARD, fg=INK,
                     font=FH3, width=3).pack(side="left")
            tk.Button(f, text="+", bg=SURFACE, fg=ACCENT, relief="flat",
                      font=FH3, width=2,
                      command=lambda: var.set(min(hi, var.get()+1))).pack(side="left")
            f.pack(fill="x", pady=1)

        _stpr(ctrl, "Outer iter", self._n_outer_var, 1, 20)
        tk.Label(ctrl, text="HQS loops — more = better quality, slower (8 is a good default)",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w")
        _stpr(ctrl, "TV iter",    self._n_tv_var,    5, 100)
        tk.Label(ctrl, text="TV denoising steps per HQS loop (40 is a good default)",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w")

        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Run buttons
        btn_row = tk.Frame(ctrl, bg=CARD)
        btn_row.pack(fill="x")

        def _mk_btn(parent, text, cmd, bg, fg):
            b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                          relief="flat", font=FH3, padx=14, pady=6,
                          cursor="hand2", activebackground=_lerp_color(bg,"#000000",0.1))
            b.pack(side="left", padx=3)
            return b

        _mk_btn(btn_row, "⚡ Fast Preview", self._fast, SURFACE, ACCENT)
        _mk_btn(btn_row, "▶ Full Run",       self._full, ACCENT,  "white")
        tk.Label(ctrl, text="Fast = 2 outer × 10 TV (quick check)   Full = your settings above",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w", pady=(3, 0))

        # Benchmark shortcut
        bench_row = tk.Frame(ctrl, bg=CARD)
        bench_row.pack(fill="x", pady=(6, 0))
        _mk_btn(bench_row, "📊 Benchmark This Config", self._bench_this,
                "#1e3a5f", "#4fc3f7")
        tk.Label(ctrl,
                 text="sends current λ, γ, iters → Benchmark tab and starts a quick run",
                 bg=CARD, fg=LGRAY, font=FTINY).pack(anchor="w", pady=(2, 0))

        # ── RIGHT: Preview panel ───────────────────────────────────────────────
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True, pady=4)

        # Kernel preview
        k_row = tk.Frame(right, bg=BG)
        k_row.pack(fill="x", pady=(0,8))
        tk.Label(k_row, text="PSF Kernel (live)", bg=BG, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0,4))
        self._kernel_cv = KernelCanvas(right)
        self._kernel_cv.pack(anchor="w", pady=(0,10))
        self._kernel_cv.update(15, 45)

        # Reveal slider
        tk.Label(right, text="Before / After  (drag to reveal)",
                 bg=BG, fg=GRAY, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self._reveal = RevealSlider(right, width=420, height=190)
        self._reveal.pack(pady=(4,10))

        # Lab metrics
        m_row = tk.Frame(right, bg=BG)
        m_row.pack(anchor="w")
        self._lab_sharp = MetricBadge(m_row, "Sharp Δ", "", GREEN)
        self._lab_psnr  = MetricBadge(m_row, "PSNR",    "dB", AMBER)
        self._lab_ssim  = MetricBadge(m_row, "SSIM",    "",   INK)
        for b in (self._lab_sharp, self._lab_psnr, self._lab_ssim):
            b.pack(side="left", padx=4)

    def _params_changed(self, _=None):
        """Update kernel preview instantly on any change."""
        self._kernel_cv.update(
            int(self._sl_len.value),
            float(self._dial.value))

    def _get_params(self, fast=False):
        return {
            "angle_hint":  float(self._dial.value),
            "length_hint": int(self._sl_len.value),
            "lam":         float(self._sl_lam.value),
            "gamma":       float(self._sl_gam.value),
            "n_outer":     2  if fast else self._n_outer_var.get(),
            "n_tv":        10 if fast else self._n_tv_var.get(),
        }

    def _fast(self): self._on_fast(self._get_params(fast=True))
    def _full(self): self._on_full(self._get_params(fast=False))

    def _bench_this(self):
        if self._on_bench:
            self._on_bench(self._get_params(fast=False))

    def get_params(self, fast=False) -> dict:
        return self._get_params(fast=fast)

    def load_image(self, blurred_arr: np.ndarray):
        self._blurred_arr = blurred_arr

    def show_results(self, results: dict, blurred_arr: np.ndarray):
        self._results = results
        m = results["metrics"]
        self._lab_sharp.set(m["sharpness_gain"], "",
                            GREEN if m["sharpness_gain"] > 0 else RED)
        self._lab_psnr.set(m.get("psnr"), "dB", AMBER)
        self._lab_ssim.set(m.get("ssim"), "", INK)
        self._reveal.load(blurred_arr,
                          results["intermediates"]["final_rgb"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BENCHMARK TAB
# ══════════════════════════════════════════════════════════════════════════════

class BenchTab(tk.Frame):
    def __init__(self, parent, get_lab_params=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._running        = False
        self._rows           = []
        self._get_lab_params = get_lab_params
        # Solver params (mirrored from LabTab or set independently)
        self._lam_var     = tk.DoubleVar(value=0.02)
        self._gamma_var   = tk.DoubleVar(value=0.01)
        self._n_outer_var = tk.IntVar(value=8)
        self._n_tv_var    = tk.IntVar(value=40)
        self._build()

    def _build(self):
        tk.Label(self, text="Synthetic Benchmark", bg=BG, fg=INK,
                 font=FH1).pack(anchor="w", padx=16, pady=(14,8))

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(0,10))

        def _btn(text, cmd, fg=ACCENT, bg=SURFACE):
            b = tk.Button(top, text=text, command=cmd, bg=bg, fg=fg,
                          relief="flat", font=FH3, padx=14, pady=6,
                          cursor="hand2")
            b.pack(side="left", padx=4)
        _btn("⚡ Quick  (3 configs)", self._run_quick)
        _btn("▶ Full  (80 runs)",    self._run_full, "white", ACCENT)

        # ── Solver params row ─────────────────────────────────────────────────
        prow = tk.Frame(self, bg=SURFACE,
                        highlightbackground=BORDER, highlightthickness=1)
        prow.pack(fill="x", padx=16, pady=(0, 6))

        pk = tk.Frame(prow, bg=SURFACE)
        pk.pack(side="left", padx=10, pady=6)

        def _lbl(parent, txt):
            tk.Label(parent, text=txt, bg=SURFACE, fg=GRAY, font=FTINY).pack(side="left")

        def _entry(parent, var, width=6):
            e = tk.Entry(parent, textvariable=var, bg=CARD, fg=INK, insertbackground=INK,
                         font=FSM, width=width, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1)
            e.pack(side="left", padx=2)
            return e

        def _stpr(parent, var, lo, hi):
            tk.Button(parent, text="−", bg=CARD, fg=ACCENT, relief="flat",
                      font=FSM, width=2,
                      command=lambda: var.set(max(lo, var.get() - 1))).pack(side="left")
            tk.Label(parent, textvariable=var, bg=SURFACE, fg=INK,
                     font=FSM, width=3).pack(side="left")
            tk.Button(parent, text="+", bg=CARD, fg=ACCENT, relief="flat",
                      font=FSM, width=2,
                      command=lambda: var.set(min(hi, var.get() + 1))).pack(side="left")

        _lbl(pk, "λ ="); _entry(pk, self._lam_var, 5)
        tk.Frame(pk, bg=BORDER, width=1).pack(side="left", fill="y", padx=6)
        _lbl(pk, "γ ="); _entry(pk, self._gamma_var, 5)
        tk.Frame(pk, bg=BORDER, width=1).pack(side="left", fill="y", padx=6)
        _lbl(pk, "outer="); _stpr(pk, self._n_outer_var, 1, 20)
        tk.Frame(pk, bg=BORDER, width=1).pack(side="left", fill="y", padx=6)
        _lbl(pk, "TV=");    _stpr(pk, self._n_tv_var,    5, 100)

        tk.Button(prow, text="⟳ Sync from Lab", command=self._sync_from_lab,
                  bg=CARD, fg="#4fc3f7", relief="flat", font=FTINY,
                  padx=8, pady=4, cursor="hand2").pack(side="right", padx=10)

        tk.Label(prow, text="These solver params are applied to all benchmark runs",
                 bg=SURFACE, fg=LGRAY, font=FTINY).pack(side="right", padx=4)

        self._prog_lbl = tk.Label(self, text="", bg=BG, fg=GRAY, font=FSM)
        self._prog_lbl.pack(anchor="w", padx=16)

        # Results table
        import tkinter.ttk as ttk
        style = ttk.Style()
        style.configure("Bench.Treeview",
                        background=CARD, fieldbackground=CARD,
                        foreground=INK, rowheight=24, font=FMONO_SM)
        style.configure("Bench.Treeview.Heading",
                        background=SURFACE, foreground=INK,
                        font=("Segoe UI", 9, "bold"))

        cols = ("image","config","psnr_in","psnr_out","gain","ssim","sharp","ocr")
        self._tree = ttk.Treeview(self, columns=cols, show="headings",
                                   height=14, style="Bench.Treeview")
        heads = ("Image","Config","PSNR blur","PSNR deblur","ΔPSNR","SSIM","ΔSharp","OCR")
        widths = (90, 90, 80, 90, 65, 65, 75, 130)
        for c, h, w in zip(cols, heads, widths):
            self._tree.heading(c, text=h)
            self._tree.column(c, width=w, anchor="center")
        self._tree.pack(fill="both", expand=True, padx=16, pady=(6,0))

        sb = tk.Scrollbar(self, command=self._tree.yview)
        self._tree.config(yscrollcommand=sb.set)

    def _run_quick(self): self._launch(quick=True)
    def _run_full(self):  self._launch(quick=False)

    def run_quick(self):
        """Public: trigger a quick benchmark (called from Lab tab shortcut)."""
        self._launch(quick=True)

    def sync_params(self, params: dict):
        """Copy solver params from a dict (e.g. from LabTab.get_params())."""
        if "lam"     in params: self._lam_var.set(round(params["lam"],   4))
        if "gamma"   in params: self._gamma_var.set(round(params["gamma"], 4))
        if "n_outer" in params: self._n_outer_var.set(int(params["n_outer"]))
        if "n_tv"    in params: self._n_tv_var.set(int(params["n_tv"]))

    def _sync_from_lab(self):
        if self._get_lab_params:
            self.sync_params(self._get_lab_params())

    def _launch(self, quick):
        if self._running:
            messagebox.showinfo("Busy", "Benchmark already running.")
            return
        self._running = True
        for item in self._tree.get_children(): self._tree.delete(item)
        self._rows = []
        lam     = self._lam_var.get()
        gamma   = self._gamma_var.get()
        n_outer = self._n_outer_var.get()
        n_tv    = self._n_tv_var.get()
        self._prog_lbl.config(
            text=f"Starting…  (λ={lam:.3f}  γ={gamma:.3f}  outer={n_outer}  TV={n_tv})",
            fg=AMBER)

        def _worker():
            try:
                from synthetic_bench import run_benchmark, QUICK_CONFIGS, CONFIGS
                configs = QUICK_CONFIGS if quick else CONFIGS
                rows = run_benchmark(configs=configs, verbose=False,
                                     lam=lam, gamma=gamma,
                                     n_outer=n_outer, n_tv=n_tv)
                for r in rows:
                    self.after(0, self._add_row, r)
                self.after(0, self._done, rows)
            except Exception as e:
                self.after(0, self._prog_lbl.config, {"text": f"Error: {e}", "fg": RED})
                self._running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _add_row(self, r: dict):
        if "psnr_deblur" not in r:
            vals = (r.get("image","?"), r.get("config","?"),
                    "—","—","—","—","—", r.get("error","ERR"))
        else:
            vals = (
                r["image"], r["config"],
                f"{r['psnr_blurred']:.2f}", f"{r['psnr_deblur']:.2f}",
                f"{r.get('psnr_gain',0):+.2f}", f"{r['ssim']:.4f}",
                f"{r.get('sharp_gain',0):+.0f}",
                (r.get("ocr_tess") or "—")[:18],
            )
        self._tree.insert("", "end", values=vals)
        self._tree.yview_moveto(1)  # scroll to bottom

    def _done(self, rows):
        valid = [r for r in rows if r.get("psnr_deblur")]
        if valid:
            import numpy as np
            avg_g = np.mean([r.get("psnr_gain",0) for r in valid])
            avg_s = np.mean([r["ssim"] for r in valid])
            self._prog_lbl.config(
                text=f"Done  —  {len(valid)} runs  |  Avg PSNR gain: {avg_g:+.2f} dB  |  Avg SSIM: {avg_s:.4f}",
                fg=GREEN)
        else:
            self._prog_lbl.config(text="Done (no valid results).", fg=GRAY)
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HISTORY TAB
# ══════════════════════════════════════════════════════════════════════════════

class HistoryTab(tk.Frame):
    MAX = 10

    def __init__(self, parent, on_restore, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._on_restore = on_restore
        self._history    = []
        self._build()

    def _build(self):
        tk.Label(self, text="Run History", bg=BG, fg=INK,
                 font=FH1).pack(anchor="w", padx=16, pady=(14,8))
        tk.Label(self, text=f"Last {self.MAX} runs  ·  click any card to restore",
                 bg=BG, fg=GRAY, font=FSM).pack(anchor="w", padx=16)

        self._scroll = tk.Frame(self, bg=BG)
        self._scroll.pack(fill="both", expand=True, padx=16, pady=10)

        self._empty_lbl = tk.Label(
            self._scroll, text="No runs yet.\nRun the pipeline to see history here.",
            bg=BG, fg=LGRAY, font=FBODY, justify="center")
        self._empty_lbl.pack(pady=60)

    def push(self, results: dict, thumb_arr: np.ndarray):
        self._history.insert(0, (results, thumb_arr))
        if len(self._history) > self.MAX:
            self._history = self._history[:self.MAX]
        self._refresh()

    def _refresh(self):
        for w in self._scroll.winfo_children(): w.destroy()
        if not self._history:
            self._empty_lbl = tk.Label(
                self._scroll,
                text="No runs yet.\nRun the pipeline to see history here.",
                bg=BG, fg=LGRAY, font=FBODY, justify="center")
            self._empty_lbl.pack(pady=60)
            return

        for i, (res, thumb) in enumerate(self._history):
            card = self._make_card(res, thumb, i)
            card.pack(fill="x", pady=4)

    def _make_card(self, res, thumb, idx):
        card = tk.Frame(self._scroll, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1,
                        cursor="hand2")

        # Thumbnail
        img = Image.fromarray(thumb.astype(np.uint8))
        img.thumbnail((90, 50), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl_img = tk.Label(card, image=photo, bg=CARD)
        lbl_img.image = photo
        lbl_img.pack(side="left", padx=8, pady=6)

        info = tk.Frame(card, bg=CARD)
        info.pack(side="left", fill="x", expand=True, padx=4)

        p = res["params"]; m = res["metrics"]
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        tk.Label(info, text=Path(res["image_path"]).name,
                 bg=CARD, fg=INK, font=FH3).pack(anchor="w")
        tk.Label(info,
                 text=f"  {p['angle']:.0f}°  ·  {p['length']}px  ·  "
                      f"Sharp Δ {m['sharpness_gain']:+.0f}",
                 bg=CARD, fg=GRAY, font=FSM).pack(anchor="w")
        tess = res["ocr"].get("tesseract") or "—"
        tk.Label(info, text=f"  Tesseract: {tess}",
                 bg=CARD, fg=ACCENT, font=FMONO_SM).pack(anchor="w")

        restore_btn = tk.Button(card, text="Restore",
                                bg=SURFACE, fg=ACCENT, relief="flat",
                                font=FSM, padx=10,
                                command=lambda r=res: self._on_restore(r))
        restore_btn.pack(side="right", padx=10)

        for w in (card, lbl_img, info):
            w.bind("<Button-1>", lambda _, r=res: self._on_restore(r))

        return card


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ZOOM WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class ZoomWindow(tk.Toplevel):
    def __init__(self, parent, step_id: str, arr: np.ndarray):
        super().__init__(parent)
        self.title(f"PlateReveal — {step_id}")
        self.configure(bg=INK)
        self.resizable(True, True)

        W, H = min(1200, arr.shape[1]*2 + 40), min(800, arr.shape[0]*2 + 60)
        self.geometry(f"{W}x{H}")

        img = Image.fromarray(arr.astype(np.uint8))
        img.thumbnail((W-30, H-50), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        tk.Label(self, text=step_id.upper().replace("_"," "),
                 bg=INK, fg="white", font=FH1).pack(pady=8)
        tk.Label(self, image=self._photo, bg=INK).pack(expand=True)
        tk.Button(self, text="Close", command=self.destroy,
                  bg=ACCENT, fg="white", relief="flat",
                  font=FH3, padx=20).pack(pady=8)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7B — MATH VIEW TAB
# ══════════════════════════════════════════════════════════════════════════════

_MATH_SECTIONS = [
    ("Degradation Model", "Step 1 — Problem Setup", """
  Observed image:    y = k ∗ x + n

  y  =  blurred observation (what we see)
  x  =  unknown sharp image (what we want)
  k  =  Point Spread Function (motion blur kernel)
  n  =  sensor noise
  ∗  =  convolution operator

  Goal: recover x given only y — a blind (k unknown) inverse problem.
"""),
    ("Blur Angle — Hough on FFT", "Step 1A — Angle Estimation", """
  Cepstrum:  C(y) = F⁻¹{ log|F{y}| }

  A motion blur kernel at angle θ suppresses energy perpendicular
  to θ in the Fourier spectrum — creating a directional stripe.

  Algorithm:
    1. Apply Hanning window to suppress spectral leakage
    2. Compute log-magnitude FFT
    3. Scan 360 radial lines through the cepstrum centre
    4. For each angle α, find the minimum of Gaussian-smoothed profile
    5. θ̂ = argmin_α  min_r  C_α(r)

  Output: estimated blur angle θ̂ in degrees
"""),
    ("Blur Length — Cepstrum", "Step 1B — Length Estimation", """
  A motion kernel of length L introduces zeros spaced 1/L apart
  along the blur axis in the frequency domain.

  In the cepstrum, this appears as a sharp dip at distance L from
  the centre along the estimated blur direction.

  Algorithm:
    1. Slice cepstrum along θ̂ using bilinear interpolation
    2. Smooth with Gaussian (σ = 1.5)
    3. L̂ = argmin_{r ∈ [6, min(H,W)/2]} C̃_θ̂(r)

  Refinement:  L_final = 0.6·L̂_cepstrum + 0.4·L̂_TPS
"""),
    ("Spatially-Varying PSF — TPS", "Step 1C — Thin Plate Spline", """
  Camera shake can cause spatially varying blur.
  We sample 9 patches in a 3×3 grid, run Hough+Cepstrum on each.

  Thin Plate Spline interpolation:

    f_TPS(p) = a₀ + aᵀp + Σᵢ wᵢ φ(‖p − pᵢ‖)
    φ(r) = r² log r   (radial basis function)

  Outlier patches (local angle deviation > 30° from global) are
  rejected before fitting. The result: dense angle & length maps.
"""),
    ("Wiener Filter", "Step 2 — Rough Deblur", """
  Closed-form deconvolution in the Fourier domain:

    X̂(ω) = K̄(ω) · Y(ω)  /  (|K(ω)|² + K_noise)

  K_noise estimated adaptively via Median Absolute Deviation:
    σ̂_n  =  median|y − G_σ∗y| / 0.6745
    K_noise = clip(σ̂_n² / Var(y),  10⁻⁴,  0.1)

  Applied per RGB channel. Fast but produces Gibbs ringing.
  Output used as warm start for HQS and to extract character mask
  via Otsu thresholding.
"""),
    ("Objective — TV + Hough", "Step 3 — Optimization Problem", """
  Full objective (solved by HQS):

    min_{x ≥ 0}  ½‖k∗x − y‖²  +  λ·TV(x)  +  γ·H(x)

  TV regularization (promotes sharp edges, smooth regions):
    TV(x) = Σᵢⱼ ‖∇xᵢⱼ‖₂

  ◆ Hough Geometry Term (novel — our contribution):
    H(x) = Σᵢⱼ (1 − cos²(∠∇xᵢⱼ − θ_blur)) · ‖∇xᵢⱼ‖₂

  Gradients parallel to blur direction → zero penalty (preserved).
  Perpendicular gradients → penalised → streak artifacts removed.
  λ controls TV strength.  γ controls directional suppression.
"""),
    ("HQS Solver", "Step 4 — Half-Quadratic Splitting", """
  HQS decouples the problem by introducing auxiliary variable z:

    min_{x,z}  ½‖k∗x − y‖²  +  μ/2·‖x − z‖²  +  λ·TV(z) + γ·H(z)

  Alternates between two steps:

  x-step (deconvolution — Wiener in Fourier domain):
    x^{t+1} = F⁻¹{ (K̄·F{y} + μ·F{z^t}) / (|K|² + μ) }

  z-step (denoising — Chambolle–Pock primal-dual):
    Solves the proximal TV+Hough problem iteratively.
    n_outer outer loops × n_tv inner Chambolle–Pock iterations.

  ◆ EVP step (Extremal Value Push — our contribution):
    After each z-step, push intermediate grays toward {0,1}:
      EVP(x; τ) = x − τ if x < 0.5   else   x + τ
    Exploits binary plate prior: characters are dark, bg is bright.

  μ increases each outer iteration to enforce x ≈ z convergence.
"""),
    ("OCR Strategy", "Step 5 — Character Recognition", """
  Two OCR routes are compared:

  Route 1:  deblur(y) → OCR       (our pipeline)
  Route 2:  OCR(y)                (direct on blurred image)

  Tesseract sweep: 80 combinations
    8 preprocessing variants × 5 PSM modes × 2 OEM settings
    Preprocessing: CLAHE, NLM denoise, Otsu binarization,
                   morphological cleaning, upscaling

  EasyOCR: deep-learning engine with preprocessing variants
    (upscale, CLAHE, unsharp mask)

  Best result selected by length and plate character whitelist:
    ABCDEFGHJKLMNPRSTUVWXYZ0123456789
"""),
]


class MathTab(tk.Frame):
    """Interactive pipeline node diagram. Click a node to see its math + animation."""

    # (node_id, label, icon, colour, x_frac, y_frac)
    _NODES = [
        ("input",    "Input\nImage",      "📷", "#3b82f6", 0.08, 0.20),
        ("fft",      "FFT\nSpectrum",     "🔵", "#8b5cf6", 0.25, 0.08),
        ("hough",    "Hough\nAngle θ",    "📏", "#7c3aed", 0.44, 0.08),
        ("cepstrum", "Cepstrum\nLength L","📊", "#6d28d9", 0.63, 0.08),
        ("tps",      "TPS\nPSF map",      "🗺", "#db2777", 0.82, 0.20),
        ("kernel",   "PSF\nKernel k",     "🔥", "#f59e0b", 0.82, 0.55),
        ("wiener",   "Wiener\nFilter",    "🌊", "#059669", 0.63, 0.78),
        ("hqs",      "HQS\nSolver",       "⚡", "#d97706", 0.44, 0.78),
        ("tv",       "TV+Hough\nReg.",    "🧮", "#dc2626", 0.25, 0.78),
        ("output",   "Final\nOutput",     "✨", "#10b981", 0.08, 0.55),
        ("ocr",      "OCR\nReader",       "🔤", "#0284c7", 0.08, 0.78),
    ]
    # directed edges (from, to)
    _EDGES = [
        ("input","fft"),("fft","hough"),("hough","cepstrum"),("cepstrum","tps"),
        ("tps","kernel"),("kernel","wiener"),("kernel","hqs"),
        ("wiener","hqs"),("hqs","tv"),("tv","hqs"),
        ("hqs","output"),("output","ocr"),
    ]
    # Detailed content for each node
    _DETAIL = {
        "input": ("Degradation Model", "#3b82f6",
            "y = k ∗ x + n\n\n"
            "y  =  blurred observation\n"
            "x  =  unknown sharp image  ← what we solve for\n"
            "k  =  Point Spread Function (motion blur)\n"
            "n  =  sensor noise\n"
            "∗  =  convolution\n\n"
            "Goal: recover x given only y (blind inverse problem)"),
        "fft": ("FFT Frequency Domain", "#8b5cf6",
            "F{y}(ω) = Σ_{x} y(x) · e^{−j2πωx}\n\n"
            "Motion blur at angle θ suppresses energy\n"
            "perpendicular to θ → creates a dark stripe\n"
            "through the log-magnitude FFT spectrum.\n\n"
            "We apply a Hanning window first to reduce\n"
            "spectral leakage at the image boundaries."),
        "hough": ("Hough on FFT", "#7c3aed",
            "Radial profile C_α(r) along angle α:\n\n"
            "  C_α(r) = log|F{y}(r·cos α, r·sin α)|\n\n"
            "Scan 360 angles, find the one with the\n"
            "minimum energy (the dark suppression stripe):\n\n"
            "  θ̂ = argmin_α  min_{r∈[10,R]} C_α(r)\n\n"
            "Output: blur angle θ̂ in degrees [0°, 180°)"),
        "cepstrum": ("Cepstrum Length", "#6d28d9",
            "Motion blur of length L creates zeros\n"
            "spaced 1/L apart in frequency domain.\n\n"
            "Cepstrum (log-spectrum inverse FT):\n"
            "  Ce(y) = F⁻¹{ log|F{y}|² }\n\n"
            "Slice along θ̂, find first prominent dip:\n"
            "  L̂ = argmin_{r∈[6,R/2]} Ce̊_θ̂(r)\n\n"
            "Final: L = 0.6·L̂_cepstrum + 0.4·L̂_TPS"),
        "tps": ("Thin Plate Spline", "#db2777",
            "Spatially-varying blur needs a dense PSF map.\n\n"
            "Sample 9 patches in a 3×3 grid, run\n"
            "Hough+Cepstrum on each patch independently.\n\n"
            "Fit a Thin Plate Spline to interpolate:\n"
            "  f_TPS(p) = a₀ + aᵀp + Σᵢ wᵢ φ(‖p−pᵢ‖)\n"
            "  φ(r) = r² log r   (radial basis)\n\n"
            "Result: dense angle_map and length_map"),
        "kernel": ("PSF Kernel Construction", "#f59e0b",
            "Motion blur PSF for angle θ, length L:\n\n"
            "  k[i,j] = 1/L  if pixel (i,j) lies on\n"
            "           the line segment at angle θ\n"
            "           of length L through centre\n"
            "         = 0   otherwise\n\n"
            "K(ω) = F{k}   (kernel in frequency domain)\n\n"
            "Used in Wiener filter and HQS solver."),
        "wiener": ("Wiener Filter", "#059669",
            "Closed-form deconvolution (per channel):\n\n"
            "  X̂(ω) = K̄(ω)·Y(ω) / (|K(ω)|² + K_noise)\n\n"
            "K_noise estimated via MAD:\n"
            "  σ̂ = median|y − G_σ∗y| / 0.6745\n"
            "  K_noise = clip(σ̂²/Var(y), 1e-4, 0.1)\n\n"
            "Fast but causes Gibbs ringing.\n"
            "Used as warm start for HQS."),
        "hqs": ("HQS Solver", "#d97706",
            "Half-Quadratic Splitting decouples:\n\n"
            "  min_{x,z}  ½‖k∗x−y‖² + μ/2·‖x−z‖² + λ·TV(z) + γ·H(z)\n\n"
            "x-step (Wiener in Fourier):\n"
            "  x^{t+1} = F⁻¹{ (K̄·F{y} + μ·F{z^t}) / (|K|²+μ) }\n\n"
            "z-step (Chambolle-Pock TV+Hough prox):\n"
            "  n_outer outer × n_tv inner iterations\n\n"
            "μ increases each outer loop (penalty schedule)."),
        "tv": ("TV + Hough Geometry Term", "#dc2626",
            "Total Variation (promotes sharp edges):\n"
            "  TV(x) = Σᵢⱼ ‖∇xᵢⱼ‖₂\n\n"
            "◆ Hough Geometry Term (our contribution):\n"
            "  H(x) = Σᵢⱼ sin²(∠∇xᵢⱼ − θ_blur) · ‖∇xᵢⱼ‖₂\n\n"
            "Gradients ∥ to blur direction → zero penalty\n"
            "Gradients ⊥ to blur direction → penalised\n"
            "→ removes directional streaking artifacts"),
        "output": ("EVP + Output", "#10b981",
            "After HQS: Extremal Value Push (our contribution)\n\n"
            "  EVP(x; τ) = x − τ  if x < 0.5\n"
            "              x + τ  if x ≥ 0.5\n\n"
            "Exploits binary plate prior: characters are dark,\n"
            "background is bright. Pushes grays toward {0,1}.\n\n"
            "Luminance coupling: weighted RGB → luma channel\n"
            "blended back to suppress colour fringing."),
        "ocr": ("OCR Strategy", "#0284c7",
            "Two engines compared on the deblurred output:\n\n"
            "Tesseract sweep: 80 configs\n"
            "  8 preprocessing × 5 PSM × 2 OEM\n"
            "  Preprocessing: CLAHE, NLM, Otsu, morph, upscale\n\n"
            "EasyOCR: deep-learning engine + preprocessing\n"
            "  (upscale, CLAHE, unsharp mask)\n\n"
            "Best result selected by whitelist:\n"
            "  ABCDEFGHJKLMNPRSTUVWXYZ0123456789"),
    }

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._selected  = None
        self._anim_id   = None
        self._anim_t    = 0
        self._params_str = "—  run the pipeline first"
        self._build()

    def _build(self):
        tk.Label(self, text="Algorithm Diagram", bg=BG, fg=INK, font=FH1
                 ).pack(anchor="w", padx=16, pady=(12, 2))
        tk.Label(self, text="Click any node to see its formula, role, and a live animation",
                 bg=BG, fg=GRAY, font=FSM).pack(anchor="w", padx=16, pady=(0, 6))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        # Left: node diagram canvas
        self._diag = tk.Canvas(body, bg="#0d0010", width=520, height=340,
                               highlightthickness=1, highlightbackground=BORDER)
        self._diag.pack(side="left", fill="y")
        self._diag.bind("<Configure>", self._redraw_diagram)
        self._diag.bind("<Button-1>", self._on_click)

        # Right: detail + mini-animation panel
        right = tk.Frame(body, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self._detail_title = tk.Label(right, text="Select a node →",
                                       bg=CARD, fg=INK, font=FH2, anchor="w")
        self._detail_title.pack(anchor="w", padx=14, pady=(12, 4))

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=14)

        self._mini_anim = tk.Canvas(right, bg="#020008", width=340, height=130,
                                     highlightthickness=0)
        self._mini_anim.pack(padx=14, pady=(8, 4))

        self._detail_text = tk.Text(right, font=FMONO_SM, bg=CARD, fg=INK,
                                     relief="flat", wrap="word",
                                     height=10, width=44, state="disabled")
        self._detail_text.pack(padx=14, pady=(0, 8), fill="both", expand=True)

        # Bottom: live params bar
        params_bar = tk.Frame(self, bg=SURFACE,
                              highlightbackground=BORDER, highlightthickness=1)
        params_bar.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(params_bar, text="  Last run: ", bg=SURFACE, fg=GRAY,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=6)
        self._params_lbl = tk.Label(params_bar, text=self._params_str,
                                     bg=SURFACE, fg=INK, font=FMONO_SM)
        self._params_lbl.pack(side="left", padx=6)

        self.after(100, self._redraw_diagram)

    def _node_coords(self, w: int, h: int):
        pad = 30
        return {nid: (int(pad + x * (w - 2*pad)), int(pad + y * (h - 2*pad)))
                for nid, _, _, _, x, y in self._NODES}

    def _redraw_diagram(self, _=None):
        self._diag.delete("all")
        w = self._diag.winfo_width() or 520
        h = self._diag.winfo_height() or 340
        coords = self._node_coords(w, h)
        nw, nh = 72, 48
        # Draw edges first
        for (a, b) in self._EDGES:
            if a not in coords or b not in coords: continue
            ax, ay = coords[a]; bx, by = coords[b]
            self._diag.create_line(ax, ay, bx, by, fill="#333355", width=1,
                                    arrow="last", arrowshape=(6, 8, 3))
        # Draw nodes
        for nid, label, icon, colour, *_ in self._NODES:
            if nid not in coords: continue
            cx, cy = coords[nid]
            x0, y0, x1, y1 = cx-nw//2, cy-nh//2, cx+nw//2, cy+nh//2
            is_sel = (nid == self._selected)
            # Dim version of the node colour for unselected state
            r = int(colour[1:3], 16); g = int(colour[3:5], 16); b = int(colour[5:7], 16)
            dim_fill = f"#{r//4:02x}{g//4:02x}{b//4:02x}"
            bg = colour if is_sel else dim_fill
            outline_col = "white" if is_sel else colour
            bw = 3 if is_sel else 2
            # Shadow
            self._diag.create_rectangle(x0+3, y0+3, x1+3, y1+3, fill="#08040f", outline="")
            self._diag.create_rectangle(x0, y0, x1, y1, fill=bg,
                                         outline=outline_col, width=bw)
            self._diag.create_text(cx, cy - 9, text=icon,
                                    fill="white", font=("Segoe UI", 12))
            self._diag.create_text(cx, cy + 10, text=label,
                                    fill="white", font=FTINY, justify="center")

    def _on_click(self, event):
        w = self._diag.winfo_width() or 520
        h = self._diag.winfo_height() or 340
        coords = self._node_coords(w, h)
        nw, nh = 72, 48
        for nid, *_ in self._NODES:
            if nid not in coords: continue
            cx, cy = coords[nid]
            if (cx - nw//2 <= event.x <= cx + nw//2 and
                    cy - nh//2 <= event.y <= cy + nh//2):
                self._select(nid)
                return

    def _select(self, nid: str):
        self._selected = nid
        self._redraw_diagram()
        if nid not in self._DETAIL: return
        title, colour, body = self._DETAIL[nid]
        self._detail_title.config(text=title, fg=colour)
        self._detail_text.config(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("end", body)
        self._detail_text.config(state="disabled")
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        self._anim_t = 0
        self._animate_node(nid, colour)

    def _animate_node(self, nid: str, colour: str):
        c = self._mini_anim
        c.delete("all")
        t = self._anim_t
        W, H = 340, 130

        if nid == "fft":
            # Animated frequency bars
            c.configure(bg="#020008")
            n = 18
            for i in range(n):
                freq = abs(math.sin(t * 0.05 + i * 0.6)) * 0.8 + 0.2 * abs(math.sin(i))
                bh = int(freq * (H - 20))
                x = 10 + i * (W - 20) // n
                bw = max(1, (W - 20) // n - 3)
                bright = int(80 + 175 * freq)
                col = f"#{bright//2:02x}{bright//3:02x}{bright:02x}"
                c.create_rectangle(x, H - 10 - bh, x + bw, H - 10, fill=col, outline="")
            c.create_text(W // 2, 10, text="Frequency Domain — energy over ω",
                          fill="#8888ff", font=FTINY)

        elif nid == "hough":
            # Rotating scan lines around centre point
            c.configure(bg="#020008")
            cx, cy = W // 2, H // 2
            c.create_oval(cx-60, cy-60, cx+60, cy+60, outline="#222", width=1)
            for i in range(12):
                ang = math.radians(i * 15 + t * 1.5)
                bright = int(80 + 175 * abs(math.sin(i + t * 0.05)))
                col = f"#{bright//2:02x}{bright//4:02x}{bright:02x}"
                c.create_line(cx - 60*math.cos(ang), cy - 60*math.sin(ang),
                              cx + 60*math.cos(ang), cy + 60*math.sin(ang),
                              fill=col, width=1)
            # Highlight minimum energy angle
            best_ang = math.radians(t % 180)
            c.create_line(cx - 64*math.cos(best_ang), cy - 64*math.sin(best_ang),
                          cx + 64*math.cos(best_ang), cy + 64*math.sin(best_ang),
                          fill="#ff6600", width=2)
            c.create_text(W//2, H-12, text="Scanning all angles → min energy = blur direction",
                          fill="#ffaa66", font=FTINY)

        elif nid == "kernel":
            # Animated kernel drawing (line sweeping)
            c.configure(bg="#020008")
            cx, cy = W // 2, H // 2
            L = 50 + 20 * abs(math.sin(t * 0.04))
            ang = math.radians((t * 0.8) % 180)
            dx, dy = math.cos(ang), -math.sin(ang)
            steps = int(L)
            for i in range(steps):
                frac = i / max(steps, 1)
                px = cx + (frac - 0.5) * L * dx
                py = cy + (frac - 0.5) * L * dy
                bright = int(100 + 155 * abs(math.sin(frac * math.pi)))
                r = max(1, int(bright / 80))
                c.create_oval(px-r, py-r, px+r, py+r,
                              fill=f"#{bright:02x}{bright//2:02x}00", outline="")
            c.create_text(W//2, H-14, text=f"PSF kernel  L={int(L)}px  θ={int(t*0.8)%180}°",
                          fill="#ffaa44", font=FTINY)

        elif nid == "hqs":
            # Convergence curve animation
            c.configure(bg="#020008")
            pts = []
            n_pts = 30
            for i in range(n_pts):
                x = 20 + i * (W - 40) // n_pts
                progress = min(1.0, i / max(t * 0.15, 1))
                y = H - 20 - int((H - 40) * (1 - math.exp(-progress * 3)))
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(*pts, fill=ACCENT, width=2, smooth=True)
            # Animated cursor
            cur_i = int(t * 0.15) % n_pts
            cx2 = 20 + cur_i * (W - 40) // n_pts
            c.create_oval(cx2-4, H-20 - int((H-40)*(1-math.exp(-min(1,cur_i/(max(t*0.15,1)))*3)))-4,
                          cx2+4, H-20 - int((H-40)*(1-math.exp(-min(1,cur_i/(max(t*0.15,1)))*3)))+4,
                          fill="#ff8c00", outline="")
            c.create_text(W//2, 12, text="PSNR convergence over HQS iterations",
                          fill="#aaaaff", font=FTINY)

        elif nid == "tv":
            # Signal before/after TV denoising
            c.configure(bg="#020008")
            n = 60
            for i in range(n - 1):
                x1 = 10 + i * (W - 20) // n
                x2 = 10 + (i+1) * (W - 20) // n
                # Noisy signal
                noisy = 0.5 + 0.35*math.sin(i*0.4) + 0.15*math.sin(i*3.1+t*0.1)
                noisy2 = 0.5 + 0.35*math.sin((i+1)*0.4) + 0.15*math.sin((i+1)*3.1+t*0.1)
                # Smooth signal
                smooth = 0.5 + 0.35*math.sin(i*0.4)
                smooth2 = 0.5 + 0.35*math.sin((i+1)*0.4)
                c.create_line(x1, H//2 - int(noisy*50), x2, H//2 - int(noisy2*50),
                              fill="#ff6666", width=1)
                c.create_line(x1, H//2 - int(smooth*50), x2, H//2 - int(smooth2*50),
                              fill="#00ff88", width=2)
            c.create_text(10, 8, text="noisy", fill="#ff6666", font=FTINY, anchor="w")
            c.create_text(10, 20, text="TV-smooth", fill="#00ff88", font=FTINY, anchor="w")

        elif nid == "wiener":
            # Fourier division animation
            c.configure(bg="#020008")
            n = 40
            for i in range(n - 1):
                x1 = 10 + i * (W - 20) // n
                x2 = 10 + (i + 1) * (W - 20) // n
                fy = 0.3 + 0.5 * abs(math.sin(i * 0.5))
                ky = 0.1 + 0.4 * abs(math.sin(i * 0.5 + 0.2))
                result = fy / (ky + 0.01)
                result = min(result, 1.5)
                c.create_line(x1, H//3 - int(fy*30), x2, H//3 - int(0.3+0.5*abs(math.sin((i+1)*0.5)))*30,
                              fill="#8888ff", width=1)
                c.create_line(x1, H*2//3 - int(result*25), x2,
                              H*2//3 - int(min(fy/(0.1+0.4*abs(math.sin((i+1)*0.5+0.2)))+0.01,1.5)*25),
                              fill="#00ff88", width=2)
            c.create_text(W//2, H-12, text="Y(ω) / K(ω)  →  deconvolved X̂(ω)",
                          fill="#aaffaa", font=FTINY)

        elif nid == "ocr":
            # Character reveal animation
            c.configure(bg="#020008")
            plate = "AB123CD"
            shown = int(t * 0.25) % (len(plate) + 4)
            for i, ch in enumerate(plate[:shown]):
                x = 30 + i * 42
                match = (i % 3 != 1)  # demo correct/wrong
                col = "#00ff88" if match else "#ff4444"
                c.create_rectangle(x, 25, x+34, 85, fill="#1a1030", outline=col, width=2)
                c.create_text(x+17, 55, text=ch, fill=col,
                              font=("Consolas", 20, "bold"))
            c.create_text(W//2, H-12, text="Tesseract sweeping 80 config combinations",
                          fill="#88aaff", font=FTINY)

        else:
            # Generic pulse
            c.configure(bg="#020008")
            r = 20 + 15 * abs(math.sin(t * 0.1))
            for node_info in self._NODES:
                if node_info[0] == nid:
                    _, _, icon, colour, *_ = node_info
                    break
            else:
                colour = ACCENT
            c.create_oval(W//2-r, H//2-r, W//2+r, H//2+r,
                          outline=colour, width=3)
            c.create_text(W//2, H//2, text=nid.upper(), fill=colour, font=FH2)

        self._anim_t += 1
        _col = colour
        self._anim_id = self.after(55, lambda n=nid, c=_col: self._animate_node(n, c))

    def update_params(self, results: dict):
        p = results["params"]; m = results["metrics"]
        self._params_lbl.config(
            text=(f"θ={p['angle']:.1f}°  L={p['length']}px  "
                  f"λ={p['lam']:.3f}  γ={p['gamma']:.3f}  "
                  f"ΔSharp={m['sharpness_gain']:+.1f}  "
                  + (f"PSNR={m['psnr']:.2f}dB" if m.get('psnr') else "PSNR=N/A")))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7C — OCR COMPARISON TAB
# ══════════════════════════════════════════════════════════════════════════════

_GT = {
    "img1": "GJW115A1138", "img2": "TN52U1580",  "img3": "WB06F9209",
    "img4": "DL10CG4693",  "img5": "TN45BA1065", "img6": "MH14EP4660",
    "img7": "TN21AU7234",  "img8": "HR26DA0471",
}


class OCRCompareTab(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._rows   = []
        self._running = False
        self._build()

    def _build(self):
        tk.Label(self, text="OCR Engine Comparison", bg=BG, fg=INK,
                 font=FH1).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(self,
                 text="Runs Tesseract (80-strategy sweep) and EasyOCR on every "
                      "deblurred output image. Green = winner.",
                 bg=BG, fg=GRAY, font=FSM).pack(anchor="w", padx=16, pady=(0, 8))

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=(0, 8))
        tk.Button(top, text="▶  Run OCR Comparison",
                  bg=ACCENT, fg="white", relief="flat", font=FH3,
                  padx=16, pady=6, cursor="hand2",
                  command=self._run).pack(side="left")
        self._status = tk.Label(top, text="", bg=BG, fg=GRAY, font=FSM)
        self._status.pack(side="left", padx=12)

        # Treeview
        import tkinter.ttk as ttk
        style = ttk.Style()
        style.configure("OCR.Treeview",
                        background=CARD, fieldbackground=CARD,
                        foreground=INK, rowheight=26, font=FMONO_SM)
        style.configure("OCR.Treeview.Heading",
                        background=SURFACE, foreground=INK,
                        font=("Segoe UI", 9, "bold"))

        cols = ("img","gt","tess","easy","t_sim","e_sim","winner")
        self._tree = ttk.Treeview(self, columns=cols, show="headings",
                                   height=10, style="OCR.Treeview")
        heads  = ("Image","Ground Truth","Tesseract","EasyOCR","Tess Sim","Easy Sim","Winner")
        widths = (70, 120, 140, 140, 75, 75, 80)
        for c, h, w in zip(cols, heads, widths):
            self._tree.heading(c, text=h)
            self._tree.column(c, width=w, anchor="center")
        self._tree.pack(fill="x", padx=16, pady=(0, 8))

        # Charts area (matplotlib embedded)
        self._chart_frame = tk.Frame(self, bg=BG)
        self._chart_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self._fig = Figure(figsize=(10, 3.2), facecolor=BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._chart_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _similarity(self, pred, gt):
        if not pred or not gt: return 0.0
        n = min(len(pred), len(gt))
        return sum(a == b for a, b in zip(pred, gt)) / len(gt)

    def _run(self):
        if self._running:
            return
        self._running = True
        self._status.config(text="Loading OCR engines…", fg=AMBER)
        for item in self._tree.get_children(): self._tree.delete(item)
        self._rows = []

        def _worker():
            try:
                from ocr_plate import recognize as tess_rec
                try:
                    from ocr_easyocr import recognize_plate as easy_rec
                    _easy = True
                except Exception:
                    _easy = False

                # Collect ALL deblurred outputs from every run directory
                candidates = []
                if OUTPUT_DIR.exists():
                    for run_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True)[:20]:
                        fin = run_dir / "05_final_deblur.png"
                        if fin.exists():
                            candidates.append(fin)
                # Also include raw input images if no output found
                if not candidates:
                    for d in (INPUT_DIR, NOT_BLURRED):
                        if d and d.exists():
                            for ext in ("*.png", "*.jpg", "*.jpeg"):
                                candidates += list(d.glob(ext))[:4]

                rows = []
                total = len(candidates)
                for i, img_path in enumerate(candidates):
                    self.after(0, self._status.config,
                               {"text": f"Processing {i+1}/{total}…", "fg": AMBER})
                    from PIL import Image as PILImg
                    import numpy as np
                    arr = np.array(PILImg.open(img_path).convert("RGB"))
                    tess = tess_rec(arr)
                    easy = easy_rec(arr) if _easy else "N/A"
                    stem = img_path.stem.replace("05_final_deblur","").strip("_")
                    # Try to match a GT key
                    gt = ""
                    for k in _GT:
                        if k in img_path.parts[-2] or k in stem:
                            gt = _GT[k]; break
                    t_sim = self._similarity(tess, gt) if gt else None
                    e_sim = self._similarity(easy, gt) if gt else None
                    rows.append(dict(img=img_path.parent.name[:14],
                                     gt=gt or "?",
                                     tess=tess or "—",
                                     easy=easy or "—",
                                     t_sim=t_sim, e_sim=e_sim))
                    self.after(0, self._add_row, rows[-1])

                self.after(0, self._finish, rows)
            except Exception as e:
                self.after(0, self._status.config,
                           {"text": f"Error: {e}", "fg": RED})
                self._running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _add_row(self, r):
        ts = f"{r['t_sim']:.2f}" if r['t_sim'] is not None else "—"
        es = f"{r['e_sim']:.2f}" if r['e_sim'] is not None else "—"
        if r['t_sim'] is not None and r['e_sim'] is not None:
            winner = "Tesseract" if r['t_sim'] >= r['e_sim'] else "EasyOCR"
        else:
            winner = "—"
        self._tree.insert("", "end",
                          values=(r['img'], r['gt'], r['tess'], r['easy'],
                                  ts, es, winner))

    def _finish(self, rows):
        self._rows = rows
        self._status.config(
            text=f"Done — {len(rows)} images compared", fg=GREEN)
        self._running = False
        self._draw_charts(rows)

    def _draw_charts(self, rows):
        self._fig.clear()
        valid = [r for r in rows
                 if r['t_sim'] is not None and r['e_sim'] is not None]
        if not valid:
            return

        import numpy as np
        ax1 = self._fig.add_subplot(1, 2, 1)
        ax2 = self._fig.add_subplot(1, 2, 2)
        self._fig.patch.set_facecolor(BG)

        labels = [r['img'][:10] for r in valid]
        t_sims = [r['t_sim'] for r in valid]
        e_sims = [r['e_sim'] for r in valid]
        x = np.arange(len(valid))

        # Bar chart
        ax1.set_facecolor(CARD)
        ax1.bar(x - 0.18, t_sims, 0.35, color=ACCENT,  label="Tesseract", alpha=0.85)
        ax1.bar(x + 0.18, e_sims, 0.35, color=ROSE,    label="EasyOCR",   alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=40,
                                                 ha="right", fontsize=8, color=GRAY)
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Similarity by Image", color=INK, fontsize=10, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.tick_params(colors=GRAY)
        ax1.set_ylabel("Character Similarity", color=GRAY)
        for s in ax1.spines.values(): s.set_color(BORDER)

        # Pie chart — wins
        t_wins = sum(1 for r in valid if r['t_sim'] >= r['e_sim'])
        e_wins = len(valid) - t_wins
        ax2.set_facecolor(CARD)
        wedges, texts, pcts = ax2.pie(
            [t_wins, e_wins],
            labels=["Tesseract", "EasyOCR"],
            colors=[ACCENT, ROSE],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": INK, "fontsize": 9},
        )
        ax2.set_title("Win Rate", color=INK, fontsize=10, fontweight="bold")

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7D — HISTOGRAMS / ANALYSIS TAB
# ══════════════════════════════════════════════════════════════════════════════

class HistogramTab(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._fig    = None
        self._canvas = None
        self._build()

    def _build(self):
        tk.Label(self, text="Image Analysis", bg=BG, fg=INK,
                 font=FH1).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(self,
                 text="Pixel distributions, sharpness evolution, PSF kernel, and quality metrics. "
                      "Auto-refreshes after every pipeline run.",
                 bg=BG, fg=GRAY, font=FSM).pack(anchor="w", padx=16, pady=(0, 8))

        self._empty = tk.Label(self,
                               text="Run the pipeline to populate the analysis panels.",
                               bg=BG, fg=LGRAY, font=FBODY)
        self._empty.pack(pady=60)

        self._chart_frame = tk.Frame(self, bg=BG)
        # not packed until we have data

    def update(self, results: dict):
        self._empty.pack_forget()
        self._chart_frame.pack(fill="both", expand=True, padx=8, pady=(0,8))

        if self._fig:
            plt.close(self._fig)
        if self._canvas:
            self._canvas.get_tk_widget().destroy()

        interm  = results["intermediates"]
        blurred = interm["blurred_rgb"].astype(np.float64)
        final   = interm["final_rgb"].astype(np.float64)
        kernel  = interm["kernel"]
        mask    = interm["mask"].astype(np.float64)
        m       = results["metrics"]

        self._fig = Figure(figsize=(13, 7), facecolor=BG)
        axes = self._fig.subplots(2, 3)
        self._fig.subplots_adjust(hspace=0.42, wspace=0.32,
                                   left=0.06, right=0.97,
                                   top=0.93, bottom=0.08)

        def _style(ax, title):
            ax.set_facecolor(CARD)
            ax.set_title(title, color=INK, fontsize=9, fontweight="bold")
            ax.tick_params(colors=GRAY, labelsize=7)
            for s in ax.spines.values(): s.set_color(BORDER)

        # ── Panel 1: Luminance histogram ──────────────────────────────────────
        ax = axes[0, 0]; _style(ax, "Luminance Histogram")
        lum_b = 0.299*blurred[:,:,0]+0.587*blurred[:,:,1]+0.114*blurred[:,:,2]
        lum_f = 0.299*final[:,:,0] +0.587*final[:,:,1] +0.114*final[:,:,2]
        ax.hist(lum_b.ravel(), bins=80, color=ROSE,   alpha=0.6, label="Blurred",   density=True)
        ax.hist(lum_f.ravel(), bins=80, color=ACCENT, alpha=0.6, label="Deblurred", density=True)
        ax.legend(fontsize=7, facecolor=CARD, labelcolor=INK)
        ax.set_xlabel("Pixel value", color=GRAY, fontsize=7)

        # ── Panel 2: RGB channel histograms ───────────────────────────────────
        ax = axes[0, 1]; _style(ax, "RGB Channels (deblurred)")
        for c, col, lbl in ((0,"#ef4444","R"),(1,"#10b981","G"),(2,"#3b82f6","B")):
            ax.hist(final[:,:,c].ravel(), bins=60, color=col, alpha=0.55,
                    label=lbl, density=True)
        ax.legend(fontsize=7, facecolor=CARD, labelcolor=INK)
        ax.set_xlabel("Pixel value", color=GRAY, fontsize=7)

        # ── Panel 3: PSF Kernel heatmap ───────────────────────────────────────
        ax = axes[0, 2]; _style(ax, "PSF Kernel")
        ax.imshow(kernel / (kernel.max()+1e-10), cmap="hot", aspect="auto",
                  interpolation="nearest")
        ax.set_xlabel(f"θ={results['params']['angle']:.0f}°  "
                      f"L={results['params']['length']}px", color=GRAY, fontsize=7)

        # ── Panel 4: Sharpness bar ────────────────────────────────────────────
        ax = axes[1, 0]; _style(ax, "Sharpness (Laplacian Var.)")
        bars = ax.bar(["Blurred", "Deblurred"],
                      [m["sharpness_input"], m["sharpness_output"]],
                      color=[ROSE, GREEN], alpha=0.85, width=0.5)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK, padding=3)
        gain = m["sharpness_gain"]
        ax.set_title(f"Sharpness  (Δ = {gain:+.0f})",
                     color=GREEN if gain > 0 else RED,
                     fontsize=9, fontweight="bold")

        # ── Panel 5: Character mask density ───────────────────────────────────
        ax = axes[1, 1]; _style(ax, "Character Mask Density")
        ax.imshow(mask, cmap="Purples", aspect="auto", vmin=0, vmax=1)
        pct = mask.mean() * 100
        ax.set_xlabel(f"Mask coverage: {pct:.1f}% of pixels",
                      color=GRAY, fontsize=7)

        # ── Panel 6: Quality metrics bar ──────────────────────────────────────
        ax = axes[1, 2]; _style(ax, "Quality Metrics")
        metric_names = ["Sharp\nGain", "PSNR\n(dB)", "SSIM\n×100"]
        vals = [
            max(0, min(100, m["sharpness_gain"] / 10)),
            m.get("psnr") or 0,
            (m.get("ssim") or 0) * 100,
        ]
        colors = [GREEN if v > 0 else RED for v in vals]
        bars2 = ax.bar(metric_names, vals, color=colors, alpha=0.85, width=0.5)
        ax.bar_label(bars2, fmt="%.1f", fontsize=8, color=INK, padding=3)
        ax.set_ylim(0, max(vals) * 1.25 + 1)
        if m.get("psnr") is None:
            ax.set_xlabel("PSNR/SSIM require ground truth (--gt)",
                          color=LGRAY, fontsize=7)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._chart_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class PlateReveal(tk.Tk):
    TABS = ["Pipeline", "Lab", "Math", "OCR Compare", "Histograms", "Benchmark", "History"]

    def __init__(self, initial_image=None):
        super().__init__()
        self.title("PlateReveal Studio")
        self.geometry("1340x820")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self._q        = queue.Queue()
        self._running  = False
        self._last_res = None
        self._last_bl  = None  # blurred array from last run
        self._lab_mode = False  # True when run triggered from Lab

        self._build_ui()

        if initial_image:
            self.after(200, lambda: self._sidebar.set_image(initial_image))

        self._poll()

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top header bar
        hdr = tk.Frame(self, bg=ACCENT, height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        tk.Label(hdr, text="◉  PlateReveal Studio",
                 bg=ACCENT, fg="white", font=FH1).pack(side="left", padx=18)

        # Custom tab buttons
        self._tab_btns = {}
        self._active_tab = tk.StringVar(value="Pipeline")
        tab_bar = tk.Frame(hdr, bg=ACCENT)
        tab_bar.pack(side="left", padx=20)

        for tab in self.TABS:
            btn = tk.Button(tab_bar, text=tab,
                            bg=ACCENT, fg="white", relief="flat",
                            font=("Segoe UI", 10, "bold"),
                            padx=16, pady=8, cursor="hand2",
                            activebackground=A_DARK, activeforeground="white",
                            command=lambda t=tab: self._switch_tab(t))
            btn.pack(side="left")
            self._tab_btns[tab] = btn

        # Status bar (bottom)
        self._status_bar = tk.Frame(self, bg=SURFACE, height=28)
        self._status_bar.pack(side="bottom", fill="x")
        self._status_bar.pack_propagate(False)
        self._status_lbl = tk.Label(self._status_bar, text="Ready",
                                     bg=SURFACE, fg=GRAY, font=FSM)
        self._status_lbl.pack(side="left", padx=12)

        # Main horizontal split
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        # Sidebar
        self._sidebar = SideBar(main, on_run=self._on_run_requested)
        self._sidebar.pack(side="left", fill="y")

        # Content area
        self._content = tk.Frame(main, bg=BG)
        self._content.pack(side="left", fill="both", expand=True)

        # Tabs (all packed but only one visible)
        self._pipeline_tab = PipelineTab(self._content,
                                          on_zoom=self._open_zoom)
        self._lab_tab      = LabTab(self._content,
                                     on_fast=self._on_lab_fast,
                                     on_full=self._on_lab_full,
                                     on_bench=self._on_lab_bench)
        self._bench_tab    = BenchTab(self._content,
                                      get_lab_params=lambda: self._lab_tab.get_params())
        self._history_tab  = HistoryTab(self._content,
                                         on_restore=self._on_restore)
        self._math_tab     = MathTab(self._content)
        self._ocr_tab      = OCRCompareTab(self._content)
        self._hist_tab     = HistogramTab(self._content)

        self._tab_frames = {
            "Pipeline":    self._pipeline_tab,
            "Lab":         self._lab_tab,
            "Math":        self._math_tab,
            "OCR Compare": self._ocr_tab,
            "Histograms":  self._hist_tab,
            "Benchmark":   self._bench_tab,
            "History":     self._history_tab,
        }
        # Now that _tab_frames exists, wire up the active tab highlight
        self._switch_tab("Pipeline")

    def _switch_tab(self, name: str):
        for t, f in self._tab_frames.items():
            if t == name:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()
        self._active_tab.set(name)
        for t, btn in self._tab_btns.items():
            btn.config(bg=A_DARK if t == name else ACCENT)

    # ── Threading ─────────────────────────────────────────────────────────────

    def _on_run_requested(self, params: dict):
        if self._running:
            messagebox.showinfo("Busy", "Pipeline is still running.")
            return
        self._running  = True
        self._lab_mode = False
        self._pipeline_tab.reset()
        self._switch_tab("Pipeline")
        self._sidebar.set_status("Running…", AMBER)
        self._set_status("Running pipeline…")

        # Show blurred as first card immediately
        def _worker():
            try:
                from pipeline_run import run_pipeline
                _skip = {"angle_hint","length_hint","lam","gamma","gt_text"}
                results = run_pipeline(
                    **{k: v for k, v in params.items() if k not in _skip},
                    lam         = params.get("lam", 0.02),
                    gamma       = params.get("gamma", 0.01),
                    angle_hint  = params.get("angle_hint"),
                    length_hint = params.get("length_hint"),
                    gt_text     = params.get("gt_text"),
                    verbose=False,
                    step_callback=lambda msg: self._q.put(msg),
                )
                self._q.put({"type": "done", "results": results})
            except Exception as e:
                import traceback
                self._q.put({"type": "error", "message": str(e),
                             "tb": traceback.format_exc()})
        threading.Thread(target=_worker, daemon=True).start()

    def _on_lab_fast(self, lab_params: dict):
        self._run_with_lab_params(lab_params)

    def _on_lab_full(self, lab_params: dict):
        self._run_with_lab_params(lab_params)

    def _on_lab_bench(self, lab_params: dict):
        """Sync Lab params to BenchTab, switch there, and start a quick run."""
        self._bench_tab.sync_params(lab_params)
        self._switch_tab("Benchmark")
        self._bench_tab.run_quick()

    def _run_with_lab_params(self, lab_params: dict):
        if not self._last_res:
            messagebox.showinfo("No image", "Run the pipeline from the sidebar first.")
            return
        if self._running:
            messagebox.showinfo("Busy", "Pipeline still running.")
            return
        self._running  = True
        self._lab_mode = True
        self._set_status("Lab run in progress…")
        self._switch_tab("Lab")
        img_path = self._last_res["image_path"]
        gt_path  = self._last_res.get("gt_path")
        synth    = self._last_res.get("synthetic", False)

        def _worker():
            try:
                from pipeline_run import run_pipeline
                results = run_pipeline(
                    image_path   = img_path,
                    gt_path      = gt_path,
                    synthetic    = synth,
                    angle_hint   = lab_params.get("angle_hint"),
                    length_hint  = lab_params.get("length_hint"),
                    n_outer      = lab_params.get("n_outer", 8),
                    n_tv         = lab_params.get("n_tv", 40),
                    lam          = lab_params.get("lam", 0.02),
                    gamma        = lab_params.get("gamma", 0.01),
                    verbose=False,
                    step_callback=lambda msg: self._q.put(msg),
                )
                self._q.put({"type": "done", "results": results})
            except Exception as e:
                self._q.put({"type": "error", "message": str(e)})
        threading.Thread(target=_worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                try:
                    msg = self._q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._dispatch(msg)
                except Exception as _exc:
                    import traceback as _tb
                    err = _tb.format_exc()
                    self._running = False
                    self._sidebar.set_status("Error", RED)
                    self._set_status(f"UI error: {_exc}")
                    messagebox.showerror(
                        "PlateReveal Error",
                        f"A UI error occurred while processing a pipeline message.\n\n"
                        f"msg type: {msg.get('type')}  name: {msg.get('name')}\n\n"
                        f"{err[:1200]}")
        finally:
            self.after(60, self._poll)  # always reschedule, no matter what

    def _dispatch(self, msg: dict):
        t    = msg.get("type", "step")
        name = msg.get("name", "")

        if t == "step":
            if name == "loaded":
                arr = msg.get("image")
                if arr is not None:
                    self._last_bl = arr
                    self._pipeline_tab.step_running("blurred")
                    self._pipeline_tab.step_done("blurred", arr)
                    self._lab_tab.load_image(arr)
                    self._set_status("Loaded image — estimating blur…")
            elif name == "fft":
                self._pipeline_tab.step_running("fft")
                if msg.get("image") is not None:
                    self._pipeline_tab.step_done("fft", msg["image"])
                self._set_status("FFT spectrum computed — running Hough angle scan…")
            elif name == "cepstrum":
                ang = msg.get("angle", 0); lgt = msg.get("length", 0)
                cep_img = msg.get("cep_image")
                self._pipeline_tab.step_running("cepstrum")
                self._pipeline_tab.step_done("cepstrum", cep_img,
                                              cep_image=cep_img,
                                              angle=ang, length=lgt)
                self._set_status(f"Cepstrum: θ={ang}°  L={lgt}px — building PSF kernel…")
            elif name == "tps_start":
                self._set_status(
                    "TPS PSF map: sampling 9 patches + Thin Plate Spline fitting…")
            elif name == "kernel":
                self._pipeline_tab.step_running("kernel")
                k = msg.get("kernel")
                ang = msg.get("angle", 0); lgt = msg.get("length", 0)
                if k is not None:
                    self._pipeline_tab.step_done("kernel", None,
                                                  kernel=k, angle=ang, length=lgt)
                self._set_status(f"Kernel ready: {ang}°/{lgt}px — Wiener rough pass…")
            elif name == "wiener_start":
                self._pipeline_tab.step_running("wiener")
                self._set_status("Wiener deconvolution running… (FFT-based, per channel)")
            elif name == "wiener":
                if msg.get("image") is not None:
                    self._pipeline_tab.step_done("wiener", msg["image"])
                self._set_status("Wiener pass done — starting HQS solver…")
            elif name == "hqs_start":
                self._pipeline_tab.show_hqs_start(
                    msg.get("n_outer", 8), msg.get("n_tv", 40), delay_ms=1500)
                self._set_status(
                    f"HQS running: {msg.get('n_outer')} outer × {msg.get('n_tv')} TV iters…")
            elif name == "hqs":
                if msg.get("image") is not None:
                    self._pipeline_tab.step_done("hqs", msg["image"])
                self._set_status("HQS done — running OCR…")
            elif name == "ocr":
                self._set_status(
                    f"OCR done  Tess: {msg.get('tesseract','?')}  "
                    f"Easy: {msg.get('easyocr','?')}")

        elif t == "done":
            res = msg["results"]
            self._last_res = res
            self._running  = False
            self._sidebar.set_status("Done ✓", GREEN)
            self._set_status(
                f"Complete — {res['params']['angle']:.0f}° / {res['params']['length']}px  "
                f"· Tess: {res['ocr']['tesseract'] or '—'}")
            # Core display — always run
            if self._lab_mode:
                bl = self._last_bl or res["intermediates"]["blurred_rgb"]
                self._lab_tab.show_results(res, bl)
            else:
                self._pipeline_tab.show_results(res)
                bl = res["intermediates"]["blurred_rgb"]
                self._last_bl = bl
                self._lab_tab.show_results(res, bl)
            self._history_tab.push(res, res["intermediates"]["final_rgb"])
            # Optional extras — failures here must NOT hide the result
            try:
                self._math_tab.update_params(res)
            except Exception:
                pass
            try:
                self._hist_tab.update(res)
            except Exception:
                pass

        elif t == "error":
            self._running = False
            self._sidebar.set_status("Error", RED)
            self._set_status(f"Error: {msg['message']}")
            messagebox.showerror("Pipeline Error",
                                 msg.get("message","Unknown error") + "\n\n" +
                                 msg.get("tb",""))

    def _on_restore(self, results: dict):
        self._last_res = results
        self._pipeline_tab.show_results(results)
        interm = results["intermediates"]
        for step_id, arr in (("blurred", interm["blurred_rgb"]),
                              ("wiener",  interm["rough_rgb"]),
                              ("hqs",     interm["final_rgb"])):
            self._pipeline_tab.step_done(step_id, arr)
        self._switch_tab("Pipeline")
        self._set_status("History restored.")

    def _open_zoom(self, step_id: str, arr: np.ndarray):
        ZoomWindow(self, step_id, arr)

    def _set_status(self, msg: str):
        self._status_lbl.config(text=msg)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Pre-load this image on startup")
    args = ap.parse_args()

    app = PlateReveal(initial_image=args.image)
    app.mainloop()
