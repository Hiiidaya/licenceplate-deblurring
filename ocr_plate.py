"""
ocr_plate.py — Maximum-power classical license plate OCR
=========================================================
No neural networks.

Pipeline
--------
1. Plate crop   — isolate bright white/yellow rectangle
2. Inner crop   — remove dark frame, keep ONLY the white interior
                  (this fixes the inverted-polarity bug that made Tesseract
                  read the dark border as text instead of background)
3. Tesseract    — 8 preprocessing strategies × 5 PSM × 2 OEM  (80 attempts)
4. Classical    — VPP segmentation + multi-font HOG+NCC matching

Usage (CLI)
-----------
    python ocr_plate.py image.png
    python ocr_plate.py image.png --debug

Import
------
    from ocr_plate import recognize
    text = recognize("deblurred.png")
    text = recognize(numpy_uint8_rgb_array)
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from scipy.ndimage import (gaussian_filter, label as _slabel,
                                binary_closing, binary_dilation, uniform_filter)
    _SCI = True
except ImportError:
    _SCI = False

_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def _load(src):
    if isinstance(src, str):
        return np.array(Image.open(src).convert("RGB"))
    if isinstance(src, Image.Image):
        return np.array(src.convert("RGB"))
    arr = np.asarray(src, dtype=np.uint8)
    if arr.ndim == 2:
        return np.stack([arr]*3, axis=2)
    return arr[:, :, :3] if arr.shape[2] == 4 else arr.copy()


def _gray(rgb):
    return np.clip(
        0.2989*rgb[:,:,0].astype(float) +
        0.5870*rgb[:,:,1].astype(float) +
        0.1140*rgb[:,:,2].astype(float), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PLATE CROP  (find the bright outer rectangle)
# ══════════════════════════════════════════════════════════════════════════════

def _plate_crop(rgb, debug=False):
    """Find the license-plate rectangle, return a perspective-corrected crop."""
    if not _CV2:
        return rgb

    gray = _gray(rgb)
    img_h, img_w = gray.shape

    clahe = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    _, bright = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(img_w*0.06), 3), 3))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(int(img_h*0.04), 3)))
    closed = cv2.morphologyEx(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kh),
                              cv2.MORPH_CLOSE, kv)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        af = w * h / (img_h * img_w + 1e-5)
        ar = w / (h + 1e-5)
        if not (0.08 <= af <= 0.96) or not (1.2 <= ar <= 8.0):
            continue
        score = gray[y:y+h, x:x+w].mean() * af
        if score > best_score:
            best_score = score; best = (x, y, w, h)

    if best:
        x, y, w, h = best
        p = 6
        crop = rgb[max(0,y-p):min(img_h,y+h+p), max(0,x-p):min(img_w,x+w+p)]
        if crop.size > 0:
            if debug: print(f"  [plate_crop] ({x},{y}) {w}×{h}")
            return crop

    # Fallback: brightness-profile crop
    row_m = gray.mean(axis=1); col_m = gray.mean(axis=0)
    thr = max(gray.mean() * 1.2, 100)
    br = np.where(row_m > thr)[0]; bc = np.where(col_m > thr)[0]
    if br.size and bc.size:
        p = 6
        crop = rgb[max(0,br[0]-p):min(img_h,br[-1]+p),
                   max(0,bc[0]-p):min(img_w,bc[-1]+p)]
        if crop.size > 0:
            if debug: print(f"  [plate_crop] brightness profile fallback")
            return crop

    if debug: print("  [plate_crop] no crop found — using full image")
    return rgb


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — INNER CROP  (remove dark border, keep white interior ONLY)
#
# This is the critical fix:
#   _plate_crop returns the outer rectangle which INCLUDES the dark car frame.
#   When we binarize that, plate-background becomes white but the frame also
#   becomes white → _polarity inverts the whole thing → characters end up
#   WHITE on BLACK → Tesseract reads garbage.
#
#   After inner_crop the image is ONLY the light plate background + dark chars,
#   so Otsu correctly gives BLACK chars on WHITE background.
# ══════════════════════════════════════════════════════════════════════════════

def _inner_crop(plate_rgb, debug=False):
    """Crop away the dark border — keep only the bright plate interior."""
    gray = _gray(plate_rgb)
    h, w = gray.shape

    # Use a high fixed threshold to isolate only the bright plate background
    brightness_thresh = max(int(gray.mean() * 1.15), 120)
    bright_mask = gray > brightness_thresh

    row_sums = bright_mask.sum(axis=1)
    col_sums = bright_mask.sum(axis=0)

    # Keep rows/cols where at least 20% of pixels are bright (plate interior)
    min_frac = 0.20
    bright_rows = np.where(row_sums > w * min_frac)[0]
    bright_cols = np.where(col_sums > h * min_frac)[0]

    if bright_rows.size < 4 or bright_cols.size < 4:
        if debug: print("  [inner_crop] not enough bright rows/cols — skipping")
        return plate_rgb

    r0, r1 = int(bright_rows[0]),  int(bright_rows[-1])
    c0, c1 = int(bright_cols[0]),  int(bright_cols[-1])

    # Sanity: resulting crop must be landscape and reasonably sized
    crop_h = r1 - r0
    crop_w = c1 - c0
    if crop_w < w * 0.3 or crop_h < h * 0.2 or crop_w < crop_h:
        if debug: print("  [inner_crop] crop too small/portrait — skipping")
        return plate_rgb

    pad = 4
    crop = plate_rgb[max(0,r0-pad):min(h,r1+pad),
                     max(0,c0-pad):min(w,c1+pad)]
    if debug:
        print(f"  [inner_crop] ({c0},{r0})→({c1},{r1}), result {crop.shape[1]}×{crop.shape[0]}")
    return crop if crop.size > 0 else plate_rgb


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _upscale(arr, s):
    if _CV2:
        return cv2.resize(arr, (arr.shape[1]*s, arr.shape[0]*s),
                          interpolation=cv2.INTER_LANCZOS4)
    return np.array(Image.fromarray(arr).resize(
        (arr.shape[1]*s, arr.shape[0]*s), Image.LANCZOS))


def _clahe(g, clip=3.0, tile=8):
    if _CV2:
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile)).apply(g)
    hist, _ = np.histogram(g.ravel(), 256, [0,256])
    cdf = hist.cumsum().astype(float); m = cdf[cdf>0][0]
    lut = np.round((cdf-m)/(g.size-m+1e-10)*255).clip(0,255).astype(np.uint8)
    return lut[g]


def _nlm(g, h=20):
    return cv2.fastNlMeansDenoising(g, h=h, templateWindowSize=7,
                                    searchWindowSize=21) if _CV2 else g


def _gauss(g, k):
    if _CV2: return cv2.GaussianBlur(g, (k, k), 0)
    if _SCI: return gaussian_filter(g.astype(float), k/3).clip(0,255).astype(np.uint8)
    return g


def _unsharp(g, sigma=2, amount=2.0):
    if _CV2:
        b = cv2.GaussianBlur(g, (0,0), sigma)
        return cv2.addWeighted(g, 1+amount, b, -amount, 0)
    return g


def _otsu(g):
    if _CV2:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b
    hist, bins = np.histogram(g.ravel(), 256, [0,256])
    h = hist.astype(float)/(hist.sum()+1e-10); c=(bins[:-1]+bins[1:])/2
    cw=np.cumsum(h); cm=np.cumsum(h*c); tm=cm[-1]; bv,bt=-1.0,128.0
    for t in range(1,len(h)-1):
        w0,w1=cw[t],1-cw[t]
        if w0<1e-6 or w1<1e-6: continue
        v=w0*w1*(cm[t]/w0-(tm-cm[t])/w1)**2
        if v>bv: bv,bt=v,c[t]
    return ((g>=bt*255)*255).astype(np.uint8)


def _adaptive(g, block=31, c=10):
    if _CV2:
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, c)
    if _SCI:
        local = uniform_filter(g.astype(float), size=block)
        return ((g.astype(float) > local-c)*255).astype(np.uint8)
    return _otsu(g)


def _morph_open(b, k=3):
    if _CV2:
        return cv2.morphologyEx(b, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    if _SCI:
        s = np.ones((k,k),bool)
        closed = binary_closing(b>128, s)  # open = erode+dilate, approx via close complement
        return (closed*255).astype(np.uint8)
    return b


def _morph_close(b, k=3):
    if _CV2:
        return cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    if _SCI:
        return (binary_closing(b>128, np.ones((k,k),bool))*255).astype(np.uint8)
    return b


def _ensure_dark_chars(binary):
    """Ensure characters are DARK (0) on LIGHT (255) background.
    Count dark pixels — if more dark than light, the image is inverted."""
    dark_px = (binary < 128).sum()
    light_px = (binary >= 128).sum()
    # Characters should be the MINORITY (< 50% of pixels)
    return 255 - binary if dark_px > light_px else binary


def _add_border(arr, pad=20):
    """Add white border — Tesseract performs better with padding."""
    if _CV2:
        return cv2.copyMakeBorder(arr, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=255)
    return np.pad(arr, pad, mode='constant', constant_values=255)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING STRATEGY FACTORY
# All strategies produce: BLACK chars on WHITE background + white border
# ══════════════════════════════════════════════════════════════════════════════

def _make_variants(interior_rgb, scale=4):
    """Return list of (name, PIL.Image) ready for Tesseract."""
    g = _gray(interior_rgb)
    big = _upscale(g, scale)

    def _finalize(b):
        b = _ensure_dark_chars(b)
        b = _add_border(b, 20)
        return Image.fromarray(b)

    variants = []
    fns = {
        # Key insight: blur first to kill ringing → then Otsu
        "blur9+otsu":       lambda g: _morph_close(_morph_open(_otsu(_gauss(g, 9)), 3), 5),
        "blur7+otsu":       lambda g: _morph_close(_morph_open(_otsu(_gauss(g, 7)), 2), 4),
        "nlm20+otsu":       lambda g: _morph_close(_morph_open(_otsu(_nlm(g, 20)), 3), 5),
        "nlm30+otsu":       lambda g: _morph_close(_morph_open(_otsu(_nlm(g, 30)), 3), 5),
        "clahe+blur7+otsu": lambda g: _morph_close(_morph_open(_otsu(_gauss(_clahe(g), 7)), 2), 4),
        "clahe+nlm20+otsu": lambda g: _morph_close(_morph_open(_otsu(_nlm(_clahe(g), 20)), 3), 5),
        "adaptive31":       lambda g: _morph_close(_morph_open(_adaptive(_nlm(g,15), 31, 8), 2), 3),
        "grayscale":        lambda g: g,   # let Tesseract's internal Sauvola handle it
    }

    for name, fn in fns.items():
        try:
            arr = fn(big)
            variants.append((name, _finalize(arr)))
        except Exception:
            pass

    return variants


# ══════════════════════════════════════════════════════════════════════════════
# TESSERACT ENGINE  (primary)
# ══════════════════════════════════════════════════════════════════════════════

_WL  = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789 "
_PSM = (6, 7, 3, 11, 13)
_OEM = (3, 0)


def _setup_tess():
    import pytesseract
    if os.name == "nt":
        for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p; break
    return pytesseract


def _clean(raw):
    return ''.join(c for c in raw.upper() if c in _CHARS)


def ocr_tesseract(interior_rgb, debug=False):
    """Sweep all variants × PSM × OEM. Return longest valid result."""
    try:
        tess = _setup_tess()
    except ImportError:
        return None

    variants = _make_variants(interior_rgb, scale=4)
    best_text, best_score = "", 0

    for name, pil in variants:
        for oem in _OEM:
            for psm in _PSM:
                cfg = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={_WL}"
                try:
                    text = _clean(tess.image_to_string(pil, config=cfg))
                    s    = len(text)
                    if s > best_score:
                        best_score, best_text = s, text
                        if debug:
                            print(f"  [tess] {name:25s} oem={oem} psm={psm} → '{text}'")
                except Exception:
                    pass

    return best_text if best_score >= 2 else None


# ══════════════════════════════════════════════════════════════════════════════
# VPP SEGMENTATION  (classical fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _proj(binary, axis):
    return (binary > 128).sum(axis=axis).astype(float)


def _smooth(p, s=2.0):
    if _SCI: return gaussian_filter(p, s)
    k = max(int(s*2),1)
    return np.convolve(p, np.ones(k)/k, mode='same')


def _split_rows(binary):
    h = binary.shape[0]
    hp = _smooth(_proj(binary, axis=1), 1.5)
    thr = hp.max() * 0.08
    rows, in_r, r0 = [], False, 0
    for r, v in enumerate(hp):
        if not in_r and v > thr: r0=r; in_r=True
        elif in_r and v <= thr:
            if r-r0 >= h*0.12: rows.append((max(0,r0-2), min(h,r+2)))
            in_r = False
    if in_r and h-r0 >= h*0.12: rows.append((max(0,r0-2), h))
    return rows or [(0, h)]


def _vpp_chars(row_bin):
    vp  = _smooth(_proj(row_bin, axis=0), 1.5)
    thr = max(vp.max()*0.06, 1.0)
    w   = row_bin.shape[1]
    chars, in_c, c0 = [], False, 0
    for c, v in enumerate(vp):
        if not in_c and v > thr: c0=c; in_c=True
        elif in_c and v <= thr:
            if c-c0 >= 3: chars.append((max(0,c0-1), min(w,c+1)))
            in_c = False
    if in_c and w-c0 >= 3: chars.append((max(0,c0-1), w))
    if not chars: return []
    widths = [c1-c0 for c0,c1 in chars]; mw=sorted(widths)[len(widths)//2]
    merged, i = [], 0
    while i < len(chars):
        c0,c1 = chars[i]
        while i+1 < len(chars) and (c1-c0) < 0.28*mw: i+=1; c1=chars[i][1]
        merged.append((c0,c1)); i+=1
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# HOG + TEMPLATE BANK  (classical fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _hog(gray, ch=8, cw=6, nb=9):
    f = gray.astype(float)
    if _CV2:
        gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    else:
        gx = np.gradient(f, axis=1); gy = np.gradient(f, axis=0)
    mag = np.sqrt(gx**2+gy**2)
    ang = (np.degrees(np.arctan2(gy,gx))%180).astype(float)
    bw  = 180.0/nb
    feats = []
    for r in range(0, gray.shape[0], ch):
        for c in range(0, gray.shape[1], cw):
            m=mag[r:r+ch,c:c+cw].ravel(); a=ang[r:r+ch,c:c+cw].ravel()
            hist=np.zeros(nb)
            for mi,ai in zip(m,a): hist[int(ai/bw)%nb]+=mi
            feats.append(hist)
    v=np.concatenate(feats); return v/(np.linalg.norm(v)+1e-10)


def _build_bank(ch=40, cw=24):
    fonts = []
    # Heavy fonts first — best match for thick licence-plate strokes
    for name in ("ariblk.ttf", "impact.ttf", "tahomabd.ttf", "verdanab.ttf",
                 "arialbd.ttf", "GOTHICB.TTF", "calibrib.ttf", "consolab.ttf",
                 "courbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try: fonts.append(ImageFont.truetype(name, int(ch * 0.85)))
        except: pass
    if not fonts: fonts.append(ImageFont.load_default())
    bank = {c: [] for c in _CHARS}
    for font in fonts:
        for c in _CHARS:
            canvas = Image.new("L", (cw*3, ch*3), 255)
            ImageDraw.Draw(canvas).text((4, 4), c, fill=0, font=font)
            arr = np.array(canvas)
            rows, cols = np.any(arr < 128, axis=1), np.any(arr < 128, axis=0)
            if rows.any() and cols.any():
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                arr = arr[r0:r1+1, c0:c1+1]
            resized = np.array(Image.fromarray(arr).resize((cw, ch), Image.LANCZOS), float)
            norm = 1.0 - resized / 255.0
            bank[c].append((norm, _hog(resized.astype(np.uint8), ch//5, cw//4)))
            # Blurred variant — matches blurry plate images where HOG edges are soft
            if _SCI:
                from scipy.ndimage import gaussian_filter
                blurred = gaussian_filter(resized, sigma=2.5)
                norm_b = 1.0 - blurred / 255.0
                bank[c].append((norm_b, _hog(blurred.astype(np.uint8), ch//5, cw//4)))
    return bank, ch, cw


def _match(crop, bank, ch, cw):
    r=np.array(Image.fromarray(crop).resize((cw,ch),Image.LANCZOS),float)
    n=1.0-r/255.0; hg=_hog(r.astype(np.uint8),ch//5,cw//4)
    best_c,best_s='?',-2.0
    for c,tmps in bank.items():
        s=max(0.55*float(np.sum((n-n.mean())*(t-t.mean()))/(np.linalg.norm(n-n.mean())*np.linalg.norm(t-t.mean())+1e-10))
              +0.45*float(np.dot(hg,hv)/(np.linalg.norm(hg)*np.linalg.norm(hv)+1e-10))
              for t,hv,*_ in tmps)
        if s>best_s: best_s,best_c=s,c
    return best_c,best_s


def _best_binary_for_classical(gray_big):
    """Pick the preprocessing that gives char density closest to 20%."""
    fns = [
        lambda g: _morph_close(_morph_open(_otsu(_gauss(g,9)),3),5),
        lambda g: _morph_close(_morph_open(_otsu(_nlm(g,20)),3),5),
        lambda g: _morph_close(_morph_open(_adaptive(_nlm(g,15),31,8),2),3),
    ]
    best = None; best_d = 999
    for fn in fns:
        try:
            b = fn(gray_big)
            b = _ensure_dark_chars(b)
            d = (b < 128).mean()     # fraction of dark (char) pixels
            if abs(d - 0.20) < best_d:
                best_d = abs(d-0.20); best = b
        except: pass
    return best if best is not None else _ensure_dark_chars(_otsu(_clahe(gray_big)))


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX MATCHING ENGINE
# Each character blob → 40×40 binary matrix of 0s and 1s.
# Compare against reference matrices for every A-Z and 0-9.
# The character whose reference matrix has the highest pixel overlap wins.
# ══════════════════════════════════════════════════════════════════════════════

_MATRIX_H = 40
_MATRIX_W  = 32

def _render_one(ch, font, dilation_px=0):
    """Render one character, tight-crop, return binary ink mask (1=ink, 0=bg)."""
    canvas = Image.new("L", (120, 120), 255)
    ImageDraw.Draw(canvas).text((10, 10), ch, fill=0, font=font)
    arr = np.array(canvas)
    rows = np.any(arr < 128, axis=1); cols = np.any(arr < 128, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    arr = arr[r0:r1+1, c0:c1+1]
    ink = (arr < 128).astype(np.uint8)
    if dilation_px > 0 and _CV2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px*2+1,)*2)
        ink = cv2.dilate(ink, k)
    return ink.astype(np.float32)


def _ink_to_dt(ink_float):
    """Convert ink mask to distance-transform map, normalised to [0,1].
    Each pixel value = distance to nearest background pixel (normalised).
    This gives a 'stroke density' map: thick stroke centres are bright,
    edges are dim, and background is zero — highly discriminative."""
    ink_u8 = (ink_float > 0.5).astype(np.uint8)
    if _CV2:
        dt = cv2.distanceTransform(ink_u8, cv2.DIST_L2, 5)
    else:
        from scipy.ndimage import distance_transform_edt
        dt = distance_transform_edt(ink_u8).astype(np.float32)
    mx = dt.max()
    return (dt / mx if mx > 0 else dt).astype(np.float32)


def _to_matrix(ink_float):
    """Resize binary ink mask to _MATRIX_H × _MATRIX_W preserving aspect ratio."""
    h_in, w_in = ink_float.shape
    scale = min(_MATRIX_H / (h_in+1e-5), _MATRIX_W / (w_in+1e-5))
    new_h = max(1, int(h_in * scale))
    new_w = max(1, int(w_in * scale))
    arr = (ink_float * 255).astype(np.uint8)
    if _CV2:
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS))
    matrix = np.zeros((_MATRIX_H, _MATRIX_W), dtype=np.float32)
    y_off = (_MATRIX_H - new_h) // 2
    x_off = (_MATRIX_W - new_w) // 2
    matrix[y_off:y_off+new_h, x_off:x_off+new_w] = (resized > 64).astype(np.float32)
    return matrix


def _build_reference_matrices():
    """Build reference matrices for A–Z / 0–9.
    Each character gets 3 variants:
      • thin (dilation 0) — matches clean plates
      • medium (dilation 2) — matches typical deblurred plates
      • thick (dilation 4) — matches heavily blurred/blobby chars
    Multiple bold fonts are used to cover licence-plate typeface variations.
    Returns dict: char → list of _MATRIX_H × _MATRIX_W float arrays."""
    refs = {c: [] for c in _CHARS}

    fonts = []
    # Heavy/black fonts first — closest to licence-plate stroke weight
    for name in ("ariblk.ttf", "impact.ttf", "tahomabd.ttf", "verdanab.ttf",
                 "arialbd.ttf", "Arial Bold.ttf", "calibrib.ttf",
                 "consolab.ttf", "courbd.ttf",
                 "DejaVuSans-Bold.ttf", "FreeSansBold.ttf",
                 "LiberationSans-Bold.ttf",
                 "arial.ttf", "DejaVuSans.ttf"):
        try:
            fonts.append(ImageFont.truetype(name, 72))
        except Exception:
            pass
    if not fonts:
        fonts.append(ImageFont.load_default())

    for font in fonts:
        for ch in _CHARS:
            # Only dilation 0 — heavy fonts already have thick strokes;
            # dilation on top over-fills and loses character shape
            ink = _render_one(ch, font, dilation_px=0)
            if ink is None:
                continue
            refs[ch].append(_to_matrix(ink))

    return refs


def _thin(ink_mask):
    """Zhang-Suen thinning: reduce thick strokes to 1-pixel skeleton.
    Works on a boolean array where True = ink.
    Makes matching font-independent (skeleton shape = character identity)."""
    img = ink_mask.astype(np.uint8)
    prev = np.zeros_like(img)
    while True:
        # Sub-iteration 1
        marked = np.zeros_like(img)
        h, w = img.shape
        for _ in range(2):  # two sub-iterations
            p2 = np.roll(img, -1, 0); p4 = np.roll(img, -1, 1)
            p6 = np.roll(img,  1, 0); p8 = np.roll(img,  1, 1)
            p3 = np.roll(p2, -1, 1);  p7 = np.roll(p6,  1, 1)
            p9 = np.roll(p6, -1, 1);  p5 = np.roll(p2,  1, 1)
            # Neighbours
            nb = p2+p3+p4+p5+p6+p7+p8+p9
            # Transitions 0→1 in (p2,p3,p4,p5,p6,p7,p8,p9,p2)
            stacked = np.stack([p2,p3,p4,p5,p6,p7,p8,p9,p2],axis=0)
            trans = ((stacked[:-1]==0)&(stacked[1:]==1)).sum(axis=0)
            cond1 = (nb >= 2) & (nb <= 6)
            cond2 = trans == 1
            if _ == 0:
                cond3 = (p2*p4*p6 == 0)
                cond4 = (p4*p6*p8 == 0)
            else:
                cond3 = (p2*p4*p8 == 0)
                cond4 = (p2*p6*p8 == 0)
            marked |= (img==1) & cond1 & cond2 & cond3 & cond4
        img = img & ~marked
        if np.array_equal(img, prev):
            break
        prev = img.copy()
    return img.astype(bool)


def _blob_to_matrix(blob_binary, use_skeleton=True):
    """Convert a character blob to a (_MATRIX_H × _MATRIX_W) float matrix.
    blob_binary: dark (0) chars on white (255) background.
    With use_skeleton=True the strokes are thinned to 1px before comparison —
    this makes matching font-independent."""
    ink = blob_binary < 128
    if not ink.any():
        return np.zeros((_MATRIX_H, _MATRIX_W), dtype=np.float32)

    rows = np.any(ink, axis=1); cols = np.any(ink, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    cropped = ink[r0:r1+1, c0:c1+1]

    if use_skeleton:
        cropped = _thin(cropped)

    return _to_matrix(cropped.astype(np.float32))


def _vec_ncc(a, b):
    """Normalized cross-correlation of two 1D vectors → [0, 1]."""
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / denom * 0.5 + 0.5)


def _edge_matrix(m):
    """Morphological boundary = dilation − erosion (captures stroke outlines)."""
    if _CV2:
        k = np.ones((3, 3), np.uint8)
        u8 = (m * 255).astype(np.uint8)
        edge = cv2.dilate(u8, k).astype(float) - cv2.erode(u8, k).astype(float)
        return np.clip(edge / 255.0, 0, 1).astype(np.float32)
    # Fallback without cv2
    from scipy.ndimage import binary_dilation, binary_erosion
    ink = m > 0.5
    return (binary_dilation(ink) ^ binary_erosion(ink)).astype(np.float32)


def _hu_moments(m):
    """Compute log-scale Hu moments from a binary matrix (shape descriptor)."""
    if not _CV2:
        return np.zeros(7)
    u8 = (m * 255).astype(np.uint8)
    mom = cv2.moments(u8)
    hu = cv2.HuMoments(mom).flatten()
    # Log-scale (standard for Hu moments comparison)
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def _hu_similarity(blob_m, ref_m):
    """Hu-moment similarity: 1 = identical shape, 0 = very different."""
    h1 = _hu_moments(blob_m)
    h2 = _hu_moments(ref_m)
    dist = np.sum(np.abs(h1 - h2))
    return float(np.exp(-dist / 3.0))   # scale so ~same char → ~0.8+


def _matrix_score(blob_m, ref_m):
    """Multi-feature binary matrix similarity:
    - IoU + NCC on binary shapes
    - Projection profiles (HPP + VPP)
    - Hu moments (rotation/scale-invariant shape descriptor)
    """
    # IoU
    intersection = (blob_m * ref_m).sum()
    union = np.clip(blob_m + ref_m, 0, 1).sum()
    iou = float(intersection / (union + 1e-10))

    # 2D NCC
    a = blob_m - blob_m.mean(); b = ref_m - ref_m.mean()
    ncc = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # Projection profiles
    hpp = _vec_ncc(blob_m.sum(axis=1), ref_m.sum(axis=1))
    vpp = _vec_ncc(blob_m.sum(axis=0), ref_m.sum(axis=0))

    # Hu moments
    hu_sim = _hu_similarity(blob_m, ref_m)

    return (0.25 * iou
            + 0.20 * ((ncc + 1) / 2)
            + 0.15 * hpp
            + 0.15 * vpp
            + 0.25 * hu_sim)


# Characters with exactly 1 enclosed interior region (topological hole)
_ONE_HOLE  = set('0DGOQCBRS69')
# Characters with 2 enclosed interior regions
_TWO_HOLES = set('8B')
# Characters with 0 holes (open)
_NO_HOLES  = set('1234567AEFHJKLMNPTUVWXYZ')


def _count_holes(blob_binary):
    """Count enclosed white regions inside a character blob.
    blob_binary: dark (0) chars on white (255) background.
    Returns integer: 0 = open char (M, H, 4...), 1 = single hole (0, D, 6...), 2+ = (8, B...).

    Noise-robust: applies morphological closing to fill tiny gaps before counting,
    and ignores holes smaller than 1% of the blob area."""
    if not _CV2:
        return 0
    h, w = blob_binary.shape
    # 10% threshold: detects large interior loops (0, 9, W-gap) while
    # ignoring M's V-notches. D gets holes=0 due to large close_k filling it.
    min_hole_px = max(4, int(h * w * 0.10))

    # Close small noise gaps in the ink (fill tiny holes in strokes)
    ink = (blob_binary < 128).astype(np.uint8)
    close_k = max(3, min(h, w) // 15)
    close_k = close_k if close_k % 2 == 1 else close_k + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kern)

    bg_mask = (1 - ink_closed).astype(np.uint8)   # 1=white (background+holes)
    padded = np.zeros((h+2, w+2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = bg_mask
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    holes_mask = (flood[1:h+1, 1:w+1] == 1).astype(np.uint8)

    if holes_mask.sum() < min_hole_px:
        return 0

    # Count significant connected components in holes_mask
    n_labels, labels = cv2.connectedComponents(holes_mask)
    count = 0
    for lbl in range(1, n_labels):
        if (labels == lbl).sum() >= min_hole_px:
            count += 1
    return count


def _extract_blobs(binary, debug=False):
    """Connected-component blob extraction.
    binary: dark (0) chars on white (255) background.

    Key technique: pad the binary with white rows/cols before CC analysis so
    that chars touching the image border are NOT merged with border artifacts.
    Wide blobs (merged chars) are returned as-is; caller splits them via VPP.
    """
    img_h, img_w = binary.shape

    # Pad with white border so chars never touch image edge → no border merging
    PAD = 4
    padded = np.full((img_h + 2*PAD, img_w + 2*PAD), 255, dtype=np.uint8)
    padded[PAD:PAD+img_h, PAD:PAD+img_w] = binary

    inv = 255 - padded   # chars become white
    boxes = []

    if _CV2:
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        inv_d = cv2.dilate(inv, kern, iterations=1)
        contours, _ = cv2.findContours(inv_d, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Convert back to un-padded coordinates
            x -= PAD; y -= PAD
            # Height: 10% – 96% of original image height
            if h < img_h * 0.10 or h > img_h * 0.96:
                continue
            if w < img_w * 0.02:          # too narrow = noise
                continue
            if w > img_w * 0.95:          # full-width = frame artifact
                continue
            if w / (h + 1e-5) > 6.0:     # extreme horizontal line = noise
                continue
            if (inv_d[y+PAD:y+PAD+h, x+PAD:x+PAD+w] > 128).mean() < 0.05:
                continue
            boxes.append((x, y, x+w, y+h))
    else:
        labeled, n = _slabel(inv > 128)
        for cid in range(1, n+1):
            ys, xs = np.where(labeled == cid)
            r0, r1 = int(ys.min()) - PAD, int(ys.max()) - PAD
            c0, c1 = int(xs.min()) - PAD, int(xs.max()) - PAD
            h_c, w_c = r1-r0+1, c1-c0+1
            if not (img_h*0.10 <= h_c <= img_h*0.96): continue
            if w_c < img_w*0.02 or w_c > img_w*0.95: continue
            if w_c / (h_c + 1e-5) > 6.0: continue
            boxes.append((c0, r0, c1, r1))

    # Sort into reading order: row by row, left to right
    if not boxes:
        if debug: print("  [blobs] no blobs found after filtering")
        return []
    heights = [b[3]-b[1]+1 for b in boxes]
    med_h = sorted(heights)[len(heights)//2]
    rows = []
    for box in sorted(boxes, key=lambda b: (b[1]+b[3])/2):
        cy = (box[1]+box[3])/2
        placed = False
        for row in rows:
            row_cy = sum((b[1]+b[3])/2 for b in row) / len(row)
            if abs(cy - row_cy) < 0.65 * med_h:
                row.append(box); placed = True; break
        if not placed:
            rows.append([box])
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    if debug:
        print(f"  [blobs] found {len(ordered)} character blobs in {len(rows)} row(s)")
    return ordered


def _split_wide_blob(blob_binary, expected_n=2):
    """Split a wide blob (merged chars) into expected_n sub-spans using VPP valleys.
    Returns list of (col_start, col_end) spans relative to blob."""
    h, w = blob_binary.shape
    if expected_n <= 1:
        return [(0, w)]

    # Vertical projection profile of ink pixels
    ink_col = (blob_binary < 128).sum(axis=0).astype(float)
    ink_col = _smooth(ink_col, 3.0)

    if expected_n == 2:
        # Find the deepest valley in the middle half
        ms, me = w // 4, 3 * w // 4
        v_idx = int(np.argmin(ink_col[ms:me])) + ms
        return [(0, v_idx), (v_idx, w)]

    # For N chars: find N-1 deepest valleys across the whole profile
    # Use equal-interval search: divide into expected_n bands, find valley in each gap
    unit = w / expected_n
    splits = []
    for i in range(1, expected_n):
        band_s = max(0, int(unit * i - unit * 0.4))
        band_e = min(w, int(unit * i + unit * 0.4))
        if band_e <= band_s:
            splits.append(int(unit * i))
        else:
            splits.append(int(np.argmin(ink_col[band_s:band_e])) + band_s)

    spans = []
    prev = 0
    for sp in sorted(splits):
        if sp > prev:
            spans.append((prev, sp))
        prev = sp
    spans.append((prev, w))
    return spans if len(spans) >= 2 else [(0, w)]


def ocr_by_matrix(interior_rgb, debug=False):
    """
    Character recognition using HOG + NCC on individual character blobs.
      1. Upscale and binarize the plate interior
      2. Extract character blobs via padded connected-components
      3. Wide blobs (merged chars) are split using VPP
      4. Each blob grey-image matched against HOG + NCC reference bank
      7. The character with the highest score wins
    """
    gray   = _gray(interior_rgb)
    big    = _upscale(gray, 4)

    binary = _morph_close(_morph_open(_otsu(_nlm(big, 15)), 2), 3)
    binary = _ensure_dark_chars(binary)

    img_h, img_w = binary.shape

    boxes = _extract_blobs(binary, debug=debug)
    if not boxes:
        # Single large blob (plate border merging with chars) — trim outer edge and retry
        for _trim_pct in (0.03, 0.05, 0.07, 0.10):
            _m = max(5, int(_trim_pct * min(img_h, img_w)))
            _btrim = binary.copy()
            _btrim[:_m, :] = 255; _btrim[-_m:, :] = 255
            _btrim[:, :_m] = 255; _btrim[:, -_m:] = 255
            _boxes_try = _extract_blobs(_btrim, debug=debug)
            # Filter out residual border artifacts (blobs > 40% of image width)
            _boxes_try = [(c0,r0,c1,r1) for (c0,r0,c1,r1) in _boxes_try
                          if (c1-c0) < img_w * 0.40]
            if _boxes_try:
                boxes = _boxes_try
                binary = _btrim
                break
    if not boxes:
        if debug: print("  [matrix] no blobs found")
        return "?"

    # Use plate-width / 12 as single-char-width estimate.
    # Using median(blob_widths) over-estimates when chars are merged.
    unit_w = img_w // 12

    # Build HOG reference bank (used for all blobs)
    bank, bch, bcw = _build_bank(ch=40, cw=24)

    # Filter out corner-decoration blobs: much smaller than median height
    heights = [r1-r0 for (c0,r0,c1,r1) in boxes]
    med_h = sorted(heights)[len(heights)//2]
    boxes = [(c0,r0,c1,r1) for (c0,r0,c1,r1) in boxes if (r1-r0) >= med_h * 0.5]

    result = ""
    for (c0, r0, c1, r1) in boxes:
        blob_w = c1 - c0

        # Split if blob is wider than ~1.4× estimated single-char width
        sub_spans = [(0, blob_w)]
        if blob_w > unit_w * 1.4:
            sub_spans = _split_wide_blob(binary[r0:r1, c0:c1],
                                         expected_n=max(1, round(blob_w / unit_w)))
            if debug and len(sub_spans) > 1:
                print(f"  [matrix] blob ({c0},{r0})→({c1},{r1}) width={blob_w} unit={unit_w} "
                      f"→ split into {len(sub_spans)} sub-blobs")

        for (sc0, sc1) in sub_spans:
            sub_gray = big[r0:r1, c0+sc0:c0+sc1]
            sub_bin  = binary[r0:r1, c0+sc0:c0+sc1]
            n_holes  = _count_holes(sub_bin)

            from PIL import Image as _PILImg

            def _ncc1d(a, b):
                da = a - a.mean(); db = b - b.mean()
                return float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-8))

            # HOG + NCC match (gray appearance)
            r   = np.array(_PILImg.fromarray(sub_gray).resize((bcw, bch), _PILImg.LANCZOS), float)
            n_g = 1.0 - r / 255.0
            hg  = _hog(r.astype(np.uint8), bch // 5, bcw // 4)
            blob_hpp = n_g.sum(axis=1)

            all_scores = {}
            for ch_cand, tmps in bank.items():
                all_scores[ch_cand] = max(
                    0.55 * float(np.sum((n_g - n_g.mean()) * (t - t.mean())) /
                                 (np.linalg.norm(n_g - n_g.mean()) * np.linalg.norm(t - t.mean()) + 1e-10))
                    + 0.45 * float(np.dot(hg, hv) / (np.linalg.norm(hg) * np.linalg.norm(hv) + 1e-10))
                    for t, hv in tmps)

            # When exactly 1 topological hole is detected, the blob is a closed-loop
            # char (0, 6, 9, D, W...). HPP (horizontal projection) strongly discriminates
            # these from open chars (4, F, etc.). Also penalize 2-hole chars (8, B)
            # since they shouldn't appear in a 1-hole context.
            if n_holes == 1:
                for ch_cand, tmps in bank.items():
                    hpp_score = max(_ncc1d(blob_hpp, t.sum(axis=1)) for t, hv in tmps)
                    adjusted = 0.65 * all_scores[ch_cand] + 0.35 * hpp_score
                    if ch_cand in _TWO_HOLES:
                        adjusted -= 0.05
                    all_scores[ch_cand] = adjusted

            best_ch = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_ch]

            result += best_ch
            if debug:
                abs_c0 = c0 + sc0; abs_c1 = c0 + sc1
                print(f"  [matrix] blob ({abs_c0},{r0})→({abs_c1},{r1})  holes={n_holes}"
                      f"  → '{best_ch}'  score={best_score:.3f}")

    return result


def ocr_classical(interior_rgb, debug=False):
    """Wrapper: runs the matrix-matching engine (primary classical method)."""
    return ocr_by_matrix(interior_rgb, debug=debug)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def recognize(src, debug=False, save_debug_prefix=None):
    """
    Recognize a license plate.

    Parameters
    ----------
    src                : str path | PIL.Image | numpy uint8 array
    debug              : print intermediate scores
    save_debug_prefix  : if set, saves plate_crop and inner_crop as PNG files

    Returns
    -------
    str — uppercase alphanumeric plate text
    """
    rgb = _load(src)

    # Step 1 — find outer plate rectangle
    plate = _plate_crop(rgb, debug=debug)

    # Step 2 — crop away dark border, keep bright interior ONLY
    interior = _inner_crop(plate, debug=debug)

    if save_debug_prefix:
        Image.fromarray(plate).save(f"{save_debug_prefix}_plate.png")
        Image.fromarray(interior).save(f"{save_debug_prefix}_interior.png")
        if debug:
            print(f"  Saved {save_debug_prefix}_plate.png  and  _interior.png")

    if debug:
        print(f"  Interior size: {interior.shape[1]}×{interior.shape[0]}")

    # Step 3 — Tesseract (primary)
    try:
        text = ocr_tesseract(interior, debug=debug)
        if text and len(text) >= 2:
            if debug: print(f"\n  [WINNER — tesseract] '{text}'")
            return text
    except Exception as e:
        if debug: print(f"  [tesseract error: {e}]")

    # Step 4 — classical VPP + HOG (fallback)
    text = ocr_classical(interior, debug=debug)
    if debug: print(f"\n  [WINNER — classical] '{text}'")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_plate.py image.png [--debug]")
        sys.exit(1)
    path  = sys.argv[1]
    debug = "--debug" in sys.argv
    if not os.path.exists(path):
        print(f"File not found: {path}"); sys.exit(1)
    print(f"\nProcessing: {path}")
    result = recognize(path, debug=debug, save_debug_prefix="ocr_debug")
    print(f"\n{'═'*42}\n  Plate :  {result}\n{'═'*42}\n")