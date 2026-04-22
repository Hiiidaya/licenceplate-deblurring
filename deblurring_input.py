"""
License Plate Deblurring — Real Image Input
============================================
Dependencies: numpy, scipy, matplotlib, pillow

Takes a real blurred image and deblurs it. No ground truth needed.
All algorithms identical to deblurring_full.py — angle/length are
estimated from the blurred image itself, not from a known kernel.

Usage:
    python deblurring_input.py blurred.png
    python deblurring_input.py blurred.png out.png
    python deblurring_input.py blurred.png out.png 300     # custom iterations
    python deblurring_input.py blurred.png out.png 300 35 20  # override angle/length
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import gaussian_filter, label, binary_dilation
from scipy.interpolate import RBFInterpolator


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1A — ANGLE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def fft_magnitude_map(gray):
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2(gray * win))
    log_mag = np.log1p(np.abs(F))
    log_mag -= log_mag.min()
    log_mag /= (log_mag.max() + 1e-10)
    return log_mag


def hough_on_fft(gray, n_angles=360):
    log_mag = fft_magnitude_map(gray)
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fft2(gray * win)
    log_mag_cep = np.log(np.maximum(np.abs(F), 1e-3))
    cep = np.fft.fftshift(np.fft.ifft2(log_mag_cep).real)

    cx, cy = w // 2, h // 2
    r_min = 4
    r_max = max(min(h, w) // 2 - 4, r_min + 2)

    # Patch too small to do a meaningful Hough scan — return neutral angle
    if r_max <= r_min:
        return 0, log_mag

    angles = np.linspace(0, 180, n_angles, endpoint=False)
    best_score = 0.0
    best_angle = 0

    for ang in angles:
        rad = np.deg2rad(ang)
        ca, sa = np.cos(rad), np.sin(rad)
        rs = np.arange(r_min, r_max)
        if len(rs) == 0:
            continue
        prof = np.zeros(len(rs))
        for i, r in enumerate(rs):
            vals = []
            for sign in (1, -1):
                x = cx + int(round(sign * r * ca))
                y = cy + int(round(sign * r * sa))
                if 0 <= x < w and 0 <= y < h:
                    vals.append(cep[y, x])
            prof[i] = np.mean(vals) if vals else 0.0
        prof_sm = gaussian_filter(prof, 1.5)
        if prof_sm.size == 0:
            continue
        min_val = prof_sm.min()
        if min_val < best_score:
            best_score = min_val
            best_angle = int(round(ang)) % 180

    return best_angle, log_mag


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1B — LENGTH ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cepstrum(gray):
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fft2(gray * win)
    log_mag = np.log(np.maximum(np.abs(F), 1e-3))
    cep = np.fft.fftshift(np.fft.ifft2(log_mag).real)
    return cep


def cep_profile(cep, angle_deg, r_min, r_max):
    h, w = cep.shape
    cx, cy = w // 2, h // 2
    rad = np.deg2rad(angle_deg)
    ca, sa = np.cos(rad), np.sin(rad)
    rs = np.arange(r_min, r_max)
    profile = np.zeros(len(rs))
    for i, r in enumerate(rs):
        vals = []
        for sign in (1, -1):
            for frac in np.linspace(0.8, 1.2, 5):
                fx = cx + sign * r * frac * ca
                fy = cy + sign * r * frac * sa
                x0, y0 = int(np.floor(fx)), int(np.floor(fy))
                dx, dy = fx - x0, fy - y0
                acc, wsum = 0.0, 0.0
                for xi, wx in ((x0, 1 - dx), (x0 + 1, dx)):
                    for yi, wy in ((y0, 1 - dy), (y0 + 1, dy)):
                        if 0 <= xi < w and 0 <= yi < h:
                            acc += wx * wy * cep[yi, xi]
                            wsum += wx * wy
                if wsum > 1e-6:
                    vals.append(acc / wsum)
        profile[i] = np.mean(vals) if vals else 0.0
    return rs, profile


def estimate_length_cepstrum(gray, angle_deg, r_min=6, r_max=None):
    h, w = gray.shape
    if r_max is None:
        r_max = min(h, w) // 2 - 2
    cep = compute_cepstrum(gray)
    rs, prof = cep_profile(cep, float(angle_deg), r_min, r_max)
    prof_sm = gaussian_filter(prof, sigma=1.5)
    est_length = int(rs[np.argmin(prof_sm)])
    return max(est_length, 3), cep


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1C — LOCAL PSF VIA THIN PLATE SPLINE
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_local_psf_params(gray, global_angle=None):
    h, w = gray.shape
    pad = min(h, w) // 6
    rows = [pad, h // 2, h - pad]
    cols = [pad, w // 2, w - pad]
    sample_points = [(r, c) for r in rows for c in cols]

    results = []
    for (r, c) in sample_points:
        ph = min(h // 3, 80)
        pw = min(w // 3, 120)
        r0, r1 = max(0, r - ph // 2), min(h, r + ph // 2)
        c0, c1 = max(0, c - pw // 2), min(w, c + pw // 2)
        patch = gray[r0:r1, c0:c1]
        if patch.size < 400 or min(patch.shape) < 20:
            continue
        a, _ = hough_on_fft(patch, n_angles=180)
        if global_angle is not None:
            ae = min(abs(a - global_angle), 180 - abs(a - global_angle))
            if ae > 30:
                continue
        l, _ = estimate_length_cepstrum(patch, a, r_min=4,
                                         r_max=min(patch.shape) // 2 - 2)
        results.append((float(r), float(c), float(a), float(l)))
    return results


def tps_interpolate_psf(sample_results, h, w, grid_step=32):
    if len(sample_results) < 3:
        a0 = sample_results[0][2] if sample_results else 0.0
        l0 = sample_results[0][3] if sample_results else 15.0
        return np.full((h, w), a0), np.full((h, w), l0)

    pts    = np.array([[r, c] for (r, c, a, l) in sample_results])
    angles  = np.array([a     for (r, c, a, l) in sample_results])
    lengths = np.array([l     for (r, c, a, l) in sample_results])

    rbf_angle  = RBFInterpolator(pts, angles,  kernel='thin_plate_spline', smoothing=1.0)
    rbf_length = RBFInterpolator(pts, lengths, kernel='thin_plate_spline', smoothing=1.0)

    rs = np.arange(0, h, grid_step)
    cs = np.arange(0, w, grid_step)
    RR, CC = np.meshgrid(rs, cs, indexing='ij')
    query = np.stack([RR.ravel(), CC.ravel()], axis=1).astype(float)

    a_vals = rbf_angle(query).reshape(RR.shape)
    l_vals = rbf_length(query).reshape(RR.shape)

    from scipy.ndimage import zoom
    angle_map  = zoom(a_vals, (h / a_vals.shape[0], w / a_vals.shape[1]), order=1)[:h, :w]
    length_map = zoom(l_vals, (h / l_vals.shape[0], w / l_vals.shape[1]), order=1)[:h, :w]

    angle_map  = np.clip(angle_map,  0, 179).astype(np.float32)
    length_map = np.clip(length_map, 3, 80).astype(np.float32)
    return angle_map, length_map


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def make_motion_kernel(length, angle_deg):
    length = max(int(length), 3)
    k = length if length % 2 == 1 else length + 1
    kernel = np.zeros((k, k), dtype=np.float64)
    cx = k // 2
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    for i in range(length):
        d = i - length // 2
        fx = cx + d * cos_a
        fy = cx + d * sin_a
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        dx, dy = fx - x0, fy - y0
        for xi, wx in ((x0, 1 - dx), (x0 + 1, dx)):
            for yi, wy in ((y0, 1 - dy), (y0 + 1, dy)):
                if 0 <= xi < k and 0 <= yi < k:
                    kernel[yi, xi] += wx * wy
    s = kernel.sum()
    kernel /= s if s > 0 else 1.0
    return kernel


def rgb_to_gray(image_rgb):
    return (0.2989 * image_rgb[:, :, 0] +
            0.5870 * image_rgb[:, :, 1] +
            0.1140 * image_rgb[:, :, 2])


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — WIENER ROUGH PASS + CHARACTER MASK
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_wiener_K(blurred_gray):
    smooth = gaussian_filter(blurred_gray.astype(np.float64), sigma=1.0)
    residual = blurred_gray.astype(np.float64) - smooth
    noise_var = (np.median(np.abs(residual)) / 0.6745) ** 2
    signal_var = np.var(blurred_gray.astype(np.float64))
    return float(np.clip(noise_var / (signal_var + 1e-10), 1e-4, 0.1))


def pad_reflect(img, pad):
    return np.pad(img, pad, mode='reflect')


def wiener_channel(blurred, kernel, K=0.008):
    h, w = blurred.shape
    kh, kw = kernel.shape
    pad = max(kh, kw)
    blurred_pad = pad_reflect(blurred.astype(np.float64), pad)
    ph, pw = blurred_pad.shape
    kp = np.zeros((ph, pw), dtype=np.float64)
    kp[:kh, :kw] = kernel
    kp = np.roll(np.roll(kp, -kh // 2, axis=0), -kw // 2, axis=1)
    Kf = np.fft.fft2(kp)
    Bf = np.fft.fft2(blurred_pad)
    W = np.conj(Kf) / (np.abs(Kf) ** 2 + K)
    restored = np.fft.ifft2(W * Bf).real
    return np.clip(restored[pad:pad + h, pad:pad + w], 0, 255)


def multi_otsu_threshold(image_norm):
    hist, bins = np.histogram(image_norm.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-10
    centers = (bins[:-1] + bins[1:]) / 2
    cum_w = np.cumsum(hist)
    cum_m = np.cumsum(hist * centers)
    total_mean = cum_m[-1]
    best_var, best_thresh = -1.0, 0.5
    for t in range(1, len(hist) - 1):
        w0 = cum_w[t]
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = cum_m[t] / w0
        m1 = (total_mean - cum_m[t]) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = centers[t]
    return best_thresh


def rough_deblur_and_mask(blurred_gray, kernel, K=None):
    if K is None:
        K = estimate_wiener_K(blurred_gray)
    rough = wiener_channel(blurred_gray, kernel, K)
    rough_norm = rough / 255.0
    thresh = multi_otsu_threshold(rough_norm)
    mask = (rough_norm < thresh).astype(np.float32)
    h, w = blurred_gray.shape
    min_size = max(20, int(h * w * 0.0003))
    labeled, _ = label(mask)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    mask_clean = (sizes > min_size)[labeled].astype(np.float32)
    struct = np.ones((3, 3), dtype=bool)
    mask_clean = binary_dilation(mask_clean.astype(bool), structure=struct).astype(np.float32)
    return rough_norm, mask_clean, thresh


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3+4 — CHAMBOLLE–POCK SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_2d(x):
    return np.roll(x, -1, axis=0) - x, np.roll(x, -1, axis=1) - x


def divergence_2d(py, px):
    return (py - np.roll(py, 1, axis=0)) + (px - np.roll(px, 1, axis=1))


def kernel_fft(kernel, h, w):
    kh, kw = kernel.shape
    kp = np.zeros((h, w), dtype=np.float64)
    kp[:kh, :kw] = kernel
    kp = np.roll(np.roll(kp, -kh // 2, axis=0), -kw // 2, axis=1)
    return np.fft.fft2(kp)


def hough_geometry_weights(h, w, plate_angle_deg):
    Y, X = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist_from_center = np.sqrt(((Y - cy) / h)**2 + ((X - cx) / w)**2)
    return (0.3 + 0.7 * dist_from_center).astype(np.float32)


def chambolle_pock_deblur(blurred_norm, kernel, mask,
                           plate_angle_deg=0,
                           lam=0.04, sigma_evp=0.15, gamma=0.02,
                           K_wiener=0.006, n_iter=300, evp_step=0.02):
    h, w = blurred_norm.shape
    kh, kw = kernel.shape
    pad = max(kh, kw)

    y_pad = pad_reflect(blurred_norm, pad)
    m_pad = pad_reflect(mask, pad)
    ph, pw = y_pad.shape

    Kf      = kernel_fft(kernel, ph, pw)
    Kf_conj = np.conj(Kf)
    Kf_abs2 = np.abs(Kf) ** 2

    Dy = np.fft.fft2(np.array([[1, 0], [-1, 0]], dtype=float), s=(ph, pw))
    Dx = np.fft.fft2(np.array([[1, -1]], dtype=float), s=(ph, pw))
    D_abs2 = np.abs(Dy)**2 + np.abs(Dx)**2

    w_tv    = np.where(m_pad > 0.5, 0.5, 1.0).astype(np.float64)
    denom_x = Kf_abs2 + K_wiener * D_abs2 + gamma * np.ones_like(D_abs2)

    # ── Hough geometry dual variables ─────────────────────────────────────────
    # Implements Term 4:  γ · H(x) = γ · Σᵢ (1 - cos²(∠∇xᵢ - θ_plate)) · |∇xᵢ|
    # h_weights encodes how "misaligned" each pixel's gradient is with the plate
    # direction: 0 = perfectly aligned (preserve), 1 = perpendicular (penalise).
    # Weights are re-estimated every 15 iters (majorisation–minimisation style).
    plate_rad = np.deg2rad(plate_angle_deg)
    qy        = np.zeros((ph, pw), dtype=np.float64)
    qx        = np.zeros((ph, pw), dtype=np.float64)
    h_weights = np.ones((ph, pw),  dtype=np.float64)

    tau   = 0.6
    sigma = 1.0 / (8.0 * tau)

    k_est = kernel.copy()
    Yf    = np.fft.fft2(y_pad)

    # Initialize from Wiener estimate — solver starts close to solution instead
    # of the blurred input, so all iterations go toward refinement not deblurring.
    x     = np.clip(np.fft.ifft2(Kf_conj / (Kf_abs2 + K_wiener) * Yf).real, 0, 1)
    x_bar = x.copy()
    py    = np.zeros_like(x)
    px    = np.zeros_like(x)

    print(f"    Running Chambolle–Pock + Hough geometry ({n_iter} iterations)...")

    for it in range(n_iter):

        # ── Refresh Hough geometry weights every 15 iterations ────────────────
        # h_i = 1 - cos²(∠∇x_i - θ_plate)
        #   → 0 when gradient aligns with plate direction  (good, no penalty)
        #   → 1 when gradient is perpendicular to it       (bad, penalise)
        if it % 15 == 0:
            gy_cur, gx_cur = gradient_2d(x)
            grad_angle = np.arctan2(gy_cur, gx_cur)
            cos_align  = np.cos(grad_angle - plate_rad)
            h_weights  = np.clip(1.0 - cos_align ** 2, 0.0, 1.0)

        # ── Dual (isotropic TV) update ─────────────────────────────────────────
        gy, gx = gradient_2d(x_bar)
        py_new = py + sigma * gy
        px_new = px + sigma * gx
        norm   = np.sqrt(py_new**2 + px_new**2)
        factor = np.maximum(1.0, norm / (lam * w_tv + 1e-10))
        py, px = py_new / factor, px_new / factor

        # ── Dual (Hough geometry) update ───────────────────────────────────────
        # Same prox structure as TV but with per-pixel ball radius γ·h_weights.
        # Gradients mis-aligned with plate get a tighter ball → stronger pull.
        qy_new     = qy + sigma * gy
        qx_new     = qx + sigma * gx
        hough_norm = np.sqrt(qy_new**2 + qx_new**2)
        hough_rad  = gamma * (h_weights + 1e-3)
        hough_fac  = np.maximum(1.0, hough_norm / (hough_rad + 1e-10))
        qy, qx     = qy_new / hough_fac, qx_new / hough_fac

        # ── Primal (image) update — TV + Hough geometry divergences ───────────
        div_p  = divergence_2d(py, px)
        div_q  = divergence_2d(qy, qx)
        rhs    = x - tau * (-(div_p + div_q))
        Xf     = np.fft.fft2(rhs)
        x_new  = np.fft.ifft2((Xf + tau * Kf_conj * Yf) / (1 + tau * denom_x)).real

        # ── EVP proximal step (graduated non-convexity via sigmoid ramp) ──────
        sigmoid     = 1.0 / (1.0 + np.exp(-10.0 * (it / n_iter - 0.5)))
        evp_thresh  = sigma_evp * sigmoid
        char_region = m_pad > 0.5
        x_char      = x_new[char_region]
        x_char      = np.where(x_char > 0.5,
                               np.minimum(1.0, x_char + evp_thresh * evp_step),
                               np.maximum(0.0, x_char - evp_thresh * evp_step))
        x_new[char_region] = x_char
        x_new = np.clip(x_new, 0, 1)

        # Blind kernel update disabled — can lock in a wrong kernel before convergence

        # ── Over-relaxation ────────────────────────────────────────────────────
        x_bar = 2 * x_new - x

        # ── Early stopping ─────────────────────────────────────────────────────
        residual = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        x = x_new
        # Only allow early stop after EVP has fully ramped up (sigmoid > 0.95 → it > 0.8*n_iter)
        if residual < 1e-5 and it > int(0.8 * n_iter):
            print(f"    Early stop at iteration {it + 1} (residual={residual:.2e})")
            break

    print("    Done.")
    return np.clip(x[pad:pad + h, pad:pad + w], 0, 1), k_est


# ═══════════════════════════════════════════════════════════════════════════════
# HALF-QUADRATIC SPLITTING SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def tv_prox(f, lam, plate_angle_deg=0.0, gamma=0.0, n_iter=40):
    """Solve  min_z  lam·TV(z) + gamma·H(z) + ½‖z - f‖²  via Chambolle–Pock.

    H(z) is the Hough geometry term.  Both TV and H share the same dual
    variable with a per-pixel ball radius:
        ball_i = lam + gamma · (1 - cos²(∠∇z_i - θ_plate))
    This is pure denoising — no deconvolution kernel is involved.
    """
    z     = f.copy()
    z_bar = z.copy()
    py    = np.zeros_like(z)
    px    = np.zeros_like(z)
    tau   = 0.25
    sigma = 1.0 / (8.0 * tau)

    if gamma > 0.0:
        plate_rad = np.deg2rad(plate_angle_deg)
        gy0, gx0  = gradient_2d(f)
        cos_align = np.cos(np.arctan2(gy0, gx0) - plate_rad)
        ball_rad  = lam + gamma * np.clip(1.0 - cos_align ** 2, 0.0, 1.0)
    else:
        ball_rad = lam

    for _ in range(n_iter):
        gy, gx   = gradient_2d(z_bar)
        py_n     = py + sigma * gy
        px_n     = px + sigma * gx
        factor   = np.maximum(1.0, np.sqrt(py_n**2 + px_n**2) / ball_rad)
        py, px   = py_n / factor, px_n / factor

        div_p    = divergence_2d(py, px)
        z_new    = (z + tau * div_p + tau * f) / (1.0 + tau)
        z_new    = np.clip(z_new, 0, 1)
        z_bar    = 2.0 * z_new - z
        z        = z_new

    return z


def hqs_deblur(blurred_norm, kernel, mask, plate_angle_deg=0,
               lam=0.02, gamma=0.01, K_wiener=0.008,
               n_outer=8, n_tv=40, evp_step=0.03):
    """Half-Quadratic Splitting deblurring.

    Reformulates  min_x ½‖k*x-y‖² + λ·TV(x) + γ·H(x)  by introducing z:

        x-step:  x = ifft2( (K̄·Ŷ + μ·Ẑ) / (|K̂|² + μ) )  — pure Wiener, sharp
        z-step:  z = tv_prox(x,  lam/μ,  gamma/μ)          — pure denoising

    Key insight: x-step is ALWAYS a Wiener filter so it never blurs.
    z-step is ALWAYS pure denoising so TV does its job without fighting
    the deconvolution.  μ grows ×2 each outer iteration (continuation).
    """
    h, w = blurred_norm.shape
    kh, kw = kernel.shape
    pad = max(kh, kw)

    y_pad = pad_reflect(blurred_norm, pad)
    m_pad = pad_reflect(mask, pad)
    ph, pw = y_pad.shape

    Kf      = kernel_fft(kernel, ph, pw)
    Kf_conj = np.conj(Kf)
    Kf_abs2 = np.abs(Kf) ** 2
    Yf      = np.fft.fft2(y_pad)

    # Warm-start from Wiener — all iterations refine, none spend time deblurring
    x  = np.clip(np.fft.ifft2(Kf_conj / (Kf_abs2 + K_wiener) * Yf).real, 0, 1)
    z  = x.copy()
    mu = K_wiener
    char_region = m_pad > 0.5

    print(f"    Running HQS ({n_outer} outer × {n_tv} TV iters)...")

    for k in range(n_outer):
        # ── x-step: closed-form Wiener (deconvolution only) ───────────────────
        Zf = np.fft.fft2(z)
        x  = np.clip(np.fft.ifft2((Kf_conj * Yf + mu * Zf) / (Kf_abs2 + mu)).real, 0, 1)

        # ── z-step: TV + Hough proximal (denoising only, no convolution) ──────
        z = tv_prox(x, lam=lam / mu, plate_angle_deg=plate_angle_deg,
                    gamma=gamma / mu, n_iter=n_tv)

        # ── EVP: graduated push of character pixels toward {0, 1} ─────────────
        t       = (k + 1) / n_outer
        sigmoid = 1.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))
        push    = sigmoid * evp_step
        z_char  = z[char_region]
        z_char  = np.where(z_char > 0.5,
                           np.minimum(1.0, z_char + push),
                           np.maximum(0.0, z_char - push))
        z[char_region] = z_char
        z  = np.clip(z, 0, 1)

        print(f"      k={k+1}/{n_outer}  μ={mu:.5f}  λ_eff={lam/mu:.4f}")
        mu *= 2.0

    # Final x-step: pull back the sharp deconvolution anchored on clean z
    Zf    = np.fft.fft2(z)
    x_out = np.clip(np.fft.ifft2((Kf_conj * Yf + mu * Zf) / (Kf_abs2 + mu)).real, 0, 1)

    print("    Done.")
    return np.clip(x_out[pad:pad + h, pad:pad + w], 0, 1), kernel


def _directional_sharpen(image_f64, mask, plate_angle_deg):
    """Unsharp mask steered perpendicular to the blur direction."""
    rad  = np.deg2rad(plate_angle_deg + 90.0)
    sz, half = 5, 2
    kern = np.zeros((sz, sz), dtype=np.float64)
    ca, sa = np.cos(rad), np.sin(rad)
    for i in range(sz):
        d  = i - half
        fx, fy = half + d * ca, half + d * sa
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        for xi, wx in ((x0, 1 - (fx - x0)), (x0 + 1, fx - x0)):
            for yi, wy in ((y0, 1 - (fy - y0)), (y0 + 1, fy - y0)):
                if 0 <= xi < sz and 0 <= yi < sz:
                    kern[yi, xi] += wx * wy
    s = kern.sum()
    if s > 0:
        kern /= s
    from scipy.ndimage import convolve
    blurred_d = convolve(image_f64, kern, mode='reflect')
    usm   = image_f64 + 1.5 * (image_f64 - blurred_d)
    alpha = np.clip(0.4 + 0.6 * mask, 0.4, 1.0)
    return np.clip(alpha * usm + (1.0 - alpha) * image_f64, 0.0, 1.0)


def deblur_full(blurred_rgb, kernel, mask, plate_angle_deg=0, n_iter=300, K_wiener=0.02):
    # Deblur each channel WITHOUT EVP (evp_step=0) — EVP applied on luminance only below
    channels = []
    for c in range(3):
        ch = blurred_rgb[:, :, c].astype(np.float64) / 255.0
        result, _ = hqs_deblur(ch, kernel, mask,
                                plate_angle_deg=plate_angle_deg,
                                lam=0.02, gamma=0.01, K_wiener=K_wiener,
                                n_outer=8, n_tv=40, evp_step=0.0)
        result = _directional_sharpen(result, mask, plate_angle_deg)
        channels.append(result)

    # Cross-channel luminance coupling to suppress colour fringing
    weights  = np.array([0.2989, 0.5870, 0.1140])
    lum      = sum(channels[c] * weights[c] for c in range(3))
    lum_norm = lum / (lum.max() + 1e-10)
    coupled  = []
    for c in range(3):
        blended = 0.92 * channels[c] + 0.08 * (lum_norm * (channels[c].max() + 1e-10))
        coupled.append(np.clip(blended, 0, 1))

    # EVP applied to luminance only — push character pixels toward {0,1} in Y channel
    # then redistribute the push back to RGB proportionally.
    # This avoids per-channel binary extremes that cause colour fringing.
    lum_coupled = sum(coupled[c] * weights[c] for c in range(3))
    m_pad_crop  = mask  # already same shape as image
    char_region = m_pad_crop > 0.5
    lum_evp     = lum_coupled.copy()
    lum_char    = lum_evp[char_region]
    lum_char    = np.where(lum_char > 0.5,
                           np.minimum(1.0, lum_char + 0.03),
                           np.maximum(0.0, lum_char - 0.03))
    lum_evp[char_region] = lum_char

    # Scale each RGB channel by the luminance EVP ratio
    lum_ratio = lum_evp / (lum_coupled + 1e-10)
    final = []
    for c in range(3):
        final.append(np.clip(coupled[c] * lum_ratio, 0, 1))

    return np.clip(np.stack(final, axis=2) * 255, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — 4-PANEL FIGURE (no ground truth)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(blurred, rough_deblur, final_deblur,
                est_kernel, fft_map, mask,
                est_angle, est_length, out_path):

    BG = "#0d0d0d"
    C1 = "#ff8a65"   # orange  — blurred input
    C2 = "#aed581"   # yellow-green — FFT / mask
    C3 = "#4fc3f7"   # cyan  — Wiener
    C4 = "#81c784"   # green — final
    CM = "#ce93d8"   # purple — metrics

    fig = plt.figure(figsize=(22, 9), facecolor=BG)
    gs = GridSpec(2, 4, figure=fig,
                  height_ratios=[3.2, 1.1],
                  hspace=0.22, wspace=0.14,
                  left=0.01, right=0.98,
                  top=0.91, bottom=0.03)

    def show_img(ax, data, title, color, cmap=None):
        ax.imshow(data, aspect='equal', cmap=cmap)
        ax.set_title(title, color=color, fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')

    def show_kern(ax, k, title, color):
        knorm = k / (k.max() + 1e-10)
        ax.imshow(knorm, cmap='hot', vmin=0, vmax=1,
                  interpolation='nearest', aspect='equal')
        ax.set_title(title, color=color, fontsize=7.5, fontweight='bold', pad=3)
        ax.axis('off')

    # Panel 1 — Blurred input
    ax0 = fig.add_subplot(gs[0, 0])
    show_img(ax0, blurred, "① BLURRED INPUT", C1)
    bk0 = fig.add_subplot(gs[1, 0])
    bk0.set_facecolor(BG)
    bk0.text(0.5, 0.5, "input image", color="#555",
             ha='center', va='center', fontsize=9, style='italic',
             transform=bk0.transAxes)
    bk0.axis('off')

    # Panel 2 — FFT + mask
    ax1 = fig.add_subplot(gs[0, 1])
    show_img(ax1, fft_map,
             f"② FFT MAGNITUDE\n(Hough → {est_angle}°)", C2, cmap='inferno')
    bk1 = fig.add_subplot(gs[1, 1])
    show_img(bk1, mask, "Character Mask", C2, cmap='gray')
    bk1.axis('off')

    # Panel 3 — Wiener rough
    ax2 = fig.add_subplot(gs[0, 2])
    show_img(ax2, rough_deblur, "③ WIENER (ROUGH)", C3)
    bk2 = fig.add_subplot(gs[1, 2])
    show_kern(bk2, est_kernel, f"Est. kernel  {est_angle}° · {est_length}px", C3)

    # Panel 4 — Final
    ax3 = fig.add_subplot(gs[0, 3])
    show_img(ax3, final_deblur, "④ CHAMBOLLE–POCK (FINAL)", C4)
    bk3 = fig.add_subplot(gs[1, 3])
    bk3.set_facecolor("#111")
    bk3.axis('off')
    bk3.text(0.5, 0.7, f"Est. angle:  {est_angle}°",
             color=CM, fontsize=8, ha='center', transform=bk3.transAxes)
    bk3.text(0.5, 0.45, f"Est. length: {est_length} px",
             color=CM, fontsize=8, ha='center', transform=bk3.transAxes)
    bk3.text(0.5, 0.2, "No ground truth — blind deblur",
             color="#555", fontsize=7, style='italic',
             ha='center', transform=bk3.transAxes)

    fig.text(0.5, 0.948,
             "License Plate Deblurring — Real Input  ·  "
             "Hough + Cepstrum + TPS + TV + EVP + Chambolle–Pock  "
             "(NumPy / SciPy / PIL)",
             color="#ddd", fontsize=10, ha='center', va='center', fontweight='bold')

    for xf in [0.245, 0.490, 0.735]:
        fig.add_artist(plt.Line2D([xf, xf], [0.03, 0.93],
                                   transform=fig.transFigure,
                                   color="#222", linewidth=1, linestyle='--'))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(image_path, out_path="deblurring_result.png", n_iter=300,
        angle=None, length=None):

    print(f"\n  Loading: {image_path}")
    blurred_rgb  = np.array(Image.open(image_path).convert("RGB"))
    blurred_gray = (0.2989 * blurred_rgb[:, :, 0] +
                    0.5870 * blurred_rgb[:, :, 1] +
                    0.1140 * blurred_rgb[:, :, 2])

    # ── STEP 1A — angle ────────────────────────────────────────────────────────
    if angle is not None:
        est_angle_hough = int(angle)
        _, fft_map = hough_on_fft(blurred_gray, n_angles=4)  # just for display
        print(f"  Angle (override): {est_angle_hough}°")
    else:
        print("\n  Step 1A: Hough on FFT (angle estimation)...")
        est_angle_hough, fft_map = hough_on_fft(blurred_gray, n_angles=360)
        print(f"  Hough angle: {est_angle_hough}°")

    # ── STEP 1B — length ───────────────────────────────────────────────────────
    if length is not None:
        est_length = int(length)
        print(f"  Length (override): {est_length} px")
    else:
        print("  Step 1B: Cepstrum (length estimation)...")
        est_length, _ = estimate_length_cepstrum(blurred_gray, est_angle_hough)
        print(f"  Cepstrum length: {est_length} px")

    # ── STEP 1C — TPS local PSF ────────────────────────────────────────────────
    print("  Step 1C: TPS local PSF estimation...")
    h_img, w_img = blurred_gray.shape
    sample_results = estimate_local_psf_params(blurred_gray, global_angle=est_angle_hough)
    angle_map, length_map = tps_interpolate_psf(sample_results, h_img, w_img)
    tps_angle  = int(round(angle_map[h_img // 2, w_img // 2])) % 180
    tps_length = max(3, int(round(length_map[h_img // 2, w_img // 2])))
    ae_direct  = min(abs(tps_angle - est_angle_hough),
                     180 - abs(tps_angle - est_angle_hough))
    if angle is None:
        est_angle  = tps_angle if ae_direct <= 20 else est_angle_hough
        est_length = max(3, int(round(0.6 * est_length + 0.4 * tps_length)))
    else:
        est_angle = est_angle_hough
    print(f"  TPS center: {tps_angle}°/{tps_length}px → final: {est_angle}°/{est_length}px")

    est_kernel = make_motion_kernel(est_length, est_angle)

    # ── STEP 2 — Wiener + mask ─────────────────────────────────────────────────
    print("\n  Step 2: Wiener rough pass + character mask...")
    adaptive_K = estimate_wiener_K(blurred_gray)
    print(f"  Adaptive Wiener K: {adaptive_K:.5f}")
    rough_norm, mask, thresh = rough_deblur_and_mask(blurred_gray, est_kernel, K=adaptive_K)
    rough_deblur_rgb = np.stack([
        np.clip(wiener_channel(blurred_rgb[:, :, c].astype(float),
                               est_kernel, K=adaptive_K), 0, 255).astype(np.uint8)
        for c in range(3)
    ], axis=2)
    print(f"  Mask threshold: {thresh:.3f}  |  Masked pixels: {mask.mean()*100:.1f}%")

    # ── STEP 3+4 — Chambolle–Pock ──────────────────────────────────────────────
    print(f"\n  Step 3+4: Chambolle–Pock solver ({n_iter} iterations)...")
    final_rgb = deblur_full(blurred_rgb, est_kernel, mask,
                             plate_angle_deg=est_angle,
                             n_iter=n_iter, K_wiener=adaptive_K)

    # ── Figure ─────────────────────────────────────────────────────────────────
    make_figure(
        blurred=blurred_rgb,
        rough_deblur=rough_deblur_rgb,
        final_deblur=final_rgb,
        est_kernel=est_kernel,
        fft_map=fft_map,
        mask=mask,
        est_angle=est_angle,
        est_length=est_length,
        out_path=out_path,
    )

    # Also save the deblurred image on its own
    clean_out = out_path.replace(".png", "_clean.png")
    Image.fromarray(final_rgb).save(clean_out)
    print(f"  Clean deblurred image → {clean_out}")

    # ── STEP 5 — OCR ───────────────────────────────────────────────────────────
    try:
        from ocr_plate import recognize as _ocr_recognize
        ocr_text = _ocr_recognize(final_rgb)
        print(f"\n  ┌─────────────────────────────────┐")
        print(f"  │  OCR Plate Text :  {ocr_text:<14s} │")
        print(f"  └─────────────────────────────────┘")
    except Exception as _e:
        print(f"\n  OCR failed: {_e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deblurring_input.py blurred.png [out.png] [n_iter] [angle] [length]")
        sys.exit(1)
    image_path = sys.argv[1]
    out_p      = sys.argv[2] if len(sys.argv) > 2 else "deblurring_result.png"
    n_iter     = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    angle      = int(sys.argv[4]) if len(sys.argv) > 4 else None
    length     = int(sys.argv[5]) if len(sys.argv) > 5 else None
    run(image_path, out_path=out_p, n_iter=n_iter, angle=angle, length=length)