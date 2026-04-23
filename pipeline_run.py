"""
pipeline_run.py — Full pipeline runner with intermediate capture
================================================================
Calls deblurring_input.py and ocr_plate.py without modifying them.
Saves every intermediate step + one big zoomable pipeline figure.

Usage
-----
    # blurred image, no ground truth
    python pipeline_run.py input/img1.png

    # blurred image WITH ground truth (enables PSNR/SSIM)
    python pipeline_run.py input/img1.png --gt ../BEST_TEST_IM/not_blurred/nimg1.png

    # unblurred image → apply synthetic blur then deblur (enables PSNR/SSIM)
    python pipeline_run.py ../BEST_TEST_IM/not_blurred/nimg1.png --synthetic --angle 35 --length 20

    # also update LaTeX report after run
    python pipeline_run.py input/img1.png --gt ../BEST_TEST_IM/not_blurred/nimg1.png --latex
"""

import sys
import os
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent.resolve()
PIPELINE    = HERE.parent
NOT_BLURRED = PIPELINE / "BEST_TEST_IM" / "not_blurred"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPELINE))

# ── Import existing code (DO NOT MODIFY THOSE FILES) ─────────────────────────
from deblurring_input import (
    hough_on_fft,
    estimate_length_cepstrum,
    estimate_local_psf_params,
    tps_interpolate_psf,
    make_motion_kernel,
    estimate_wiener_K,
    rough_deblur_and_mask,
    wiener_channel,
    deblur_full,
    hqs_deblur,
    pad_reflect,
    kernel_fft,
)
from ocr_plate import recognize as tesseract_recognize

try:
    from ocr_easyocr import recognize_plate as easyocr_recognize
    _EASYOCR = True
except Exception:
    _EASYOCR = False


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def psnr(original: np.ndarray, restored: np.ndarray) -> float:
    orig = original.astype(np.float64)
    rest = restored.astype(np.float64)
    mse  = np.mean((orig - rest) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    from scipy.ndimage import uniform_filter
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a  = uniform_filter(a, win)
    mu_b  = uniform_filter(b, win)
    mu_a2 = uniform_filter(a * a, win) - mu_a ** 2
    mu_b2 = uniform_filter(b * b, win) - mu_b ** 2
    mu_ab = uniform_filter(a * b, win) - mu_a * mu_b
    num   = (2 * mu_a * mu_b + C1) * (2 * mu_ab + C2)
    den   = (mu_a ** 2 + mu_b ** 2 + C1) * (mu_a2 + mu_b2 + C2)
    return float(np.mean(num / (den + 1e-10)))


def laplacian_sharpness(img: np.ndarray) -> float:
    gray = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1]
            + 0.1140 * img[:, :, 2]).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    lap  = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.ndimage import convolve
    return float(np.var(convolve(gray, lap)))


def compute_metrics(blurred_rgb, final_rgb, gt_rgb=None):
    metrics = {
        "sharpness_input":  laplacian_sharpness(blurred_rgb),
        "sharpness_output": laplacian_sharpness(final_rgb),
        "psnr":  None,
        "ssim":  None,
        "psnr_input": None,
    }
    metrics["sharpness_gain"] = metrics["sharpness_output"] - metrics["sharpness_input"]
    if gt_rgb is not None:
        gt = gt_rgb.astype(np.float64)
        fi = final_rgb.astype(np.float64)
        bl = blurred_rgb.astype(np.float64)
        if gt.shape == fi.shape:
            metrics["psnr"]       = psnr(gt_rgb, final_rgb)
            metrics["ssim"]       = ssim(gt_rgb, final_rgb)
            metrics["psnr_input"] = psnr(gt_rgb, blurred_rgb)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC BLUR
# ══════════════════════════════════════════════════════════════════════════════

def apply_motion_blur(img_rgb: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    kernel = make_motion_kernel(length, angle_deg)
    kh, kw = kernel.shape
    result_channels = []
    for c in range(3):
        ch   = img_rgb[:, :, c].astype(np.float64)
        pad  = max(kh, kw)
        cp   = pad_reflect(ch, pad)
        ph, pw = cp.shape
        kp   = np.zeros((ph, pw))
        kp[:kh, :kw] = kernel
        kp   = np.roll(np.roll(kp, -kh // 2, 0), -kw // 2, 1)
        Kf   = np.fft.fft2(kp)
        Cf   = np.fft.fft2(cp)
        blurred = np.fft.ifft2(Kf * Cf).real
        blurred = np.clip(blurred[pad:pad + ch.shape[0], pad:pad + ch.shape[1]], 0, 255)
        result_channels.append(blurred.astype(np.uint8))
    return np.stack(result_channels, axis=2)


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_path:    str,
    gt_path:       str   = None,
    synthetic:     bool  = False,
    angle_hint:    int   = None,
    length_hint:   int   = None,
    n_outer:       int   = 8,
    n_tv:          int   = 40,
    lam:           float = 0.02,
    gamma:         float = 0.01,
    evp_step:      float = 0.0,
    out_base:      str   = None,
    verbose:       bool  = True,
    gt_text:       str   = None,    # ground-truth plate string for OCR comparison
    step_callback         = None,   # callable(dict) — called after each step for GUI
) -> dict:
    """
    Run the full deblurring + OCR pipeline and return a dict with every
    intermediate image, PSF parameters, metrics, and OCR results.
    """

    # ── Load image ────────────────────────────────────────────────────────────
    gt_rgb     = None
    blurred_rgb = None

    if synthetic:
        # image_path is the SHARP image; we blur it synthetically
        sharp_rgb = np.array(Image.open(image_path).convert("RGB"))
        ang = angle_hint  if angle_hint  is not None else 35
        lgt = length_hint if length_hint is not None else 20
        blurred_rgb = apply_motion_blur(sharp_rgb, lgt, ang)
        gt_rgb      = sharp_rgb
        true_angle, true_length = ang, lgt
        if verbose:
            print(f"  Synthetic blur applied: angle={ang}°  length={lgt}px")
    else:
        blurred_rgb = np.array(Image.open(image_path).convert("RGB"))
        true_angle, true_length = None, None
        if gt_path:
            gt_rgb = np.array(Image.open(gt_path).convert("RGB"))
            gt_rgb = np.array(Image.fromarray(gt_rgb).resize(
                (blurred_rgb.shape[1], blurred_rgb.shape[0]), Image.LANCZOS))

    blurred_gray = (0.2989 * blurred_rgb[:, :, 0] +
                    0.5870 * blurred_rgb[:, :, 1] +
                    0.1140 * blurred_rgb[:, :, 2])

    def _cb(msg):
        if step_callback:
            step_callback(msg)

    _cb({"type": "step", "name": "loaded", "image": blurred_rgb})

    # ── Step 1A: Angle estimation ──────────────────────────────────────────────
    if verbose: print("\n  [1A] Hough on FFT — angle estimation...")
    if angle_hint is not None and not synthetic:
        est_angle_hough = int(angle_hint)
        _, fft_map = hough_on_fft(blurred_gray, n_angles=4)
    else:
        est_angle_hough, fft_map = hough_on_fft(blurred_gray, n_angles=360)
    if verbose: print(f"       Hough angle: {est_angle_hough}°")
    _cb({"type": "step", "name": "fft", "image": _fft_to_rgb(fft_map)})

    # ── Step 1B: Length estimation ─────────────────────────────────────────────
    if verbose: print("  [1B] Cepstrum — length estimation...")
    if length_hint is not None and not synthetic:
        est_length = int(length_hint)
        cep = None
    else:
        est_length, cep = estimate_length_cepstrum(blurred_gray, est_angle_hough)
    if verbose: print(f"       Cepstrum length: {est_length}px")
    # Render cepstrum as RGB for GUI display
    cep_rgb = None
    if cep is not None:
        cep_norm = np.abs(cep)
        cep_norm = np.log1p(cep_norm)
        cep_norm = (cep_norm / (cep_norm.max() + 1e-10) * 255).astype(np.uint8)
        cep_rgb  = np.stack([cep_norm, cep_norm, cep_norm], axis=2)
    _cb({"type": "step", "name": "cepstrum",
         "angle": est_angle_hough, "length": est_length, "cep_image": cep_rgb})

    # ── Step 1C: TPS local PSF ────────────────────────────────────────────────
    if verbose: print("  [1C] TPS local PSF estimation...")
    _cb({"type": "step", "name": "tps_start"})  # signal UI — TPS samples 9 patches
    h_img, w_img  = blurred_gray.shape
    sample_results = estimate_local_psf_params(blurred_gray, global_angle=est_angle_hough)
    angle_map, length_map = tps_interpolate_psf(sample_results, h_img, w_img)
    tps_angle  = int(round(angle_map[h_img // 2, w_img // 2])) % 180
    tps_length = max(3, int(round(length_map[h_img // 2, w_img // 2])))
    ae_direct  = min(abs(tps_angle - est_angle_hough), 180 - abs(tps_angle - est_angle_hough))
    if angle_hint is None or synthetic:
        est_angle  = tps_angle if ae_direct <= 20 else est_angle_hough
        est_length = max(3, int(round(0.6 * est_length + 0.4 * tps_length)))
    else:
        est_angle = est_angle_hough
    if verbose: print(f"       TPS: {tps_angle}°/{tps_length}px → final: {est_angle}°/{est_length}px")

    est_kernel = make_motion_kernel(est_length, est_angle)
    _cb({"type": "step", "name": "kernel", "kernel": est_kernel,
         "angle": est_angle, "length": est_length})

    # ── Step 2: Wiener rough + character mask ─────────────────────────────────
    if verbose: print("  [2]  Wiener rough pass + character mask...")
    _cb({"type": "step", "name": "wiener_start"})   # signal UI before heavy work
    adaptive_K = estimate_wiener_K(blurred_gray)
    rough_norm, mask, thresh = rough_deblur_and_mask(blurred_gray, est_kernel, K=adaptive_K)
    rough_rgb = np.stack([
        np.clip(wiener_channel(blurred_rgb[:, :, c].astype(float),
                               est_kernel, K=adaptive_K), 0, 255).astype(np.uint8)
        for c in range(3)
    ], axis=2)
    if verbose:
        print(f"       K={adaptive_K:.5f}  thresh={thresh:.3f}  mask={mask.mean()*100:.1f}%")
    _cb({"type": "step", "name": "wiener", "image": rough_rgb,
         "mask": (mask * 255).astype(np.uint8)})

    # ── Step 3+4: HQS deblur (full or custom params) ─────────────────────────
    if verbose: print(f"  [3+4] HQS solver (n_outer={n_outer}, n_tv={n_tv})...")
    _cb({"type": "step", "name": "hqs_start", "n_outer": n_outer, "n_tv": n_tv})
    channels = []
    for c in range(3):
        ch = blurred_rgb[:, :, c].astype(np.float64) / 255.0
        result, _ = hqs_deblur(
            ch, est_kernel, mask,
            plate_angle_deg=est_angle,
            lam=lam, gamma=gamma,
            K_wiener=adaptive_K,
            n_outer=n_outer, n_tv=n_tv,
            evp_step=evp_step,
        )
        channels.append(result)

    # Luminance coupling + EVP (from deblur_full logic)
    weights     = np.array([0.2989, 0.5870, 0.1140])
    lum         = sum(channels[c] * weights[c] for c in range(3))
    lum_norm    = lum / (lum.max() + 1e-10)
    coupled     = []
    for c in range(3):
        blended = 0.92 * channels[c] + 0.08 * (lum_norm * (channels[c].max() + 1e-10))
        coupled.append(np.clip(blended, 0, 1))
    lum_coupled = sum(coupled[c] * weights[c] for c in range(3))
    char_region = mask > 0.5
    lum_evp     = lum_coupled.copy()
    lum_char    = lum_evp[char_region]
    lum_char    = np.where(lum_char > 0.5,
                           np.minimum(1.0, lum_char + 0.03),
                           np.maximum(0.0, lum_char - 0.03))
    lum_evp[char_region] = lum_char
    lum_ratio = lum_evp / (lum_coupled + 1e-10)
    final_channels = [np.clip(coupled[c] * lum_ratio, 0, 1) for c in range(3)]
    final_rgb = np.clip(np.stack(final_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    _cb({"type": "step", "name": "hqs", "image": final_rgb})

    # ── Step 5: OCR ───────────────────────────────────────────────────────────
    if verbose: print("  [5]  Running OCR...")
    tess_text = tesseract_recognize(final_rgb)
    easy_text = easyocr_recognize(final_rgb) if _EASYOCR else "N/A"
    if verbose:
        print(f"       Tesseract : {tess_text}")
        print(f"       EasyOCR   : {easy_text}")
    _cb({"type": "step", "name": "ocr", "tesseract": tess_text, "easyocr": easy_text})

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(blurred_rgb, final_rgb, gt_rgb)

    if verbose:
        print(f"\n  ── Metrics ──────────────────────────────")
        print(f"     Sharpness input : {metrics['sharpness_input']:.2f}")
        print(f"     Sharpness output: {metrics['sharpness_output']:.2f}  (Δ {metrics['sharpness_gain']:+.2f})")
        if metrics["psnr"] is not None:
            print(f"     PSNR  (vs GT)   : {metrics['psnr']:.2f} dB")
            print(f"     SSIM  (vs GT)   : {metrics['ssim']:.4f}")
            print(f"     PSNR  (blurred) : {metrics['psnr_input']:.2f} dB")

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_base) if out_base else HERE / "output" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(blurred_rgb).save(run_dir / "01_blurred.png")
    _save_fft(fft_map,  run_dir / "02_fft_map.png")
    _save_mask(mask,    run_dir / "03_char_mask.png")
    Image.fromarray(rough_rgb).save(run_dir / "04_wiener_rough.png")
    Image.fromarray(final_rgb).save(run_dir / "05_final_deblur.png")
    _save_kernel(est_kernel, run_dir / "06_psf_kernel.png")
    if gt_rgb is not None:
        Image.fromarray(gt_rgb).save(run_dir / "00_ground_truth.png")

    # ── Big pipeline figure ───────────────────────────────────────────────────
    fig_path = str(run_dir / "pipeline_figure.png")
    make_pipeline_figure(
        blurred_rgb=blurred_rgb,
        fft_map=fft_map,
        mask=mask,
        rough_rgb=rough_rgb,
        kernel=est_kernel,
        final_rgb=final_rgb,
        gt_rgb=gt_rgb,
        metrics=metrics,
        psf_params=dict(angle=est_angle, length=est_length, K=adaptive_K, thresh=thresh),
        tess_text=tess_text,
        easy_text=easy_text,
        out_path=fig_path,
        true_angle=true_angle,
        true_length=true_length,
    )
    if verbose:
        print(f"\n  Outputs saved → {run_dir}")
        print(f"  Pipeline figure → {fig_path}")

    return {
        "image_path": image_path,
        "gt_path":    gt_path,
        "synthetic":  synthetic,
        "params": dict(angle=est_angle, length=est_length, K=adaptive_K,
                       lam=lam, gamma=gamma, n_outer=n_outer, n_tv=n_tv),
        "intermediates": dict(
            blurred_rgb=blurred_rgb,
            gt_rgb=gt_rgb,
            fft_map=fft_map,
            kernel=est_kernel,
            mask=mask,
            rough_rgb=rough_rgb,
            final_rgb=final_rgb,
        ),
        "metrics":  metrics,
        "gt_text":  gt_text,
        "ocr":      dict(tesseract=tess_text, easyocr=easy_text),
        "output_dir":      str(run_dir),
        "pipeline_figure": fig_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fft_to_rgb(fft_map: np.ndarray) -> np.ndarray:
    """Convert FFT magnitude map (float [0,1]) to an RGB array for display."""
    import matplotlib.cm as cm
    colored = cm.inferno(fft_map)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def _save_fft(fft_map, path):
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="black")
    ax.imshow(fft_map, cmap="inferno", aspect="auto")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor="black", dpi=120)
    plt.close(fig)


def _save_mask(mask, path):
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="black")
    ax.imshow(mask, cmap="gray", aspect="auto")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor="black", dpi=120)
    plt.close(fig)


def _save_kernel(kernel, path):
    fig, ax = plt.subplots(figsize=(2, 2), facecolor="black")
    ax.imshow(kernel / (kernel.max() + 1e-10), cmap="hot", aspect="auto")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor="black", dpi=120)
    plt.close(fig)


def make_pipeline_figure(
    blurred_rgb, fft_map, mask, rough_rgb, kernel,
    final_rgb, gt_rgb, metrics, psf_params,
    tess_text, easy_text, out_path,
    true_angle=None, true_length=None,
):
    BG = "#0d0d0d"
    has_gt = gt_rgb is not None
    n_top  = 6 if has_gt else 5

    fig = plt.figure(figsize=(24, 11), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, n_top, figure=fig,
        height_ratios=[3.8, 1.2],
        hspace=0.18, wspace=0.10,
        left=0.01, right=0.99,
        top=0.90, bottom=0.02,
    )

    COLORS = ["#ff8a65", "#aed581", "#4fc3f7", "#f48fb1", "#81c784", "#ce93d8"]
    TITLES = (
        (["① GROUND TRUTH",  "② BLURRED INPUT"] if has_gt else ["① BLURRED INPUT"]) +
        [f"③ FFT MAGNITUDE\n(Hough → {psf_params['angle']}°)",
         "④ CHARACTER MASK",
         "⑤ WIENER ROUGH",
         f"⑥ FINAL DEBLUR\n(HQS + TV + EVP)"]
    )
    imgs = ([gt_rgb, blurred_rgb] if has_gt else [blurred_rgb]) + [None, None, rough_rgb, final_rgb]

    def show(ax, data, title, col, cmap=None):
        ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title, color=col, fontsize=9, fontweight="bold", pad=5)
        ax.axis("off")

    for i, (title, img, col) in enumerate(zip(TITLES, imgs, COLORS)):
        ax = fig.add_subplot(gs[0, i])
        if title.startswith("③"):
            show(ax, fft_map, title, col, cmap="inferno")
        elif title.startswith("④"):
            show(ax, mask,    title, col, cmap="gray")
        else:
            show(ax, img, title, col)

    # ── Bottom row ─────────────────────────────────────────────────────────────
    # Kernel
    ax_k = fig.add_subplot(gs[1, 0 if not has_gt else 1])
    knorm = kernel / (kernel.max() + 1e-10)
    ax_k.imshow(knorm, cmap="hot", aspect="auto", interpolation="nearest")
    ang_str = f"True: {true_angle}°  Est: {psf_params['angle']}°" if true_angle else f"Est: {psf_params['angle']}°"
    len_str = f"True: {true_length}px  Est: {psf_params['length']}px" if true_length else f"Est: {psf_params['length']}px"
    ax_k.set_title(f"PSF Kernel\nAngle {ang_str}\nLength {len_str}",
                   color="#4fc3f7", fontsize=7.5, pad=3)
    ax_k.axis("off")

    # Metrics panel
    ax_m = fig.add_subplot(gs[1, 2 if not has_gt else 3])
    ax_m.set_facecolor("#111")
    ax_m.axis("off")
    lines = [
        f"Sharpness in : {metrics['sharpness_input']:.1f}",
        f"Sharpness out: {metrics['sharpness_output']:.1f}  (Δ {metrics['sharpness_gain']:+.1f})",
    ]
    if metrics["psnr"] is not None:
        lines += [
            f"PSNR (blurred): {metrics['psnr_input']:.2f} dB",
            f"PSNR (deblur) : {metrics['psnr']:.2f} dB",
            f"SSIM          : {metrics['ssim']:.4f}",
        ]
    for j, line in enumerate(lines):
        ax_m.text(0.05, 0.88 - j * 0.18, line, color="#ce93d8",
                  fontsize=8, transform=ax_m.transAxes, fontfamily="monospace")

    # OCR panel
    ax_o = fig.add_subplot(gs[1, 3 if not has_gt else 4])
    ax_o.set_facecolor("#111")
    ax_o.axis("off")
    ax_o.text(0.05, 0.85, "OCR Results",     color="#aed581", fontsize=9, fontweight="bold", transform=ax_o.transAxes)
    ax_o.text(0.05, 0.60, f"Tesseract: {tess_text}", color="#fff", fontsize=9, fontfamily="monospace", transform=ax_o.transAxes)
    ax_o.text(0.05, 0.35, f"EasyOCR:   {easy_text}", color="#fff", fontsize=9, fontfamily="monospace", transform=ax_o.transAxes)

    # Histogram
    ax_h = fig.add_subplot(gs[1, 4 if not has_gt else 5])
    ax_h.set_facecolor("#111")
    gray_b = (0.2989*blurred_rgb[:,:,0] + 0.5870*blurred_rgb[:,:,1] + 0.1140*blurred_rgb[:,:,2]).ravel()
    gray_f = (0.2989*final_rgb[:,:,0]   + 0.5870*final_rgb[:,:,1]   + 0.1140*final_rgb[:,:,2]).ravel()
    ax_h.hist(gray_b, bins=64, range=(0,255), color="#ff8a65", alpha=0.6, density=True, label="blurred")
    ax_h.hist(gray_f, bins=64, range=(0,255), color="#81c784", alpha=0.6, density=True, label="deblurred")
    ax_h.set_title("Pixel Histogram", color="#aed581", fontsize=8, pad=3)
    ax_h.tick_params(labelsize=6, colors="gray")
    ax_h.legend(fontsize=6, facecolor="#222", labelcolor="white")
    for s in ax_h.spines.values():
        s.set_color("#333")
    ax_h.set_facecolor("#111")

    # Title
    fig.text(0.5, 0.945,
             "License Plate Deblurring Pipeline  ·  "
             "Hough + Cepstrum + TPS + HQS + TV + Hough-Geometry + EVP",
             color="#ddd", fontsize=11, ha="center", fontweight="bold")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Pipeline figure saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    W = 58  # terminal width

    ap = argparse.ArgumentParser(
        description="PlateReveal — Blind Motion Deblurring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # already-blurred image (auto-estimate kernel):
      python pipeline_run.py input/img1.png

  # apply known blur then deblur (synthetic benchmark):
      python pipeline_run.py sharp/nimg1.png --unblurred 45 20

  # blurred image with known ground truth (enables PSNR/SSIM):
      python pipeline_run.py input/img1.png --gt ../BEST_TEST_IM/not_blurred/nimg1.png
""")
    ap.add_argument("image", help="Input image (blurred, or sharp when --unblurred is used)")
    ap.add_argument("--unblurred", nargs=2, type=float, metavar=("ANGLE", "LENGTH"),
                    help="Image is SHARP — apply blur at ANGLE° LENGTH px, then deblur")
    ap.add_argument("--gt",      help="Ground-truth sharp image (enables PSNR/SSIM)")
    ap.add_argument("--angle",   type=int,   default=None, help="Blur angle hint (deg)")
    ap.add_argument("--length",  type=int,   default=None, help="Blur length hint (px)")
    ap.add_argument("--n-outer", type=int,   default=8,    help="HQS outer iterations (default 8)")
    ap.add_argument("--n-tv",    type=int,   default=40,   help="TV inner iterations (default 40)")
    ap.add_argument("--lam",     type=float, default=0.02, help="TV weight λ (default 0.02)")
    ap.add_argument("--gamma",   type=float, default=0.01, help="Hough weight γ (default 0.01)")
    ap.add_argument("--latex",   action="store_true",      help="Update report_long.tex after run")
    args = ap.parse_args()

    synthetic    = False
    angle_hint   = args.angle
    length_hint  = args.length
    true_angle_cli = None
    true_len_cli   = None

    if args.unblurred:
        synthetic       = True
        true_angle_cli  = args.unblurred[0]
        true_len_cli    = int(args.unblurred[1])
        angle_hint      = true_angle_cli
        length_hint     = true_len_cli

    BAR = "─" * W
    print(f"\n╔{'═'*W}╗")
    print(f"║{'  PLATE REVEAL — Blind Deblurring Pipeline':^{W}}║")
    print(f"╚{'═'*W}╝")
    print(f"  Image  : {args.image}")
    if synthetic:
        print(f"  Mode   : Unblurred → apply blur ({true_angle_cli:.0f}°, {true_len_cli}px) → deblur")
    else:
        print(f"  Mode   : Blurred input  (auto-estimate kernel)")
    if args.gt:
        print(f"  GT     : {args.gt}")
    print(f"  Params : λ={args.lam}  γ={args.gamma}  "
          f"n_outer={args.n_outer}  n_tv={args.n_tv}")
    print(f"  {BAR}")

    _steps_done = [0]
    _step_labels = {
        "fft":      "Angle estimation (Hough/FFT)  ",
        "cepstrum": "Length estimation (Cepstrum)  ",
        "kernel":   "TPS PSF + kernel construction ",
        "wiener":   "Wiener rough pass + mask      ",
        "hqs":      f"HQS fine deblur ({args.n_outer}×{args.n_tv})        ",
        "ocr":      "OCR (Tesseract + EasyOCR)     ",
    }
    _step_vals = {}

    def _cli_step(msg):
        name = msg.get("name")
        if name not in _step_labels:
            return
        label = _step_labels[name]
        extra = ""
        if name == "fft":
            extra = f"  {msg.get('image', '').__class__.__name__}"
        elif name == "cepstrum":
            extra = f"  {msg.get('angle', '?')}°  {msg.get('length', '?')}px"
        elif name == "kernel":
            extra = f"  {msg.get('angle','?')}° / {msg.get('length','?')}px"
        elif name == "ocr":
            extra = f"  Tess: {msg.get('tesseract','—') or '—'}"
        _steps_done[0] += 1
        n = _steps_done[0]
        print(f"  [{n}/6] {label}  ✓{extra}")
        _step_vals[name] = msg

    results = run_pipeline(
        image_path    = args.image,
        gt_path       = args.gt,
        synthetic     = synthetic,
        angle_hint    = angle_hint,
        length_hint   = length_hint,
        n_outer       = args.n_outer,
        n_tv          = args.n_tv,
        lam           = args.lam,
        gamma         = args.gamma,
        verbose       = False,
        step_callback = _cli_step,
    )

    m = results["metrics"]
    p = results["params"]
    o = results["ocr"]

    print(f"\n  {BAR}")
    print(f"  METRICS")
    print(f"  {BAR}")
    print(f"  Angle estimated   : {p['angle']:.1f}°" +
          (f"  (true: {true_angle_cli:.0f}°)" if true_angle_cli else ""))
    print(f"  Length estimated  : {p['length']} px" +
          (f"  (true: {true_len_cli}px)" if true_len_cli else ""))
    print(f"  Sharpness  before : {m['sharpness_input']:.1f}")
    print(f"  Sharpness  after  : {m['sharpness_output']:.1f}  (Δ {m['sharpness_gain']:+.1f})")
    if m.get("psnr") is not None:
        print(f"  PSNR    blurred  : {m['psnr_input']:.2f} dB")
        print(f"  PSNR    deblurred: {m['psnr']:.2f} dB  (Δ {m['psnr']-m['psnr_input']:+.2f} dB)")
        print(f"  SSIM             : {m['ssim']:.4f}")
    print(f"\n  {BAR}")
    print(f"  OCR RESULTS")
    print(f"  {BAR}")
    print(f"  Tesseract : {o['tesseract'] or '(no result)'}")
    print(f"  EasyOCR   : {o['easyocr']   or '(no result)'}")
    print(f"\n  {BAR}")
    print(f"  OUTPUT")
    print(f"  {BAR}")
    print(f"  Directory : {results['output_dir']}")
    print(f"  Figure    : {results['pipeline_figure']}")
    print(f"  {'═'*W}\n")

    if args.latex:
        try:
            from report_updater import update_report, pipeline_fig_latex
            m = results["metrics"]
            fig_rel = Path(results["pipeline_figure"]).name

            metrics_tex = (
                "\\begin{itemize}[noitemsep, topsep=2pt]\n"
                f"  \\item Estimated angle: ${results['params']['angle']:.1f}°$, "
                f"length: ${results['params']['length']}\\,\\mathrm{{px}}$\n"
            )
            if m.get("psnr") is not None:
                metrics_tex += (
                    f"  \\item PSNR blurred: ${m['psnr_input']:.2f}\\,\\mathrm{{dB}}$, "
                    f"deblurred: ${m['psnr']:.2f}\\,\\mathrm{{dB}}$ "
                    f"($\\Delta = {m['psnr']-m['psnr_input']:+.2f}\\,\\mathrm{{dB}}$)\n"
                    f"  \\item SSIM: ${m['ssim']:.4f}$, "
                    f"sharpness gain: ${m['sharpness_gain']:+.1f}$\n"
                )
            metrics_tex += (
                f"  \\item Tesseract OCR: \\texttt{{{results['ocr']['tesseract'] or '(none)'}}}\n"
                f"  \\item EasyOCR: \\texttt{{{results['ocr']['easyocr'] or '(none)'}}}\n"
                "\\end{itemize}"
            )

            update_report(str(HERE / "report_long.tex"), {
                "pipeline_fig":     pipeline_fig_latex(fig_rel),
                "pipeline_metrics": metrics_tex,
            })
        except Exception as e:
            print(f"  LaTeX update failed: {e}")
