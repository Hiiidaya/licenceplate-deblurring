"""
report_updater.py — Auto-update tagged sections of the LaTeX report
====================================================================
Finds markers of the form:

    %% AUTO:section_name_start %%
    ... old content ...
    %% AUTO:section_name_end %%

and replaces the content between them with the supplied text.

Usage (standalone)
------------------
    from report_updater import update_report
    update_report("report_long.tex", {"bench_latex": table_str, "pipeline_fig": fig_latex})

Called automatically from pipeline_run.py and synthetic_bench.py when --latex is passed.

Supported section keys
----------------------
    pipeline_fig      — \includegraphics for the latest pipeline figure
    bench_table       — LaTeX tabular from synthetic benchmark
    ocr_table         — OCR Tesseract vs EasyOCR comparison table
    psnr_chart_fig    — \includegraphics for benchmark bar chart
    scatter_fig       — \includegraphics for PSNR vs length scatter
"""

import re
import sys
import shutil
import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
DEFAULT_REPORT = HERE / "report_long.tex"

# ── Marker regex ──────────────────────────────────────────────────────────────
_START = r"%%\s*AUTO:({key})_start\s*%%"
_END   = r"%%\s*AUTO:({key})_end\s*%%"


def _replace_section(text: str, key: str, content: str) -> str:
    """Replace content between AUTO markers for `key`. Returns updated text."""
    pattern = (
        rf"(%%\s*AUTO:{re.escape(key)}_start\s*%%)"
        rf"(.*?)"
        rf"(%%\s*AUTO:{re.escape(key)}_end\s*%%)"
    )
    # Use a callable replacement so backslashes in `content` aren't treated as
    # regex escape sequences (e.g. \c in \centering would raise re.error).
    def _repl(m):
        return m.group(1) + "\n" + content + "\n" + m.group(3)

    new_text, n = re.subn(pattern, _repl, text, flags=re.DOTALL)
    if n == 0:
        print(f"  [report_updater] WARNING: marker AUTO:{key} not found in report.")
    else:
        print(f"  [report_updater] Updated section: {key}")
    return new_text


def update_report(report_path: str, sections: dict, backup: bool = True) -> bool:
    """
    Update tagged sections in a LaTeX report file.

    Parameters
    ----------
    report_path : str — path to the .tex file
    sections    : dict[str, str] — {section_key: latex_content}
    backup      : bool — if True, save a timestamped backup before writing

    Returns
    -------
    True on success, False on error.
    """
    path = Path(report_path)
    if not path.exists():
        print(f"  [report_updater] ERROR: report not found: {path}")
        return False

    text = path.read_text(encoding="utf-8")

    if backup:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(f".bak_{ts}.tex")
        shutil.copy2(path, bak)
        print(f"  [report_updater] Backup saved → {bak.name}")

    for key, content in sections.items():
        text = _replace_section(text, key, content)

    path.write_text(text, encoding="utf-8")
    print(f"  [report_updater] Report saved → {path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Helper builders — generate the LaTeX snippets from pipeline results
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_fig_latex(fig_path: str, caption: str = "", label: str = "fig:pipeline") -> str:
    """Generate a \figure block for a pipeline output image."""
    cap = caption or "Full pipeline output: blurred → FFT → Wiener → HQS deblurred."
    return (
        "\\begin{figure}[H]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{fig_path}}}\n"
        f"  \\caption{{{cap}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{figure}"
    )


def bench_table_latex(rows: list[dict]) -> str:
    """Generate PSNR/SSIM LaTeX table from synthetic benchmark rows."""
    valid = [r for r in rows if "psnr_deblur" in r and r.get("psnr_deblur") is not None]
    if not valid:
        return "% No valid benchmark results."

    import numpy as np
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Synthetic Benchmark --- PSNR, SSIM, and Sharpness Gain "
        "across 10 motion blur configurations on 8 unblurred reference images.}",
        "\\label{tab:synth_bench}",
        "\\small",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        "Image & Config & "
        "PSNR$_{\\text{blur}}$ & PSNR$_{\\text{deblur}}$ & "
        "$\\Delta$PSNR & SSIM & $\\Delta$Sharp \\\\",
        "\\midrule",
    ]
    for r in valid:
        gain  = r.get("psnr_gain", 0) or 0
        sgain = r.get("sharp_gain", 0) or 0
        lines.append(
            f"\\texttt{{{r['image']}}} & {r['config']} & "
            f"{r['psnr_blurred']:.2f} & {r['psnr_deblur']:.2f} & "
            f"{gain:+.2f} & {r['ssim']:.4f} & "
            f"{sgain:+.1f} \\\\"
        )
    avg_psnr = float(np.mean([r["psnr_deblur"]  for r in valid]))
    avg_ssim = float(np.mean([r["ssim"]          for r in valid]))
    avg_gain = float(np.mean([r.get("psnr_gain", 0) or 0 for r in valid]))
    lines += [
        "\\midrule",
        f"\\textbf{{Average}} & & & {avg_psnr:.2f} & {avg_gain:+.2f} & {avg_ssim:.4f} & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def ocr_table_latex(ocr_rows: list[dict]) -> str:
    """
    Generate OCR comparison table (Tesseract vs EasyOCR).

    Each row: {image, gt, tesseract, easyocr, tess_sim, easy_sim}
    """
    if not ocr_rows:
        return "% No OCR comparison data."

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{OCR Comparison --- Tesseract vs EasyOCR on deblurred license plates. "
        "Similarity = character-level positional accuracy.}",
        "\\label{tab:ocr_compare}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llllcc@{}}",
        "\\toprule",
        "Image & Ground Truth & Tesseract & EasyOCR & Tess Sim & Easy Sim \\\\",
        "\\midrule",
    ]
    for r in ocr_rows:
        ts   = r.get("tess_sim",  0) or 0
        es   = r.get("easy_sim",  0) or 0
        tess = r.get("tesseract", "—") or "—"
        easy = r.get("easyocr",   "—") or "—"
        winner_t = "\\textcolor{mygreen}{" if ts > es else ""
        winner_e = "\\textcolor{mygreen}{" if es > ts else ""
        close_t  = "}" if ts > es else ""
        close_e  = "}" if es > ts else ""
        lines.append(
            f"\\texttt{{{r['image']}}} & "
            f"\\texttt{{{r.get('gt', '?')}}} & "
            f"{winner_t}\\texttt{{{tess}}}{close_t} & "
            f"{winner_e}\\texttt{{{easy}}}{close_e} & "
            f"{ts:.2f} & {es:.2f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def psnr_chart_latex(chart_path: str) -> str:
    """LaTeX figure for the benchmark bar chart."""
    return (
        "\\begin{figure}[H]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{chart_path}}}\n"
        "  \\caption{Synthetic benchmark bar charts: PSNR (blurred vs.\\ deblurred), "
        "SSIM, and Laplacian sharpness gain across all 10 blur configurations.}\n"
        "  \\label{fig:bench_bars}\n"
        "\\end{figure}"
    )


def scatter_fig_latex(scatter_path: str) -> str:
    """LaTeX figure for PSNR vs blur length scatter."""
    return (
        "\\begin{figure}[H]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.75\\linewidth]{{{scatter_path}}}\n"
        "  \\caption{PSNR of deblurred output vs.\\ applied blur severity (kernel length). "
        "Longer kernels generally yield lower PSNR as estimation becomes harder.}\n"
        "  \\label{fig:bench_scatter}\n"
        "\\end{figure}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI — for testing / manual invocation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Manually update report_long.tex markers")
    ap.add_argument("--report", default=str(DEFAULT_REPORT), help="Path to .tex file")
    ap.add_argument("--list",   action="store_true", help="List AUTO markers found in report")
    args = ap.parse_args()

    rp = Path(args.report)
    if not rp.exists():
        print(f"Report not found: {rp}")
        sys.exit(1)

    text = rp.read_text(encoding="utf-8")

    if args.list:
        markers = re.findall(r"%%\s*AUTO:(\w+)_start\s*%%", text)
        if markers:
            print(f"AUTO markers in {rp.name}:")
            for m in markers:
                print(f"  {m}")
        else:
            print("No AUTO markers found.")
    else:
        print(f"Use --list to show markers in {rp.name}")
        print("Import update_report() from this module to update sections programmatically.")
