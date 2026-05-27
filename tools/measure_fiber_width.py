"""Measure the fiber half-width distribution across all GT labels, in microns.

Motivation
----------
The evaluation tolerance must sit *above* the noise floor of the ground-truth
annotation itself. A GT centerline, after skeletonization, is one pixel wide —
but its localization relative to the true anatomical center is uncertain, and
the lower bound of that uncertainty is the fiber half-width. If the tolerance
is smaller than the half-width, the comparison mistakes the GT's own centerline
ambiguity for reconstruction error.

This tool reports the half-width distribution in **micrometres**, so the
tolerance can be chosen directly in physical units (resolution-independent).

Method
------
For every ``label.png`` (GT label):

  1. ``binary  = label > 0``
  2. ``edt     = distance_transform_edt(binary)`` — each foreground pixel's
     distance to the nearest boundary.
  3. ``skel    = skeletonize(binary)`` — single-pixel centerline.
  4. ``half_widths_px = edt[skel]`` — the EDT value on each centerline pixel
     is the distance from the centerline to the nearest edge, i.e. the fiber
     half-width at that point. No division by 2 is needed.
  5. ``half_widths_um = half_widths_px * (um/px of that sample)``.

This dataset mixes pixel scales (e.g. 0.32 and 0.64 um/px), so the conversion
is done per sample *before* pooling — the pooled micron distribution is
therefore comparable across the whole dataset.

Reported statistics
-------------------
Median, mean and a high percentile (P90 / P95) of the half-width — these are
robust. The maximum is shown for reference only: it is dominated by the single
thickest fiber or a staining halo and should not anchor any argument.

Run:
    uv run python tools/measure_fiber_width.py
    uv run python tools/measure_fiber_width.py --data-dir data_orig --k 1.0
    uv run python tools/measure_fiber_width.py --output output/fiber_width.csv \\
        --plot output/fiber_width_hist.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import percentileofscore
from skimage.morphology import skeletonize
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-sample measurement
# ---------------------------------------------------------------------------
def measure_half_widths_px(label_path: Path) -> np.ndarray:
    """Return the per-centerline-pixel half-width array (pixels) for one label.

    Empty array if the label has no foreground / no centerline.
    """
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[:, :, 0]
    binary = label > 0
    if not binary.any():
        return np.empty(0, dtype=np.float64)

    edt = np.asarray(distance_transform_edt(binary))
    skeleton = skeletonize(binary)
    return edt[skeleton].astype(np.float64)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
_STAT_KEYS = ("n", "mean", "median", "p90", "p95", "std", "min", "max")


def summarize(values: np.ndarray) -> dict[str, float]:
    """Robust summary of a 1-D half-width array."""
    if values.size == 0:
        return {k: float("nan") for k in _STAT_KEYS}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _fmt_summary(label: str, s: dict[str, float]) -> str:
    if not np.isfinite(s["median"]):
        return f"  {label}: (no data)"
    return (
        f"  {label}:\n"
        f"    n (centerline px) : {int(s['n']):,}\n"
        f"    median half-width : {s['median']:.3f} um\n"
        f"    mean   half-width : {s['mean']:.3f} um\n"
        f"    P90    half-width : {s['p90']:.3f} um\n"
        f"    P95    half-width : {s['p95']:.3f} um\n"
        f"    std               : {s['std']:.3f} um\n"
        f"    max (ref. only)   : {s['max']:.3f} um  "
        f"<- outlier-sensitive, do not anchor arguments on this"
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def save_histogram(half_widths_um: np.ndarray, k: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize(half_widths_um)
    fig, ax = plt.subplots(figsize=(8, 5))
    upper = float(np.percentile(half_widths_um, 99.5))
    ax.hist(half_widths_um, bins=80, range=(0, upper), color="#4C72B0",
            alpha=0.85, edgecolor="white", linewidth=0.3)
    for x, name, color in [
        (s["median"], f"median = {s['median']:.2f} um", "#2A2A2A"),
        (s["p90"], f"P90 = {s['p90']:.2f} um", "#DD8452"),
        (s["p95"], f"P95 = {s['p95']:.2f} um", "#C44E52"),
        (k, f"k = {k:g} um", "#55A868"),
    ]:
        ax.axvline(x, color=color, linestyle="--", linewidth=1.6, label=name)
    ax.set_xlabel("Fiber half-width (micrometres)")
    ax.set_ylabel("Centerline pixel count")
    ax.set_title("GT fiber half-width distribution (all centerline pixels)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the GT fiber half-width distribution in microns "
                    "via EDT on the skeletonized centerline."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data_orig"),
        help="Directory whose subfolders (sample IDs) contain label.png. "
             "Default: data_orig",
    )
    parser.add_argument(
        "--px-um", type=Path, default=None,
        help="Path to px_um.json (sample_id -> um/px). "
             "Default: <data-dir>/px_um.json",
    )
    parser.add_argument(
        "--k", type=float, default=1.0,
        help="Candidate tolerance in MICRONS to locate within the "
             "distribution. Default: 1.0",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional CSV path for per-sample statistics.",
    )
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="Optional PNG path for the pooled half-width histogram.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        print(f"Error: --data-dir not found: {data_dir}", file=sys.stderr)
        return 1

    px_um_path = args.px_um or (data_dir / "px_um.json")
    if not px_um_path.exists():
        print(f"Error: px_um.json required for micron conversion, not found "
              f"at {px_um_path}", file=sys.stderr)
        return 1
    with px_um_path.open(encoding="utf-8") as f:
        px_um: dict[str, float] = json.load(f)

    sample_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and (p / "label.png").exists()
    )
    if not sample_dirs:
        print(f"Error: no <sample>/label.png under {data_dir}", file=sys.stderr)
        return 1

    print(f"Measuring fiber half-widths in {len(sample_dirs)} GT label(s) "
          f"from {data_dir}\n")

    per_sample: list[dict] = []          # rows for CSV / breakdown
    all_um: list[np.ndarray] = []        # pooled micron half-widths
    by_scale: dict[float, list[np.ndarray]] = {}   # micron half-widths per um/px

    for sample_dir in tqdm(sample_dirs, desc="Samples"):
        sid = sample_dir.name
        ratio = px_um.get(sid)
        if ratio is None:
            print(f"  [skip] {sid}: not in px_um.json — cannot convert to um",
                  file=sys.stderr)
            continue
        ratio = float(ratio)

        hw_px = measure_half_widths_px(sample_dir / "label.png")
        if hw_px.size == 0:
            print(f"  [skip] {sid}: empty label / no centerline", file=sys.stderr)
            continue

        hw_um = hw_px * ratio
        all_um.append(hw_um)
        by_scale.setdefault(ratio, []).append(hw_um)
        per_sample.append({
            "sample_id": sid,
            "px_um": ratio,
            "stats": summarize(hw_um),
        })

    if not all_um:
        print("Error: no measurable labels.", file=sys.stderr)
        return 1

    pooled_um = np.concatenate(all_um)
    s = summarize(pooled_um)

    print("=" * 72)
    print("POOLED HALF-WIDTH DISTRIBUTION — MICRONS "
          "(all centerline pixels, all samples)")
    print("=" * 72)
    print(_fmt_summary("Half-width (um)", s))

    # Per-scale breakdown: confirms the two acquisition resolutions agree on
    # the physical fiber caliber.
    if len(by_scale) > 1:
        print("\n" + "-" * 72)
        print("BREAKDOWN BY ACQUISITION SCALE (sanity check — should roughly agree)")
        print("-" * 72)
        for ratio in sorted(by_scale):
            grp = np.concatenate(by_scale[ratio])
            gs = summarize(grp)
            print(f"  {ratio:g} um/px  ({len(by_scale[ratio])} samples):  "
                  f"median {gs['median']:.3f} um / "
                  f"P90 {gs['p90']:.3f} um / P95 {gs['p95']:.3f} um")

    # Locate the candidate tolerance k within the distribution.
    k = args.k
    k_pct = percentileofscore(pooled_um, k, kind="weak")
    print("\n" + "-" * 72)
    print("INTERPRETATION — choosing the tolerance (in microns)")
    print("-" * 72)
    print("  Centerline localization uncertainty lower bound = fiber half-width.")
    print("  Pick the tolerance >= a high percentile of the half-width so the")
    print("  comparison happens above the annotation noise floor:")
    print(f"    median half-width : {s['median']:.3f} um")
    print(f"    P90    half-width : {s['p90']:.3f} um   <- covers 90% of centerlines")
    print(f"    P95    half-width : {s['p95']:.3f} um   <- covers 95% of centerlines")
    print(f"  Candidate k = {k:g} um sits at the {k_pct:.1f}th percentile")
    print(f"    -> k = {k:g} um covers {k_pct:.1f}% of all GT centerline half-widths.")
    print("=" * 72)

    if args.output is not None:
        _write_csv(per_sample, args.output)
        print(f"\nPer-sample CSV written to: {args.output}")

    if args.plot is not None:
        save_histogram(pooled_um, k, args.plot)
        print(f"Histogram written to: {args.plot}")

    return 0


def _write_csv(per_sample: list[dict], out_path: Path) -> None:
    fields = [
        "sample_id", "px_um", "n_centerline_px",
        "median_um", "mean_um", "p90_um", "p95_um", "max_um",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_sample:
            st = row["stats"]
            writer.writerow({
                "sample_id": row["sample_id"],
                "px_um": row["px_um"],
                "n_centerline_px": int(st["n"]),
                "median_um": st["median"], "mean_um": st["mean"],
                "p90_um": st["p90"], "p95_um": st["p95"], "max_um": st["max"],
            })


if __name__ == "__main__":
    sys.exit(main())
