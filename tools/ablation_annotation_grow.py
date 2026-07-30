"""Structural ablation of the annotation_grow pipeline.

Two groups of switches are ablated, all relative to the ``full`` reference.

A. Image-enhancement (cost-map preprocessing), three sequential steps:

    1. white-top-hat  (background removal by morphological opening + subtract)
    2. CLAHE          (contrast-limited adaptive histogram equalization)
    3. Sato           (multi-scale vesselness filter)

B. Connection step (the actual annotation linking), ablated internally:

    cost_mode  — "exp" (default) | "linear" | "uniform"
                 uniform = constant cost ⇒ Dijkstra routes by hop-count only
                 (straight-line bridging, ignores image intensity)
    do_prune   — drop the high-cost edge cutoff before MST
    do_mst     — keep the pruned graph as-is instead of reducing to a tree

Configurations (the first, ``full``, is the REFERENCE):

    full         — everything enabled
    no/w_*       — image-enhancement ablations
    linear_cost  — cost_mode="linear"
    uniform_cost — cost_mode="uniform"  (straight-line bridges)
    no_prune     — prune off
    no_mst       — MST off (cycles allowed)

For every config it runs full inference + evaluation across the dataset,
caches per-sample (hd95, avg_hd, cldice) to JSON, then prints a comparison
table. Each non-ref cell receives a ``*`` when a paired one-sided Wilcoxon
signed-rank test (p < 0.05) shows the configuration is *significantly worse*
than ``full``:

    hd95   — worse means cmp > ref
    avg_hd — worse means cmp > ref
    cldice — worse means cmp < ref

Re-running the script reuses cached per-config results unless ``--rerun`` is
passed.

Run:
    uv run python tools/ablation_annotation_grow.py \\
        --data-dir data/ \\
        --output-dir output/ablation_annotation_grow \\
        --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import networkx as nx
import numpy as np
import skimage as ski
from PIL import Image
from scipy import stats
from tqdm import tqdm

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (
    build_component_graph,
    find_meeting_points,
    minimum_spanning_forest,
    prune_edges,
)
from neural_reconstruction.algorithms.annotation_grow.skeleton import (
    build_result_graph,
)
from neural_reconstruction.core.crosses_detection import run_crossing_analysis
from neural_reconstruction.core.evaluation import (
    compute_average_hausdorff_distance,
    compute_cldice,
    compute_hd95,
    extract_graph_points,
)
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.dataset import DatasetLoader, SampleFiles


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linker parameters (kept fixed across ablations; only the switches in
# DEFAULT_SWITCHES differ between configurations).
# ---------------------------------------------------------------------------
FIXED_PARAMS: dict[str, Any] = dict(
    offset_px=50,
    bg_kernel_size=5,
    clahe_clip=30.0,
    clahe_grid=(768, 768),
    sato_sigmas_start=1,
    sato_sigmas_stop=4,
    connectivity=8,
    prune_threshold=20.0,
    min_tree_components=1,
    stub_length_threshold=3,
)

# clDice spatial tolerance as a physical distance (micrometres). It is
# converted to pixels *per sample* via px_um.json so the slack is identical
# across acquisition resolutions: tolerance_px = CLDICE_TOLERANCE_UM / px_um.
CLDICE_TOLERANCE_UM = 1.28
ALPHA = 0.05

# Default switches = the ``full`` reference configuration. Each ablation row
# overrides one (or a few) of these via ``_cfg``.
DEFAULT_SWITCHES: dict[str, object] = {
    # image-enhancement (group A)
    "use_wth": True,
    "use_clahe": True,
    "use_sato": True,
    # connection step (group B)
    "cost_mode": "exp",   # "exp" | "linear" | "uniform"
    "do_connect": True,   # False = skip all bridge connections (annotation only)
    "do_prune": True,
    "do_mst": True,
}


def _cfg(**overrides: object) -> dict[str, object]:
    """Build a switch dict from DEFAULT_SWITCHES with the given overrides."""
    unknown = set(overrides) - set(DEFAULT_SWITCHES)
    if unknown:
        raise KeyError(f"unknown switch keys: {sorted(unknown)}")
    cfg = dict(DEFAULT_SWITCHES)
    cfg.update(overrides)
    return cfg


# Per-ablation switches. The first entry is treated as the reference.
ABLATIONS: list[tuple[str, dict[str, object]]] = [
    ("full",         _cfg()),
    # ── group A: image enhancement ─────────────────────────────────────
    ("no_wth",       _cfg(use_wth=False)),
    ("no_clahe",       _cfg(use_clahe=False)),
    ("no_sato",      _cfg(use_sato=False)),

    ("no",           _cfg(use_wth=False, use_clahe=False, use_sato=False)),
    ("w_bg",         _cfg(use_clahe=False, use_sato=False)),
    ("only_sato",    _cfg(use_wth=False, use_clahe=False)),
    
    # ── group B: connection step ───────────────────────────────────────
    ("linear_cost",  _cfg(cost_mode="linear")),
    ("uniform_cost", _cfg(cost_mode="uniform")),
    ("no_prune",     _cfg(do_prune=False)),
    ("no_mst",       _cfg(do_mst=False)),
    ("no_connect",   _cfg(do_connect=False)),
]
REF_NAME = ABLATIONS[0][0]


# ---------------------------------------------------------------------------
# Preprocessing with on/off toggles (mirrors build_enhanced_image but
# selectively skips stages). Lives in the ablation script so production code
# stays untouched.
# ---------------------------------------------------------------------------
def build_enhanced_image_ablated(
    green: np.ndarray,
    roi_mask: np.ndarray,
    *,
    use_wth: bool,
    use_clahe: bool,
    use_sato: bool,
    bg_kernel_size: int,
    clahe_clip: float,
    clahe_grid: tuple[int, int],
    sato_sigmas: range,
) -> np.ndarray:
    img = green
    if use_wth and bg_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
        )
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.subtract(img, background)

    img = cv2.bitwise_and(img, img, mask=roi_mask)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        img = clahe.apply(img)

    if use_sato:
        img = ski.filters.sato(img, sigmas=sato_sigmas, black_ridges=False)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin) * 255
        img = img.astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def build_cost_map_ablated(enhanced: np.ndarray, *, mode: str) -> np.ndarray:
    """Convert an enhanced image into a Dijkstra traversal cost map.

    Mirrors ``cost_map.build_cost_map`` but supports connection-step ablations:

        "exp"     — cost = exp(1 - norm) - 1   (production: bright ⇒ low cost)
        "linear"  — cost = 1 - norm            (no exponential emphasis)
        "uniform" — cost = 1 everywhere        (intensity ignored; Dijkstra
                    minimises hop count ⇒ straight-line bridging)
    """
    if mode == "uniform":
        return np.ones_like(enhanced, dtype=np.float32)

    norm = enhanced.astype(np.float32) / 255.0
    if mode == "linear":
        cost = 1.0 - norm
    elif mode == "exp":
        cost = np.exp(1.0 - norm) - 1.0
    else:
        raise ValueError(f"unknown cost_mode: {mode!r}")
    return cost.astype(np.float32)


# ---------------------------------------------------------------------------
# A free-function reimplementation of AnnotationGrowLinker.run() that swaps
# in the ablated preprocessing and connection step. The rest of the pipeline
# is identical.
# ---------------------------------------------------------------------------
def run_ablated_inference(
    image: np.ndarray,
    mask: np.ndarray,
    annotation: np.ndarray,
    *,
    switches: dict[str, object],
):
    if image.ndim == 3:
        green = image[:, :, 1]
    else:
        green = image
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if annotation.ndim == 3:
        annotation = annotation[:, :, 0]

    roi_mask = dilate_epidermis_vertically(mask, offset_px=FIXED_PARAMS["offset_px"])

    roi_image = build_enhanced_image_ablated(
        green=green,
        roi_mask=roi_mask,
        use_wth=switches["use_wth"],
        use_clahe=switches["use_clahe"],
        use_sato=switches["use_sato"],
        bg_kernel_size=FIXED_PARAMS["bg_kernel_size"],
        clahe_clip=FIXED_PARAMS["clahe_clip"],
        clahe_grid=FIXED_PARAMS["clahe_grid"],
        sato_sigmas=range(
            FIXED_PARAMS["sato_sigmas_start"], FIXED_PARAMS["sato_sigmas_stop"]
        ),
    )

    roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
    cost_map = build_cost_map_ablated(roi_image, mode=str(switches["cost_mode"]))

    annotation_bin = (roi_annotation > 127).astype(np.uint8)
    annot_labeled = get_components(annotation_bin)
    n_components = int(annot_labeled.max())

    # Connection-step ablations. do_connect=False skips bridging entirely so the
    # annotation components stay isolated (an empty bridge graph ⇒ build_result_graph
    # skeletonises only the annotation bodies). Otherwise prune cutoff and tree
    # (MST) constraint are the remaining switches.
    if switches.get("do_connect", True):
        owner_map, dist_map, prev_y, prev_x = multi_source_dijkstra(
            cost_map,
            annot_labeled,
            connectivity=FIXED_PARAMS["connectivity"],
            roi_mask=(roi_mask > 127),
        )
        connections = find_meeting_points(owner_map, dist_map, prev_y, prev_x)
        g = build_component_graph(connections, n_components)
        if switches["do_prune"]:
            g = prune_edges(g, threshold=FIXED_PARAMS["prune_threshold"])
        # build_result_graph only reads each edge's 'path'; passing the (pruned)
        # graph instead of its MST keeps every surviving connection (cycles allowed).
        mst = minimum_spanning_forest(g) if switches["do_mst"] else g
    else:
        mst = nx.Graph()
    result_graph = build_result_graph(
        mst,
        annotation_bin,
        dilate_radius=3,
        stub_length_threshold=FIXED_PARAMS["stub_length_threshold"],
    )
    _valid_count, labeled_graph = run_crossing_analysis(
        result_graph,
        mask,
        annot_labeled,
        min_tree_components=FIXED_PARAMS["min_tree_components"],
    )
    return labeled_graph, roi_mask


# ---------------------------------------------------------------------------
# Per-sample evaluation (hd95 + cldice). Mirrors evaluate_dataset._evaluate_sample
# but trimmed to just the two metrics we report.
# ---------------------------------------------------------------------------
@dataclass
class SampleMetrics:
    sample_id: str
    hd95: Optional[float]
    avg_hd: Optional[float]
    cldice: Optional[float]
    error: Optional[str] = None


_local = threading.local()


def _topology_builder() -> TopologyBuilder:
    if not hasattr(_local, "tb"):
        _local.tb = TopologyBuilder()
    return _local.tb


def evaluate_sample(
    sample: SampleFiles,
    switches: dict[str, object],
    px_um_ratios: dict[str, float],
) -> SampleMetrics:
    is_complete, reason = sample.is_complete()
    if not is_complete:
        return SampleMetrics(sample.sample_id, None, None, None, error=f"skip:{reason}")
    if sample.label_path is None or not sample.label_path.exists():
        return SampleMetrics(sample.sample_id, None, None, None, error="skip:missing_label")

    try:
        image = np.array(Image.open(sample.image_path))
        mask = np.array(Image.open(sample.mask_path))
        annotation = np.array(Image.open(sample.annotation_path))
        gt_label = np.array(Image.open(sample.label_path))

        pred_graph, roi_mask = run_ablated_inference(
            image, mask, annotation, switches=switches
        )

        if gt_label.ndim == 3:
            gt_label = gt_label[:, :, 0]
        roi_label = cv2.bitwise_and(gt_label, gt_label, mask=roi_mask)

        graph_gt = _topology_builder().build_seed_graph(roi_label)
        pred_pts = extract_graph_points(pred_graph)
        gt_pts = extract_graph_points(graph_gt)

        # HD95 and average Hausdorff distance share the same point sets; both
        # are scaled to microns when a px/um ratio is available.
        ratio = px_um_ratios.get(sample.sample_id)
        scale = ratio if ratio is not None else 1.0
        hd95_val, _, _ = compute_hd95(pred_pts, gt_pts)
        hd95_val *= scale
        avg_hd_val, _, _ = compute_average_hausdorff_distance(pred_pts, gt_pts)
        avg_hd_val *= scale

        # clDice tolerance is a fixed physical distance (microns) converted to
        # pixels per sample so the slack is identical across acquisition
        # resolutions: tolerance_px = CLDICE_TOLERANCE_UM / px_um.
        if ratio is not None and ratio > 0:
            cldice_tol_px = round(CLDICE_TOLERANCE_UM / ratio)
        else:
            cldice_tol_px = round(CLDICE_TOLERANCE_UM)

        cld, _tp, _ts = compute_cldice(
            pred_graph, roi_label, tolerance_px=cldice_tol_px
        )
        return SampleMetrics(
            sample.sample_id, float(hd95_val), float(avg_hd_val), float(cld)
        )

    except Exception as e:
        logger.exception("Sample %s failed: %s", sample.sample_id, e)
        return SampleMetrics(sample.sample_id, None, None, None, error=f"fail:{e}")


# ---------------------------------------------------------------------------
# Caching: one JSON file per ablation, written incrementally.
# ---------------------------------------------------------------------------
def cache_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.json"


def load_cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f).get("samples", {})


def save_cached(
    path: Path, switches: dict[str, object], samples: dict[str, dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {"switches": switches, "fixed_params": _serializable(FIXED_PARAMS),
             "samples": samples},
            f, indent=2,
        )


def _serializable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def run_ablation(
    name: str,
    switches: dict[str, object],
    samples: list[SampleFiles],
    px_um_ratios: dict[str, float],
    out_dir: Path,
    workers: int,
    rerun: bool,
) -> dict[str, dict]:
    path = cache_path(out_dir, name)
    cached = {} if rerun else load_cached(path)
    # Re-run any sample that isn't cached as a clean success, so stale "failed"
    # entries (e.g. from an older code version) don't silently poison the table.
    todo = [
        s for s in samples
        if cached.get(s.sample_id, {}).get("status") != "success"
    ]
    if not todo:
        logger.info("[%s] all %d samples cached, skipping", name, len(samples))
        return cached

    logger.info(
        "[%s] running %d / %d samples (switches=%s)",
        name, len(todo), len(samples), switches,
    )

    results = dict(cached)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(evaluate_sample, s, switches, px_um_ratios): s
            for s in todo
        }
        with tqdm(total=len(futs), desc=f"  {name}") as bar:
            for fut in as_completed(futs):
                m = fut.result()
                row: dict = {"sample_id": m.sample_id}
                if m.error is None:
                    row["status"] = "success"
                    row["hd95"] = m.hd95
                    row["avg_hd"] = m.avg_hd
                    row["cldice"] = m.cldice
                else:
                    row["status"] = "failed"
                    row["error"] = m.error
                results[m.sample_id] = row
                save_cached(path, switches, results)
                bar.update(1)

    return results


# ---------------------------------------------------------------------------
# Wilcoxon comparison table (mirrors compute_pvalues_legacy.py output style)
# ---------------------------------------------------------------------------
METRIC_SPECS = {
    # name: (json key, lower_is_better, decimals)
    "hd95":   ("hd95",   True,  3),
    "avg_hd": ("avg_hd", True,  4),
    "cldice": ("cldice", False, 4),
}


def collect_metric(samples: dict[str, dict], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for sid, s in samples.items():
        if s.get("status") != "success":
            continue
        v = s.get(key)
        if v is None or not np.isfinite(v):
            continue
        out[sid] = float(v)
    return out


def wilcoxon_p_cmp_worse(
    ref_vals: dict[str, float],
    cmp_vals: dict[str, float],
    lower_is_better: bool,
) -> float:
    common = sorted(set(ref_vals) & set(cmp_vals))
    if len(common) < 2:
        return float("nan")
    r = np.asarray([ref_vals[s] for s in common])
    c = np.asarray([cmp_vals[s] for s in common])
    if np.all(r == c):
        return float("nan")
    alternative = "greater" if lower_is_better else "less"
    try:
        res = stats.wilcoxon(c, r, zero_method="wilcox", alternative=alternative)
    except ValueError:
        return float("nan")
    return float(res.pvalue)  # type: ignore[union-attr]


def fmt_pvalue(p: float) -> str:
    """Format a p-value compactly: '<.001' for tiny values, else 3 decimals."""
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def fmt_cell(metric: str, value: float, std: float, p: Optional[float]) -> str:
    if not np.isfinite(value):
        return "nan"
    decimals = METRIC_SPECS[metric][2]
    body = f"{value:.{decimals}f}"
    if np.isfinite(std):
        body += f"±{std:.{decimals}f}"
    if p is not None and np.isfinite(p):
        star = "*" if p < ALPHA else ""
        ptxt = fmt_pvalue(p)
        sep = "" if ptxt.startswith("<") else "="
        body += f" (p{sep}{ptxt}){star}"
    return body


def print_table(rows: list[dict], metrics: list[str], ref_name: str) -> None:
    name_w = max(
        max((len(r["name"]) for r in rows), default=0),
        len(ref_name) + len(" (ref)"),
        12,
    )
    col_w = {
        m: max(
            len(m),
            max((len(r["cells"][m]) for r in rows), default=0),
        )
        for m in metrics
    }
    header = f"{'ablation':<{name_w}} {'n':>3}"
    for m in metrics:
        header += f" {m:>{col_w[m]}}"
    bar = "-" * len(header)

    cmp_rows = [r for r in rows if r["name"] != ref_name]
    ref_row = next(r for r in rows if r["name"] == ref_name)

    print()
    print(header)
    print(bar)
    for r in cmp_rows:
        line = f"{r['name']:<{name_w}} {r['n']:>3}"
        for m in metrics:
            line += f" {r['cells'][m]:>{col_w[m]}}"
        print(line)
    print(bar)
    line = f"{ref_row['name'] + ' (ref)':<{name_w}} {ref_row['n']:>3}"
    for m in metrics:
        line += f" {ref_row['cells'][m]:>{col_w[m]}}"
    print(line)
    print()
    print(
        f"  (p=…) = paired one-sided Wilcoxon p-value; "
        f"* marks p < {ALPHA:g} (ablation significantly WORSE than {ref_name})"
    )
    print("  (lower is better: hd95, avg_hd; higher is better: cldice)")


def write_csv(rows: list[dict], metrics: list[str], path: Path) -> None:
    fields = ["ablation", "is_reference", "n"]
    for m in metrics:
        fields += [m, f"{m}_std", f"{m}_p_cmp_worse_than_ref"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {
                "ablation": r["name"],
                "is_reference": r["name"] == REF_NAME,
                "n": r["n"],
            }
            for m in metrics:
                row[m] = r["values"][m]
                row[f"{m}_std"] = r["stds"][m]
                row[f"{m}_p_cmp_worse_than_ref"] = r["pvalues"][m]
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def load_px_um(data_dir: Path) -> dict[str, float]:
    path = data_dir / "px_um.json"
    if not path.exists():
        logger.warning("px_um.json not found at %s — HD95 stays in pixels", path)
        return {}
    with path.open() as f:
        return json.load(f)


def build_summary_rows(
    ablation_samples: dict[str, dict[str, dict]],
) -> list[dict]:
    """Aggregate per-ablation samples into table rows + run Wilcoxon vs. ref."""
    ref_metric_vals: dict[str, dict[str, float]] = {
        m: collect_metric(ablation_samples[REF_NAME], METRIC_SPECS[m][0])
        for m in METRIC_SPECS
    }

    rows: list[dict] = []
    for name, _switches in ABLATIONS:
        samples = ablation_samples[name]
        values: dict[str, float] = {}
        stds: dict[str, float] = {}
        pvalues: dict[str, Optional[float]] = {}
        cells: dict[str, str] = {}
        for metric, (key, lower_is_better, _dec) in METRIC_SPECS.items():
            per_sample = collect_metric(samples, key)
            vals = list(per_sample.values())
            v = float(np.mean(vals)) if vals else float("nan")
            # sample std (ddof=1); needs >=2 samples, else undefined
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            values[metric] = v
            stds[metric] = s
            if name == REF_NAME:
                pvalues[metric] = None
            else:
                pvalues[metric] = wilcoxon_p_cmp_worse(
                    ref_metric_vals[metric], per_sample, lower_is_better
                )
            cells[metric] = fmt_cell(metric, v, s, pvalues[metric])
        n_success = sum(1 for s in samples.values() if s.get("status") == "success")
        rows.append({
            "name": name,
            "n": n_success,
            "values": values,
            "stds": stds,
            "pvalues": pvalues,
            "cells": cells,
        })
    return rows


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Dataset root (containing sample subdirs and px_um.json).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where per-ablation results.json files are cached.")
    parser.add_argument("--sample-ids", nargs="+", default=None,
                        help="Restrict to these sample IDs (default: all).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Per-ablation thread count.")
    parser.add_argument("--rerun", action="store_true",
                        help="Ignore cached results and recompute everything.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional CSV output for the summary table.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = DatasetLoader(args.data_dir).load_samples(args.sample_ids)
    if not samples:
        logger.error("No samples found under %s", args.data_dir)
        return 1
    logger.info("Loaded %d samples", len(samples))

    px_um = load_px_um(args.data_dir)

    ablation_samples: dict[str, dict[str, dict]] = {}
    for name, switches in ABLATIONS:
        ablation_samples[name] = run_ablation(
            name=name,
            switches=switches,
            samples=samples,
            px_um_ratios=px_um,
            out_dir=args.output_dir,
            workers=args.workers,
            rerun=args.rerun,
        )

    rows = build_summary_rows(ablation_samples)
    print_table(rows, list(METRIC_SPECS.keys()), REF_NAME)

    if args.csv is not None:
        write_csv(rows, list(METRIC_SPECS.keys()), args.csv)
        print(f"CSV written to: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
