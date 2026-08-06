"""Topology metrics (cldice / hd95 / avg_hd / count / length) for existing
LinkerResult pickles (mst_15 / annotation_grow) vs GT — folds in
tools/length_rmse.py's total-length comparison so everything lives in one CSV.

GT graph is built on the fly from data_0510/{id}/label.png, cropped with each
method's own ROI mask (stored in the pickled LinkerResult), via TopologyBuilder
— identical to tools/ablation_annotation_grow.py's evaluate_sample.

Run:
    uv run python tools/topology_metrics.py
"""

import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import pearsonr, wilcoxon

from neural_reconstruction.core.evaluation import (
    compute_average_hausdorff_distance,
    compute_cldice,
    compute_hd95,
    extract_graph_points,
)
from neural_reconstruction.core.topology import TopologyBuilder

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "msf": "output/ref/mst_15",
    "Proposed": "output/ref/annotation_grow",
}
DATA_DIR = ROOT / "data_0510"
OUT_CSV = ROOT / "output" / "topology_metrics.csv"
OUT_SUMMARY_CSV = ROOT / "output" / "topology_metrics_summary.csv"

# clDice tolerance as a physical distance (micrometres), matches
# tools/ablation_annotation_grow.py's CLDICE_TOLERANCE_UM.
CLDICE_TOLERANCE_UM = 1.28

METRICS = ["cldice", "hd95", "avg_hd"]
LOWER_IS_BETTER = {"cldice": False, "hd95": True, "avg_hd": True}


def path_length(path) -> float:
    if path is None or len(path) < 2:
        return 0.0
    pts = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def graph_length(graph) -> float:
    return sum(path_length(d.get("path")) for _, _, d in graph.edges(data=True))


def sample_metrics(pred_graph, gt_label: np.ndarray, mask: np.ndarray, ratio: float) -> dict:
    roi_label = cv2.bitwise_and(gt_label, gt_label, mask=mask)
    gt_graph = TopologyBuilder().build_seed_graph(roi_label)

    pred_pts = extract_graph_points(pred_graph)
    gt_pts = extract_graph_points(gt_graph)

    hd95, _, _ = compute_hd95(pred_pts, gt_pts)
    avg_hd, _, _ = compute_average_hausdorff_distance(pred_pts, gt_pts)
    tol_px = round(CLDICE_TOLERANCE_UM / ratio) if ratio > 0 else round(CLDICE_TOLERANCE_UM)
    cldice, _, _ = compute_cldice(pred_graph, roi_label, tolerance_px=tol_px)

    return {"cldice": cldice, "hd95": hd95 * ratio, "avg_hd": avg_hd * ratio}


def add_scalar_error_stats(summary_rows: dict, rows: list, prefix: str, gt_key: str) -> None:
    """MAE/RMSE (Wilcoxon p, msf vs Proposed) + Pearson r (its own t-test p)
    for a scalar quantity (count, length_um, ...) predicted per method vs GT."""
    gt_vals = np.array([r[gt_key] for r in rows], dtype=float)
    diffs = {
        name: np.array([r[f"{name}_{prefix}"] for r in rows], dtype=float) - gt_vals
        for name in METHODS
    }

    _, mae_p = wilcoxon(np.abs(diffs["msf"]), np.abs(diffs["Proposed"]), alternative="greater")
    _, rmse_p = wilcoxon(diffs["msf"] ** 2, diffs["Proposed"] ** 2, alternative="greater")
    mae_p_by_name = {"msf": mae_p, "Proposed": None}
    rmse_p_by_name = {"msf": rmse_p, "Proposed": None}

    for name in METHODS:
        diff = diffs[name]
        ae = np.abs(diff)
        summary_rows[name][f"{prefix}_mae"] = ae.mean()
        summary_rows[name][f"{prefix}_mae_std"] = ae.std(ddof=1)
        summary_rows[name][f"{prefix}_mae_p"] = mae_p_by_name[name]
        summary_rows[name][f"{prefix}_rmse"] = np.sqrt((diff ** 2).mean())
        summary_rows[name][f"{prefix}_rmse_std"] = diff.std(ddof=1)
        summary_rows[name][f"{prefix}_rmse_p"] = rmse_p_by_name[name]

        pred_vals = gt_vals + diff
        n = len(pred_vals)
        r, r_p = pearsonr(pred_vals, gt_vals)
        summary_rows[name][f"{prefix}_pearson_r"] = r
        # Standard error of r under H0 (same quantity the t-test itself uses).
        summary_rows[name][f"{prefix}_pearson_r_std"] = np.sqrt((1 - r ** 2) / (n - 2)) if n > 2 else float("nan")
        summary_rows[name][f"{prefix}_pearson_p"] = r_p


def main() -> None:
    px_um = json.loads((DATA_DIR / "px_um.json").read_text())
    gt_counts = json.loads((DATA_DIR / "count.json").read_text())

    sample_ids = sorted(p.stem for p in (ROOT / METHODS["msf"]).glob("*.pkl"))

    rows = []
    for sid in sample_ids:
        ratio = px_um.get(sid)
        gt_count = gt_counts.get(sid)
        if ratio is None or gt_count is None:
            continue

        gt_label = np.array(Image.open(DATA_DIR / sid / "label.png"))
        if gt_label.ndim == 3:
            gt_label = gt_label[:, :, 0]

        row = {"sample_id": sid, "gt_count": gt_count}
        msf_mask = None
        for name, rel in METHODS.items():
            with open(ROOT / rel / f"{sid}.pkl", "rb") as f:
                result = pickle.load(f)
            m = sample_metrics(result.graph, gt_label, result.mask, ratio)
            for metric in METRICS:
                row[f"{name}_{metric}"] = m[metric]
            row[f"{name}_count"] = result.valid_count
            row[f"{name}_length_um"] = graph_length(result.graph) * ratio
            if name == "msf":
                msf_mask = result.mask

        # GT length uses msf's ROI mask (both methods' masks are effectively
        # identical) — same convention as tools/length_rmse.py.
        gt_roi_label = cv2.bitwise_and(gt_label, gt_label, mask=msf_mask)
        gt_graph = TopologyBuilder().build_seed_graph(gt_roi_label)
        row["gt_length_um"] = graph_length(gt_graph) * ratio
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-sample CSV: {OUT_CSV}")

    # Paired Wilcoxon signed-rank, H1: msf significantly worse than Proposed
    # (same convention as tools/length_rmse.py: p attached to msf's row only).
    summary_rows = {name: {"method": name} for name in METHODS}
    for metric in METRICS:
        lower_is_better = LOWER_IS_BETTER[metric]
        msf_vals = np.array([r[f"msf_{metric}"] for r in rows])
        proposed_vals = np.array([r[f"Proposed_{metric}"] for r in rows])
        alt = "greater" if lower_is_better else "less"  # msf worse than Proposed
        try:
            _, p = wilcoxon(msf_vals, proposed_vals, alternative=alt)
        except ValueError:
            p = float("nan")
        p_by_name = {"msf": p, "Proposed": None}

        for name in METHODS:
            vals = np.array([r[f"{name}_{metric}"] for r in rows])
            summary_rows[name][metric] = vals.mean()
            summary_rows[name][f"{metric}_std"] = vals.std(ddof=1)
            summary_rows[name][f"{metric}_p"] = p_by_name[name]

    # Count (valid_count vs GT count.json) and total length (vs GT topology
    # length): MAE/RMSE like tools/length_rmse.py, plus Pearson correlation.
    add_scalar_error_stats(summary_rows, rows, "count", "gt_count")
    add_scalar_error_stats(summary_rows, rows, "length_um", "gt_length_um")

    with open(OUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows["msf"].keys()))
        writer.writeheader()
        writer.writerows(summary_rows.values())
    print(f"summary CSV: {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
