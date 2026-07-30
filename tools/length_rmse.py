"""Total topology length error (mst_15 / annotation_grow vs GT) in µm.

For each sample, total length = sum of Euclidean path length over every graph
edge (NOT the 'weight' attribute — that's an A* cost, ~1e-05 for cheap edges,
not a real length). GT graph is built on the fly from data_0510/{id}/label.png,
cropped with the sample's ROI mask (identical for both methods) via
TopologyBuilder, exactly like tools/evaluate_dataset.py does for Hausdorff.

Run:
    uv run python tools/length_rmse.py
"""

import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon

from neural_reconstruction.core.topology import TopologyBuilder

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "msf": "output/ref/mst_15",
    "Proposed": "output/ref/annotation_grow",
}
DATA_DIR = ROOT / "data_0510"
OUT_CSV = ROOT / "output" / "length_rmse.csv"


def path_length(path) -> float:
    if path is None or len(path) < 2:
        return 0.0
    pts = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def graph_length(graph) -> float:
    return sum(path_length(d.get("path")) for _, _, d in graph.edges(data=True))


def gt_length_px(sample_id: str, mask: np.ndarray) -> float:
    label_path = DATA_DIR / sample_id / "label.png"
    gt_label = np.array(Image.open(label_path))
    roi_label = cv2.bitwise_and(gt_label, gt_label, mask=mask)
    gt_graph = TopologyBuilder().build_seed_graph(roi_label)
    return graph_length(gt_graph)


def main() -> None:
    px_um = json.loads((DATA_DIR / "px_um.json").read_text())

    sample_ids = sorted(
        p.stem for p in (ROOT / METHODS["msf"]).glob("*.pkl")
    )

    rows = []
    for sid in sample_ids:
        ratio = px_um.get(sid)
        if ratio is None:
            continue

        with open(ROOT / METHODS["msf"] / f"{sid}.pkl", "rb") as f:
            msf_result = pickle.load(f)

        gt_um = gt_length_px(sid, msf_result.mask) * ratio

        row = {"sample_id": sid, "gt_length_um": gt_um}
        for name, rel in METHODS.items():
            pkl_path = ROOT / rel / f"{sid}.pkl"
            with open(pkl_path, "rb") as f:
                result = pickle.load(f)
            pred_um = graph_length(result.graph) * ratio
            row[f"{name}_length_um"] = pred_um
            row[f"{name}_diff_um"] = pred_um - gt_um
            row[f"{name}_diff_pct"] = 100.0 * (pred_um - gt_um) / gt_um
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-sample CSV: {OUT_CSV}")

    # Wilcoxon signed-rank, paired per-sample, H1: msf significantly worse.
    # MAE  -> compare |error|   (mean of this is exactly MAE)
    # RMSE -> compare error^2   (mean of this is exactly RMSE^2)
    msf_diffs = np.array([r["msf_diff_um"] for r in rows])
    proposed_diffs = np.array([r["Proposed_diff_um"] for r in rows])
    _, p_mae = wilcoxon(np.abs(msf_diffs), np.abs(proposed_diffs), alternative="greater")
    _, p_rmse = wilcoxon(msf_diffs ** 2, proposed_diffs ** 2, alternative="greater")
    p_mae_by_name = {"msf": p_mae, "Proposed": None}
    p_rmse_by_name = {"msf": p_rmse, "Proposed": None}

    header = (f"{'方法':<12} {'MAE±SD':>18} {'p(MAE)':>10} "
              f"{'RMSE±SD':>18} {'p(RMSE)':>10}")
    print(header)
    print("-" * len(header))
    for name in METHODS:
        diffs = np.array([r[f"{name}_diff_um"] for r in rows])
        ae = np.abs(diffs)
        mae, mae_sd = ae.mean(), ae.std(ddof=1)
        rmse, rmse_sd = np.sqrt((diffs ** 2).mean()), diffs.std(ddof=1)
        p_mae_str = f"{p_mae_by_name[name]:.4g}" if p_mae_by_name[name] is not None else "-"
        p_rmse_str = f"{p_rmse_by_name[name]:.4g}" if p_rmse_by_name[name] is not None else "-"
        print(f"{name:<12} {f'{mae:.2f}±{mae_sd:.2f}':>18} {p_mae_str:>10} "
              f"{f'{rmse:.2f}±{rmse_sd:.2f}':>18} {p_rmse_str:>10}")


if __name__ == "__main__":
    main()
