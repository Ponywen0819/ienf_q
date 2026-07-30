"""Count-error MAE / RMSE (with std dev) per method from results.json files.

Each results.json has per-sample valid_count_pred / gt_count. The signed count
error is e = valid_count_pred - gt_count. We report:

    bias  = mean(e)                      (signed: +over-counts, -under-counts)
    MAE   = mean(|e|)      ± SD(|e|)
    RMSE  = sqrt(mean(e^2)) ± SD(e)      (std of per-sample errors, count units;
                                          RMSE^2 ~ bias^2 + SD(e)^2)

Run:
    python tools/count_rmse.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "mst":      "output/ref/mst_15/results.json",
    "Proposed": "output/ref/annotation_grow/results.json",
}


def count_errors(results_path: Path) -> np.ndarray:
    """Signed per-sample count errors (pred - gt) over successful samples."""
    d = json.loads(results_path.read_text())
    errs = []
    for s in d["samples"].values():
        if s.get("status") != "success":
            continue
        pred, gt = s.get("valid_count_pred"), s.get("gt_count")
        if pred is None or gt is None:
            continue
        errs.append(pred - gt)
    return np.asarray(errs, dtype=float)


def main() -> None:
    header = f"{'method':<12} {'n':>3} {'bias':>8} {'MAE±SD':>16} {'RMSE±SD':>16}"
    print(header)
    print("-" * len(header))
    for name, rel in METHODS.items():
        e = count_errors(ROOT / rel)
        n = len(e)
        if n == 0:
            print(f"{name:<12} {0:>3}  (no samples)")
            continue
        bias = e.mean()
        ae = np.abs(e)
        mae, mae_sd = ae.mean(), (ae.std(ddof=1) if n > 1 else float("nan"))
        rmse = np.sqrt((e ** 2).mean())
        # SD of the per-sample errors (count units), same recipe as MAE's SD.
        e_sd = e.std(ddof=1) if n > 1 else float("nan")
        print(f"{name:<12} {n:>3} {bias:>+8.3f} "
              f"{f'{mae:.3f}±{mae_sd:.3f}':>16} "
              f"{f'{rmse:.3f}±{e_sd:.3f}':>16}")


if __name__ == "__main__":
    main()
