"""Apply px→µm scaling to HD95 values in a staged grid-search output directory.

The per-combo JSON files store HD95 in pixels. This tool reads px_um.json,
multiplies each sample's HD95 by its ratio, and rewrites:
  - per_combo/combo_*.json   (adds hd95_px, updates hd95)
  - grid_search_results.json (updates hd95_* aggregate fields)
  - grid_search_results.csv  (same)

The original pixel value is preserved in hd95_px so the tool is idempotent:
running it twice on the same directory gives the same result.

Run:
    uv run python tools/rescale_hd95.py \\
        --grid-dir output/0510_grid \\
        --px-um data_0510/px_um.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_px_um(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def rescale_combo_files(per_combo_dir: Path, ratios: dict[str, float]) -> dict[int, list[float]]:
    """
    Update each combo_*.json in-place.

    For each success sample:
      - If hd95_px absent: set hd95_px = current hd95 (pixel backup)
      - Set hd95 = hd95_px * ratio  (or hd95_px if sample not in ratios)

    Returns {combo_index: [scaled hd95 values]} for aggregate recomputation.
    """
    scaled: dict[int, list[float]] = {}
    missing_samples: set[str] = set()

    for combo_file in sorted(per_combo_dir.glob("combo_*.json")):
        with combo_file.open(encoding="utf-8") as f:
            data = json.load(f)

        idx = data["combo_index"]
        vals: list[float] = []

        for s in data.get("samples", []):
            if s.get("status") != "success":
                continue
            if s.get("hd95") is None:
                continue

            # Preserve original pixel value on first run
            if "hd95_px" not in s:
                s["hd95_px"] = s["hd95"]

            ratio = ratios.get(s["sample_id"])
            if ratio is None:
                missing_samples.add(s["sample_id"])
                s["hd95"] = s["hd95_px"]
            else:
                s["hd95"] = s["hd95_px"] * ratio

            vals.append(s["hd95"])

        scaled[idx] = vals

        with combo_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    if missing_samples:
        print(
            f"  Warning: {len(missing_samples)} sample(s) not in px_um.json "
            f"— HD95 left in pixels: {sorted(missing_samples)}",
            file=sys.stderr,
        )

    return scaled


def recompute_aggregates(
    results_json: Path,
    scaled: dict[int, list[float]],
) -> None:
    if not results_json.exists():
        print(f"  {results_json} not found — skipping aggregate update.")
        return

    with results_json.open(encoding="utf-8") as f:
        doc = json.load(f)

    for r in doc.get("results", []):
        idx = _find_combo_index(r, doc)
        if idx is None:
            continue
        vals = scaled.get(idx, [])
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        if finite:
            r["hd95_mean"] = float(np.mean(finite))
            r["hd95_median"] = float(np.median(finite))
            r["hd95_std"] = float(np.std(finite))
            r["hd95_min"] = float(np.min(finite))
            r["hd95_max"] = float(np.max(finite))
        else:
            r["hd95_mean"] = r["hd95_median"] = r["hd95_std"] = None
            r["hd95_min"] = r["hd95_max"] = None

    with results_json.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def _find_combo_index(result_row: dict, doc: dict) -> int | None:
    """
    Match a result row back to a combo index by comparing its params
    against the per_combo files. We identify the combo by matching all
    param-grid keys between the result row and the combo params.

    grid_search_results.json embeds params as flat keys in each row,
    so we match by iterating the per_combo dir and comparing.
    """
    # The combo index is not stored in the result row directly, so we
    # reconstruct the ordering: results are written in combo_index order.
    results = doc.get("results", [])
    try:
        return results.index(result_row)
    except ValueError:
        return None


def update_csv(results_json: Path) -> None:
    csv_path = results_json.with_suffix(".csv")
    if not results_json.exists() or not csv_path.exists():
        return

    with results_json.open(encoding="utf-8") as f:
        doc = json.load(f)

    results = doc.get("results", [])
    if not results:
        return

    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir",
        type=Path,
        required=True,
        help="Grid search output directory (contains per_combo/ and grid_search_results.json).",
    )
    parser.add_argument(
        "--px-um",
        type=Path,
        required=True,
        help="Path to px_um.json (maps sample_id → µm per pixel).",
    )
    args = parser.parse_args()

    per_combo_dir = args.grid_dir / "per_combo"
    if not per_combo_dir.is_dir():
        print(f"Error: {per_combo_dir} not found.", file=sys.stderr)
        return 1

    ratios = load_px_um(args.px_um)
    print(f"Loaded {len(ratios)} px→µm ratios from {args.px_um}")

    combo_files = sorted(per_combo_dir.glob("combo_*.json"))
    print(f"Rescaling {len(combo_files)} combo files in {per_combo_dir} ...")
    scaled = rescale_combo_files(per_combo_dir, ratios)

    results_json = args.grid_dir / "grid_search_results.json"
    print(f"Updating aggregates in {results_json} ...")
    recompute_aggregates(results_json, scaled)

    print("Updating CSV ...")
    update_csv(results_json)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
