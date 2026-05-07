"""
從 evaluation results.json 抽取每個樣本的 valid_count_pred，
輸出成 {sample_id: count} 的字典。

使用範例:
    python tools/extract_valid_counts.py output/label_test/results.json
    python tools/extract_valid_counts.py output/label_test/results.json -o counts.json
"""

import argparse
import json
from pathlib import Path


def extract_valid_counts(results_path: Path) -> dict:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", {})
    return {sid: info["valid_count_pred"] for sid, info in samples.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Extract valid_count_pred from evaluation results.json"
    )
    parser.add_argument("results", type=Path, help="Path to results.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    counts = extract_valid_counts(args.results)
    payload = json.dumps(counts, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {len(counts)} entries to {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
