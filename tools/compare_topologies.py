"""
拓樸比對工具 (Topology Comparison Tool)

專注於比對兩個拓樸圖，計算平均 Hausdorff 距離。
不依賴於影像處理 Pipeline，直接從拓樸檔案讀取。

支援的拓樸檔案格式：
- GraphML (.graphml) - 推薦，保留所有屬性
- Pickle (.pkl, .pickle) - NetworkX 原生格式
- GML (.gml) - 簡單文字格式
- JSON (.json) - 自訂 JSON 格式

使用範例：
    # 比對兩個拓樸檔案
    python tools/compare_topologies.py \
        --topology1 output/pred_topology.graphml \
        --topology2 output/gt_topology.graphml

    # 批次比對目錄中的所有拓樸對
    python tools/compare_topologies.py \
        --batch \
        --pred-dir output/predictions/ \
        --gt-dir output/ground_truth/ \
        --output results.csv

作者: Claude Code
日期: 2026-02-09
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import csv

import numpy as np

# 使用新的評估模組
from neural_reconstruction.core.evaluation import (
    TopologyLoader,
    TopologyComparator,
)


# ============================================================================
# 主程式
# ============================================================================


def setup_logging(verbose: bool = False):
    """設定日誌"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def compare_single_pair(args):
    """比對單一對拓樸"""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("拓樸比對工具")
    logger.info("=" * 80)

    # 載入拓樸
    loader = TopologyLoader()

    logger.info(f"載入拓樸 1: {args.topology1}")
    graph1 = loader.load(Path(args.topology1))
    if graph1 is None:
        logger.error("載入拓樸 1 失敗")
        return 1

    logger.info(f"載入拓樸 2: {args.topology2}")
    graph2 = loader.load(Path(args.topology2))
    if graph2 is None:
        logger.error("載入拓樸 2 失敗")
        return 1

    # 比對
    comparator = TopologyComparator()
    result = comparator.compare(
        graph1,
        graph2,
        label1=Path(args.topology1).stem,
        label2=Path(args.topology2).stem,
    )

    # 轉換為字典以便輸出
    result_dict = result.to_dict()

    # 輸出結果
    print("\n" + "=" * 80)
    print("比對結果")
    print("=" * 80)
    print(f"拓樸 1: {result_dict['label1']}")
    print(f"  節點數: {result_dict['num_nodes1']}")
    print(f"  邊數: {result_dict['num_edges1']}")
    print(f"  總點數: {result_dict['num_points1']}")
    print()
    print(f"拓樸 2: {result_dict['label2']}")
    print(f"  節點數: {result_dict['num_nodes2']}")
    print(f"  邊數: {result_dict['num_edges2']}")
    print(f"  總點數: {result_dict['num_points2']}")
    print()

    if result_dict["status"] == "success":
        print(f"平均 Hausdorff 距離: {result_dict['hausdorff_distance']:.4f} 像素")
    else:
        print(f"狀態: {result_dict['status']}")
        print(f"錯誤: {result_dict['error']}")

    print("=" * 80)

    # 儲存結果（如果指定）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"結果已儲存: {output_path}")

    return 0 if result_dict["status"] == "success" else 1


def compare_batch(args):
    """批次比對拓樸"""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("批次拓樸比對")
    logger.info("=" * 80)

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.is_dir():
        logger.error(f"預測目錄不存在: {pred_dir}")
        return 1

    if not gt_dir.is_dir():
        logger.error(f"GT 目錄不存在: {gt_dir}")
        return 1

    # 尋找配對的拓樸檔案
    loader = TopologyLoader()
    comparator = TopologyComparator()
    results = []

    # 支援的副檔名
    extensions = [".graphml", ".pkl", ".pickle", ".gml", ".json"]

    pred_files = []
    for ext in extensions:
        pred_files.extend(pred_dir.glob(f"*{ext}"))

    logger.info(f"找到 {len(pred_files)} 個預測拓樸檔案")

    for pred_file in sorted(pred_files):
        # 尋找對應的 GT 檔案
        sample_id = pred_file.stem
        gt_file = None

        for ext in extensions:
            candidate = gt_dir / f"{sample_id}{ext}"
            if candidate.exists():
                gt_file = candidate
                break

        if gt_file is None:
            logger.warning(f"找不到對應的 GT 檔案: {sample_id}")
            continue

        logger.info(f"比對: {pred_file.name} vs {gt_file.name}")

        # 載入
        graph_pred = loader.load(pred_file)
        graph_gt = loader.load(gt_file)

        if graph_pred is None or graph_gt is None:
            logger.warning(f"載入失敗，跳過: {sample_id}")
            continue

        # 比對
        result = comparator.compare(
            graph_pred, graph_gt, label1=f"{sample_id}_pred", label2=f"{sample_id}_gt"
        )
        # 轉換為字典並添加 sample_id
        result_dict = result.to_dict()
        result_dict["sample_id"] = sample_id
        results.append(result_dict)

    # 輸出統計
    print("\n" + "=" * 80)
    print("批次比對統計")
    print("=" * 80)
    print(f"總共比對: {len(results)} 對")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"成功: {len(successful)}")
    print(f"失敗: {len(failed)}")

    if successful:
        distances = [r["hausdorff_distance"] for r in successful]
        print()
        print("平均 Hausdorff 距離統計:")
        print(f"  平均值: {np.mean(distances):.4f}")
        print(f"  中位數: {np.median(distances):.4f}")
        print(f"  標準差: {np.std(distances):.4f}")
        print(f"  最小值: {np.min(distances):.4f}")
        print(f"  最大值: {np.max(distances):.4f}")

    print("=" * 80)

    # 儲存結果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV 格式
        if output_path.suffix == ".csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "sample_id",
                    "hausdorff_distance",
                    "num_nodes1",
                    "num_nodes2",
                    "num_edges1",
                    "num_edges2",
                    "num_points1",
                    "num_points2",
                    "status",
                    "error",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        else:
            # JSON 格式
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"結果已儲存: {output_path}")

    return 0


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="拓樸比對工具 - 計算兩個拓樸圖之間的平均 Hausdorff 距離",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 比對兩個拓樸檔案
  %(prog)s --topology1 pred.graphml --topology2 gt.graphml

  # 批次比對
  %(prog)s --batch --pred-dir predictions/ --gt-dir ground_truth/ --output results.csv

  # 詳細輸出
  %(prog)s --topology1 pred.pkl --topology2 gt.pkl --verbose
        """,
    )

    # 模式選擇
    parser.add_argument(
        "--batch", action="store_true", help="批次模式：比對兩個目錄中的所有配對拓樸"
    )

    # 單一比對參數
    parser.add_argument("--topology1", type=str, help="第一個拓樸檔案路徑")

    parser.add_argument("--topology2", type=str, help="第二個拓樸檔案路徑")

    # 批次比對參數
    parser.add_argument("--pred-dir", type=str, help="預測拓樸目錄（批次模式）")

    parser.add_argument("--gt-dir", type=str, help="Ground truth 拓樸目錄（批次模式）")

    # 共用參數
    parser.add_argument("--output", type=str, help="輸出檔案路徑（JSON 或 CSV 格式）")

    parser.add_argument("--verbose", action="store_true", help="啟用詳細日誌輸出")

    args = parser.parse_args()

    # 驗證參數
    if args.batch:
        if not args.pred_dir or not args.gt_dir:
            parser.error("批次模式需要 --pred-dir 和 --gt-dir 參數")
        return compare_batch(args)
    else:
        if not args.topology1 or not args.topology2:
            parser.error("單一比對模式需要 --topology1 和 --topology2 參數")
        return compare_single_pair(args)


if __name__ == "__main__":
    sys.exit(main())
