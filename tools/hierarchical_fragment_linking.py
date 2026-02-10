#!/usr/bin/env python3
"""
階層式片段連接算法（含完整預處理）

這個算法實現了完整的神經纖維重建流程：
1. 預處理：ROI提取、背景減除、伪標註生成
2. 階段1：高信心端點延伸（嚴格角度限制、小搜索半徑）
3. 階段2：生成MST候選邊（寬鬆角度、大搜索半徑）+ MST優化

用法:
    python tools/hierarchical_fragment_linking.py \
        --image data/S1585-2_a/image.png \
        --mask data/S1585-2_a/mask.png \
        --annotation data/S1585-2_a/annotation.png \
        --output output/hierarchical_linking/S1585-2_a_result.pkl
"""

import argparse
import sys
from pathlib import Path

import cv2

from neural_reconstruction.core.evaluation import TopologyLoader
from neural_reconstruction.algorithms.fragment_linking import HierarchicalFragmentLinker


# =============================================================================
# 命令行接口
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="階層式片段連接算法（含完整預處理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python tools/hierarchical_fragment_linking.py \\
      --image data/S1585-2_a/image.png \\
      --mask data/S1585-2_a/mask.png \\
      --annotation data/S1585-2_a/annotation.png \\
      --output output/hierarchical_linking/S1585-2_a_result.pkl

  # 詳細模式
  python tools/hierarchical_fragment_linking.py \\
      --image data/S1585-2_a/image.png \\
      --mask data/S1585-2_a/mask.png \\
      --annotation data/S1585-2_a/annotation.png \\
      --output output/hierarchical_linking/S1585-2_a_result.pkl \\
      --verbose

  # 自定義參數
  python tools/hierarchical_fragment_linking.py \\
      --image data/S1585-2_a/image.png \\
      --mask data/S1585-2_a/mask.png \\
      --annotation data/S1585-2_a/annotation.png \\
      --output output/hierarchical_linking/S1585-2_a_result.pkl \\
      --offset-px 150 \\
      --segment-length 5.0 \\
      --search-radius-phase1 15.0 \\
      --search-radius-phase2 25.0 \\
      --verbose
        """,
    )

    # 必需參數
    parser.add_argument(
        "--image", type=Path, required=True, help="輸入原始圖像路徑 (RGB或灰度)"
    )

    parser.add_argument("--mask", type=Path, required=True, help="輸入表皮遮罩圖像路徑")

    parser.add_argument(
        "--annotation", type=Path, required=True, help="輸入手工標註圖像路徑"
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="輸出拓撲文件路徑 (.pkl)"
    )

    # 預處理參數
    parser.add_argument(
        "--offset-px",
        type=int,
        default=100,
        help="ROI垂直膨脹偏移量 (default: 100)",
    )

    parser.add_argument(
        "--rolling-ball-radius",
        type=int,
        default=2,
        help="背景減除 rolling ball 半徑 (default: 2)",
    )

    parser.add_argument(
        "--sato-weight",
        type=float,
        default=0.0,
        help="Sato濾波器權重 (default: 0.0, 不使用)",
    )

    parser.add_argument(
        "--opening-kernel-size",
        type=int,
        default=3,
        help="形態學 opening 核大小 (default: 3)",
    )

    # 重建參數
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3.0,
        help="種子圖分段長度 (default: 3.0)",
    )

    parser.add_argument(
        "--search-radius-pathfinding",
        type=float,
        default=50.0,
        help="路徑查找搜索半徑 (default: 50.0)",
    )

    parser.add_argument(
        "--search-radius-phase1",
        type=float,
        default=10.0,
        help="階段1搜索半徑 (default: 10.0)",
    )

    parser.add_argument(
        "--max-angle-phase1",
        type=float,
        default=75.0,
        help="階段1最大角度 (default: 75.0)",
    )

    parser.add_argument(
        "--search-radius-phase2",
        type=float,
        default=20.0,
        help="階段2搜索半徑 (default: 20.0)",
    )

    parser.add_argument(
        "--max-angle-phase2",
        type=float,
        default=90.0,
        help="階段2最大角度 (default: 90.0)",
    )

    parser.add_argument(
        "--max-cost-threshold-phase2",
        type=float,
        default=0.75,
        help="階段2成本閾值 (default: 0.75)",
    )

    parser.add_argument(
        "--phase1-weight-discount",
        type=float,
        default=0.5,
        help="階段1邊權重折扣 (default: 0.5)",
    )

    parser.add_argument("--verbose", action="store_true", help="詳細輸出")

    args = parser.parse_args()

    # 檢查輸入文件
    if not args.image.exists():
        print(f"錯誤: 圖像文件不存在: {args.image}")
        return 1

    if not args.mask.exists():
        print(f"錯誤: 遮罩文件不存在: {args.mask}")
        return 1

    if not args.annotation.exists():
        print(f"錯誤: 標註文件不存在: {args.annotation}")
        return 1

    # 創建輸出目錄
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 載入圖像
    if args.verbose:
        print(f"載入圖像: {args.image}")
        print(f"載入遮罩: {args.mask}")
        print(f"載入標註: {args.annotation}")

    # 載入原始圖像 (支持 RGB 和灰度)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        # 嘗試灰度模式
        image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    annotation = cv2.imread(str(args.annotation), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"錯誤: 無法載入圖像: {args.image}")
        return 1

    if mask is None:
        print(f"錯誤: 無法載入遮罩: {args.mask}")
        return 1

    if annotation is None:
        print(f"錯誤: 無法載入標註: {args.annotation}")
        return 1

    # 創建連接器
    linker = HierarchicalFragmentLinker(
        # 預處理參數
        offset_px=args.offset_px,
        rolling_ball_radius=args.rolling_ball_radius,
        sato_weight=args.sato_weight,
        opening_kernel_size=args.opening_kernel_size,
        # 重建參數
        segment_length=args.segment_length,
        search_radius_pathfinding=args.search_radius_pathfinding,
        search_radius_phase1=args.search_radius_phase1,
        max_angle_phase1=args.max_angle_phase1,
        search_radius_phase2=args.search_radius_phase2,
        max_angle_phase2=args.max_angle_phase2,
        max_cost_threshold_phase2=args.max_cost_threshold_phase2,
        phase1_weight_discount=args.phase1_weight_discount,
        verbose=args.verbose,
    )

    # 運行算法
    mst_result = linker.run(image, mask, annotation)

    # 保存結果
    if args.verbose:
        print(f"\n保存結果到: {args.output}")

    loader = TopologyLoader()
    loader.save(mst_result, args.output, format="pickle")

    print(f"\n✓ 完成！結果已保存到: {args.output}")
    print(f"  - 節點數: {mst_result.number_of_nodes()}")
    print(f"  - 邊數: {mst_result.number_of_edges()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
