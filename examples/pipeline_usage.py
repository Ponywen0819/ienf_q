#!/usr/bin/env python3
"""
主要 Pipeline 使用範例

展示如何使用 NeuralReconstructionPipeline 執行完整的神經重建流程。
"""

import logging
from pathlib import Path

from neural_reconstruction.ui.main_pipeline import NeuralReconstructionPipeline

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_default_config():
    """範例 1: 使用預設配置"""
    logger.info("\n" + "=" * 80)
    logger.info("範例 1: 使用預設配置")
    logger.info("=" * 80)

    # 建立 pipeline（使用預設配置）
    pipeline = NeuralReconstructionPipeline()

    # 執行 pipeline
    result = pipeline.run_from_files(
        label_path="data/Label/S163-2_a.tif",
        mask_path="data/Mask/S163-2_a.tif",
        image_path="data/Original/S163-2_a.tif"
    )

    # 取得結果
    logger.info(f"\n結果:")
    logger.info(f"  節點數: {result.num_nodes}")
    logger.info(f"  邊數: {result.num_edges}")
    logger.info(f"  連通分量數: {result.num_components}")


def example_2_custom_config():
    """範例 2: 自訂配置"""
    logger.info("\n" + "=" * 80)
    logger.info("範例 2: 自訂配置")
    logger.info("=" * 80)

    # 自訂前處理配置
    preprocessing_config = {
        'morphology': {
            'closing_kernel': 5,  # 增大 closing kernel
            'opening_kernel': 3
        },
        'mask': {
            'dilate_offset': 100  # 增大 dilation offset
        },
        'background': {
            'method': 'rolling_ball',  # 使用 rolling ball 方法
            'radius': 20,
            'light_background': True
        },
        'threshold': {
            'method': 'binary'
        },
        'normalization': {
            'enabled': True  # 啟用區域正規化
        }
    }

    # 自訂重建配置
    reconstruction_config = {
        'connectivity': 4,
        'min_area': 30,  # 降低最小面積閾值
        'segment_length': 3.0,  # 減小種子間隔
        'search_radius': 100.0,  # 增大搜尋半徑
        'max_cost_threshold': 0.95,  # 降低成本閾值
        'intensity_weight': 0.7,  # 增加強度權重
        'shape_weight': 0.3
    }

    # 建立 pipeline
    pipeline = NeuralReconstructionPipeline(
        preprocessing_config=preprocessing_config,
        reconstruction_config=reconstruction_config
    )

    # 執行 pipeline
    result = pipeline.run_from_files(
        label_path="data/Label/S163-2_a.tif",
        mask_path="data/Mask/S163-2_a.tif",
        image_path="data/Original/S163-2_a.tif"
    )

    # 取得結果
    logger.info(f"\n結果:")
    logger.info(f"  節點數: {result.num_nodes}")
    logger.info(f"  邊數: {result.num_edges}")
    logger.info(f"  連通分量數: {result.num_components}")


def example_3_direct_array_input():
    """範例 3: 直接使用 NumPy 陣列輸入"""
    logger.info("\n" + "=" * 80)
    logger.info("範例 3: 直接使用 NumPy 陣列輸入")
    logger.info("=" * 80)

    import numpy as np
    from PIL import Image

    # 載入影像
    label_image = np.array(Image.open("data/Label/S163-2_a.tif"))
    mask_image = np.array(Image.open("data/Mask/S163-2_a.tif"))
    original_image = np.array(Image.open("data/Original/S163-2_a.tif"))

    # 建立 pipeline
    pipeline = NeuralReconstructionPipeline()

    # 執行 pipeline
    result = pipeline.run(
        label_image=label_image,
        mask_image=mask_image,
        original_image=original_image,
        debug=False
    )

    # 取得結果
    logger.info(f"\n結果:")
    logger.info(f"  MST 森林: {result.mst_forest}")
    logger.info(f"  節點數: {result.num_nodes}")
    logger.info(f"  邊數: {result.num_edges}")

    # 存取圖資料
    logger.info(f"\n圖資料:")
    logger.info(f"  節點: {list(result.mst_forest.nodes())[:5]}...")  # 顯示前 5 個節點
    if result.mst_forest.number_of_edges() > 0:
        logger.info(f"  邊: {list(result.mst_forest.edges())[:5]}...")  # 顯示前 5 條邊


def example_4_save_results():
    """範例 4: 儲存結果"""
    logger.info("\n" + "=" * 80)
    logger.info("範例 4: 儲存結果")
    logger.info("=" * 80)

    import numpy as np
    from PIL import Image
    import networkx as nx

    # 建立 pipeline
    pipeline = NeuralReconstructionPipeline()

    # 執行 pipeline
    result = pipeline.run_from_files(
        label_path="data/Label/S163-2_a.tif",
        mask_path="data/Mask/S163-2_a.tif",
        image_path="data/Original/S163-2_a.tif"
    )

    # 建立輸出目錄
    output_dir = Path("output/pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 儲存最終標註影像
    final_label_path = output_dir / "final_label.png"
    Image.fromarray(result.final_label).save(final_label_path)
    logger.info(f"✓ 已儲存最終標註: {final_label_path}")

    # 儲存 ROI 影像
    roi_path = output_dir / "roi_image.png"
    Image.fromarray(result.roi_image).save(roi_path)
    logger.info(f"✓ 已儲存 ROI 影像: {roi_path}")

    # 儲存 MST 森林（GraphML 格式）
    graph_path = output_dir / "mst_forest.graphml"
    nx.write_graphml(result.mst_forest, graph_path)
    logger.info(f"✓ 已儲存 MST 森林: {graph_path}")

    # 儲存結果摘要
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("神經重建結果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"節點數: {result.num_nodes}\n")
        f.write(f"邊數: {result.num_edges}\n")
        f.write(f"連通分量數: {result.num_components}\n")
        f.write(f"最終標註形狀: {result.final_label.shape}\n")
        f.write(f"ROI 影像形狀: {result.roi_image.shape}\n")
    logger.info(f"✓ 已儲存結果摘要: {summary_path}")


if __name__ == "__main__":
    # 執行所有範例（依序執行，避免混淆）

    # 範例 1: 使用預設配置
    # example_1_default_config()

    # 範例 2: 自訂配置
    # example_2_custom_config()

    # 範例 3: 直接使用 NumPy 陣列輸入
    # example_3_direct_array_input()

    # 範例 4: 儲存結果
    example_4_save_results()
