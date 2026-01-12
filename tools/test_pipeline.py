#!/usr/bin/env python3
"""
測試主要 Pipeline

驗證從前處理到神經重建的完整流程。
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


def main():
    """主測試函式"""
    # 設定檔案路徑
    data_dir = Path("data")
    label_path = data_dir / "Label" / "S163-2_a.tif"
    mask_path = data_dir / "Mask" / "S163-2_a.tif"
    image_path = data_dir / "Original" / "S163-2_a.tif"

    # 檢查檔案是否存在
    for path in [label_path, mask_path, image_path]:
        if not path.exists():
            logger.error(f"檔案不存在: {path}")
            return

    logger.info("=" * 80)
    logger.info("測試主要 Pipeline")
    logger.info("=" * 80)

    # 建立 pipeline（使用預設配置）
    pipeline = NeuralReconstructionPipeline()

    # 執行 pipeline
    result = pipeline.run_from_files(
        label_path=str(label_path),
        mask_path=str(mask_path),
        image_path=str(image_path),
        debug=False
    )

    # 顯示結果
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline 執行結果摘要")
    logger.info("=" * 80)
    logger.info(f"MST 節點數: {result.num_nodes}")
    logger.info(f"MST 邊數: {result.num_edges}")
    logger.info(f"連通分量數: {result.num_components}")
    logger.info(f"最終標註形狀: {result.final_label.shape}")
    logger.info(f"ROI 影像形狀: {result.roi_image.shape}")
    logger.info("=" * 80)

    # 驗證結果
    if result.num_nodes > 0 and result.num_edges > 0:
        logger.info("✅ Pipeline 測試成功！")
    else:
        logger.warning("⚠️  Pipeline 執行完成但未產生有效結果")


if __name__ == "__main__":
    main()
