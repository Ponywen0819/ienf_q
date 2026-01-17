"""
主要 Pipeline：從前處理到神經網路重建
(Main Pipeline: From Preprocessing to Neural Network Reconstruction)

整合完整流程：
1. 前處理 (Preprocessing) - 形態學處理、背景校正、ROI 萃取
2. 神經重建 (Neural Reconstruction) - 元件分析、連接圖建構、骨架萃取

使用範例:
    from neural_reconstruction.ui.main_pipeline import NeuralReconstructionPipeline

    # 建立 pipeline
    pipeline = NeuralReconstructionPipeline()

    # 執行完整流程
    result = pipeline.run(
        label_image=label_img,
        mask_image=mask_img,
        original_image=orig_img
    )

    # 取得結果
    mst_forest = result.mst_forest
    final_label = result.final_label
    roi_image = result.roi_image

作者: Claude Code
日期: 2026-01-12
"""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import networkx as nx
from PIL import Image

from neural_reconstruction.core.preprocessing.pipeline import (
    SkinAnalysisPipeline,
    PipelineConfig as PreprocessingPipelineConfig,
)

from neural_reconstruction.core.construction.main import build_neural_network

# 設定 logger
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Pipeline 完整執行結果

    Attributes:
        mst_forest: 重建的神經網路骨架（NetworkX Graph）
        final_label: 前處理後的最終標註影像
        roi_image: ROI 區域影像
        preprocessing_debug: 前處理的 debug 輸出（如果啟用）
        num_nodes: MST 節點數
        num_edges: MST 邊數
        num_components: MST 連通分量數
    """

    mst_forest: nx.Graph
    final_label: np.ndarray
    roi_image: np.ndarray
    preprocessing_debug: Optional[Any] = None
    num_nodes: int = 0
    num_edges: int = 0
    num_components: int = 0


class NeuralReconstructionPipeline:
    """
    神經重建主要 Pipeline

    整合前處理和重建流程，提供簡潔的 API 介面。

    Example:
        >>> pipeline = NeuralReconstructionPipeline()
        >>> result = pipeline.run(label_img, mask_img, orig_img)
        >>> print(f"重建完成: {result.num_nodes} 節點, {result.num_edges} 條邊")
    """

    def __init__(
        self,
        preprocessing_config: Optional[PreprocessingPipelineConfig] = None,
        reconstruction_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 Pipeline

        Args:
            preprocessing_config: 前處理配置字典，若為 None 則使用預設值
            reconstruction_config: 重建配置字典，若為 None 則使用預設值
        """
        # 設定前處理配置
        if preprocessing_config is None:
            preprocessing_config = self._get_default_preprocessing_config()
        self.preprocessing_config = preprocessing_config

        # 設定重建配置
        if reconstruction_config is None:
            reconstruction_config = self._get_default_reconstruction_config()
        self.reconstruction_config = reconstruction_config

        logger.info("Pipeline 初始化完成")
        logger.info(f"前處理配置: {preprocessing_config}")
        logger.info(f"重建配置: {reconstruction_config}")

    @staticmethod
    def _get_default_preprocessing_config() -> PreprocessingPipelineConfig:
        """取得預設前處理配置"""
        config = PreprocessingPipelineConfig.from_dict(
            {
                "morphology": {"closing_kernel": 0, "opening_kernel": 3},
                "mask": {"dilate_offset": 50},
                "background": {
                    "method": "rolling_ball",
                    "radius": 25,
                },
                "threshold": {"use_full_roi": False},
                "normalization": {"enabled": False},
            }
        )
        return config

    @staticmethod
    def _get_default_reconstruction_config() -> Dict[str, Any]:
        """取得預設重建配置"""
        return {
            "connectivity": 4,
            "min_area": 50,
            "segment_length": 5.0,
            "min_edge_length": None,
            "prune_threshold": 5.0,
            "spacing": 1.0,
            "search_radius": 50.0,
            "max_cost_threshold": 0.98,
            "intensity_weight": 0.6,
            "shape_weight": 0.4,
        }

    def run(
        self,
        label_image: np.ndarray,
        mask_image: np.ndarray,
        original_image: np.ndarray,
        debug: bool = False,
    ) -> PipelineResult:
        """
        執行完整的神經重建 Pipeline

        流程：
        1. 前處理 - 形態學處理、背景校正、ROI 萃取
        2. 神經重建 - 元件分析、連接圖建構、骨架萃取

        Args:
            label_image: 二值化標註影像 (H, W)，uint8
            mask_image: 表皮 mask 影像 (H, W)，uint8
            original_image: 原始影像，可以是灰階 (H, W) 或 RGB (H, W, 3)
            debug: 是否啟用 debug 模式，記錄中間結果

        Returns:
            PipelineResult: 包含 MST 森林、最終標註、ROI 影像等結果

        Raises:
            ValueError: 輸入影像尺寸不符或格式錯誤

        Example:
            >>> pipeline = NeuralReconstructionPipeline()
            >>> result = pipeline.run(label_img, mask_img, orig_img)
            >>> print(f"節點數: {result.num_nodes}")
        """
        logger.info("\n" + "=" * 80)
        logger.info("開始執行神經重建 Pipeline")
        logger.info("=" * 80)

        # 驗證輸入
        self._validate_inputs(label_image, mask_image, original_image)

        # 萃取綠色通道（前處理需要綠色通道）
        green_channel_input = self._extract_green_channel(original_image)

        # ========== 階段 1: 影像預處理 ==========
        logger.info("\n【階段 1/2】影像預處理")
        logger.info("-" * 80)

        final_label, roi_image, preprocessing_debug = self._run_preprocessing(
            label_image, mask_image, green_channel_input
        )

        logger.info("✓ 前處理完成")
        logger.info(f"  - 最終標註形狀: {final_label.shape}")
        logger.info(f"  - ROI 影像形狀: {roi_image.shape}")

        # 萃取綠色通道（用於神經重建）
        green_channel = self._extract_green_channel(roi_image)
        logger.info(f"  - 綠色通道形狀: {green_channel.shape}")

        # ========== 階段 2: 神經網路重建 ==========
        logger.info("\n【階段 2/2】神經網路重建")
        logger.info("-" * 80)

        mst_forest = self._run_reconstruction(final_label, green_channel)

        # 統計結果
        num_nodes = mst_forest.number_of_nodes()
        num_edges = mst_forest.number_of_edges()
        num_components = nx.number_connected_components(mst_forest)

        logger.info("✓ 重建完成")
        logger.info(f"  - 節點數: {num_nodes}")
        logger.info(f"  - 邊數: {num_edges}")
        logger.info(f"  - 連通分量: {num_components}")

        # ========== 完成 ==========
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline 執行完成")
        logger.info("=" * 80)

        # 建立結果物件
        result = PipelineResult(
            mst_forest=mst_forest,
            final_label=final_label,
            roi_image=roi_image,
            preprocessing_debug=preprocessing_debug,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_components=num_components,
        )

        return result

    def run_from_files(
        self,
        label_path: str,
        mask_path: str,
        image_path: str,
        debug: bool = False,
    ) -> PipelineResult:
        """
        從檔案路徑執行 Pipeline

        Args:
            label_path: 標註影像路徑
            mask_path: Mask 影像路徑
            image_path: 原始影像路徑
            debug: 是否啟用 debug 模式

        Returns:
            PipelineResult: Pipeline 執行結果

        Example:
            >>> pipeline = NeuralReconstructionPipeline()
            >>> result = pipeline.run_from_files(
            ...     "data/Label/S163-2_a.tif",
            ...     "data/Mask/S163-2_a.tif",
            ...     "data/Original/S163-2_a.tif"
            ... )
        """
        logger.info("從檔案載入影像:")
        logger.info(f"  - 標註: {label_path}")
        logger.info(f"  - Mask: {mask_path}")
        logger.info(f"  - 原始: {image_path}")

        # 載入影像
        label_image = np.array(Image.open(label_path))
        mask_image = np.array(Image.open(mask_path))
        original_image = np.array(Image.open(image_path))

        logger.info("✓ 影像載入完成")

        # 執行 pipeline
        return self.run(label_image, mask_image, original_image, debug)

    def _validate_inputs(
        self,
        label_image: np.ndarray,
        mask_image: np.ndarray,
        original_image: np.ndarray,
    ) -> None:
        """驗證輸入影像"""
        if label_image.shape != mask_image.shape:
            raise ValueError(
                f"標註影像 {label_image.shape} 與 mask {mask_image.shape} 尺寸不符"
            )

        if original_image.shape[:2] != mask_image.shape:
            raise ValueError(
                f"原始影像 {original_image.shape} 與 mask {mask_image.shape} 尺寸不符"
            )

    def _run_preprocessing(
        self,
        label_image: np.ndarray,
        mask_image: np.ndarray,
        original_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
        """執行前處理 Pipeline"""
        # 建立前處理 pipeline
        preprocessing_pipeline = SkinAnalysisPipeline(self.preprocessing_config)

        # 執行前處理
        final_label, roi_image = preprocessing_pipeline.run(
            label_image, mask_image, original_image
        )

        preprocessing_debug = None
        # if debug:
        #     # 如果需要 debug 資訊，可以在此處理
        #     preprocessing_debug = {"config": self.preprocessing_config}

        return final_label, roi_image, preprocessing_debug

    def _extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """
        萃取綠色通道

        根據 CLAUDE.md 說明，神經纖維在綠色通道有最強的訊號。

        Args:
            image: 輸入影像，可以是灰階 (H, W) 或 RGB (H, W, 3)

        Returns:
            綠色通道影像 (H, W)，uint8
        """
        if image.ndim == 2:
            # 灰階影像，直接使用
            return image.astype(np.uint8)
        elif image.ndim == 3 and image.shape[2] >= 3:
            # RGB 影像，萃取綠色通道（索引 1）
            logger.info(f"從 RGB 影像 {image.shape} 萃取綠色通道")
            return image[:, :, 1].astype(np.uint8)
        else:
            raise ValueError(f"不支援的影像格式: {image.shape}")

    def _run_reconstruction(
        self, label_image: np.ndarray, green_channel: np.ndarray
    ) -> nx.Graph:
        """執行神經重建"""
        mst_forest = build_neural_network(
            label_image=label_image,
            green_channel=green_channel,
            **self.reconstruction_config,
        )

        return mst_forest


__all__ = ["NeuralReconstructionPipeline", "PipelineResult"]
