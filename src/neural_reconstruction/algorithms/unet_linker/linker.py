"""
UNet 分割連接器模組 (UNet Segmentation Linker Module)

主控制器，協調完整的 UNet 神經纖維重建流程：
1. 預處理：ROI 提取、背景減除
2. UNet 推論：對 ROI 影像進行滑動窗口推論，取得預測遮罩
3. 骨架化與種子切分（TopologyBuilder）
4. 元件間連接路徑查找（PathFinder）
5. MST 骨架萃取
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import networkx as nx
import torch
from scipy.spatial import KDTree
from skimage.measure import label
import cv2

from neural_reconstruction.core.preprocessing import SkinAnalysisPipeline
from neural_reconstruction.core.segmentation import UNet, predict_full_image
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.common.data_types import LinkerResult

logger = logging.getLogger(__name__)


class UnetLinker:
    """
    UNet 神經重建連接器（含預處理）

    使用訓練好的 UNet 模型對 ROI 影像進行分割，
    再以 MST 演算法重建神經纖維網路。

    Examples:
        >>> linker = UnetLinker(checkpoint_path="output/unet_best.pth")
        >>> result = linker.run(image, mask, annotation)
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        # UNet 參數
        base_channels: int = 32,
        patch_size: int = 512,
        overlap: int = 64,
        threshold: float = 0.5,
        device: str = "auto",
    ):
        # UNet 參數
        self.checkpoint_path = Path(checkpoint_path)
        self.base_channels = base_channels
        self.patch_size = patch_size
        self.stride = patch_size - overlap
        self.threshold = threshold

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 延遲載入模型（首次 run 時載入）
        self._model: Optional[UNet] = None

    def _load_model(self) -> UNet:
        """載入 UNet 模型（僅在首次呼叫時執行）"""
        if self._model is not None:
            return self._model

        logger.info(f"  - 載入模型: {self.checkpoint_path}")
        model = UNet(
            in_channels=1, out_channels=2, base_channels=self.base_channels
        ).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        self._model = model
        return model

    def run(
        self, image: np.ndarray, mask: np.ndarray, annotation: np.ndarray
    ) -> LinkerResult:
        """
        運行完整的 UNet 神經纖維重建流程（含預處理）

        Args:
            image: 原始圖像 (H, W) 或 (H, W, 3)
            mask: 表皮遮罩 (H, W)

        Returns:
            LinkerResult
        """
        logger.info("1. 圖像預處理...")

        pipeline = SkinAnalysisPipeline()
        pipeline.config.mask.dilate_offset = 1  # 擴張遮罩以包含邊緣纖維
        orig_img = image[:, :, 1]  # 綠色通道
        roi_mask = pipeline._create_dilated_mask(mask)
        roi_image = cv2.bitwise_and(orig_img, orig_img, mask=roi_mask)

        logger.info("  ✓ 預處理完成")

        # 2. UNet 推論
        logger.info("2. UNet 推論...")
        model = self._load_model()

        prob_map = predict_full_image(
            model,
            orig_img,
            patch_size=self.patch_size,
            stride=self.stride,
            device=self.device,
        )

        # 二值化預測結果
        unet_pred = ((prob_map > self.threshold).astype(np.uint8)) * 255
        unet_pred = cv2.bitwise_and(unet_pred, unet_pred, mask=roi_mask)
        fg_ratio = unet_pred.mean() / 255
        logger.info(f"  ✓ UNet 推論完成 (fg_ratio={fg_ratio:.3%})")

        builder = TopologyBuilder()
        unet_annotation = builder.build_seed_graph(unet_pred)

        return LinkerResult(
            annotation=annotation,
            image=roi_image,
            mask=roi_mask,
            graph=unet_annotation,
        )
