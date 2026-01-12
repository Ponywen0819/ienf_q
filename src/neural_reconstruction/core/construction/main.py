#!/usr/bin/env python3
"""
神經重建主要進入點 (Neural Reconstruction Main Entry Point)

提供完整的神經網路重建流程，串接以下模組：
1. 連通元件分析 (Connected Components Analysis)
2. 元件拓樸分析 (Component Topology Analysis)
3. 元件配對與連接圖建構 (Component Pairing & Connection Graph Building)
4. 主要骨架萃取 (Backbone Extraction via MST)

使用範例:
    from neural_reconstruction.core.construction.main import build_neural_network

    mst_forest = build_neural_network(
        label_image=binary_label,
        green_channel=green_channel_image,
        connectivity=4,
        min_area=50,
        segment_length=5.0,
        search_radius=50.0,
        max_cost_threshold=0.98,
    )

    # mst_forest 是 NetworkX Graph，包含重建的神經網路骨架

作者: Claude Code
日期: 2026-01-12
"""

import logging
from typing import Optional, List

import numpy as np
import networkx as nx
from skimage.measure import label, regionprops

from .component_analyzer.analyzer import ComponentAnalyzer
from .connection_graph_builder.builder import NetworkBuilder
from .backbone_extractor.extractor import BackboneExtractor
from neural_reconstruction.common.data_types import ComponentAnalysisResult

# 設定 logger
logger = logging.getLogger(__name__)


def build_neural_network(
    label_image: np.ndarray,
    green_channel: np.ndarray,
    connectivity: int = 4,
    min_area: int = 0,
    segment_length: float = 5.0,
    min_edge_length: Optional[float] = None,
    prune_threshold: float = 5.0,
    spacing: float = 1.0,
    search_radius: float = 50.0,
    max_cost_threshold: float = 0.98,
    intensity_weight: float = 0.6,
    shape_weight: float = 0.4,
) -> nx.Graph:
    """
    建構神經網路骨架

    完整流程：
    1. 連通元件分析 - 識別離散的纖維段落
    2. 元件分析 - 骨架化、拓樸建構、種子萃取
    3. 連接圖建構 - 使用 A* 計算元件間連接路徑與成本
    4. 骨架萃取 - 使用 MST 演算法萃取主要神經組織

    Args:
        label_image: 二值化標註影像 (0/255 或 0/1)
        green_channel: 綠色通道影像 (uint8, 0-255)，用於路徑搜尋
        connectivity: 連通性 (4 或 8)，預設 4
        min_area: 最小元件面積過濾（像素數），預設 0
        segment_length: 種子間隔長度（像素），預設 5.0
        min_edge_length: 最小邊長度閾值（像素），預設使用 segment_length
        prune_threshold: 剪枝閾值（像素），預設 5.0
        spacing: 像素間距（用於距離計算），預設 1.0
        search_radius: 搜尋半徑（像素），預設 50.0
        max_cost_threshold: 最大成本閾值比例 (0-1)，預設 0.98
        intensity_weight: 路徑搜尋的強度權重，預設 0.6
        shape_weight: 路徑搜尋的形狀權重，預設 0.4

    Returns:
        nx.Graph: MST 森林（主要神經網路骨架）
            - 節點：種子點座標 (y, x)
            - 邊屬性：weight（成本）、distance（距離）、path（路徑座標列表）
    """
    logger.info("=" * 70)
    logger.info("開始神經網路重建流程")
    logger.info("=" * 70)

    # 參數處理
    if min_edge_length is None:
        min_edge_length = segment_length

    # 邊界條件檢查
    if label_image is None or label_image.size == 0:
        logger.warning("輸入影像為空，返回空圖")
        return nx.Graph()

    if green_channel is None or green_channel.size == 0:
        logger.warning("綠色通道影像為空，返回空圖")
        return nx.Graph()

    # ========== 階段 1: 連通元件分析 ==========
    logger.info("\n" + "=" * 70)
    logger.info("階段 1: 連通元件分析")
    logger.info("=" * 70)

    # 二值化處理（確保是 0/1）
    binary_image = (label_image > 0).astype(np.uint8)

    # 轉換 connectivity 參數（4 -> 1, 8 -> 2）
    # scikit-image 使用 1 表示 4-連通，2 表示 8-連通
    skimage_connectivity = 1 if connectivity == 4 else 2

    # 連通元件標記
    labeled_image = label(binary_image, connectivity=skimage_connectivity)
    regions = regionprops(labeled_image)

    logger.info(f"連通性: {connectivity}-connected (scikit-image: {skimage_connectivity})")
    logger.info(f"偵測到 {len(regions)} 個連通元件")

    # 面積過濾
    if min_area > 0:
        original_count = len(regions)
        regions = [r for r in regions if r.area >= min_area]
        filtered_count = original_count - len(regions)
        logger.info(f"最小面積閾值: {min_area} 像素")
        logger.info(f"過濾掉 {filtered_count} 個小元件，剩餘 {len(regions)} 個")

    if len(regions) == 0:
        logger.warning("沒有連通元件，返回空圖")
        return nx.Graph()

    logger.info(f"✓ 階段 1 完成: {len(regions)} 個連通元件")

    # ========== 階段 2: 元件分析（骨架化、拓樸建構、種子萃取） ==========
    logger.info("\n" + "=" * 70)
    logger.info("階段 2: 元件分析（骨架化、拓樸建構、種子萃取）")
    logger.info("=" * 70)

    # 初始化元件分析器
    component_analyzer = ComponentAnalyzer(
        segment_length=segment_length,
        min_edge_length=min_edge_length,
        prune_threshold=prune_threshold,
        spacing=spacing,
    )

    logger.info(f"種子間隔: {segment_length} 像素")
    logger.info(f"最小邊長度: {min_edge_length} 像素")
    logger.info(f"剪枝閾值: {prune_threshold} 像素")

    # 批次分析所有元件
    component_results: List[ComponentAnalysisResult] = []
    total_seeds = 0

    for i, region in enumerate(regions):
        result = component_analyzer.analyze(region)
        component_results.append(result)

        # 統計種子數（從拓樸圖的節點數）
        num_seeds = result.topology.number_of_nodes()
        total_seeds += num_seeds

        # 進度報告（每 10 個元件或最後一個）
        if (i + 1) % 10 == 0 or (i + 1) == len(regions):
            logger.info(f"  已處理 {i + 1}/{len(regions)} 個元件")

    logger.info(f"✓ 階段 2 完成: 萃取 {total_seeds} 個種子點")

    # ========== 階段 3: 連接圖建構 ==========
    logger.info("\n" + "=" * 70)
    logger.info("階段 3: 連接圖建構")
    logger.info("=" * 70)

    # 初始化網路建構器
    network_builder = NetworkBuilder(
        image=green_channel,
        search_radius=search_radius,
        max_cost_threshold=max_cost_threshold,
        intensity_weight=intensity_weight,
        shape_weight=shape_weight,
    )

    # 建構連接圖
    connection_result = network_builder.build_graph(component_results)
    connection_graph = connection_result.graph

    if connection_graph.number_of_nodes() == 0:
        logger.warning("連接圖為空，返回空圖")
        return nx.Graph()

    logger.info(f"✓ 階段 3 完成: {connection_graph.number_of_edges()} 條連接")

    # ========== 階段 4: 主要骨架萃取（MST） ==========
    logger.info("\n" + "=" * 70)
    logger.info("階段 4: 主要骨架萃取（MST）")
    logger.info("=" * 70)

    # 初始化骨架萃取器
    backbone_extractor = BackboneExtractor()

    # 萃取 MST 森林
    mst_forest = backbone_extractor.extract(connection_graph)

    if mst_forest.number_of_nodes() == 0:
        logger.warning("MST 森林為空")
        return nx.Graph()

    # 統計連通分量
    num_components = nx.number_connected_components(mst_forest)

    logger.info("✓ 階段 4 完成:")
    logger.info(f"  - 節點數: {mst_forest.number_of_nodes()}")
    logger.info(f"  - 邊數: {mst_forest.number_of_edges()}")
    logger.info(f"  - 連通分量: {num_components}")

    # ========== 完成 ==========
    logger.info("\n" + "=" * 70)
    logger.info("神經網路重建流程完成")
    logger.info("=" * 70)

    return mst_forest


__all__ = ["build_neural_network"]
