#!/usr/bin/env python3
"""
神經重建完整流程 (Neural Reconstruction Pipeline)

整合所有模組，執行完整的神經重建流程：
1. 連通元件提取 (Connected Components Analysis)
2. 骨架化 (Skeletonization)
3. 骨架拓樸建構與種子萃取 (Topology Building & Seed Extraction)
4. 元件配對與連接拓樸建構 (Component Pairing & Connection Topology)
5. MST 神經重建 (Neural Reconstruction) - TODO

使用範例:
    from src.nueral_reconstruction.pipeline import NeuralReconstructionPipeline

    # 設定 logging level
    import logging
    logging.basicConfig(level=logging.INFO)

    pipeline = NeuralReconstructionPipeline(
        connectivity=8,
        min_area=50,
        config_path='config/default.yaml'
    )

    results = pipeline.run(
        input_image_path='path/to/annotation.png',
        green_channel_path='path/to/green_channel.png',
        output_dir='output/reconstruction'
    )

作者: Generated with Claude Code
日期: 2025-11-17
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import cv2
import numpy as np
import networkx as nx

from .connected_components import ConnectedComponentsAnalyzer
from .skeletonization import SkeletonAnalyzer
from .seed_extraction import SkeletonTopologyBuilder, EdgeSeedExtractor
from .component_pairing import ComponentPairAnalyzer
from .graph_builder import ComponentGraphBuilder
from .mst_builder import MSTBuilder
from .config_loader import load_config, IENFConfig

# 設定 logger
logger = logging.getLogger(__name__)


class NeuralReconstructionPipeline:
    """神經重建完整流程管理器"""

    def __init__(
        self,
        # 連通元件參數（可選，用於覆蓋配置）
        connectivity: Optional[int] = None,
        min_area: Optional[int] = None,
        # 配置文件
        config: Optional[IENFConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        初始化神經重建流程

        Args:
            connectivity: 連通性 (4 或 8)，可選，用於覆蓋配置值
            min_area: 最小元件面積（像素），可選，用於覆蓋配置值
            config: IENFConfig 配置物件（可選）
            config_path: 配置文件路徑（可選）
        """
        # 載入配置
        if config is not None:
            self.config = config
            logger.info("✓ 使用提供的配置物件")
        elif config_path is not None:
            self.config = load_config(config_path)
            logger.info(f"✓ 已載入配置文件: {config_path}")
        else:
            # 嘗試使用預設配置
            try:
                self.config = load_config()
                logger.info("✓ 已載入預設配置文件")
            except FileNotFoundError:
                logger.warning("⚠️  未找到配置文件，使用內建預設值")
                self.config = IENFConfig()

        # 允許參數覆蓋配置
        if connectivity is not None:
            self.config.connected_components.connectivity = connectivity
            logger.info(f"  覆蓋配置: connectivity = {connectivity}")
        if min_area is not None:
            self.config.connected_components.min_area = min_area
            logger.info(f"  覆蓋配置: min_area = {min_area}")

        # 初始化各階段的分析器
        self.cc_analyzer = ConnectedComponentsAnalyzer(
            connectivity=self.config.connected_components.connectivity,
            min_area=self.config.connected_components.min_area
        )

        self.skeleton_analyzer = SkeletonAnalyzer()

        # 初始化種子提取器
        self.topology_builder = SkeletonTopologyBuilder()
        self.seed_extractor = EdgeSeedExtractor(
            min_edge_length=self.config.seed_extraction.base_segment_length
        )

        # 初始化元件配對分析器（需要綠色通道影像，在 run() 時設定）
        self.component_pair_analyzer = None

        # 初始化 MST 重建模組
        self.graph_builder = ComponentGraphBuilder()
        self.mst_builder = MSTBuilder()

        logger.info("=" * 70)
        logger.info("神經重建流程初始化完成")
        logger.info("=" * 70)

    def _build_mst_with_paths(
        self,
        mst_forest: nx.Graph,
        all_connections: List[Dict]
    ) -> Dict:
        """
        Build enriched MST with full path and seed information.

        Args:
            mst_forest: MST forest graph
            all_connections: All connections from pairing stage

        Returns:
            Dictionary with MST edges enriched with path data
        """
        # Create lookup for connections
        connection_lookup = {}
        for conn in all_connections:
            edge = tuple(sorted([conn['component_a_id'], conn['component_b_id']]))
            connection_lookup[edge] = conn

        # Build enriched MST edges
        mst_edges_with_paths = []
        for u, v, data in mst_forest.edges(data=True):
            edge = tuple(sorted([u, v]))
            if edge in connection_lookup:
                conn_data = connection_lookup[edge]
                mst_edges_with_paths.append({
                    'component_a_id': u,
                    'component_b_id': v,
                    'cost': data.get('weight', conn_data['cost']),
                    'distance': conn_data['distance'],
                    'seed_pair': conn_data['seed_pair'],
                    'path': conn_data['path']
                })

        return {
            'edges': mst_edges_with_paths,
            'num_edges': len(mst_edges_with_paths)
        }

    def run(
        self,
        input_image_path: str,
        green_channel_path: str,
        output_dir: Optional[str] = None,
        save_intermediates: bool = True
    ) -> Dict[str, Any]:
        """
        執行完整的神經重建流程

        Args:
            input_image_path: 輸入二值標註影像路徑
            green_channel_path: 綠色通道影像路徑（用於路徑規劃）
            output_dir: 輸出目錄（可選）
            save_intermediates: 是否保存中間結果

        Returns:
            results: 包含所有階段結果的字典
        """
        logger.info("\n" + "=" * 70)
        logger.info("開始神經重建流程")
        logger.info("=" * 70)
        logger.info(f"輸入影像: {input_image_path}")
        logger.info(f"綠色通道: {green_channel_path}")

        # 載入綠色通道影像
        green_channel = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
        if green_channel is None:
            raise FileNotFoundError(f"無法載入綠色通道影像: {green_channel_path}")
        logger.info(f"綠色通道影像大小: {green_channel.shape}")

        # 初始化元件配對分析器
        self.component_pair_analyzer = ComponentPairAnalyzer(
            green_channel=green_channel,
            max_distance_threshold=self.config.component_pairing.max_distance_threshold,
            max_cost_threshold=self.config.component_pairing.max_cost_threshold
        )

        # 建立輸出目錄
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"輸出目錄: {output_dir}")

        results = {
            'input_path': input_image_path,
            'green_channel_path': green_channel_path,
            'output_dir': output_dir,
            'stages': {}
        }

        # ========== 階段 1: 連通元件提取 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 1: 連通元件提取")
        logger.info("=" * 70)

        regions = self.cc_analyzer.process(input_image_path)

        results['stages']['connected_components'] = {
            'num_components': len(regions),
            'regions': regions
        }

        logger.info(f"\n✓ 階段 1 完成: 提取了 {len(regions)} 個連通元件")

        # ========== 階段 2: 骨架化 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 2: 骨架化分析")
        logger.info("=" * 70)

        skeleton_results = self.skeleton_analyzer.batch_process(regions)

        results['stages']['skeletonization'] = {
            'num_skeletons': len(skeleton_results),
            'skeleton_data': skeleton_results
        }

        # 統計骨架資訊
        total_endpoints = sum(s['num_endpoints'] for s in skeleton_results)
        total_branchpoints = sum(s['num_branchpoints'] for s in skeleton_results)

        logger.info(f"\n✓ 階段 2 完成: 處理了 {len(skeleton_results)} 個骨架")
        logger.info(f"  總端點數: {total_endpoints}")
        logger.info(f"  總分支點數: {total_branchpoints}")

        # ========== 階段 3: 骨架拓樸建構與種子萃取 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 3: 骨架拓樸建構與種子萃取")
        logger.info("=" * 70)

        all_topologies = []
        all_seeds = []
        total_nodes = 0
        total_edges = 0

        for skeleton_data in skeleton_results:
            # 從 region.label 獲取元件 ID
            component_id = skeleton_data['region'].label
            skeleton_mask = skeleton_data['skeleton']
            endpoints = skeleton_data['endpoints']
            branchpoints = skeleton_data['branchpoints']

            logger.info(f"\n處理元件 {component_id}...")

            # 將 endpoints 和 branchpoints 從字典格式轉換為 tuple 格式
            # 格式從 {'x': int, 'y': int} 轉換為 (y, x)
            endpoints_tuples = [(pt['y'], pt['x']) for pt in endpoints]
            branchpoints_tuples = [(pt['y'], pt['x']) for pt in branchpoints]

            # 建構拓樸
            topology = self.topology_builder.build_topology(
                skeleton_mask, endpoints_tuples, branchpoints_tuples
            )

            # 將座標從局部(相對於組件 bounding box)轉換為全局(相對於完整影像)
            # 這樣視覺化和配對分析才能正確使用座標
            region = skeleton_data['region']
            minr, minc, maxr, maxc = region.bbox

            # 轉換拓樸節點位置到全局座標
            for node in topology['nodes']:
                y, x = node['position']
                node['position'] = (y + minr, x + minc)

            # 轉換拓樸邊路徑到全局座標
            for edge in topology['edges']:
                global_path = []
                for y, x in edge['path']:
                    global_path.append((y + minr, x + minc))
                edge['path'] = global_path

            total_nodes += len(topology['nodes'])
            total_edges += len(topology['edges'])

            # 從拓樸邊抽取種子
            seeds = self.seed_extractor.extract_seeds_from_topology(
                topology,
                segment_length=self.config.seed_extraction.base_segment_length
            )

            # 添加節點作為種子（端點和分支點）
            for node in topology['nodes']:
                seeds.append({
                    'position': node['position'],
                    'type': node['type'],
                    'component_id': component_id
                })

            # 記錄元件 ID
            for seed in seeds:
                if 'component_id' not in seed:
                    seed['component_id'] = component_id

            all_topologies.append({
                'component_id': component_id,
                'topology': topology
            })

            all_seeds.extend(seeds)

            logger.info(f"  元件 {component_id}: {len(topology['nodes'])} 節點, "
                       f"{len(topology['edges'])} 邊, {len(seeds)} 種子")

        results['stages']['topology_and_seeds'] = {
            'num_components': len(all_topologies),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_seeds': len(all_seeds),
            'topologies': all_topologies,
            'seeds': all_seeds
        }

        logger.info(f"\n✓ 階段 3 完成:")
        logger.info(f"  總節點數: {total_nodes}")
        logger.info(f"  總邊數: {total_edges}")
        logger.info(f"  總種子數: {len(all_seeds)}")

        # ========== 階段 4: 元件配對與連接拓樸建構 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 4: 元件配對與連接拓樸建構")
        logger.info("=" * 70)

        # 準備元件資料（每個元件的種子列表）
        components_data = []
        for topo_data in all_topologies:
            component_id = topo_data['component_id']
            # 找出屬於這個元件的所有種子
            component_seeds = [s for s in all_seeds if s['component_id'] == component_id]
            components_data.append({
                'component_id': component_id,
                'seeds': component_seeds
            })

        logger.info(f"準備配對分析: {len(components_data)} 個元件")

        # 執行批次元件配對分析
        pairing_results = self.component_pair_analyzer.batch_analyze_components(
            components_data
        )

        results['stages']['component_pairing'] = {
            'num_components': pairing_results['num_components'],
            'num_pairs_analyzed': pairing_results['num_pairs_analyzed'],
            'num_connections': pairing_results['num_connections'],
            'connections': pairing_results['connections'],
            'all_pair_results': pairing_results['all_pair_results']
        }

        logger.info(f"\n✓ 階段 4 完成:")
        logger.info(f"  分析配對數: {pairing_results['num_pairs_analyzed']}")
        logger.info(f"  建議連接數: {pairing_results['num_connections']}")
        logger.info(f"  連接拓樸已建立")

        # ========== 階段 5: MST 重建 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 5: MST 神經重建")
        logger.info("=" * 70)

        # Step 1: Build component graph from pairing results
        logger.info("步驟 1/4: 建構元件圖...")
        component_graph = self.graph_builder.build_graph(pairing_results)
        logger.info(f"  ✓ 元件圖: {component_graph.number_of_nodes()} 節點, "
                   f"{component_graph.number_of_edges()} 邊")

        # Step 2: Build MST forest from component graph
        logger.info("步驟 2/4: 建構 MST 森林...")
        mst_forest = self.mst_builder.build_mst_forest(component_graph)
        num_connected_components = nx.number_connected_components(mst_forest)
        logger.info(f"  ✓ MST 森林: {mst_forest.number_of_nodes()} 節點, "
                   f"{mst_forest.number_of_edges()} 邊, "
                   f"{num_connected_components} 個連通元件")

        # Step 3: Filter connections based on MST
        logger.info("步驟 3/4: 根據 MST 過濾連接...")
        filtered_connections = self.graph_builder.filter_connections_by_mst(
            pairing_results,
            mst_forest
        )
        logger.info(f"  ✓ 保留連接: {filtered_connections['num_kept']}")
        logger.info(f"  ✓ 移除連接: {filtered_connections['num_removed']}")

        # Step 4: Build enriched MST with full path information
        logger.info("步驟 4/4: 建構包含路徑資訊的豐富 MST...")
        mst_with_paths = self._build_mst_with_paths(
            mst_forest,
            pairing_results['connections']
        )
        logger.info(f"  ✓ 豐富 MST 邊數: {mst_with_paths['num_edges']}")

        # Store results
        results['stages']['mst_reconstruction'] = {
            'status': 'completed',
            'num_mst_edges': mst_forest.number_of_edges(),
            'num_mst_nodes': mst_forest.number_of_nodes(),
            'num_connected_components': num_connected_components,
            'connections_kept': filtered_connections['num_kept'],
            'connections_removed': filtered_connections['num_removed'],
            'mst_forest': mst_forest,
            'mst_with_paths': mst_with_paths,
            'filtered_connections': filtered_connections
        }

        # ========== 流程完成 ==========
        logger.info("\n" + "=" * 70)
        logger.info("神經重建流程執行完成")
        logger.info("=" * 70)
        logger.info(f"✓ 已完成所有階段: 連通元件提取, 骨架化, 拓樸建構與種子萃取, "
                   f"元件配對與連接拓樸建構, MST 重建")
        logger.info("=" * 70)

        return results