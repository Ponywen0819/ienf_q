#!/usr/bin/env python3
"""
神經重建完整流程 (Neural Reconstruction Pipeline)

整合所有模組，執行完整的神經重建流程：
1. 連通元件提取 (Connected Components Analysis)
2. 元件分析 (Component Analysis) - 骨架化、拓樸建構、種子萃取
3. 元件配對與連接拓樸建構 (Component Pairing & Connection Topology)
4. MST 神經重建 (Neural Reconstruction)

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
        input_image=annotation_image,
        green_image=green_channel,
        output_dir='output/reconstruction'
    )

作者: Generated with Claude Code
日期: 2025-11-17
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import networkx as nx

from .connected_components import ConnectedComponentsAnalyzer
from .component_analyzer import ComponentAnalyzer
from .data_types import SeedPoint, ComponentAnalysisResult
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

        # 初始化元件分析器（整合骨架化、拓樸建構、種子萃取）
        self.component_analyzer = ComponentAnalyzer(
            segment_length=self.config.seed_extraction.base_segment_length,
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

    def _to_global_coords(
        self,
        result: ComponentAnalysisResult
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        將元件分析結果從局部座標轉換為全局座標

        Args:
            result: ComponentAnalysisResult（局部座標）

        Returns:
            (seeds_global, topology_dict): 全局座標的種子列表和拓樸字典
        """
        minr, minc = result.bbox[0], result.bbox[1]

        # 轉換種子座標
        seeds_global = []
        for seed in result.seeds:
            y, x = seed.position
            seeds_global.append({
                'position': (y + minr, x + minc),
                'type': seed.seed_type,
                'component_id': seed.component_id,
                'edge_id': seed.edge_id
            })

        # 轉換拓樸座標
        nodes_global = []
        for node in result.topology.nodes:
            y, x = node.position
            nodes_global.append({
                'id': node.node_id,
                'position': (y + minr, x + minc),
                'type': node.node_type
            })

        edges_global = []
        for edge in result.topology.edges:
            global_path = [(y + minr, x + minc) for y, x in edge.path]
            edges_global.append({
                'source': edge.source_id,
                'target': edge.target_id,
                'path': global_path,
                'length': edge.length
            })

        topology_dict = {
            'nodes': nodes_global,
            'edges': edges_global
        }

        return seeds_global, topology_dict

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
        input_image: np.ndarray,
        green_image: np.ndarray,
        output_dir: Optional[str] = None,
        save_intermediates: bool = True,
        stop_step: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        執行完整的神經重建流程

        Args:
            input_image: 輸入二值標註影像
            green_image: 綠色通道影像（用於路徑規劃）
            output_dir: 輸出目錄（可選）
            save_intermediates: 是否保存中間結果
            stop_step: 停止步驟（可選）
                - 'connected_components': 階段 1 後停止
                - 'component_analysis': 階段 2 後停止
                - 'component_pairing': 階段 3 後停止

        Returns:
            results: 包含所有階段結果的字典
        """
        logger.info("\n" + "=" * 70)
        logger.info("開始神經重建流程")
        logger.info("=" * 70)

        # 初始化元件配對分析器
        self.component_pair_analyzer = ComponentPairAnalyzer(
            green_channel=green_image,
            max_distance_threshold=self.config.component_pairing.max_distance_threshold,
            max_cost_threshold=self.config.component_pairing.max_cost_threshold
        )

        # 建立輸出目錄
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"輸出目錄: {output_dir}")

        results = {
            'output_dir': output_dir,
            'stages': {}
        }

        # ========== 階段 1: 連通元件提取 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 1: 連通元件提取")
        logger.info("=" * 70)

        regions = self.cc_analyzer.analyze(input_image)

        logger.info(f"\n✓ 階段 1 完成: 提取了 {len(regions)} 個連通元件")

        if stop_step == 'connected_components':
            logger.info("流程在階段 1 停止")
            return results

        # ========== 階段 2: 元件分析（骨架化 + 拓樸建構 + 種子萃取） ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 2: 元件分析（骨架化、拓樸建構、種子萃取）")
        logger.info("=" * 70)

        # 批次分析所有元件（局部座標）
        component_results = self.component_analyzer.batch_analyze(regions)

        # 轉換為全局座標並收集結果
        all_topologies = []
        all_seeds = []
        total_nodes = 0
        total_edges = 0

        for result in component_results:
            # 轉換為全局座標
            seeds_global, topology_dict = self._to_global_coords(result)

            all_topologies.append({
                'component_id': result.component_id,
                'topology': topology_dict
            })

            all_seeds.extend(seeds_global)

            total_nodes += len(result.topology.nodes)
            total_edges += len(result.topology.edges)

            logger.info(f"  元件 {result.component_id}: "
                       f"{len(result.topology.nodes)} 節點, "
                       f"{len(result.topology.edges)} 邊, "
                       f"{len(result.seeds)} 種子")

        results['stages']['component_analysis'] = {
            'num_components': len(component_results),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_seeds': len(all_seeds),
            'component_results': component_results,
            'topologies': all_topologies,
            'seeds': all_seeds
        }

        logger.info(f"\n✓ 階段 2 完成:")
        logger.info(f"  處理元件數: {len(component_results)}")
        logger.info(f"  總節點數: {total_nodes}")
        logger.info(f"  總邊數: {total_edges}")
        logger.info(f"  總種子數: {len(all_seeds)}")

        if stop_step == 'component_analysis':
            logger.info("流程在階段 2 停止")
            return results

        # ========== 階段 3: 元件配對與連接拓樸建構 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 3: 元件配對與連接拓樸建構")
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

        logger.info(f"\n✓ 階段 3 完成:")
        logger.info(f"  分析配對數: {pairing_results['num_pairs_analyzed']}")
        logger.info(f"  建議連接數: {pairing_results['num_connections']}")
        logger.info(f"  連接拓樸已建立")

        if stop_step == 'component_pairing':
            logger.info("流程在階段 3 停止")
            return results

        # ========== 階段 4: MST 重建 ==========
        logger.info("\n" + "=" * 70)
        logger.info("階段 4: MST 神經重建")
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
        logger.info(f"✓ 已完成所有階段: 連通元件提取, 元件分析, "
                   f"元件配對與連接拓樸建構, MST 重建")
        logger.info("=" * 70)

        return results
