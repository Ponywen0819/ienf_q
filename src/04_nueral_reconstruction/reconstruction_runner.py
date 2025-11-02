"""
重建流程協調器

整合所有模組，執行完整的 MST 重建流程
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import networkx as nx
import json
import cv2

from mst_builder import MSTBuilder
from visualization import MSTVisualizer


@dataclass
class ReconstructionConfig:
    """重建配置"""
    max_edge_cost: float = 150        # MST 邊成本閾值
    min_branch_angle: float = 30      # 銳角分支閾值（度）
    min_quality_threshold: float = 80  # 路徑質量閾值
    verbose: bool = True


class ReconstructionRunner:
    """
    重建流程協調器

    整合 MST 構建、拓撲驗證、質量評估和視覺化
    """

    def __init__(self, config: ReconstructionConfig):
        """
        初始化重建流程

        Args:
            config: 重建配置
        """
        self.config = config
        self.mst_builder = MSTBuilder(max_edge_cost=config.max_edge_cost)

        # 稍後初始化的模組（階段二）
        self.validator = None
        self.quality_checker = None
        self.visualizer = None

    def run(
        self,
        graph_path: str,
        seeds_path: str,
        green_channel_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        執行完整重建流程

        Args:
            graph_path: 網路圖檔案路徑 (network.graphml)
            seeds_path: 種子檔案路徑 (seeds.json)
            green_channel_path: 綠色通道影像路徑
            output_dir: 輸出目錄

        Returns:
            results: 包含所有統計和驗證結果的字典
        """
        print("=" * 60)
        print("MST 神經纖維重建")
        print("=" * 60)

        # 創建輸出目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ========== 步驟 1: 載入資料 ==========
        print("\n[1/5] 載入資料...")
        try:
            G = self._load_graph(graph_path)
            seeds = self._load_seeds(seeds_path)
            green_channel = self._load_image(green_channel_path)
            print(f"  ✓ 圖: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
            print(f"  ✓ 種子: {len(seeds)} 個")
            print(f"  ✓ 影像: {green_channel.shape}")
        except Exception as e:
            print(f"  ✗ 載入失敗: {e}")
            raise

        # ========== 步驟 2: 構建 MST 森林 ==========
        print("\n[2/5] 構建 MST 森林...")
        try:
            forest = self.mst_builder.build_constrained_mst_forest(G)
            stats = self.mst_builder.get_forest_statistics(forest)

            if self.config.verbose:
                self.mst_builder.print_statistics(stats, verbose=True)
        except Exception as e:
            print(f"  ✗ MST 構建失敗: {e}")
            raise

        # ========== 步驟 3: 拓撲驗證（階段二實作） ==========
        print("\n[3/5] 拓撲驗證...")
        print("  ⚠️  尚未實作（階段二）")
        validation_results = {}

        # ========== 步驟 4: 路徑質量評估（階段二實作） ==========
        print("\n[4/5] 路徑質量評估...")
        print("  ⚠️  尚未實作（階段二）")
        quality_results = {}

        # ========== 步驟 5: 保存結果 ==========
        print("\n[5/5] 保存結果...")
        try:
            # 初始化視覺化器
            visualizer = MSTVisualizer(green_channel)
            visualizer.set_seeds(seeds)

            self._save_outputs(
                forest=forest,
                seeds=seeds,
                stats=stats,
                validation_results=validation_results,
                quality_results=quality_results,
                visualizer=visualizer,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"  ✗ 保存失敗: {e}")
            raise

        print("\n" + "=" * 60)
        print("✓ 重建完成！")
        print("=" * 60)

        # 返回結果
        return {
            'forest': forest,
            'stats': stats,
            'validation': validation_results,
            'quality': quality_results
        }

    def _load_graph(self, graph_path: str) -> nx.Graph:
        """載入 NetworkX 圖"""
        if not Path(graph_path).exists():
            raise FileNotFoundError(f"圖檔案不存在: {graph_path}")

        G = nx.read_graphml(graph_path)
        return G

    def _load_seeds(self, seeds_path: str) -> list:
        """載入種子資料"""
        if not Path(seeds_path).exists():
            raise FileNotFoundError(f"種子檔案不存在: {seeds_path}")

        with open(seeds_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 從 JSON 中提取 seeds 列表
        seeds_data = data.get('seeds', [])
        if not seeds_data:
            raise ValueError(f"seeds.json 格式錯誤：找不到 'seeds' 欄位")

        # 簡化的種子物件（只需要座標）
        seeds = []
        for seed in seeds_data:
            seeds.append({
                'id': seed['seed_id'],
                'x': seed['position']['x'],
                'y': seed['position']['y'],
                'component_id': seed.get('component_id'),
                'seed_type': seed.get('type')
            })

        return seeds

    def _load_image(self, image_path: str):
        """載入影像"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"影像檔案不存在: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        return image

    def _save_outputs(
        self,
        forest: nx.Graph,
        seeds: list,
        stats: Dict[str, Any],
        validation_results: Dict[str, Any],
        quality_results: Dict[str, Any],
        visualizer: MSTVisualizer,
        output_dir: Path
    ):
        """保存所有輸出檔案"""

        # 1. 保存 MST 森林（GraphML）
        try:
            graphml_file = output_dir / 'mst_forest.graphml'
            nx.write_graphml(forest, str(graphml_file))
            print(f"  ✓ MST 森林圖: {graphml_file}")
        except Exception as e:
            print(f"  ✗ 無法保存 GraphML: {e}")

        # 2. 保存 MST 森林（JSON with paths）
        try:
            json_file = output_dir / 'mst_forest_with_paths.json'
            forest_data = self._forest_to_json(forest)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(forest_data, f, indent=2, default=str)
            print(f"  ✓ MST 森林資料: {json_file}")
        except Exception as e:
            print(f"  ✗ 無法保存 JSON: {e}")

        # 3. 保存統計報告
        try:
            summary_file = output_dir / 'reconstruction_summary.txt'
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("MST 神經纖維重建統計報告\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"連通分量數: {stats['num_components']}\n")
                f.write(f"總節點數: {stats['total_nodes']}\n")
                f.write(f"總邊數: {stats['total_edges']}\n\n")

                f.write("各分量詳情：\n")
                f.write("-" * 60 + "\n")
                for info in stats['components_info'][:20]:  # 前 20 個
                    f.write(f"分量 {info['component_id']}: "
                           f"{info['num_nodes']} 節點, "
                           f"{info['num_edges']} 邊, "
                           f"平均權重={info['avg_weight']:.2f}\n")

                if len(stats['components_info']) > 20:
                    f.write(f"\n... ({len(stats['components_info']) - 20} 個分量未顯示)\n")

            print(f"  ✓ 統計報告: {summary_file}")
        except Exception as e:
            print(f"  ✗ 無法保存統計報告: {e}")

        # 4. 保存驗證報告（階段二）
        if validation_results:
            try:
                validation_file = output_dir / 'validation_report.json'
                with open(validation_file, 'w', encoding='utf-8') as f:
                    json.dump(validation_results, f, indent=2)
                print(f"  ✓ 驗證報告: {validation_file}")
            except Exception as e:
                print(f"  ✗ 無法保存驗證報告: {e}")

        # 5. 保存質量評估（階段二）
        if quality_results:
            try:
                quality_file = output_dir / 'quality_assessment.json'
                with open(quality_file, 'w', encoding='utf-8') as f:
                    json.dump(quality_results, f, indent=2)
                print(f"  ✓ 質量評估: {quality_file}")
            except Exception as e:
                print(f"  ✗ 無法保存質量評估: {e}")

        # 6. 生成視覺化
        print("\n  生成視覺化...")
        try:
            # 6.1 完整森林視覺化
            full_viz = output_dir / 'mst_forest_full.png'
            visualizer.visualize_mst_forest(forest, str(full_viz), zoom=False)

            # 6.2 放大森林視覺化
            zoomed_viz = output_dir / 'mst_forest_zoomed.png'
            visualizer.visualize_mst_forest(forest, str(zoomed_viz), zoom=True, zoom_radius=200)

            # 6.3 分量分解視覺化
            breakdown_viz = output_dir / 'component_breakdown.png'
            visualizer.visualize_component_breakdown(forest, str(breakdown_viz), max_components=12)

            # 6.4 路徑質量熱力圖
            heatmap_viz = output_dir / 'quality_heatmap.png'
            visualizer.visualize_quality_heatmap(forest, str(heatmap_viz))

        except Exception as e:
            print(f"  ✗ 視覺化生成失敗: {e}")
            import traceback
            traceback.print_exc()

    def _forest_to_json(self, forest: nx.Graph) -> Dict[str, Any]:
        """將森林轉換為 JSON 格式"""
        nodes = []
        for node_id, data in forest.nodes(data=True):
            nodes.append({
                'id': node_id,
                **data
            })

        edges = []
        for u, v, data in forest.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                **data
            })

        return {
            'nodes': nodes,
            'edges': edges,
            'num_components': nx.number_connected_components(forest)
        }


if __name__ == '__main__':
    # 測試程式碼
    print("重建流程測試")

    config = ReconstructionConfig(
        max_edge_cost=150,
        verbose=True
    )

    runner = ReconstructionRunner(config)

    # 測試（需要實際檔案）
    # results = runner.run(
    #     graph_path='output/network/network.graphml',
    #     seeds_path='output/seeds/seeds.json',
    #     green_channel_path='test/green_channel.png',
    #     output_dir='output/reconstruction'
    # )
