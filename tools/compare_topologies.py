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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import csv

import numpy as np
import networkx as nx

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.spatial.distance import cdist


# ============================================================================
# 平均 Hausdorff 距離計算
# ============================================================================


def compute_average_hausdorff(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> float:
    """
    計算兩個點集之間的平均 Hausdorff 距離

    Average Hausdorff Distance 定義:
    - d(A→B) = mean(min_distance(a, B) for a in A)
    - d(B→A) = mean(min_distance(b, A) for b in B)
    - avg_hausdorff(A, B) = (d(A→B) + d(B→A)) / 2

    與傳統的 Hausdorff 距離（取最大值）不同，平均 Hausdorff 距離
    對離群點更加穩健，能更好地反映整體的相似度。

    Args:
        points_a: 點集 A，形狀 (M, 2)，每行為 [y, x]
        points_b: 點集 B，形狀 (N, 2)，每行為 [y, x]

    Returns:
        平均 Hausdorff 距離（像素單位）

    Raises:
        ValueError: 如果任一點集為空
    """
    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError("點集不能為空")

    # 驗證輸入形狀
    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError(f"points_a 應為形狀 (M, 2)，實際為 {points_a.shape}")
    if points_b.ndim != 2 or points_b.shape[1] != 2:
        raise ValueError(f"points_b 應為形狀 (N, 2)，實際為 {points_b.shape}")

    # 計算距離矩陣：dist_matrix[i, j] = distance(points_a[i], points_b[j])
    # 形狀: (M, N)，其中 M = len(points_a), N = len(points_b)
    dist_matrix = cdist(points_a, points_b, metric='euclidean')

    # 對每個點在 A 中，找到它到 B 中所有點的最小距離
    # min_dist_a_to_b[i] = min_j(dist_matrix[i, j])
    min_dist_a_to_b = np.min(dist_matrix, axis=1)  # 形狀: (M,)

    # 對每個點在 B 中，找到它到 A 中所有點的最小距離
    # min_dist_b_to_a[j] = min_i(dist_matrix[i, j])
    min_dist_b_to_a = np.min(dist_matrix, axis=0)  # 形狀: (N,)

    # 計算雙向平均距離
    d_a_to_b = np.mean(min_dist_a_to_b)
    d_b_to_a = np.mean(min_dist_b_to_a)

    # 返回對稱的平均距離
    avg_hausdorff_dist = (d_a_to_b + d_b_to_a) / 2.0

    return float(avg_hausdorff_dist)


# ============================================================================
# 拓樸載入器
# ============================================================================


class TopologyLoader:
    """
    拓樸載入器

    支援多種拓樸檔案格式的讀取。
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load(self, file_path: Path) -> Optional[nx.Graph]:
        """
        從檔案載入拓樸圖

        Args:
            file_path: 拓樸檔案路徑

        Returns:
            NetworkX 圖，失敗則返回 None
        """
        if not file_path.exists():
            self.logger.error(f"檔案不存在: {file_path}")
            return None

        suffix = file_path.suffix.lower()

        try:
            if suffix == '.graphml':
                return self._load_graphml(file_path)
            elif suffix in ['.pkl', '.pickle']:
                return self._load_pickle(file_path)
            elif suffix == '.gml':
                return self._load_gml(file_path)
            elif suffix == '.json':
                return self._load_json(file_path)
            else:
                self.logger.error(f"不支援的檔案格式: {suffix}")
                return None
        except Exception as e:
            self.logger.error(f"載入拓樸失敗 {file_path}: {e}", exc_info=True)
            return None

    def _load_graphml(self, file_path: Path) -> nx.Graph:
        """載入 GraphML 格式"""
        self.logger.debug(f"載入 GraphML: {file_path}")
        graph = nx.read_graphml(file_path)

        # GraphML 可能將節點存儲為字符串，需要轉換回元組
        return self._convert_node_labels(graph)

    def _load_pickle(self, file_path: Path) -> nx.Graph:
        """載入 Pickle 格式"""
        self.logger.debug(f"載入 Pickle: {file_path}")
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _load_gml(self, file_path: Path) -> nx.Graph:
        """載入 GML 格式"""
        self.logger.debug(f"載入 GML: {file_path}")
        graph = nx.read_gml(file_path)
        return self._convert_node_labels(graph)

    def _load_json(self, file_path: Path) -> nx.Graph:
        """載入自訂 JSON 格式"""
        self.logger.debug(f"載入 JSON: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = nx.Graph()

        # 添加節點
        for node_data in data.get('nodes', []):
            node_id = tuple(node_data['id'])  # (y, x)
            attrs = {k: v for k, v in node_data.items() if k != 'id'}
            graph.add_node(node_id, **attrs)

        # 添加邊
        for edge_data in data.get('edges', []):
            source = tuple(edge_data['source'])
            target = tuple(edge_data['target'])
            attrs = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}

            # 轉換 path 列表為元組列表
            if 'path' in attrs and isinstance(attrs['path'], list):
                attrs['path'] = [tuple(p) if isinstance(p, list) else p for p in attrs['path']]
            if 'path-coordinates' in attrs and isinstance(attrs['path-coordinates'], list):
                attrs['path-coordinates'] = [tuple(p) if isinstance(p, list) else p for p in attrs['path-coordinates']]

            graph.add_edge(source, target, **attrs)

        return graph

    def _convert_node_labels(self, graph: nx.Graph) -> nx.Graph:
        """
        轉換節點標籤為 (y, x) 元組格式

        GraphML 等格式可能將節點存儲為字符串，需要轉換回元組。
        """
        # 檢查第一個節點的格式
        if graph.number_of_nodes() == 0:
            return graph

        first_node = list(graph.nodes())[0]

        # 如果已經是元組，直接返回
        if isinstance(first_node, tuple):
            return graph

        # 如果是字符串，需要解析
        if isinstance(first_node, str):
            mapping = {}
            for node in graph.nodes():
                # 解析 "(y, x)" 格式的字符串
                try:
                    # 移除括號和空格，分割
                    clean = node.strip('()').replace(' ', '')
                    parts = clean.split(',')
                    if len(parts) == 2:
                        y, x = float(parts[0]), float(parts[1])
                        mapping[node] = (y, x)
                    else:
                        self.logger.warning(f"無法解析節點標籤: {node}")
                        mapping[node] = node
                except Exception as e:
                    self.logger.warning(f"解析節點標籤失敗 {node}: {e}")
                    mapping[node] = node

            return nx.relabel_nodes(graph, mapping)

        return graph

    def save(self, graph: nx.Graph, file_path: Path, format: str = 'graphml'):
        """
        儲存拓樸圖到檔案

        Args:
            graph: NetworkX 圖
            file_path: 輸出檔案路徑
            format: 輸出格式 (graphml, pickle, gml, json)
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'graphml':
            self._save_graphml(graph, file_path)
        elif format == 'pickle':
            self._save_pickle(graph, file_path)
        elif format == 'gml':
            self._save_gml(graph, file_path)
        elif format == 'json':
            self._save_json(graph, file_path)
        else:
            raise ValueError(f"不支援的輸出格式: {format}")

    def _save_graphml(self, graph: nx.Graph, file_path: Path):
        """儲存為 GraphML 格式"""
        # NetworkX 的 write_graphml 需要字符串節點
        # 將元組節點轉換為字符串
        graph_copy = nx.Graph()

        for node, attrs in graph.nodes(data=True):
            node_str = f"{node[0]},{node[1]}"
            graph_copy.add_node(node_str, **attrs)

        for u, v, attrs in graph.edges(data=True):
            u_str = f"{u[0]},{u[1]}"
            v_str = f"{v[0]},{v[1]}"
            # GraphML 不支持列表，需要轉換
            attrs_copy = {}
            for k, val in attrs.items():
                if isinstance(val, (list, np.ndarray)):
                    attrs_copy[k] = str(val)
                else:
                    attrs_copy[k] = val
            graph_copy.add_edge(u_str, v_str, **attrs_copy)

        nx.write_graphml(graph_copy, file_path)
        self.logger.info(f"已儲存 GraphML: {file_path}")

    def _save_pickle(self, graph: nx.Graph, file_path: Path):
        """儲存為 Pickle 格式"""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)
        self.logger.info(f"已儲存 Pickle: {file_path}")

    def _save_gml(self, graph: nx.Graph, file_path: Path):
        """儲存為 GML 格式"""
        # GML 也需要字符串節點
        graph_copy = nx.Graph()

        for node, attrs in graph.nodes(data=True):
            node_str = f"{node[0]},{node[1]}"
            graph_copy.add_node(node_str, **attrs)

        for u, v, attrs in graph.edges(data=True):
            u_str = f"{u[0]},{u[1]}"
            v_str = f"{v[0]},{v[1]}"
            graph_copy.add_edge(u_str, v_str, **attrs)

        nx.write_gml(graph_copy, file_path)
        self.logger.info(f"已儲存 GML: {file_path}")

    def _save_json(self, graph: nx.Graph, file_path: Path):
        """儲存為自訂 JSON 格式"""
        data = {
            'nodes': [],
            'edges': []
        }

        # 添加節點
        for node, attrs in graph.nodes(data=True):
            node_data = {'id': list(node)}
            node_data.update(attrs)
            data['nodes'].append(node_data)

        # 添加邊
        for u, v, attrs in graph.edges(data=True):
            edge_data = {
                'source': list(u),
                'target': list(v)
            }
            # 轉換屬性
            for k, val in attrs.items():
                if isinstance(val, (list, tuple)):
                    edge_data[k] = [list(p) if isinstance(p, tuple) else p for p in val]
                elif isinstance(val, np.ndarray):
                    edge_data[k] = val.tolist()
                else:
                    edge_data[k] = val
            data['edges'].append(edge_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"已儲存 JSON: {file_path}")


# ============================================================================
# 拓樸比對器
# ============================================================================


class TopologyComparator:
    """
    拓樸比對器

    計算兩個拓樸圖之間的平均 Hausdorff 距離。
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _extract_all_points(self, graph: nx.Graph) -> np.ndarray:
        """
        從圖中提取所有點（包括節點和邊上的路徑點）

        此方法會提取：
        1. 圖的所有節點座標
        2. 圖的所有邊上的路徑點座標

        邊路徑支援兩種屬性名稱：
        - 'path': 預測圖（Pipeline 生成）使用
        - 'path-coordinates': GT 圖（從標註生成）使用

        Args:
            graph: NetworkX 圖，節點為 (y, x) 座標

        Returns:
            所有點的陣列，形狀 (N, 2)，每行為 [y, x]
            如果圖為空，返回形狀為 (0, 2) 的空陣列
        """
        # 提取所有節點
        points = list(graph.nodes())
        num_nodes = len(points)

        # 提取所有邊上的路徑點
        num_path_points = 0
        edges_with_path = 0

        for u, v, edge_data in graph.edges(data=True):
            # 嘗試兩種屬性名稱
            path = edge_data.get('path')
            if path is None:
                path = edge_data.get('path-coordinates')

            if path is not None and len(path) > 0:
                # path 是一個 (y, x) 元組的列表
                points.extend(path)
                num_path_points += len(path)
                edges_with_path += 1

        # 記錄提取資訊
        total_edges = graph.number_of_edges()
        if total_edges > 0:
            self.logger.debug(
                f"提取點集: {num_nodes} 節點 + {num_path_points} 邊路徑點 "
                f"({edges_with_path}/{total_edges} 條邊包含路徑) = {len(points)} 總點數"
            )
        else:
            self.logger.debug(f"提取點集: {num_nodes} 節點（無邊）")

        # 轉換為 numpy 陣列
        if len(points) == 0:
            return np.array([]).reshape(0, 2)

        points_array = np.array(points, dtype=np.float64)

        # 去除重複點（例如節點可能與邊的端點重複）
        # 這可以減少計算量並提高效率
        points_unique = np.unique(points_array, axis=0)

        if len(points_unique) < len(points_array):
            self.logger.debug(
                f"去除 {len(points_array) - len(points_unique)} 個重複點，"
                f"剩餘 {len(points_unique)} 個唯一點"
            )

        return points_unique

    def compare(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        label1: str = "圖1",
        label2: str = "圖2"
    ) -> Dict:
        """
        比對兩個拓樸圖

        Args:
            graph1: 第一個圖
            graph2: 第二個圖
            label1: 第一個圖的標籤
            label2: 第二個圖的標籤

        Returns:
            包含比對結果的字典
        """
        result = {
            'label1': label1,
            'label2': label2,
            'num_nodes1': graph1.number_of_nodes(),
            'num_nodes2': graph2.number_of_nodes(),
            'num_edges1': graph1.number_of_edges(),
            'num_edges2': graph2.number_of_edges(),
            'num_points1': None,
            'num_points2': None,
            'hausdorff_distance': None,
            'status': 'success',
            'error': None
        }

        try:
            # 提取點集
            points1 = self._extract_all_points(graph1)
            points2 = self._extract_all_points(graph2)

            result['num_points1'] = len(points1)
            result['num_points2'] = len(points2)

            if len(points1) == 0 or len(points2) == 0:
                result['status'] = 'failed'
                result['error'] = 'empty_point_set'
                self.logger.warning(f"點集為空: {label1}={len(points1)}, {label2}={len(points2)}")
                return result

            # 計算平均 Hausdorff 距離
            distance = compute_average_hausdorff(points1, points2)
            result['hausdorff_distance'] = float(distance)

            self.logger.info(
                f"比對完成: {label1} vs {label2} = {distance:.4f} "
                f"(點數: {len(points1)} vs {len(points2)})"
            )

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.logger.error(f"比對失敗: {e}", exc_info=True)

        return result


# ============================================================================
# 主程式
# ============================================================================


def setup_logging(verbose: bool = False):
    """設定日誌"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        graph1, graph2,
        label1=Path(args.topology1).stem,
        label2=Path(args.topology2).stem
    )

    # 輸出結果
    print("\n" + "=" * 80)
    print("比對結果")
    print("=" * 80)
    print(f"拓樸 1: {result['label1']}")
    print(f"  節點數: {result['num_nodes1']}")
    print(f"  邊數: {result['num_edges1']}")
    print(f"  總點數: {result['num_points1']}")
    print()
    print(f"拓樸 2: {result['label2']}")
    print(f"  節點數: {result['num_nodes2']}")
    print(f"  邊數: {result['num_edges2']}")
    print(f"  總點數: {result['num_points2']}")
    print()

    if result['status'] == 'success':
        print(f"平均 Hausdorff 距離: {result['hausdorff_distance']:.4f} 像素")
    else:
        print(f"狀態: {result['status']}")
        print(f"錯誤: {result['error']}")

    print("=" * 80)

    # 儲存結果（如果指定）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"結果已儲存: {output_path}")

    return 0 if result['status'] == 'success' else 1


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
    extensions = ['.graphml', '.pkl', '.pickle', '.gml', '.json']

    pred_files = []
    for ext in extensions:
        pred_files.extend(pred_dir.glob(f'*{ext}'))

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
            graph_pred, graph_gt,
            label1=f"{sample_id}_pred",
            label2=f"{sample_id}_gt"
        )
        result['sample_id'] = sample_id
        results.append(result)

    # 輸出統計
    print("\n" + "=" * 80)
    print("批次比對統計")
    print("=" * 80)
    print(f"總共比對: {len(results)} 對")

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"成功: {len(successful)}")
    print(f"失敗: {len(failed)}")

    if successful:
        distances = [r['hausdorff_distance'] for r in successful]
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
        if output_path.suffix == '.csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'sample_id', 'hausdorff_distance',
                    'num_nodes1', 'num_nodes2',
                    'num_edges1', 'num_edges2',
                    'num_points1', 'num_points2',
                    'status', 'error'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        else:
            # JSON 格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"結果已儲存: {output_path}")

    return 0


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='拓樸比對工具 - 計算兩個拓樸圖之間的平均 Hausdorff 距離',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 比對兩個拓樸檔案
  %(prog)s --topology1 pred.graphml --topology2 gt.graphml

  # 批次比對
  %(prog)s --batch --pred-dir predictions/ --gt-dir ground_truth/ --output results.csv

  # 詳細輸出
  %(prog)s --topology1 pred.pkl --topology2 gt.pkl --verbose
        """
    )

    # 模式選擇
    parser.add_argument(
        '--batch',
        action='store_true',
        help='批次模式：比對兩個目錄中的所有配對拓樸'
    )

    # 單一比對參數
    parser.add_argument(
        '--topology1',
        type=str,
        help='第一個拓樸檔案路徑'
    )

    parser.add_argument(
        '--topology2',
        type=str,
        help='第二個拓樸檔案路徑'
    )

    # 批次比對參數
    parser.add_argument(
        '--pred-dir',
        type=str,
        help='預測拓樸目錄（批次模式）'
    )

    parser.add_argument(
        '--gt-dir',
        type=str,
        help='Ground truth 拓樸目錄（批次模式）'
    )

    # 共用參數
    parser.add_argument(
        '--output',
        type=str,
        help='輸出檔案路徑（JSON 或 CSV 格式）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='啟用詳細日誌輸出'
    )

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
