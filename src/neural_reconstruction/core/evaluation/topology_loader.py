"""
拓樸載入器模組 (Topology Loader Module)

支援多種拓樸檔案格式的讀寫：
- GraphML (.graphml) - 推薦，保留所有屬性
- Pickle (.pkl, .pickle) - NetworkX 原生格式
- GML (.gml) - 簡單文字格式
- JSON (.json) - 自訂 JSON 格式

設計原則：
- 統一的檔案格式處理介面
- 自動格式偵測
- 節點座標格式自動轉換 (string ↔ tuple)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np


class TopologyLoader:
    """
    拓樸載入器

    支援多種拓樸檔案格式的讀取和寫入。

    Examples:
        >>> loader = TopologyLoader()
        >>> graph = loader.load(Path("topology.graphml"))
        >>> loader.save(graph, Path("output.pkl"), format='pickle')
    """

    SUPPORTED_EXTENSIONS = {
        'load': ['.graphml', '.pkl', '.pickle', '.gml', '.json'],
        'save': ['graphml', 'pickle', 'gml', 'json']
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def load(self, file_path: Path) -> Optional[nx.Graph]:
        """
        從檔案載入拓樸圖

        根據副檔名自動選擇載入方法。

        Args:
            file_path: 拓樸檔案路徑

        Returns:
            NetworkX 圖，失敗則返回 None
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"檔案不存在: {file_path}")
            return None

        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS['load']:
            self.logger.error(
                f"不支援的檔案格式: {suffix}，"
                f"支援的格式: {self.SUPPORTED_EXTENSIONS['load']}"
            )
            return None

        try:
            if suffix == '.graphml':
                return self._load_graphml(file_path)
            elif suffix in ['.pkl', '.pickle']:
                return self._load_pickle(file_path)
            elif suffix == '.gml':
                return self._load_gml(file_path)
            elif suffix == '.json':
                return self._load_json(file_path)
        except Exception as e:
            self.logger.error(f"載入拓樸失敗 {file_path}: {e}", exc_info=True)
            return None

    def save(
        self,
        graph: nx.Graph,
        file_path: Path,
        format: str = 'graphml'
    ) -> None:
        """
        儲存拓樸圖到檔案

        Args:
            graph: NetworkX 圖
            file_path: 輸出檔案路徑
            format: 輸出格式 (graphml, pickle, gml, json)

        Raises:
            ValueError: 如果格式不支援
        """
        if format not in self.SUPPORTED_EXTENSIONS['save']:
            raise ValueError(
                f"不支援的輸出格式: {format}，"
                f"支援的格式: {self.SUPPORTED_EXTENSIONS['save']}"
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'graphml':
            self._save_graphml(graph, file_path)
        elif format == 'pickle':
            self._save_pickle(graph, file_path)
        elif format == 'gml':
            self._save_gml(graph, file_path)
        elif format == 'json':
            self._save_json(graph, file_path)

    # ========== 載入方法 ==========

    def _load_graphml(self, file_path: Path) -> nx.Graph:
        """載入 GraphML 格式"""
        self.logger.debug(f"載入 GraphML: {file_path}")
        graph = nx.read_graphml(file_path)
        return self._convert_node_labels_to_tuples(graph)

    def _load_pickle(self, file_path: Path) -> nx.Graph:
        """載入 Pickle 格式"""
        self.logger.debug(f"載入 Pickle: {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _load_gml(self, file_path: Path) -> nx.Graph:
        """載入 GML 格式"""
        self.logger.debug(f"載入 GML: {file_path}")
        graph = nx.read_gml(file_path)
        return self._convert_node_labels_to_tuples(graph)

    def _load_json(self, file_path: Path) -> nx.Graph:
        """載入自訂 JSON 格式"""
        self.logger.debug(f"載入 JSON: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = nx.Graph()

        # 添加節點
        for node_data in data.get('nodes', []):
            node_id = tuple(node_data['id'])
            attrs = {k: v for k, v in node_data.items() if k != 'id'}
            graph.add_node(node_id, **attrs)

        # 添加邊
        for edge_data in data.get('edges', []):
            source = tuple(edge_data['source'])
            target = tuple(edge_data['target'])
            attrs = {k: v for k, v in edge_data.items()
                    if k not in ['source', 'target']}

            # 轉換路徑列表
            for path_attr in ['path', 'path-coordinates']:
                if path_attr in attrs and isinstance(attrs[path_attr], list):
                    attrs[path_attr] = [
                        tuple(p) if isinstance(p, list) else p
                        for p in attrs[path_attr]
                    ]

            graph.add_edge(source, target, **attrs)

        return graph

    # ========== 儲存方法 ==========

    def _save_graphml(self, graph: nx.Graph, file_path: Path):
        """儲存為 GraphML 格式"""
        # 轉換節點為字串（GraphML 要求）
        graph_copy = self._convert_graph_for_string_format(graph)
        nx.write_graphml(graph_copy, file_path)
        self.logger.info(f"已儲存 GraphML: {file_path}")

    def _save_pickle(self, graph: nx.Graph, file_path: Path):
        """儲存為 Pickle 格式"""
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)
        self.logger.info(f"已儲存 Pickle: {file_path}")

    def _save_gml(self, graph: nx.Graph, file_path: Path):
        """儲存為 GML 格式"""
        graph_copy = self._convert_graph_for_string_format(graph)
        nx.write_gml(graph_copy, file_path)
        self.logger.info(f"已儲存 GML: {file_path}")

    def _save_json(self, graph: nx.Graph, file_path: Path):
        """儲存為自訂 JSON 格式"""
        data = {'nodes': [], 'edges': []}

        # 節點
        for node, attrs in graph.nodes(data=True):
            node_data = {'id': list(node)}
            node_data.update(attrs)
            data['nodes'].append(node_data)

        # 邊
        for u, v, attrs in graph.edges(data=True):
            edge_data = {'source': list(u), 'target': list(v)}
            for k, val in attrs.items():
                if isinstance(val, (list, tuple)):
                    edge_data[k] = [
                        list(p) if isinstance(p, tuple) else p
                        for p in val
                    ]
                elif isinstance(val, np.ndarray):
                    edge_data[k] = val.tolist()
                else:
                    edge_data[k] = val
            data['edges'].append(edge_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"已儲存 JSON: {file_path}")

    # ========== 輔助方法 ==========

    def _convert_node_labels_to_tuples(self, graph: nx.Graph) -> nx.Graph:
        """
        將字串節點標籤轉換為 (y, x) 元組

        GraphML/GML 等格式可能將節點存儲為字符串 "(y, x)"，
        需要轉換回元組格式。
        """
        if graph.number_of_nodes() == 0:
            return graph

        first_node = list(graph.nodes())[0]
        if isinstance(first_node, tuple):
            return graph

        if isinstance(first_node, str):
            mapping = {}
            for node in graph.nodes():
                try:
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

    def _convert_graph_for_string_format(self, graph: nx.Graph) -> nx.Graph:
        """
        將圖轉換為字串節點格式（用於 GraphML/GML）

        將 (y, x) 元組節點轉換為 "y,x" 字串格式。
        """
        graph_copy = nx.Graph()

        for node, attrs in graph.nodes(data=True):
            node_str = f"{node[0]},{node[1]}"
            graph_copy.add_node(node_str, **attrs)

        for u, v, attrs in graph.edges(data=True):
            u_str = f"{u[0]},{u[1]}"
            v_str = f"{v[0]},{v[1]}"
            # 轉換列表屬性為字串
            attrs_copy = {}
            for k, val in attrs.items():
                if isinstance(val, (list, np.ndarray)):
                    attrs_copy[k] = str(val)
                else:
                    attrs_copy[k] = val
            graph_copy.add_edge(u_str, v_str, **attrs_copy)

        return graph_copy
