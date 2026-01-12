"""
區域標注器 (Region Labeler)

為神經纖維拓樸結構中的每個節點標注其所屬區域（表皮/真皮），
並為每條邊標注是否跨越表皮/真皮邊界。
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
import copy

logger = logging.getLogger(__name__)


class RegionLabeler:
    """
    區域標注器 - 為拓樸節點標注區域屬性，為邊標注跨越屬性

    根據表皮遮罩判斷每個節點位於表皮(epidermis)或真皮(dermis)區域，
    並標注每條邊是否跨越表皮/真皮邊界。
    """

    def __init__(self):
        """初始化區域標注器"""
        logger.info("Initialized RegionLabeler")

    def label_topology(
        self,
        topology: Dict,
        epidermis_mask: np.ndarray
    ) -> Dict:
        """
        為拓樸結構標注區域資訊

        Args:
            topology: 拓樸結構字典
                {
                    'nodes': [{'id': int, 'position': (y, x), 'type': str}, ...],
                    'edges': [{'source': int, 'target': int, 'path': [...], 'length': float}, ...]
                }
            epidermis_mask: 表皮遮罩 (uint8, 255=表皮, 0=真皮)

        Returns:
            labeled_topology: 標注後的拓樸結構
                {
                    'nodes': [
                        {'id': int, 'position': (y, x), 'type': str, 'region': 'epidermis'|'dermis'},
                        ...
                    ],
                    'edges': [
                        {'source': int, 'target': int, 'path': [...], 'length': float, 'is_crossing': bool},
                        ...
                    ]
                }
        """
        # 深拷貝以避免修改原始資料
        labeled_topology = copy.deepcopy(topology)

        # 建立節點 ID 到區域的映射
        node_region_map: Dict[int, str] = {}

        # 標注每個節點的區域
        for node in labeled_topology['nodes']:
            node_id = node['id']
            position = node['position']
            region = self._get_node_region(position, epidermis_mask)
            node['region'] = region
            node_region_map[node_id] = region

        # 統計節點區域分布
        epidermis_count = sum(1 for r in node_region_map.values() if r == 'epidermis')
        dermis_count = len(node_region_map) - epidermis_count
        logger.info(f"節點區域分布: 表皮={epidermis_count}, 真皮={dermis_count}")

        # 標注每條邊是否跨越邊界
        crossing_count = 0
        for edge in labeled_topology['edges']:
            source_region = node_region_map[edge['source']]
            target_region = node_region_map[edge['target']]
            is_crossing = source_region != target_region
            edge['is_crossing'] = is_crossing
            if is_crossing:
                crossing_count += 1

        logger.info(f"跨越邊數量: {crossing_count} / {len(labeled_topology['edges'])}")

        return labeled_topology

    def _get_node_region(
        self,
        position: Tuple[int, int],
        epidermis_mask: np.ndarray
    ) -> str:
        """
        判斷節點所屬區域

        Args:
            position: 節點座標 (y, x)
            epidermis_mask: 表皮遮罩

        Returns:
            region: 'epidermis' 或 'dermis'
        """
        y, x = position
        height, width = epidermis_mask.shape

        # 邊界檢查
        if y < 0 or y >= height or x < 0 or x >= width:
            logger.warning(f"節點座標 ({y}, {x}) 超出遮罩範圍，預設為真皮區域")
            return 'dermis'

        # 查詢遮罩值
        mask_value = epidermis_mask[y, x]

        # 255 = 表皮, 0 = 真皮
        if mask_value > 127:  # 使用閾值以處理可能的中間值
            return 'epidermis'
        else:
            return 'dermis'

    def batch_label_topologies(
        self,
        topologies: List[Dict],
        epidermis_mask: np.ndarray
    ) -> List[Dict]:
        """
        批次處理多個拓樸結構

        Args:
            topologies: 拓樸結構列表，每個元素為 {'component_id': int, 'topology': Dict}
            epidermis_mask: 表皮遮罩

        Returns:
            labeled_topologies: 標注後的拓樸結構列表
        """
        logger.info(f"批次標注 {len(topologies)} 個拓樸結構...")

        labeled_topologies = []
        total_crossings = 0

        for topo_data in topologies:
            component_id = topo_data.get('component_id', -1)
            topology = topo_data['topology']

            labeled_topology = self.label_topology(topology, epidermis_mask)

            # 計算此元件的跨越邊數
            crossings = sum(1 for e in labeled_topology['edges'] if e.get('is_crossing', False))
            total_crossings += crossings

            labeled_topologies.append({
                'component_id': component_id,
                'topology': labeled_topology
            })

        logger.info(f"批次標注完成，總跨越邊數: {total_crossings}")

        return labeled_topologies
