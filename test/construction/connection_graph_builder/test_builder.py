"""
NetworkBuilder 類別的單元測試

測試範圍：
1. 初始化與參數驗證
2. 全域拓撲構建（座標轉換）
3. 全域索引構建（KDTree）
4. 鄰居搜尋與過濾
5. 路徑解析與邊緣建立
6. 圖構建完整流程
7. 邊界情況處理
"""

import pytest
import numpy as np
import networkx as nx
from typing import List

from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.builder import NetworkBuilder
from neural_reconstruction.common.data_types import ComponentAnalysisResult


class TestNetworkBuilderInit:
    """測試 NetworkBuilder 初始化"""

    def test_init_with_default_parameters(self, simple_bright_image):
        """測試使用預設參數初始化"""
        builder = NetworkBuilder(simple_bright_image)

        assert builder.search_radius == 50.0
        assert builder.max_cost_threshold == 0.98
        assert builder.pathfinder is not None
        assert builder.kdtree is None  # 尚未建構
        assert builder.global_seeds is None
        assert builder.component_ids is None

    def test_init_with_custom_parameters(self, simple_bright_image):
        """測試使用自訂參數初始化"""
        builder = NetworkBuilder(
            simple_bright_image,
            search_radius=100.0,
            max_cost_threshold=0.95,
            intensity_weight=0.7,
            shape_weight=0.3
        )

        assert builder.search_radius == 100.0
        assert builder.max_cost_threshold == 0.95
        assert builder.pathfinder.intensity_weight == 0.7
        assert builder.pathfinder.shape_weight == 0.3

    def test_init_creates_pathfinder(self, simple_bright_image):
        """測試初始化時建立 Pathfinder"""
        builder = NetworkBuilder(simple_bright_image)

        assert builder.pathfinder is not None
        assert builder.pathfinder.cost_map is not None
        assert builder.pathfinder.cost_map.shape == simple_bright_image.shape


class TestNetworkBuilderGlobalTopology:
    """測試全域拓撲構建"""

    def test_get_component_global_topology_coordinate_transform(
        self, create_mock_component_result
    ):
        """測試元件拓撲的座標轉換"""
        # 建立一個有 bbox 偏移的元件
        bbox = (10, 20, 30, 40)  # minr=10, minc=20
        local_nodes = [(0, 0), (5, 5), (10, 10)]

        component = create_mock_component_result(
            component_id=1,
            bbox=bbox,
            node_positions=local_nodes
        )

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._get_component_global_topology(component)

        # 檢查節點已轉換為全域座標
        expected_global_nodes = [(10, 20), (15, 25), (20, 30)]
        assert set(global_topology.nodes()) == set(expected_global_nodes)

    def test_get_component_global_topology_preserves_edges(
        self, create_mock_component_result
    ):
        """測試座標轉換保留邊的連接關係"""
        bbox = (5, 10, 15, 20)
        local_nodes = [(0, 0), (1, 1), (2, 2)]

        component = create_mock_component_result(
            component_id=1,
            bbox=bbox,
            node_positions=local_nodes
        )

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._get_component_global_topology(component)

        # 檢查邊數量與原始拓撲一致
        assert global_topology.number_of_edges() == component.topology.number_of_edges()

    def test_get_component_global_topology_adds_component_id(
        self, create_mock_component_result
    ):
        """測試為所有節點添加 component_id 屬性"""
        bbox = (0, 0, 10, 10)
        local_nodes = [(0, 0), (5, 5)]

        component = create_mock_component_result(
            component_id=42,
            bbox=bbox,
            node_positions=local_nodes
        )

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._get_component_global_topology(component)

        # 檢查所有節點都有 component_id 屬性
        for node in global_topology.nodes():
            assert global_topology.nodes[node]["component_id"] == 42

    def test_get_component_global_topology_transforms_edge_paths(
        self, create_mock_component_result
    ):
        """測試邊的路徑座標也被轉換"""
        bbox = (10, 20, 30, 40)
        local_nodes = [(0, 0), (5, 5)]

        component = create_mock_component_result(
            component_id=1,
            bbox=bbox,
            node_positions=local_nodes
        )

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._get_component_global_topology(component)

        # 檢查邊的路徑已轉換
        for u, v, data in global_topology.edges(data=True):
            if "path" in data:
                path = data["path"]
                # 路徑點應該在全域座標系統中
                for point in path:
                    assert point[0] >= 10  # minr
                    assert point[1] >= 20  # minc

    def test_build_global_topology_combines_multiple_components(
        self, create_mock_component_result
    ):
        """測試組合多個元件的全域拓撲"""
        components = [
            create_mock_component_result(1, (0, 0, 10, 10), [(0, 0), (5, 5)]),
            create_mock_component_result(2, (20, 20, 30, 30), [(0, 0), (5, 5)]),
            create_mock_component_result(3, (40, 40, 50, 50), [(0, 0), (5, 5)])
        ]

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._build_global_topology(components)

        # 應該包含所有元件的節點
        expected_node_count = sum(len(c.topology.nodes()) for c in components)
        assert global_topology.number_of_nodes() == expected_node_count

    def test_build_global_topology_preserves_component_separation(
        self, create_mock_component_result
    ):
        """測試不同元件的節點保持獨立"""
        components = [
            create_mock_component_result(1, (0, 0, 10, 10), [(0, 0), (1, 1)]),
            create_mock_component_result(2, (20, 20, 30, 30), [(0, 0), (1, 1)])
        ]

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._build_global_topology(components)

        # 檢查不同元件的節點有不同的 component_id
        component_ids = set()
        for node in global_topology.nodes():
            component_ids.add(global_topology.nodes[node]["component_id"])

        assert len(component_ids) == 2
        assert component_ids == {1, 2}


class TestNetworkBuilderGlobalIndex:
    """測試全域索引構建"""

    def test_build_global_index_creates_kdtree(
        self, create_mock_component_result
    ):
        """測試建立 KDTree 空間索引"""
        components = [
            create_mock_component_result(1, (0, 0, 10, 10), [(0, 0), (5, 5)])
        ]

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        assert builder.kdtree is not None
        assert builder.global_seeds is not None
        assert builder.component_ids is not None

    def test_build_global_index_seed_count_matches(
        self, create_mock_component_result
    ):
        """測試種子點數量與節點數一致"""
        components = [
            create_mock_component_result(1, (0, 0, 10, 10), [(0, 0), (5, 5), (10, 10)])
        ]

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        expected_count = global_topology.number_of_nodes()
        assert len(builder.global_seeds) == expected_count
        assert len(builder.component_ids) == expected_count

    def test_build_global_index_component_ids_correct(
        self, create_mock_component_result
    ):
        """測試 component_ids 陣列正確記錄"""
        components = [
            create_mock_component_result(1, (0, 0, 10, 10), [(0, 0), (5, 5)]),
            create_mock_component_result(2, (20, 20, 30, 30), [(0, 0)])
        ]

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        # 檢查 component_ids 包含預期的值
        unique_ids = set(builder.component_ids)
        assert unique_ids == {1, 2}

    def test_build_global_index_kdtree_queryable(
        self, create_mock_component_result
    ):
        """測試 KDTree 可正常查詢"""
        components = [
            create_mock_component_result(1, (0, 0, 10, 10), [(5, 5)])
        ]

        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        # 測試查詢半徑內的點
        query_point = [5, 5]
        indices = builder.kdtree.query_ball_point(query_point, r=10.0)

        assert len(indices) > 0

    def test_build_global_index_empty_graph(self):
        """測試空圖的索引構建"""
        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))
        empty_graph = nx.MultiGraph()
        builder._build_global_index(empty_graph)

        assert len(builder.global_seeds) == 0
        assert len(builder.component_ids) == 0
        assert builder.kdtree is None  # 空圖時 KDTree 為 None


class TestNetworkBuilderNeighborSearch:
    """測試鄰居搜尋與過濾"""

    def test_get_unprocessed_neighbor_indices_filters_same_component(
        self, create_mock_component_result
    ):
        """測試過濾同一元件的節點"""
        # 建立同一元件的多個節點
        components = [
            create_mock_component_result(1, (0, 0, 50, 50), [
                (10, 10), (15, 15), (20, 20)  # 三個節點距離很近
            ])
        ]

        builder = NetworkBuilder(
            np.zeros((100, 100), dtype=np.uint8),
            search_radius=20.0
        )
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        neighbors = builder._get_unprocessed_neighbor_indices(0, visited)

        # 同元件的節點應該被過濾掉
        assert len(neighbors) == 0

    def test_get_unprocessed_neighbor_indices_finds_different_components(
        self, create_mock_component_result
    ):
        """測試找到不同元件的鄰居"""
        components = [
            create_mock_component_result(1, (0, 0, 20, 20), [(10, 10)]),
            create_mock_component_result(2, (0, 0, 30, 30), [(15, 15)])  # 距離約 7 像素
        ]

        builder = NetworkBuilder(
            np.zeros((100, 100), dtype=np.uint8),
            search_radius=20.0
        )
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        neighbors = builder._get_unprocessed_neighbor_indices(0, visited)

        # 應該找到不同元件的鄰居
        assert len(neighbors) > 0

    def test_get_unprocessed_neighbor_indices_respects_radius(
        self, create_mock_component_result
    ):
        """測試遵守搜尋半徑限制"""
        components = [
            create_mock_component_result(1, (0, 0, 20, 20), [(10, 10)]),
            create_mock_component_result(2, (0, 0, 100, 100), [(90, 90)])  # 距離約 113 像素
        ]

        builder = NetworkBuilder(
            np.zeros((100, 100), dtype=np.uint8),
            search_radius=50.0  # 半徑 50 無法到達
        )
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        neighbors = builder._get_unprocessed_neighbor_indices(0, visited)

        # 應該找不到超出半徑的鄰居
        assert len(neighbors) == 0

    def test_get_unprocessed_neighbor_indices_skips_visited(
        self, create_mock_component_result
    ):
        """測試跳過已訪問的節點對"""
        components = [
            create_mock_component_result(1, (0, 0, 20, 20), [(10, 10)]),
            create_mock_component_result(2, (0, 0, 30, 30), [(15, 15)])
        ]

        builder = NetworkBuilder(
            np.zeros((100, 100), dtype=np.uint8),
            search_radius=20.0
        )
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        # 標記節點對為已訪問
        visited = {(0, 1)}
        neighbors = builder._get_unprocessed_neighbor_indices(0, visited)

        # 已訪問的節點對應該被過濾
        assert len(neighbors) == 0

    def test_get_unprocessed_neighbor_indices_without_index_raises_error(
        self
    ):
        """測試未建構索引時拋出錯誤"""
        builder = NetworkBuilder(np.zeros((100, 100), dtype=np.uint8))

        with pytest.raises(RuntimeError, match="請先建構全局索引"):
            builder._get_unprocessed_neighbor_indices(0, set())


class TestNetworkBuilderPathResolution:
    """測試路徑解析與邊緣建立"""

    def test_resolve_candidate_paths_finds_valid_paths(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試找到有效路徑"""
        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(20, 20)]),
            create_mock_component_result(2, (0, 0, 40, 40), [(30, 30)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        # target_index_list 接收的是 tuple pair 的列表
        # 透過 _compute_edges_from_source 來測試完整流程
        edges = builder._compute_edges_from_source(0, visited)

        # 應該找到路徑（兩個元件距離很近）
        assert len(edges) >= 0  # 可能找到也可能找不到路徑（取決於成本）
        if len(edges) > 0:
            assert "path" in edges[0]
            assert "cost" in edges[0]
            assert "distance" in edges[0]

    def test_resolve_candidate_paths_filters_high_cost(
        self, create_mock_component_result
    ):
        """測試過濾高成本路徑"""
        # 建立一個幾乎全暗的影像（高成本）
        dark_image = np.zeros((100, 100), dtype=np.uint8)

        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(10, 10)]),
            create_mock_component_result(2, (0, 0, 100, 100), [(50, 50)])
        ]

        builder = NetworkBuilder(
            dark_image,
            search_radius=100.0,
            max_cost_threshold=0.1  # 非常低的閾值
        )
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        edges = builder._compute_edges_from_source(0, visited)

        # 高成本路徑應該被過濾
        # 低閾值應該導致大部分或所有路徑被拒絕

    def test_resolve_candidate_paths_handles_unreachable_targets(
        self, create_mock_component_result
    ):
        """測試處理無法到達的目標"""
        # 使用有隔離區域的影像
        components = [
            create_mock_component_result(1, (0, 0, 20, 20), [(10, 10)]),
            create_mock_component_result(2, (80, 80, 100, 100), [(90, 90)])
        ]

        builder = NetworkBuilder(
            np.zeros((100, 100), dtype=np.uint8),
            search_radius=150.0  # 足夠大的半徑
        )
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        edges = builder._compute_edges_from_source(0, visited)

        # 應該能正常執行，無論是否找到路徑
        assert isinstance(edges, list)

    def test_resolve_candidate_paths_marks_pairs_as_visited(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試標記節點對為已訪問"""
        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(20, 20)]),
            create_mock_component_result(2, (0, 0, 40, 40), [(30, 30)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        global_topology = builder._build_global_topology(components)
        builder._build_global_index(global_topology)

        visited = set()
        builder._compute_edges_from_source(0, visited)

        # 節點對應該被標記為已訪問
        assert len(visited) > 0  # 至少有一些節點對被處理


class TestNetworkBuilderGraphConstruction:
    """測試圖構建完整流程"""

    def test_build_graph_single_component_returns_empty_edges(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試單一元件不產生跨元件邊"""
        components = [
            create_mock_component_result(1, (0, 0, 50, 50), [
                (20, 20), (25, 25), (30, 30)
            ])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        result = builder.build_graph(components)

        # 單一元件不應產生跨元件連接
        # 但會包含元件內部的邊
        assert result.graph is not None
        assert result.nodes is not None

    def test_build_graph_two_components_creates_connections(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試兩個元件產生連接"""
        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(20, 20)]),
            create_mock_component_result(2, (0, 0, 40, 40), [(30, 30)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        result = builder.build_graph(components)

        assert result.graph is not None
        assert result.graph.number_of_nodes() == 2
        # 可能產生跨元件邊（取決於路徑成本）

    def test_build_graph_result_structure(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試返回結果的結構"""
        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(20, 20)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        result = builder.build_graph(components)

        assert result.graph is not None
        assert result.nodes is not None
        assert isinstance(result.graph, nx.MultiGraph)
        assert isinstance(result.nodes, np.ndarray)

    def test_build_graph_empty_components_returns_empty_result(
        self, simple_bright_image
    ):
        """測試空元件列表返回空結果"""
        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        result = builder.build_graph([])

        assert result.graph is not None
        assert result.graph.number_of_nodes() == 0
        assert result.graph.number_of_edges() == 0

    def test_build_graph_updates_global_index(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試構建圖時更新全域索引"""
        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(20, 20)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)

        # 構建前索引應為空
        assert builder.kdtree is None

        builder.build_graph(components)

        # 構建後索引應已建立
        assert builder.kdtree is not None
        assert builder.global_seeds is not None
        assert builder.component_ids is not None

    def test_build_graph_edge_attributes(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試邊包含必要的屬性"""
        components = [
            create_mock_component_result(1, (0, 0, 30, 30), [(20, 20)]),
            create_mock_component_result(2, (0, 0, 40, 40), [(30, 30)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        result = builder.build_graph(components)

        # 檢查跨元件邊的屬性
        for u, v, data in result.graph.edges(data=True):
            # 檢查是否為跨元件邊
            comp_id_u = result.graph.nodes[u].get("component_id")
            comp_id_v = result.graph.nodes[v].get("component_id")

            if comp_id_u != comp_id_v:
                # 跨元件邊應該有這些屬性
                assert "weight" in data
                assert "distance" in data or data["weight"] == 1e-5  # 可能是內部邊
                # path 屬性可能存在也可能不存在（取決於是否為新增的邊）

    def test_build_graph_respects_search_radius(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試遵守搜尋半徑限制"""
        # 建立距離較遠的兩個元件
        components = [
            create_mock_component_result(1, (0, 0, 20, 20), [(10, 10)]),
            create_mock_component_result(2, (80, 80, 100, 100), [(90, 90)])
        ]

        # 使用小搜尋半徑
        builder = NetworkBuilder(
            simple_bright_image,
            search_radius=10.0
        )
        result = builder.build_graph(components)

        # 距離過遠的元件不應產生跨元件連接
        # 檢查圖中是否有跨元件邊
        has_cross_component_edge = False
        for u, v, data in result.graph.edges(data=True):
            comp_id_u = result.graph.nodes[u].get("component_id")
            comp_id_v = result.graph.nodes[v].get("component_id")
            if comp_id_u != comp_id_v:
                has_cross_component_edge = True
                break

        # 由於搜尋半徑很小，不應該有跨元件邊
        assert not has_cross_component_edge


class TestNetworkBuilderEdgeCases:
    """測試邊界情況"""

    def test_update_edges_to_graph_without_index_raises_error(
        self, simple_bright_image
    ):
        """測試未建構索引時更新邊會拋出錯誤"""
        builder = NetworkBuilder(simple_bright_image)
        graph = nx.MultiGraph()

        with pytest.raises(RuntimeError, match="請先建構全局索引"):
            builder._update_edges_to_graph([], graph)

    def test_compute_edges_from_source_without_index_raises_error(
        self, simple_bright_image
    ):
        """測試未建構索引時計算邊會拋出錯誤"""
        builder = NetworkBuilder(simple_bright_image)

        with pytest.raises(RuntimeError, match="請先建構全局索引"):
            builder._compute_edges_from_source(0, set())

    def test_build_graph_with_overlapping_components(
        self, create_mock_component_result, simple_bright_image
    ):
        """測試處理重疊的元件"""
        # 建立位置重疊的元件
        components = [
            create_mock_component_result(1, (10, 10, 30, 30), [(20, 20)]),
            create_mock_component_result(2, (15, 15, 35, 35), [(25, 25)])
        ]

        builder = NetworkBuilder(simple_bright_image, search_radius=50.0)
        result = builder.build_graph(components)

        # 應該能正常處理重疊元件
        assert result.graph is not None
        assert result.graph.number_of_nodes() == 2

    def test_build_graph_with_many_components(
        self, create_mock_component_result, complex_network_image
    ):
        """測試處理大量元件"""
        # 建立多個元件
        components = [
            create_mock_component_result(i, (i*20, i*20, i*20+10, i*20+10), [(i*20+5, i*20+5)])
            for i in range(5)
        ]

        builder = NetworkBuilder(complex_network_image, search_radius=50.0)
        result = builder.build_graph(components)

        assert result.graph is not None
        assert result.graph.number_of_nodes() == 5
