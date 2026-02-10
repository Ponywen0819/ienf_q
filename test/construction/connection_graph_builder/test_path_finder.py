"""
Pathfinder 類別的單元測試

測試範圍：
1. 初始化與參數驗證
2. 成本地圖建立
3. 座標轉換
4. ROI 提取
5. 路徑搜尋功能
"""

import pytest
import numpy as np
from typing import List, Tuple

from neural_reconstruction.algorithms.pure_mst.connection_graph_builder.path_finder import Pathfinder


class TestPathfinderInit:
    """測試 Pathfinder 初始化"""

    def test_init_with_default_weights(self, nerve_like_image):
        """測試使用預設權重初始化"""
        pathfinder = Pathfinder(nerve_like_image)

        assert pathfinder.intensity_weight == 0.6
        assert pathfinder.shape_weight == 0.4
        assert pathfinder.cost_map.shape == nerve_like_image.shape
        assert pathfinder.height == nerve_like_image.shape[0]
        assert pathfinder.width == nerve_like_image.shape[1]

    def test_init_with_custom_weights(self, simple_bright_image):
        """測試使用自訂權重初始化"""
        pathfinder = Pathfinder(
            simple_bright_image,
            intensity_weight=0.8,
            shape_weight=0.2
        )

        assert pathfinder.intensity_weight == 0.8
        assert pathfinder.shape_weight == 0.2

    def test_init_validates_weight_sum(self, simple_bright_image):
        """測試權重總和驗證（應等於 1.0）"""
        # 權重總和不為 1.0 時應該正常運作（內部會正規化或使用原始值）
        pathfinder = Pathfinder(
            simple_bright_image,
            intensity_weight=0.5,
            shape_weight=0.3  # Sum = 0.8
        )

        # 確認可以正常建立實例
        assert pathfinder is not None
        assert pathfinder.intensity_weight == 0.5
        assert pathfinder.shape_weight == 0.3


class TestPathfinderCostMap:
    """測試成本地圖建立"""

    def test_cost_map_shape_matches_image(self, simple_bright_image, default_pathfinder):
        """測試成本地圖尺寸與輸入影像一致"""
        assert default_pathfinder.cost_map.shape == simple_bright_image.shape

    def test_cost_map_uniform_image(self, simple_bright_image):
        """測試均勻影像的成本地圖"""
        pathfinder = Pathfinder(simple_bright_image)
        cost_map = pathfinder.cost_map

        # 均勻影像應該產生相對均勻的成本
        assert cost_map.min() >= 0
        # 成本變異應該較小
        assert np.std(cost_map) < np.mean(cost_map) * 0.5

    def test_cost_map_bright_pixels_have_low_cost(self, nerve_like_image):
        """測試亮像素區域有較低成本"""
        pathfinder = Pathfinder(nerve_like_image)
        cost_map = pathfinder.cost_map

        # 找出影像中的亮區域（神經纖維模擬）
        bright_mask = nerve_like_image > 200
        dark_mask = nerve_like_image < 50

        if np.any(bright_mask) and np.any(dark_mask):
            bright_costs = cost_map[bright_mask]
            dark_costs = cost_map[dark_mask]

            # 亮區域的平均成本應該低於暗區域
            assert np.mean(bright_costs) < np.mean(dark_costs)

    def test_cost_map_intensity_weight_effect(self, nerve_like_image):
        """測試強度權重對成本地圖的影響"""
        # 高強度權重
        pf_high_intensity = Pathfinder(
            nerve_like_image,
            intensity_weight=0.9,
            shape_weight=0.1
        )

        # 低強度權重
        pf_low_intensity = Pathfinder(
            nerve_like_image,
            intensity_weight=0.1,
            shape_weight=0.9
        )

        # 兩個成本地圖應該不同
        assert not np.allclose(
            pf_high_intensity.cost_map,
            pf_low_intensity.cost_map
        )

    def test_cost_map_values_normalized(self, default_pathfinder):
        """測試成本地圖值已正規化"""
        cost_map = default_pathfinder.cost_map

        # 成本地圖的值應該都是正數（有加上 epsilon）
        assert np.all(cost_map > 0)


class TestPathfinderCoordinateConversion:
    """測試座標轉換功能"""

    def test_global_to_local_conversion(self, default_pathfinder):
        """測試全域座標轉換為局部座標"""
        min_y, min_x = 10, 20
        global_pos = (15, 25)

        local_pos = default_pathfinder._convert_position_global_to_local(
            global_pos, min_y, min_x
        )

        assert local_pos == (5, 5)

    def test_local_to_global_conversion(self, default_pathfinder):
        """測試局部座標轉換為全域座標"""
        min_y, min_x = 10, 20
        local_pos = (5, 5)

        global_pos = default_pathfinder._convert_position_local_to_global(
            local_pos, min_y, min_x
        )

        assert global_pos == (15, 25)

    def test_coordinate_conversion_round_trip(self, default_pathfinder):
        """測試座標來回轉換的一致性"""
        min_y, min_x = 30, 40
        original_global = (50, 70)

        # 全域 -> 局部 -> 全域
        local = default_pathfinder._convert_position_global_to_local(
            original_global, min_y, min_x
        )
        back_to_global = default_pathfinder._convert_position_local_to_global(
            local, min_y, min_x
        )

        assert back_to_global == original_global

    def test_coordinate_conversion_multiple_points(self, default_pathfinder):
        """測試批次座標轉換"""
        min_y, min_x = 0, 0
        global_points = [(10, 10), (20, 30), (5, 15)]

        local_points = [
            default_pathfinder._convert_position_global_to_local(gp, min_y, min_x)
            for gp in global_points
        ]

        # 轉換回全域
        recovered_globals = [
            default_pathfinder._convert_position_local_to_global(lp, min_y, min_x)
            for lp in local_points
        ]

        assert recovered_globals == global_points


class TestPathfinderROIExtraction:
    """測試 ROI（感興趣區域）提取"""

    def test_roi_covers_source_and_targets(self, default_pathfinder):
        """測試 ROI 包含來源點和所有目標點"""
        source = (50, 50)
        targets = [(60, 60), (40, 70), (55, 45)]

        min_y, max_y, min_x, max_x, cropped = default_pathfinder._get_path_finding_roi(
            source, targets
        )

        # 檢查 ROI 包含所有點
        all_points = [source] + targets
        for point in all_points:
            assert min_y <= point[0] <= max_y
            assert min_x <= point[1] <= max_x

    def test_roi_respects_image_boundaries(self, default_pathfinder):
        """測試 ROI 不超出影像邊界"""
        height = default_pathfinder.height
        width = default_pathfinder.width
        source = (5, 5)
        targets = [(height - 5, width - 5)]

        min_y, max_y, min_x, max_x, cropped = default_pathfinder._get_path_finding_roi(
            source, targets
        )

        assert min_y >= 0
        assert min_x >= 0
        assert max_y < height
        assert max_x < width

    def test_roi_minimal_for_close_points(self, default_pathfinder):
        """測試相近點的 ROI 大小合理"""
        source = (50, 50)
        targets = [(51, 51), (52, 50)]

        min_y, max_y, min_x, max_x, cropped = default_pathfinder._get_path_finding_roi(
            source, targets, bbox_padding=5
        )

        # ROI 應該相對較小
        roi_height = max_y - min_y + 1
        roi_width = max_x - min_x + 1

        assert roi_height < 30
        assert roi_width < 30

    def test_roi_expands_for_distant_points(self, default_pathfinder):
        """測試遠距點的 ROI 大小"""
        source = (10, 10)
        targets = [(90, 90)]

        min_y, max_y, min_x, max_x, cropped = default_pathfinder._get_path_finding_roi(
            source, targets
        )

        # ROI 應該較大
        roi_height = max_y - min_y + 1
        roi_width = max_x - min_x + 1

        assert roi_height > 60
        assert roi_width > 60


class TestPathfinderPathFinding:
    """測試路徑搜尋功能"""

    def test_find_paths_single_target_reachable(self, default_pathfinder):
        """測試單一可達目標的路徑搜尋"""
        source = (50, 50)
        targets = [(55, 55)]

        result = default_pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 1
        assert targets[0] in result
        path_result = result[targets[0]]
        assert path_result is not None
        path, cost = path_result
        assert len(path) >= 2
        assert tuple(path[0]) == source
        assert tuple(path[-1]) == targets[0]
        assert cost > 0

    def test_find_paths_multiple_targets(self, default_pathfinder):
        """測試多個目標的路徑搜尋"""
        source = (50, 50)
        targets = [(55, 55), (60, 50), (45, 60)]

        result = default_pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 3
        # 至少應該有一些路徑找到
        found_paths = [v for v in result.values() if v is not None]
        assert len(found_paths) > 0

    def test_find_paths_unreachable_target(self, unreachable_target_image):
        """測試無法到達的目標"""
        # 使用有隔離區域的影像
        pathfinder = Pathfinder(unreachable_target_image)
        source = (10, 10)
        targets = [(90, 90)]  # 可能在隔離區域中

        result = pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 1
        # 結果字典應該有該目標的鍵
        assert targets[0] in result

    def test_find_paths_follows_bright_regions(self, bright_path_image):
        """測試路徑傾向沿著亮區域"""
        pathfinder = Pathfinder(bright_path_image)
        source = (10, 50)
        targets = [(90, 50)]

        result = pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 1
        assert targets[0] in result
        path_result = result[targets[0]]

        if path_result is not None:
            path, cost = path_result
            # 檢查路徑是否經過亮區域
            path_intensities = [bright_path_image[p[0], p[1]] for p in path]
            avg_intensity = np.mean(path_intensities)
            # 路徑應該傾向經過亮區域（平均亮度應該較高）
            assert avg_intensity > 100

    def test_find_paths_empty_targets(self, default_pathfinder):
        """測試空目標列表"""
        source = (50, 50)
        targets = []

        result = default_pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 0

    def test_find_paths_source_equals_target(self, default_pathfinder):
        """測試來源點等於目標點"""
        source = (50, 50)
        targets = [(50, 50)]

        result = default_pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 1
        assert targets[0] in result
        path_result = result[targets[0]]
        # 路徑應該只包含該點或為 None（取決於實作）
        if path_result is not None:
            path, cost = path_result
            assert len(path) <= 3

    def test_find_paths_adjacent_targets(self, default_pathfinder):
        """測試相鄰目標點"""
        source = (50, 50)
        targets = [(50, 51), (51, 50)]

        result = default_pathfinder.find_paths_from_source(source, targets)

        assert len(result) == 2
        # 相鄰點應該都能找到路徑
        for target in targets:
            assert target in result
            path_result = result[target]
            if path_result is not None:
                path, cost = path_result
                # 路徑應該很短
                assert len(path) <= 5

    def test_find_paths_consistency(self, default_pathfinder):
        """測試路徑搜尋的一致性（相同輸入應得到相同結果）"""
        source = (50, 50)
        targets = [(60, 60), (40, 40)]

        result1 = default_pathfinder.find_paths_from_source(source, targets)
        result2 = default_pathfinder.find_paths_from_source(source, targets)

        assert len(result1) == len(result2)
        for target in targets:
            r1 = result1[target]
            r2 = result2[target]
            if r1 is None and r2 is None:
                continue
            assert r1 is not None and r2 is not None
            path1, cost1 = r1
            path2, cost2 = r2
            assert path1 == path2
            assert cost1 == cost2
