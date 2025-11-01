'''
獨立測試腳本 for CostCalculator (階段 B)
'''
import numpy as np
import sys
from dataclasses import dataclass
import os

# 確保可以從當前目錄匯入模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathfinding import ImagePathfinder
from cost_calculator import CostCalculator

# 為了獨立測試,我們在腳本內定義一個模擬的 Seed 類別
@dataclass
class MockSeed:
    id: int
    x: int
    y: int
    component_id: int = 0
    seed_type: str = 'test'
    curvature_degrees: float = 0.0
    path_id: int = 0


def create_test_image():
    '''
    建立一個 20x20 的合成測試影像 (模擬 green_channel)
    
    - 影像中間有一條高亮度 (低成本) 的水平通道
    - 通道上下是低亮度 (高成本) 的障礙區
    - 右下角有一個被完全隔離的區域
    '''
    # 影像亮度越高, A* 成本越低 (cost = 255 - brightness)
    image = np.full((20, 20), 50, dtype=np.uint8)  # 背景: 高成本

    # 中間的低成本通道 (亮度 250)
    image[9:12, :] = 250

    # 右下角的隔離區 (用成本無限大的牆壁圍起來)
    image[15:, 15:] = 250 # 隔離區內部
    image[14, 14:] = 0   # 上方牆壁 (成本 255)
    image[15:, 14] = 0   # 左方牆壁 (成本 255)

    return image


def run_tests():
    '''執行所有測試案例'''
    print("="*50)
    print("🚀 開始測試 CostCalculator (階段 B)")
    print("="*50)

    # 1. 建立測試環境
    green_channel = create_test_image()
    pathfinder = ImagePathfinder(green_channel, verbose=False)
    cost_calculator = CostCalculator(pathfinder, verbose=False)

    print("✓ 測試環境建立成功 (合成影像、Pathfinder、CostCalculator)")
    print(f"  - 影像尺寸: {green_channel.shape}")
    print(f"  - 成本權重 (α, β, γ): ({cost_calculator.alpha}, {cost_calculator.beta}, {cost_calculator.gamma})")
    print("-"*50)

    # 2. 定義測試種子
    # 案例 1: 在低成本通道內
    seed_A = MockSeed(id=1, x=2, y=10)
    seed_B = MockSeed(id=2, x=18, y=10)

    # 案例 2: 需穿越高成本區域
    seed_C = MockSeed(id=3, x=5, y=5)
    seed_D = MockSeed(id=4, x=15, y=5)

    # 案例 3: 一個在隔離區內,一個在外面
    seed_E = MockSeed(id=5, x=17, y=17) # 隔離區內
    seed_F = MockSeed(id=6, x=5, y=17)  # 隔離區外

    # --- 執行測試案例 ---

    # 案例 1: Happy Path
    print("▶️  案例 1: 低成本路徑 (Happy Path)")
    costs_1 = cost_calculator.calculate_total_cost(seed_A, seed_B)
    print(f"  - 種子 A: ({seed_A.x}, {seed_A.y}), 種子 B: ({seed_B.x}, {seed_B.y})")
    print(f"  - 總成本: {costs_1['total_cost']:.2f}")
    print(f"  - 幾何成本: {costs_1['geometric_cost']:.2f}")
    print(f"  - 影像成本: {costs_1['image_cost']:.2f} (預期較低)")
    print(f"  - 曲率成本: {costs_1['curvature_cost']:.2f}")
    print(f"  - 彎曲度: {costs_1['tortuosity']:.2f} (預期接近 1.0)")
    assert costs_1['total_cost'] < 100, "案例1成本應較低"
    print("  ✅ 驗證通過: 成本合理")
    print("-"*50)

    # 案例 2: 高成本路徑
    print("▶️  案例 2: 高成本路徑 (穿過障礙)")
    costs_2 = cost_calculator.calculate_total_cost(seed_C, seed_D)
    print(f"  - 種子 C: ({seed_C.x}, {seed_C.y}), 種子 D: ({seed_D.x}, {seed_D.y})")
    print(f"  - 總成本: {costs_2['total_cost']:.2f}")
    print(f"  - 幾何成本: {costs_2['geometric_cost']:.2f}")
    print(f"  - 影像成本: {costs_2['image_cost']:.2f} (預期較高)")
    print(f"  - 曲率成本: {costs_2['curvature_cost']:.2f}")
    print(f"  - 彎曲度: {costs_2['tortuosity']:.2f}")
    assert costs_2['image_cost'] > costs_1['image_cost'], "案例2影像成本應較高"
    print("  ✅ 驗證通過: 影像成本顯著高於案例1")
    print("-"*50)

    # 案例 3: 無路徑
    print("▶️  案例 3: 無法到達的路徑 (隔離區)")
    costs_3 = cost_calculator.calculate_total_cost(seed_E, seed_F)
    print(f"  - 種子 E: ({seed_E.x}, {seed_E.y}), 種子 F: ({seed_F.x}, {seed_F.y})")
    print(f"  - 總成本: {costs_3['total_cost']}")
    print(f"  - 影像成本: {costs_3['image_cost']}")
    assert costs_3['total_cost'] == float('inf'), "案例3應回傳無限大成本"
    print("  ✅ 驗證通過: 總成本為無限大")
    print("-"*50)

    print("🎉 所有測試案例執行完畢!")
    print("="*50)


if __name__ == "__main__":
    run_tests()