"""
測試邊界連接去重修復

驗證修改後的代碼是否正確去除了重複的邊界連接。
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.boundary_crossing.boundary_connector import BoundaryConnector


def main():
    print("=" * 70)
    print("測試邊界連接去重修復")
    print("=" * 70)
    print()

    # Test data paths
    epidermis_mst = "together/epidermis/mst_forest_with_paths.json"
    dermis_mst = "together/dermis/mst_forest_with_paths.json"
    epidermis_mask = "split/S163-2_a_epidermis_virdis.png"
    green_channel = "split/S163-2_a_epidermis_virdis.png"
    output_dir = "test_boundary_fix_output"

    # Create connector with verbose output
    connector = BoundaryConnector(
        boundary_tolerance=10,
        max_crossing_distance=100,
        verbose=True
    )

    # Run the connection process
    try:
        result = connector.connect_layers(
            epidermis_mst=epidermis_mst,
            dermis_mst=dermis_mst,
            epidermis_mask=epidermis_mask,
            green_channel=green_channel,
            output_dir=output_dir
        )

        print("\n" + "=" * 70)
        print("✓ 測試成功！")
        print("=" * 70)
        print()
        print("如果看到以下訊息，表示修復有效：")
        print("  1. '⚠ Warning: Removed X duplicate crossing edges' - 去重生效")
        print("  2. '✓ Validation passed' - 驗證通過")
        print()
        print(f"檢查輸出檔案：{output_dir}/merged_mst_forest.json")

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ 測試失敗！")
        print("=" * 70)
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
