#!/usr/bin/env python3
"""
連通元件分析腳本 (Connected Components Analysis)

從二值標註影像中提取所有獨立的白色區塊（連通元件），
並儲存為標籤影像和 JSON 元數據，供後續骨架化和種子提取使用。

使用方式:
    python connected_components.py -i annotation.png -o output/components -v

作者: Generated with Claude Code
日期: 2025-10-22
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.color import label2rgb


class ConnectedComponentsAnalyzer:
    """連通元件分析器"""

    def __init__(
        self,
        connectivity: int = 8,
        min_area: int = 10,
        verbose: bool = False
    ):
        """
        初始化連通元件分析器

        Args:
            connectivity: 連通性 (4 或 8)
            min_area: 最小元件面積（像素），小於此值的元件會被過濾
            verbose: 是否輸出詳細資訊
        """
        self.connectivity = connectivity
        self.min_area = min_area
        self.verbose = verbose

        # 驗證參數
        if connectivity not in [4, 8]:
            raise ValueError(f"connectivity 必須是 4 或 8，但收到 {connectivity}")

        # scikit-image 的 connectivity 參數: 1 = 4-連通, 2 = 8-連通
        self.skimage_connectivity = 1 if connectivity == 4 else 2

    def load_binary_image(self, image_path: str) -> np.ndarray:
        """
        載入並驗證二值影像

        Args:
            image_path: 影像檔案路徑

        Returns:
            二值影像陣列 (0 或 255)

        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 影像格式不正確
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"影像檔案不存在: {image_path}")

        # 載入為灰階影像
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        # 檢查是否為二值影像
        unique_values = np.unique(image)
        if not (len(unique_values) <= 2 and all(v in [0, 255] for v in unique_values)):
            if self.verbose:
                print(f"警告: 影像不是標準二值影像（只包含 0 和 255）")
                print(f"  唯一像素值: {unique_values}")
                print(f"  將進行二值化處理（閾值 = 127）")
            # 自動二值化
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        if self.verbose:
            print(f"✓ 成功載入二值影像: {image_path}")
            print(f"  影像尺寸: {image.shape[1]}x{image.shape[0]} (寬x高)")
            print(f"  白色像素數: {np.sum(image == 255)} ({np.sum(image == 255) / image.size * 100:.2f}%)")

        return image

    def analyze(self, binary_image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        執行連通元件分析

        Args:
            binary_image: 二值影像 (0 或 255)

        Returns:
            (標籤影像, 元件列表)
            - 標籤影像: 每個像素值代表所屬元件的 ID (0 = 背景)
            - 元件列表: 每個元件的屬性字典
        """
        if self.verbose:
            print(f"\n執行連通元件分析...")
            print(f"  連通性: {self.connectivity}-連通")
            print(f"  最小面積過濾: {self.min_area} 像素")

        # 1. 執行連通元件標記
        # 將二值影像轉為 0/1
        binary = (binary_image > 0).astype(np.uint8)

        # 使用 scikit-image 標記連通元件
        labeled_image = measure.label(binary, connectivity=self.skimage_connectivity)

        initial_num_components = labeled_image.max()

        if self.verbose:
            print(f"\n  初始檢測到 {initial_num_components} 個連通元件")

        # 2. 提取元件屬性
        regions = measure.regionprops(labeled_image)

        # 3. 過濾小元件並重新編號
        valid_components = []
        filtered_count = 0

        for region in regions:
            if region.area >= self.min_area:
                valid_components.append(region)
            else:
                filtered_count += 1

        if self.verbose:
            print(f"  過濾掉 {filtered_count} 個小元件（面積 < {self.min_area}）")
            print(f"  保留 {len(valid_components)} 個有效元件")

        # 4. 創建新的標籤影像（重新編號 1, 2, 3, ...）
        new_labeled_image = np.zeros_like(labeled_image, dtype=np.uint16)
        components_list = []

        for new_id, region in enumerate(valid_components, start=1):
            # 在新標籤影像中標記此元件
            new_labeled_image[labeled_image == region.label] = new_id

            # 提取元件屬性
            component_info = self.extract_component_properties(region, new_id)
            components_list.append(component_info)

        if self.verbose:
            print(f"\n✓ 連通元件分析完成")
            print(f"  最終元件數: {len(components_list)}")
            if components_list:
                total_area = sum(c['area'] for c in components_list)
                avg_area = total_area / len(components_list)
                print(f"  總標註面積: {total_area} 像素")
                print(f"  平均元件面積: {avg_area:.1f} 像素")
                print(f"  最大元件面積: {max(c['area'] for c in components_list)} 像素")
                print(f"  最小元件面積: {min(c['area'] for c in components_list)} 像素")

        return new_labeled_image, components_list

    def extract_component_properties(self, region, component_id: int) -> Dict:
        """
        提取單個元件的屬性

        Args:
            region: scikit-image regionprops 物件
            component_id: 元件 ID

        Returns:
            元件屬性字典
        """
        # 邊界框: (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox

        # 質心: (row, col) -> (y, x)
        centroid_y, centroid_x = region.centroid

        properties = {
            'id': int(component_id),
            'area': int(region.area),
            'bbox': {
                'x_min': int(min_col),
                'y_min': int(min_row),
                'x_max': int(max_col),
                'y_max': int(max_row)
            },
            'centroid': {
                'x': float(centroid_x),
                'y': float(centroid_y)
            },
            'perimeter': float(region.perimeter),
            'equivalent_diameter': float(region.equivalent_diameter),
            'solidity': float(region.solidity)
        }

        return properties

    def save_labeled_image(self, labeled_image: np.ndarray, output_path: str) -> None:
        """
        儲存標籤影像為 16-bit PNG

        Args:
            labeled_image: 標籤影像（元件 ID）
            output_path: 輸出檔案路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 確保為 16-bit 格式（支援 > 255 個元件）
        labeled_16bit = labeled_image.astype(np.uint16)

        success = cv2.imwrite(str(output_path), labeled_16bit)
        if not success:
            raise IOError(f"無法儲存標籤影像: {output_path}")

        if self.verbose:
            print(f"✓ 標籤影像已儲存: {output_path}")
            print(f"  格式: 16-bit PNG")
            print(f"  最大元件 ID: {labeled_image.max()}")

    def save_metadata(
        self,
        components: List[Dict],
        image_shape: Tuple[int, int],
        input_path: str,
        output_path: str
    ) -> None:
        """
        儲存元件元數據為 JSON

        Args:
            components: 元件列表
            image_shape: 影像尺寸 (height, width)
            input_path: 輸入影像路徑
            output_path: 輸出 JSON 路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 計算統計資訊
        total_area = sum(c['area'] for c in components) if components else 0

        metadata = {
            'metadata': {
                'image_shape': list(image_shape),
                'total_components': len(components),
                'timestamp': datetime.now().isoformat(),
                'input_image': str(Path(input_path).absolute()),
                'connectivity': self.connectivity,
                'min_area': self.min_area
            },
            'components': components,
            'statistics': {
                'total_area': total_area,
                'average_area': total_area / len(components) if components else 0,
                'max_area': max((c['area'] for c in components), default=0),
                'min_area': min((c['area'] for c in components), default=0)
            }
        }

        # 儲存為格式化的 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"✓ 元數據已儲存: {output_path}")
            print(f"  格式: JSON")
            print(f"  元件數量: {len(components)}")

    def visualize_components(
        self,
        labeled_image: np.ndarray,
        output_path: str
    ) -> None:
        """
        生成偽彩色視覺化

        Args:
            labeled_image: 標籤影像
            output_path: 輸出圖片路徑
        """
        if self.verbose:
            print(f"\n生成偽彩色視覺化...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use scikit-image's label2rgb to generate pseudo-color
        # bg_label=0 means background is black
        colored = label2rgb(labeled_image, bg_label=0, bg_color=(0, 0, 0))

        # Save as high-resolution image
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(colored)
        ax.axis('off')
        ax.set_title(
            f'Connected Components Visualization ({labeled_image.max()} components)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ 視覺化已儲存: {output_path}")

    def process(
        self,
        input_path: str,
        output_dir: str,
        visualize: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        完整的連通元件分析流程

        Args:
            input_path: 輸入二值標註影像路徑
            output_dir: 輸出目錄
            visualize: 是否生成視覺化圖

        Returns:
            (標籤影像, 元件列表)
        """
        if self.verbose:
            print("=" * 60)
            print("連通元件分析")
            print("=" * 60)

        # 1. 載入二值影像
        binary_image = self.load_binary_image(input_path)

        # 2. 執行連通元件分析
        labeled_image, components = self.analyze(binary_image)

        # 3. 建立輸出目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 儲存標籤影像
        labeled_image_path = output_dir / 'labeled_components.png'
        self.save_labeled_image(labeled_image, str(labeled_image_path))

        # 5. 儲存元數據
        metadata_path = output_dir / 'components.json'
        self.save_metadata(
            components,
            binary_image.shape,
            input_path,
            str(metadata_path)
        )

        # 6. 生成視覺化（可選）
        if visualize:
            viz_path = output_dir / 'visualization.png'
            self.visualize_components(labeled_image, str(viz_path))

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 連通元件分析完成!")
            print("=" * 60)
            print(f"\n輸出檔案:")
            print(f"  - 標籤影像: {labeled_image_path}")
            print(f"  - 元數據: {metadata_path}")
            if visualize:
                print(f"  - 視覺化: {viz_path}")

        return labeled_image, components


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='連通元件分析工具 - 從二值標註影像提取獨立區塊',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用
  python %(prog)s -i annotation.png -o output/components

  # 帶視覺化
  python %(prog)s -i annotation.png -o output/components -v --verbose

  # 調整連通性和過濾參數
  python %(prog)s -i annotation.png --connectivity 4 --min-area 20 -v

  # 完整參數
  python %(prog)s -i data/annotation_binary.png -o output/components --connectivity 8 --min-area 10 -v --verbose

輸出說明:
  output_dir/
  ├── labeled_components.png    # 16-bit 標籤影像（每個元件有唯一 ID）
  ├── components.json           # 元件元數據和統計資訊
  └── visualization.png         # 偽彩色視覺化（--visualize）

標籤影像說明:
  - 像素值 0 = 背景
  - 像素值 1, 2, 3, ... = 各個連通元件的 ID
  - 使用 16-bit PNG 格式，支援最多 65535 個元件

後續使用:
  骨架化腳本可以讀取標籤影像和 JSON，逐個處理元件:

  import cv2
  import json

  labeled = cv2.imread('labeled_components.png', cv2.IMREAD_UNCHANGED)
  with open('components.json') as f:
      data = json.load(f)

  for comp in data['components']:
      comp_id = comp['id']
      mask = (labeled == comp_id).astype(np.uint8) * 255
      # 對 mask 執行骨架化...

參數說明:
  --connectivity:
    4-連通: 只考慮上下左右鄰居
    8-連通: 考慮 8 個方向鄰居（預設，適合神經纖維）

  --min-area:
    過濾面積小於此值的元件（預設 10 像素）
    用於去除雜訊和小瑕疵
        """
    )

    # 必填參數
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='輸入二值標註影像路徑（白色區域 = 神經纖維標註）'
    )

    # 選填參數
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./output/components',
        help='輸出目錄（預設: ./output/components）'
    )

    parser.add_argument(
        '--connectivity',
        type=int,
        choices=[4, 8],
        default=8,
        help='連通性（預設: 8）'
    )

    parser.add_argument(
        '--min-area',
        type=int,
        default=10,
        metavar='PIXELS',
        help='最小元件面積，小於此值會被過濾（預設: 10 像素）'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='生成偽彩色視覺化圖'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='輸出詳細處理資訊'
    )

    return parser.parse_args()


def main():
    """主程式進入點"""
    args = parse_arguments()

    # 參數驗證
    if args.min_area < 1:
        print(f"警告: min_area ({args.min_area}) 過小，可能無法有效過濾雜訊", file=sys.stderr)

    try:
        # 建立連通元件分析器
        analyzer = ConnectedComponentsAnalyzer(
            connectivity=args.connectivity,
            min_area=args.min_area,
            verbose=args.verbose
        )

        # 執行分析
        analyzer.process(
            input_path=args.input,
            output_dir=args.output_dir,
            visualize=args.visualize
        )

        return 0

    except FileNotFoundError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"未預期的錯誤: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
