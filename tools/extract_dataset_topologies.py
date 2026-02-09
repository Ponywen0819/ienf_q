#!/usr/bin/env python3
"""
數據集拓樸提取工具

掃描 data 資料夾，從所有包含 label.png 的樣本中提取 GT 拓樸並儲存為 Pickle 格式。

用法:
    python tools/extract_dataset_topologies.py
    python tools/extract_dataset_topologies.py --data-dir data --output-dir output/topologies
    python tools/extract_dataset_topologies.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.evaluate_dataset import TopologyExtractor
from tools.compare_topologies import TopologyLoader


class DatasetTopologyExtractor:
    """數據集拓樸提取器"""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: 數據集根目錄
            output_dir: 輸出目錄
            verbose: 詳細日志模式
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # 設置日志
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化提取器和加載器
        self.topology_extractor = TopologyExtractor()
        self.topology_loader = TopologyLoader()

    def scan_dataset(self) -> List[Dict[str, Path]]:
        """
        掃描數據集目錄，找到所有包含 label.png 的樣本

        Returns:
            樣本信息列表，每個元素包含:
            - sample_id: 樣本 ID
            - label_path: label.png 路徑
            - image_path: image.png 路徑（如果存在）
            - mask_path: mask.png 路徑（如果存在）
            - annotation_path: annotation.png 路徑（如果存在）
        """
        samples = []

        if not self.data_dir.exists():
            self.logger.error(f"數據集目錄不存在: {self.data_dir}")
            return samples

        # 掃描所有子目錄
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue

            # 跳過特殊目錄
            if subdir.name.startswith(".") or subdir.name == "datasets":
                continue

            # 檢查是否有 label.png
            label_path = subdir / "label.png"
            if not label_path.exists():
                continue

            # 收集所有相關文件
            sample_info = {
                "sample_id": subdir.name,
                "label_path": label_path,
                "image_path": subdir / "image.png" if (subdir / "image.png").exists() else None,
                "mask_path": subdir / "mask.png" if (subdir / "mask.png").exists() else None,
                "annotation_path": subdir / "annotation.png" if (subdir / "annotation.png").exists() else None,
            }

            samples.append(sample_info)
            self.logger.debug(f"找到樣本: {subdir.name}")

        self.logger.info(f"掃描完成，找到 {len(samples)} 個包含 label.png 的樣本")
        return samples

    def extract_topology(self, sample_info: Dict[str, Path]) -> Optional[Dict]:
        """
        從單個樣本提取拓樸

        Args:
            sample_info: 樣本信息字典

        Returns:
            提取結果字典，包含:
            - sample_id: 樣本 ID
            - success: 是否成功
            - output_path: 輸出文件路徑（如果成功）
            - num_nodes: 節點數（如果成功）
            - num_edges: 邊數（如果成功）
            - error: 錯誤信息（如果失敗）
        """
        sample_id = sample_info["sample_id"]
        label_path = sample_info["label_path"]

        result = {
            "sample_id": sample_id,
            "success": False,
        }

        try:
            # 載入 label 圖像
            self.logger.debug(f"[{sample_id}] 載入 label.png...")
            label_img = np.array(Image.open(label_path))

            # 提取拓樸
            self.logger.debug(f"[{sample_id}] 提取拓樸...")
            graph = self.topology_extractor.extract_from_gt(label_img)

            if graph is None or graph.number_of_nodes() == 0:
                raise ValueError("拓樸提取失敗或圖為空")

            # 儲存為 Pickle 格式
            output_filename = f"{sample_id}_gt.pkl"
            output_path = self.output_dir / output_filename

            self.logger.debug(f"[{sample_id}] 儲存拓樸到 {output_path}...")
            self.topology_loader.save(graph, output_path, format="pickle")

            # 記錄成功
            result.update({
                "success": True,
                "output_path": output_path,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            })

            self.logger.info(
                f"✓ [{sample_id}] 成功 - {result['num_nodes']} 節點, "
                f"{result['num_edges']} 邊 -> {output_filename}"
            )

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"✗ [{sample_id}] 失敗: {e}")

        return result

    def extract_all(self) -> Dict:
        """
        提取所有樣本的拓樸

        Returns:
            統計信息字典
        """
        # 掃描數據集
        samples = self.scan_dataset()

        if not samples:
            self.logger.warning("未找到任何樣本")
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "results": [],
            }

        # 提取所有拓樸
        results = []
        success_count = 0
        failed_count = 0

        print("=" * 80)
        print(f"開始提取 {len(samples)} 個樣本的拓樸")
        print("=" * 80)

        for i, sample_info in enumerate(samples, 1):
            sample_id = sample_info["sample_id"]
            print(f"\n進度: [{i}/{len(samples)}] 處理 {sample_id}...")

            result = self.extract_topology(sample_info)
            results.append(result)

            if result["success"]:
                success_count += 1
            else:
                failed_count += 1

        # 彙總統計
        stats = {
            "total": len(samples),
            "success": success_count,
            "failed": failed_count,
            "results": results,
        }

        # 打印摘要
        self._print_summary(stats)

        return stats

    def _print_summary(self, stats: Dict):
        """打印提取摘要"""
        print("\n" + "=" * 80)
        print("提取摘要")
        print("=" * 80)
        print(f"總樣本數: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失敗: {stats['failed']}")
        print(f"成功率: {stats['success'] / stats['total'] * 100:.1f}%")

        if stats['success'] > 0:
            # 統計節點和邊
            successful_results = [r for r in stats['results'] if r['success']]
            total_nodes = sum(r['num_nodes'] for r in successful_results)
            total_edges = sum(r['num_edges'] for r in successful_results)
            avg_nodes = total_nodes / len(successful_results)
            avg_edges = total_edges / len(successful_results)

            print(f"\n拓樸統計:")
            print(f"  平均節點數: {avg_nodes:.1f}")
            print(f"  平均邊數: {avg_edges:.1f}")
            print(f"  總節點數: {total_nodes}")
            print(f"  總邊數: {total_edges}")

        if stats['failed'] > 0:
            print(f"\n失敗樣本:")
            for result in stats['results']:
                if not result['success']:
                    print(f"  - {result['sample_id']}: {result.get('error', 'Unknown error')}")

        print(f"\n輸出目錄: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="數據集拓樸提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默認路徑 (data/ -> output/topologies/)
  python tools/extract_dataset_topologies.py

  # 指定輸入輸出目錄
  python tools/extract_dataset_topologies.py \\
      --data-dir data \\
      --output-dir output/gt_topologies

  # 詳細日志模式
  python tools/extract_dataset_topologies.py --verbose
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="數據集根目錄 (默認: data)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/topologies"),
        help="輸出目錄 (默認: output/topologies)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細日志模式",
    )

    args = parser.parse_args()

    # 創建提取器並運行
    extractor = DatasetTopologyExtractor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    stats = extractor.extract_all()

    # 返回退出碼
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
