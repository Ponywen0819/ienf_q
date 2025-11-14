#!/usr/bin/env python3
"""
IENF 量化分析 Pipeline - 完整工作流程

從原始影像和標註輸入，自動執行完整流程，輸出神經纖維重建結果。

使用方式:
    # 基本使用
    python run_pipeline.py \\
        --image data/original_image.png \\
        --annotation data/manual_annotation.png \\
        --output output/reconstruction

    # 使用配置文件
    python run_pipeline.py \\
        --image data/original_image.png \\
        --annotation data/manual_annotation.png \\
        --output output/reconstruction \\
        --config config/high_quality.yaml

    # 保存中間產物（用於除錯）
    python run_pipeline.py \\
        --image data/original_image.png \\
        --annotation data/manual_annotation.png \\
        --output output/reconstruction \\
        --save-intermediates
"""

import argparse
import sys
import logging
import shutil
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import cv2
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration loader
from src.config_loader import load_config, IENFConfig


def _import_module_from_path(module_name: str, file_path: Path):
    """動態導入模組（支援數字開頭的目錄名）"""
    # 將模組所在目錄添加到 sys.path 最前面，以支援相對導入
    # 並確保優先於其他已添加的目錄
    module_dir = str(file_path.parent)

    # 如果已經在 sys.path 中，先移除
    if module_dir in sys.path:
        sys.path.remove(module_dir)

    # 添加到最前面
    sys.path.insert(0, module_dir)

    # 清除可能已緩存的相關模組（避免模組名稱衝突）
    modules_to_clear = []
    for key in sys.modules:
        # 不清除主模組本身和標準庫模組
        if key == module_name or key.startswith('_') or '.' not in key:
            continue
        # 清除可能衝突的模組（如 visualization, mst_builder 等）
        if key in ['visualization', 'mst_builder', 'seed_loader', 'cost_calculator',
                   'density_estimator', 'pathfinding', 'seed_pairing', 'graph_builder']:
            modules_to_clear.append(key)

    for key in modules_to_clear:
        del sys.modules[key]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class IENFPipeline:
    """IENF 神經纖維重建完整流程"""

    def __init__(self, config: IENFConfig, save_intermediates: bool = False):
        """
        初始化 pipeline

        Args:
            config: 完整配置
            save_intermediates: 是否保存中間產物
        """
        self.config = config
        self.save_intermediates = save_intermediates
        self.logger = self._setup_logging()
        self.temp_dir = None

    def _setup_logging(self) -> logging.Logger:
        """設定日誌系統"""
        logger = logging.getLogger('IENF_Pipeline')
        logger.setLevel(getattr(logging, self.config.pipeline.logging.level))
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        if self.config.pipeline.logging.log_to_file:
            log_file = Path(self.config.pipeline.logging.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

        return logger

    def _get_working_dir(self, output_dir: str) -> Path:
        """獲取工作目錄"""
        if self.save_intermediates:
            working_dir = Path(output_dir).parent / "intermediates"
            working_dir.mkdir(parents=True, exist_ok=True)
            return working_dir
        else:
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp(prefix="ienf_pipeline_")
            return Path(self.temp_dir)

    def _cleanup_temp_dir(self):
        """清理臨時目錄"""
        if self.temp_dir and not self.save_intermediates:
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"已清理臨時目錄: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"清理臨時目錄失敗: {e}")

    def _preprocess_image(
        self,
        image_path: str,
        working_dir: Path
    ) -> str:
        """
        前處理：提取綠色通道

        Args:
            image_path: 原始影像路徑
            working_dir: 工作目錄

        Returns:
            綠色通道影像路徑
        """
        self.logger.info("步驟 1/6: 影像前處理（提取綠色通道）")
        self.logger.info("-" * 70)

        # 讀取影像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        # 提取綠色通道
        if len(image.shape) == 3:
            green_channel = image[:, :, 1]  # OpenCV uses BGR
            self.logger.info(f"✓ 提取綠色通道: {green_channel.shape}")
        else:
            green_channel = image
            self.logger.info(f"✓ 使用灰階影像: {green_channel.shape}")

        # 儲存綠色通道
        green_path = working_dir / "green_channel.png"
        cv2.imwrite(str(green_path), green_channel)
        self.logger.info("")

        return str(green_path)

    def _process_annotation(
        self,
        annotation_path: str,
        working_dir: Path
    ) -> Path:
        """
        處理標註：連通元件分析

        Args:
            annotation_path: 標註影像路徑
            working_dir: 工作目錄

        Returns:
            連通元件目錄路徑
        """
        self.logger.info("步驟 2/6: 標註處理（連通元件分析）")
        self.logger.info("-" * 70)

        # 讀取標註
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            raise ValueError(f"無法讀取標註: {annotation_path}")

        # 二值化
        _, binary = cv2.threshold(annotation, 127, 255, cv2.THRESH_BINARY)

        # 連通元件分析
        from skimage import measure
        labeled_image = measure.label(binary > 0, connectivity=2)
        props = measure.regionprops(labeled_image)

        self.logger.info(f"✓ 找到 {len(props)} 個連通元件")

        # 儲存連通元件
        components_dir = working_dir / "components"
        components_dir.mkdir(exist_ok=True)

        # 儲存標籤影像
        labeled_path = components_dir / "labeled_components.png"
        cv2.imwrite(str(labeled_path), labeled_image.astype(np.uint16))

        # 儲存元件資訊
        import json
        components_info = {
            'metadata': {
                'total_components': len(props),
                'timestamp': datetime.now().isoformat(),
                'source': str(annotation_path)
            },
            'components': []
        }

        for i, prop in enumerate(props, start=1):
            component = {
                'id': i,
                'area': int(prop.area),
                'centroid': {
                    'y': float(prop.centroid[0]),
                    'x': float(prop.centroid[1])
                },
                'bbox': {
                    'min_row': int(prop.bbox[0]),
                    'min_col': int(prop.bbox[1]),
                    'max_row': int(prop.bbox[2]),
                    'max_col': int(prop.bbox[3])
                }
            }
            components_info['components'].append(component)

        json_path = components_dir / "components.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(components_info, f, indent=2, ensure_ascii=False)

        self.logger.info("")
        return components_dir

    def _skeletonize(
        self,
        components_dir: Path,
        working_dir: Path
    ) -> Path:
        """
        骨架化處理

        Args:
            components_dir: 連通元件目錄
            working_dir: 工作目錄

        Returns:
            骨架目錄路徑
        """
        self.logger.info("步驟 3/6: 骨架化處理")
        self.logger.info("-" * 70)

        # 動態導入 skeletonization 模組
        skeleton_module = _import_module_from_path(
            "skeletonization",
            project_root / "src" / "02_seed_extraction" / "skeletonization.py"
        )
        SkeletonAnalyzer = skeleton_module.SkeletonAnalyzer

        skeletons_dir = working_dir / "skeletons"
        skeletons_dir.mkdir(exist_ok=True)

        analyzer = SkeletonAnalyzer(
            method='zhang-suen',
            spur_threshold=3,
            verbose=False
        )

        # 執行骨架化
        analyzer.process(
            input_dir=str(components_dir),
            output_dir=str(skeletons_dir),
            visualize_skeleton=False,
            visualize_keypoints=False,
            visualize_overlay=False
        )

        self.logger.info(f"✓ 骨架化完成")
        self.logger.info("")
        return skeletons_dir

    def _extract_seeds(
        self,
        skeletons_dir: Path,
        working_dir: Path
    ) -> Path:
        """
        種子提取

        Args:
            skeletons_dir: 骨架目錄
            working_dir: 工作目錄

        Returns:
            種子目錄路徑
        """
        self.logger.info("步驟 4/6: 種子提取")
        self.logger.info("-" * 70)

        # 動態導入 seed_extraction 模組
        seed_module = _import_module_from_path(
            "seed_extraction",
            project_root / "src" / "02_seed_extraction" / "seed_extraction.py"
        )
        SeedExtractionPipeline = seed_module.SeedExtractionPipeline

        seeds_dir = working_dir / "seeds"

        pipeline = SeedExtractionPipeline(
            window_size=self.config.seed_extraction.window_size,
            base_segment_length=self.config.seed_extraction.base_segment_length,
            max_segment_length=self.config.seed_extraction.max_segment_length,
            curvature_threshold=self.config.seed_extraction.curvature_threshold,
            skip_branchpoint_range=self.config.seed_extraction.skip_branchpoint_range,
            min_path_points=self.config.seed_extraction.min_path_points,
            verbose=False
        )

        all_seeds, _ = pipeline.process(
            input_dir=str(skeletons_dir),
            output_dir=str(seeds_dir),
            visualize_seeds=False,
            visualize_overlay=False,
            visualize_curvature=False
        )

        self.logger.info(f"✓ 提取 {len(all_seeds)} 個種子點")
        self.logger.info("")
        return seeds_dir

    def _build_network(
        self,
        seeds_dir: Path,
        green_channel_path: str,
        working_dir: Path
    ) -> Path:
        """
        網路建構

        Args:
            seeds_dir: 種子目錄
            green_channel_path: 綠色通道影像路徑
            working_dir: 工作目錄

        Returns:
            網路目錄路徑
        """
        self.logger.info("步驟 5/6: 網路建構")
        self.logger.info("-" * 70)

        # 動態導入 network_building 模組
        network_module = _import_module_from_path(
            "network_builder",
            project_root / "src" / "03_network_building" / "network_builder.py"
        )
        NetworkBuilder = network_module.NetworkBuilder
        NetworkConfig = network_module.NetworkConfig

        network_dir = working_dir / "network"
        seeds_json = seeds_dir / "seeds.json"

        network_config = NetworkConfig(
            k_neighbors=self.config.network_building.network.k_neighbors,
            max_edge_cost=self.config.network_building.network.max_edge_cost,
            verbose=False,
            # Cost weights
            alpha=self.config.network_building.cost_weights.alpha,
            beta=self.config.network_building.cost_weights.beta,
            gamma=self.config.network_building.cost_weights.gamma,
            # Density parameters
            dense_threshold=self.config.network_building.density.dense_threshold,
            moderate_threshold=self.config.network_building.density.moderate_threshold,
            dense_radius=self.config.network_building.density.dense_radius,
            moderate_radius=self.config.network_building.density.moderate_radius,
            sparse_radius=self.config.network_building.density.sparse_radius,
            # Pathfinding parameters
            max_distance_multiplier=self.config.network_building.pathfinding.max_distance_multiplier,
            distance_from_start_cutoff=self.config.network_building.pathfinding.distance_from_start_cutoff
        )

        builder = NetworkBuilder(network_config)

        G = builder.build_network(
            seeds_json=str(seeds_json),
            green_channel_image=green_channel_path,
            output_dir=str(network_dir)
        )

        self.logger.info(f"✓ 建構網路: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
        self.logger.info("")
        return network_dir

    def _reconstruct(
        self,
        network_dir: Path,
        seeds_dir: Path,
        green_channel_path: str,
        output_dir: str,
        mask_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        神經重建

        Args:
            network_dir: 網路目錄
            seeds_dir: 種子目錄
            green_channel_path: 綠色通道影像路徑
            output_dir: 輸出目錄
            mask_path: 表皮 mask 路徑

        Returns:
            重建結果
        """
        self.logger.info("步驟 6/6: 神經重建")
        self.logger.info("-" * 70)

        # 動態導入 reconstruction 模組
        recon_module = _import_module_from_path(
            "reconstruction_runner",
            project_root / "src" / "04_nueral_reconstruction" / "reconstruction_runner.py"
        )
        ReconstructionRunner = recon_module.ReconstructionRunner
        ReconstructionConfig = recon_module.ReconstructionConfig

        network_graphml = network_dir / "network.graphml"
        seeds_json = seeds_dir / "seeds.json"

        recon_config = ReconstructionConfig(
            max_edge_cost=self.config.reconstruction.max_edge_cost,
            min_branch_angle=self.config.reconstruction.min_branch_angle,
            min_quality_threshold=self.config.reconstruction.min_quality_threshold,
            verbose=False
        )

        runner = ReconstructionRunner(recon_config)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = runner.run(
            graph_path=str(network_graphml),
            seeds_path=str(seeds_json),
            green_channel_path=green_channel_path,
            output_dir=str(output_path),
            mask_path=mask_path
        )

        self.logger.info(f"✓ 重建完成")
        self.logger.info(f"  - MST 森林: {results['stats']['total_nodes']} 節點, "
                       f"{results['stats']['total_edges']} 邊")
        self.logger.info(f"  - 連通分量: {results['stats']['num_components']} 個")
        self.logger.info("")

        return results

    def run(
        self,
        image_path: str,
        annotation_path: str,
        output_dir: str,
        mask_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        執行完整的 IENF 量化流程

        Args:
            image_path: 原始影像路徑（RGB 或灰階）
            annotation_path: 手動標註影像路徑（二值化）
            output_dir: 最終輸出目錄（只包含重建結果）
            mask_path: 表皮 mask 路徑（可選，用於視覺化）

        Returns:
            包含統計資訊的結果字典
        """
        start_time = datetime.now()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("IENF 神經纖維重建 Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"原始影像: {image_path}")
        self.logger.info(f"手動標註: {annotation_path}")
        self.logger.info(f"輸出目錄: {output_dir}")

        if self.save_intermediates:
            self.logger.info(f"中間產物: 將保存")
        else:
            self.logger.info(f"中間產物: 不保存（使用臨時目錄）")

        self.logger.info("")

        try:
            # 驗證輸入檔案
            if not Path(image_path).exists():
                raise FileNotFoundError(f"影像檔案不存在: {image_path}")
            if not Path(annotation_path).exists():
                raise FileNotFoundError(f"標註檔案不存在: {annotation_path}")

            # 獲取工作目錄
            working_dir = self._get_working_dir(output_dir)

            # 執行完整流程
            green_channel_path = self._preprocess_image(image_path, working_dir)
            components_dir = self._process_annotation(annotation_path, working_dir)
            skeletons_dir = self._skeletonize(components_dir, working_dir)
            seeds_dir = self._extract_seeds(skeletons_dir, working_dir)
            network_dir = self._build_network(seeds_dir, green_channel_path, working_dir)
            results = self._reconstruct(
                network_dir, seeds_dir, green_channel_path, output_dir, mask_path
            )

            # 清理或保留中間產物
            if not self.save_intermediates:
                self._cleanup_temp_dir()
                self.logger.info("✓ 已清理中間產物")
            else:
                self.logger.info(f"✓ 中間產物已保存至: {working_dir}")

            # 完成
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("\n" + "=" * 70)
            self.logger.info("Pipeline 執行完成")
            self.logger.info("=" * 70)
            self.logger.info(f"總耗時: {duration}")
            self.logger.info(f"輸出目錄: {output_dir}")
            self.logger.info("=" * 70 + "\n")

            return {
                'success': True,
                'duration': str(duration),
                'output_dir': str(output_dir),
                'statistics': results['stats'],
                'intermediates_saved': self.save_intermediates,
                'intermediates_dir': str(working_dir) if self.save_intermediates else None
            }

        except Exception as e:
            self.logger.error(f"\n✗ Pipeline 執行失敗: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_temp_dir()
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description='IENF 神經纖維重建完整流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本使用
  python run_pipeline.py \\
      --image data/original_image.png \\
      --annotation data/manual_annotation.png \\
      --output output/reconstruction

  # 使用高品質配置
  python run_pipeline.py \\
      --image data/original_image.png \\
      --annotation data/manual_annotation.png \\
      --output output/reconstruction \\
      --config config/high_quality.yaml

  # 保存中間產物（用於除錯）
  python run_pipeline.py \\
      --image data/original_image.png \\
      --annotation data/manual_annotation.png \\
      --output output/reconstruction \\
      --save-intermediates

  # 覆蓋配置參數
  python run_pipeline.py \\
      --image data/original_image.png \\
      --annotation data/manual_annotation.png \\
      --output output/reconstruction \\
      --config config/default.yaml \\
      --curvature-threshold 28

說明:
  - 輸入：原始影像（RGB/灰階）+ 手動標註影像（二值化）
  - 自動執行：前處理 → 連通元件 → 骨架化 → 種子提取 → 網路建構 → 重建
  - 輸出：只包含最終的重建結果（MST 森林、視覺化）
  - 中間產物：預設不保存，使用臨時目錄自動清理
        """
    )

    # 必要參數
    required = parser.add_argument_group('必要參數')
    required.add_argument(
        '--image',
        required=True,
        metavar='PATH',
        help='原始影像路徑（RGB 或灰階）'
    )

    required.add_argument(
        '--annotation',
        required=True,
        metavar='PATH',
        help='手動標註影像路徑（二值化，神經纖維為白色）'
    )

    required.add_argument(
        '--output',
        required=True,
        metavar='DIR',
        help='輸出目錄（只包含最終重建結果）'
    )

    # 可選參數
    optional = parser.add_argument_group('可選參數')
    optional.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        metavar='PATH',
        help='YAML 配置文件路徑（預設: config/default.yaml）'
    )

    optional.add_argument(
        '--mask',
        type=str,
        default=None,
        metavar='PATH',
        help='表皮 mask 路徑（可選，用於視覺化）'
    )

    optional.add_argument(
        '--save-intermediates',
        action='store_true',
        help='保存中間產物（前處理、骨架、種子、網路）到 intermediates/ 目錄'
    )

    # 參數覆蓋
    override = parser.add_argument_group('參數覆蓋（覆蓋配置文件）')
    override.add_argument(
        '--curvature-threshold',
        type=float,
        default=None,
        metavar='DEGREES',
        help='曲率閾值（度數）'
    )

    override.add_argument(
        '--max-edge-cost',
        type=float,
        default=None,
        metavar='FLOAT',
        help='最大邊成本閾值'
    )

    override.add_argument(
        '--alpha',
        type=float,
        default=None,
        metavar='FLOAT',
        help='幾何成本權重'
    )

    override.add_argument(
        '--beta',
        type=float,
        default=None,
        metavar='FLOAT',
        help='影像成本權重'
    )

    override.add_argument(
        '--gamma',
        type=float,
        default=None,
        metavar='FLOAT',
        help='曲率成本權重'
    )

    args = parser.parse_args()

    # 載入配置
    try:
        config = load_config(args.config)
        print(f"✓ 載入配置: {args.config}")
    except Exception as e:
        print(f"✗ 載入配置失敗: {e}", file=sys.stderr)
        return 1

    # 應用 CLI 覆蓋
    if args.curvature_threshold is not None:
        config.seed_extraction.curvature_threshold = args.curvature_threshold
        print(f"  覆蓋參數: curvature_threshold = {args.curvature_threshold}")

    if args.max_edge_cost is not None:
        config.network_building.network.max_edge_cost = args.max_edge_cost
        config.reconstruction.max_edge_cost = args.max_edge_cost
        print(f"  覆蓋參數: max_edge_cost = {args.max_edge_cost}")

    if args.alpha is not None:
        config.network_building.cost_weights.alpha = args.alpha
        print(f"  覆蓋參數: alpha = {args.alpha}")

    if args.beta is not None:
        config.network_building.cost_weights.beta = args.beta
        print(f"  覆蓋參數: beta = {args.beta}")

    if args.gamma is not None:
        config.network_building.cost_weights.gamma = args.gamma
        print(f"  覆蓋參數: gamma = {args.gamma}")

    # 執行 pipeline
    try:
        save_intermediates = args.save_intermediates or config.pipeline.intermediates.save

        pipeline = IENFPipeline(
            config=config,
            save_intermediates=save_intermediates
        )

        results = pipeline.run(
            image_path=args.image,
            annotation_path=args.annotation,
            output_dir=args.output,
            mask_path=args.mask
        )

        if results['success']:
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n\n✗ 使用者中斷執行")
        return 130
    except Exception as e:
        print(f"\n✗ 執行失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
