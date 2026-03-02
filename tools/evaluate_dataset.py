"""
資料集評測腳本 (Dataset Evaluation Script)

對整個資料集進行批次處理與評估：
1. 驗證資料集完整性
2. 執行神經重建 Pipeline
3. 計算 Hausdorff 距離（預留 GT 比較）
4. 生成統計報告

使用範例:
    python tools/evaluate_dataset.py \
        --data-dir data/ \
        --output-dir output/evaluation

作者: Claude Code
日期: 2026-01-15
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import csv

import numpy as np
import networkx as nx
from PIL import Image
from tqdm import tqdm

# 添加專案根目錄到 Python 路徑

from neural_reconstruction.algorithms.pure_mst.linker import PureMstLinker
from neural_reconstruction.algorithms.fragment_linking.linker import HierarchicalFragmentLinker
from neural_reconstruction.core.topology import TopologyBuilder

# 使用新的評估模組
from neural_reconstruction.core.evaluation import (
    TopologyComparator,
    extract_graph_points,
)


# ============================================================================
# 資料結構定義
# ============================================================================


@dataclass
class SampleFiles:
    """樣本檔案路徑"""

    sample_id: str
    image_path: Path
    mask_path: Path
    annotation_path: Path
    label_path: Optional[Path] = None  # GT，目前預留

    def is_complete(self) -> Tuple[bool, str]:
        """
        檢查必要檔案是否完整

        Returns:
            (is_complete, missing_reason)
        """
        if not self.image_path.exists():
            return False, "missing_image"
        if not self.mask_path.exists():
            return False, "missing_mask"
        if not self.annotation_path.exists():
            return False, "missing_annotation"
        # label_path 目前是可選的
        return True, ""


@dataclass
class SampleResult:
    """單一樣本的評測結果"""

    sample_id: str
    status: str  # success, skipped, failed
    hausdorff_distance: Optional[float] = None
    num_nodes_pred: Optional[int] = None
    num_nodes_gt: Optional[int] = None
    num_edges_pred: Optional[int] = None
    num_edges_gt: Optional[int] = None
    num_components_pred: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class EvaluationSummary:
    """評測統計摘要"""

    total_samples: int
    processed: int
    skipped: int
    failed: int
    hausdorff_mean: Optional[float] = None
    hausdorff_median: Optional[float] = None
    hausdorff_std: Optional[float] = None
    hausdorff_min: Optional[float] = None
    hausdorff_max: Optional[float] = None


# ============================================================================
# 資料集載入器
# ============================================================================


class DatasetLoader:
    """
    資料集載入器

    掃描資料集目錄，驗證檔案完整性，返回樣本列表。
    """

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: 資料集根目錄
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def load_samples(self, sample_ids: Optional[List[str]] = None) -> List[SampleFiles]:
        """
        載入資料集樣本

        Args:
            sample_ids: 指定要載入的樣本 ID，None 則載入全部

        Returns:
            樣本檔案列表
        """
        self.logger.info(f"掃描資料集目錄: {self.data_dir}")

        # 獲取所有樣本目錄
        if sample_ids:
            sample_dirs = [
                self.data_dir / sid
                for sid in sample_ids
                if (self.data_dir / sid).is_dir()
            ]
        else:
            sample_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        self.logger.info(f"找到 {len(sample_dirs)} 個樣本目錄")

        samples = []
        for sample_dir in sorted(sample_dirs):
            sample_id = sample_dir.name

            # 檢查 GT 檔案（支援 label.png 和 lable.png 兩種拼法）
            label_path = None
            if (sample_dir / "label.png").exists():
                label_path = sample_dir / "label.png"
            elif (sample_dir / "lable.png").exists():
                label_path = sample_dir / "lable.png"

            sample_files = SampleFiles(
                sample_id=sample_id,
                image_path=sample_dir / "image.png",
                mask_path=sample_dir / "mask.png",
                annotation_path=sample_dir / "annotation.png",
                label_path=label_path,
            )
            samples.append(sample_files)

        return samples


# ============================================================================
# 拓樸萃取器
# ============================================================================


class TopologyExtractor:
    """
    拓樸萃取器

    統一處理 GT 和 Pipeline 結果的拓樸建構。
    """

    def __init__(self, linker: Any):
        """
        Args:
            linker: 實作 run(image, mask, annotation) -> nx.Graph 的 linker 實例
        """
        self.linker = linker
        self.gt_builder = TopologyBuilder()
        self.logger = logging.getLogger(__name__)

    def extract_from_pipeline(
        self, image: np.ndarray, mask: np.ndarray, annotation: np.ndarray
    ) -> Optional[nx.Graph]:
        """
        從 Pipeline 萃取拓樸

        Args:
            image: 原始影像
            mask: 遮罩影像
            annotation: 標註影像

        Returns:
            NetworkX Graph，失敗則返回 None
        """
        try:
            return self.linker.run(image, mask, annotation)
        except Exception as e:
            self.logger.error(f"Pipeline 執行失敗: {e}")
            return None

    def extract_from_gt(self, gt_label: np.ndarray) -> Optional[nx.Graph]:
        """
        從 GT 標註萃取拓樸

        直接對整張 GT 影像呼叫 build_seed_graph，無需逐連通分量處理。

        Args:
            gt_label: GT 標註影像（二值，0 或 255）

        Returns:
            NetworkX Graph，失敗則返回 None
        """
        try:
            graph = self.gt_builder.build_seed_graph(gt_label)
            self.logger.debug(
                f"GT 拓樸建構完成: {graph.number_of_nodes()} 節點, "
                f"{graph.number_of_edges()} 邊"
            )
            return graph
        except Exception as e:
            self.logger.error(f"GT 拓樸建構失敗: {e}", exc_info=True)
            return None


# ============================================================================
# Hausdorff 距離計算器
# ============================================================================


# 已移至 neural_reconstruction.core.evaluation 模組
# 使用 TopologyComparator 取代 HausdorffCalculator


# ============================================================================
# 評測報告器
# ============================================================================


class EvaluationReporter:
    """
    評測報告器

    彙整結果並生成報告（JSON、CSV、終端輸出）。
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_report(
        self, results: List[SampleResult], config: Dict[str, Any]
    ) -> EvaluationSummary:
        """
        生成完整評測報告

        Args:
            results: 所有樣本的評測結果
            config: 評測配置

        Returns:
            評測統計摘要
        """
        self.logger.info("生成評測報告...")

        # 計算統計摘要
        summary = self._compute_summary(results)

        # 生成 JSON 報告
        self._save_json_report(results, summary, config)

        # 生成 CSV 報告
        self._save_csv_report(results)

        # 終端輸出摘要
        self._print_summary(summary)

        return summary

    def _compute_summary(self, results: List[SampleResult]) -> EvaluationSummary:
        """計算統計摘要"""
        total = len(results)
        processed = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")

        # 收集所有有效的 Hausdorff 距離
        valid_distances = [
            r.hausdorff_distance
            for r in results
            if r.status == "success" and r.hausdorff_distance is not None
        ]

        if valid_distances:
            hausdorff_mean = float(np.mean(valid_distances))
            hausdorff_median = float(np.median(valid_distances))
            hausdorff_std = float(np.std(valid_distances))
            hausdorff_min = float(np.min(valid_distances))
            hausdorff_max = float(np.max(valid_distances))
        else:
            hausdorff_mean = None
            hausdorff_median = None
            hausdorff_std = None
            hausdorff_min = None
            hausdorff_max = None

        return EvaluationSummary(
            total_samples=total,
            processed=processed,
            skipped=skipped,
            failed=failed,
            hausdorff_mean=hausdorff_mean,
            hausdorff_median=hausdorff_median,
            hausdorff_std=hausdorff_std,
            hausdorff_min=hausdorff_min,
            hausdorff_max=hausdorff_max,
        )

    def _save_json_report(
        self,
        results: List[SampleResult],
        summary: EvaluationSummary,
        config: Dict[str, Any],
    ):
        """儲存 JSON 報告"""
        json_path = self.output_dir / "results.json"

        # 轉換為字典格式
        samples_dict = {r.sample_id: asdict(r) for r in results}
        skipped_samples = [r.sample_id for r in results if r.status == "skipped"]

        report = {
            "summary": asdict(summary),
            "samples": samples_dict,
            "skipped_samples": skipped_samples,
            "config": config,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON 報告已儲存: {json_path}")

    def _save_csv_report(self, results: List[SampleResult]):
        """儲存 CSV 報告"""
        csv_path = self.output_dir / "results.csv"

        fieldnames = [
            "sample_id",
            "status",
            "hausdorff_distance",
            "num_nodes_pred",
            "num_nodes_gt",
            "num_edges_pred",
            "num_edges_gt",
            "num_components_pred",
            "error_message",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

        self.logger.info(f"CSV 報告已儲存: {csv_path}")

    def _print_summary(self, summary: EvaluationSummary):
        """終端輸出摘要"""
        print("\n" + "=" * 80)
        print("評測摘要 (Evaluation Summary)")
        print("=" * 80)
        print(f"總樣本數:     {summary.total_samples}")
        print(f"成功處理:     {summary.processed}")
        print(f"跳過:         {summary.skipped}")
        print(f"失敗:         {summary.failed}")
        print("-" * 80)

        if summary.hausdorff_mean is not None:
            print("平均 Hausdorff 距離統計:")
            print(f"  平均值:     {summary.hausdorff_mean:.4f}")
            print(f"  中位數:     {summary.hausdorff_median:.4f}")
            print(f"  標準差:     {summary.hausdorff_std:.4f}")
            print(f"  最小值:     {summary.hausdorff_min:.4f}")
            print(f"  最大值:     {summary.hausdorff_max:.4f}")
        else:
            print("平均 Hausdorff 距離: 無有效數據")

        print("=" * 80 + "\n")


# ============================================================================
# 主評測器
# ============================================================================


class DatasetEvaluator:
    """
    資料集評測器

    整合所有元件，執行完整的評測流程。
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        linker: Any,
    ):
        """
        Args:
            data_dir: 資料集目錄
            output_dir: 輸出目錄
            linker: 實作 run(image, mask, annotation) -> nx.Graph 的 linker 實例
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        # 建立元件
        self.loader = DatasetLoader(data_dir)
        self.extractor = TopologyExtractor(linker)
        self.comparator = TopologyComparator()
        self.reporter = EvaluationReporter(output_dir)

        self.config = {"linker": type(linker).__name__, "params": vars(linker)}

        self.logger = logging.getLogger(__name__)

    def evaluate(self, sample_ids: Optional[List[str]] = None) -> EvaluationSummary:
        """
        執行評測

        Args:
            sample_ids: 指定要評測的樣本 ID，None 則評測全部

        Returns:
            評測統計摘要
        """
        self.logger.info("開始資料集評測...")

        # 載入樣本
        samples = self.loader.load_samples(sample_ids)

        # 處理每個樣本
        results = []
        for sample in tqdm(samples, desc="評測進度"):
            result = self._evaluate_sample(sample)
            results.append(result)

        # 生成報告
        summary = self.reporter.generate_report(results, self.config)

        self.logger.info("評測完成")
        return summary

    def _evaluate_sample(self, sample: SampleFiles) -> SampleResult:
        """
        評測單一樣本

        Args:
            sample: 樣本檔案

        Returns:
            評測結果
        """
        # 必須有 label.png 才能計算 Hausdorff distance
        if not sample.label_path or not sample.label_path.exists():
            self.logger.info(f"樣本 {sample.sample_id} 跳過: 缺少 label.png")
            return SampleResult(
                sample_id=sample.sample_id,
                status="skipped",
                error_message="missing_label",
            )

        # 檢查其他必要檔案完整性
        is_complete, missing_reason = sample.is_complete()
        if not is_complete:
            self.logger.warning(f"樣本 {sample.sample_id} 跳過: {missing_reason}")
            return SampleResult(
                sample_id=sample.sample_id,
                status="skipped",
                error_message=missing_reason,
            )

        try:
            # 載入影像
            image = np.array(Image.open(sample.image_path))
            mask = np.array(Image.open(sample.mask_path))
            annotation = np.array(Image.open(sample.annotation_path))

            # 萃取 Pipeline 拓樸
            graph_pred = self.extractor.extract_from_pipeline(image, mask, annotation)

            if graph_pred is None:
                return SampleResult(
                    sample_id=sample.sample_id,
                    status="failed",
                    error_message="pipeline_failed",
                )

            # 萃取 GT 拓樸
            gt_label = np.array(Image.open(sample.label_path))
            graph_gt = self.extractor.extract_from_gt(gt_label)

            # 計算 Hausdorff 距離
            hausdorff_dist = None
            if graph_gt is not None:
                result = self.comparator.compare(
                    graph_pred, graph_gt, label1="pred", label2="gt"
                )
                if result.status == "success":
                    hausdorff_dist = result.hausdorff_distance

            # 收集統計資訊
            num_nodes_pred = graph_pred.number_of_nodes()
            num_edges_pred = graph_pred.number_of_edges()
            num_components_pred = nx.number_connected_components(graph_pred)

            num_nodes_gt = graph_gt.number_of_nodes() if graph_gt is not None else None
            num_edges_gt = graph_gt.number_of_edges() if graph_gt is not None else None

            return SampleResult(
                sample_id=sample.sample_id,
                status="success",
                hausdorff_distance=hausdorff_dist,
                num_nodes_pred=num_nodes_pred,
                num_nodes_gt=num_nodes_gt,
                num_edges_pred=num_edges_pred,
                num_edges_gt=num_edges_gt,
                num_components_pred=num_components_pred,
            )

        except Exception as e:
            self.logger.error(f"樣本 {sample.sample_id} 處理失敗: {e}", exc_info=True)
            return SampleResult(
                sample_id=sample.sample_id, status="failed", error_message=str(e)
            )


# ============================================================================
# 命令列介面
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    """設定日誌"""
    log_path = output_dir / "evaluation.log"

    # 設定日誌格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 設定檔案 handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # 設定終端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # 設定 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="資料集評測腳本 - 批次處理並評估神經重建結果"
    )

    parser.add_argument("--data-dir", type=Path, required=True, help="資料集根目錄")

    parser.add_argument("--output-dir", type=Path, required=True, help="輸出目錄")

    parser.add_argument(
        "--sample-ids", nargs="+", help="指定要評測的樣本 ID（可選，預設評測全部）"
    )

    parser.add_argument(
        "--algorithm",
        choices=["pure_mst", "hierarchical"],
        default="pure_mst",
        help="使用的重建演算法 (預設: pure_mst)",
    )

    parser.add_argument("--verbose", action="store_true", help="啟用詳細日誌輸出")

    args = parser.parse_args()

    # 建立輸出目錄
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 設定日誌
    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("資料集評測腳本")
    logger.info("=" * 80)
    logger.info(f"資料集目錄: {args.data_dir}")
    logger.info(f"輸出目錄: {args.output_dir}")
    logger.info(f"演算法: {args.algorithm}")

    # 建立 linker
    if args.algorithm == "pure_mst":
        linker = PureMstLinker(
            offset_px=100,
            rolling_ball_radius=2,
            sato_weight=0.0,
            opening_kernel_size=3,
            segment_length=5.0,
            search_radius=20.0,
            max_cost_threshold=0.98,
            intensity_weight=0.6,
        )
    else:  # hierarchical
        linker = HierarchicalFragmentLinker(
            offset_px=100,
            rolling_ball_radius=2,
            sato_weight=0.0,
            opening_kernel_size=3,
            segment_length=3.0,
            search_radius_pathfinding=50.0,
            search_radius_endpoint_extension=10.0,
            max_angle_endpoint_extension=75.0,
            search_radius_mst=20.0,
            max_angle_mst=90.0,
            max_cost_threshold_mst=0.75,
        )

    # 建立評測器
    evaluator = DatasetEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        linker=linker,
    )

    # 執行評測
    summary = evaluator.evaluate(sample_ids=args.sample_ids)

    logger.info("=" * 80)
    logger.info("評測完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
