"""
資料集評測腳本 (Dataset Evaluation Script)

從推論結果（pkl 檔）計算評測指標，生成統計報告。
推論步驟請先執行 run_inference.py。

使用範例:
    python tools/evaluate_dataset.py \
        --inference-dir output/inference \
        --data-dir data/ \
        --output-dir output/evaluation

作者: Claude Code
日期: 2026-01-15
"""

import argparse
import csv
import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.evaluation import (
    extract_graph_points,
    compute_average_hausdorff_distance,
)


# ============================================================================
# 視覺化工具
# ============================================================================

# GT 綠色 / Pred 橘紅色
_COLOR_GT   = (0, 220, 0)      # BGR green
_COLOR_PRED = (0, 100, 255)    # BGR orange-red


def _draw_graph(canvas: np.ndarray, graph: nx.Graph, color: tuple, thickness: int = 1) -> None:
    """將 graph 的邊（優先使用 path 像素座標）畫到 canvas 上（in-place）。"""
    for u, v, data in graph.edges(data=True):
        path = data.get("path") or data.get("path-coordinates")
        if path and len(path) >= 2:
            pts = np.array(path, dtype=np.int32)   # (N, 2) as (y, x)
            for i in range(len(pts) - 1):
                cv2.line(
                    canvas,
                    (int(pts[i][1]),     int(pts[i][0])),      # (x, y)
                    (int(pts[i + 1][1]), int(pts[i + 1][0])),
                    color, thickness, lineType=cv2.LINE_AA,
                )
        else:
            cv2.line(
                canvas,
                (int(u[1]), int(u[0])),
                (int(v[1]), int(v[0])),
                color, thickness, lineType=cv2.LINE_AA,
            )


def save_overlay_visualization(
    sample_id: str,
    image_path: Optional[Path],
    roi_image: np.ndarray,
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    vis_dir: Path,
) -> None:
    """
    輸出三層疊加圖：原始輸入圖像 → GT 拓樸（綠）→ 預測拓樸（橘紅）。

    Args:
        image_path: 原始 image.png 路徑（存在時優先使用，否則用 roi_image）
        roi_image:  evaluate_sample 中的已處理 ROI 灰階影像（fallback）
        pred_graph: 預測的重建圖
        gt_graph:   GT 拓樸圖
        vis_dir:    輸出目錄
    """
    # --- 背景：優先使用原始圖像 ---
    if image_path is not None and image_path.exists():
        bg = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bg is None:
            bg = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    else:
        if roi_image.ndim == 2:
            bg = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        else:
            bg = roi_image.copy()

    canvas = bg.copy()
    _draw_graph(canvas, gt_graph,   _COLOR_GT,   thickness=1)
    _draw_graph(canvas, pred_graph, _COLOR_PRED, thickness=1)

    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{sample_id}.png"
    cv2.imwrite(str(out_path), canvas)


# ============================================================================
# 資料結構定義
# ============================================================================


@dataclass
class SampleResult:
    """單一樣本的評測結果"""

    sample_id: str
    status: str  # success, skipped, failed
    hausdorff_distance: Optional[float] = None
    hausdorff_distance_pred_to_gt: Optional[float] = None
    hausdorff_distance_gt_to_pred: Optional[float] = None
    num_nodes_pred: Optional[int] = None
    num_nodes_gt: Optional[int] = None
    num_edges_pred: Optional[int] = None
    num_edges_gt: Optional[int] = None
    num_components_pred: Optional[int] = None
    valid_count_pred: Optional[int] = None
    gt_count: Optional[int] = None
    count_error: Optional[float] = None  # |pred - gt|
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
    count_mae_mean: Optional[float] = None
    count_mae_median: Optional[float] = None
    count_mae_std: Optional[float] = None
    count_pearson_r: Optional[float] = None
    count_pearson_p: Optional[float] = None
    count_spearman_r: Optional[float] = None
    count_spearman_p: Optional[float] = None
    count_n: int = 0


# ============================================================================
# 評測報告器
# ============================================================================


class EvaluationReporter:
    """
    評測報告器

    彙整結果並生成報告（JSON、CSV、終端輸出）。
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_report(
        self, results: List[SampleResult], config: Dict[str, Any]
    ) -> EvaluationSummary:
        """生成完整評測報告"""
        self.logger.info("生成評測報告...")

        summary = self._compute_summary(results)
        self._save_json_report(results, summary, config)
        self._save_csv_report(results)
        self._print_summary(summary)

        return summary

    def _compute_summary(self, results: List[SampleResult]) -> EvaluationSummary:
        """計算統計摘要"""
        total = len(results)
        processed = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")

        # Hausdorff
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
            hausdorff_mean = hausdorff_median = hausdorff_std = None
            hausdorff_min = hausdorff_max = None

        # Count MAE + correlation
        count_pairs = [
            (r.valid_count_pred, r.gt_count)
            for r in results
            if r.status == "success"
            and r.valid_count_pred is not None
            and r.gt_count is not None
        ]
        count_mae_mean = count_mae_median = count_mae_std = None
        count_pearson_r = count_pearson_p = None
        count_spearman_r = count_spearman_p = None
        count_n = len(count_pairs)

        if count_pairs:
            preds, gts = zip(*count_pairs)
            errors = np.abs(np.array(preds, dtype=float) - np.array(gts, dtype=float))
            count_mae_mean = float(np.mean(errors))
            count_mae_median = float(np.median(errors))
            count_mae_std = float(np.std(errors))

            if count_n >= 2:
                pr, pp = pearsonr(preds, gts)
                sr, sp = spearmanr(preds, gts)
                count_pearson_r = float(pr)
                count_pearson_p = float(pp)
                count_spearman_r = float(sr)
                count_spearman_p = float(sp)

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
            count_mae_mean=count_mae_mean,
            count_mae_median=count_mae_median,
            count_mae_std=count_mae_std,
            count_pearson_r=count_pearson_r,
            count_pearson_p=count_pearson_p,
            count_spearman_r=count_spearman_r,
            count_spearman_p=count_spearman_p,
            count_n=count_n,
        )

    def _save_json_report(
        self,
        results: List[SampleResult],
        summary: EvaluationSummary,
        config: Dict[str, Any],
    ):
        """儲存 JSON 報告"""
        json_path = self.output_dir / "results.json"

        samples_dict = {r.sample_id: asdict(r) for r in results}
        skipped_samples = [r.sample_id for r in results if r.status == "skipped"]

        report = {
            "summary": asdict(summary),
            "config": config,
            "samples": samples_dict,
            "skipped_samples": skipped_samples,
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
            "hausdorff_distance_gt_to_pred",
            "hausdorff_distance_pred_to_gt",
            "num_nodes_pred",
            "num_nodes_gt",
            "num_edges_pred",
            "num_edges_gt",
            "num_components_pred",
            "valid_count_pred",
            "gt_count",
            "count_error",
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
            print("Hausdorff Distance:")
            print(f"  Mean:       {summary.hausdorff_mean:.4f}")
            print(f"  Median:     {summary.hausdorff_median:.4f}")
            print(f"  Std:        {summary.hausdorff_std:.4f}")
            print(f"  Min:        {summary.hausdorff_min:.4f}")
            print(f"  Max:        {summary.hausdorff_max:.4f}")
        else:
            print("Hausdorff Distance: no valid data")

        print("-" * 80)

        if summary.count_n > 0:
            print(f"Valid Count  (n={summary.count_n}):")
            print(f"  MAE Mean:   {summary.count_mae_mean:.4f}")
            print(f"  MAE Median: {summary.count_mae_median:.4f}")
            print(f"  MAE Std:    {summary.count_mae_std:.4f}")
            if summary.count_pearson_r is not None:
                print(
                    f"  Pearson r:  {summary.count_pearson_r:.4f}  (p={summary.count_pearson_p:.4f})"
                )
                print(
                    f"  Spearman r: {summary.count_spearman_r:.4f}  (p={summary.count_spearman_p:.4f})"
                )
        else:
            print("Valid Count: no GT data (count.json missing or no matching samples)")

        print("=" * 80 + "\n")


# ============================================================================
# 主評測器
# ============================================================================


class DatasetEvaluator:
    """
    資料集評測器

    從推論結果（pkl 檔）計算 Hausdorff 距離、count 誤差等指標。
    """

    def __init__(
        self,
        inference_dir: Path,
        data_dir: Path,
        output_dir: Path,
        num_workers: int = 1,
        visualize: bool = False,
    ):
        """
        Args:
            inference_dir: 推論結果目錄（含 {sample_id}.pkl 檔）
            data_dir: 資料集目錄（用於讀取 GT label 與 count.json）
            output_dir: 評測報告輸出目錄
            num_workers: 平行執行緒數量
            visualize: 是否輸出視覺化疊加圖至 output_dir/vis/
        """
        self.inference_dir = Path(inference_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.visualize = visualize
        self.vis_dir = Path(output_dir) / "vis" if visualize else None

        self.reporter = EvaluationReporter(output_dir)
        self.gt_counts = self._load_gt_counts(data_dir)

        self.config = {
            "inference_dir": str(inference_dir),
            "data_dir": str(data_dir),
        }
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _load_gt_counts(data_dir: Path) -> Dict[str, int]:
        count_path = Path(data_dir) / "count.json"
        if not count_path.exists():
            logging.getLogger(__name__).warning(
                f"count.json not found at {count_path} — count metrics will be skipped"
            )
            return {}
        with open(count_path, encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, sample_ids: Optional[List[str]] = None) -> EvaluationSummary:
        """
        執行評測

        Args:
            sample_ids: 指定要評測的樣本 ID；None 則自動掃描 inference_dir 內所有 pkl

        Returns:
            評測統計摘要
        """
        self.logger.info("開始資料集評測...")

        # 取得要評測的樣本清單
        if sample_ids is None:
            pkl_files = sorted(self.inference_dir.glob("*.pkl"))
            sample_ids = [p.stem for p in pkl_files]
            self.logger.info(f"在 {self.inference_dir} 找到 {len(sample_ids)} 個推論結果")
        else:
            self.logger.info(f"指定評測 {len(sample_ids)} 個樣本")

        results = []
        for sid in tqdm(sample_ids, desc="評測進度"):
            results.append(self._evaluate_sample(sid))

        summary = self.reporter.generate_report(results, self.config)
        self.logger.info("評測完成")
        return summary

    def _evaluate_sample(self, sample_id: str) -> SampleResult:
        """評測單一樣本"""
        pkl_path = self.inference_dir / f"{sample_id}.pkl"

        if not pkl_path.exists():
            self.logger.warning(f"樣本 {sample_id} 跳過: 找不到 {pkl_path}")
            return SampleResult(
                sample_id=sample_id,
                status="skipped",
                error_message="missing_inference_result",
            )

        try:
            with open(pkl_path, "rb") as f:
                extract_result = pickle.load(f)

            # --- Hausdorff ---
            avg_dist = d_pred_to_gt = d_gt_to_pred = None
            num_nodes_gt = num_edges_gt = None

            gt_label_path = self.data_dir / sample_id / "label.png"
            if gt_label_path.exists():
                from PIL import Image

                gt_label = np.array(Image.open(gt_label_path))
                roi_label = cv2.bitwise_and(
                    gt_label, gt_label, mask=extract_result.mask
                )
                graph_gt = TopologyBuilder().build_seed_graph(roi_label)

                pred_points = extract_graph_points(extract_result.graph)
                gt_points = extract_graph_points(graph_gt)

                avg_dist, d_pred_to_gt, d_gt_to_pred = compute_average_hausdorff_distance(
                    pred_points, gt_points
                )
                num_nodes_gt = graph_gt.number_of_nodes()
                num_edges_gt = graph_gt.number_of_edges()

                if self.visualize:
                    save_overlay_visualization(
                        sample_id=sample_id,
                        image_path=self.data_dir / sample_id / "image.png",
                        roi_image=extract_result.image,
                        pred_graph=extract_result.graph,
                        gt_graph=graph_gt,
                        vis_dir=self.vis_dir,
                    )

            # --- Count ---
            gt_count = self.gt_counts.get(sample_id)
            valid_count_pred = extract_result.valid_count
            count_error = (
                float(abs(valid_count_pred - gt_count))
                if gt_count is not None
                else None
            )

            return SampleResult(
                sample_id=sample_id,
                status="success",
                hausdorff_distance=avg_dist,
                hausdorff_distance_pred_to_gt=d_pred_to_gt,
                hausdorff_distance_gt_to_pred=d_gt_to_pred,
                num_nodes_pred=extract_result.graph.number_of_nodes(),
                num_nodes_gt=num_nodes_gt,
                num_edges_pred=extract_result.graph.number_of_edges(),
                num_edges_gt=num_edges_gt,
                num_components_pred=nx.number_connected_components(
                    extract_result.graph
                ),
                valid_count_pred=valid_count_pred,
                gt_count=gt_count,
                count_error=count_error,
            )

        except Exception as e:
            self.logger.error(f"樣本 {sample_id} 評測失敗: {e}", exc_info=True)
            return SampleResult(
                sample_id=sample_id, status="failed", error_message=str(e)
            )


# ============================================================================
# 命令列介面
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    """設定日誌"""
    log_path = output_dir / "evaluation.log"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(
        description="資料集評測腳本 - 從推論結果計算 Hausdorff 距離與 count 指標"
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        required=True,
        help="推論結果目錄（run_inference.py 輸出的 pkl 檔目錄）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="資料集根目錄（用於讀取 GT label 與 count.json）",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="報告輸出目錄")
    parser.add_argument(
        "--sample-ids", nargs="+", help="指定要評測的樣本 ID（可選，預設評測全部）"
    )
    parser.add_argument("--verbose", action="store_true", help="啟用詳細日誌輸出")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="輸出視覺化疊加圖（預測拓樸 + GT 拓樸疊加在輸入圖像上），儲存至 output-dir/vis/",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("資料集評測腳本")
    logger.info("=" * 80)
    logger.info(f"推論結果目錄: {args.inference_dir}")
    logger.info(f"資料集目錄:   {args.data_dir}")
    logger.info(f"輸出目錄:     {args.output_dir}")

    evaluator = DatasetEvaluator(
        inference_dir=args.inference_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=args.visualize,
    )

    evaluator.evaluate(sample_ids=args.sample_ids)

    logger.info("=" * 80)
    logger.info("評測完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
