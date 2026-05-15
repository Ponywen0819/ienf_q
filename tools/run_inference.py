"""
推論腳本 (Inference Script)

對整個資料集執行推論，將每個樣本的 LinkerResult 儲存為 pkl 檔，
供後續 evaluate_dataset.py 計算指標使用。

使用範例:
    python tools/run_inference.py \
        --data-dir data/ \
        --output-dir output/inference \
        --algorithm pure_mst \
        --workers 4
"""

import argparse
import concurrent.futures
import copy
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from neural_reconstruction.dataset import SampleFiles, DatasetLoader
from neural_reconstruction.algorithms.pure_mst.linker import PureMstLinker
from neural_reconstruction.algorithms.annotation_grow import AnnotationGrowLinker
from neural_reconstruction.algorithms.label_linker import LabelLinker
from neural_reconstruction.algorithms.skeleton_linker import SkeletonLinker


# ============================================================================
# 推論器
# ============================================================================


class DatasetInferencer:
    """
    資料集推論器

    對每個樣本執行 linker，將 LinkerResult 儲存為 pkl 檔。
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        linker: Any,
        num_workers: int = 1,
    ):
        """
        Args:
            data_dir: 資料集目錄
            output_dir: 輸出目錄（pkl 檔存放處）
            linker: 實作 run(image, mask, annotation) -> LinkerResult 的 linker 實例
            num_workers: 平行執行緒數量
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._linker = linker
        self.num_workers = num_workers

        self.loader = DatasetLoader(data_dir)
        self._thread_local = threading.local()
        self.logger = logging.getLogger(__name__)

    def _get_linker(self) -> Any:
        """取得當前執行緒專用的 linker（deep copy 確保執行緒安全）"""
        if not hasattr(self._thread_local, "linker"):
            self._thread_local.linker = copy.deepcopy(self._linker)
        return self._thread_local.linker

    def run(self, sample_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        執行推論

        Args:
            sample_ids: 指定要處理的樣本 ID，None 則處理全部

        Returns:
            {'success': [...], 'skipped': [...], 'failed': [...]}
        """
        self.logger.info("開始資料集推論...")
        self.logger.info(f"平行執行緒數: {self.num_workers}")

        samples = self.loader.load_samples(sample_ids)
        results: Dict[str, List[str]] = {"success": [], "skipped": [], "failed": []}

        if self.num_workers <= 1:
            for sample in tqdm(samples, desc="推論進度"):
                status, sid = self._process_sample(sample)
                results[status].append(sid)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = {
                    executor.submit(self._process_sample, sample): sample
                    for sample in samples
                }
                with tqdm(total=len(futures), desc="推論進度") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        status, sid = future.result()
                        results[status].append(sid)
                        pbar.update(1)

        self.logger.info(
            f"推論完成: 成功={len(results['success'])}, "
            f"跳過={len(results['skipped'])}, "
            f"失敗={len(results['failed'])}"
        )
        return results

    def _process_sample(self, sample: SampleFiles):
        """處理單一樣本，儲存推論結果為 pkl"""
        try:
            linker = self._get_linker()

            if isinstance(linker, LabelLinker):
                if sample.label_path is None or not sample.label_path.exists():
                    self.logger.warning(
                        f"樣本 {sample.sample_id} 跳過: label.png 不存在"
                    )
                    return "skipped", sample.sample_id
                if not sample.mask_path.exists():
                    self.logger.warning(f"樣本 {sample.sample_id} 跳過: missing_mask")
                    return "skipped", sample.sample_id
                if not sample.annotation_path.exists():
                    self.logger.warning(
                        f"樣本 {sample.sample_id} 跳過: missing_annotation"
                    )
                    return "skipped", sample.sample_id
                mask = np.array(Image.open(sample.mask_path))
                label = np.array(Image.open(sample.label_path))
                annotation = np.array(Image.open(sample.annotation_path))
                result = linker.run(mask, label, annotation)
            else:
                is_complete, reason = sample.is_complete()
                if not is_complete:
                    self.logger.warning(f"樣本 {sample.sample_id} 跳過: {reason}")
                    return "skipped", sample.sample_id
                image = np.array(Image.open(sample.image_path))
                mask = np.array(Image.open(sample.mask_path))
                annotation = np.array(Image.open(sample.annotation_path))
                result = linker.run(image, mask, annotation)

            if result is None:
                self.logger.error(f"樣本 {sample.sample_id} 推論失敗: 返回 None")
                return "failed", sample.sample_id

            out_path = self.output_dir / f"{sample.sample_id}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(result, f)

            self.logger.debug(f"樣本 {sample.sample_id} 推論完成，儲存至 {out_path}")
            return "success", sample.sample_id

        except Exception as e:
            self.logger.error(f"樣本 {sample.sample_id} 處理失敗: {e}", exc_info=True)
            return "failed", sample.sample_id


# ============================================================================
# 命令列介面
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    """設定日誌"""
    log_path = output_dir / "inference.log"
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


def build_linker(algorithm: str) -> Any:
    """根據演算法名稱建立 linker"""
    if algorithm == "pure_mst":
        return PureMstLinker(
            offset_px=50,
            bg_kernel_size=3,
            clahe_grid=(768, 768),
            clahe_clip=20.0,
            sato_sigmas_start=3,
            sato_sigmas_stop=8,
            segment_length=5.0,
            search_radius=50.0,
            min_component_length=3.0,
        )
    elif algorithm == "annotation_grow":
        return AnnotationGrowLinker(
            offset_px=50,
            bg_kernel_size=3,
            clahe_grid=(768, 768),
            clahe_clip=20.0,
            sato_sigmas_start=3,
            sato_sigmas_stop=8,
            prune_threshold=20.0,
        )
    elif algorithm == "label":
        return LabelLinker(offset_px=50)
    elif algorithm == "skeleton":
        return SkeletonLinker(offset_px=50, segment_length=3.0)
    else:
        raise ValueError(f"未知的演算法選項: {algorithm}")


def main():
    parser = argparse.ArgumentParser(
        description="資料集推論腳本 - 執行 linker 並儲存推論結果"
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="資料集根目錄")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="推論結果輸出目錄（pkl 檔）"
    )
    parser.add_argument(
        "--sample-ids", nargs="+", help="指定要處理的樣本 ID（可選，預設處理全部）"
    )
    parser.add_argument(
        "--algorithm",
        choices=[
            "pure_mst",
            "annotation_grow",
            "label",
            "skeleton",
        ],
        default="pure_mst",
        help="使用的重建演算法 (預設: pure_mst)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"平行處理的執行緒數量（預設: 1）",
    )
    parser.add_argument("--verbose", action="store_true", help="啟用詳細日誌輸出")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("資料集推論腳本")
    logger.info("=" * 80)
    logger.info(f"資料集目錄: {args.data_dir}")
    logger.info(f"輸出目錄:   {args.output_dir}")
    logger.info(f"演算法:     {args.algorithm}")
    logger.info(f"執行緒數:   {args.workers}")

    linker = build_linker(args.algorithm)

    inferencer = DatasetInferencer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        linker=linker,
        num_workers=args.workers,
    )

    inferencer.run(sample_ids=args.sample_ids)

    logger.info("=" * 80)
    logger.info("推論完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
