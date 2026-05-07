"""
統計資料集影像尺寸範圍。

使用範例:
    python tools/image_size_stats.py --data-dir data_0331
    python tools/image_size_stats.py --data-dir data_0331 --image-name mask.png
    python tools/image_size_stats.py --data-dir data_0331 --all-images --csv output/image_sizes.csv
    python tools/image_size_stats.py --data-dir data_0331 --json output/image_size_summary.json
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Optional

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageRecord:
    """單張圖片的尺寸紀錄。"""

    sample_id: str
    path: str
    width: int
    height: int
    area: int
    mode: str


@dataclass
class RangeStats:
    """一組數值的摘要統計。"""

    min: int
    max: int
    mean: float
    median: float


def find_images(data_dir: Path, image_name: str, all_images: bool) -> List[Path]:
    """依照資料集結構或遞迴模式找出影像路徑。"""
    if all_images:
        return sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    return sorted(
        path
        for sample_dir in data_dir.iterdir()
        if sample_dir.is_dir()
        for path in [sample_dir / image_name]
        if path.exists()
    )


def read_image_records(paths: Iterable[Path], data_dir: Path) -> List[ImageRecord]:
    """讀取影像尺寸，遇到壞檔會跳過並印出警告。"""
    records: List[ImageRecord] = []

    for path in paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        except (OSError, UnidentifiedImageError) as exc:
            print(f"[WARN] 無法讀取 {path}: {exc}")
            continue

        relative_path = path.relative_to(data_dir)
        sample_id = relative_path.parts[0] if len(relative_path.parts) > 1 else path.stem
        records.append(
            ImageRecord(
                sample_id=sample_id,
                path=str(relative_path),
                width=width,
                height=height,
                area=width * height,
                mode=mode,
            )
        )

    return records


def summarize_values(values: List[int]) -> RangeStats:
    """建立 min/max/mean/median 摘要。"""
    return RangeStats(
        min=min(values),
        max=max(values),
        mean=round(mean(values), 2),
        median=round(median(values), 2),
    )


def summarize(records: List[ImageRecord]) -> dict:
    """彙整影像尺寸統計。"""
    widths = [record.width for record in records]
    heights = [record.height for record in records]
    areas = [record.area for record in records]
    sizes = Counter(f"{record.width}x{record.height}" for record in records)
    modes = Counter(record.mode for record in records)

    min_area_record = min(records, key=lambda record: record.area)
    max_area_record = max(records, key=lambda record: record.area)

    return {
        "count": len(records),
        "width": asdict(summarize_values(widths)),
        "height": asdict(summarize_values(heights)),
        "area": asdict(summarize_values(areas)),
        "unique_sizes": len(sizes),
        "size_distribution": dict(sorted(sizes.items())),
        "mode_distribution": dict(sorted(modes.items())),
        "smallest_image": asdict(min_area_record),
        "largest_image": asdict(max_area_record),
    }


def write_csv(records: List[ImageRecord], output_path: Path) -> None:
    """將逐張影像尺寸寫成 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "path", "width", "height", "area", "mode"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_json(summary: dict, output_path: Path) -> None:
    """將摘要統計寫成 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def print_summary(summary: dict, csv_path: Optional[Path], json_path: Optional[Path]) -> None:
    """印出終端摘要。"""
    print("影像尺寸統計")
    print("=" * 48)
    print(f"影像數量: {summary['count']}")
    print(
        "寬度範圍: "
        f"{summary['width']['min']} - {summary['width']['max']} px "
        f"(平均 {summary['width']['mean']}, 中位數 {summary['width']['median']})"
    )
    print(
        "高度範圍: "
        f"{summary['height']['min']} - {summary['height']['max']} px "
        f"(平均 {summary['height']['mean']}, 中位數 {summary['height']['median']})"
    )
    print(
        "面積範圍: "
        f"{summary['area']['min']} - {summary['area']['max']} px^2 "
        f"(平均 {summary['area']['mean']}, 中位數 {summary['area']['median']})"
    )
    print(f"不同尺寸數量: {summary['unique_sizes']}")
    print(f"色彩模式分佈: {summary['mode_distribution']}")
    print(
        "最小影像: "
        f"{summary['smallest_image']['path']} "
        f"({summary['smallest_image']['width']}x{summary['smallest_image']['height']})"
    )
    print(
        "最大影像: "
        f"{summary['largest_image']['path']} "
        f"({summary['largest_image']['width']}x{summary['largest_image']['height']})"
    )

    print("\n尺寸分佈:")
    for size, count in summary["size_distribution"].items():
        print(f"  {size}: {count}")

    if csv_path:
        print(f"\nCSV 已輸出: {csv_path}")
    if json_path:
        print(f"JSON 已輸出: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="統計 data 目錄中的影像尺寸範圍")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_0331"),
        help="資料集根目錄，預設為 data_0331",
    )
    parser.add_argument(
        "--image-name",
        default="image.png",
        help="每個 sample 目錄中要統計的影像檔名，預設為 image.png",
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="遞迴統計 data-dir 下所有常見影像檔，而不是只統計每個 sample 的 image-name",
    )
    parser.add_argument("--csv", type=Path, help="輸出逐張影像尺寸 CSV")
    parser.add_argument("--json", type=Path, help="輸出摘要統計 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"資料目錄不存在: {data_dir}")

    image_paths = find_images(data_dir, args.image_name, args.all_images)
    if not image_paths:
        mode_message = "所有影像" if args.all_images else f"*/{args.image_name}"
        raise FileNotFoundError(f"在 {data_dir} 找不到 {mode_message}")

    records = read_image_records(image_paths, data_dir)
    if not records:
        raise RuntimeError("沒有成功讀取任何影像")

    summary = summarize(records)

    if args.csv:
        write_csv(records, args.csv)
    if args.json:
        write_json(summary, args.json)

    print_summary(summary, args.csv, args.json)


if __name__ == "__main__":
    main()
