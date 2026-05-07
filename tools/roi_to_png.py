"""
將 ImageJ .roi 檔(多邊形)轉成與原始影像同尺寸的二值 PNG mask。
路徑直接寫死在檔案開頭。

每個 .roi 必須對應 IMAGE_DIR 中同名的影像(支援 .tif / .png),
以取得正確的畫布尺寸。輸出檔名為 <sample_id>.png,內容為 0/255。
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from read_roi import read_roi_file

ROI_DIR = Path("/home/pony/projects/ienf_q/nas/neuroimages/20250214/第一批/Epider roi_cal")
IMAGE_DIR = Path("/home/pony/projects/ienf_q/nas/neuroimages/20250214/第一批/Image/Original")
OUTPUT_DIR = Path("/home/pony/projects/ienf_q/data_0331/epidermis_masks")
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def find_image(sample_id: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = IMAGE_DIR / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def roi_to_mask(roi_path: Path, width: int, height: int) -> np.ndarray:
    rois = read_roi_file(str(roi_path))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for roi in rois.values():
        roi_type = roi.get("type")
        xs, ys = roi.get("x"), roi.get("y")
        if xs is None or ys is None:
            print(f"  skip ROI '{roi.get('name')}' (type={roi_type}, no coords)")
            continue
        points = list(zip(xs, ys))
        if roi_type in ("polygon", "freehand", "traced", "polyline", "freeline"):
            if len(points) >= 3:
                draw.polygon(points, fill=255)
            else:
                draw.line(points, fill=255, width=1)
        else:
            draw.polygon(points, fill=255)
    return np.array(mask, dtype=np.uint8)


def main() -> None:
    roi_files = sorted(ROI_DIR.glob("*.roi"))
    print(f"ROI files: {len(roi_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, missing, failed = 0, 0, 0

    for roi_path in roi_files:
        sample_id = roi_path.stem
        image_path = find_image(sample_id)
        if image_path is None:
            print(f"  [missing image] {sample_id}")
            missing += 1
            continue
        try:
            with Image.open(image_path) as im:
                width, height = im.size
            mask = roi_to_mask(roi_path, width, height)
            out_path = OUTPUT_DIR / f"{sample_id}.png"
            Image.fromarray(mask).save(out_path)
            ok += 1
        except Exception as e:
            print(f"  [failed] {sample_id}: {e}")
            failed += 1

    print(f"\nDone. OK={ok}, missing image={missing}, failed={failed}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
