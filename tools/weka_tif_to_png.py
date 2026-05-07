"""
將 Weka 二值 TIF (值為 0/1) 轉成可視化的 PNG (反轉後值為 255/0,
即原本 0 → 255、原本 1 → 0)。
路徑直接寫死在檔案開頭。
"""

import numpy as np
from pathlib import Path
from PIL import Image

INPUT_DIR = Path("/home/pony/projects/ienf_q/nas/neuroimages/20250214/第一批/Weka3_1_1")
OUTPUT_DIR = Path("/home/pony/projects/ienf_q/data_0331/weka_masks")


def main() -> None:
    tif_files = sorted(INPUT_DIR.glob("*.tif"))
    print(f"TIF files: {len(tif_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0

    for tif_path in tif_files:
        sample_id = tif_path.stem
        try:
            with Image.open(tif_path) as im:
                arr = np.array(im)
            mask = np.where(arr > 0, 0, 255).astype(np.uint8)
            out_path = OUTPUT_DIR / f"{sample_id}.png"
            Image.fromarray(mask).save(out_path)
            ok += 1
        except Exception as e:
            print(f"  [failed] {sample_id}: {e}")
            failed += 1

    print(f"\nDone. OK={ok}, failed={failed}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
