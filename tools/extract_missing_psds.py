"""
檢查 PSD 資料夾中,sample_id 沒有出現在 count.json 的 PSD 檔,
並將這些缺漏的 PSD 複製到輸出資料夾。

PSD 檔名格式假設為 `<sample_id>_stack.psd`(例如 S1140-2_a_stack.psd)。
路徑直接寫死在檔案開頭。
"""

import json
import shutil
from pathlib import Path

PSD_DIR = Path("/home/pony/projects/ienf_q/nas/neuroimages/Control_psd_traced_20260430")
JSON_PATH = Path("/home/pony/projects/ienf_q/data_0331/count.json")
OUTPUT_DIR = Path("/home/pony/projects/ienf_q/psd_0503")


def psd_to_sample_id(psd_path: Path) -> str:
    name = psd_path.stem
    return name.removesuffix("_stack")


def main() -> None:
    with JSON_PATH.open("r", encoding="utf-8") as f:
        known_ids = set(json.load(f).keys())

    psd_files = sorted(PSD_DIR.glob("*.psd"))
    missing = [p for p in psd_files if psd_to_sample_id(p) not in known_ids]

    print(f"PSD files scanned: {len(psd_files)}")
    print(f"Sample IDs in JSON: {len(known_ids)}")
    print(f"Missing PSDs: {len(missing)}")

    if not missing:
        print("Nothing to copy.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for psd in missing:
        dest = OUTPUT_DIR / psd.name
        shutil.copy2(psd, dest)
        print(f"  copied {psd.name} -> {dest}")

    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
