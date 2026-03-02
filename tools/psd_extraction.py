from psd_tools import PSDImage
from PIL import Image
import numpy as np
from pathlib import Path

base_path = "/home/pony/projects/ienf_q/label_0302"
output_base = Path("./data_0320")
# 設定要找的圖層名稱
target_layer_name = "refine"

# 1. 搜索所有 .psd 檔案
psd_files = list(Path(base_path).glob("*.psd"))
print(f"找到 {len(psd_files)} 個 PSD 檔案")

for psd_file in psd_files:
    # 2. 從檔案名稱提取 ID (例如: S222-2_a_stack.psd -> S222-2_a)
    file_name = psd_file.stem  # 去除副檔名
    if "_stack" in file_name:
        sample_id = file_name.replace("_stack", "")
    else:
        sample_id = file_name

    print(f"\n處理檔案: {psd_file.name}")
    print(f"樣本 ID: {sample_id}")

    # 3. 載入 PSD 檔案
    try:
        psd = PSDImage.open(str(psd_file))
    except Exception as e:
        print(f"  ⚠️  無法載入 PSD: {e}")
        continue

    # 4. 遍歷圖層
    found = False
    for layer in psd:
        if layer.name == target_layer_name:
            print(f"  ✓ 找到圖層 '{target_layer_name}'！")

            try:
                image = layer.composite()
                image = np.array(image).astype(np.uint8)
                aplpha_channel = image[:, :, 3]  # 取得 alpha 通道

                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                target_pixels = (
                    (aplpha_channel > 0)
                    & (image[:, :, 0] > 127)
                    & (image[:, :, 1] > 127)
                    & (image[:, :, 2] > 127)
                )  # 非透明部分的像素位置
                mask[target_pixels] = 255  # 取得非透明部分的像素

                # 5. 建立輸出路徑並儲存
                output_dir = output_base / sample_id
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "label.png"

                Image.fromarray(mask).save(str(output_path))
                print(f"  ✓ 已儲存 mask 到: {output_path}")
                found = True
                break
            except Exception as e:
                print(f"  ⚠️  處理圖層時發生錯誤: {e}")
                continue

    if not found:
        print(f"  ⚠️  未找到圖層 '{target_layer_name}'")
