from psd_tools import PSDImage
from PIL import Image
import numpy as np
from pathlib import Path

base_path = "/home/pony/projects/ienf_q/nas/jia/neuro_psd"
output_base = Path("./data_0331")

# 設定要抽取的圖層，格式: {圖層名稱: 輸出檔案名稱}
target_layers = {
    "new_weka": "weka.png",
    "new_label": "label.png",
}

# 1. 搜索所有 .psd 檔案
psd_files = list(Path(base_path).glob("S222-2_a_stack.psd"))
print(f"Found {len(psd_files)} PSD files")

for psd_file in psd_files:
    # 2. 從檔案名稱提取 ID (例如: S222-2_a_stack.psd -> S222-2_a)
    file_name = psd_file.stem
    if "_stack" in file_name:
        sample_id = file_name.replace("_stack", "")
    else:
        sample_id = file_name

    print(f"\nProcessing: {psd_file.name}  (sample ID: {sample_id})")

    try:
        psd = PSDImage.open(str(psd_file))
    except Exception as e:
        print(f"  WARNING: Cannot open PSD: {e}")
        continue

    output_dir = output_base / sample_id

    # 3. 遍歷圖層，找到所有目標圖層
    found_layers = {}
    for layer in psd:
        if layer.name in target_layers:
            found_layers[layer.name] = layer

    for layer_name, output_filename in target_layers.items():
        if layer_name not in found_layers:
            print(f"  WARNING: Layer '{layer_name}' not found")
            continue

        layer = found_layers[layer_name]
        try:
            image = layer.composite()
            image = np.array(image).astype(np.uint8)

            # 二值化成遮罩：非透明 且 有顏色 (RGB > 126) 的部分為 255
            alpha = image[:, :, 3]
            has_color = np.any(image[:, :, :3] > 126, axis=2)
            layer_mask = np.where((alpha > 0) & has_color, 255, 0).astype(np.uint8)

            # 將圖層貼到 canvas 大小的全零底圖（處理偏移與尺寸差異）
            canvas_h = psd.height  # type: ignore[attr-defined]
            canvas_w = psd.width  # type: ignore[attr-defined]
            mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

            # 圖層在 canvas 上的位置
            x0 = layer.left
            y0 = layer.top
            x1 = x0 + layer_mask.shape[1]
            y1 = y0 + layer_mask.shape[0]

            # 計算有效的貼入範圍（防止超出 canvas 邊界）
            cx0 = max(x0, 0)
            cy0 = max(y0, 0)
            cx1 = min(x1, canvas_w)
            cy1 = min(y1, canvas_h)

            # 對應到 layer_mask 的裁切範圍
            lx0 = cx0 - x0
            ly0 = cy0 - y0
            lx1 = lx0 + (cx1 - cx0)
            ly1 = ly0 + (cy1 - cy0)

            if cx1 > cx0 and cy1 > cy0:
                mask[cy0:cy1, cx0:cx1] = layer_mask[ly0:ly1, lx0:lx1]

            print(
                f"  Layer '{layer_name}': size={layer_mask.shape}, offset=({x0},{y0}), canvas=({canvas_w}x{canvas_h})"
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename
            Image.fromarray(mask).save(str(output_path))
            print(f"  OK: '{layer_name}' -> {output_path}")
        except Exception as e:
            print(f"  WARNING: Error processing layer '{layer_name}': {e}")
