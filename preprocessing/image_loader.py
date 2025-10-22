"""
圖像載入模組

負責載入和保存多通道 TIFF 圖像
"""

from typing import Optional
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile


class ImageLoader:
    """圖像載入器類別"""

    def __init__(self, verbose: bool = True):
        """
        初始化圖像載入器

        Args:
            verbose: 是否輸出詳細信息
        """
        self.verbose = verbose

    def load(self, image_path: str | Path) -> np.ndarray:
        """
        載入圖像文件

        支援格式: TIFF, TIF, PNG, JPG, JPEG

        Args:
            image_path: 圖像文件路徑

        Returns:
            numpy array 格式的圖像數據

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的圖像格式
        """
        image_path = Path(image_path)

        # 檢查文件是否存在
        if not image_path.exists():
            raise FileNotFoundError(f"圖像文件不存在: {image_path}")

        # 獲取文件擴展名
        ext = image_path.suffix.lower()

        # 根據擴展名選擇載入方法
        if ext in ['.tif', '.tiff']:
            image = self._load_tiff(image_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            image = self._load_standard(image_path)
        else:
            raise ValueError(f"不支持的圖像格式: {ext}")

        if self.verbose:
            print(f"✓ 載入圖像: {image_path.name}")
            print(f"  - 尺寸: {image.shape}")
            print(f"  - 類型: {image.dtype}")
            print(f"  - 範圍: [{image.min()}, {image.max()}]")

        return image

    def _load_tiff(self, image_path: Path) -> np.ndarray:
        """
        載入 TIFF 圖像（支援多通道和高位深度）

        Args:
            image_path: TIFF 圖像路徑

        Returns:
            numpy array 格式的圖像數據
        """
        try:
            # 使用 tifffile 載入，支援各種 TIFF 格式
            image = tifffile.imread(image_path)
            return image
        except Exception as e:
            # 如果 tifffile 失敗，嘗試使用 PIL
            try:
                image = np.array(Image.open(image_path))
                return image
            except Exception as e2:
                raise ValueError(f"無法載入 TIFF 圖像: {e}, {e2}")

    def _load_standard(self, image_path: Path) -> np.ndarray:
        """
        載入標準圖像格式（PNG, JPG 等）

        Args:
            image_path: 圖像路徑

        Returns:
            numpy array 格式的圖像數據
        """
        try:
            image = Image.open(image_path)
            image = np.array(image)
            return image
        except Exception as e:
            raise ValueError(f"無法載入圖像: {e}")

    def save(self, image: np.ndarray, output_path: str | Path, compress: bool = True) -> None:
        """
        保存圖像到文件

        Args:
            image: numpy array 格式的圖像數據
            output_path: 輸出文件路徑
            compress: 是否壓縮（僅對 TIFF 有效）

        Raises:
            ValueError: 圖像數據格式錯誤
        """
        output_path = Path(output_path)

        # 驗證圖像
        if not validate_image(image):
            raise ValueError("圖像數據無效")

        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 根據擴展名選擇保存方法
        ext = output_path.suffix.lower()

        if ext in ['.tif', '.tiff']:
            self._save_tiff(image, output_path, compress)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            self._save_standard(image, output_path)
        else:
            raise ValueError(f"不支持的輸出格式: {ext}")

        if self.verbose:
            print(f"✓ 保存圖像: {output_path.name}")

    def _save_tiff(self, image: np.ndarray, output_path: Path, compress: bool) -> None:
        """
        保存為 TIFF 格式

        Args:
            image: 圖像數據
            output_path: 輸出路徑
            compress: 是否壓縮
        """
        try:
            if compress:
                tifffile.imwrite(output_path, image, compression='lzw')
            else:
                tifffile.imwrite(output_path, image)
        except Exception as e:
            raise ValueError(f"無法保存 TIFF 圖像: {e}")

    def _save_standard(self, image: np.ndarray, output_path: Path) -> None:
        """
        保存為標準圖像格式

        Args:
            image: 圖像數據
            output_path: 輸出路徑
        """
        try:
            # 如果是浮點數，轉換為 uint8
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)

            # 如果是 16-bit，轉換為 8-bit
            elif image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)

            Image.fromarray(image).save(output_path)
        except Exception as e:
            raise ValueError(f"無法保存圖像: {e}")


def load_image(image_path: str | Path) -> np.ndarray:
    """
    載入圖像文件（便捷函數）

    Args:
        image_path: 圖像文件路徑

    Returns:
        numpy array 格式的圖像數據

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的圖像格式
    """
    loader = ImageLoader(verbose=False)
    return loader.load(image_path)


def save_image(image: np.ndarray, output_path: str | Path, compress: bool = True) -> None:
    """
    保存圖像到文件（便捷函數）

    Args:
        image: numpy array 格式的圖像數據
        output_path: 輸出文件路徑
        compress: 是否壓縮（僅對 TIFF 有效）

    Raises:
        ValueError: 圖像數據格式錯誤
    """
    loader = ImageLoader(verbose=False)
    loader.save(image, output_path, compress)


def extract_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    從 RGB 圖像中提取紅色和綠色通道

    Args:
        image: RGB 圖像 (H, W, 3) 或更多通道

    Returns:
        (red_channel, green_channel): 紅色和綠色通道的元組

    Raises:
        ValueError: 圖像格式不正確
    """
    # 檢查圖像維度
    if len(image.shape) < 3:
        raise ValueError(f"圖像必須是多通道格式，收到 {len(image.shape)}D")

    if image.shape[2] < 2:
        raise ValueError(f"圖像至少需要 2 個通道，收到 {image.shape[2]} 個")

    # 提取通道（假設是 RGB 順序）
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]

    return red_channel, green_channel


def validate_image(image: np.ndarray) -> bool:
    """
    驗證圖像數據是否有效

    Args:
        image: numpy array 格式的圖像數據

    Returns:
        True 如果圖像有效，否則 False
    """
    # 檢查是否為 numpy array
    if not isinstance(image, np.ndarray):
        return False

    # 檢查維度（2D 或 3D）
    if len(image.shape) not in [2, 3]:
        return False

    # 檢查尺寸是否合理
    if any(dim <= 0 for dim in image.shape):
        return False

    # 檢查數據類型
    if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        return False

    return True
