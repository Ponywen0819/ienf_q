"""
評估資料型別模組 (Evaluation Data Types)

定義評估和比對相關的資料結構。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ComparisonResult:
    """
    拓樸比對結果

    包含兩個圖的統計資訊和度量結果。

    Attributes:
        label1: 第一個圖的標籤
        label2: 第二個圖的標籤
        num_nodes1: 第一個圖的節點數
        num_nodes2: 第二個圖的節點數
        num_edges1: 第一個圖的邊數
        num_edges2: 第二個圖的邊數
        num_points1: 第一個圖的總點數（節點+路徑點）
        num_points2: 第二個圖的總點數（節點+路徑點）
        hausdorff_distance: 平均 Hausdorff 距離
        hausdorff_a_to_b: 單向距離 d(A→B)
        hausdorff_b_to_a: 單向距離 d(B→A)
        status: 比對狀態 ('success', 'failed')
        error: 錯誤訊息（如果失敗）
    """
    label1: str
    label2: str
    num_nodes1: int
    num_nodes2: int
    num_edges1: int
    num_edges2: int
    num_points1: Optional[int] = None
    num_points2: Optional[int] = None
    hausdorff_distance: Optional[float] = None
    hausdorff_a_to_b: Optional[float] = None
    hausdorff_b_to_a: Optional[float] = None
    status: str = 'success'
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """
        轉換為字典

        Returns:
            包含所有欄位的字典
        """
        return {
            'label1': self.label1,
            'label2': self.label2,
            'num_nodes1': self.num_nodes1,
            'num_nodes2': self.num_nodes2,
            'num_edges1': self.num_edges1,
            'num_edges2': self.num_edges2,
            'num_points1': self.num_points1,
            'num_points2': self.num_points2,
            'hausdorff_distance': self.hausdorff_distance,
            'hausdorff_a_to_b': self.hausdorff_a_to_b,
            'hausdorff_b_to_a': self.hausdorff_b_to_a,
            'status': self.status,
            'error': self.error
        }


@dataclass
class EvaluationMetrics:
    """
    評估度量集合

    可擴展支援多種度量。

    Attributes:
        hausdorff_distance: 平均 Hausdorff 距離
        hausdorff_a_to_b: 單向距離 d(A→B)
        hausdorff_b_to_a: 單向距離 d(B→A)
    """
    hausdorff_distance: float
    hausdorff_a_to_b: Optional[float] = None
    hausdorff_b_to_a: Optional[float] = None

    def to_dict(self) -> dict:
        """
        轉換為字典

        Returns:
            包含所有度量的字典
        """
        return {
            'hausdorff_distance': self.hausdorff_distance,
            'hausdorff_a_to_b': self.hausdorff_a_to_b,
            'hausdorff_b_to_a': self.hausdorff_b_to_a
        }
