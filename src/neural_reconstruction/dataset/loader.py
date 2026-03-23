"""
Dataset loader for IENF-Q sample directories.

Each sample directory is expected to contain:
  image.png      - original image
  mask.png       - epidermis mask
  annotation.png - manual annotation
  label.png      - ground truth label (optional, for evaluation)
  lable.png      - alternative spelling tolerated
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class SampleFiles:
    """File paths for a single dataset sample."""

    sample_id: str
    image_path: Path
    mask_path: Path
    annotation_path: Path
    label_path: Optional[Path] = None  # GT label, optional

    def is_complete(self) -> Tuple[bool, str]:
        """Check that required files exist.

        Returns:
            (is_complete, missing_reason)
        """
        if not self.image_path.exists():
            return False, "missing_image"
        if not self.mask_path.exists():
            return False, "missing_mask"
        if not self.annotation_path.exists():
            return False, "missing_annotation"
        return True, ""


class DatasetLoader:
    """Scan a dataset root directory and return a list of SampleFiles."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def load_samples(self, sample_ids: Optional[List[str]] = None) -> List[SampleFiles]:
        """Load dataset samples.

        Args:
            sample_ids: Restrict to these sample IDs; None loads all.

        Returns:
            Sorted list of SampleFiles.
        """
        self.logger.info(f"Scanning dataset directory: {self.data_dir}")

        if sample_ids:
            sample_dirs = [
                self.data_dir / sid
                for sid in sample_ids
                if (self.data_dir / sid).is_dir()
            ]
        else:
            sample_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        self.logger.info(f"Found {len(sample_dirs)} sample directories")

        samples = []
        for sample_dir in sorted(sample_dirs):
            sample_id = sample_dir.name

            # Tolerate both spellings of "label"
            label_path: Optional[Path] = None
            if (sample_dir / "label.png").exists():
                label_path = sample_dir / "label.png"
            elif (sample_dir / "lable.png").exists():
                label_path = sample_dir / "lable.png"

            samples.append(
                SampleFiles(
                    sample_id=sample_id,
                    image_path=sample_dir / "image.png",
                    mask_path=sample_dir / "mask.png",
                    annotation_path=sample_dir / "weka.png",
                    label_path=label_path,
                )
            )

        return samples
