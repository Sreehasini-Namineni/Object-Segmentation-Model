"""
This package contains utility functions for:
- Data I/O (OCID dataset)
- Evaluation metrics (IoU-based, Hungarian matching)
- Visualization
- Dataset loading
"""

from .ocid_io import (
    Intrinsics,
    read_rgb,
    read_depth_png,
    read_label_png
)

from .segmentation_types import SegmentationResult

from .evaluation import (
    compute_iou,
    match_instances_hungarian,
    evaluate_single_frame,
    evaluate_dataset
)

from .dataset_loader import load_dataset

__all__ = [
    # Data I/O
    'Intrinsics',
    'read_rgb',
    'read_depth_png',
    'read_label_png',
    # Types
    'SegmentationResult',
    # Evaluation
    'compute_iou',
    'match_instances_hungarian',
    'evaluate_single_frame',
    'evaluate_dataset',
    # Dataset
    'load_dataset',
]

