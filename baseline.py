from __future__ import annotations
from typing import Dict, Any
import numpy as np

from utils.segmentation_types import SegmentationResult
from utils.segmentation import (
    table_mask_from_ransac,
    foreground_from_plane,
    connected_components,
    cleanup_mask
)
from utils.ocid_io import Intrinsics


DEFAULT_PARAMS = {
    'max_range_m': 2.0,
    'min_object_size': 800,
    'ransac': {
        'iters': 2000,
        'inlier_thresh_m': 0.008,
        'min_inliers': 5000,
        'seed': 0
    }
}


def baseline_solve(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: Intrinsics,
    params: Dict[str, Any] = None,
    seed: int = 42
) -> SegmentationResult:
    """
    Baseline segmentation: RANSAC + connected components.
    
    Args:
        rgb: (H, W, 3) RGB image (not used in this baseline)
        depth_m: (H, W) depth in meters
        K: camera intrinsics
        params: configuration dict
        seed: random seed (not used in this method)
    
    Returns:
        SegmentationResult with labels and diagnostics
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    H, W = depth_m.shape

    # Get config parameters
    max_range = float(params.get('max_range_m', 2.0))
    ransac_cfg = params.get('ransac', {})
    min_area = int(params.get('min_object_size', 800))

    table_mask, plane = table_mask_from_ransac(
        depth_m, K, max_range, ransac_cfg
    )

    if plane is None:
        # No table found - return empty segmentation
        return SegmentationResult(
            labels=np.zeros((H, W), dtype=np.int32),
            num_objects=0,
            diagnostics={'error': 'No table plane found'}
        )

    fg_mask = foreground_from_plane(
        depth_m, K, plane, max_range, margin_m=0.01
    )

    fg_clean = cleanup_mask(fg_mask, open_iters=1, close_iters=2)

    labels, num_objects = connected_components(
        fg_clean, min_area_px=min_area, connectivity=2
    )
    
    return SegmentationResult(
        labels=labels,
        num_objects=num_objects,
        diagnostics={
            'table_inliers': int(table_mask.sum()),
            'fg_pixels': int(fg_mask.sum()),
            'plane_normal': plane.n.tolist() if plane else None,
            'plane_offset': float(plane.d) if plane else None
        }
    )
