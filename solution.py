from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

from utils.segmentation_types import SegmentationResult
from utils.ocid_io import Intrinsics
from utils.segmentation import (
    table_mask_from_ransac,
    foreground_from_plane,
    connected_components,
    cleanup_mask
)

DEFAULT_PARAMS = {
    'max_range_m': 2.0,
    'min_object_size': 800,
    'ransac': {
        'iters': 200,
        'inlier_thresh_m': 0.008,
        'min_inliers': 5000,
        'seed': 0
    }
}

def solve(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: Intrinsics,
    params: Optional[Dict[str, Any]] = None,
) -> SegmentationResult:
    """
    Implement your instance segmentation method here.

    Args:
        rgb: (H, W, 3) uint8 RGB image (RGB order)
        depth_m: (H, W) float32 depth in meters
        K: camera intrinsics (fx, fy, cx, cy)
        params: optional config dict (you can define your own keys)

    Returns:
        SegmentationResult:
            labels: (H, W) int32, 0=background, 1..N=instances
            num_objects: int, number of instances
    """
    if params is None:
        params = DEFAULT_PARAMS

    max_range_m = params.get("max_range_m", 2.0)
    ransac_cfg = params.get("ransac_cfg", {
        "iters": 1000,
        "inlier_thresh_m": 0.008,
        "min_inliers": 5000,
        "seed": 0,
    })
    margin_m = params.get("margin_m", 0.01)
    min_area = params.get("min_area_px", 800)

    # Fit table plane with RANSAC
    table_mask, plane = table_mask_from_ransac(depth_m, K, max_range_m, ransac_cfg)
    H, W = depth_m.shape
    if plane is None:
        return SegmentationResult(
            labels=np.zeros((H, W), dtype=np.int32),
            num_objects=0, 
            diagnostics={'error': 'No table plane found'}
        )

    # Get foreground (points above table)
    fg = foreground_from_plane(depth_m, K, plane, max_range_m, margin_m=margin_m)
    fg = fg & (table_mask == False)

    # Clean up mask
    fg_clean = cleanup_mask(fg)

    # Connected components
    labels, num_objects = connected_components(fg_clean, min_area_px=min_area, connectivity=2)

    #Check for merged object with RGB
    label = np.zeros((H, W), dtype = np.int32)
    curr = 1

    for i in range(1, num_objects + 1):
        mask = labels == i
        area = np.sum(mask)

        if area < 3000:
            label[mask] = curr
            curr += 1

        else:
            pix = rgb[mask].reshape(-1, 3).astype(np.float32)
            clusters, centers = clustering(pix)
            rgb_diff = np.linalg.norm(centers[0] - centers[1])

            if rgb_diff < 35: #colors too similar
                label[mask] = curr
                curr += 1

            else:
                for n in [0, 1]:
                    cluster_m = mask.copy()
                    cluster_m[mask] = clusters == n
                    label[cluster_m] = curr
                    curr += 1

    result = SegmentationResult(label, curr - 1)
    return result

def clustering(pixels, iters=5):

    idx = np.random.choice(len(pixels), 2, False)
    centers = pixels[idx].astype(np.float32)

    for _ in range(iters):
        d0 = np.linalg.norm(pixels - centers[0], axis=1)
        d1 = np.linalg.norm(pixels - centers[1], axis=1)
        labels = d0 < d1

        if np.sum(labels) > 0:
            centers[0] = pixels[labels].mean(axis=0)

        if np.sum(labels == False) > 0:
            centers[1] = pixels[labels == False].mean(axis=0)

    return labels.astype(int), centers
