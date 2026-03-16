from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from scipy import ndimage as ndi
from .ocid_io import Intrinsics
from .ransac_plane import ransac_plane, PlaneModel


def backproject(depth_m: np.ndarray, K: Intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project a depth image to 3D point cloud.

    Args:
        depth_m: (H, W) numpy array of depth values in meters.
        K: Intrinsics object with camera parameters (fx, fy, cx, cy).

    Returns:
        pts: (N, 3) float32 array of 3D points.
        idxs: (N, 2) int32 array of (v, u) indices for each point.
    """
    H, W = depth_m.shape
    valid = np.isfinite(depth_m)
    vs, us = np.nonzero(valid)  # Pixel coordinates
    z = depth_m[vs, us].astype(np.float32)
    x = (us.astype(np.float32) - float(K.cx)) * z / float(K.fx)
    y = (vs.astype(np.float32) - float(K.cy)) * z / float(K.fy)
    pts = np.stack((x, y, z), axis=1)
    idxs = np.stack((vs, us), axis=1)
    return pts, idxs


def table_mask_from_ransac(
    depth_m: np.ndarray,
    K: Intrinsics,
    max_range_m: float,
    ransac_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, PlaneModel | None]:
    """
    Fit a plane to the depth image with RANSAC and return a mask indicating table pixels.

    Args:
        depth_m: (H, W) depth map in meters.
        K: Intrinsics object.
        max_range_m: Ignore depths beyond this range.
        ransac_cfg: Dict with keys 'iters', 'inlier_thresh_m', 'min_inliers', 'seed'.

    Returns:
        table_mask: (H, W) bool array (True for table).
        plane: PlaneModel instance or None.
    """
    H, W = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m < max_range_m)

    if np.count_nonzero(valid) < 3:
        return np.zeros((H, W), dtype=bool), None

    pts, idxs = backproject(depth_m, K)
    # Retain only 3D points within the specified range
    within_range = pts[:, 2] < max_range_m
    pts = pts[within_range]
    idxs = idxs[within_range]

    # Fit plane using RANSAC
    plane, inliers = ransac_plane(
        pts,
        iters=int(ransac_cfg.get("iters", 2000)),
        inlier_thresh=float(ransac_cfg.get("inlier_thresh_m", 0.008)),
        min_inliers=int(ransac_cfg.get("min_inliers", 5000)),
        seed=int(ransac_cfg.get("seed", 0)),
    )
    table_mask = np.zeros((H, W), dtype=bool)
    if plane is None or inliers is None:
        return table_mask, None

    inlier_pixels = idxs[inliers]  # (M, 2), pixel coordinates of inliers
    table_mask[inlier_pixels[:, 0], inlier_pixels[:, 1]] = True
    return table_mask, plane


def foreground_from_plane(
    depth_m: np.ndarray,
    K: Intrinsics,
    plane: PlaneModel,
    max_range_m: float,
    margin_m: float = 0.01,
) -> np.ndarray:
    """
    Segment foreground by keeping points on the camera side of the plane, away from the plane.

    Args:
        depth_m: (H, W) depth map in meters.
        K: camera intrinsics.
        plane: table plane model.
        max_range_m: Ignore depths beyond this range.
        margin_m: Margin away from the plane in meters.

    Returns:
        fg: (H, W) bool mask (True for foreground).
    """
    H, W = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m < max_range_m)
    fg = np.zeros((H, W), dtype=bool)
    if not np.any(valid):
        return fg

    pts, idxs = backproject(depth_m, K)
    within_range = pts[:, 2] < max_range_m
    pts = pts[within_range]
    idxs = idxs[within_range]

    s0 = float(plane.d)  # Plane offset (signed distance at origin)
    sd = plane.signed_distance(pts)  # Signed distances of all valid points

    # Camera-side means same sign as at the origin (0, 0, 0), plus margin.
    camera_side = (sd * s0) > float(margin_m)
    fg_pixels = idxs[camera_side]
    fg[fg_pixels[:, 0], fg_pixels[:, 1]] = True
    return fg


def cleanup_mask(
    binary: np.ndarray,
    open_iters: int = 1,
    close_iters: int = 2,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Clean up a binary mask using morphological operations.

    Args:
        binary: input boolean mask.
        open_iters: number of opening iterations.
        close_iters: number of closing iterations.
        fill_holes: whether to fill holes.

    Returns:
        Cleaned up binary mask.
    """
    st = np.ones((3, 3), dtype=bool)
    m = binary.astype(bool)
    if open_iters > 0:
        m = ndi.binary_opening(m, structure=st, iterations=int(open_iters))
    if close_iters > 0:
        m = ndi.binary_closing(m, structure=st, iterations=int(close_iters))
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    return m


def connected_components(
    mask: np.ndarray,
    min_area_px: int = 800,
    connectivity: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    Compute connected components and filter out small components.

    Args:
        mask: (H, W) binary mask.
        min_area_px: Ignore components smaller than this.
        connectivity: Pixel connectivity (1=4-connected, 2=8-connected).

    Returns:
        labels: (H, W) array of component labels, 0 for background.
        num: number of components (excluding background).
    """
    if connectivity == 1:
        st = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)
    else:
        st = np.ones((3, 3), dtype=bool)

    labels, num = ndi.label(mask > 0, structure=st)

    if num == 0:
        return labels.astype(np.int32), 0

    if min_area_px > 0:
        # Compute area of each component (labels start from 1)
        counts = np.bincount(labels.ravel())
        too_small = counts < int(min_area_px)
        too_small[0] = False  # Always keep background
        # Set labels of small components to 0 (background)
        mask_filtered = too_small[labels]
        labels[mask_filtered] = 0
        labels, num = ndi.label(labels > 0, structure=st)

    return labels.astype(np.int32), int(num)
