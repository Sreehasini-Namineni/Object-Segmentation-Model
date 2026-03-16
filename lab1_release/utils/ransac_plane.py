from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class PlaneModel:
    """
    Represents a plane in 3D space using the implicit equation: n · X + d = 0.
    
    The plane is defined by:
    - n: unit normal vector (shape (3,), ||n|| = 1)
    - d: scalar offset
    
    For any point X on the plane: n · X + d = 0
    """
    n: np.ndarray  # (3,) unit normal vector
    d: float       # scalar offset

    def __post_init__(self):
        """Validate that the normal vector has correct shape and is normalized."""
        if self.n.shape != (3,):
            raise ValueError(f"Normal vector must have shape (3,), got {self.n.shape}")
        norm = np.linalg.norm(self.n)
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"Normal vector must be unit length (||n||=1), got ||n||={norm:.6f}")

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute signed distance from points to the plane.
        
        Positive values: point is on the side of the plane in the direction of n.
        Negative values: point is on the opposite side.
        Zero: point lies on the plane.
        
        Args:
            pts: Array of shape (N, 3) containing N 3D points.
        
        Returns:
            Array of shape (N,) containing signed distances.
        """
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
        return pts @ self.n + self.d

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute absolute (unsigned) distance from points to the plane.
        
        This is the perpendicular distance from each point to the plane surface.
        
        Args:
            pts: Array of shape (N, 3) containing N 3D points.
        
        Returns:
            Array of shape (N,) containing non-negative distances.
        """
        return np.abs(self.signed_distance(pts))


def fit_plane_from_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[PlaneModel]:
    """
    Fit a plane through 3 non-collinear points.
        
    The plane should be computed by:
    1. Creating two vectors from p1 to p2 and p1 to p3
    2. Computing the cross product to get the normal vector
    3. Normalizing the normal vector
    4. Computing the offset d = -n · p1
    
    Args:
        p1, p2, p3: Three 3D points, each with shape (3,).
    
    Returns:
        PlaneModel if the points are non-collinear, None otherwise.
    """
    # Validate input shapes
    for i, pt in enumerate([p1, p2, p3], 1):
        if pt.shape != (3,):
            raise ValueError(f"Point p{i} must have shape (3,), got {pt.shape}")
        
    v1 = p2 - p1
    v2 = p3 - p1
    normal_v = np.cross(v1, v2)

    # colinear check (crossproduct is 0)
    if np.allclose(normal_v, 0):
        return None
    
    normal_v = normal_v / np.linalg.norm(normal_v)
    d = -np.dot(normal_v, p1)

    return PlaneModel(normal_v, d)
    



def refine_plane_svd(pts: np.ndarray) -> Optional[PlaneModel]:
    """
    Least-squares plane fit via SVD (Singular Value Decomposition).
    
    This method finds the best-fit plane by:
    1. Centering the points at the origin
    2. Using SVD to find the direction of least variance (normal to the plane)
    3. The normal is the right singular vector corresponding to the smallest singular value
    
    Args:
        pts: Array of shape (N, 3) containing N 3D points (at least 3 required).
    
    Returns:
        PlaneModel if successful, None if insufficient points or degenerate case.
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
    
    if pts.shape[0] < 3:
        return None
    
    # Center the points at the origin
    centroid = pts.mean(axis=0)
    X = pts - centroid
    
    # SVD: X = U @ S @ Vh
    # The normal is the right singular vector (vh) corresponding to smallest singular value
    # This is the direction of least variance, i.e., perpendicular to the best-fit plane
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]  # Last row of Vh corresponds to smallest singular value
    
    # Normalize (should already be unit, but ensure for numerical stability)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    
    # Compute offset: d = -n · centroid
    d = -float(np.dot(n, centroid))
    
    return PlaneModel(n=n.astype(np.float32), d=float(d))


def ransac_plane(pts: np.ndarray,
                 iters: int = 2000,
                 inlier_thresh: float = 0.008,
                 min_inliers: int = 5000,
                 seed: int = 0) -> Tuple[Optional[PlaneModel], Optional[np.ndarray]]:
    """
    RANSAC (RANdom SAmple Consensus) algorithm for robust plane fitting.
        
    The algorithm should:
    1. Randomly sample 3 points and fit a plane using fit_plane_from_3pts
    2. Count inliers (points within threshold distance of the plane)
    3. Keep the plane with the most inliers
    4. Refine the best plane using SVD on all inliers (refine_plane_svd)
    5. Return None if fewer than min_inliers are found
    
    Args:
        pts: Array of shape (N, 3) containing N 3D points (float32 recommended).
        iters: Maximum number of RANSAC iterations.
        inlier_thresh: Distance threshold for considering a point an inlier.
        min_inliers: Minimum number of inliers required for success.
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (best_plane, inlier_mask) where:
        - best_plane: PlaneModel if successful, None otherwise
        - inlier_mask: Boolean array of shape (N,) indicating inliers, None if failed
    """
    # Validate input
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
    
    if pts.shape[0] < 3:
        return None, None
    
    if inlier_thresh <= 0:
        raise ValueError(f"inlier_thresh must be positive, got {inlier_thresh}")
    
    if min_inliers < 3:
        raise ValueError(f"min_inliers must be at least 3, got {min_inliers}")
    
    max = 0
    inlier_mask = None

    for _ in range(iters):
        
        points = np.random.choice(len(pts), 3, False)
        p1, p2, p3 = (pts[points[0]], pts[points[1]], pts[points[2]])
        plane = fit_plane_from_3pts(p1, p2, p3)
        if plane == None:
            continue

        dist = plane.distance(pts)
        inliers = dist < inlier_thresh
        num = np.sum(inliers)

        if num > max:
            max = num
            inlier_mask = inliers

    if max < min_inliers:
        return None, None
    
    best_plane = refine_plane_svd(pts[inlier_mask])
    return best_plane, inlier_mask
