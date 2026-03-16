from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Intrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float


def read_rgb(path: Union[str, Path]) -> np.ndarray:
    """
    Read an RGB image from the given path and return as a uint8 HxWx3 array.
    
    Args:
        path: Path to the image file.

    Returns:
        NumPy array of shape (H, W, 3), dtype uint8.
    """
    p = Path(path)
    # .convert("RGB") ensures the result is always RGB, even if the file is grayscale or RGBA
    with Image.open(p) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def read_depth_png(
    path: Union[str, Path], 
    depth_scale_m_per_unit: float = 0.001
) -> np.ndarray:
    """
    Read a 16-bit PNG depth image and convert to meters.
    Invalid pixels (value <= 0) are set to np.nan.

    Args:
        path: Path to 16-bit PNG depth image.
        depth_scale_m_per_unit: Scale to get meters from pixel values.

    Returns:
        NumPy array of shape (H, W), dtype float32, values in meters, np.nan for invalid.
    """
    p = Path(path)
    with Image.open(p) as img:
        d = np.array(img).astype(np.float32)
    d[d <= 0] = np.nan
    return d * float(depth_scale_m_per_unit)


def read_label_png(path: Union[str, Path]) -> np.ndarray:
    """
    Read a label PNG image (instance segmentation mask).
    
    Args:
        path: Path to label PNG file.
    
    Returns:
        NumPy array of shape (H, W), dtype int32.
    """
    p = Path(path)
    with Image.open(p) as img:
        return np.array(img, dtype=np.int32)


def validate_pair(
    rgb_path: Union[str, Path], 
    depth_path: Union[str, Path]
) -> Tuple[Path, Path]:
    """
    Validates that the RGB and depth paths exist and returns them as Path objects.

    Args:
        rgb_path: Path to RGB image.
        depth_path: Path to depth image.

    Returns:
        Tuple of (Path to RGB, Path to depth).

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    rp = Path(rgb_path)
    dp = Path(depth_path)
    if not rp.is_file():
        raise FileNotFoundError(f"RGB path not found: {rp}")
    if not dp.is_file():
        raise FileNotFoundError(f"Depth path not found: {dp}")
    return rp, dp


def collect_pairs_from_roots(
    rgb_root: Union[str, Path],
    depth_root: Union[str, Path],
    rgb_ext: str = ".png",
    depth_ext: str = ".png"
) -> List[Tuple[Path, Path]]:
    """
    Given 2 root directories, collects pairs of RGB and depth images by matching filenames (stem).
    Assumes directory structure is flat, or recursively scans for files with the given extensions.

    Args:
        rgb_root: Root directory containing RGB images.
        depth_root: Root directory containing depth images.
        rgb_ext: Extension for RGB images (e.g., '.png').
        depth_ext: Extension for depth images (e.g., '.png').

    Returns:
        List of (Path to RGB, Path to depth), sorted by RGB filename.

    Notes:
        - Only pairs where both RGB and depth image exist (by matching filename stem) are returned.
        - Adapt this function if your OCID arrangement differs.
    """
    rgb_root = Path(rgb_root)
    depth_root = Path(depth_root)

    rgb_files = sorted(rgb_root.rglob(f"*{rgb_ext}"))
    depth_files = list(depth_root.rglob(f"*{depth_ext}"))
    depth_map = {p.stem: p for p in depth_files}

    pairs: List[Tuple[Path, Path]] = []
    for rp in rgb_files:
        dp = depth_map.get(rp.stem)
        if dp is not None:
            pairs.append((rp, dp))
    return pairs


def load_frame(rgb_path, depth_path, label_path):
    """Load a single frame (RGB, depth, label)."""
    rgb = read_rgb(rgb_path)
    depth = read_depth_png(depth_path)
    label = read_label_png(label_path)
    return rgb, depth, label

