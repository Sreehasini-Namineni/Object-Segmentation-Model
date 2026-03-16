from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt


def label_to_color_image(labels: np.ndarray) -> np.ndarray:
    """
    Convert a labels array (H, W) to an RGB color image.
    Background (0) is mapped to black. All other ids get deterministic colors.

    Args:
        labels: (H, W) int32/uint32/uint8 array of instance or semantic labels.

    Returns:
        (H, W, 3) float32 RGB image with values in [0, 1].
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D label map, got shape {labels.shape}")
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    ids = np.unique(labels)
    ids = ids[ids != 0]   # ignore background label 0
    if ids.size == 0:
        return out

    # Assign distinct colors to each id, with deterministic hashing.
    for i in ids:
        # Deterministic hash to color (in [0,1])
        r = ((int(i) * 123457) % 256) / 255.0
        g = ((int(i) * 234569) % 256) / 255.0
        b = ((int(i) * 345679) % 256) / 255.0
        out[labels == i, :] = (r, g, b)
    return out


def overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.55
) -> np.ndarray:
    """
    Overlay color-coded labels over an RGB image.

    Args:
        rgb: (H, W, 3) uint8 or float RGB image.
        labels: (H, W) int label mask (0=background).
        alpha: Blending coefficient for the labels [0, 1]. Higher = more label color.

    Returns:
        Overlay image (H, W, 3), float in [0, 1].
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
    if labels.shape != rgb.shape[:2]:
        raise ValueError(f"labels shape {labels.shape} does not match rgb shape {rgb.shape[:2]}")
    # Normalize RGB to float [0, 1]
    rgb_f = rgb.astype(np.float32)
    if rgb_f.max() > 1.5:
        rgb_f /= 255.0
    color = label_to_color_image(labels)
    mask = labels > 0
    out = rgb_f.copy()
    out[mask] = (1.0 - float(alpha)) * rgb_f[mask] + float(alpha) * color[mask]
    return np.clip(out, 0.0, 1.0)


def save_side_by_side(
    rgb: np.ndarray,
    labels: np.ndarray,
    out_path: Union[str, Path],
    title_left: str = "Original",
    title_right: str = "Segmentation",
    alpha: float = 0.55
) -> None:
    """
    Save a side-by-side visualization of an RGB image and its segmentation.

    Args:
        rgb: Input RGB image (H, W, 3), uint8 or float in [0, 1].
        labels: Label mask (H, W), int.
        out_path: Output file path.
        title_left: Title above the left image.
        title_right: Title above the right image.
        alpha: Blending coefficient for overlaying the segmentation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov = overlay(rgb, labels, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax1, ax2 = axes

    ax1.imshow(np.clip(rgb, 0, 255).astype(np.uint8) if rgb.max() > 1.5 else np.clip(rgb, 0, 1))
    ax1.set_title(title_left)
    ax1.axis('off')

    ax2.imshow(ov)
    ax2.set_title(f"{title_right} (instances={int(np.max(labels))})")
    ax2.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
