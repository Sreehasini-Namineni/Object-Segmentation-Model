"""
Dataset loader utility for CS3630 Lab 1
"""

from pathlib import Path
import json
from typing import List, Dict, Any
from .ocid_io import Intrinsics


def load_dataset(dataset_path: Path | str, config_path: Path | str = None) -> List[Dict[str, Any]]:
    """
    Load dataset frames from directory.
    
    Args:
        dataset_path: Path to dataset root containing rgb/, depth/, label/ subdirectories
        config_path: Path to config.json (default: lab1_release/config.json)
    
    Returns:
        List of frame dictionaries with paths and intrinsics
    """
    dataset_path = Path(dataset_path)
    
    # Load config
    if config_path is None:
        # Default to lab1_release/config.json
        config_path = Path(__file__).parent.parent / "config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Get intrinsics from config
    intr_data = config['intrinsics']
    K = Intrinsics(
        fx=intr_data['fx'],
        fy=intr_data['fy'],
        cx=intr_data['cx'],
        cy=intr_data['cy']
    )
    
    # Find directories
    rgb_dir = dataset_path / "rgb"
    depth_dir = dataset_path / "depth"
    label_dir = dataset_path / "label"
    
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    # Get all RGB files
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise ValueError(f"No PNG files found in {rgb_dir}")
    
    frames = []
    for rgb_path in rgb_files:
        frame_id = rgb_path.stem
        depth_path = depth_dir / f"{frame_id}.png"
        label_path = label_dir / f"{frame_id}.png"
        
        if not depth_path.exists():
            print(f"⚠️  Skipping {frame_id}: depth not found")
            continue
        if not label_path.exists():
            print(f"⚠️  Skipping {frame_id}: label not found")
            continue
        
        frames.append({
            'frame_id': frame_id,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'label_path': label_path,
            'K': K
        })
    
    return frames
