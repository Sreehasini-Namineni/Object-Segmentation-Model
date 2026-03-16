"""
Type definitions for segmentation results and diagnostics.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np


class SegmentationResult:
    """
    Result of a segmentation method.
    
    Supports both old (instance_mask/num_instances) and new (labels/num_objects) parameter names.
    """
    
    def __init__(
        self, 
        labels: Optional[np.ndarray] = None,
        num_objects: Optional[int] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        # Alternative parameter names for backward compatibility
        instance_mask: Optional[np.ndarray] = None,
        num_instances: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize with either old or new parameter names."""
        # Handle alternative parameter names
        if instance_mask is not None:
            labels = instance_mask
        if num_instances is not None:
            num_objects = num_instances
        if metadata is not None and diagnostics is None:
            diagnostics = metadata
        
        # Ensure we have the required fields
        if labels is None:
            raise ValueError("Either 'labels' or 'instance_mask' must be provided")
        if num_objects is None:
            raise ValueError("Either 'num_objects' or 'num_instances' must be provided")
        
        # Set attributes
        self.labels = labels.astype(np.int32) if labels.dtype != np.int32 else labels
        self.num_objects = num_objects
        self.diagnostics = diagnostics if diagnostics is not None else {}
        
        # Verify num_objects matches actual labels
        actual_num = int(self.labels.max())
        if actual_num != self.num_objects:
            self.num_objects = actual_num
    
    @property
    def instance_mask(self):
        """Alias for labels (for backward compatibility with eval code)."""
        return self.labels
    
    @property
    def num_instances(self):
        """Alias for num_objects (for backward compatibility)."""
        return self.num_objects
