from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)
        
    Returns:
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_instances_hungarian(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Hungarian (optimal) instance matching by IoU.
    
    Uses scipy.optimize.linear_sum_assignment to find optimal assignment
    that maximizes total IoU (minimizes cost = 1 - IoU).
    
    Args:
        pred_mask: (H, W) prediction mask, 0=background, 1..N=objects
        gt_mask: (H, W) ground truth mask, 0=background, 1..N=objects
        iou_threshold: Minimum IoU to consider a match valid
    
    Returns:
        Dictionary containing:
            - pq: **Primary metric** - Panoptic Quality = SQ × RQ
            - sq: Segmentation Quality (mean IoU over matched pairs)
            - rq: Recognition Quality (TP / (TP + 0.5*FP + 0.5*FN))
            - tp: Number of matched instances (matched pairs with IoU >= threshold)
            - fp: False positives (unmatched predictions)
            - fn: False negatives (unmatched ground truths)
            - num_pred: Number of predicted instances
            - num_gt: Number of ground truth instances
            - matched_pairs: List of (pred_id, gt_id, iou) tuples
            - mean_matched_iou: Mean IoU over matched pairs (same as sq, kept for compatibility)
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 2]  # GT: 0, 1, 2 are background; objects start from 3
    
    num_pred = len(pred_ids)
    num_gt = len(gt_ids)
    
    # Special case: both empty is perfect match
    if num_pred == 0 and num_gt == 0:
        return {
            'pq': 1.0,  # **Primary Metric**: Perfect panoptic quality
            'sq': 1.0,  # Perfect segmentation (no instances to segment)
            'rq': 1.0,  # Perfect recognition (no instances to recognize)
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'num_pred': 0,
            'num_gt': 0,
            'matched_pairs': [],
            'mean_matched_iou': 1.0,  # Perfect: both correctly predict no objects
        }
    
    # One empty, one not: total failure
    if num_pred == 0 or num_gt == 0:
        return {
            'pq': 0.0,  # **Primary Metric**: Total failure
            'sq': 0.0,  # No matched pairs
            'rq': 0.0,  # Total recognition failure
            'tp': 0,
            'fp': num_pred,
            'fn': num_gt,
            'num_pred': num_pred,
            'num_gt': num_gt,
            'matched_pairs': [],
            'mean_matched_iou': 0.0,
        }
    
    # Build IoU cost matrix (num_pred x num_gt)
    # Cost = 1 - IoU for valid matches (IoU >= threshold)
    # For IoU < threshold, set a large penalty to prevent Hungarian from matching them
    LARGE_COST = 1e9  # Large penalty for invalid matches
    cost_matrix = np.full((num_pred, num_gt), LARGE_COST, dtype=np.float64)
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)
    
    for i, pid in enumerate(pred_ids):
        pred_binary = (pred_mask == pid)
        for j, gid in enumerate(gt_ids):
            gt_binary = (gt_mask == gid)
            iou = compute_iou(pred_binary, gt_binary)
            iou_matrix[i, j] = iou
            
            # Only allow matching if IoU >= threshold
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1.0 - iou
            # else: keep LARGE_COST to prevent this match
    
    # Hungarian algorithm - now naturally avoids low-IoU matches
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Collect valid matches (Hungarian already filtered by threshold via cost matrix)
    matched_pairs = []
    matched_ious = []
    matched_pred = set()
    matched_gt = set()
    
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        # Double-check threshold (should always pass due to cost matrix design)
        if iou >= iou_threshold:
            matched_pairs.append((int(pred_ids[r]), int(gt_ids[c]), iou))
            matched_ious.append(iou)
            matched_pred.add(pred_ids[r])
            matched_gt.add(gt_ids[c])
    
    tp = len(matched_pairs)
    fp = num_pred - tp
    fn = num_gt - len(matched_gt)
    
    # Compute PQ metrics
    # SQ = Segmentation Quality = mean IoU over matched pairs (TP)
    sq = np.mean(matched_ious) if matched_ious else 0.0
    
    # RQ = Recognition Quality = TP / (TP + 0.5*FP + 0.5*FN)
    denominator = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denominator if denominator > 0 else 0.0
    
    # PQ = Panoptic Quality = SQ × RQ
    pq = sq * rq
    
    return {
        'pq': pq,  # **Primary Metric**: Panoptic Quality = SQ × RQ
        'sq': sq,  # Segmentation Quality (mean IoU over matched pairs)
        'rq': rq,  # Recognition Quality (TP / (TP + 0.5*FP + 0.5*FN))
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_pred': num_pred,
        'num_gt': num_gt,
        'matched_pairs': matched_pairs,
        'mean_matched_iou': sq,  # Same as SQ, kept for compatibility
    }


def evaluate_single_frame(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a single frame prediction against ground truth.
    
    Primary metric: PQ (Panoptic Quality) = SQ × RQ
    - SQ (Segmentation Quality): mean IoU over matched pairs
    - RQ (Recognition Quality): TP / (TP + 0.5*FP + 0.5*FN)
    
    Args:
        pred_mask: (H, W) prediction mask
        gt_mask: (H, W) ground truth mask
        iou_threshold: Minimum IoU for valid match (default: 0.5)
    
    Returns:
        Dictionary with metrics:
            - pq: Primary metric - Panoptic Quality
            - sq: Segmentation Quality
            - rq: Recognition Quality
            - tp: Number of matched pairs
            - fp: False positives
            - fn: False negatives
            - num_pred: Number of predicted instances
            - num_gt: Number of ground truth instances
            - matched_pairs: List of (pred_id, gt_id, iou) tuples
    """
    return match_instances_hungarian(pred_mask, gt_mask, iou_threshold)


def evaluate_dataset(predictions: list, ground_truths: list, iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate predictions on entire dataset.
    
    Args:
        predictions: List of prediction masks
        ground_truths: List of ground truth masks
        iou_threshold: Minimum IoU for valid match
        
    Returns:
        Dictionary with per-frame results and aggregate PQ metrics
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have same length"
    
    frame_results = []
    all_pqs = []
    all_sqs = []
    all_rqs = []
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = evaluate_single_frame(pred, gt, iou_threshold)
        frame_results.append(metrics)
        all_pqs.append(metrics['pq'])
        all_sqs.append(metrics['sq'])
        all_rqs.append(metrics['rq'])
    
    # Aggregate PQ metrics across all frames
    mean_pq = np.mean(all_pqs) if all_pqs else 0.0
    mean_sq = np.mean(all_sqs) if all_sqs else 0.0
    mean_rq = np.mean(all_rqs) if all_rqs else 0.0
    
    return {
        'frame_results': frame_results,
        'aggregate': {
            'mean_pq': mean_pq,  # Primary metric: average PQ across frames
            'mean_sq': mean_sq,  # Average SQ across frames
            'mean_rq': mean_rq,  # Average RQ across frames
            'num_frames': len(predictions)
        }
    }
