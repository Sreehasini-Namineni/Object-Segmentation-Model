import argparse
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

from solution import solve
from utils import (
    read_rgb, read_depth_png,
    evaluate_single_frame, load_dataset
)
from utils.visualize import save_side_by_side


def load_dataset_frames(dataset_path: Path, config_path: Path = None):
    """
    Load dataset frames from directory using utils.load_dataset().
    
    Expected structure:
        dataset_path/
          rgb/frame_0000.png, ...
          depth/frame_0000.png, ...
          label/frame_0000.png, ...
    
    Camera intrinsics loaded from config.json (default: lab1_release/config.json)
    
    Args:
        dataset_path: Path to dataset directory
        config_path: Path to config.json (default: lab1_release/config.json)
    
    Returns:
        List of frame dictionaries with paths and intrinsics
    """
    return load_dataset(dataset_path, config_path)


def run_evaluation(test_frames, iou_threshold=0.5, visualize=False, vis_output_dir=None):
    """
    Run evaluation on test frames.
    
    Args:
        test_frames: List of frame dictionaries
        iou_threshold: IoU threshold for matching
        visualize: Whether to generate visualizations
        vis_output_dir: Directory to save visualizations
    
    Returns:
        Dictionary with frame results and aggregate PQ metrics
    """
    print(f"\n{'='*60}")
    print(f"Starting Evaluation: {len(test_frames)} frames")
    print(f"Matching Threshold: IoU >= {iou_threshold}")
    print(f"Primary Metric: Panoptic Quality (PQ = SQ × RQ)")
    print(f"{'='*60}\n")
    
    frame_results = []
    all_matched_ious = []
    
    if visualize and vis_output_dir:
        vis_path = Path(vis_output_dir)
        vis_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {vis_path}\n")
    
    for i, frame_info in enumerate(test_frames, 1):
        frame_id = frame_info['frame_id']
        
        try:
            # Load data
            rgb = read_rgb(frame_info['rgb_path'])
            depth = read_depth_png(frame_info['depth_path'])
            gt_label = np.array(Image.open(frame_info['label_path'])).astype(np.int32)
            K = frame_info['K']
            
            # Run student solution
            result = solve(rgb, depth, K)
            pred_mask = result.labels
            
            # Evaluate
            metrics = evaluate_single_frame(pred_mask, gt_label, iou_threshold)
            
            # Store results
            frame_results.append({
                'frame_id': frame_id,
                'pq': metrics['pq'],
                'sq': metrics['sq'],
                'rq': metrics['rq'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'num_pred': metrics['num_pred'],
                'num_gt': metrics['num_gt'],
            })
            
            # Update totals
            all_matched_ious.append(metrics['pq'])
            
            # Print progress
            print(f"[{i:3d}/{len(test_frames)}] {frame_id:15s} | "
                  f"PQ={metrics['pq']:.3f} (SQ={metrics['sq']:.3f} RQ={metrics['rq']:.3f}) | "
                  f"TP={metrics['tp']:2d} FP={metrics['fp']:2d} FN={metrics['fn']:2d}")
            
            # Generate simple visualization (RGB + segmentation)
            if visualize and vis_output_dir:
                vis_file = vis_path / f"{frame_id}_result.png"
                save_side_by_side(
                    rgb, pred_mask, 
                    str(vis_file),
                    title_left="RGB",
                    title_right=f"Segmentation (PQ={metrics['pq']:.3f})"
                )
            
        except Exception as e:
            print(f"Error processing {frame_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute aggregate metrics
    mean_pq = np.mean([r['pq'] for r in frame_results]) if frame_results else 0.0
    mean_sq = np.mean([r['sq'] for r in frame_results]) if frame_results else 0.0
    mean_rq = np.mean([r['rq'] for r in frame_results]) if frame_results else 0.0
    total_tp = sum(r['tp'] for r in frame_results)
    total_fp = sum(r['fp'] for r in frame_results)
    total_fn = sum(r['fn'] for r in frame_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Frames:           {len(frame_results)}")
    print(f"Total TP/FP/FN:         {total_tp}/{total_fp}/{total_fn}")
    print(f"-" * 60)
    print(f"Mean PQ (Primary):      {mean_pq:.4f}  (Panoptic Quality)")
    print(f"Mean SQ:                {mean_sq:.4f}  (Segmentation Quality)")
    print(f"Mean RQ:                {mean_rq:.4f}  (Recognition Quality)")
    print(f"{'='*60}\n")
    
    results = {
        'frame_results': frame_results,
        'aggregate': {
            'mean_pq': mean_pq,
            'mean_sq': mean_sq,
            'mean_rq': mean_rq,
            'num_frames': len(frame_results),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='CS3630 Lab 1: Test Segmentation Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python test.py --dataset /path/to/dataset
  
  # Custom IoU threshold
  python test.py --dataset /path/to/dataset --iou 0.6
  
  # Save results and visualizations
  python test.py --dataset /path/to/dataset --output results.json --visualize --vis-dir visualizations/
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset (must contain rgb/, depth/, label/ subdirectories)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (default: lab1_release/config.json)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for instance matching (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON file (with PQ metrics)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--vis-dir', type=str, default='visualizations',
                        help='Directory to save visualizations (default: visualizations/)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("CS3630 Lab 1: Segmentation Evaluation")
    print(f"{'='*60}")
    print(f"Dataset:      {dataset_path}")
    print(f"Match Thresh: IoU >= {args.iou}")
    print(f"Metric:       Panoptic Quality (PQ)")
    print(f"Visualize:    {args.visualize}")
    
    # Load dataset
    try:
        config_path = Path(args.config) if args.config else None
        test_frames = load_dataset_frames(dataset_path, config_path)
        print(f"Loaded:       {len(test_frames)} frames")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run evaluation
    try:
        results = run_evaluation(
            test_frames, 
            iou_threshold=args.iou,
            visualize=args.visualize,
            vis_output_dir=args.vis_dir if args.visualize else None
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    print(f"\nEvaluation complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
