#!/usr/bin/env python3
"""
Unified MapQR AV2 Evaluation Script
======================================
Runs MapQR inference on Argoverse 2 with configurable camera inputs AND evaluates with camera-specific FOV clipping.

Features:
1. Run MapQR inference with selectable AV2 cameras (all 7 ring cameras)
2. Evaluate predictions with camera-specific FOV clipping
3. Compare different camera configurations
4. Use EXACT MapTR official evaluation code adapted for AV2

Camera Configuration Options:
- Single camera: --cameras ring_front_center
- Multiple cameras: --cameras ring_front_center ring_rear_left
- All cameras: --cameras all
- Front cameras: --cameras ring_front_center ring_front_left ring_front_right

Usage Examples:
---------------
# Front center camera only
python tools/mapqr_av2_eval_unified.py --cameras ring_front_center

# Front + Rear cameras
python tools/mapqr_av2_eval_unified.py --cameras ring_front_center ring_rear_left ring_rear_right

# All 7 cameras (baseline)
python tools/mapqr_av2_eval_unified.py --cameras all

# Front hemisphere only
python tools/mapqr_av2_eval_unified.py --cameras ring_front_center ring_front_left ring_front_right

# Skip inference if predictions already exist
python tools/mapqr_av2_eval_unified.py --skip-inference --predictions-pkl existing_preds.pkl

# Full BEV evaluation without FOV clipping
python tools/mapqr_av2_eval_unified.py --no-clipping --cameras ring_front_center
"""

import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
import pickle
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# MapTR imports
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import get_root_logger
import os.path as osp

# Geometry and dataset utilities
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString, Point, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

# Import shared camera FOV utilities
from camera_fov_utils import (
    VectorizedLocalMap,
    CameraFOVClipper,
    extract_gt_vectors,
)
from camera_fov_utils_av2 import (
    extract_gt_with_fov_clipping_av2,
    process_predictions_with_fov_clipping_av2
)

# Add MapQR project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MapTR's Chamfer distance implementation
try:
    from projects.mmdet3d_plugin.datasets.map_utils.tpfp_chamfer import custom_polyline_score
except ImportError:
    print("Warning: Could not import MapTR's custom_polyline_score. Using fallback implementation.")
    custom_polyline_score = None


# ==================== CAMERA CONFIGURATION ====================
# Argoverse 2 has 7 ring cameras (from custom_av2_map_converter.py)
CAMERA_MAP = {
    'ring_front_center': 0,
    'ring_front_right': 1,
    'ring_front_left': 2,
    'ring_rear_right': 3,
    'ring_rear_left': 4,
    'ring_side_right': 5,
    'ring_side_left': 6
}

def parse_camera_config(camera_args: List[str]) -> List[int]:
    """
    Parse camera configuration from command line arguments.
    Handles both space-separated and comma-separated camera names.
    
    Args:
        camera_args: List of camera names or 'all'
                    Examples: ['CAM_FRONT', 'CAM_BACK'] or ['CAM_FRONT,CAM_BACK']
    
    Returns:
        List of camera indices (0-5)
    """
    if not camera_args or camera_args[0] == 'all':
        return list(range(7))  # All cameras
    
    # Handle comma-separated camera names
    # Split any comma-separated arguments into individual camera names
    camera_names_flat = []
    for arg in camera_args:
        if ',' in arg:
            # Split on comma and strip whitespace
            camera_names_flat.extend([name.strip() for name in arg.split(',')])
        else:
            camera_names_flat.append(arg)
    
    camera_indices = []
    for cam_name in camera_names_flat:
        if cam_name in CAMERA_MAP:
            camera_indices.append(CAMERA_MAP[cam_name])
        else:
            print(f"Warning: Unknown camera name '{cam_name}'. Valid names: {list(CAMERA_MAP.keys())}")
    
    if not camera_indices:
        print("Warning: No valid cameras specified. Using all cameras.")
        return list(range(7))
    
    return camera_indices


# ==================== INFERENCE CODE (ADAPTED FROM MapTR save_maptr_predictions.py) ====================
def run_mapqr_inference(
    config_path: str,
    checkpoint_path: str,
    output_pkl: str,
    camera_indices: List[int],
    score_thresh: float = 0.0,
    samples_pkl: str = None,
    av2_data_path: str = None
) -> str:
    """
    Run MapQR inference on Argoverse 2 with specified camera configuration.
    Adapted from MapTR save_maptr_predictions.py for AV2
    
    Args:
        config_path: Path to MapQR config file
        checkpoint_path: Path to MapQR checkpoint
        output_pkl: Output pickle file path
        camera_indices: List of camera indices to use (0-6 for AV2)
        score_thresh: Score threshold for filtering predictions
        samples_pkl: Path to override dataset annotation file
        av2_data_path: Path to override AV2 data root
    
    Returns:
        Path to saved predictions pickle file
    """
    print("\n" + "="*80)
    print("STEP 1: Running MapQR Inference")
    print("="*80)
    
    cfg = Config.fromfile(config_path)
    
    # Override dataset paths if provided
    if av2_data_path is not None:
        cfg.data.test.data_root = av2_data_path
        print(f"Overriding AV2 data root to: {av2_data_path}")
    if samples_pkl is not None:
        cfg.data.test.ann_file = samples_pkl
        print(f"Overriding dataset annotation file to: {samples_pkl}")
    
    # Import plugin modules
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)
    
    # Setup CUDA
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    
    # Setup logger
    logger = get_root_logger()
    logger.info('Building dataset...')
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    
    # ✨ NEW: Patch image paths to use correct data_root if needed (Ported from StreamMapNet)
    # Get data_root from dataset config
    data_root = cfg.data.test.get('data_root', '')
    if not data_root:
        # Try to get from dataset object
        if hasattr(dataset, 'data_root'):
            data_root = dataset.data_root
        elif hasattr(dataset, 'nusc') and hasattr(dataset.nusc, 'dataroot'):
            data_root = dataset.nusc.dataroot
    
    if data_root:
        from projects.mmdet3d_plugin.datasets.av2_offlinemap_dataset import CustomAV2OfflineLocalMapDataset
        original_get_data_info = CustomAV2OfflineLocalMapDataset.get_data_info
        
        def patched_get_data_info_with_data_root(self, index):
            """Patched version that fixes image paths with correct data_root."""
            input_dict = original_get_data_info(self, index)
            
            # Fix image paths to use correct data_root
            fixed_img_filenames = []
            # In MapTR's get_data_info, img_filename is a list of paths
            if 'img_filename' in input_dict:
                for img_path in input_dict['img_filename']:
                    if img_path:
                        # Normalize the path
                        img_path_abs = os.path.abspath(img_path) if not os.path.isabs(img_path) else img_path
                        
                        # If path doesn't exist, try to fix it using data_root
                        if not os.path.exists(img_path_abs):
                            # Remove leading ./ or relative path components
                            img_path_clean = img_path.lstrip('./')
                            
                            # Check if it contains 'nuscenes/' to extract the relative part
                            if 'nuscenes/' in img_path_clean:
                                # Extract the part after 'nuscenes/'
                                parts = img_path_clean.split('nuscenes/', 1)
                                if len(parts) > 1:
                                    # Reconstruct with correct data_root
                                    fixed_path = os.path.join(data_root, parts[1])
                                    if os.path.exists(fixed_path):
                                        fixed_img_filenames.append(fixed_path)
                                        continue
                            
                            # Try joining with data_root/samples/ directly
                            # Path format: ./data/nuscenes/samples/CAM_FRONT/...
                            # We want: data_root/samples/CAM_FRONT/...
                            if 'samples/' in img_path_clean:
                                parts = img_path_clean.split('samples/', 1)
                                if len(parts) > 1:
                                    fixed_path = os.path.join(data_root, 'samples', parts[1])
                                    if os.path.exists(fixed_path):
                                        fixed_img_filenames.append(fixed_path)
                                        continue
                            
                            # Last resort: try joining with data_root directly
                            fixed_path = os.path.join(data_root, img_path_clean)
                            if os.path.exists(fixed_path):
                                fixed_img_filenames.append(fixed_path)
                            else:
                                # Keep original path if fixed path doesn't exist
                                logger.warning(f"Could not find image at {img_path} or {fixed_path}")
                                fixed_img_filenames.append(img_path)
                        else:
                            # Path exists, use it as-is
                            fixed_img_filenames.append(img_path_abs)
                    else:
                        fixed_img_filenames.append(img_path)
                
                input_dict['img_filename'] = fixed_img_filenames
                
            return input_dict
        
        CustomAV2OfflineLocalMapDataset.get_data_info = patched_get_data_info_with_data_root
        logger.info(f'Patched CustomAV2OfflineLocalMapDataset.get_data_info to fix image paths with data_root: {data_root}')

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info(f'Built dataset with {len(dataset)} samples')
    
    # Build model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    logger.info(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    logger.info('Model loaded and ready')
    
    # Print camera configuration
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    logger.info(f'\nCamera configuration:')
    logger.info(f'  Active cameras ({len(camera_indices)}/6): {", ".join(camera_names)}')
    if len(camera_indices) < 7:
        inactive_names = [name for name, idx in CAMERA_MAP.items() if idx not in camera_indices]
        logger.info(f'  Inactive cameras (zeroed out): {", ".join(inactive_names)}')
    
    # Storage for predictions
    predictions = {}
    
    # Run inference
    logger.info('\nRunning inference...')
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        try:
            img_metas = data['img_metas'][0].data[0]
            
            # DEBUG: Print what's actually in img_metas on first sample
            if i == 0:
                logger.info(f"DEBUG: img_metas[0] keys: {img_metas[0].keys()}")
                if 'scene_token' in img_metas[0]:
                    logger.info(f"DEBUG: scene_token (log_id) = {img_metas[0]['scene_token']}")
                if 'timestamp' in img_metas[0]:
                    logger.info(f"DEBUG: timestamp = {img_metas[0]['timestamp']}")
            
            # Get sample token - AV2 format: log_id + '_' + timestamp
            # From converter line 212: token=str(log_id+'_'+str(ts))
            # Note: log_id is passed as 'scene_token' in img_metas (see av2_dataset.py line 957)
            if 'scene_token' in img_metas[0] and 'timestamp' in img_metas[0]:
                # AV2: construct token as log_id_timestamp
                log_id = img_metas[0]['scene_token']  # scene_token = log_id
                timestamp = img_metas[0]['timestamp']
                sample_token = f"{log_id}_{timestamp}"
                if i == 0:
                    logger.info(f"DEBUG: Constructed token = '{sample_token}'")
            elif 'sample_idx' in img_metas[0]:
                sample_token = img_metas[0]['sample_idx']
            else:
                pts_filename = img_metas[0]['pts_filename']
                sample_token = osp.basename(pts_filename).replace('__LIDAR_TOP__', '_').split('.')[0]
            
            # Zero out inactive cameras
            if len(camera_indices) < 7 and 'img' in data and data['img'][0] is not None:
                imgs = data['img'][0].data[0]  # Shape: [B, N_views, C, H, W] or [N_views, C, H, W]
                
                if i == 0:
                    logger.info(f"DEBUG: Image tensor shape: {imgs.shape}")
                    logger.info(f"DEBUG: Keeping camera indices {camera_indices}")
                
                # Create mask: ones for active views, zeros for inactive
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    mask = torch.zeros_like(imgs)
                    for cam_idx in camera_indices:
                        mask[:, cam_idx, :, :, :] = 1
                    data['img'][0].data[0] = imgs * mask
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    mask = torch.zeros_like(imgs)
                    for cam_idx in camera_indices:
                        mask[cam_idx, :, :, :] = 1
                    data['img'][0].data[0] = imgs * mask
                else:
                    logger.warning(f"Unexpected image tensor shape: {imgs.shape}")
            
            # Run inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Extract predictions (EXACT format from MapTR)
            result_dic = result[0]['pts_bbox']
            pred_boxes = result_dic['boxes_3d']
            pred_scores = result_dic['scores_3d']
            pred_labels = result_dic['labels_3d']
            pred_vectors = result_dic['pts_3d']
            
            # Filter by score threshold
            keep = pred_scores > score_thresh
            pred_vectors = pred_vectors[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            
            # Convert to numpy
            if torch.is_tensor(pred_vectors):
                pred_vectors = pred_vectors.cpu().numpy()
            if torch.is_tensor(pred_labels):
                pred_labels = pred_labels.cpu().numpy()
            if torch.is_tensor(pred_scores):
                pred_scores = pred_scores.cpu().numpy()
            
            # Store predictions
            predictions[sample_token] = {
                'vectors': pred_vectors,
                'labels': pred_labels,
                'scores': pred_scores
            }
            
        except Exception as e:
            logger.warning(f'Error processing sample {i}: {str(e)}')
        
        prog_bar.update()
    
    # Save predictions
    logger.info(f'\nSaving {len(predictions)} predictions to {output_pkl}...')
    with open(output_pkl, 'wb') as f:
        pickle.dump(predictions, f)
    
    logger.info('✓ Inference complete!')
    return output_pkl


# ==================== EVALUATION CODE (EXACT COPY FROM evaluate_with_fov_clipping_standalone.py) ====================
class CameraSpecificEvaluator:
    """
    Evaluator using EXACT MapTR official evaluation method.
    EXACT COPY from evaluate_with_fov_clipping_standalone.py
    
    Applies camera-specific FOV clipping and rotation to both GT and predictions,
    then evaluates using MapTR's official matching algorithm.
    """
    
    def __init__(
        self,
        av2_data_path: str,
        pc_range: List[float] = None,
        num_sample_pts: int = 100,
        thresholds_chamfer: List[float] = None,
        camera_names: List[str] = None,
        num_workers: int = 1
    ):
        """
        Args:
            av2_data_path: Path to AV2 dataset
            pc_range: BEV range [-x, -y, -z, x, y, z]
            num_sample_pts: Number of points to resample vectors to (MUST match training: 100)
            thresholds_chamfer: Chamfer distance thresholds (MapTR uses [0.5, 1.0, 1.5])
            camera_names: List of camera names to evaluate
            num_workers: Number of workers for parallel processing (e.g., buffering geometries)
        """
        self.av2_data_path = av2_data_path
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.num_sample_pts: int = num_sample_pts
        self.thresholds_chamfer = thresholds_chamfer or [0.5, 1.0, 1.5]
        self.camera_names = camera_names or ['CAM_FRONT']
        self.num_workers = num_workers # NEW: Store num_workers for buffering step
        
        # Calculate patch size from pc_range
        self.patch_size = (self.pc_range[4] - self.pc_range[1], self.pc_range[3] - self.pc_range[0])
        
        # Accumulators
        self.reset()
    
    def reset(self):
        """Reset accumulators"""
        self.predictions_per_camera = {cam: [] for cam in self.camera_names}
        self.ground_truths_per_camera = {cam: [] for cam in self.camera_names}
        self.num_samples_processed = 0
    
    def resample_vector_linestring(self, vector: np.ndarray, num_sample: int) -> np.ndarray:
        """
        Resample a vector to fixed number of points using LineString interpolation.
        Matches original MapTR implementation with support for both 2D and 3D vectors.
        
        Original MapQR design:
        - NuScenes: code_size=2 (2D evaluation)
        - AV2: code_size=3 (3D evaluation with Z coordinate)
        
        Auto-detects dimensionality from input and preserves it.
        """
        if len(vector) < 2:
            if num_sample > len(vector):
                # Detect dimensionality for padding
                dim = vector.shape[1] if len(vector.shape) > 1 else 2
                padding = np.zeros((num_sample - len(vector), dim))
                return np.vstack([vector, padding])
            return vector
        
        # Detect dimensionality: 2D or 3D
        is_3d = vector.shape[1] == 3 if len(vector.shape) > 1 else False
        
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances])
        
        # Reshape based on detected dimensionality
        if is_3d:
            return sampled_points.reshape(-1, 3)  # 3D for AV2 (code_size=3)
        else:
            return sampled_points.reshape(-1, 2)  # 2D for NuScenes (code_size=2)
    
    def process_gt_with_fov_clipping(
        self,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and process GT vectors for a specific camera.
        Uses shared extract_gt_with_fov_clipping_av2() for 100% identical logic.
        """
        gt_data = extract_gt_with_fov_clipping_av2(
            sample_info=sample_info,
            av2_data_path=self.av2_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            fixed_num=20,
            apply_clipping=apply_clipping
        )
        
        vectors = gt_data['vectors']
        gt_labels = gt_data['labels']
        
        if len(vectors) == 0:
            return np.array([]), np.array([])
        
        # Resample to num_sample_pts (100) for evaluation
        final_vectors = []
        for vector in vectors:
            if len(vector) >= 2:
                resampled_vec = self.resample_vector_linestring(vector, self.num_sample_pts)
                final_vectors.append(resampled_vec)
        
        if len(final_vectors) == 0:
            return np.array([]), np.array([])
        
        return np.array(final_vectors), np.array(gt_labels)
    
    def process_predictions_with_fov_clipping_and_rotation(
        self,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply optional FOV clipping AND camera-centric rotation to predictions.
        Uses shared process_predictions_with_fov_clipping_av2() for 100% identical logic.
        """
        if len(pred_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        vectors, labels, scores = process_predictions_with_fov_clipping_av2(
            pred_vectors=pred_vectors,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            sample_info=sample_info,
            av2_data_path=self.av2_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            apply_clipping=apply_clipping
        )
        
        if len(vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Resample to num_sample_pts (100) for evaluation
        final_vectors = []
        for vector in vectors:
            if len(vector) >= 2:
                resampled_vec = self.resample_vector_linestring(vector, self.num_sample_pts)
                final_vectors.append(resampled_vec)
        
        if len(final_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(final_vectors), np.array(labels), np.array(scores)
    
    def compute_chamfer_distance_matrix_maptr_official(self,
                                                        pred_vectors: np.ndarray,
                                                        gt_vectors: np.ndarray,
                                                        linewidth: float = 2.0,
                                                        pred_geometries: List = None,
                                                        gt_geometries: List = None) -> np.ndarray:
        """
        Compute Chamfer Distance matrix using EXACT MapTR official method.
        EXACT copy from MapTR's tpfp_chamfer.py:custom_polyline_score()
        
        OPTIMIZED: Accepts pre-computed geometries to avoid redundant buffering.
        
        Returns NEGATIVE CD values (higher = better match).
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        if num_preds == 0 or num_gts == 0:
            return np.full((num_preds, num_gts), -100.0)
        
        # Use pre-computed geometries if provided, otherwise create them
        if pred_geometries is None:
            pred_lines_shapely = [
                LineString(pred_vectors[i]).buffer(
                    linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                for i in range(num_preds)
            ]
        else:
            pred_lines_shapely = pred_geometries
        
        if gt_geometries is None:
            gt_lines_shapely = [
                LineString(gt_vectors[i]).buffer(
                    linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                for i in range(num_gts)
            ]
        else:
            gt_lines_shapely = gt_geometries
        
        # STRtree spatial indexing
        tree = STRtree(pred_lines_shapely)
        
        # Initialize with -100.0 for non-intersecting pairs
        cd_matrix = np.full((num_preds, num_gts), -100.0)
        
        # Compute CD only for intersecting buffered geometries
        for i, gt_line in enumerate(gt_lines_shapely):
            query_result = tree.query(gt_line)
            
            # Handle both Shapely 1.x and 2.x
            if len(query_result) > 0 and isinstance(query_result[0], (int, np.integer)):
                # Shapely 2.x: returns indices
                for pred_idx in query_result:
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()
                        valid_ba = dist_mat.min(axis=0).mean()
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
            else:
                # Shapely 1.x: returns geometries
                for pred_idx in range(num_preds):
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()
                        valid_ba = dist_mat.min(axis=0).mean()
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
        
        return cd_matrix
    
    def precompute_shapely_geometries(self,
                                     vectors: np.ndarray,
                                     linewidth: float = 2.0) -> List:
        """
        Pre-compute buffered Shapely geometries for a set of vectors.
        This is a performance optimization to avoid recomputing geometries
        for each threshold evaluation.
        
        Args:
            vectors: Array of shape (N, num_points, 2)
            linewidth: Buffer width for LineString
        
        Returns:
            List of buffered Shapely polygons
        """
        if len(vectors) == 0:
            return []
        
        geometries = [
            LineString(vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(len(vectors))
        ]
        return geometries
    
    def compute_chamfer_distance_torch(self,
                                       pred_vectors: np.ndarray,
                                       gt_vectors: np.ndarray) -> float:
        """
        Compute Chamfer Distance for monitoring (returns POSITIVE distance).
        Matches original MapQR: uses dimensionality based on input vectors.
        
        - NuScenes: 2D Chamfer distance (code_size=2)
        - AV2: 3D Chamfer distance (code_size=3)
        """
        if len(pred_vectors) == 0 or len(gt_vectors) == 0:
            return float('inf')
        
        # AV2 MUST use 3D vectors (code_size=3)
        # Validate dimensionality to catch upstream bugs
        pred_dim = pred_vectors.shape[-1] if len(pred_vectors.shape) > 2 else 2
        gt_dim = gt_vectors.shape[-1] if len(gt_vectors.shape) > 2 else 2
        
        # Raise error if 2D vectors are passed (indicates bug in pipeline)
        if pred_dim != 3 or gt_dim != 3:
            raise ValueError(
                f"AV2 evaluation requires 3D vectors (code_size=3), but got:\n"
                f"  Pred vectors: shape={pred_vectors.shape}, dim={pred_dim}\n"
                f"  GT vectors: shape={gt_vectors.shape}, dim={gt_dim}\n"
                f"This indicates a bug in the upstream processing pipeline.\n"
                f"Check resample_vector_linestring() and FOV clipping utilities."
            )
        
        pred_points = pred_vectors.reshape(-1, 3)
        gt_points = gt_vectors.reshape(-1, 3)
        
        # Compute Euclidean distance (automatically uses correct dimensionality)
        dist_matrix = distance.cdist(pred_points, gt_points, 'euclidean')
        
        valid_ab = dist_matrix.min(axis=1).mean()
        valid_ba = dist_matrix.min(axis=0).mean()
        
        chamfer_dist = (valid_ab + valid_ba) / 2.0
        
        return chamfer_dist
    
    def accumulate_sample(
        self,
        sample_info: Dict,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        apply_clipping: bool = True
    ):
        """
        Process one sample and accumulate results for each camera.
        """
        for camera_name in self.camera_names:
            # Extract GT vectors with FOV clipping
            gt_vectors, gt_labels = self.process_gt_with_fov_clipping(
                sample_info, camera_name, apply_clipping
            )
            
            # Process predictions with FOV clipping
            pred_vectors_clipped, pred_labels_clipped, pred_scores_clipped = \
                self.process_predictions_with_fov_clipping_and_rotation(
                    pred_vectors, pred_labels, pred_scores,
                    sample_info, camera_name, apply_clipping
                )
            
            # Debug logging for first sample
            if len(self.predictions_per_camera[camera_name]) == 0:
                print(f"\n[DEBUG] First sample processing for camera: {camera_name}")
                print(f"  Input predictions: {len(pred_vectors)} vectors")
                print(f"  After FOV+rotation: {len(pred_vectors_clipped)} vectors")
                print(f"  GT vectors: {len(gt_vectors)} vectors")
            
            # Store for this camera
            self.predictions_per_camera[camera_name].append({
                'vectors': pred_vectors_clipped,
                'labels': pred_labels_clipped,
                'scores': pred_scores_clipped
            })
            
            self.ground_truths_per_camera[camera_name].append({
                'vectors': gt_vectors,
                'labels': gt_labels
            })
        
        self.num_samples_processed += 1
    
    def match_predictions_to_gt_maptr_official(self,
                                               pred_vectors: np.ndarray,
                                               pred_scores: np.ndarray,
                                               gt_vectors: np.ndarray,
                                               threshold: float,
                                               pred_geometries: List = None,
                                               gt_geometries: List = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match predictions to GT using MapTR's EXACT OFFICIAL method.
        EXACT copy from MapTR's tpfp.py:custom_tpfp_gen()
        
        OPTIMIZED: Accepts pre-computed geometries to avoid redundant buffering.
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        
        if num_gts == 0:
            fp[:] = 1
            return tp, fp
        
        if num_preds == 0:
            return tp, fp
        
        # Convert threshold to NEGATIVE
        if threshold > 0:
            threshold = -threshold
        
        # Compute CD matrix (with optional pre-computed geometries)
        cd_matrix = self.compute_chamfer_distance_matrix_maptr_official(
            pred_vectors, gt_vectors, linewidth=2.0,
            pred_geometries=pred_geometries, gt_geometries=gt_geometries)
        
        # Find best matching GT for each prediction
        matrix_max = cd_matrix.max(axis=1)
        matrix_argmax = cd_matrix.argmax(axis=1)
        
        # Sort by confidence (descending)
        sort_inds = np.argsort(-pred_scores)
        
        # Track matched GTs
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        # Greedy matching
        for i in sort_inds:
            if matrix_max[i] >= threshold:
                matched_gt = matrix_argmax[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        return tp, fp
    
    def compute_ap_area_based(self,
                              recalls: np.ndarray,
                              precisions: np.ndarray) -> float:
        """
        Compute Average Precision using area under PR curve.
        """
        mrec = np.concatenate([[0], recalls, [1]])
        mpre = np.concatenate([[0], precisions, [0]])
        
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        
        return float(ap)
    
    def compute_ap_for_class(
        self,
        pred_vectors_list: List[np.ndarray],
        pred_scores_list: List[np.ndarray],
        gt_vectors_list: List[np.ndarray],
        threshold: float,
        class_name: str = "",
        pred_geoms_list: List = None,
        gt_geoms_list: List = None
    ) -> Tuple[float, float]:
        """
        Compute AP and average CD for a single class at given threshold.
        
        Args:
            pred_geoms_list: Optional pre-computed prediction geometries (avoids redundant buffering)
            gt_geoms_list: Optional pre-computed GT geometries (avoids redundant buffering)
        """
        num_gts = sum(len(gts) for gts in gt_vectors_list)
        
        if num_gts == 0:
            return 0.0, float('inf')
        
        all_tp = []
        all_fp = []
        all_scores = []
        chamfer_distances_per_sample = []
        
        # Use pre-computed geometries if provided, otherwise compute them
        if pred_geoms_list is None or gt_geoms_list is None:
            pred_geoms_list = []
            gt_geoms_list = []
            
            desc = f"Pre-computing geometries ({class_name})" if class_name else "Pre-computing geometries"
            for pred_vecs, gt_vecs in tqdm(zip(pred_vectors_list, gt_vectors_list), 
                                           total=len(pred_vectors_list),
                                           desc=desc,
                                           leave=False,
                                           disable=len(pred_vectors_list) < 100):
                if len(pred_vecs) > 0:
                    pred_geoms = self.precompute_shapely_geometries(pred_vecs, linewidth=2.0)
                else:
                    pred_geoms = []
                
                if len(gt_vecs) > 0:
                    gt_geoms = self.precompute_shapely_geometries(gt_vecs, linewidth=2.0)
                else:
                    gt_geoms = []
                
                pred_geoms_list.append(pred_geoms)
                gt_geoms_list.append(gt_geoms)

        
        # Match predictions using pre-computed geometries
        iterator = zip(pred_vectors_list, pred_scores_list, gt_vectors_list, 
                      pred_geoms_list, gt_geoms_list)
        
        for pred_vecs, pred_scores, gt_vecs, pred_geoms, gt_geoms in iterator:
            if len(pred_vecs) == 0:
                continue
            
            if len(gt_vecs) == 0:
                all_tp.append(np.zeros(len(pred_vecs), dtype=np.float32))
                all_fp.append(np.ones(len(pred_vecs), dtype=np.float32))
                all_scores.append(pred_scores)
                continue
            
            # Match predictions to GT (using pre-computed geometries)
            tp, fp = self.match_predictions_to_gt_maptr_official(
                pred_vecs, pred_scores, gt_vecs, threshold,
                pred_geometries=pred_geoms, gt_geometries=gt_geoms)
            
            all_tp.append(tp)
            all_fp.append(fp)
            all_scores.append(pred_scores)
            
            # Compute chamfer distance
            cd_sample = self.compute_chamfer_distance_torch(pred_vecs, gt_vecs)
            chamfer_distances_per_sample.append(cd_sample)
        
        if len(all_tp) == 0:
            return 0.0, float('inf')
        
        # Concatenate all predictions
        all_tp = np.concatenate(all_tp)
        all_fp = np.concatenate(all_fp)
        all_scores = np.concatenate(all_scores)
        
        # Sort by confidence
        sort_inds = np.argsort(-all_scores)
        tp = all_tp[sort_inds]
        fp = all_fp[sort_inds]
        
        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall
        eps = np.finfo(np.float32).eps
        recalls = tp_cumsum / np.maximum(num_gts, eps)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        
        # Compute AP
        ap = self.compute_ap_area_based(recalls, precisions)
        
        # Average CD
        avg_cd = np.mean(chamfer_distances_per_sample) if chamfer_distances_per_sample else float('inf')
        
        return ap, avg_cd
    

# ==================== PARALLEL GEOMETRY BUFFERING WORKER ====================
def buffer_geometries_worker(pred_vectors: np.ndarray, gt_vectors: np.ndarray, linewidth: float = 2.0):
    """
    Worker function to buffer geometries for one sample in parallel.
    Returns buffered shapely geometries for predictions and GT.
    """
    from shapely.geometry import LineString
    from shapely.geometry import CAP_STYLE, JOIN_STYLE
    
    def buffer_vectors(vectors):
        if len(vectors) == 0:
            return []
        geometries = []
        for vec in vectors:
            if len(vec) >= 2:
                line = LineString(vec)
                buffered = line.buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                geometries.append(buffered)
            else:
                geometries.append(None)
        return geometries
    
    pred_geoms = buffer_vectors(pred_vectors)
    gt_geoms = buffer_vectors(gt_vectors)
    
    return pred_geoms, gt_geoms


def _camera_specific_evaluator_evaluate(self) -> Dict:
        """
        Compute final metrics across all cameras and classes.
        """
        results = {}
        class_names = ['divider', 'ped_crossing', 'boundary']
        
        print(f"\nComputing metrics for {len(self.camera_names)} camera(s), {len(class_names)} classes, {len(self.thresholds_chamfer)} thresholds...")
        
        for camera_name in self.camera_names:
            camera_results = {}
            all_aps = []
            
            camera_preds = self.predictions_per_camera[camera_name]
            camera_gts = self.ground_truths_per_camera[camera_name]
            
            # PRE-COMPUTE GEOMETRIES ONCE PER CAMERA (not per class!)
            # This avoids redundant buffering: 7 cameras × 3 classes × 3 thresholds = 63 runs
            # Optimized to: 7 cameras = 7 runs (9x speedup!)
            # NOW PARALLELIZED: Use multiprocessing for additional 8x speedup
            print(f"Pre-computing geometries for {camera_name}...")
            
            # Prepare data for parallel buffering
            buffer_tasks = [(pred_data['vectors'], gt_data['vectors']) 
                           for pred_data, gt_data in zip(camera_preds, camera_gts)]
            
            # Use multiprocessing to buffer geometries in parallel
            num_workers = self.num_workers  # Use configured num_workers
            if num_workers > 1:
                # Clamp to cpu_count() to avoid excessive overhead if user requests too many
                num_workers = min(num_workers, cpu_count())
            
            if num_workers > 1:
                print(f"  Using {num_workers} parallel workers for geometry buffering...")
                with Pool(processes=num_workers) as pool:
                    geom_results = list(tqdm(
                        pool.starmap(buffer_geometries_worker, buffer_tasks),
                        total=len(buffer_tasks),
                        desc=f"Buffering geometries ({camera_name})",
                        leave=False
                    ))
                all_pred_geoms = [r[0] for r in geom_results]
                all_gt_geoms = [r[1] for r in geom_results]
            else:
                # Serial fallback
                all_pred_geoms = []
                all_gt_geoms = []
                for pred_data, gt_data in tqdm(zip(camera_preds, camera_gts),
                                              total=len(camera_preds),
                                              desc=f"Buffering geometries ({camera_name})",
                                              leave=False):
                    if len(pred_data['vectors']) > 0:
                        pred_geoms = self.precompute_shapely_geometries(pred_data['vectors'], linewidth=2.0)
                    else:
                        pred_geoms = []
                    
                    if len(gt_data['vectors']) > 0:
                        gt_geoms = self.precompute_shapely_geometries(gt_data['vectors'], linewidth=2.0)
                    else:
                        gt_geoms = []
                    
                    all_pred_geoms.append(pred_geoms)
                    all_gt_geoms.append(gt_geoms)
            
            # Evaluate each class
            for class_id, class_name in enumerate(class_names):
                class_results = {}
                
                # Extract predictions and GT for this class
                pred_vectors_list = []
                pred_scores_list = []
                gt_vectors_list = []
                
                # Also extract the corresponding pre-computed geometries for this class
                pred_geoms_for_class = []
                gt_geoms_for_class = []
                
                for pred_data, gt_data, pred_geoms, gt_geoms in zip(camera_preds, camera_gts,
                                                                     all_pred_geoms, all_gt_geoms):
                    pred_mask = pred_data['labels'] == class_id
                    gt_mask = gt_data['labels'] == class_id
                    
                    pred_vectors_list.append(pred_data['vectors'][pred_mask])
                    pred_scores_list.append(pred_data['scores'][pred_mask])
                    gt_vectors_list.append(gt_data['vectors'][gt_mask])
                    
                    # Extract geometries for this class using the same mask
                    if len(pred_geoms) > 0:
                        pred_geoms_for_class.append([pred_geoms[i] for i in range(len(pred_geoms)) if pred_mask[i]])
                    else:
                        pred_geoms_for_class.append([])
                    
                    if len(gt_geoms) > 0:
                        gt_geoms_for_class.append([gt_geoms[i] for i in range(len(gt_geoms)) if gt_mask[i]])
                    else:
                        gt_geoms_for_class.append([])
                
                # Compute AP at each threshold
                # Geometries are pre-computed ONCE per camera and reused across all thresholds and classes
                avg_cd = None
                for threshold in self.thresholds_chamfer:
                    ap, cd = self.compute_ap_for_class(
                        pred_vectors_list, pred_scores_list, gt_vectors_list, 
                        threshold, class_name=class_name,
                        pred_geoms_list=pred_geoms_for_class,
                        gt_geoms_list=gt_geoms_for_class)
                    
                    class_results[f'AP@{threshold}m'] = ap
                    all_aps.append(ap)
                    
                    if avg_cd is None:
                        avg_cd = cd
                
                class_results['avg_chamfer_distance'] = avg_cd if avg_cd is not None else float('inf')
                
                camera_results[class_name] = class_results
            
            # Compute mAP
            camera_results['mAP'] = np.mean(all_aps) if all_aps else 0.0
            
            results[camera_name] = camera_results
        
        # Compute average across cameras if multiple
        if len(self.camera_names) > 1:
            avg_results = {}
            
            for class_name in class_names:
                class_avg = {}
                for threshold in self.thresholds_chamfer:
                    threshold_key = f'AP@{threshold}m'
                    aps = [results[cam][class_name][threshold_key] 
                           for cam in self.camera_names 
                           if class_name in results[cam]]
                    class_avg[threshold_key] = np.mean(aps) if aps else 0.0
                
                cds = [results[cam][class_name]['avg_chamfer_distance'] 
                      for cam in self.camera_names 
                      if class_name in results[cam] and results[cam][class_name]['avg_chamfer_distance'] != float('inf')]
                class_avg['avg_chamfer_distance'] = np.mean(cds) if cds else float('inf')
                
                avg_results[class_name] = class_avg
            
            all_camera_maps = [results[cam]['mAP'] for cam in self.camera_names]
            avg_results['mAP'] = np.mean(all_camera_maps) if all_camera_maps else 0.0
            
            results['AVERAGE'] = avg_results
        
        return results


# Attach evaluate method to the class
CameraSpecificEvaluator.evaluate = _camera_specific_evaluator_evaluate


# ==================== PARALLEL PROCESSING WORKER ====================
def process_sample_worker_av2(
    sample_info: Dict,
    pred_data: Dict,
    av2_data_path: str,
    pc_range: List[float],
    num_sample_pts: int,
    camera_names: List[str],
    apply_clipping: bool
) -> Dict:
    """
    Worker function to process a single AV2 sample in parallel.
    Returns processed GT and predictions for all cameras.
    
    This function is picklable and can be used with multiprocessing.Pool
    """
    from camera_fov_utils_av2 import extract_gt_with_fov_clipping_av2, process_predictions_with_fov_clipping_av2
    from shapely.geometry import LineString
    
    def resample_vector(vector: np.ndarray, num_sample: int) -> np.ndarray:
        """Resample vector preserving dimensionality (2D or 3D)"""
        if len(vector) < 2:
            return vector
        
        # Detect dimensionality
        is_3d = vector.shape[1] == 3 if len(vector.shape) > 1 else False
        
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances])
        
        # Preserve dimensionality
        if is_3d:
            return sampled_points.reshape(-1, 3)
        else:
            return sampled_points.reshape(-1, 2)
    
    results_per_camera = {}
    
    for camera_name in camera_names:
        # Process GT
        gt_data = extract_gt_with_fov_clipping_av2(
            sample_info=sample_info,
            av2_data_path=av2_data_path,
            pc_range=pc_range,
            camera_name=camera_name,
            fixed_num=20,
            apply_clipping=apply_clipping
        )
        
        gt_vectors = gt_data['vectors']
        gt_labels = gt_data['labels']
        
        # Resample GT to num_sample_pts
        if len(gt_vectors) > 0:
            final_gt_vectors = []
            for vector in gt_vectors:
                if len(vector) >= 2:
                    resampled_vec = resample_vector(vector, num_sample_pts)
                    final_gt_vectors.append(resampled_vec)
            gt_vectors_array = np.array(final_gt_vectors) if final_gt_vectors else np.array([])
            gt_labels_array = np.array(gt_labels) if final_gt_vectors else np.array([])
        else:
            gt_vectors_array = np.array([])
            gt_labels_array = np.array([])
        
        # Process predictions
        pred_vectors_processed, pred_labels_processed, pred_scores_processed = \
            process_predictions_with_fov_clipping_av2(
                pred_vectors=pred_data['vectors'],
                pred_labels=pred_data['labels'],
                pred_scores=pred_data['scores'],
                sample_info=sample_info,
                av2_data_path=av2_data_path,
                pc_range=pc_range,
                camera_name=camera_name,
                apply_clipping=apply_clipping
            )
        
        # Resample predictions to num_sample_pts
        if len(pred_vectors_processed) > 0:
            final_pred_vectors = []
            final_pred_labels = []
            final_pred_scores = []
            for vec, label, score in zip(pred_vectors_processed, pred_labels_processed, pred_scores_processed):
                if len(vec) >= 2:
                    resampled_vec = resample_vector(vec, num_sample_pts)
                    final_pred_vectors.append(resampled_vec)
                    final_pred_labels.append(label)
                    final_pred_scores.append(score)
            
            pred_vectors_array = np.array(final_pred_vectors) if final_pred_vectors else np.array([])
            pred_labels_array = np.array(final_pred_labels) if final_pred_labels else np.array([])
            pred_scores_array = np.array(final_pred_scores) if final_pred_labels else np.array([])
        else:
            pred_vectors_array = np.array([])
            pred_labels_array = np.array([])
            pred_scores_array = np.array([])
        
        results_per_camera[camera_name] = {
            'gt': {
                'vectors': gt_vectors_array,
                'labels': gt_labels_array
            },
            'pred': {
                'vectors': pred_vectors_array,
                'labels': pred_labels_array,
                'scores': pred_scores_array
            }
        }
    
    return results_per_camera


# ==================== MAIN UNIFIED WORKFLOW ====================
def main():
    parser = argparse.ArgumentParser(
        description='Unified MapQR AV2 Evaluation: Inference + Camera-Specific FOV Clipping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Front center camera only
  python %(prog)s --cameras ring_front_center
  
  # Front + Rear cameras
  python %(prog)s --cameras ring_front_center ring_rear_left ring_rear_right
  
  # All 7 cameras (baseline)
  python %(prog)s --cameras all
  
  # Skip inference if predictions exist
  python %(prog)s --skip-inference --predictions-pkl existing_preds.pkl --cameras ring_front_center
  
  # Full BEV evaluation without FOV clipping
  python %(prog)s --no-clipping --cameras ring_front_center
        """)
    
    # Inference arguments
    parser.add_argument('--config', type=str,
                       default='config/mapqr_av2_3d_r50_6ep.py',
                       help='MapQR AV2 config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='ckpts/mapqr_av2_3d_r50_6ep.pth',
                       help='MapQR AV2 checkpoint file path')
    parser.add_argument('--score-thresh', type=float, default=0.0,
                       help='Score threshold for predictions (default: 0.0 = keep all)')
    
    # Camera configuration
    parser.add_argument('--cameras', type=str, nargs='+', default=['ring_front_center'],
                       help='Camera names to use (ring_front_center, ring_rear_left, etc.) or "all" for all 7 cameras')
    
    # Evaluation arguments
    parser.add_argument('--av2-data-path', type=str,
                       default=None,
                       help='Path to AV2 sensor dataset root (default: use path from config file)')
    parser.add_argument('--samples-pkl', type=str,
                       default=None,
                       help='Path to samples pickle file (default: use path from config file)')
    parser.add_argument('--pc-range', type=float, nargs=6,
                       default=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                       help='Point cloud range (AV2 default: [-30,-15,-5,30,15,3])')
    parser.add_argument('--num-sample-pts', type=int, default=100,
                       help='Number of points to resample vectors to (default: 100)')
    
    # Output arguments
    parser.add_argument('--predictions-pkl', type=str, default=None,
                       help='Output/input pickle file for predictions (auto-generated if not specified)')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Output JSON file for results (auto-generated if not specified)')
    
    # Control flow
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference step (use existing predictions)')
    parser.add_argument('--apply-clipping', action='store_true',
                       help='Apply camera FOV clipping (default: True)')
    parser.add_argument('--no-clipping', dest='apply_clipping', action='store_false',
                       help='Disable FOV clipping (full BEV evaluation)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help=f'Number of parallel workers for evaluation (default: 8, set to 1 for serial, max: {cpu_count()})')
    parser.set_defaults(apply_clipping=True)
    
    args = parser.parse_args()
    
    # Parse camera configuration
    camera_indices = parse_camera_config(args.cameras)
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    
    # Generate default filenames if not specified
    camera_suffix = '_'.join([name.replace('CAM_', '').lower() for name in camera_names])
    if args.predictions_pkl is None:
        args.predictions_pkl = f'mapqr_av2_predictions_{camera_suffix}.pkl'
    if args.output_json is None:
        args.output_json = f'mapqr_results_{camera_suffix}.json'
    
    print("\n" + "="*80)
    print("UNIFIED MapQR EVALUATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  AV2 data path: {args.av2_data_path or 'from config file'}")
    print(f"  Samples pickle: {args.samples_pkl or 'from config file'}")
    print(f"  Camera configuration: {camera_names} ({len(camera_names)}/7 cameras)")
    print(f"  FOV clipping: {'ENABLED' if args.apply_clipping else 'DISABLED (full BEV)'}")
    print(f"  Predictions file: {args.predictions_pkl}")
    print(f"  Output file: {args.output_json}")
    
    # STEP 1: Run inference (unless skipped)
    if not args.skip_inference:
        predictions_pkl = run_mapqr_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_pkl=args.predictions_pkl,
            camera_indices=camera_indices,
            score_thresh=args.score_thresh,
            samples_pkl=args.samples_pkl,
            av2_data_path=args.av2_data_path
        )
    else:
        print("\n" + "="*80)
        print("STEP 1: Skipping Inference (using existing predictions)")
        print("="*80)
        predictions_pkl = args.predictions_pkl
        if not os.path.exists(predictions_pkl):
            raise FileNotFoundError(f"Predictions file not found: {predictions_pkl}")
        print(f"Using predictions from: {predictions_pkl}")
    
    # STEP 2: Load samples and predictions
    print("\n" + "="*80)
    print("STEP 2: Loading Data")
    print("="*80)
    
    # Get actual samples path (from args or infer from config)
    # Always load config to get load_interval
    cfg = Config.fromfile(args.config)
    
    # Get actual samples path (from args or infer from config)
    if args.samples_pkl:
        samples_path = args.samples_pkl
    else:
        samples_path = cfg.data.test.ann_file
    
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples_data = pickle.load(f)
    
    # CRITICAL: Sort by timestamp before slicing to match CustomAV2OfflineLocalMapDataset behavior
    # The dataset class does: data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))
    # then: data_infos = data_infos[::self.load_interval]
    all_samples = list(sorted(samples_data['samples'], key=lambda e: e['timestamp']))
    
    # Apply load_interval from config if specified
    # AV2 default: load_interval=4 (10Hz -> 2.5Hz, similar to NuScenes 2Hz)
    load_interval = getattr(cfg.data.test, 'load_interval', 1)
    if load_interval > 1:
        samples = all_samples[::load_interval]
        print(f"✓ Loaded {len(samples)} samples (load_interval={load_interval}, total in file: {len(all_samples)})")
    else:
        samples = all_samples
        print(f"✓ Loaded {len(samples)} samples")

    
    print(f"\nLoading predictions from {predictions_pkl}...")
    with open(predictions_pkl, 'rb') as f:
        predictions_by_token = pickle.load(f)
    print(f"✓ Loaded predictions for {len(predictions_by_token)} samples")
    
    # STEP 3: Evaluate
    print("\n" + "="*80)
    print("STEP 3: Evaluating with Camera-Specific FOV Clipping")
    print("="*80)
    
    # Get actual AV2 path for evaluation
    if args.av2_data_path:
        av2_eval_path = args.av2_data_path
    else:
        # Load config to get the data root
        cfg = Config.fromfile(args.config)
        av2_eval_path = cfg.data.test.data_root
    
    # Create evaluator
    evaluator = CameraSpecificEvaluator(
        av2_data_path=av2_eval_path,
        pc_range=args.pc_range,
        num_sample_pts=args.num_sample_pts,
        thresholds_chamfer=[0.5, 1.0, 1.5],
        camera_names=camera_names,
        num_workers=max(1, args.num_workers)  # Pass configured worker count
    )
    
    print(f"\nEvaluator configuration:")
    print(f"  PC range: {args.pc_range}")
    print(f"  Patch size: {evaluator.patch_size}")
    print(f"  Sample points per vector: {args.num_sample_pts} (MapTR standard)")
    print(f"  Chamfer thresholds: {evaluator.thresholds_chamfer} meters")
    print(f"  Evaluation method: Official MapTR (STRtree, linewidth=2m)")
    
    # Accumulate predictions and GT
    mode_str = "camera-specific FOV clipping" if args.apply_clipping else "full BEV (no clipping)"
    
    # Determine number of workers (default is 8)
    num_workers = max(1, args.num_workers)  # At least 1 worker
    
    print(f"\nProcessing {len(samples)} samples with {mode_str}...")
    print(f"Using {num_workers} parallel workers (CPU count: {cpu_count()})")
    
    # Prepare data for parallel processing
    samples_with_preds = []
    for sample_info in samples:
        sample_token = sample_info['token']
        if sample_token in predictions_by_token:
            samples_with_preds.append((sample_info, predictions_by_token[sample_token]))
    
    print(f"Matched {len(samples_with_preds)} samples with predictions")
    
    if num_workers == 1:
        # Serial processing (for debugging or when multiprocessing has issues)
        print("Running in SERIAL mode...")
        results_list = []
        for sample_info, pred_data in tqdm(samples_with_preds, desc="Processing samples"):
            result = process_sample_worker_av2(
                sample_info, pred_data, av2_eval_path, args.pc_range,
                args.num_sample_pts, camera_names, args.apply_clipping
            )
            results_list.append(result)
    else:
        # Parallel processing
        print("Running in PARALLEL mode...")
        worker_fn = partial(
            process_sample_worker_av2,
            av2_data_path=av2_eval_path,
            pc_range=args.pc_range,
            num_sample_pts=args.num_sample_pts,
            camera_names=camera_names,
            apply_clipping=args.apply_clipping
        )
        
        with Pool(processes=num_workers) as pool:
            # Use imap for progress bar
            results_list = list(tqdm(
                pool.starmap(worker_fn, samples_with_preds),
                total=len(samples_with_preds),
                desc="Processing samples"
            ))
    
    # Merge results into evaluator
    print("\nMerging results from parallel workers...")
    for result in results_list:
        for camera_name in camera_names:
            camera_result = result[camera_name]
            
            evaluator.predictions_per_camera[camera_name].append({
                'vectors': camera_result['pred']['vectors'],
                'labels': camera_result['pred']['labels'],
                'scores': camera_result['pred']['scores']
            })
            
            evaluator.ground_truths_per_camera[camera_name].append({
                'vectors': camera_result['gt']['vectors'],
                'labels': camera_result['gt']['labels']
            })
        
        evaluator.num_samples_processed += 1
    
    # Compute metrics
    print("\nComputing metrics...")
    results = evaluator.evaluate()
    
    # STEP 4: Print and save results
    print("\n" + "="*80)
    print("STEP 4: EVALUATION RESULTS")
    print("="*80)
    
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    # Print average first if multiple cameras
    if 'AVERAGE' in results:
        camera_results = results['AVERAGE']
        print(f"\nAVERAGE (across {len(camera_names)} cameras):")
        print(f"  mAP (all classes & thresholds): {camera_results['mAP']:.4f}")
        print()
        for class_name in class_names:
            if class_name not in camera_results:
                continue
            class_results = camera_results[class_name]
            print(f"  {class_name}:")
            for threshold in evaluator.thresholds_chamfer:
                ap = class_results[f'AP@{threshold}m']
                print(f"    AP@{threshold}m: {ap:.4f}")
            cd = class_results['avg_chamfer_distance']
            cd_str = f"{cd:.4f}m" if cd != float('inf') else "N/A"
            print(f"    Avg CD: {cd_str}")
        print("\n" + "-"*80)
    
    # Print per-camera results
    for camera_name, camera_results in results.items():
        if camera_name == 'AVERAGE':
            continue
        
        print(f"\n{camera_name}:")
        
        if 'mAP' in camera_results:
            print(f"  mAP (all classes & thresholds): {camera_results['mAP']:.4f}")
        
        print()
        for class_name in class_names:
            if class_name not in camera_results:
                continue
            class_results = camera_results[class_name]
            print(f"  {class_name}:")
            for threshold in evaluator.thresholds_chamfer:
                ap = class_results[f'AP@{threshold}m']
                print(f"    AP@{threshold}m: {ap:.4f}")
            cd = class_results['avg_chamfer_distance']
            cd_str = f"{cd:.4f}m" if cd != float('inf') else "N/A"
            print(f"    Avg CD: {cd_str}")
    
    # Save results
    print(f"\n\nSaving results to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nFiles generated:")
    print(f"  - Predictions: {args.predictions_pkl}")
    print(f"  - Results: {args.output_json}")
    print()


if __name__ == '__main__':
    main()
