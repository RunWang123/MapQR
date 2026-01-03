#!/usr/bin/env python3
"""
AV2-Specific utilities for camera-specific FOV clipping and rotation.
Simplified version that uses existing AV2 dataset components.

This module:
1. Imports VectorizedAV2LocalMap from av2_offlinemap_dataset.py (already implemented)
2. Provides AV2-adapted versions of extract_gt_with_fov_clipping() and process_predictions_with_fov_clipping()
3. Reuses CameraFOVClipper from camera_fov_utils.py (generic, works for both datasets)

Key AV2 Adaptations:
- Camera transforms from 'extrinsics' (4x4 matrix) instead of sensor2ego quaternion
- Per-camera ego2global transforms (e2g_rotation/e2g_translation)
- Lidar2ego is identity matrix in AV2
"""

import numpy as np
from typing import Dict, List, Tuple
from shapely.geometry import LineString
from scipy.spatial.transform import Rotation
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

# Import VectorizedAV2LocalMap from existing AV2 dataset code
from projects.mmdet3d_plugin.datasets.av2_offlinemap_dataset import VectorizedAV2LocalMap

# Import generic CameraFOVClipper (works for both NuScenes and AV2)
from tools.camera_fov_utils import CameraFOVClipper


# ==================== 3D FOV CLIPPING FUNCTIONS ====================

def project_3d_points_to_image(
    points_3d: np.ndarray,
    cam_extrinsics: np.ndarray,
    cam_intrinsics: np.ndarray,
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to camera image plane using actual 3D projection.
    
    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        cam_extrinsics: 4×4 camera extrinsics matrix (cam2world or cam2bev)
        cam_intrinsics: 3×3 camera intrinsics matrix
        image_size: (height, width) tuple
        
    Returns:
        points_2d: (N, 2) array of 2D image coordinates
        valid_mask: (N,) boolean array indicating which points project inside image bounds
                    and are in front of camera
    """
    height, width = image_size
    
    # Step 1: Transform 3D points to camera frame
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
    
    # Transform from world to camera (invert extrinsics)
    world_to_cam = np.linalg.inv(cam_extrinsics)
    points_cam = (world_to_cam @ points_homo.T).T
    
    # Step 2: Project to image using intrinsics
    # Perspective projection: [x, y, z] -> [x/z, y/z, 1]
    z_cam = points_cam[:, 2:3]
    
    # Check if points are in front of camera
    in_front = (z_cam[:, 0] > 0.1)  # Small threshold to avoid numerical issues
    
    # Avoid division by zero
    z_safe = np.where(z_cam > 0.1, z_cam, 1.0)
    
    # Project to normalized camera coordinates
    points_normalized = points_cam[:, :2] / z_safe
    
    # Apply intrinsics to get pixel coordinates
    points_2d_homo = np.column_stack([points_normalized, np.ones(len(points_normalized))])
    points_2d = (cam_intrinsics @ points_2d_homo.T).T[:, :2]
    
    # Step 3: Check which points are inside image bounds
    in_bounds_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width)
    in_bounds_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    in_bounds = in_bounds_x & in_bounds_y
    
    # Combine all validity checks
    valid_mask = in_front & in_bounds
    
    return points_2d, valid_mask


def clip_3d_line_to_image_bounds(
    vector_3d: np.ndarray,
    cam_extrinsics: np.ndarray,
    cam_intrinsics: np.ndarray,
    image_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Clip a 3D vector to camera FOV using actual 3D projection.
    Handles line segments that cross image boundaries by interpolating intersection points.
    
    Args:
        vector_3d: (N, 3) array of 3D points representing a polyline
        cam_extrinsics: 4×4 camera extrinsics matrix
        cam_intrinsics: 3×3 camera intrinsics matrix
        image_size: (height, width) tuple
        
    Returns:
        List of clipped 3D vectors (each is np.ndarray of shape (M, 3))
        May return multiple segments if line crosses in/out of view
    """
    if len(vector_3d) < 2:
        return []
    
    height, width = image_size
    
    # Project all points to image
    points_2d, valid_mask = project_3d_points_to_image(
        vector_3d, cam_extrinsics, cam_intrinsics, image_size
    )
    
    # If all points are valid, return original vector
    if np.all(valid_mask):
        return [vector_3d]
    
    # If no points are valid, check if line segments cross the FOV
    if not np.any(valid_mask):
        # Still need to check if any segment crosses through the FOV
        # For now, return empty (conservative approach)
        return []
    
    # Build clipped segments
    clipped_segments = []
    current_segment = []
    
    for i in range(len(vector_3d)):
        if valid_mask[i]:
            # Point is inside FOV
            current_segment.append(vector_3d[i])
        
        # Check if we need to add boundary intersection points
        if i < len(vector_3d) - 1:
            # Check transition between consecutive points
            p1_valid = valid_mask[i]
            p2_valid = valid_mask[i + 1]
            
            if p1_valid != p2_valid:
                # Boundary crossing - compute intersection point
                p1_3d = vector_3d[i]
                p2_3d = vector_3d[i + 1]
                
                # Find intersection with image boundary
                intersection_3d = find_boundary_intersection_3d(
                    p1_3d, p2_3d, 
                    cam_extrinsics, cam_intrinsics, 
                    image_size, p1_valid
                )
                
                if intersection_3d is not None:
                    if p1_valid:
                        # Going from inside to outside
                        current_segment.append(intersection_3d)
                        # Finish current segment
                        if len(current_segment) >= 2:
                            clipped_segments.append(np.array(current_segment))
                        current_segment = []
                    else:
                        # Going from outside to inside
                        current_segment.append(intersection_3d)
            
            elif not p1_valid and not p2_valid:
                # Both outside - finalize any current segment
                if len(current_segment) >= 2:
                    clipped_segments.append(np.array(current_segment))
                current_segment = []
    
    # Add final segment if exists
    if len(current_segment) >= 2:
        clipped_segments.append(np.array(current_segment))
    
    return clipped_segments


def find_boundary_intersection_3d(
    p1_3d: np.ndarray,
    p2_3d: np.ndarray,
    cam_extrinsics: np.ndarray,
    cam_intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    p1_inside: bool,
    num_iterations: int = 10
) -> np.ndarray:
    """
    Find the 3D point where a line segment crosses the image boundary.
    Uses binary search to find the intersection point.
    
    Args:
        p1_3d: First 3D point (shape: (3,))
        p2_3d: Second 3D point (shape: (3,))
        cam_extrinsics: 4×4 camera extrinsics matrix
        cam_intrinsics: 3×3 camera intrinsics matrix
        image_size: (height, width) tuple
        p1_inside: Whether p1 is inside the FOV
        num_iterations: Number of binary search iterations
        
    Returns:
        intersection_3d: 3D point at boundary (shape: (3,)) or None if not found
    """
    height, width = image_size
    
    # Binary search for intersection point
    t_min, t_max = 0.0, 1.0
    
    for _ in range(num_iterations):
        t_mid = (t_min + t_max) / 2.0
        p_mid_3d = p1_3d + t_mid * (p2_3d - p1_3d)
        
        # Project to check if inside
        p_mid_2d, valid_mask = project_3d_points_to_image(
            p_mid_3d.reshape(1, 3), 
            cam_extrinsics, cam_intrinsics, image_size
        )
        
        is_inside = valid_mask[0]
        
        if is_inside == p1_inside:
            t_min = t_mid
        else:
            t_max = t_mid
    
    # Return the midpoint of final interval
    t_final = (t_min + t_max) / 2.0
    intersection_3d = p1_3d + t_final * (p2_3d - p1_3d)
    
    return intersection_3d


def clip_vectors_to_fov_3d(
    vectors_3d: List[np.ndarray],
    labels: List[int],
    cam_extrinsics: np.ndarray,
    cam_intrinsics: np.ndarray,
    image_size: Tuple[int, int] = (900, 1600),
    debug: bool = False
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Clip a list of 3D vectors to camera FOV, preserving Z coordinates.
    
    Args:
        vectors_3d: List of 3D vectors, each is (N, 3) array
        labels: List of integer labels
        cam_extrinsics: 4×4 camera extrinsics matrix
        cam_intrinsics: 3×3 camera intrinsics matrix
        image_size: (height, width) tuple
        debug: Enable debug logging
        
    Returns:
        clipped_vectors: List of clipped 3D vectors
        clipped_labels: List of corresponding labels
    """
    clipped_vectors = []
    clipped_labels = []
    
    total_input = len(vectors_3d)
    total_output = 0
    
    for vector_3d, label in zip(vectors_3d, labels):
        # Clip each vector
        clipped_segments = clip_3d_line_to_image_bounds(
            vector_3d, cam_extrinsics, cam_intrinsics, image_size
        )
        
        # Add all resulting segments
        for segment in clipped_segments:
            if len(segment) >= 2:  # Only keep valid polylines
                clipped_vectors.append(segment)
                clipped_labels.append(label)
                total_output += 1
    
    if debug:
        retention_rate = (total_output / total_input * 100) if total_input > 0 else 0
        print(f"[DEBUG] 3D FOV Clipping: {total_input} vectors -> {total_output} vectors ({retention_rate:.1f}% retention)")
    
    return clipped_vectors, clipped_labels


def extract_gt_vectors_av2(sample_info: Dict, av2_data_path: str, pc_range: list, fixed_num: int = 20, debug: bool = False) -> Dict:
    """
    Extract GT vectors from AV2 map with MapTR's resampling method.
    Adapted from NuScenes version to use AV2's data structure.
    
    Args:
        sample_info: AV2 sample info dict
        av2_data_path: Path to AV2 dataset root
        pc_range: Point cloud range
        fixed_num: Number of points to resample to (default: 20)
        debug: Enable debug logging (default: False)
        
    Returns:
        Dict with 'vectors' (list of numpy arrays) and 'labels' (list of ints)
    """
    # Setup patch size
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    patch_size = (patch_h, patch_w)
    
    # Create AV2 vector map
    # Signature: __init__(self, canvas_size, patch_size, map_classes, ...)
    # Note: VectorizedAV2LocalMap doesn't take dataroot as parameter!
    # It uses map annotations directly in gen_vectorized_samples()
    vector_map = VectorizedAV2LocalMap(
        canvas_size=(200, 100),
        patch_size=patch_size,
        map_classes=['divider', 'ped_crossing', 'boundary']
    )
    
    # AV2: lidar2ego is identity
    lidar2ego = np.eye(4)
    
    # AV2: ego2global from sample_info
    ego2global = np.eye(4)
    ego2global[:3, :3] = sample_info['e2g_rotation']
    ego2global[:3, 3] = sample_info['e2g_translation']
    
    lidar2global = ego2global @ lidar2ego  # Since lidar2ego is identity, this is just ego2global
    lidar2global_translation = list(lidar2global[:3, 3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    
    # AV2: Get map annotation from sample_info
    # The annotation is already extracted in the pickle file
    map_annotation = sample_info['annotation']
    
    if debug:
        print(f"[DEBUG] extract_gt_vectors_av2:")
        print(f"  annotation keys: {map_annotation.keys()}")
        for key, val in map_annotation.items():
            print(f"    {key}: {len(val)} items")
    
    # Generate vectorized map from annotation
    # Signature: gen_vectorized_samples(self, map_annotation, example=None, feat_down_sample=32)
    anns_results = vector_map.gen_vectorized_samples(map_annotation)
    
    if debug:
        print(f"  gen_vectorized_samples returned keys: {anns_results.keys()}")
        for key, val in anns_results.items():
            print(f"    {key}: type={type(val)}")
    
    # AV2's gen_vectorized_samples returns (line 665-670):
    # - 'gt_vecs_pts_loc': LiDARInstanceLines object
    # - 'gt_vecs_label': list of labels
    gt_instance_lines = anns_results['gt_vecs_pts_loc']  # LiDARInstanceLines object
    gt_labels = anns_results['gt_vecs_label']  # Correct key!
    
    if debug:
        print(f"  gt_instance_lines type: {type(gt_instance_lines)}")
        print(f"  gt_labels: {len(gt_labels)} labels")
    
    # Extract instances from LiDARInstanceLines
    # The instance_list contains LineString objects
    gt_instances = gt_instance_lines.instance_list
    
    if debug:
        print(f"  Extracted {len(gt_instances)} instances from LiDARInstanceLines")
        if len(gt_instances) > 0:
            print(f"  First instance type: {type(gt_instances[0])}")
    
    # Resample using MapTR's method
    resampled_vectors = []
    final_labels = []
    
    for instance, label in zip(gt_instances, gt_labels):
        if not hasattr(instance, 'length') or instance.length == 0:
            continue
        
        # Check if LineString has Z coordinates
        has_z = instance.has_z if hasattr(instance, 'has_z') else False
        
        # Resample along the line
        distances = np.linspace(0, instance.length, fixed_num)
        sampled_points = np.array([list(instance.interpolate(distance).coords) 
                                   for distance in distances])
        
        # Reshape based on dimensionality
        if has_z and sampled_points.shape[-1] == 3:
            sampled_points = sampled_points.reshape(-1, 3)  # Preserve 3D
        else:
            sampled_points = sampled_points.reshape(-1, 2)  # 2D only
        
        resampled_vectors.append(sampled_points)
        final_labels.append(label)
    
    if debug:
        print(f"  Final: {len(resampled_vectors)} vectors after resampling")
        if len(resampled_vectors) > 0:
            print(f"  Vector dimensionality: {resampled_vectors[0].shape[1]}D")
    
    return {
        'vectors': resampled_vectors,
        'labels': final_labels
    }


def extract_gt_with_fov_clipping_av2(
    sample_info: Dict,
    av2_data_path: str,
    pc_range: list,
    camera_name: str = 'ring_front_center',
    fixed_num: int = 20,
    apply_clipping: bool = True,
    debug: bool = False
) -> Dict:
    """
    Extract GT vectors for AV2 with optional camera-specific FOV clipping and rotation.
    Adapted from NuScenes version for AV2's data structure.
    
    Processing pipeline:
    1. Extract GT vectors from AV2 map with 20-point resampling
    2. Optionally apply camera-specific FOV clipping
    3. Rotate to camera-centric coordinates
    
    Args:
        sample_info: AV2 sample info dict
        av2_data_path: Path to AV2 dataset root
        pc_range: BEV range
        camera_name: Camera name (e.g., 'ring_front_center')
        fixed_num: Number of points for resampling (default: 20)
        apply_clipping: If True, apply FOV clipping (default: True)
        debug: Enable debug logging (default: False)
        
    Returns:
        Dict with 'vectors' and 'labels' in camera-centric coordinates
    """
    # Step 1: Extract GT vectors from AV2 map
    gt_data = extract_gt_vectors_av2(sample_info, av2_data_path, pc_range, fixed_num=fixed_num, debug=debug)
    vectors = gt_data['vectors']
    gt_labels = gt_data['labels']
    
    if len(vectors) == 0:
        return {'vectors': [], 'labels': []}
    
    # Step 2: Get camera transforms (AV2 format)
    cam_info = sample_info['cams'][camera_name]
    
    # AV2 DIFFERENCE: intrinsics is already a 3x3 matrix (or might need padding to 4x4)
    if 'intrinsics' in cam_info:
        cam_intrinsic_3x3 = cam_info['intrinsics']
        if cam_intrinsic_3x3.shape == (3, 3):
            cam_intrinsic = np.eye(4)
            cam_intrinsic[:3, :3] = cam_intrinsic_3x3
        else:
            cam_intrinsic = cam_intrinsic_3x3
    elif 'cam_intrinsic' in cam_info:
        cam_intrinsic = np.array(cam_info['cam_intrinsic'])
    else:
        raise KeyError(f"Camera info missing intrinsics: {cam_info.keys()}")
    
    # AV2 CRITICAL: extrinsics is EGO2CAM, not CAM2EGO!
    # See av2_offlinemap_dataset.py line 975: ego2cam_rt = cam_info['extrinsics']
    # We need to INVERT it to get cam2ego
    ego2cam = cam_info['extrinsics']
    cam2ego = np.linalg.inv(ego2cam)
    
    # AV2: Per-camera ego2global
    ego2global = np.eye(4)
    ego2global[:3, :3] = cam_info['e2g_rotation']
    ego2global[:3, 3] = cam_info['e2g_translation']
    
    # AV2: lidar2ego is identity
    lidar2ego = np.eye(4)
    
    lidar2global = ego2global @ lidar2ego  # Just ego2global since lidar2ego is identity
    cam2global = ego2global @ cam2ego
    
    # Get lidar2global rotation for BEV alignment
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    rotation = Quaternion(lidar2global_rotation)
    patch_angle_deg = quaternion_yaw(rotation) / np.pi * 180
    patch_angle_rad = np.radians(patch_angle_deg)
    
    # Rotate camera transformation to align with BEV coordinate system
    cos_a = np.cos(-patch_angle_rad)
    sin_a = np.sin(-patch_angle_rad)
    
    rotation_matrix_bev = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    cam_extrinsics_bev = np.eye(4)
    cam_extrinsics_bev[:3, :3] = rotation_matrix_bev @ cam2global[:3, :3]
    cam_extrinsics_bev[:3, 3] = rotation_matrix_bev @ (cam2global[:3, 3] - lidar2global[:3, 3])
    
    # Step 3: Optionally apply FOV clipping
    if apply_clipping:
        # Extract 3x3 intrinsics for clipper
        cam_intrinsic_3x3 = cam_intrinsic[:3, :3] if cam_intrinsic.shape[0] == 4 else cam_intrinsic
        
        if debug:
            print(f"\n[DEBUG] Before 3D FOV clipping:")
            print(f"  Total vectors: {len(vectors)}")
            if len(vectors) > 0:
                # Show bounds of first vector
                first_vec = vectors[0]
                print(f"  First vector shape: {first_vec.shape}")
                print(f"  First vector x bounds: [{first_vec[:, 0].min():.2f}, {first_vec[:, 0].max():.2f}]")
                print(f"  First vector y bounds: [{first_vec[:, 1].min():.2f}, {first_vec[:, 1].max():.2f}]")
                if first_vec.shape[1] == 3:
                    print(f"  First vector z bounds: [{first_vec[:, 2].min():.2f}, {first_vec[:, 2].max():.2f}]")
            print(f"  Camera extrinsics translation: {cam_extrinsics_bev[:3, 3]}")
            print(f"  Camera intrinsics:\n{cam_intrinsic_3x3}")
        
        # Use 3D FOV clipping that preserves Z coordinates
        # AV2 image size: 2048 (height) x 1550 (width)
        cropped_vectors, cropped_labels = clip_vectors_to_fov_3d(
            vectors_3d=vectors,
            labels=gt_labels,
            cam_extrinsics=cam_extrinsics_bev,
            cam_intrinsics=cam_intrinsic_3x3,
            image_size=(2048, 1550),
            debug=debug
        )
        
        if len(cropped_vectors) == 0:
            if debug:
                print(f"  ❌ All vectors removed by 3D FOV clipping!")
            return {'vectors': [], 'labels': []}
    else:
        # Skip clipping - keep all vectors
        cropped_vectors = vectors
        cropped_labels = gt_labels
    
    # Step 4: Rotate to camera-centric coordinates (simple 90 degree clockwise rotation)
    # (x, y) -> (-y, x) aligns +Y with camera forward direction
    rotation_90_clockwise = np.array([
        [0, -1],
        [1, 0]
    ])
    
    # Apply rotation and resample
    final_vectors = []
    final_labels = []
    
    for vector, label in zip(cropped_vectors, cropped_labels):
        if len(vector) >= 2:
            # Check dimensionality
            is_3d = vector.shape[1] == 3
            
            # Rotate only x,y coordinates by 90 degrees clockwise
            if is_3d:
                xy_rotated = (rotation_90_clockwise @ vector[:, :2].T).T
                # Preserve z coordinate
                vector_rotated = np.column_stack([xy_rotated, vector[:, 2]])
            else:
                vector_rotated = (rotation_90_clockwise @ vector.T).T
            
            # Create LineString for resampling (use 2D for length calculation)
            line = LineString(vector_rotated[:, :2] if is_3d else vector_rotated)
            if line.length > 0:
                distances = np.linspace(0, line.length, fixed_num)
                
                if is_3d:
                    # Resample 3D line by interpolating z along with x,y
                    line_3d = LineString(vector_rotated)  # Full 3D LineString
                    resampled_points = np.array([list(line_3d.interpolate(distance).coords) 
                                                for distance in distances]).reshape(-1, 3)
                else:
                    # Resample 2D line
                    resampled_points = np.array([list(line.interpolate(distance).coords) 
                                                for distance in distances]).reshape(-1, 2)
                
                final_vectors.append(resampled_points)
                final_labels.append(label)
            else:
                # Degenerate case: zero length
                final_vectors.append(np.tile(vector_rotated[0], (fixed_num, 1)))
                final_labels.append(label)
    
    return {
        'vectors': final_vectors,
        'labels': final_labels
    }


def process_predictions_with_fov_clipping_av2(
    pred_vectors: List[np.ndarray],
    pred_labels: List[int],
    pred_scores: List[float],
    sample_info: Dict,
    av2_data_path: str,
    pc_range: list,
    camera_name: str = 'ring_front_center',
    apply_clipping: bool = True
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Process predictions with FOV clipping (AV2 version).
    Identical processing pipeline to GT for fair comparison.
    
    Processing steps:
    1. Convert 3D vectors to 2D (x, y only)
    2. Apply FOV clipping if enabled
    3. Rotate to camera-centric coordinates
    4. Resample to 20 points
    
    Args:
        pred_vectors: List of predicted vectors (N, 20, 3) or (N, 20, 2)
        pred_labels: List of predicted labels
        pred_scores: List of prediction scores
        sample_info: AV2 sample info dict
        av2_data_path: Path to AV2 dataset
        pc_range: BEV range
        camera_name: Camera name
        apply_clipping: If True, apply FOV clipping
        
    Returns:
        Tuple of (vectors, labels, scores) after processing
    """
    if len(pred_vectors) == 0:
        return [], [], []
    
    # CRITICAL for AV2: Model predicts in 3D (code_size=3)
    # - Use 3D FOV clipping to preserve z-coordinates
    # - Original MapTRv2 AV2: gt_z_flag=True, pred_z_flag=True
    
    # Store original 3D vectors (ensure they have z-coordinates)
    vectors_3d_original = []
    for vec in pred_vectors:
        if vec.shape[-1] == 3:
            vectors_3d_original.append(vec.copy())
        elif vec.shape[-1] == 2:
            # Add z=0 if missing
            z_zeros = np.zeros((len(vec), 1))
            vectors_3d_original.append(np.hstack([vec, z_zeros]))
        else:
            raise ValueError(f"Unexpected vector shape: {vec.shape}, expected (20, 2) or (20, 3)")
    
    # Get camera transforms (AV2 format)
    cam_info = sample_info['cams'][camera_name]
    
    # Get intrinsics
    if 'intrinsics' in cam_info:
        cam_intrinsic_3x3 = cam_info['intrinsics']
        if cam_intrinsic_3x3.shape == (3, 3):
            cam_intrinsic = np.eye(4)
            cam_intrinsic[:3, :3] = cam_intrinsic_3x3
        else:
            cam_intrinsic = cam_intrinsic_3x3
    elif 'cam_intrinsic' in cam_info:
        cam_intrinsic = np.array(cam_info['cam_intrinsic'])
    else:
        raise KeyError(f"Camera info missing intrinsics: {cam_info.keys()}")
    
    # AV2 CRITICAL: extrinsics is EGO2CAM, not CAM2EGO!
    # Must invert to get cam2ego
    ego2cam = cam_info['extrinsics']
    cam2ego = np.linalg.inv(ego2cam)
    
    # AV2: Per-camera ego2global
    ego2global = np.eye(4)
    ego2global[:3, :3] = cam_info['e2g_rotation']
    ego2global[:3, 3] = cam_info['e2g_translation']
    
    # AV2: lidar2ego is identity
    lidar2ego = np.eye(4)
    
    lidar2global = ego2global @ lidar2ego
    cam2global = ego2global @ cam2ego
    
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    rotation = Quaternion(lidar2global_rotation)
    patch_angle_deg = quaternion_yaw(rotation) / np.pi * 180
    patch_angle_rad = np.radians(patch_angle_deg)
    
    cos_a = np.cos(-patch_angle_rad)
    sin_a = np.sin(-patch_angle_rad)
    
    rotation_matrix_bev = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    cam_extrinsics_bev = np.eye(4)
    cam_extrinsics_bev[:3, :3] = rotation_matrix_bev @ cam2global[:3, :3]
    cam_extrinsics_bev[:3, 3] = rotation_matrix_bev @ (cam2global[:3, 3] - lidar2global[:3, 3])
    
    # Simple 90 degree clockwise rotation: (x, y) -> (-y, x)
    rotation_90_clockwise = np.array([
        [0, -1],
        [1, 0]
    ])
    
    # Process predictions with FOV clipping in 3D
    final_vectors = []
    final_labels = []
    final_scores = []
    
    if apply_clipping:
        # Extract 3x3 intrinsics
        cam_intrinsic_3x3 = cam_intrinsic[:3, :3] if cam_intrinsic.shape[0] == 4 else cam_intrinsic
        
        # Use 3D FOV clipping for each prediction
        # AV2 image size: 2048 (height) x 1550 (width)
        for vec_3d, label, score in zip(vectors_3d_original, pred_labels, pred_scores):
            # Clip using 3D clipping
            clipped_segments = clip_3d_line_to_image_bounds(
                vec_3d, cam_extrinsics_bev, cam_intrinsic_3x3, image_size=(2048, 1550)
            )
            
            if len(clipped_segments) > 0 and len(clipped_segments[0]) >= 2:
                # Take the first/longest segment (predictions should be continuous)
                cropped_vec_3d = clipped_segments[0]
                
                # Rotate to camera-centric coordinates by 90 degrees clockwise (only x,y rotation)
                vector_rotated_2d = (rotation_90_clockwise @ cropped_vec_3d[:, :2].T).T
                # Combine rotated x,y with original z
                vector_rotated_3d = np.hstack([vector_rotated_2d, cropped_vec_3d[:, 2:3]])
                
                line = LineString(vector_rotated_2d)  # Use 2D for length check
                if line.length > 0:
                    # Resample in 3D
                    line_3d = LineString(vector_rotated_3d)
                    distances = np.linspace(0, line_3d.length, 20)
                    resampled_points_3d = np.array([list(line_3d.interpolate(distance).coords)
                                                   for distance in distances]).reshape(-1, 3)
                    final_vectors.append(resampled_points_3d)
                    final_labels.append(label)
                    final_scores.append(score)
                else:
                    final_vectors.append(np.tile(vector_rotated_3d[0], (20, 1)))
                    final_labels.append(label)
                    final_scores.append(score)
    else:
        # Skip clipping - just rotate by 90 degrees and resample in 3D
        for vec_3d, label, score in zip(vectors_3d_original, pred_labels, pred_scores):
            if len(vec_3d) >= 2:
                # Rotate only x,y coordinates by 90 degrees clockwise
                vector_rotated_2d = (rotation_90_clockwise @ vec_3d[:, :2].T).T
                # Keep z as-is
                vector_rotated_3d = np.hstack([vector_rotated_2d, vec_3d[:, 2:3]])
                
                line_2d = LineString(vector_rotated_2d)
                if line_2d.length > 0:
                    # Resample in 3D
                    line_3d = LineString(vector_rotated_3d)
                    distances = np.linspace(0, line_3d.length, 20)
                    resampled_points_3d = np.array([list(line_3d.interpolate(distance).coords) 
                                                for distance in distances]).reshape(-1, 3)
                    final_vectors.append(resampled_points_3d)
                    final_labels.append(label)
                    final_scores.append(score)
                else:
                    final_vectors.append(np.tile(vector_rotated_3d[0], (20, 1)))
                    final_labels.append(label)
                    final_scores.append(score)
    
    return final_vectors, final_labels, final_scores
