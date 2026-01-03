#!/usr/bin/env python3
"""
Shared utilities for camera-specific FOV clipping and rotation.
EXACT code from training_6pv_enhance used by BOTH visualization and evaluation.

This module contains:
1. VectorizedLocalMap - Extract GT vectors from NuScenes map
2. CameraFOVClipper - Clip vectors to camera FOV
3. GT extraction with 20-point MapTR resampling
4. Camera-specific FOV clipping and rotation functions

Both visualize_maptr_predictions.py and evaluate_with_fov_clipping_standalone.py
import from this file to ensure 100% identical processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, box
from shapely import ops
from scipy.spatial.transform import Rotation
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


# ==================== VECTORIZED MAP EXTRACTION ====================

class VectorizedLocalMap(object):
    """
    Extract vectorized map from NuScenes.
    EXACT copy from training_6pv_enhance/train_utils/vectorized_map.py
    """
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }
    
    def __init__(self, dataroot, patch_size,
                 map_classes=['divider','ped_crossing','boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane']):
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        self.patch_size = patch_size

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)     
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
        
        gt_instance = []
        gt_labels = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)
        
        return {'gt_vecs_pts_loc': gt_instance, 'gt_vecs_label': gt_labels, 'gt_labels': gt_labels}

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        """Convert line geometries to Shapely LineString instances (for later resampling)"""
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line.geoms:
                        line_vectors.append(l)
                elif line.geom_type == 'LineString':
                    line_vectors.append(line)
        return line_vectors

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = {}
        for line_type, geom in line_geom:
            line_instances_dict[line_type] = self._one_type_line_geom_to_vectors(geom)
        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        """Convert line geometries to Shapely LineString instances (for later resampling)"""
        line_instances = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
        return line_instances
    
    def ped_poly_geoms_to_instances(self, ped_geom):
        """Convert pedestrian crossing polygons to line instances (CORRECT training_6pv_enhance version)"""
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        """Convert polygon geometries to boundary instances (CORRECT training_6pv_enhance version)"""
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)


# ==================== CAMERA FOV CLIPPER ====================

class CameraFOVClipper:
    """
    Clips vectors to camera field of view
    EXACT copy from training_6pv_beta/train_utils/vectorized_map.py
    Uses Shapely intersection for proper geometric clipping.
    """
    
    def __init__(self, image_size=(900, 1600), lidar_height_above_ground=1.84, max_range=None):
        self.image_size = image_size
        self.lidar_height_above_ground = lidar_height_above_ground
        self.max_range = max_range  # Maximum range for backprojection (e.g., double PC range)
        self.boundary_intersection_points = []
        
    def project_points_to_camera(self, points: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D points to camera image plane"""
        # Convert points to homogeneous coordinates at ground plane level
        # In BEV coordinates, ground plane is at z = -lidar_height_above_ground
        ground_z = np.full(len(points), -self.lidar_height_above_ground)
        points_homo = np.column_stack([points, ground_z])
        points_homo = np.column_stack([points_homo, np.ones(len(points))])
        
        # Transform to camera frame (invert extrinsics to go from world to camera)
        world_to_cam = np.linalg.inv(extrinsics)
        points_cam = (world_to_cam @ points_homo.T).T
        
        # Project to image plane
        z_safe = np.maximum(points_cam[:, 2:3], 1e-6)
        points_2d = points_cam[:, :2] / z_safe
        
        # Apply intrinsics
        points_2d = (intrinsics[:2, :2] @ points_2d.T).T + intrinsics[:2, 2]
        
        # Check validity (points in front of camera)
        min_z_threshold = 0.1 # meters
        valid_mask = points_cam[:, 2] > min_z_threshold
        
        return points_2d, valid_mask, points_cam
    
    def _clip_polyline_at_camera_plane(self, polyline: np.ndarray, points_cam: np.ndarray, valid_mask: np.ndarray, extrinsics: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Clip polyline at camera safe distance plane"""
        if len(polyline) < 2:
            return polyline, []
        
        min_z_threshold = 0.5
        clipped_points = []
        intersection_points = []
        
        for i in range(len(polyline)):
            if valid_mask[i]:
                clipped_points.append(polyline[i])
            
            if i < len(polyline) - 1:
                if valid_mask[i] != valid_mask[i + 1]:
                    # Find intersection with Z=min_z_threshold plane
                    p1_cam = points_cam[i, :3]
                    p2_cam = points_cam[i + 1, :3]
                    
                    z1, z2 = p1_cam[2], p2_cam[2]
                    if abs(z2 - z1) > 1e-6:
                        t = (min_z_threshold - z1) / (z2 - z1)
                        t = np.clip(t, 0.0, 1.0)
                        
                        intersection_cam = p1_cam + t * (p2_cam - p1_cam)
                        intersection_cam_homo = np.append(intersection_cam, 1.0)
                        intersection_world_homo = extrinsics @ intersection_cam_homo
                        intersection_world = intersection_world_homo[:2]
                        
                        if not valid_mask[i] and valid_mask[i + 1]:
                            clipped_points.append(intersection_world)
                            intersection_points.append(intersection_world)
                        elif valid_mask[i] and not valid_mask[i + 1]:
                            clipped_points.append(intersection_world)
                            intersection_points.append(intersection_world)
        
        if len(clipped_points) == 0:
            return np.array([]), intersection_points
        
        return np.array(clipped_points), intersection_points
    
    def _backproject_image_point_to_ground_plane(self, u: float, v: float, extrinsics: np.ndarray, intrinsics: np.ndarray, lidar_height_above_ground: float = 1.84) -> np.ndarray:
        """Back-project an image point to the actual ground plane
        
        For downward-tilted cameras (like AV2), ground plane intersection may be behind camera.
        In this case, project forward along the ray for max_range distance.
        """
        # Unproject to normalized camera coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # Ray direction in camera frame
        ray_dir_cam = np.array([x_norm, y_norm, 1.0])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
        
        # Transform ray to BEV world space (where lidar is at origin)
        cam_pos_world = extrinsics[:3, 3]
        ray_dir_world = extrinsics[:3, :3] @ ray_dir_cam
        
        # Find intersection with actual ground plane (z = -lidar_height_above_ground in BEV coordinates)
        ground_plane_z = -lidar_height_above_ground
        
        # Check if ray is nearly parallel to ground
        if abs(ray_dir_world[2]) < 1e-6:
            # Ray parallel to ground - project forward by max_range
            use_distance = self.max_range if self.max_range is not None else 50.0
            intersection = cam_pos_world + use_distance * ray_dir_world
            return np.array([intersection[0], intersection[1]])
        
        t = (ground_plane_z - cam_pos_world[2]) / ray_dir_world[2]
        
        # Handle different cases
        if t < 0:
            # Ground plane intersection is BEHIND camera (common for downward-tilted cameras)
            # Project FORWARD along ray direction instead
            use_distance = self.max_range if self.max_range is not None else 50.0
            intersection = cam_pos_world + use_distance * ray_dir_world
        elif self.max_range is not None and t > self.max_range:
            # Intersection too far - clamp to max_range
            intersection = cam_pos_world + self.max_range * ray_dir_world
        else:
            # Normal case: ground plane intersection in front of camera
            intersection = cam_pos_world + t * ray_dir_world
        
        return np.array([intersection[0], intersection[1]])
    
    def _backproject_linestring_to_world(self, geometry, extrinsics: np.ndarray, intrinsics: np.ndarray) -> List[np.ndarray]:
        """
        Back-project Shapely LineString, LinearRing, or MultiLineString from image space to world space.
        
        This handles the geometric results from Shapely intersection and converts them
        back to world coordinates, preserving topology (closed polygons remain closed).
        
        Args:
            geometry: Shapely LineString, LinearRing, MultiLineString, or GeometryCollection
            extrinsics: Camera extrinsics matrix
            intrinsics: Camera intrinsics matrix
            
        Returns:
            List of numpy arrays, each representing a polyline in world coordinates
        """
        results = []
        
        # Handle different geometry types
        if geometry.geom_type == 'LineString' or geometry.geom_type == 'LinearRing':
            # Single line segment or closed ring (from polygon.exterior)
            coords_2d = np.array(geometry.coords)
            coords_world = []
            for u, v in coords_2d:
                world_pt = self._backproject_image_point_to_ground_plane(
                    u, v, extrinsics, intrinsics, self.lidar_height_above_ground
                )
                coords_world.append(world_pt)
            results.append(np.array(coords_world))
            
        elif geometry.geom_type == 'MultiLineString':
            # Multiple line segments (e.g., polygon clipped at boundaries)
            for line in geometry.geoms:
                coords_2d = np.array(line.coords)
                coords_world = []
                for u, v in coords_2d:
                    world_pt = self._backproject_image_point_to_ground_plane(
                        u, v, extrinsics, intrinsics, self.lidar_height_above_ground
                    )
                    coords_world.append(world_pt)
                if len(coords_world) >= 2:
                    results.append(np.array(coords_world))
                    
        elif geometry.geom_type == 'GeometryCollection':
            # Handle mixed geometry types
            for geom in geometry.geoms:
                sub_results = self._backproject_linestring_to_world(geom, extrinsics, intrinsics)
                results.extend(sub_results)
        
        return results
    
    def _crop_polyline_to_fov_shapely(self, polyline_world: np.ndarray, extrinsics: np.ndarray, 
                                      intrinsics: np.ndarray) -> List[np.ndarray]:
        """
        Crop polyline/polygon to FOV using correct Shapely intersection.
        
        KEY INSIGHT:
        - For POLYGONS: Use Polygon.intersection(box) → returns closed clipped polygon
        - For LINES: Use LineString.intersection(box) → returns line segments
        
        CRITICAL: Detect polygon closure in WORLD SPACE before projection!
        Camera projection can distort geometry, so we must check if the original
        polyline is closed before projecting to image space.
        
        Algorithm:
        1. Check if closed polygon IN WORLD SPACE (before projection!)
        2. Project polyline to image space
        3. Use appropriate Shapely intersection (Polygon or LineString)
        4. Back-project result to world space
        5. Closed polygons automatically include FOV boundary edges!
        
        Args:
            polyline_world: Polyline in world/BEV coordinates (Nx2)
            extrinsics: Camera extrinsics matrix
            intrinsics: Camera intrinsics matrix
            
        Returns:
            List of clipped polyline segments in world coordinates
        """
        if len(polyline_world) < 2:
            return []
        
        # Step 1: Detect if this is a closed polygon IN WORLD SPACE (BEFORE projection!)
        # This is CRITICAL: must check before camera plane clipping which can open polygons
        is_closed_polygon = False
        if len(polyline_world) >= 3:
            dist = np.linalg.norm(polyline_world[0] - polyline_world[-1])
            # Use stricter tolerance: 1cm instead of 10cm
            # This prevents near-open polylines from being incorrectly treated as polygons
            if dist < 0.01:  # 0.01 meter (1cm) tolerance in world space
                is_closed_polygon = True
        
        # Step 2: Project to image space
        points_2d, valid_mask, points_cam = self.project_points_to_camera(
            polyline_world, extrinsics, intrinsics
        )
        
        if not valid_mask.any():
            return []
        
        # Step 3: Check if we need camera plane clipping (points behind camera)
        if not valid_mask.all():
            # Some points behind camera - clip at camera plane first
            # NOTE: This may "open" a closed polygon by removing the back portion,
            # but we keep is_closed_polygon=True because we trust the world-space decision
            # and let Shapely properly handle the geometry in image space
            camera_clipped, _ = self._clip_polyline_at_camera_plane(
                polyline_world, points_cam, valid_mask, extrinsics
            )
            if len(camera_clipped) < 2:
                return []
            
            # Re-project clipped polyline
            points_2d, valid_mask, points_cam = self.project_points_to_camera(
                camera_clipped, extrinsics, intrinsics
            )
            polyline_to_use = camera_clipped
        else:
            polyline_to_use = polyline_world
        
        # Step 4: If originally closed polygon, ensure endpoints are connected in image space
        # DON'T force-close if camera clipping opened it - only if endpoints are exact match
        # This avoids creating artificial diagonal lines across the FOV
        if is_closed_polygon and len(points_2d) >= 3:
            # Only close if endpoints are exactly the same (camera plane didn't separate them)
            if not np.array_equal(points_2d[0], points_2d[-1]):
                # Camera clipping opened the polygon - treat as polyline instead
                is_closed_polygon = False
        
        # Step 5: Create FOV box in image space
        h, w = self.image_size
        fov_box = box(0, 0, w, h)
        
        # Step 6: Use CORRECT Shapely intersection based on geometry type
        if is_closed_polygon:
            # POLYGON: Use Polygon.intersection(box) for closed result
            # This automatically adds edges along FOV boundaries (like cutting a cake!)
            poly_2d = Polygon(points_2d)
            
            # Validate polygon - buffer(0) can fix self-intersections and other issues
            if not poly_2d.is_valid:
                poly_2d = poly_2d.buffer(0)
            
            # If still invalid after buffer, treat as linestring instead
            if not poly_2d.is_valid or poly_2d.is_empty:
                is_closed_polygon = False
            else:
                clipped_geom = poly_2d.intersection(fov_box)
                
                if clipped_geom.is_empty:
                    return []
                
                # Extract exterior boundary (always closed)
                if clipped_geom.geom_type == 'Polygon':
                    # Single clipped polygon - extract its boundary (LinearRing)
                    boundary_2d = clipped_geom.exterior
                    world_segments = self._backproject_linestring_to_world(
                        boundary_2d, extrinsics, intrinsics
                    )
                elif clipped_geom.geom_type == 'MultiPolygon':
                    # Multiple polygons (rare) - extract all boundaries
                    world_segments = []
                    for poly in clipped_geom.geoms:
                        boundary_2d = poly.exterior
                        segments = self._backproject_linestring_to_world(
                            boundary_2d, extrinsics, intrinsics
                        )
                        world_segments.extend(segments)
                else:
                    # Unexpected type (shouldn't happen)
                    return []
        
        if not is_closed_polygon:
            # LINESTRING: Use LineString.intersection(box) for line segments
            line_2d = LineString(points_2d)
            clipped_geom = line_2d.intersection(fov_box)
            
            if clipped_geom.is_empty:
                return []
            
            # Merge touching segments if possible
            if isinstance(clipped_geom, MultiLineString):
                clipped_geom = ops.linemerge(clipped_geom)
            
            # Back-project to world space
            world_segments = self._backproject_linestring_to_world(
                clipped_geom, extrinsics, intrinsics
            )
        
        return world_segments
    
    def crop_vectors_to_fov(self, vectors: List[List[np.ndarray]], labels: List[int],
                           extrinsics: np.ndarray, intrinsics: np.ndarray) -> Tuple[List[List[np.ndarray]], List[int], List[np.ndarray]]:
        """
        Crop ground truth vectors to camera field of view using Shapely intersection.
        
        NEW IMPLEMENTATION: Uses Shapely geometric intersection (same approach as patch cropping).
        
        Benefits:
        - Automatically inserts intersection points at FOV boundaries
        - Preserves closed polygon topology when possible  
        - Handles complex edge cases (corners, tangents, etc.)
        - Consistent with patch cropping logic
        
        The workflow:
        1. Project polyline to image space
        2. Use Shapely's .intersection(fov_box) to clip
        3. Back-project result to world space
        4. Shapely automatically preserves topology (closed polygons remain closed!)
        
        Args:
            vectors: List of vector lists, where each vector is a polyline [N, 2]
            labels: Class labels for each vector group
            extrinsics: Camera extrinsics matrix (4x4)
            intrinsics: Camera intrinsics matrix (3x3)
            
        Returns:
            Tuple of (cropped_vectors, cropped_labels, boundary_intersection_points)
        """
        self.boundary_intersection_points = []
        
        cropped_vectors = []
        cropped_labels = []
        
        for vecs, label in zip(vectors, labels):
            for vector in vecs:
                if len(vector) < 2:
                    continue
                
                # Use new Shapely-based FOV cropping
                clipped_segments = self._crop_polyline_to_fov_shapely(
                    vector, extrinsics, intrinsics
                )
                
                # Filter valid segments
                valid_segments = []
                for segment in clipped_segments:
                    if len(segment) >= 2:
                        valid_segments.append(segment)
                        # Track boundary points for visualization
                        self.boundary_intersection_points.append(segment[0])
                        self.boundary_intersection_points.append(segment[-1])
                
                if valid_segments:
                    cropped_vectors.append(valid_segments)
                    cropped_labels.append(label)
        
        return cropped_vectors, cropped_labels, self.boundary_intersection_points
# ==================== GT EXTRACTION WITH MAPTR RESAMPLING ====================

def extract_gt_vectors(sample_info: Dict, nuscenes_path: str, pc_range: list, fixed_num: int = 20) -> Dict:
    """
    Extract GT vectors for visualization/evaluation with MapTR's resampling method.
    EXACT resampling logic from MapTR's fixed_num_sampled_points property.
    
    Args:
        sample_info: Sample info dict
        nuscenes_path: Path to NuScenes dataset
        pc_range: Point cloud range
        fixed_num: Number of points to resample to (MapTR default: 20)
        
    Returns:
        Dict with 'vectors' (list of numpy arrays) and 'labels' (list of ints)
    """
    # Setup patch size
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    patch_size = (patch_h, patch_w)
    
    # Create vector map
    vector_map = VectorizedLocalMap(
        nuscenes_path,
        patch_size=patch_size,
        map_classes=['divider', 'ped_crossing', 'boundary']
    )
    
    # Get transformation matrices
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = sample_info['lidar2ego_translation']
    
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = sample_info['ego2global_translation']
    
    lidar2global = ego2global @ lidar2ego
    lidar2global_translation = list(lidar2global[:3, 3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    
    # Extract location
    location = sample_info['map_location']
    
    # Generate vectorized map (returns Shapely LineString objects)
    anns_results = vector_map.gen_vectorized_samples(
        location, lidar2global_translation, lidar2global_rotation)
    
    gt_instances = anns_results['gt_vecs_pts_loc']  # Shapely LineString objects
    gt_labels = anns_results['gt_labels']
    
    # Resample using MapTR's EXACT method from fixed_num_sampled_points property
    # Reference: MapTR/projects/mmdet3d_plugin/datasets/nuscenes_map_dataset.py lines 147-148
    resampled_vectors = []
    final_labels = []
    
    for instance, label in zip(gt_instances, gt_labels):
        # EXACT MapTR resampling: use LineString.length and interpolate()
        distances = np.linspace(0, instance.length, fixed_num)
        sampled_points = np.array([list(instance.interpolate(distance).coords) 
                                   for distance in distances]).reshape(-1, 2)
        resampled_vectors.append(sampled_points)
        final_labels.append(label)
    
    return {
        'vectors': resampled_vectors,
        'labels': final_labels
    }


# ==================== CAMERA-SPECIFIC FOV CLIPPING AND ROTATION ====================

def extract_gt_with_fov_clipping(
    sample_info: Dict,
    nuscenes_path: str,
    pc_range: list,
    camera_name: str = 'CAM_FRONT',
    fixed_num: int = 20,
    apply_clipping: bool = True
) -> Dict:
    """
    Extract GT vectors with optional camera-specific FOV clipping and rotation.
    EXACT logic from training_6pv_enhance/visualize_per_camera_with_correct_fov_backup.py
    
    Processing pipeline:
    1. Extract GT vectors from map with 20-point resampling (MapTR standard)
    2. Optionally apply camera-specific FOV clipping
    3. Rotate to camera-centric coordinates (camera forward = +Y)
    
    Args:
        sample_info: Sample info dict
        nuscenes_path: Path to NuScenes data
        pc_range: BEV range
        camera_name: Camera name (e.g., 'CAM_FRONT')
        fixed_num: Number of points for initial resampling (default: 20)
        apply_clipping: If True, apply FOV clipping; if False, skip clipping (default: True)
        
    Returns:
        Dict with 'vectors' and 'labels' in camera-centric coordinate system
    """
    # Step 1: Extract GT vectors with MapTR's 20-point resampling
    gt_data = extract_gt_vectors(sample_info, nuscenes_path, pc_range, fixed_num=fixed_num)
    vectors = gt_data['vectors']  # List of [20, 2] numpy arrays
    gt_labels = gt_data['labels']
    
    if len(vectors) == 0:
        return {'vectors': [], 'labels': []}
    
    # Step 2: Get camera extrinsics and intrinsics (EXACT training logic)
    cam_info = sample_info['cams'][camera_name]
    cam_intrinsic = np.array(cam_info['cam_intrinsic'])
    
    # Convert quaternion format: nuScenes uses [w,x,y,z], scipy uses [x,y,z,w]
    quat_wxyz = cam_info['sensor2ego_rotation']
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    sensor2ego_rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = sensor2ego_rot_matrix
    cam2ego[:3, 3] = cam_info['sensor2ego_translation']
    
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = sample_info['ego2global_translation']
    
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = sample_info['lidar2ego_translation']
    
    lidar2global = ego2global @ lidar2ego
    cam2global = ego2global @ cam2ego
    
    # Get lidar2global rotation for BEV alignment
    lidar2global_translation = list(lidar2global[:3, 3])
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
        # Apply FOV clipping (EXACT training logic)
        lidar_height = sample_info['lidar2ego_translation'][2]
        clipper = CameraFOVClipper(image_size=(900, 1600), lidar_height_above_ground=lidar_height)
        
        # Group vectors by label for FOV clipper
        vectors_by_label = {}
        for vec, label in zip(vectors, gt_labels):
            if label not in vectors_by_label:
                vectors_by_label[label] = []
            vectors_by_label[label].append(vec)
        
        vectors_list = []
        labels_list = []
        for label, vecs in vectors_by_label.items():
            vectors_list.append(vecs)
            labels_list.append(label)
        
        cropped_vectors, cropped_labels, _ = clipper.crop_vectors_to_fov(
            vectors_list, labels_list, cam_extrinsics_bev, cam_intrinsic
        )
        
        if len(cropped_vectors) == 0:
            return {'vectors': [], 'labels': []}
    else:
        # Skip clipping - keep all vectors, group by label for consistent structure
        vectors_by_label = {}
        for vec, label in zip(vectors, gt_labels):
            if label not in vectors_by_label:
                vectors_by_label[label] = []
            vectors_by_label[label].append(vec)
        
        cropped_vectors = []
        cropped_labels = []
        for label, vecs in vectors_by_label.items():
            cropped_vectors.append(vecs)
            cropped_labels.append(label)
    
    # Step 4: Rotate to camera-centric coordinates (EXACT training logic)
    # This aligns the camera's forward direction to +Y (upward in visualization)
    cam_pos_3d = cam_extrinsics_bev[:3, 3]
    cam_forward_3d = cam_extrinsics_bev[:3, :3] @ np.array([0, 0, 1])  # Camera forward (+Z)
    
    # Get angle of camera forward direction in XY plane
    cam_forward_angle = np.arctan2(cam_forward_3d[0], cam_forward_3d[1])  # angle from +Y axis
    
    # Rotation matrix to align camera forward with +Y (upward in BEV)
    cos_rot = np.cos(cam_forward_angle)
    sin_rot = np.sin(cam_forward_angle)
    rotation_to_camera_view = np.array([
        [cos_rot, -sin_rot],
        [sin_rot, cos_rot]
    ])
    
    # Apply rotation to all cropped vectors and resample to fixed_num points
    final_vectors = []
    final_labels = []
    
    for vecs, label in zip(cropped_vectors, cropped_labels):
        for vector in vecs:
            if len(vector) >= 2:
                # Rotate 2D points: (rotation_matrix @ points.T).T
                vector_rotated = (rotation_to_camera_view @ vector.T).T
                
                # Resample to fixed_num points after FOV clipping (using LineString interpolation)
                line = LineString(vector_rotated)
                if line.length > 0:
                    distances = np.linspace(0, line.length, fixed_num)
                    resampled_points = np.array([list(line.interpolate(distance).coords) 
                                                for distance in distances]).reshape(-1, 2)
                    final_vectors.append(resampled_points)
                    final_labels.append(label)
                else:
                    # Degenerate case: line has zero length, just replicate the point
                    final_vectors.append(np.tile(vector_rotated[0], (fixed_num, 1)))
                    final_labels.append(label)
    
    return {
        'vectors': final_vectors,
        'labels': final_labels
    }


def process_predictions_with_fov_clipping(
    pred_vectors: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    sample_info: Dict,
    nuscenes_path: str,
    pc_range: list,
    camera_name: str = 'CAM_FRONT',
    apply_clipping: bool = True
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Apply optional camera-specific FOV clipping and rotation to predictions.
    EXACT same logic as GT processing - ensures fair comparison.
    
    Args:
        pred_vectors: [N, num_pts, 2] prediction vectors in BEV coordinates
        pred_labels: [N] prediction labels
        pred_scores: [N] prediction scores
        sample_info: Sample info dict
        nuscenes_path: Path to NuScenes data
        pc_range: BEV range
        camera_name: Camera name
        apply_clipping: If True, apply FOV clipping; if False, skip clipping (default: True)
        
    Returns:
        Tuple of (vectors, labels, scores) with same processing as GT
    """
    if len(pred_vectors) == 0:
        return [], [], []
    
    # Get camera extrinsics and intrinsics (EXACT same as GT processing)
    cam_info = sample_info['cams'][camera_name]
    cam_intrinsic = np.array(cam_info['cam_intrinsic'])
    
    quat_wxyz = cam_info['sensor2ego_rotation']
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    sensor2ego_rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = sensor2ego_rot_matrix
    cam2ego[:3, 3] = cam_info['sensor2ego_translation']
    
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = sample_info['ego2global_translation']
    
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = sample_info['lidar2ego_translation']
    
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
    
    # Calculate rotation to camera-centric coordinates
    cam_forward_3d = cam_extrinsics_bev[:3, :3] @ np.array([0, 0, 1])
    cam_forward_angle = np.arctan2(cam_forward_3d[0], cam_forward_3d[1])
    
    cos_rot = np.cos(cam_forward_angle)
    sin_rot = np.sin(cam_forward_angle)
    rotation_to_camera_view = np.array([
        [cos_rot, -sin_rot],
        [sin_rot, cos_rot]
    ])
    
    # Process predictions with optional FOV clipping
    final_vectors = []
    final_labels = []
    final_scores = []
    
    if apply_clipping:
        # Apply FOV clipping to each prediction individually
        lidar_height = sample_info['lidar2ego_translation'][2]
        clipper = CameraFOVClipper(image_size=(900, 1600), lidar_height_above_ground=lidar_height)
        
        for vec, label, score in zip(pred_vectors, pred_labels, pred_scores):
            # Treat each prediction as a single-vector group
            vectors_list = [[vec]]
            labels_list = [label]
            
            # Apply FOV clipping
            cropped_vectors, cropped_labels, _ = clipper.crop_vectors_to_fov(
                vectors_list, labels_list, cam_extrinsics_bev, cam_intrinsic
            )
            
            # If this vector survived clipping, rotate and resample it
            if len(cropped_vectors) > 0 and len(cropped_vectors[0]) > 0:
                cropped_vec = cropped_vectors[0][0]  # Get the single vector
                if len(cropped_vec) >= 2:
                    # Rotate to camera-centric coordinates
                    vector_rotated = (rotation_to_camera_view @ cropped_vec.T).T
                    
                    # Resample to fixed_num=20 points after FOV clipping
                    line = LineString(vector_rotated)
                    if line.length > 0:
                        distances = np.linspace(0, line.length, 20)
                        resampled_points = np.array([list(line.interpolate(distance).coords) 
                                                    for distance in distances]).reshape(-1, 2)
                        final_vectors.append(resampled_points)
                        final_labels.append(label)
                        final_scores.append(score)
                    else:
                        # Degenerate case: replicate point
                        final_vectors.append(np.tile(vector_rotated[0], (20, 1)))
                        final_labels.append(label)
                        final_scores.append(score)
    else:
        # Skip clipping - just rotate and resample all predictions
        for vec, label, score in zip(pred_vectors, pred_labels, pred_scores):
            if len(vec) >= 2:
                # Rotate to camera-centric coordinates
                vector_rotated = (rotation_to_camera_view @ vec.T).T
                
                # Resample to 20 points for consistency
                line = LineString(vector_rotated)
                if line.length > 0:
                    distances = np.linspace(0, line.length, 20)
                    resampled_points = np.array([list(line.interpolate(distance).coords) 
                                                for distance in distances]).reshape(-1, 2)
                    final_vectors.append(resampled_points)
                    final_labels.append(label)
                    final_scores.append(score)
                else:
                    # Degenerate case
                    final_vectors.append(np.tile(vector_rotated[0], (20, 1)))
                    final_labels.append(label)
                    final_scores.append(score)
    
    return final_vectors, final_labels, final_scores

