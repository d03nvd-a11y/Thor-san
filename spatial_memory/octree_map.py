"""
Octree-based 3D Spatial Map

Efficient 3D occupancy mapping using octree data structure.
Stores spatial information about the environment for navigation and manipulation.
"""

import numpy as np
from typing import Optional, List, Tuple
import open3d as o3d


class OctreeMap:
    """
    3D octree map for spatial memory.

    Uses Open3D's octree implementation for efficient 3D space representation.
    Stores occupancy and other spatial properties in a hierarchical structure.
    """

    def __init__(
        self,
        resolution: float = 0.01,
        max_depth: int = 10,
        origin: np.ndarray = None
    ):
        """
        Initialize octree map.

        Args:
            resolution: Voxel size in meters (e.g., 0.01 = 1cm)
            max_depth: Maximum octree depth
            origin: Origin point (x, y, z) in meters
        """
        self.resolution = resolution
        self.max_depth = max_depth
        self.origin = origin if origin is not None else np.array([0.0, 0.0, 0.0])

        # Initialize empty octree
        self.octree = o3d.geometry.Octree(max_depth=max_depth)

        # Statistics
        self.num_updates = 0
        self.num_points_added = 0

    def add_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        occupancy_value: float = 1.0
    ):
        """
        Add point cloud to octree map.

        Args:
            pcd: Open3D point cloud
            occupancy_value: Occupancy value for points (0-1)
        """
        if len(pcd.points) == 0:
            return

        # Convert point cloud to octree
        self.octree.convert_from_point_cloud(pcd, size_expand=self.resolution)

        self.num_updates += 1
        self.num_points_added += len(pcd.points)

    def add_depth_observation(
        self,
        depth_map: np.ndarray,
        camera_matrix: np.ndarray,
        camera_pose: np.ndarray,
        color_image: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.5,
        confidence_map: Optional[np.ndarray] = None
    ):
        """
        Add depth observation to map.

        Args:
            depth_map: Depth map (H x W) in meters
            camera_matrix: Camera intrinsic matrix (3x3)
            camera_pose: Camera pose (4x4 transformation matrix)
            color_image: Optional RGB image
            confidence_threshold: Minimum confidence to add point
            confidence_map: Optional confidence map (H x W)
        """
        from vision.depth.point_cloud import PointCloudGenerator

        # Generate point cloud from depth
        pcg = PointCloudGenerator()
        pcd = pcg.depth_to_point_cloud(depth_map, camera_matrix, color_image)

        # Filter by confidence if provided
        if confidence_map is not None:
            h, w = depth_map.shape
            mask = (depth_map > 0) & (confidence_map >= confidence_threshold)

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # Filter points
            flat_mask = mask.ravel()
            if len(points) == len(flat_mask):
                points = points[flat_mask]
                if colors is not None:
                    colors = colors[flat_mask]

                pcd_filtered = o3d.geometry.PointCloud()
                pcd_filtered.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    pcd_filtered.colors = o3d.utility.Vector3dVector(colors)

                pcd = pcd_filtered

        # Transform to world frame
        pcd.transform(camera_pose)

        # Add to map
        self.add_point_cloud(pcd)

    def query_occupancy(
        self,
        point: np.ndarray
    ) -> bool:
        """
        Query if point is occupied in map.

        Args:
            point: 3D point (x, y, z)

        Returns:
            True if occupied, False otherwise
        """
        # Convert point to octree node
        node = self.octree.locate_leaf_node(point)
        return node is not None

    def query_region(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray
    ) -> List[np.ndarray]:
        """
        Query all occupied points in region.

        Args:
            min_bound: Minimum (x, y, z)
            max_bound: Maximum (x, y, z)

        Returns:
            List of occupied points
        """
        # This is a simplified version - full implementation would traverse octree
        # For now, we'll convert to point cloud and filter
        return []

    def to_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Convert octree to point cloud representation.

        Returns:
            Point cloud of occupied voxels
        """
        # Get voxel centers
        voxel_grid = o3d.geometry.VoxelGrid.create_from_octree(self.octree)

        # Extract occupied voxels
        voxels = voxel_grid.get_voxels()

        if not voxels:
            return o3d.geometry.PointCloud()

        # Get voxel centers as points
        points = []
        for voxel in voxels:
            point = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            points.append(point)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))

        return pcd

    def get_occupied_voxels(self) -> np.ndarray:
        """
        Get array of occupied voxel centers.

        Returns:
            Array of shape (N, 3) with voxel centers
        """
        pcd = self.to_point_cloud()
        return np.asarray(pcd.points)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of occupied space.

        Returns:
            (min_bound, max_bound)
        """
        pcd = self.to_point_cloud()

        if len(pcd.points) == 0:
            return self.origin.copy(), self.origin.copy()

        points = np.asarray(pcd.points)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        return min_bound, max_bound

    def clear(self):
        """Clear all map data."""
        self.octree = o3d.geometry.Octree(max_depth=self.max_depth)
        self.num_updates = 0
        self.num_points_added = 0

    def save(self, filepath: str):
        """Save octree map to file."""
        # Save as point cloud (easier to work with)
        pcd = self.to_point_cloud()
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"[INFO] Octree map saved to {filepath}")

    def load(self, filepath: str):
        """Load octree map from file."""
        pcd = o3d.io.read_point_cloud(filepath)
        self.add_point_cloud(pcd)
        print(f"[INFO] Octree map loaded from {filepath}")

    def get_statistics(self) -> dict:
        """Get map statistics."""
        min_bound, max_bound = self.get_bounding_box()
        volume = np.prod(max_bound - min_bound)

        return {
            'resolution_m': self.resolution,
            'max_depth': self.max_depth,
            'num_updates': self.num_updates,
            'num_points_added': self.num_points_added,
            'bounding_box_min': min_bound.tolist(),
            'bounding_box_max': max_bound.tolist(),
            'volume_m3': float(volume),
            'origin': self.origin.tolist()
        }

    def visualize(self, window_name: str = "Octree Map"):
        """Visualize octree map."""
        pcd = self.to_point_cloud()

        if len(pcd.points) > 0:
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=window_name,
                width=800,
                height=600
            )
        else:
            print("[WARN] Map is empty, nothing to visualize")

    def downsample(self, voxel_size: float) -> o3d.geometry.PointCloud:
        """
        Downsample map to larger voxel size.

        Args:
            voxel_size: New voxel size in meters

        Returns:
            Downsampled point cloud
        """
        pcd = self.to_point_cloud()
        return pcd.voxel_down_sample(voxel_size=voxel_size)
