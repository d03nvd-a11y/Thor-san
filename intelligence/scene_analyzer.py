"""
Scene Analyzer

High-level scene understanding from 3D spatial data.
Detects surfaces, segments regions, and interprets spatial layout.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import open3d as o3d


@dataclass
class Surface:
    """Detected planar surface in scene."""
    plane_model: np.ndarray  # (a, b, c, d) where ax + by + cz + d = 0
    inlier_points: np.ndarray  # Points on surface (N x 3)
    center: np.ndarray  # Surface center (x, y, z)
    normal: np.ndarray  # Surface normal (nx, ny, nz)
    area: float  # Approximate surface area (m^2)
    surface_type: str  # "horizontal", "vertical", "inclined"


@dataclass
class WorkspaceBounds:
    """Workspace boundaries for manipulation."""
    min_bound: np.ndarray  # (x, y, z)
    max_bound: np.ndarray  # (x, y, z)
    reachable_volume: float  # Volume in m^3


class SceneAnalyzer:
    """
    High-level scene understanding and analysis.

    Extracts semantic information from 3D point clouds and spatial maps.
    """

    def __init__(
        self,
        plane_threshold: float = 0.01,
        min_plane_points: int = 100
    ):
        """
        Initialize scene analyzer.

        Args:
            plane_threshold: RANSAC threshold for plane detection (meters)
            min_plane_points: Minimum points to consider a plane
        """
        self.plane_threshold = plane_threshold
        self.min_plane_points = min_plane_points

    def detect_surfaces(
        self,
        point_cloud: o3d.geometry.PointCloud,
        max_surfaces: int = 5
    ) -> List[Surface]:
        """
        Detect planar surfaces in point cloud.

        Uses RANSAC to find dominant planes (tables, walls, floors).

        Args:
            point_cloud: Input point cloud
            max_surfaces: Maximum number of surfaces to detect

        Returns:
            List of detected surfaces
        """
        surfaces = []
        pcd = point_cloud

        for _ in range(max_surfaces):
            if len(pcd.points) < self.min_plane_points:
                break

            # RANSAC plane segmentation
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.plane_threshold,
                ransac_n=3,
                num_iterations=1000
            )

            if len(inliers) < self.min_plane_points:
                break

            # Extract plane parameters
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)  # Normalize

            # Get inlier points
            inlier_cloud = pcd.select_by_index(inliers)
            inlier_points = np.asarray(inlier_cloud.points)

            # Compute center
            center = np.mean(inlier_points, axis=0)

            # Estimate area (using convex hull projection)
            area = self._estimate_surface_area(inlier_points, normal)

            # Classify surface type
            surface_type = self._classify_surface(normal)

            surface = Surface(
                plane_model=plane_model,
                inlier_points=inlier_points,
                center=center,
                normal=normal,
                area=area,
                surface_type=surface_type
            )

            surfaces.append(surface)

            # Remove inliers for next iteration
            pcd = pcd.select_by_index(inliers, invert=True)

        return surfaces

    def _estimate_surface_area(
        self,
        points: np.ndarray,
        normal: np.ndarray
    ) -> float:
        """
        Estimate surface area from points.

        Uses bounding rectangle approximation.
        """
        if len(points) < 3:
            return 0.0

        # Create coordinate system on plane
        # Choose two perpendicular vectors in plane
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, np.array([0, 0, 1]))
        else:
            u = np.cross(normal, np.array([1, 0, 0]))

        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        # Project points to 2D plane coordinates
        center = np.mean(points, axis=0)
        points_centered = points - center

        u_coords = np.dot(points_centered, u)
        v_coords = np.dot(points_centered, v)

        # Bounding rectangle area
        width = np.ptp(u_coords)  # Peak-to-peak
        height = np.ptp(v_coords)
        area = width * height

        return area

    def _classify_surface(self, normal: np.ndarray) -> str:
        """
        Classify surface orientation.

        Returns:
            "horizontal", "vertical", or "inclined"
        """
        # Angle with vertical (z-axis)
        vertical_angle = np.arccos(np.abs(normal[2]))
        vertical_angle_deg = np.degrees(vertical_angle)

        if vertical_angle_deg < 15:
            return "horizontal"
        elif vertical_angle_deg > 75:
            return "vertical"
        else:
            return "inclined"

    def find_support_surfaces(
        self,
        surfaces: List[Surface]
    ) -> List[Surface]:
        """
        Find horizontal surfaces that can support objects (tables, shelves).

        Args:
            surfaces: List of detected surfaces

        Returns:
            List of support surfaces
        """
        support_surfaces = []

        for surface in surfaces:
            # Must be horizontal
            if surface.surface_type == "horizontal":
                # Must face upward (normal points up)
                if surface.normal[2] > 0.8:
                    # Must have reasonable size
                    if surface.area > 0.01:  # 100 cm^2
                        support_surfaces.append(surface)

        # Sort by height (lowest first)
        support_surfaces.sort(key=lambda s: s.center[2])

        return support_surfaces

    def compute_workspace_bounds(
        self,
        point_cloud: o3d.geometry.PointCloud,
        robot_position: np.ndarray = np.array([0, 0, 0]),
        max_reach: float = 0.8
    ) -> WorkspaceBounds:
        """
        Compute reachable workspace bounds for robot.

        Args:
            point_cloud: Scene point cloud
            robot_position: Robot base position (x, y, z)
            max_reach: Maximum reach distance (meters)

        Returns:
            WorkspaceBounds
        """
        points = np.asarray(point_cloud.points)

        if len(points) == 0:
            return WorkspaceBounds(
                min_bound=robot_position - max_reach,
                max_bound=robot_position + max_reach,
                reachable_volume=0.0
            )

        # Filter points within reach
        distances = np.linalg.norm(points - robot_position, axis=1)
        reachable_mask = distances <= max_reach

        reachable_points = points[reachable_mask]

        if len(reachable_points) == 0:
            return WorkspaceBounds(
                min_bound=robot_position - max_reach,
                max_bound=robot_position + max_reach,
                reachable_volume=0.0
            )

        # Compute bounds
        min_bound = np.min(reachable_points, axis=0)
        max_bound = np.max(reachable_points, axis=0)

        # Compute volume
        volume = np.prod(max_bound - min_bound)

        return WorkspaceBounds(
            min_bound=min_bound,
            max_bound=max_bound,
            reachable_volume=volume
        )

    def detect_obstacles(
        self,
        point_cloud: o3d.geometry.PointCloud,
        workspace_bounds: WorkspaceBounds,
        ground_height: float = 0.0
    ) -> o3d.geometry.PointCloud:
        """
        Extract obstacle points in workspace.

        Args:
            point_cloud: Full scene point cloud
            workspace_bounds: Workspace boundaries
            ground_height: Ground plane height (z)

        Returns:
            Point cloud of obstacles
        """
        points = np.asarray(point_cloud.points)

        # Filter by workspace bounds
        in_workspace = (
            (points[:, 0] >= workspace_bounds.min_bound[0]) &
            (points[:, 0] <= workspace_bounds.max_bound[0]) &
            (points[:, 1] >= workspace_bounds.min_bound[1]) &
            (points[:, 1] <= workspace_bounds.max_bound[1]) &
            (points[:, 2] >= workspace_bounds.min_bound[2]) &
            (points[:, 2] <= workspace_bounds.max_bound[2])
        )

        # Remove ground plane
        above_ground = points[:, 2] > ground_height + 0.02  # 2cm above ground

        obstacle_mask = in_workspace & above_ground
        obstacle_points = points[obstacle_mask]

        # Create obstacle point cloud
        obstacle_pcd = o3d.geometry.PointCloud()
        obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)

        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
            obstacle_pcd.colors = o3d.utility.Vector3dVector(colors[obstacle_mask])

        return obstacle_pcd

    def cluster_objects(
        self,
        point_cloud: o3d.geometry.PointCloud,
        eps: float = 0.05,
        min_points: int = 10
    ) -> List[o3d.geometry.PointCloud]:
        """
        Cluster point cloud into separate objects using DBSCAN.

        Args:
            point_cloud: Input point cloud
            eps: DBSCAN epsilon (clustering distance)
            min_points: Minimum points per cluster

        Returns:
            List of clustered point clouds
        """
        # DBSCAN clustering
        labels = np.array(point_cloud.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        ))

        # Extract clusters
        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Noise
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = point_cloud.select_by_index(cluster_indices)
            clusters.append(cluster_pcd)

        return clusters

    def analyze_scene(
        self,
        point_cloud: o3d.geometry.PointCloud
    ) -> Dict:
        """
        Comprehensive scene analysis.

        Returns:
            Dictionary with scene analysis results
        """
        # Detect surfaces
        surfaces = self.detect_surfaces(point_cloud)
        support_surfaces = self.find_support_surfaces(surfaces)

        # Compute workspace
        workspace = self.compute_workspace_bounds(point_cloud)

        # Detect ground plane
        ground_height = 0.0
        if surfaces:
            # Lowest horizontal surface is likely ground
            horizontal = [s for s in surfaces if s.surface_type == "horizontal"]
            if horizontal:
                ground_height = min(s.center[2] for s in horizontal)

        # Detect obstacles
        obstacles = self.detect_obstacles(point_cloud, workspace, ground_height)

        # Cluster objects
        clusters = self.cluster_objects(obstacles)

        return {
            'num_surfaces': len(surfaces),
            'num_support_surfaces': len(support_surfaces),
            'support_surfaces': support_surfaces,
            'workspace_bounds': workspace,
            'ground_height': ground_height,
            'num_obstacles': len(obstacles.points),
            'num_object_clusters': len(clusters),
            'object_clusters': clusters
        }
