"""
Grasp Planner

Generates grasp poses for object manipulation.
Considers object geometry and approach constraints.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import open3d as o3d


@dataclass
class GraspPose:
    """6-DOF grasp pose for robot end-effector."""
    position: np.ndarray  # (x, y, z)
    orientation: np.ndarray  # Quaternion (x, y, z, w) or rotation matrix (3x3)
    quality: float  # Grasp quality score [0, 1]
    approach_direction: np.ndarray  # Unit vector
    width: float  # Required gripper width (meters)


class GraspPlanner:
    """
    Grasp pose generation for manipulation.

    Generates stable grasp configurations for detected objects.
    """

    def __init__(
        self,
        approach_distance: float = 0.1,
        clearance: float = 0.02,
        min_gripper_width: float = 0.0,
        max_gripper_width: float = 0.08
    ):
        """
        Initialize grasp planner.

        Args:
            approach_distance: Distance for pre-grasp approach (meters)
            clearance: Safety clearance around object (meters)
            min_gripper_width: Minimum gripper opening (meters)
            max_gripper_width: Maximum gripper opening (meters)
        """
        self.approach_distance = approach_distance
        self.clearance = clearance
        self.min_gripper_width = min_gripper_width
        self.max_gripper_width = max_gripper_width

    def generate_grasps(
        self,
        point_cloud: o3d.geometry.PointCloud,
        num_grasps: int = 10
    ) -> List[GraspPose]:
        """
        Generate candidate grasp poses for object point cloud.

        Args:
            point_cloud: Object point cloud
            num_grasps: Number of grasp candidates to generate

        Returns:
            List of grasp poses sorted by quality
        """
        if len(point_cloud.points) < 10:
            return []

        # Compute normals if not present
        if not point_cloud.has_normals():
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05,
                    max_nn=30
                )
            )

        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        # Compute object center and principal axes
        center = np.mean(points, axis=0)
        pca_axes = self._compute_pca_axes(points)

        grasps = []

        # Strategy 1: Grasp along principal axes
        for axis in pca_axes:
            grasp = self._generate_axis_aligned_grasp(points, center, axis)
            if grasp:
                grasps.append(grasp)

        # Strategy 2: Sample points and generate antipodal grasps
        sample_indices = np.random.choice(
            len(points),
            min(num_grasps * 2, len(points)),
            replace=False
        )

        for idx in sample_indices:
            point = points[idx]
            normal = normals[idx]

            grasp = self._generate_antipodal_grasp(
                points,
                point,
                normal,
                center
            )

            if grasp:
                grasps.append(grasp)

        # Sort by quality
        grasps.sort(key=lambda g: g.quality, reverse=True)

        return grasps[:num_grasps]

    def _compute_pca_axes(self, points: np.ndarray) -> np.ndarray:
        """
        Compute principal axes of point cloud using PCA.

        Returns:
            Principal axes (3x3 matrix, each row is an axis)
        """
        centered = points - np.mean(points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx].T

        return eigenvectors

    def _generate_axis_aligned_grasp(
        self,
        points: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray
    ) -> Optional[GraspPose]:
        """
        Generate grasp aligned with principal axis.

        Approach from above or sides.
        """
        # Grasp position at center
        position = center.copy()

        # Approach direction along axis
        approach_direction = axis / np.linalg.norm(axis)

        # Compute required gripper width
        # Project points onto axis and find span
        projections = np.dot(points - center, approach_direction)
        width = np.ptp(projections) + 2 * self.clearance

        # Check if graspable
        if width < self.min_gripper_width or width > self.max_gripper_width:
            return None

        # Create rotation matrix (approach direction as z-axis)
        orientation = self._create_orientation_matrix(approach_direction)

        # Quality based on width (prefer smaller widths)
        quality = 1.0 - (width - self.min_gripper_width) / (
            self.max_gripper_width - self.min_gripper_width
        )
        quality = np.clip(quality, 0.3, 1.0)

        return GraspPose(
            position=position,
            orientation=orientation,
            quality=quality,
            approach_direction=approach_direction,
            width=width
        )

    def _generate_antipodal_grasp(
        self,
        points: np.ndarray,
        point: np.ndarray,
        normal: np.ndarray,
        center: np.ndarray
    ) -> Optional[GraspPose]:
        """
        Generate antipodal grasp at surface point.

        Grasp perpendicular to surface normal.
        """
        # Approach direction is surface normal
        approach_direction = normal / np.linalg.norm(normal)

        # Find opposite point (antipodal)
        # Cast ray in opposite direction
        opposite_direction = -approach_direction
        opposite_point = self._find_opposite_point(
            points,
            point,
            opposite_direction,
            max_distance=self.max_gripper_width
        )

        if opposite_point is None:
            return None

        # Grasp position at midpoint
        position = (point + opposite_point) / 2.0

        # Gripper width
        width = np.linalg.norm(opposite_point - point) + 2 * self.clearance

        if width < self.min_gripper_width or width > self.max_gripper_width:
            return None

        # Orientation
        orientation = self._create_orientation_matrix(approach_direction)

        # Quality based on:
        # 1. Distance from center (prefer central grasps)
        # 2. Alignment with surface normal
        # 3. Gripper width
        center_distance = np.linalg.norm(position - center)
        max_center_distance = np.linalg.norm(points - center, axis=1).max()

        quality_center = 1.0 - (center_distance / (max_center_distance + 1e-6))
        quality_width = 1.0 - (width - self.min_gripper_width) / (
            self.max_gripper_width - self.min_gripper_width
        )

        quality = 0.5 * quality_center + 0.5 * quality_width
        quality = np.clip(quality, 0.1, 1.0)

        return GraspPose(
            position=position,
            orientation=orientation,
            quality=quality,
            approach_direction=approach_direction,
            width=width
        )

    def _find_opposite_point(
        self,
        points: np.ndarray,
        start_point: np.ndarray,
        direction: np.ndarray,
        max_distance: float
    ) -> Optional[np.ndarray]:
        """
        Find point on opposite side along direction.

        Uses nearest point along ray.
        """
        # Project all points onto ray
        to_points = points - start_point
        projections = np.dot(to_points, direction)

        # Filter points in direction and within max distance
        valid_mask = (projections > 0) & (projections < max_distance)

        if not np.any(valid_mask):
            return None

        valid_projections = projections[valid_mask]
        valid_points = points[valid_mask]

        # Find closest point along ray
        closest_idx = np.argmax(valid_projections)
        opposite_point = valid_points[closest_idx]

        return opposite_point

    def _create_orientation_matrix(
        self,
        approach_direction: np.ndarray
    ) -> np.ndarray:
        """
        Create rotation matrix with approach direction as z-axis.

        Returns:
            3x3 rotation matrix
        """
        z_axis = approach_direction / np.linalg.norm(approach_direction)

        # Choose x-axis perpendicular to z
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))

        x_axis = x_axis / np.linalg.norm(x_axis)

        # y-axis completes right-handed frame
        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix
        R = np.column_stack([x_axis, y_axis, z_axis])

        return R

    def get_pre_grasp_pose(self, grasp: GraspPose) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get pre-grasp pose (before final approach).

        Returns:
            (position, orientation)
        """
        pre_grasp_position = grasp.position - grasp.approach_direction * self.approach_distance
        return pre_grasp_position, grasp.orientation

    def visualize_grasp(
        self,
        grasp: GraspPose,
        point_cloud: Optional[o3d.geometry.PointCloud] = None
    ):
        """
        Visualize grasp pose with coordinate frame.

        Args:
            grasp: Grasp pose to visualize
            point_cloud: Optional object point cloud
        """
        geometries = []

        # Add point cloud
        if point_cloud is not None:
            geometries.append(point_cloud)

        # Create coordinate frame at grasp pose
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05,
            origin=grasp.position
        )

        # Rotate frame to grasp orientation
        if grasp.orientation.shape == (3, 3):
            frame.rotate(grasp.orientation, center=grasp.position)

        geometries.append(frame)

        # Create approach vector
        approach_end = grasp.position + grasp.approach_direction * self.approach_distance
        approach_line = o3d.geometry.LineSet()
        approach_line.points = o3d.utility.Vector3dVector([
            grasp.position,
            approach_end
        ])
        approach_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        approach_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red

        geometries.append(approach_line)

        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Grasp (quality={grasp.quality:.2f})"
        )

    def filter_grasps_by_collision(
        self,
        grasps: List[GraspPose],
        scene_point_cloud: o3d.geometry.PointCloud,
        collision_threshold: float = 0.01
    ) -> List[GraspPose]:
        """
        Filter grasps that would collide with environment.

        Args:
            grasps: List of candidate grasps
            scene_point_cloud: Full scene point cloud
            collision_threshold: Collision distance threshold (meters)

        Returns:
            Filtered list of collision-free grasps
        """
        # Simplified collision check
        # Real implementation would check gripper geometry

        scene_points = np.asarray(scene_point_cloud.points)
        collision_free_grasps = []

        for grasp in grasps:
            # Check if any scene points are too close to grasp position
            distances = np.linalg.norm(scene_points - grasp.position, axis=1)
            min_distance = np.min(distances)

            if min_distance > collision_threshold:
                collision_free_grasps.append(grasp)

        return collision_free_grasps
