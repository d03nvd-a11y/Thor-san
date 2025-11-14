"""
Point Cloud Generation

Creates 3D point clouds from depth maps and stereo images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import open3d as o3d


class PointCloudGenerator:
    """
    Generates 3D point clouds from depth maps.

    Supports both RGB-D and stereo reconstruction.
    """

    def __init__(self):
        """Initialize point cloud generator."""
        pass

    def depth_to_point_cloud(
        self,
        depth_map: np.ndarray,
        camera_matrix: np.ndarray,
        color_image: Optional[np.ndarray] = None,
        depth_scale: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """
        Convert depth map to 3D point cloud.

        Args:
            depth_map: Depth map (meters)
            camera_matrix: Camera intrinsic matrix (3x3)
            color_image: Optional RGB image for coloring
            depth_scale: Scale factor for depth values

        Returns:
            Open3D point cloud
        """
        h, w = depth_map.shape

        # Create camera intrinsic for Open3D
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # Convert depth to Open3D format (uint16 in millimeters)
        depth_o3d = o3d.geometry.Image((depth_map * 1000 / depth_scale).astype(np.uint16))

        # Create RGBD image if color provided
        if color_image is not None:
            if len(color_image.shape) == 3:
                color_o3d = o3d.geometry.Image(
                    cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                )
            else:
                color_o3d = o3d.geometry.Image(
                    cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
                )

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1000.0 / depth_scale,
                depth_trunc=10.0,
                convert_rgb_to_intensity=False
            )

            # Generate point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic
            )
        else:
            # Generate point cloud without color
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsic,
                depth_scale=1000.0 / depth_scale,
                depth_trunc=10.0
            )

        return pcd

    def stereo_to_point_cloud(
        self,
        img_left: np.ndarray,
        disparity: np.ndarray,
        Q: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Create point cloud from stereo disparity using Q matrix.

        Args:
            img_left: Left RGB image
            disparity: Disparity map
            Q: Reprojection matrix (4x4) from stereo calibration

        Returns:
            Open3D point cloud
        """
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q)

        # Create mask for valid points
        mask = disparity > 0

        # Extract valid 3D points
        points = points_3d[mask]

        # Extract colors
        if len(img_left.shape) == 3:
            colors = img_left[mask].astype(np.float32) / 255.0
            colors = colors[:, [2, 1, 0]]  # BGR to RGB
        else:
            gray = img_left[mask].astype(np.float32) / 255.0
            colors = np.stack([gray, gray, gray], axis=1)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def filter_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.005,
        remove_outliers: bool = True,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> o3d.geometry.PointCloud:
        """
        Filter and downsample point cloud.

        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling (meters)
            remove_outliers: Remove statistical outliers
            nb_neighbors: Number of neighbors for outlier removal
            std_ratio: Standard deviation ratio for outlier removal

        Returns:
            Filtered point cloud
        """
        # Voxel downsampling
        pcd_filtered = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Statistical outlier removal
        if remove_outliers:
            pcd_filtered, _ = pcd_filtered.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )

        return pcd_filtered

    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float = 0.05,
        max_nn: int = 30
    ) -> o3d.geometry.PointCloud:
        """
        Estimate surface normals for point cloud.

        Args:
            pcd: Input point cloud
            radius: Search radius for normal estimation
            max_nn: Maximum number of neighbors

        Returns:
            Point cloud with normals
        """
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn
            )
        )

        # Orient normals towards camera
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

        return pcd

    def save_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        filepath: str
    ):
        """
        Save point cloud to file.

        Supports: .ply, .pcd, .xyz, .xyzrgb

        Args:
            pcd: Point cloud
            filepath: Output file path
        """
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"[INFO] Point cloud saved to {filepath}")

    def load_point_cloud(self, filepath: str) -> o3d.geometry.PointCloud:
        """Load point cloud from file."""
        pcd = o3d.io.read_point_cloud(filepath)
        print(f"[INFO] Point cloud loaded from {filepath} ({len(pcd.points)} points)")
        return pcd

    def visualize_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        window_name: str = "Point Cloud"
    ):
        """
        Visualize point cloud in Open3D viewer.

        Args:
            pcd: Point cloud
            window_name: Viewer window name
        """
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=800,
            height=600,
            point_show_normal=False
        )

    def merge_point_clouds(
        self,
        point_clouds: list
    ) -> o3d.geometry.PointCloud:
        """
        Merge multiple point clouds into one.

        Args:
            point_clouds: List of Open3D point clouds

        Returns:
            Merged point cloud
        """
        if not point_clouds:
            return o3d.geometry.PointCloud()

        merged = point_clouds[0]

        for pcd in point_clouds[1:]:
            merged += pcd

        return merged

    def transform_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        transformation_matrix: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Apply transformation to point cloud.

        Args:
            pcd: Input point cloud
            transformation_matrix: 4x4 transformation matrix

        Returns:
            Transformed point cloud
        """
        pcd_transformed = pcd.transform(transformation_matrix)
        return pcd_transformed

    def get_bounding_box(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of point cloud.

        Returns:
            (min_bound, max_bound) as (x,y,z) arrays
        """
        aabb = pcd.get_axis_aligned_bounding_box()
        return np.asarray(aabb.min_bound), np.asarray(aabb.max_bound)

    def crop_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        min_bound: np.ndarray,
        max_bound: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Crop point cloud to bounding box.

        Args:
            pcd: Input point cloud
            min_bound: Minimum (x, y, z)
            max_bound: Maximum (x, y, z)

        Returns:
            Cropped point cloud
        """
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )
        pcd_cropped = pcd.crop(bbox)
        return pcd_cropped
