"""
RealSense D435i Camera Wrapper

Simple interface to Intel RealSense D435i.
Get RGB, depth, and IMU data with minimal code.
"""

import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class RealSenseConfig:
    """Configuration for RealSense camera."""
    # RGB stream
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: int = 30

    # Depth stream
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30

    # Enable streams
    enable_rgb: bool = True
    enable_depth: bool = True
    enable_imu: bool = False
    enable_infrared: bool = False

    # Post-processing
    enable_decimation: bool = True      # Reduce depth resolution for performance
    enable_spatial_filter: bool = True  # Smooth depth
    enable_temporal_filter: bool = True # Reduce noise over time
    enable_hole_filling: bool = True    # Fill depth holes

    # Alignment
    align_depth_to_color: bool = True   # Align depth to RGB frame


@dataclass
class RealSenseFrame:
    """Single frame from RealSense camera."""
    rgb: Optional[np.ndarray] = None           # RGB image (H x W x 3)
    depth: Optional[np.ndarray] = None         # Depth map in mm (H x W)
    depth_colormap: Optional[np.ndarray] = None  # Depth visualization
    infrared: Optional[np.ndarray] = None      # IR image (H x W)
    timestamp: float = 0.0
    frame_number: int = 0

    # Camera intrinsics
    depth_intrinsics: Optional[rs.intrinsics] = None
    color_intrinsics: Optional[rs.intrinsics] = None
    depth_scale: float = 0.001  # Conversion from depth units to meters


class RealSenseCamera:
    """
    Intel RealSense D435i camera wrapper.

    Super simple to use:
        camera = RealSenseCamera()
        camera.start()

        while True:
            frame = camera.get_frame()
            cv2.imshow("RGB", frame.rgb)
            cv2.imshow("Depth", frame.depth_colormap)
    """

    def __init__(self, config: RealSenseConfig = None):
        """
        Initialize RealSense camera.

        Args:
            config: RealSenseConfig or None for defaults
        """
        self.config = config or RealSenseConfig()

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()

        # Post-processing filters
        self.decimation = None
        self.spatial = None
        self.temporal = None
        self.hole_filling = None
        self.colorizer = rs.colorizer()  # For depth visualization

        # Alignment
        self.align = None

        # State
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # Camera info
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None

    def start(self):
        """Start the camera and begin streaming."""
        # Configure streams
        if self.config.enable_rgb:
            self.rs_config.enable_stream(
                rs.stream.color,
                self.config.rgb_width,
                self.config.rgb_height,
                rs.format.bgr8,
                self.config.rgb_fps
            )

        if self.config.enable_depth:
            self.rs_config.enable_stream(
                rs.stream.depth,
                self.config.depth_width,
                self.config.depth_height,
                rs.format.z16,
                self.config.depth_fps
            )

        if self.config.enable_infrared:
            self.rs_config.enable_stream(rs.stream.infrared, 1)

        if self.config.enable_imu:
            self.rs_config.enable_stream(rs.stream.accel)
            self.rs_config.enable_stream(rs.stream.gyro)

        # Start pipeline
        profile = self.pipeline.start(self.rs_config)

        # Get depth scale (convert depth units to meters)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Get intrinsics
        if self.config.enable_depth:
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        if self.config.enable_rgb:
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Setup post-processing filters
        if self.config.enable_decimation:
            self.decimation = rs.decimation_filter()

        if self.config.enable_spatial_filter:
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, 2)
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial.set_option(rs.option.filter_smooth_delta, 20)

        if self.config.enable_temporal_filter:
            self.temporal = rs.temporal_filter()
            self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        if self.config.enable_hole_filling:
            self.hole_filling = rs.hole_filling_filter()

        # Setup alignment
        if self.config.align_depth_to_color and self.config.enable_rgb:
            self.align = rs.align(rs.stream.color)

        self.running = True
        self.start_time = time.time()

        print(f"[INFO] RealSense D435i started!")
        print(f"[INFO] Depth scale: {self.depth_scale}")
        print(f"[INFO] Depth range: {0.3}m - {10}m")
        print(f"[INFO] RGB resolution: {self.config.rgb_width}x{self.config.rgb_height}")
        print(f"[INFO] Depth resolution: {self.config.depth_width}x{self.config.depth_height}")

    def get_frame(self) -> Optional[RealSenseFrame]:
        """
        Get latest frame from camera.

        Returns:
            RealSenseFrame with RGB, depth, etc.
        """
        if not self.running:
            return None

        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Align depth to color if enabled
        if self.align:
            frames = self.align.process(frames)

        # Extract frames
        depth_frame = frames.get_depth_frame() if self.config.enable_depth else None
        color_frame = frames.get_color_frame() if self.config.enable_rgb else None
        infrared_frame = frames.get_infrared_frame(1) if self.config.enable_infrared else None

        # Apply depth post-processing
        if depth_frame:
            if self.decimation:
                depth_frame = self.decimation.process(depth_frame)
            if self.spatial:
                depth_frame = self.spatial.process(depth_frame)
            if self.temporal:
                depth_frame = self.temporal.process(depth_frame)
            if self.hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)

        # Convert to numpy arrays
        rgb_image = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        infrared_image = np.asanyarray(infrared_frame.get_data()) if infrared_frame else None

        # Create depth colormap for visualization
        depth_colormap = None
        if depth_frame:
            depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        # Create frame object
        frame = RealSenseFrame(
            rgb=rgb_image,
            depth=depth_image,
            depth_colormap=depth_colormap,
            infrared=infrared_image,
            timestamp=time.time(),
            frame_number=self.frame_count,
            depth_intrinsics=self.depth_intrinsics,
            color_intrinsics=self.color_intrinsics,
            depth_scale=self.depth_scale
        )

        self.frame_count += 1

        return frame

    def get_point_cloud(self, depth_frame_data: np.ndarray = None) -> np.ndarray:
        """
        Generate 3D point cloud from depth.

        Args:
            depth_frame_data: Depth image (optional, uses latest if None)

        Returns:
            Point cloud as Nx3 array (x, y, z in meters)
        """
        if depth_frame_data is None:
            frame = self.get_frame()
            depth_frame_data = frame.depth

        if depth_frame_data is None:
            return np.array([])

        # Get intrinsics
        intrinsics = self.depth_intrinsics

        h, w = depth_frame_data.shape

        # Create meshgrid of pixel coordinates
        i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Get depth values in meters
        z = depth_frame_data * self.depth_scale

        # Deproject to 3D
        x = (j - intrinsics.ppx) * z / intrinsics.fx
        y = (i - intrinsics.ppy) * z / intrinsics.fy

        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)

        # Filter out zero depth
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]

        return points

    def stop(self):
        """Stop the camera."""
        if self.running:
            self.pipeline.stop()
            self.running = False

            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

            print(f"\n[INFO] RealSense stopped")
            print(f"[INFO] Captured {self.frame_count} frames in {elapsed:.1f}s")
            print(f"[INFO] Average FPS: {avg_fps:.1f}")

    def get_camera_info(self) -> dict:
        """Get camera information."""
        return {
            'running': self.running,
            'frame_count': self.frame_count,
            'depth_scale': self.depth_scale,
            'depth_intrinsics': {
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy
            } if self.depth_intrinsics else None,
            'color_intrinsics': {
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height,
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy
            } if self.color_intrinsics else None
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
