"""
RealSense Bag File Player

Plays back recorded .bag files as if they were a live camera.
Perfect for testing without physical hardware!
"""

import numpy as np
import pyrealsense2 as rs
from typing import Optional
import time

from .camera import RealSenseFrame


class BagFilePlayer:
    """
    RealSense bag file playback.

    Provides the exact same interface as RealSenseCamera,
    so you can swap between live camera and recorded data
    without changing any code!

    Usage:
        # Instead of: camera = RealSenseCamera()
        camera = BagFilePlayer("data/bags/scene.bag")
        camera.start()

        frame = camera.get_frame()  # Works exactly like live camera!
    """

    def __init__(
        self,
        bag_file_path: str,
        loop: bool = True,
        real_time: bool = False
    ):
        """
        Initialize bag file player.

        Args:
            bag_file_path: Path to .bag file
            loop: Loop playback when reaching end
            real_time: Play at recorded speed (True) or as fast as possible (False)
        """
        self.bag_file_path = bag_file_path
        self.loop = loop
        self.real_time = real_time

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Camera info (filled on start)
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None

        # State
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # Visualization
        self.colorizer = rs.colorizer()

        print(f"[INFO] BagFilePlayer initialized")
        print(f"[INFO] Bag file: {bag_file_path}")
        print(f"[INFO] Loop: {loop}, Real-time: {real_time}")

    def start(self):
        """Start playback from bag file."""
        # Enable playback from file
        self.config.enable_device_from_file(
            self.bag_file_path,
            repeat_playback=self.loop
        )

        # Start pipeline
        profile = self.pipeline.start(self.config)

        # Get playback device
        device = profile.get_device()
        playback = device.as_playback()

        # Set playback mode
        playback.set_real_time(self.real_time)

        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Get intrinsics
        try:
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        except:
            print("[WARN] No depth stream in bag file")

        try:
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        except:
            print("[WARN] No color stream in bag file")

        self.running = True
        self.start_time = time.time()

        print(f"[INFO] Bag file playback started!")
        print(f"[INFO] Depth scale: {self.depth_scale}")
        if self.depth_intrinsics:
            print(f"[INFO] Depth resolution: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
        if self.color_intrinsics:
            print(f"[INFO] RGB resolution: {self.color_intrinsics.width}x{self.color_intrinsics.height}")

    def get_frame(self) -> Optional[RealSenseFrame]:
        """
        Get next frame from bag file.

        Returns:
            RealSenseFrame (same as RealSenseCamera!)
        """
        if not self.running:
            return None

        try:
            # Wait for frames (5 second timeout)
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            # Extract frames
            depth_frame = frames.get_depth_frame() if frames.get_depth_frame() else None
            color_frame = frames.get_color_frame() if frames.get_color_frame() else None

            if not depth_frame and not color_frame:
                return None

            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            # Create depth colormap for visualization
            depth_colormap = None
            if depth_frame:
                depth_colormap = np.asanyarray(
                    self.colorizer.colorize(depth_frame).get_data()
                )

            # Create frame object (SAME FORMAT as RealSenseCamera!)
            frame = RealSenseFrame(
                rgb=rgb_image,
                depth=depth_image,
                depth_colormap=depth_colormap,
                timestamp=time.time(),
                frame_number=self.frame_count,
                depth_intrinsics=self.depth_intrinsics,
                color_intrinsics=self.color_intrinsics,
                depth_scale=self.depth_scale
            )

            self.frame_count += 1

            return frame

        except RuntimeError as e:
            # End of file reached
            if not self.loop:
                print(f"\n[INFO] Playback finished")
                self.running = False
            return None

    def stop(self):
        """Stop playback."""
        if self.running:
            self.pipeline.stop()
            self.running = False

            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

            print(f"\n[INFO] Bag playback stopped")
            print(f"[INFO] Processed {self.frame_count} frames in {elapsed:.1f}s")
            print(f"[INFO] Average FPS: {avg_fps:.1f}")

    def get_camera_info(self) -> dict:
        """Get camera/playback information."""
        return {
            'source': 'bag_file',
            'bag_file_path': self.bag_file_path,
            'running': self.running,
            'frame_count': self.frame_count,
            'loop': self.loop,
            'real_time': self.real_time,
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
