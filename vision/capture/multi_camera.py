"""
Multi-Camera Capture System

Manages multiple USB cameras simultaneously with independent threads.
Each camera runs in its own thread for optimal performance.
"""

import cv2
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np


@dataclass
class CameraFrame:
    """Container for a captured frame with metadata."""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    frame_number: int
    resolution: tuple


class CameraThread(threading.Thread):
    """Independent thread for capturing from a single camera."""

    def __init__(self, camera_id: str, device_index: int, resolution: tuple, fps: int):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[CameraFrame] = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_count = 0

    def run(self):
        """Main capture loop running in separate thread."""
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera {self.camera_id} at index {self.device_index}")
            return

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        print(f"[INFO] Camera {self.camera_id} started (index={self.device_index})")

        self.running = True
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                timestamp = time.time()
                camera_frame = CameraFrame(
                    camera_id=self.camera_id,
                    frame=frame.copy(),
                    timestamp=timestamp,
                    frame_number=self.frame_count,
                    resolution=(frame.shape[1], frame.shape[0])
                )

                with self.frame_lock:
                    self.latest_frame = camera_frame
                    self.frame_count += 1
            else:
                print(f"[WARN] Failed to read from camera {self.camera_id}")
                time.sleep(0.01)

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the most recent frame (thread-safe)."""
        with self.frame_lock:
            return self.latest_frame

    def stop(self):
        """Stop the capture thread."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        print(f"[INFO] Camera {self.camera_id} stopped")


class MultiCameraCapture:
    """
    Manages multiple cameras simultaneously.

    Each camera runs in its own thread for optimal frame rate.
    Provides synchronized access to latest frames from all cameras.
    """

    def __init__(self, config: dict):
        """
        Initialize multi-camera system.

        Args:
            config: Dictionary with camera configurations
        """
        self.config = config
        self.cameras: Dict[str, CameraThread] = {}
        self.running = False

    def start(self):
        """Start all cameras."""
        camera_configs = self.config.get('cameras', {})

        for cam_id, cam_config in camera_configs.items():
            # Skip disabled cameras
            if not cam_config.get('enabled', True):
                continue

            device_index = cam_config['index']
            resolution = tuple(cam_config['resolution'])
            fps = cam_config['fps']

            camera_thread = CameraThread(cam_id, device_index, resolution, fps)
            camera_thread.start()
            self.cameras[cam_id] = camera_thread

        self.running = True
        print(f"[INFO] Started {len(self.cameras)} camera(s)")

        # Give cameras time to initialize
        time.sleep(1.0)

    def get_frames(self) -> Dict[str, CameraFrame]:
        """
        Get latest frames from all cameras.

        Returns:
            Dictionary mapping camera_id to CameraFrame
        """
        frames = {}
        for cam_id, camera in self.cameras.items():
            frame = camera.get_latest_frame()
            if frame is not None:
                frames[cam_id] = frame
        return frames

    def get_frame(self, camera_id: str) -> Optional[CameraFrame]:
        """
        Get latest frame from specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            CameraFrame or None if not available
        """
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_latest_frame()
        return None

    def stop(self):
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()

        # Wait for threads to finish
        for camera in self.cameras.values():
            camera.join(timeout=2.0)

        self.cameras.clear()
        self.running = False
        print("[INFO] All cameras stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def get_camera_info(self) -> Dict[str, dict]:
        """Get information about all active cameras."""
        info = {}
        for cam_id, camera in self.cameras.items():
            frame = camera.get_latest_frame()
            info[cam_id] = {
                'device_index': camera.device_index,
                'resolution': camera.resolution,
                'fps': camera.fps,
                'frame_count': camera.frame_count,
                'latest_timestamp': frame.timestamp if frame else None,
                'running': camera.running
            }
        return info
