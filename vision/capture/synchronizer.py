"""
Frame Synchronizer

Ensures temporal alignment of frames from multiple cameras.
Critical for binocular vision - both eyes must see at the same moment.
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from collections import deque
import numpy as np

from .multi_camera import CameraFrame


@dataclass
class SynchronizedFrameSet:
    """Set of temporally aligned frames from multiple cameras."""
    frames: Dict[str, CameraFrame]
    timestamp: float  # Reference timestamp
    max_time_diff: float  # Maximum time difference between frames


class FrameSynchronizer:
    """
    Synchronizes frames from multiple cameras based on timestamps.

    Uses a buffer-based approach to find temporally aligned frames.
    Essential for binocular vision where left/right eyes must be synchronized.
    """

    def __init__(self, max_time_diff_ms: float = 50.0, buffer_size: int = 5):
        """
        Initialize frame synchronizer.

        Args:
            max_time_diff_ms: Maximum allowed time difference between frames (milliseconds)
            buffer_size: Number of frames to buffer per camera
        """
        self.max_time_diff = max_time_diff_ms / 1000.0  # Convert to seconds
        self.buffer_size = buffer_size
        self.frame_buffers: Dict[str, deque] = {}

        # Statistics
        self.sync_success_count = 0
        self.sync_fail_count = 0
        self.total_time_diffs: List[float] = []

    def add_frames(self, frames: Dict[str, CameraFrame]):
        """
        Add new frames to buffers.

        Args:
            frames: Dictionary of camera_id -> CameraFrame
        """
        for cam_id, frame in frames.items():
            if cam_id not in self.frame_buffers:
                self.frame_buffers[cam_id] = deque(maxlen=self.buffer_size)
            self.frame_buffers[cam_id].append(frame)

    def get_synchronized_frames(self, required_cameras: List[str]) -> Optional[SynchronizedFrameSet]:
        """
        Get temporally synchronized frames from specified cameras.

        Args:
            required_cameras: List of camera IDs that must be present

        Returns:
            SynchronizedFrameSet if synchronization successful, None otherwise
        """
        # Check if all required cameras have frames
        for cam_id in required_cameras:
            if cam_id not in self.frame_buffers or len(self.frame_buffers[cam_id]) == 0:
                return None

        # Try to find best temporal match
        best_match = self._find_best_match(required_cameras)

        if best_match is not None:
            self.sync_success_count += 1
            return best_match
        else:
            self.sync_fail_count += 1
            return None

    def _find_best_match(self, camera_ids: List[str]) -> Optional[SynchronizedFrameSet]:
        """
        Find best temporal match among buffered frames.

        Uses the first camera as reference and finds closest matches in others.
        """
        if not camera_ids:
            return None

        # Use first camera as reference
        ref_cam_id = camera_ids[0]
        ref_buffer = self.frame_buffers[ref_cam_id]

        # Try each frame in reference buffer (most recent first)
        for ref_frame in reversed(ref_buffer):
            ref_time = ref_frame.timestamp

            # Try to find matching frames in other cameras
            matched_frames = {ref_cam_id: ref_frame}
            max_diff = 0.0

            match_found = True
            for cam_id in camera_ids[1:]:
                closest_frame = self._find_closest_frame(
                    self.frame_buffers[cam_id],
                    ref_time
                )

                if closest_frame is None:
                    match_found = False
                    break

                time_diff = abs(closest_frame.timestamp - ref_time)
                if time_diff > self.max_time_diff:
                    match_found = False
                    break

                matched_frames[cam_id] = closest_frame
                max_diff = max(max_diff, time_diff)

            if match_found:
                self.total_time_diffs.append(max_diff)
                return SynchronizedFrameSet(
                    frames=matched_frames,
                    timestamp=ref_time,
                    max_time_diff=max_diff
                )

        return None

    def _find_closest_frame(self, buffer: deque, target_time: float) -> Optional[CameraFrame]:
        """Find frame with timestamp closest to target time."""
        if not buffer:
            return None

        closest = None
        min_diff = float('inf')

        for frame in buffer:
            diff = abs(frame.timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest = frame

        return closest

    def clear_old_frames(self, current_time: float, max_age: float = 1.0):
        """
        Remove frames older than max_age seconds.

        Args:
            current_time: Current timestamp
            max_age: Maximum age of frames to keep (seconds)
        """
        for cam_id, buffer in self.frame_buffers.items():
            while buffer and (current_time - buffer[0].timestamp) > max_age:
                buffer.popleft()

    def get_statistics(self) -> dict:
        """Get synchronization statistics."""
        total_attempts = self.sync_success_count + self.sync_fail_count
        success_rate = (self.sync_success_count / total_attempts * 100) if total_attempts > 0 else 0.0

        avg_time_diff = np.mean(self.total_time_diffs) if self.total_time_diffs else 0.0
        max_time_diff = np.max(self.total_time_diffs) if self.total_time_diffs else 0.0

        return {
            'success_count': self.sync_success_count,
            'fail_count': self.sync_fail_count,
            'success_rate': success_rate,
            'avg_time_diff_ms': avg_time_diff * 1000,
            'max_time_diff_ms': max_time_diff * 1000,
            'max_allowed_diff_ms': self.max_time_diff * 1000
        }
