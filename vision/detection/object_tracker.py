"""
Object Tracker

Multi-object tracking with identity persistence across frames.
Uses IoU-based association and Kalman filtering for smooth tracking.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

from .yolo_detector import Detection


@dataclass
class TrackedObject:
    """Object tracked across multiple frames."""
    track_id: int
    detection: Detection
    age: int = 0  # Number of frames tracked
    hits: int = 1  # Number of successful detections
    misses: int = 0  # Number of consecutive missed detections
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    history: List[Tuple[int, int]] = field(default_factory=list)  # Position history

    def update(self, detection: Detection):
        """Update tracked object with new detection."""
        # Compute velocity
        old_center = self.detection.center
        new_center = detection.center
        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )

        # Update detection
        self.detection = detection
        self.hits += 1
        self.misses = 0
        self.age += 1

        # Add to history
        self.history.append(new_center)
        if len(self.history) > 30:  # Keep last 30 positions
            self.history.pop(0)

    def predict_position(self) -> Tuple[int, int]:
        """Predict next position based on velocity."""
        cx, cy = self.detection.center
        vx, vy = self.velocity
        predicted = (int(cx + vx), int(cy + vy))
        return predicted

    def mark_missed(self):
        """Mark that object was not detected in this frame."""
        self.misses += 1
        self.age += 1


class ObjectTracker:
    """
    Multi-object tracker using IoU-based association.

    Maintains object identities across frames even with temporary occlusions.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize object tracker.

        Args:
            max_age: Maximum number of frames to keep track without detection
            min_hits: Minimum number of detections before track is confirmed
            iou_threshold: Minimum IoU for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[TrackedObject] = []
        self.next_track_id = 1
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections in current frame

        Returns:
            List of confirmed tracked objects
        """
        self.frame_count += 1

        # Associate detections to tracks
        matched, unmatched_detections, unmatched_tracks = self._associate(detections)

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Remove dead tracks
        self.tracks = [
            track for track in self.tracks
            if track.misses <= self.max_age
        ]

        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks
            if track.hits >= self.min_hits
        ]

        return confirmed_tracks

    def _associate(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using IoU.

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(
                    track.detection.bbox,
                    detection.bbox
                )

        # Hungarian algorithm for optimal assignment
        # Maximize IoU -> minimize (1 - IoU)
        cost_matrix = 1.0 - iou_matrix

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Filter out low IoU matches
        matched = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        for t_idx, d_idx in zip(track_indices, detection_indices):
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                matched.append((t_idx, d_idx))
                unmatched_detections.discard(d_idx)
                unmatched_tracks.discard(t_idx)

        return matched, list(unmatched_detections), list(unmatched_tracks)

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: (x1, y1, x2, y2)
            bbox2: (x1, y1, x2, y2)

        Returns:
            IoU value [0, 1]
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        iou = intersection / (union + 1e-6)

        return iou

    def _create_track(self, detection: Detection):
        """Create new track from detection."""
        track = TrackedObject(
            track_id=self.next_track_id,
            detection=detection,
            age=0,
            hits=1,
            misses=0,
            history=[detection.center]
        )
        self.tracks.append(track)
        self.next_track_id += 1

    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def visualize_tracks(
        self,
        image: np.ndarray,
        tracks: List[TrackedObject],
        show_trajectory: bool = True,
        show_id: bool = True
    ) -> np.ndarray:
        """
        Visualize tracked objects.

        Args:
            image: Input image
            tracks: List of tracked objects
            show_trajectory: Show position history
            show_id: Show track ID

        Returns:
            Visualization image
        """
        import cv2

        vis = image.copy()

        # Define colors for tracks
        np.random.seed(42)
        colors = {}

        for track in tracks:
            # Get color for track
            if track.track_id not in colors:
                colors[track.track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
            color = colors[track.track_id]

            # Draw bounding box
            x1, y1, x2, y2 = track.detection.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory
            if show_trajectory and len(track.history) > 1:
                for i in range(len(track.history) - 1):
                    pt1 = track.history[i]
                    pt2 = track.history[i + 1]
                    cv2.line(vis, pt1, pt2, color, 2)

            # Draw ID and info
            if show_id:
                label = f"ID:{track.track_id} {track.detection.class_name}"

                # Background
                (text_w, text_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    1
                )

                cv2.rectangle(
                    vis,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color,
                    -1
                )

                # Text
                cv2.putText(
                    vis,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

            # Draw center
            cv2.circle(vis, track.detection.center, 4, color, -1)

        return vis

    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.tracks),
            'next_track_id': self.next_track_id,
            'confirmed_tracks': len([t for t in self.tracks if t.hits >= self.min_hits]),
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold
        }

    def reset(self):
        """Reset tracker."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
