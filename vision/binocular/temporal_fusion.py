"""
Temporal Fusion

Mimics visual memory in human perception - integrating depth information
over time to reduce noise and improve stability.

Biological Inspiration:
- Human vision integrates information over ~100ms (multiple frames)
- Visual memory maintains persistent 3D representation
- Temporal integration reduces sensor noise
- More recent observations weighted more heavily (temporal decay)
"""

import numpy as np
from typing import Optional, List
from collections import deque
from dataclasses import dataclass


@dataclass
class DepthFrame:
    """Depth measurement at a specific time."""
    depth_map: np.ndarray
    confidence_map: np.ndarray
    timestamp: float


class TemporalFusion:
    """
    Temporal integration of depth maps (visual memory).

    Maintains a sliding window of depth measurements and fuses them
    with confidence-weighted averaging and temporal decay.
    """

    def __init__(
        self,
        window_size: int = 10,
        temporal_decay: float = 0.95,
        min_confidence: float = 0.3
    ):
        """
        Initialize temporal fusion.

        Args:
            window_size: Number of frames to integrate
            temporal_decay: Decay factor for older frames (0-1)
            min_confidence: Minimum confidence to include measurement
        """
        self.window_size = window_size
        self.temporal_decay = temporal_decay
        self.min_confidence = min_confidence

        self.depth_history: deque = deque(maxlen=window_size)
        self.fused_depth: Optional[np.ndarray] = None
        self.fused_confidence: Optional[np.ndarray] = None

    def add_frame(
        self,
        depth_map: np.ndarray,
        confidence_map: np.ndarray,
        timestamp: float
    ):
        """
        Add new depth frame to temporal window.

        Args:
            depth_map: Depth measurements (H x W)
            confidence_map: Confidence for each depth (H x W), range [0, 1]
            timestamp: Frame timestamp
        """
        depth_frame = DepthFrame(
            depth_map=depth_map.copy(),
            confidence_map=confidence_map.copy(),
            timestamp=timestamp
        )
        self.depth_history.append(depth_frame)

    def get_fused_depth(self) -> tuple:
        """
        Get temporally fused depth map.

        Uses confidence-weighted averaging with temporal decay:
        - Recent frames have higher weight
        - High-confidence measurements weighted more
        - Reduces temporal noise and flicker

        Returns:
            (fused_depth_map, fused_confidence_map)
        """
        if not self.depth_history:
            return None, None

        # Get image shape from most recent frame
        shape = self.depth_history[-1].depth_map.shape

        # Initialize accumulators
        weighted_depth_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        max_confidence = np.zeros(shape, dtype=np.float32)

        # Compute temporal weights (exponential decay from oldest to newest)
        n_frames = len(self.depth_history)
        temporal_weights = [self.temporal_decay ** (n_frames - 1 - i) for i in range(n_frames)]

        # Fuse depth measurements
        for frame, temporal_weight in zip(self.depth_history, temporal_weights):
            # Only use measurements above confidence threshold
            valid_mask = frame.confidence_map >= self.min_confidence

            # Combined weight = temporal_weight * confidence
            weights = temporal_weight * frame.confidence_map * valid_mask

            # Accumulate weighted depth
            weighted_depth_sum += frame.depth_map * weights
            weight_sum += weights

            # Track maximum confidence seen at each pixel
            max_confidence = np.maximum(max_confidence, frame.confidence_map)

        # Compute fused depth (avoid division by zero)
        valid_pixels = weight_sum > 1e-6
        fused_depth = np.zeros(shape, dtype=np.float32)
        fused_depth[valid_pixels] = weighted_depth_sum[valid_pixels] / weight_sum[valid_pixels]

        # Fused confidence is combination of max confidence and temporal consistency
        fused_confidence = max_confidence * np.sqrt(weight_sum / (weight_sum.max() + 1e-6))

        self.fused_depth = fused_depth
        self.fused_confidence = fused_confidence

        return fused_depth, fused_confidence

    def get_variance(self) -> np.ndarray:
        """
        Compute temporal variance of depth measurements.

        High variance indicates unstable/uncertain regions.

        Returns:
            Variance map (H x W)
        """
        if len(self.depth_history) < 2:
            return np.zeros_like(self.depth_history[-1].depth_map)

        # Stack depth maps
        depths = np.stack([f.depth_map for f in self.depth_history], axis=0)

        # Compute variance across time
        variance = np.var(depths, axis=0)

        return variance

    def get_stability_mask(self, variance_threshold: float = 0.1) -> np.ndarray:
        """
        Get mask of temporally stable regions.

        Args:
            variance_threshold: Maximum allowed variance for stability

        Returns:
            Boolean mask (H x W) where True = stable
        """
        variance = self.get_variance()
        stable = variance < variance_threshold
        return stable

    def reset(self):
        """Clear temporal history."""
        self.depth_history.clear()
        self.fused_depth = None
        self.fused_confidence = None

    def get_statistics(self) -> dict:
        """Get temporal fusion statistics."""
        return {
            'window_size': self.window_size,
            'frames_in_buffer': len(self.depth_history),
            'temporal_decay': self.temporal_decay,
            'min_confidence': self.min_confidence,
            'has_fused_depth': self.fused_depth is not None
        }

    def apply_temporal_median_filter(self) -> np.ndarray:
        """
        Apply temporal median filter (alternative to weighted average).

        More robust to outliers but less smooth.

        Returns:
            Median-filtered depth map
        """
        if not self.depth_history:
            return None

        # Stack depth maps
        depths = np.stack([f.depth_map for f in self.depth_history], axis=0)

        # Only consider high-confidence measurements
        confidences = np.stack([f.confidence_map for f in self.depth_history], axis=0)
        valid_mask = confidences >= self.min_confidence

        # Set invalid measurements to NaN
        depths_masked = depths.copy()
        depths_masked[~valid_mask] = np.nan

        # Compute median (ignoring NaNs)
        median_depth = np.nanmedian(depths_masked, axis=0)

        # Fill remaining NaNs with 0
        median_depth = np.nan_to_num(median_depth, nan=0.0)

        return median_depth
