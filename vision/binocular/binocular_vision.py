"""
Human-like Binocular Vision System

Main controller integrating all components of biomimetic binocular vision.

System Architecture (inspired by human vision):
1. Monocular preprocessing (retinal processing)
2. Feature extraction (V1 simple/complex cells)
3. Binocular correspondence matching (V1/V2 binocular neurons)
4. Temporal fusion (visual memory / persistence of vision)
5. Visual attention (attentional spotlight)
6. Dense depth reconstruction (3D perception)

This system mimics how human eyes and brain work together, NOT traditional
block-matching stereo vision used in most computer vision systems.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import time

from .correspondence_matcher import CorrespondenceMatcher, FeatureMatch
from .temporal_fusion import TemporalFusion
from .visual_attention import VisualAttention


@dataclass
class BinocularConfig:
    """Configuration for binocular vision system."""
    # Camera geometry
    baseline_m: float = 0.065  # Inter-pupillary distance (like human: 6.5cm)
    focal_length_px: float = 700.0  # Estimated focal length

    # Feature extraction
    feature_type: str = "orb"  # "orb", "sift", or "akaze"
    max_features: int = 500

    # Correspondence matching
    match_ratio_threshold: float = 0.75
    epipolar_threshold: float = 1.0
    min_match_confidence: float = 0.6

    # Temporal fusion (visual memory)
    temporal_window: int = 10
    temporal_decay: float = 0.95

    # Visual attention
    use_attention: bool = True
    attention_sigma: float = 50.0
    saliency_weight: float = 0.5
    task_weight: float = 0.5

    # Depth constraints
    min_depth_m: float = 0.2
    max_depth_m: float = 3.0


@dataclass
class BinocularOutput:
    """Output from binocular vision processing."""
    # Depth information
    depth_map: np.ndarray  # Dense depth map (H x W)
    confidence_map: np.ndarray  # Confidence for each pixel [0, 1]

    # Feature matches
    feature_matches: list  # List of FeatureMatch objects
    num_matches: int

    # Attention
    attention_map: Optional[np.ndarray] = None

    # Processing metadata
    processing_time_ms: float = 0.0
    timestamp: float = 0.0


class BinocularVision:
    """
    Human-like binocular vision system.

    Processes two monocular views to produce depth perception,
    mimicking biological vision processing.
    """

    def __init__(self, config: BinocularConfig = None):
        """
        Initialize binocular vision system.

        Args:
            config: BinocularConfig object
        """
        self.config = config or BinocularConfig()

        # Initialize subsystems
        self.correspondence_matcher = CorrespondenceMatcher(
            feature_type=self.config.feature_type,
            max_features=self.config.max_features,
            match_ratio_threshold=self.config.match_ratio_threshold,
            epipolar_threshold=self.config.epipolar_threshold
        )

        self.temporal_fusion = TemporalFusion(
            window_size=self.config.temporal_window,
            temporal_decay=self.config.temporal_decay
        )

        self.visual_attention = VisualAttention(
            attention_sigma=self.config.attention_sigma,
            saliency_enabled=self.config.use_attention,
            task_attention_enabled=self.config.use_attention
        )

        # Statistics
        self.frame_count = 0
        self.total_processing_time = 0.0

    def process_stereo_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        timestamp: float = None
    ) -> BinocularOutput:
        """
        Process stereo image pair to produce depth perception.

        This is the main entry point for binocular vision processing.

        Args:
            left_image: Left eye image
            right_image: Right eye image
            timestamp: Frame timestamp (defaults to current time)

        Returns:
            BinocularOutput with depth, confidence, and metadata
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = start_time

        # 1. MONOCULAR PREPROCESSING (retinal processing)
        left_gray = self._preprocess_monocular(left_image)
        right_gray = self._preprocess_monocular(right_image)

        # 2. FEATURE EXTRACTION (V1 simple/complex cells)
        kp_left, desc_left = self.correspondence_matcher.detect_features(left_gray)
        kp_right, desc_right = self.correspondence_matcher.detect_features(right_gray)

        # 3. CORRESPONDENCE MATCHING (binocular neurons)
        matches = self.correspondence_matcher.match_features(
            kp_left, desc_left, kp_right, desc_right
        )

        # Filter by confidence
        matches = self.correspondence_matcher.filter_by_confidence(
            matches, self.config.min_match_confidence
        )

        # Compute depths from disparities
        matches = self.correspondence_matcher.compute_depths(
            matches,
            self.config.baseline_m,
            self.config.focal_length_px
        )

        # 4. SPARSE DEPTH MAP from feature matches
        sparse_depth = self._create_sparse_depth_map(matches, left_gray.shape)

        # 5. VISUAL ATTENTION (attentional modulation)
        if self.config.use_attention:
            attention_map = self.visual_attention.get_combined_attention_map(
                left_image,
                self.config.saliency_weight,
                self.config.task_weight
            )
        else:
            attention_map = np.ones(left_gray.shape, dtype=np.float32)

        # 6. DENSE DEPTH ESTIMATION (interpolation + attention weighting)
        dense_depth, confidence = self._create_dense_depth(
            matches, left_gray.shape, attention_map
        )

        # Apply attention weighting to confidence
        if self.config.use_attention:
            confidence = self.visual_attention.apply_attention_weighting(
                confidence, attention_map
            )

        # 7. TEMPORAL FUSION (visual memory integration)
        self.temporal_fusion.add_frame(dense_depth, confidence, timestamp)
        fused_depth, fused_confidence = self.temporal_fusion.get_fused_depth()

        # Use fused depth if available
        if fused_depth is not None:
            depth_map = fused_depth
            confidence_map = fused_confidence
        else:
            depth_map = dense_depth
            confidence_map = confidence

        # Apply depth constraints
        depth_map = np.clip(depth_map, self.config.min_depth_m, self.config.max_depth_m)

        # Create output
        processing_time = (time.time() - start_time) * 1000  # ms

        output = BinocularOutput(
            depth_map=depth_map,
            confidence_map=confidence_map,
            feature_matches=matches,
            num_matches=len(matches),
            attention_map=attention_map if self.config.use_attention else None,
            processing_time_ms=processing_time,
            timestamp=timestamp
        )

        # Update statistics
        self.frame_count += 1
        self.total_processing_time += processing_time

        return output

    def _preprocess_monocular(self, image: np.ndarray) -> np.ndarray:
        """
        Monocular preprocessing (mimics retinal processing).

        Converts to grayscale and applies slight smoothing.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Slight Gaussian smoothing (mimics retinal pooling)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

        return gray

    def _create_sparse_depth_map(
        self,
        matches: list,
        shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sparse depth map from feature matches.

        Returns:
            (depth_map, confidence_map)
        """
        h, w = shape
        depth_map = np.zeros((h, w), dtype=np.float32)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        for match in matches:
            x, y = int(match.left_point[0]), int(match.left_point[1])

            if 0 <= x < w and 0 <= y < h:
                if match.depth_m is not None:
                    depth_map[y, x] = match.depth_m
                    confidence_map[y, x] = match.confidence

        return depth_map, confidence_map

    def _create_dense_depth(
        self,
        matches: list,
        shape: Tuple[int, int],
        attention_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dense depth map from sparse matches.

        Uses interpolation with attention-based weighting.
        """
        h, w = shape

        if not matches:
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        # Create sparse maps
        depth_sparse, confidence_sparse = self._create_sparse_depth_map(matches, shape)

        # Interpolate sparse depth to dense
        # Use attention-weighted interpolation

        # Prepare points for interpolation
        points = []
        values = []
        weights = []

        for match in matches:
            x, y = int(match.left_point[0]), int(match.left_point[1])

            if 0 <= x < w and 0 <= y < h:
                if match.depth_m is not None:
                    points.append([x, y])
                    values.append(match.depth_m)
                    # Weight by confidence and attention
                    weight = match.confidence * attention_map[y, x]
                    weights.append(weight)

        if not points:
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        points = np.array(points)
        values = np.array(values)
        weights = np.array(weights)

        # Create coordinate grids
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Interpolate using weighted nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(points)

        # Find k nearest neighbors
        k = min(5, len(points))
        distances, indices = tree.query(grid_points, k=k)

        # Compute interpolated depth (inverse distance weighting)
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)

        # Inverse distance weights
        idw_weights = 1.0 / distances

        # Include match confidence and attention
        match_weights = weights[indices]
        combined_weights = idw_weights * match_weights

        # Normalize weights
        weight_sum = np.sum(combined_weights, axis=1, keepdims=True)
        normalized_weights = combined_weights / (weight_sum + 1e-6)

        # Interpolate
        neighbor_values = values[indices]
        interpolated = np.sum(neighbor_values * normalized_weights, axis=1)

        # Compute confidence based on distance and match quality
        max_distance = np.max(distances, axis=1)
        confidence = np.exp(-max_distance / 50.0)  # Exponential decay

        # Reshape to image
        dense_depth = interpolated.reshape((h, w))
        dense_confidence = confidence.reshape((h, w))

        return dense_depth, dense_confidence

    def add_task_attention(self, x: int, y: int, radius: float = None, weight: float = 1.0, label: str = ""):
        """Add top-down task attention region (for focusing on objects of interest)."""
        self.visual_attention.add_task_attention_region((x, y), radius, weight, label)

    def clear_task_attention(self):
        """Clear all task attention regions."""
        self.visual_attention.clear_task_attention()

    def reset_temporal_memory(self):
        """Reset visual memory (temporal fusion buffer)."""
        self.temporal_fusion.reset()

    def get_statistics(self) -> dict:
        """Get system statistics."""
        avg_processing_time = (
            self.total_processing_time / self.frame_count
            if self.frame_count > 0 else 0.0
        )

        return {
            'frame_count': self.frame_count,
            'avg_processing_time_ms': avg_processing_time,
            'total_processing_time_ms': self.total_processing_time,
            'correspondence_matcher': {
                'feature_type': self.config.feature_type,
                'max_features': self.config.max_features
            },
            'temporal_fusion': self.temporal_fusion.get_statistics(),
            'config': {
                'baseline_m': self.config.baseline_m,
                'focal_length_px': self.config.focal_length_px,
                'temporal_window': self.config.temporal_window,
                'use_attention': self.config.use_attention
            }
        }

    def visualize_output(
        self,
        left_image: np.ndarray,
        output: BinocularOutput,
        show_features: bool = True,
        show_depth: bool = True,
        show_attention: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Create visualization of binocular vision output.

        Returns:
            Dictionary of visualization images
        """
        visualizations = {}

        # Depth map visualization
        if show_depth:
            depth_colored = self._colorize_depth(output.depth_map)
            visualizations['depth'] = depth_colored

            # Confidence map
            confidence_colored = cv2.applyColorMap(
                (output.confidence_map * 255).astype(np.uint8),
                cv2.COLORMAP_HOT
            )
            visualizations['confidence'] = confidence_colored

        # Feature matches visualization
        if show_features and len(output.feature_matches) > 0:
            # Draw features on left image
            left_with_features = left_image.copy()
            if len(left_with_features.shape) == 2:
                left_with_features = cv2.cvtColor(left_with_features, cv2.COLOR_GRAY2BGR)

            for match in output.feature_matches[:100]:  # Limit to 100 for clarity
                x, y = int(match.left_point[0]), int(match.left_point[1])
                color_intensity = int(255 * match.confidence)
                color = (0, color_intensity, 255 - color_intensity)
                cv2.circle(left_with_features, (x, y), 3, color, -1)

            visualizations['features'] = left_with_features

        # Attention map visualization
        if show_attention and output.attention_map is not None:
            attention_vis = self.visual_attention.visualize_attention(
                left_image, output.attention_map
            )
            visualizations['attention'] = attention_vis

        return visualizations

    def _colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Colorize depth map for visualization."""
        # Normalize to [0, 255]
        depth_normalized = depth_map.copy()
        valid_mask = depth_normalized > 0

        if np.any(valid_mask):
            depth_min = depth_normalized[valid_mask].min()
            depth_max = depth_normalized[valid_mask].max()

            if depth_max > depth_min:
                depth_normalized[valid_mask] = (
                    (depth_normalized[valid_mask] - depth_min) / (depth_max - depth_min) * 255
                )

        depth_uint8 = depth_normalized.astype(np.uint8)

        # Apply colormap (jet: blue=close, red=far)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        # Set invalid regions to black
        depth_colored[~valid_mask] = 0

        return depth_colored
