"""
Depth Processing Utilities

Additional depth processing and filtering for RealSense.
"""

import numpy as np
import cv2


class DepthProcessor:
    """
    Additional depth processing utilities.

    RealSense hardware does most filtering, but this provides
    additional tools for specific use cases.
    """

    def __init__(self, depth_scale: float = 0.001):
        """
        Initialize depth processor.

        Args:
            depth_scale: Scale to convert depth units to meters
        """
        self.depth_scale = depth_scale

    def depth_to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert depth image from units to meters."""
        return depth_image * self.depth_scale

    def clip_depth_range(
        self,
        depth_image: np.ndarray,
        min_depth_m: float = 0.3,
        max_depth_m: float = 3.0
    ) -> np.ndarray:
        """
        Clip depth values to valid range.

        Args:
            depth_image: Depth in millimeters
            min_depth_m: Minimum depth in meters
            max_depth_m: Maximum depth in meters

        Returns:
            Clipped depth image
        """
        depth_m = self.depth_to_meters(depth_image)
        depth_m = np.clip(depth_m, min_depth_m, max_depth_m)

        # Mask out-of-range values
        mask = (depth_m < min_depth_m) | (depth_m > max_depth_m)
        depth_m[mask] = 0

        return (depth_m / self.depth_scale).astype(depth_image.dtype)

    def edge_preserving_filter(
        self,
        depth_image: np.ndarray,
        d: int = 5,
        sigma_color: float = 50,
        sigma_space: float = 50
    ) -> np.ndarray:
        """
        Apply edge-preserving bilateral filter.

        Smooths depth while preserving edges (object boundaries).
        """
        # Convert to float
        depth_float = depth_image.astype(np.float32)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            depth_float,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

        return filtered.astype(depth_image.dtype)

    def remove_small_objects(
        self,
        depth_image: np.ndarray,
        min_area: int = 100
    ) -> np.ndarray:
        """
        Remove small isolated depth regions (likely noise).

        Args:
            depth_image: Depth image
            min_area: Minimum area in pixels to keep

        Returns:
            Cleaned depth image
        """
        # Create binary mask
        mask = (depth_image > 0).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Filter by area
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 1

        # Apply mask
        cleaned = depth_image * filtered_mask

        return cleaned

    def get_depth_at_point(
        self,
        depth_image: np.ndarray,
        x: int,
        y: int,
        window_size: int = 5
    ) -> float:
        """
        Get robust depth estimate at pixel location.

        Uses median over window to reduce noise.

        Returns:
            Depth in meters
        """
        h, w = depth_image.shape

        # Ensure point is in bounds
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0

        # Extract window
        half = window_size // 2
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)

        window = depth_image[y_min:y_max, x_min:x_max]

        # Get valid depths
        valid_depths = window[window > 0]

        if len(valid_depths) == 0:
            return 0.0

        # Return median depth in meters
        median_depth = np.median(valid_depths)
        return median_depth * self.depth_scale

    def colorize_depth(
        self,
        depth_image: np.ndarray,
        min_depth: float = 0.3,
        max_depth: float = 3.0,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create colored depth visualization.

        Args:
            depth_image: Depth in millimeters
            min_depth: Minimum depth for color scale (meters)
            max_depth: Maximum depth for color scale (meters)
            colormap: OpenCV colormap

        Returns:
            Colored depth image (BGR)
        """
        # Convert to meters
        depth_m = self.depth_to_meters(depth_image)

        # Normalize to 0-255
        depth_normalized = np.zeros_like(depth_m)
        valid_mask = depth_m > 0

        if np.any(valid_mask):
            depth_normalized[valid_mask] = np.clip(
                (depth_m[valid_mask] - min_depth) / (max_depth - min_depth) * 255,
                0,
                255
            )

        depth_uint8 = depth_normalized.astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)

        # Set invalid regions to black
        colored[~valid_mask] = 0

        return colored
