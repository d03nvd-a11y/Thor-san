"""
Disparity Map Calculation

Computes disparity and depth from rectified stereo pairs.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class DisparityCalculator:
    """
    Disparity map calculator for stereo vision.

    Provides multiple algorithms for disparity computation.
    """

    def __init__(
        self,
        method: str = "sgbm",
        min_disparity: int = 0,
        max_disparity: int = 128,
        block_size: int = 5
    ):
        """
        Initialize disparity calculator.

        Args:
            method: "bm" (Block Matching) or "sgbm" (Semi-Global Block Matching)
            min_disparity: Minimum disparity
            max_disparity: Maximum disparity (must be divisible by 16)
            block_size: Block size for matching (must be odd)
        """
        self.method = method
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.block_size = block_size

        if method == "bm":
            self.stereo = self._create_bm_matcher()
        elif method == "sgbm":
            self.stereo = self._create_sgbm_matcher()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _create_bm_matcher(self):
        """Create Block Matching stereo matcher."""
        stereo = cv2.StereoBM_create(
            numDisparities=self.max_disparity - self.min_disparity,
            blockSize=self.block_size
        )

        # Configure parameters
        stereo.setPreFilterCap(31)
        stereo.setMinDisparity(self.min_disparity)
        stereo.setTextureThreshold(10)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleWindowSize(100)
        stereo.setSpeckleRange(32)
        stereo.setDisp12MaxDiff(1)

        return stereo

    def _create_sgbm_matcher(self):
        """Create Semi-Global Block Matching stereo matcher (better quality)."""
        num_disparities = self.max_disparity - self.min_disparity

        stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,  # Smoothness penalty
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        return stereo

    def compute_disparity(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair.

        Args:
            img_left: Left rectified image
            img_right: Right rectified image

        Returns:
            Disparity map (float32)
        """
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right

        # Compute disparity
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32)

        # Convert to actual disparity (divide by 16 - OpenCV fixed-point format)
        disparity = disparity / 16.0

        return disparity

    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        baseline_m: float,
        focal_length_px: float,
        min_depth: float = 0.2,
        max_depth: float = 10.0
    ) -> np.ndarray:
        """
        Convert disparity to depth.

        Uses: depth = (baseline * focal_length) / disparity

        Args:
            disparity: Disparity map
            baseline_m: Baseline between cameras (meters)
            focal_length_px: Focal length (pixels)
            min_depth: Minimum valid depth (meters)
            max_depth: Maximum valid depth (meters)

        Returns:
            Depth map (meters)
        """
        # Avoid division by zero
        valid_disparity = disparity > 0

        depth = np.zeros_like(disparity)
        depth[valid_disparity] = (baseline_m * focal_length_px) / disparity[valid_disparity]

        # Clip to valid range
        depth = np.clip(depth, min_depth, max_depth)

        # Mask invalid regions
        depth[~valid_disparity] = 0

        return depth

    def filter_disparity(
        self,
        disparity: np.ndarray,
        filter_size: int = 5
    ) -> np.ndarray:
        """
        Apply post-processing filtering to disparity map.

        Args:
            disparity: Raw disparity map
            filter_size: Median filter size

        Returns:
            Filtered disparity map
        """
        # Create validity mask
        valid_mask = disparity > 0

        # Median filter to remove noise
        filtered = cv2.medianBlur(
            disparity.astype(np.uint8),
            filter_size
        ).astype(np.float32)

        # Restore invalid regions
        filtered[~valid_mask] = 0

        return filtered

    def compute_depth_from_stereo(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        baseline_m: float,
        focal_length_px: float,
        filter_disparity: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: compute disparity and convert to depth.

        Args:
            img_left: Left rectified image
            img_right: Right rectified image
            baseline_m: Baseline (meters)
            focal_length_px: Focal length (pixels)
            filter_disparity: Apply post-processing filter

        Returns:
            (depth_map, disparity_map)
        """
        # Compute disparity
        disparity = self.compute_disparity(img_left, img_right)

        # Filter if requested
        if filter_disparity:
            disparity = self.filter_disparity(disparity)

        # Convert to depth
        depth = self.disparity_to_depth(disparity, baseline_m, focal_length_px)

        return depth, disparity

    def visualize_disparity(
        self,
        disparity: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Visualize disparity map with colormap.

        Args:
            disparity: Disparity map
            normalize: Normalize to full range

        Returns:
            Colored disparity visualization
        """
        disp_vis = disparity.copy()

        # Mask invalid disparities
        valid_mask = disp_vis > 0

        if normalize and np.any(valid_mask):
            # Normalize to [0, 255]
            disp_min = disp_vis[valid_mask].min()
            disp_max = disp_vis[valid_mask].max()

            if disp_max > disp_min:
                disp_vis[valid_mask] = (disp_vis[valid_mask] - disp_min) / (disp_max - disp_min) * 255

        disp_vis = disp_vis.astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Set invalid regions to black
        colored[~valid_mask] = 0

        return colored

    def get_confidence_map(
        self,
        disparity: np.ndarray,
        img_left: np.ndarray
    ) -> np.ndarray:
        """
        Estimate confidence for each disparity value.

        Based on texture and disparity consistency.

        Returns:
            Confidence map [0, 1]
        """
        h, w = disparity.shape

        # Texture confidence (high texture = more reliable matching)
        if len(img_left.shape) == 3:
            gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_left

        # Compute local variance as texture measure
        kernel_size = 7
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        variance = mean_sq - mean ** 2
        texture_confidence = np.clip(variance / 1000.0, 0, 1)

        # Disparity validity (non-zero)
        validity = (disparity > 0).astype(np.float32)

        # Combined confidence
        confidence = texture_confidence * validity

        return confidence
