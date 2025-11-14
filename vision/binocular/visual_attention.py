"""
Visual Attention Mechanism

Mimics human visual attention system:
- Bottom-up saliency (automatic attention to salient features)
- Top-down task-driven attention (focus on relevant regions)
- Attention-weighted processing (allocate resources efficiently)

Biological Inspiration:
- Human vision processes high-attention regions with more detail
- Saliency driven by contrast, motion, color, orientation
- Task goals modulate attention (top-down control)
- Attention operates as a spotlight with Gaussian falloff
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class AttentionRegion:
    """Region of interest for attention."""
    center: Tuple[int, int]  # (x, y)
    radius: float  # Attention spread
    weight: float  # Importance (0-1)
    label: str = ""  # Optional label


class VisualAttention:
    """
    Visual attention mechanism for binocular vision.

    Combines bottom-up saliency and top-down task attention to
    weight depth processing spatially.
    """

    def __init__(
        self,
        attention_sigma: float = 50.0,
        saliency_enabled: bool = True,
        task_attention_enabled: bool = True
    ):
        """
        Initialize visual attention.

        Args:
            attention_sigma: Gaussian spread of attention (pixels)
            saliency_enabled: Enable bottom-up saliency
            task_attention_enabled: Enable top-down task attention
        """
        self.attention_sigma = attention_sigma
        self.saliency_enabled = saliency_enabled
        self.task_attention_enabled = task_attention_enabled

        self.task_attention_regions: List[AttentionRegion] = []

    def compute_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Compute bottom-up saliency map.

        Uses feature contrast detection:
        - Intensity contrast
        - Color contrast
        - Orientation contrast
        - Edge density

        Args:
            image: Input image (color or grayscale)

        Returns:
            Saliency map (H x W), normalized [0, 1]
        """
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # 1. Intensity contrast (center-surround)
        intensity_saliency = self._center_surround_contrast(gray)

        # 2. Edge density (high gradient regions are salient)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 5)

        # 3. Spectral residual (frequency-domain saliency)
        spectral_saliency = self._spectral_residual_saliency(gray)

        # Combine saliency cues
        saliency = (
            0.4 * intensity_saliency +
            0.3 * edge_density / (edge_density.max() + 1e-6) +
            0.3 * spectral_saliency
        )

        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)

        return saliency

    def _center_surround_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute center-surround contrast (mimics retinal processing).

        Uses difference of Gaussians (DoG).
        """
        # Center (fine scale)
        center = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)

        # Surround (coarse scale)
        surround = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 3.0)

        # Contrast
        contrast = np.abs(center - surround)

        return contrast

    def _spectral_residual_saliency(self, gray: np.ndarray) -> np.ndarray:
        """
        Spectral residual saliency detection.

        Based on frequency domain analysis - salient regions have
        unusual frequency content.
        """
        # FFT
        fft = np.fft.fft2(gray.astype(np.float32))
        amplitude = np.abs(fft)
        phase = np.angle(fft)

        # Log amplitude
        log_amplitude = np.log(amplitude + 1e-6)

        # Spectral residual (difference from average)
        mean_amplitude = cv2.GaussianBlur(log_amplitude, (3, 3), 1.0)
        residual = log_amplitude - mean_amplitude

        # Inverse FFT
        saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase)))

        # Smooth
        saliency = cv2.GaussianBlur(saliency, (7, 7), 2.0)

        return saliency

    def add_task_attention_region(
        self,
        center: Tuple[int, int],
        radius: float = None,
        weight: float = 1.0,
        label: str = ""
    ):
        """
        Add top-down attention region (task-driven).

        Args:
            center: (x, y) center of attention
            radius: Attention spread (defaults to attention_sigma)
            weight: Importance weight [0, 1]
            label: Optional label for region
        """
        if radius is None:
            radius = self.attention_sigma

        region = AttentionRegion(
            center=center,
            radius=radius,
            weight=weight,
            label=label
        )
        self.task_attention_regions.append(region)

    def clear_task_attention(self):
        """Clear all task attention regions."""
        self.task_attention_regions.clear()

    def compute_task_attention_map(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Compute top-down task attention map.

        Creates Gaussian attention spotlight around each task region.

        Args:
            image_shape: (height, width)

        Returns:
            Attention map (H x W), normalized [0, 1]
        """
        h, w = image_shape
        attention_map = np.zeros((h, w), dtype=np.float32)

        if not self.task_attention_regions:
            return np.ones((h, w), dtype=np.float32)  # Uniform attention

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Add Gaussian attention for each region
        for region in self.task_attention_regions:
            cx, cy = region.center
            sigma = region.radius

            # Gaussian attention spotlight
            gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            attention_map += region.weight * gaussian

        # Normalize
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()

        return attention_map

    def get_combined_attention_map(
        self,
        image: np.ndarray,
        saliency_weight: float = 0.5,
        task_weight: float = 0.5
    ) -> np.ndarray:
        """
        Compute combined attention map (bottom-up + top-down).

        Args:
            image: Input image
            saliency_weight: Weight for bottom-up saliency [0, 1]
            task_weight: Weight for top-down task attention [0, 1]

        Returns:
            Combined attention map (H x W), normalized [0, 1]
        """
        h, w = image.shape[:2]

        # Bottom-up saliency
        if self.saliency_enabled:
            saliency = self.compute_saliency(image)
        else:
            saliency = np.ones((h, w), dtype=np.float32)

        # Top-down task attention
        if self.task_attention_enabled:
            task_attention = self.compute_task_attention_map((h, w))
        else:
            task_attention = np.ones((h, w), dtype=np.float32)

        # Combine (multiplicative integration)
        combined = (
            saliency_weight * saliency +
            task_weight * task_attention
        )

        # Normalize to [0, 1]
        combined = combined / (combined.max() + 1e-6)

        return combined

    def apply_attention_weighting(
        self,
        data_map: np.ndarray,
        attention_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply attention weighting to data map.

        High-attention regions are preserved, low-attention regions suppressed.

        Args:
            data_map: Data to weight (e.g., depth map, confidence map)
            attention_map: Attention weights [0, 1]

        Returns:
            Attention-weighted data
        """
        return data_map * attention_map

    def visualize_attention(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize attention map overlaid on image.

        Args:
            image: Base image
            attention_map: Attention weights [0, 1]
            alpha: Overlay transparency

        Returns:
            Visualization image
        """
        # Convert image to color if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create heatmap from attention
        attention_colored = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Blend with original image
        vis = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)

        # Draw task attention regions
        for region in self.task_attention_regions:
            cv2.circle(
                vis,
                region.center,
                int(region.radius),
                (0, 255, 0),
                2
            )
            if region.label:
                cv2.putText(
                    vis,
                    region.label,
                    (region.center[0] + 10, region.center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        return vis

    def get_attention_statistics(self, attention_map: np.ndarray) -> dict:
        """Get statistics about attention distribution."""
        return {
            'mean_attention': float(np.mean(attention_map)),
            'max_attention': float(np.max(attention_map)),
            'min_attention': float(np.min(attention_map)),
            'attention_entropy': float(-np.sum(
                attention_map * np.log(attention_map + 1e-10)
            )),
            'num_task_regions': len(self.task_attention_regions)
        }
