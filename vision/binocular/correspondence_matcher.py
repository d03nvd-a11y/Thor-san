"""
Correspondence Matcher

Mimics binocular neurons in visual cortex (V1/V2) that respond to
corresponding features in left and right eyes.

Biological Inspiration:
- V1 cortex contains binocular cells that detect matching features
- These cells compute disparity through phase differences
- Human vision uses feature-based matching, not block correlation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureMatch:
    """Matched feature pair between left and right images."""
    left_point: Tuple[float, float]
    right_point: Tuple[float, float]
    disparity: float
    confidence: float
    depth_m: Optional[float] = None


class CorrespondenceMatcher:
    """
    Feature-based correspondence matching between binocular views.

    Unlike traditional block matching, this mimics how human binocular
    neurons detect corresponding features through keypoint matching.
    """

    def __init__(
        self,
        feature_type: str = "orb",
        max_features: int = 500,
        match_ratio_threshold: float = 0.75,
        epipolar_threshold: float = 1.0
    ):
        """
        Initialize correspondence matcher.

        Args:
            feature_type: "orb", "sift", or "akaze"
            max_features: Maximum number of features to detect
            match_ratio_threshold: Lowe's ratio test threshold
            epipolar_threshold: Maximum distance from epipolar line (pixels)
        """
        self.feature_type = feature_type
        self.max_features = max_features
        self.match_ratio_threshold = match_ratio_threshold
        self.epipolar_threshold = epipolar_threshold

        # Initialize feature detector
        self.detector = self._create_detector(feature_type, max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING if feature_type == "orb" else cv2.NORM_L2)

    def _create_detector(self, feature_type: str, max_features: int):
        """Create appropriate feature detector."""
        if feature_type == "orb":
            return cv2.ORB_create(nfeatures=max_features)
        elif feature_type == "sift":
            return cv2.SIFT_create(nfeatures=max_features)
        elif feature_type == "akaze":
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def detect_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect features in image (monocular processing).

        Mimics retinal processing and V1 simple cells detecting edges/corners.

        Args:
            image: Grayscale image

        Returns:
            (keypoints, descriptors)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(
        self,
        kp_left: List,
        desc_left: np.ndarray,
        kp_right: List,
        desc_right: np.ndarray
    ) -> List[FeatureMatch]:
        """
        Match features between left and right images.

        Uses:
        1. KNN matching with Lowe's ratio test (confidence scoring)
        2. Epipolar constraint (features must be on same horizontal line)
        3. Disparity computation (horizontal displacement)

        Args:
            kp_left: Left image keypoints
            desc_left: Left image descriptors
            kp_right: Right image keypoints
            desc_right: Right image descriptors

        Returns:
            List of feature matches with disparity
        """
        if desc_left is None or desc_right is None:
            return []

        if len(desc_left) < 2 or len(desc_right) < 2:
            return []

        # KNN matching (k=2 for ratio test)
        matches = self.matcher.knnMatch(desc_left, desc_right, k=2)

        # Apply Lowe's ratio test and epipolar constraint
        good_matches = []

        for match_pair in matches:
            if len(match_pair) < 2:
                continue

            m, n = match_pair

            # Lowe's ratio test (mimics confidence in biological matching)
            if m.distance < self.match_ratio_threshold * n.distance:
                left_pt = kp_left[m.queryIdx].pt
                right_pt = kp_right[m.trainIdx].pt

                # Epipolar constraint: points should be on same horizontal line
                vertical_diff = abs(left_pt[1] - right_pt[1])

                if vertical_diff < self.epipolar_threshold:
                    # Compute disparity (horizontal displacement)
                    disparity = left_pt[0] - right_pt[0]

                    # Only accept positive disparity (right image point is left of left image point)
                    if disparity > 0:
                        confidence = 1.0 - (m.distance / (n.distance + 1e-6))

                        feature_match = FeatureMatch(
                            left_point=left_pt,
                            right_point=right_pt,
                            disparity=disparity,
                            confidence=confidence
                        )
                        good_matches.append(feature_match)

        return good_matches

    def compute_depths(
        self,
        matches: List[FeatureMatch],
        baseline_m: float,
        focal_length_px: float
    ) -> List[FeatureMatch]:
        """
        Compute depth for each matched feature.

        Uses triangulation: depth = (baseline * focal_length) / disparity

        Args:
            matches: Feature matches with disparity
            baseline_m: Distance between cameras (meters)
            focal_length_px: Camera focal length (pixels)

        Returns:
            Updated matches with depth values
        """
        for match in matches:
            if match.disparity > 0:
                # Triangulation formula
                match.depth_m = (baseline_m * focal_length_px) / match.disparity

        return matches

    def filter_by_confidence(
        self,
        matches: List[FeatureMatch],
        min_confidence: float = 0.6
    ) -> List[FeatureMatch]:
        """Filter matches by confidence threshold."""
        return [m for m in matches if m.confidence >= min_confidence]

    def get_disparity_map_from_features(
        self,
        matches: List[FeatureMatch],
        image_shape: Tuple[int, int],
        interpolation: bool = True
    ) -> np.ndarray:
        """
        Create sparse or dense disparity map from feature matches.

        Args:
            matches: Feature matches with disparity
            image_shape: (height, width) of image
            interpolation: If True, interpolate sparse disparities to dense map

        Returns:
            Disparity map (height x width)
        """
        disparity_map = np.zeros(image_shape, dtype=np.float32)

        if not matches:
            return disparity_map

        # Fill in sparse disparities
        for match in matches:
            x, y = int(match.left_point[0]), int(match.left_point[1])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                disparity_map[y, x] = match.disparity

        if interpolation:
            # Interpolate sparse disparities to create dense map
            # Use inpainting for regions without features
            mask = (disparity_map > 0).astype(np.uint8) * 255
            if np.any(mask):
                disparity_map = cv2.inpaint(
                    disparity_map,
                    255 - mask,
                    inpaintRadius=5,
                    flags=cv2.INPAINT_NS
                )

        return disparity_map

    def visualize_matches(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        matches: List[FeatureMatch],
        max_matches: int = 50
    ) -> np.ndarray:
        """
        Visualize feature matches between left and right images.

        Args:
            img_left: Left image
            img_right: Right image
            matches: Feature matches
            max_matches: Maximum number of matches to draw

        Returns:
            Visualization image
        """
        # Convert to color if grayscale
        if len(img_left.shape) == 2:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        if len(img_right.shape) == 2:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

        # Concatenate images side by side
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        h = max(h1, h2)

        vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = img_left
        vis[:h2, w1:w1+w2] = img_right

        # Draw matches
        for i, match in enumerate(matches[:max_matches]):
            # Color based on confidence
            color_intensity = int(255 * match.confidence)
            color = (0, color_intensity, 255 - color_intensity)

            pt1 = (int(match.left_point[0]), int(match.left_point[1]))
            pt2 = (int(match.right_point[0]) + w1, int(match.right_point[1]))

            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)

        return vis
