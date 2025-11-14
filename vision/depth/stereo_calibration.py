"""
Stereo Camera Calibration

Calibrates stereo camera pairs to obtain intrinsic and extrinsic parameters.
Required for accurate depth reconstruction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os
import yaml


class StereoCalibrator:
    """
    Stereo camera calibration system.

    Uses chessboard pattern to calibrate camera pair.
    """

    def __init__(
        self,
        chessboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # meters (2.5cm)
    ):
        """
        Initialize stereo calibrator.

        Args:
            chessboard_size: (columns, rows) of internal chessboard corners
            square_size: Size of chessboard square in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[
            0:chessboard_size[0],
            0:chessboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size

        # Storage for calibration points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane

        # Calibration results
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix

        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi_left = None
        self.roi_right = None

    def add_calibration_images(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> bool:
        """
        Add stereo image pair for calibration.

        Args:
            img_left: Left camera image
            img_right: Right camera image

        Returns:
            True if chessboard found in both images, False otherwise
        """
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left,
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right,
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret_left and ret_right:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            corners_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria
            )
            corners_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria
            )

            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)

            return True

        return False

    def calibrate(self, image_size: Tuple[int, int]) -> dict:
        """
        Perform stereo calibration.

        Args:
            image_size: (width, height) of images

        Returns:
            Calibration results dictionary
        """
        if len(self.objpoints) < 10:
            raise ValueError(f"Need at least 10 calibration images, got {len(self.objpoints)}")

        print(f"[INFO] Calibrating with {len(self.objpoints)} image pairs...")

        # Calibrate individual cameras
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints_left,
            image_size,
            None,
            None
        )

        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints_right,
            image_size,
            None,
            None
        )

        print(f"[INFO] Left camera RMS: {ret_left:.4f}")
        print(f"[INFO] Right camera RMS: {ret_right:.4f}")

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret_stereo, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            image_size,
            criteria=criteria,
            flags=flags
        )

        print(f"[INFO] Stereo calibration RMS: {ret_stereo:.4f}")

        # Stereo rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            image_size,
            self.R,
            self.T,
            alpha=0  # 0 = only valid pixels, 1 = all pixels
        )

        # Compute baseline
        baseline = np.linalg.norm(self.T)

        results = {
            'rms_left': ret_left,
            'rms_right': ret_right,
            'rms_stereo': ret_stereo,
            'baseline_m': baseline,
            'num_calibration_images': len(self.objpoints),
            'image_size': image_size
        }

        print(f"[INFO] Baseline: {baseline:.4f} meters")

        return results

    def save_calibration(self, filepath: str):
        """Save calibration parameters to file."""
        calibration_data = {
            'camera_matrix_left': self.camera_matrix_left.tolist(),
            'dist_coeffs_left': self.dist_coeffs_left.tolist(),
            'camera_matrix_right': self.camera_matrix_right.tolist(),
            'dist_coeffs_right': self.dist_coeffs_right.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist(),
            'R1': self.R1.tolist(),
            'R2': self.R2.tolist(),
            'P1': self.P1.tolist(),
            'P2': self.P2.tolist(),
            'Q': self.Q.tolist(),
            'roi_left': self.roi_left,
            'roi_right': self.roi_right
        }

        with open(filepath, 'w') as f:
            yaml.dump(calibration_data, f)

        print(f"[INFO] Calibration saved to {filepath}")

    def load_calibration(self, filepath: str):
        """Load calibration parameters from file."""
        with open(filepath, 'r') as f:
            calibration_data = yaml.safe_load(f)

        self.camera_matrix_left = np.array(calibration_data['camera_matrix_left'])
        self.dist_coeffs_left = np.array(calibration_data['dist_coeffs_left'])
        self.camera_matrix_right = np.array(calibration_data['camera_matrix_right'])
        self.dist_coeffs_right = np.array(calibration_data['dist_coeffs_right'])
        self.R = np.array(calibration_data['R'])
        self.T = np.array(calibration_data['T'])
        self.E = np.array(calibration_data['E'])
        self.F = np.array(calibration_data['F'])
        self.R1 = np.array(calibration_data['R1'])
        self.R2 = np.array(calibration_data['R2'])
        self.P1 = np.array(calibration_data['P1'])
        self.P2 = np.array(calibration_data['P2'])
        self.Q = np.array(calibration_data['Q'])
        self.roi_left = calibration_data['roi_left']
        self.roi_right = calibration_data['roi_right']

        print(f"[INFO] Calibration loaded from {filepath}")

    def get_rectification_maps(
        self,
        image_size: Tuple[int, int]
    ) -> Tuple[Tuple, Tuple]:
        """
        Get rectification maps for undistortion and rectification.

        Returns:
            ((map_left_x, map_left_y), (map_right_x, map_right_y))
        """
        map_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.R1,
            self.P1,
            image_size,
            cv2.CV_32FC1
        )

        map_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.R2,
            self.P2,
            image_size,
            cv2.CV_32FC1
        )

        return map_left, map_right

    def rectify_images(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.

        Returns:
            (rectified_left, rectified_right)
        """
        image_size = (img_left.shape[1], img_left.shape[0])
        map_left, map_right = self.get_rectification_maps(image_size)

        rectified_left = cv2.remap(
            img_left,
            map_left[0],
            map_left[1],
            cv2.INTER_LINEAR
        )

        rectified_right = cv2.remap(
            img_right,
            map_right[0],
            map_right[1],
            cv2.INTER_LINEAR
        )

        return rectified_left, rectified_right

    def visualize_calibration_quality(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> np.ndarray:
        """
        Visualize rectification quality with epipolar lines.

        Returns:
            Visualization image
        """
        rect_left, rect_right = self.rectify_images(img_left, img_right)

        # Concatenate side by side
        h = max(rect_left.shape[0], rect_right.shape[0])
        w = rect_left.shape[1] + rect_right.shape[1]

        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[:rect_left.shape[0], :rect_left.shape[1]] = rect_left
        vis[:rect_right.shape[0], rect_left.shape[1]:] = rect_right

        # Draw horizontal epipolar lines
        for y in range(0, h, 40):
            cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)

        return vis
