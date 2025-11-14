#!/usr/bin/env python3
"""
Experiment 02: Stereo Camera Calibration

Calibrates stereo camera pair using chessboard pattern.
Generates calibration parameters for accurate depth reconstruction.
"""

import cv2
import yaml
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.capture import MultiCameraCapture, FrameSynchronizer
from vision.depth import StereoCalibrator


def main():
    print("=" * 60)
    print("EXPERIMENT 02: Stereo Camera Calibration")
    print("=" * 60)

    print("\nYou will need:")
    print("  - A chessboard calibration pattern (9x6 or similar)")
    print("  - Print it and mount on flat surface")
    print("  - Show pattern to both cameras from different angles")

    # Load camera configuration
    with open('configs/camera_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize cameras
    print("\n[INFO] Starting cameras...")
    multi_cam = MultiCameraCapture(config)
    multi_cam.start()

    synchronizer = FrameSynchronizer(
        max_time_diff_ms=config['sync']['max_time_diff_ms']
    )

    # Initialize calibrator
    calibrator = StereoCalibrator(
        chessboard_size=(9, 6),  # Adjust to your chessboard
        square_size=0.025  # 2.5cm squares (adjust to your pattern)
    )

    print("\n[CONTROLS]")
    print("  SPACE - Capture calibration image pair")
    print("  c - Complete calibration")
    print("  q - Quit without saving")
    print(f"\nTarget: Capture at least 10 image pairs from different angles")

    num_captured = 0
    required_images = 10

    try:
        while True:
            # Get synchronized frames
            frames = multi_cam.get_frames()

            if not frames:
                continue

            synchronizer.add_frames(frames)
            sync_frames = synchronizer.get_synchronized_frames(['left', 'right'])

            if not sync_frames:
                continue

            left_frame = sync_frames.frames['left'].frame
            right_frame = sync_frames.frames['right'].frame

            # Display
            display = cv2.hconcat([left_frame, right_frame])

            # Add instructions
            status = f"Captured: {num_captured}/{required_images} | Press SPACE to capture"
            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Stereo Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Capture calibration pair
                print(f"\n[INFO] Capturing image pair {num_captured + 1}...")

                success = calibrator.add_calibration_images(left_frame, right_frame)

                if success:
                    num_captured += 1
                    print(f"[SUCCESS] Captured {num_captured}/{required_images}")

                    if num_captured >= required_images:
                        print("[INFO] Sufficient images captured! Press 'c' to calibrate")
                else:
                    print("[WARN] Chessboard not found in both images. Try again.")

            elif key == ord('c'):
                if num_captured < 10:
                    print(f"\n[WARN] Need at least 10 images, have {num_captured}")
                    continue

                print("\n[INFO] Starting calibration...")
                h, w = left_frame.shape[:2]

                try:
                    results = calibrator.calibrate((w, h))

                    print("\n=== Calibration Results ===")
                    print(f"Left camera RMS: {results['rms_left']:.4f}")
                    print(f"Right camera RMS: {results['rms_right']:.4f}")
                    print(f"Stereo RMS: {results['rms_stereo']:.4f}")
                    print(f"Baseline: {results['baseline_m']*100:.2f} cm")

                    # Save calibration
                    calib_path = "data/calibration/stereo_calibration.yaml"
                    os.makedirs("data/calibration", exist_ok=True)
                    calibrator.save_calibration(calib_path)

                    print(f"\n✓ Calibration saved to: {calib_path}")

                    # Show rectification quality
                    print("\n[INFO] Showing rectification quality...")
                    vis = calibrator.visualize_calibration_quality(left_frame, right_frame)
                    cv2.imshow("Rectification Quality (epipolar lines should be horizontal)", vis)
                    cv2.waitKey(3000)

                    break

                except Exception as e:
                    print(f"\n[ERROR] Calibration failed: {e}")
                    print("[INFO] Try capturing more images from different angles")

            elif key == ord('q'):
                print("\n[INFO] Calibration cancelled")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        multi_cam.stop()
        cv2.destroyAllWindows()

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()
