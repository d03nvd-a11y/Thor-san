#!/usr/bin/env python3
"""
Experiment 03: Object Detection + Tracking with Bag Files

SAME as 03_test_detection.py but uses bag file instead of live camera.
Perfect for testing without physical RealSense!

Just change the BAG_FILE path below.
"""

import cv2
import yaml
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.realsense import BagFilePlayer, DepthProcessor  # Changed: BagFilePlayer instead of RealSenseCamera
from vision.detection import YOLODetector, ObjectTracker


# ========== CONFIGURATION ==========
# Set your bag file path here
BAG_FILE = "data/bags/outdoor_scene.bag"  # Change this to your bag file

# Or automatically find first available bag file
def find_first_bag():
    data_dir = Path(__file__).parent.parent / 'data' / 'bags'
    bag_files = list(data_dir.glob('*.bag'))
    return str(bag_files[0]) if bag_files else None


def main():
    print("=" * 60)
    print("EXPERIMENT 03: Detection + Tracking (BAG FILE)")
    print("=" * 60)

    # Find bag file
    bag_path = BAG_FILE
    if not os.path.exists(bag_path):
        print(f"[WARN] Bag file not found: {bag_path}")
        print("[INFO] Searching for available bag files...")
        bag_path = find_first_bag()

    if not bag_path or not os.path.exists(bag_path):
        print("\n[ERROR] No bag files found!")
        print("\nTo download sample bag files:")
        print("  python scripts/download_sample_bags.py")
        return

    print(f"\n[INFO] Using bag file: {bag_path}")

    # Load configs
    with open('configs/camera_config.yaml', 'r') as f:
        cam_config = yaml.safe_load(f)

    with open('configs/vision_config.yaml', 'r') as f:
        vis_config = yaml.safe_load(f)

    # Initialize BagFilePlayer (SAME INTERFACE as RealSenseCamera!)
    print("\n[INFO] Starting bag playback...")
    camera = BagFilePlayer(bag_path, loop=True, real_time=False)
    camera.start()

    depth_processor = DepthProcessor(camera.depth_scale)

    # Initialize YOLO
    print("[INFO] Loading YOLO...")
    try:
        detector = YOLODetector(
            model_name=vis_config['detection']['model'],
            confidence_threshold=vis_config['detection']['confidence_threshold'],
            device=vis_config['detection'].get('device', 'cpu')
        )
    except Exception as e:
        print(f"[ERROR] YOLO failed: {e}")
        camera.stop()
        return

    # Initialize tracker
    tracker = ObjectTracker(
        max_age=vis_config['tracking']['max_age'],
        min_hits=vis_config['tracking']['min_hits']
    )

    print("\n[CONTROLS]")
    print("  q - Quit")
    print("  d - Toggle tracking")
    print("  3 - Toggle 3D info overlay")
    print("  SPACE - Pause/Resume")

    use_tracking = True
    show_3d = True
    paused = False
    frame_count = 0

    try:
        while True:
            if not paused:
                frame = camera.get_frame()
                if frame is None:
                    print("\n[INFO] End of bag file (looping...)")
                    continue

                # Detect objects
                detections = detector.detect(frame.rgb)

                # Update tracker
                if use_tracking:
                    tracks = tracker.update(detections)

                    # Get 3D positions
                    for track in tracks:
                        cx, cy = track.detection.center

                        # Get depth at object center
                        depth_m = depth_processor.get_depth_at_point(
                            frame.depth, cx, cy
                        )

                        # Add 3D info to track
                        track.depth_m = depth_m

                    # Visualize
                    vis_frame = tracker.visualize_tracks(frame.rgb, tracks)

                    # Add 3D overlay
                    if show_3d:
                        for track in tracks:
                            if hasattr(track, 'depth_m') and track.depth_m > 0:
                                x1, y1, _, _ = track.detection.bbox
                                text = f"Z: {track.depth_m:.2f}m"
                                cv2.putText(
                                    vis_frame, text,
                                    (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 255), 2
                                )

                    info_text = f"Tracks: {len(tracks)}"
                else:
                    vis_frame = detector.visualize_detections(frame.rgb, detections)
                    info_text = f"Detections: {len(detections)}"

                # FPS info
                frame_count += 1
                cv2.putText(
                    vis_frame, info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                # Bag file indicator
                cv2.putText(
                    vis_frame, "BAG FILE",
                    (10, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 165, 255), 2
                )

                cv2.imshow("Detection + Tracking (Bag File)", vis_frame)
                if frame.depth_colormap is not None:
                    cv2.imshow("Depth", frame.depth_colormap)

            # Controls
            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d'):
                use_tracking = not use_tracking
                if not use_tracking:
                    tracker.reset()
                print(f"\n[INFO] Tracking: {'ON' if use_tracking else 'OFF'}")
            elif key == ord('3'):
                show_3d = not show_3d
                print(f"\n[INFO] 3D overlay: {'ON' if show_3d else 'OFF'}")
            elif key == ord(' '):
                paused = not paused
                print(f"\n[INFO] {'PAUSED' if paused else 'RESUMED'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\n✓ Experiment complete!")
    print("\n💡 This works exactly like the live camera version!")
    print("   Same detection, tracking, and 3D pose estimation.")


if __name__ == "__main__":
    main()
