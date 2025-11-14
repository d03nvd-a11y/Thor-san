#!/usr/bin/env python3
"""
Experiment 03: YOLO Object Detection and Tracking

Tests real-time object detection and multi-object tracking.
"""

import cv2
import yaml
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.capture import MultiCameraCapture
from vision.detection import YOLODetector, ObjectTracker


def main():
    print("=" * 60)
    print("EXPERIMENT 03: Object Detection and Tracking")
    print("=" * 60)

    # Load configurations
    with open('configs/camera_config.yaml', 'r') as f:
        cam_config = yaml.safe_load(f)

    with open('configs/vision_config.yaml', 'r') as f:
        vis_config = yaml.safe_load(f)

    # Initialize camera (use left camera)
    print("\n[INFO] Starting camera...")
    multi_cam = MultiCameraCapture(cam_config)
    multi_cam.start()

    # Initialize YOLO detector
    print("[INFO] Loading YOLO detector...")
    try:
        detector = YOLODetector(
            model_name=vis_config['detection']['model'],
            confidence_threshold=vis_config['detection']['confidence_threshold'],
            iou_threshold=vis_config['detection']['iou_threshold'],
            device=vis_config['detection']['device'],
            target_classes=vis_config['detection'].get('target_classes', None)
        )
        print("[INFO] YOLO detector ready!")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        print("[INFO] Make sure ultralytics is installed: pip install ultralytics")
        multi_cam.stop()
        return

    # Initialize tracker
    tracker = ObjectTracker(
        max_age=vis_config['tracking']['max_age'],
        min_hits=vis_config['tracking']['min_hits'],
        iou_threshold=vis_config['tracking']['iou_threshold']
    )

    print("\n[CONTROLS]")
    print("  q - Quit")
    print("  d - Toggle detection only (no tracking)")
    print("  s - Show statistics")

    frame_count = 0
    start_time = time.time()
    use_tracking = True

    try:
        while True:
            # Get frame from left camera
            frames = multi_cam.get_frames()

            if 'left' not in frames:
                time.sleep(0.01)
                continue

            frame = frames['left'].frame

            # Detect objects
            detections = detector.detect(frame)

            if use_tracking:
                # Update tracker
                tracks = tracker.update(detections)

                # Visualize tracks
                vis_frame = tracker.visualize_tracks(
                    frame,
                    tracks,
                    show_trajectory=True,
                    show_id=True
                )

                # Show info
                info_text = f"Tracks: {len(tracks)} | Detections: {len(detections)}"
            else:
                # Just show detections
                vis_frame = detector.visualize_detections(
                    frame,
                    detections,
                    show_conf=True,
                    show_labels=True
                )

                info_text = f"Detections: {len(detections)}"

            # Add FPS overlay
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                info_text += f" | FPS: {fps:.1f}"

            cv2.putText(
                vis_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Object Detection & Tracking", vis_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('d'):
                use_tracking = not use_tracking
                if not use_tracking:
                    tracker.reset()
                print(f"\n[INFO] Tracking: {'ON' if use_tracking else 'OFF'}")

            elif key == ord('s'):
                print("\n=== Statistics ===")
                if use_tracking:
                    tracker_stats = tracker.get_statistics()
                    for key, value in tracker_stats.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  Total detections: {len(detections)}")
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        print("\n[INFO] Shutting down...")
        multi_cam.stop()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n=== Final Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()
