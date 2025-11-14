#!/usr/bin/env python3
"""
Experiment 03: Object Detection + Tracking with 3D Poses

YOLO detection + multi-object tracking + 3D position from depth.
This is where RealSense shines - we get object 3D poses instantly!
"""

import cv2
import yaml
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.realsense import RealSenseCamera, RealSenseConfig, DepthProcessor
from vision.detection import YOLODetector, ObjectTracker


def main():
    print("=" * 60)
    print("EXPERIMENT 03: Detection + Tracking + 3D Poses")
    print("=" * 60)

    # Load configs
    with open('configs/camera_config.yaml', 'r') as f:
        cam_config = yaml.safe_load(f)

    with open('configs/vision_config.yaml', 'r') as f:
        vis_config = yaml.safe_load(f)

    # Initialize RealSense
    print("\n[INFO] Starting RealSense...")
    rs_cfg = cam_config['realsense']
    config = RealSenseConfig(
        enable_rgb=True,
        enable_depth=True,
        align_depth_to_color=True
    )
    camera = RealSenseCamera(config)
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

    use_tracking = True
    show_3d = True
    frame_count = 0

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
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

            cv2.imshow("Detection + Tracking + 3D", vis_frame)
            cv2.imshow("Depth", frame.depth_colormap)

            # Controls
            key = cv2.waitKey(1) & 0xFF

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

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\n✓ Experiment complete!")
    print("💡 Notice how easy it is to get 3D poses with RealSense!")


if __name__ == "__main__":
    main()
