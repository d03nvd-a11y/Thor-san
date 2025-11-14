#!/usr/bin/env python3
"""
Experiment 01: Test Multi-Camera System

Tests camera access and frame capture from multiple USB cameras.
Displays live feeds from all cameras.
"""

import cv2
import yaml
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.capture import MultiCameraCapture, FrameSynchronizer


def main():
    print("=" * 60)
    print("EXPERIMENT 01: Multi-Camera System Test")
    print("=" * 60)

    # Load camera configuration
    with open('configs/camera_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfigured cameras:")
    for cam_id, cam_config in config['cameras'].items():
        if cam_config.get('enabled', True):
            print(f"  - {cam_id}: index={cam_config['index']}, resolution={cam_config['resolution']}")

    # Initialize multi-camera capture
    print("\n[INFO] Starting cameras...")
    multi_cam = MultiCameraCapture(config)
    multi_cam.start()

    # Initialize frame synchronizer
    sync = FrameSynchronizer(
        max_time_diff_ms=config['sync']['max_time_diff_ms'],
        buffer_size=config['sync']['buffer_size']
    )

    print("\n[INFO] Cameras started! Press 'q' to quit.")
    print("[INFO] Press 's' to show synchronization statistics.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Get frames from all cameras
            frames = multi_cam.get_frames()

            # Add to synchronizer
            if frames:
                sync.add_frames(frames)

                # Get synchronized frames (left + right for binocular vision)
                required_cameras = ['left', 'right']
                sync_frames = sync.get_synchronized_frames(required_cameras)

                if sync_frames:
                    # Display frames
                    for cam_id, cam_frame in sync_frames.frames.items():
                        cv2.imshow(f"Camera: {cam_id}", cam_frame.frame)

                    frame_count += 1

                    # Show stats every 30 frames
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"[INFO] FPS: {fps:.1f}, Sync time diff: {sync_frames.max_time_diff*1000:.1f}ms")

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n=== Synchronization Statistics ===")
                stats = sync.get_statistics()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()

            elif key == ord('i'):
                print("\n=== Camera Information ===")
                info = multi_cam.get_camera_info()
                for cam_id, cam_info in info.items():
                    print(f"\n{cam_id}:")
                    for key, value in cam_info.items():
                        print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        print("\n[INFO] Stopping cameras...")
        multi_cam.stop()
        cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n=== Final Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")

        sync_stats = sync.get_statistics()
        print(f"Sync success rate: {sync_stats['success_rate']:.1f}%")

    print("\n[INFO] Experiment complete!")


if __name__ == "__main__":
    main()
