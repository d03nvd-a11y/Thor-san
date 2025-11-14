#!/usr/bin/env python3
"""
Experiment 06: Human-like Binocular Vision Demo ⭐

Demonstrates the core innovation: human-like binocular vision system.
Shows feature matching, temporal fusion, and visual attention.

This is the MOST IMPORTANT experiment showcasing biomimetic vision!
"""

import cv2
import yaml
import time
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.capture import MultiCameraCapture, FrameSynchronizer
from vision.binocular import BinocularVision, BinocularConfig


def main():
    print("=" * 60)
    print("EXPERIMENT 06: Human-like Binocular Vision Demo ⭐")
    print("=" * 60)
    print("\nThis system mimics HUMAN VISION, not traditional stereo cameras!")
    print("Key features:")
    print("  - Independent monocular processing (like human eyes)")
    print("  - Feature-based correspondence (like binocular neurons)")
    print("  - Temporal depth fusion (visual memory)")
    print("  - Visual attention mechanism")
    print("=" * 60)

    # Load configurations
    with open('configs/camera_config.yaml', 'r') as f:
        cam_config = yaml.safe_load(f)

    # Create binocular vision config (human-like parameters!)
    binocular_config = BinocularConfig(
        baseline_m=cam_config['binocular']['baseline_m'],  # Like human eyes: 6.5cm
        focal_length_px=cam_config['binocular']['focal_length_px'],
        feature_type=cam_config['binocular']['feature_type'],
        max_features=cam_config['binocular']['max_features'],
        temporal_window=cam_config['binocular']['temporal_window'],  # Visual memory
        temporal_decay=cam_config['binocular']['temporal_decay'],
        use_attention=cam_config['binocular']['use_attention']
    )

    print(f"\nBinocular Vision Configuration:")
    print(f"  Baseline: {binocular_config.baseline_m*100:.1f}cm (like human eyes!)")
    print(f"  Feature type: {binocular_config.feature_type}")
    print(f"  Temporal window: {binocular_config.temporal_window} frames (visual memory)")
    print(f"  Visual attention: {'ENABLED' if binocular_config.use_attention else 'DISABLED'}")

    # Initialize systems
    print("\n[INFO] Starting cameras...")
    multi_cam = MultiCameraCapture(cam_config)
    multi_cam.start()

    synchronizer = FrameSynchronizer(
        max_time_diff_ms=cam_config['sync']['max_time_diff_ms']
    )

    # Initialize binocular vision system
    binocular = BinocularVision(binocular_config)
    print("[INFO] Binocular vision system ready!")

    print("\n[CONTROLS]")
    print("  q - Quit")
    print("  a - Toggle attention regions")
    print("  r - Reset temporal memory")
    print("  1-4 - Switch visualization mode")
    print("  s - Show statistics")

    frame_count = 0
    start_time = time.time()
    show_attention_demo = False
    vis_mode = 1  # 1=depth, 2=features, 3=attention, 4=all

    try:
        while True:
            # Capture synchronized frames
            frames = multi_cam.get_frames()

            if not frames:
                time.sleep(0.01)
                continue

            synchronizer.add_frames(frames)
            sync_frames = synchronizer.get_synchronized_frames(['left', 'right'])

            if not sync_frames:
                continue

            # Get left and right images
            left_frame = sync_frames.frames['left'].frame
            right_frame = sync_frames.frames['right'].frame

            # Demo: Add task-driven attention region (simulating looking at center)
            if show_attention_demo:
                h, w = left_frame.shape[:2]
                binocular.add_task_attention(w//2, h//2, radius=100, weight=1.0, label="focus")

            # Process stereo pair with binocular vision
            timestamp = sync_frames.timestamp
            output = binocular.process_stereo_pair(left_frame, right_frame, timestamp)

            # Create visualizations
            visualizations = binocular.visualize_output(
                left_frame,
                output,
                show_features=True,
                show_depth=True,
                show_attention=True
            )

            # Display based on mode
            if vis_mode == 1:
                # Depth only
                if 'depth' in visualizations:
                    cv2.imshow("Binocular Depth", visualizations['depth'])
                if 'confidence' in visualizations:
                    cv2.imshow("Depth Confidence", visualizations['confidence'])

            elif vis_mode == 2:
                # Features only
                if 'features' in visualizations:
                    cv2.imshow("Feature Matches", visualizations['features'])

            elif vis_mode == 3:
                # Attention only
                if 'attention' in visualizations:
                    cv2.imshow("Visual Attention", visualizations['attention'])

            elif vis_mode == 4:
                # Show all
                for name, vis in visualizations.items():
                    cv2.imshow(name.capitalize(), vis)

            # Show input images
            stereo_view = np.hstack([left_frame, right_frame])
            cv2.putText(
                stereo_view,
                f"LEFT EYE | RIGHT EYE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("Stereo Input (Human-like Eyes)", stereo_view)

            frame_count += 1

            # Show info overlay
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"\r[INFO] FPS: {fps:.1f} | Matches: {output.num_matches} | "
                      f"Processing: {output.processing_time_ms:.1f}ms", end='')

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('a'):
                show_attention_demo = not show_attention_demo
                if not show_attention_demo:
                    binocular.clear_task_attention()
                print(f"\n[INFO] Attention demo: {'ON' if show_attention_demo else 'OFF'}")

            elif key == ord('r'):
                binocular.reset_temporal_memory()
                print("\n[INFO] Temporal memory reset")

            elif key == ord('1'):
                vis_mode = 1
                cv2.destroyAllWindows()
                print("\n[INFO] Visualization mode: DEPTH")

            elif key == ord('2'):
                vis_mode = 2
                cv2.destroyAllWindows()
                print("\n[INFO] Visualization mode: FEATURES")

            elif key == ord('3'):
                vis_mode = 3
                cv2.destroyAllWindows()
                print("\n[INFO] Visualization mode: ATTENTION")

            elif key == ord('4'):
                vis_mode = 4
                print("\n[INFO] Visualization mode: ALL")

            elif key == ord('s'):
                print("\n\n=== Binocular Vision Statistics ===")
                stats = binocular.get_statistics()
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"\n{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
                print()

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        print("\n[INFO] Shutting down...")
        multi_cam.stop()
        cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")

        stats = binocular.get_statistics()
        print(f"Average processing time: {stats['avg_processing_time_ms']:.1f}ms")
        print(f"Temporal fusion window: {stats['temporal_fusion']['window_size']} frames")

    print("\n✓ Experiment complete!")
    print("\nKey Insight: This system processes vision like HUMANS, not like")
    print("traditional stereo cameras. It uses feature correspondence, temporal")
    print("integration, and attention - just like biological vision systems!")


if __name__ == "__main__":
    main()
