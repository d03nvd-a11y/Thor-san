#!/usr/bin/env python3
"""
Experiment 01: Test RealSense D435i

Simple test to verify RealSense camera is working.
Shows RGB and depth streams.
"""

import cv2
import yaml
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.realsense import RealSenseCamera, RealSenseConfig


def main():
    print("=" * 60)
    print("EXPERIMENT 01: RealSense D435i Test")
    print("=" * 60)

    # Load config
    with open('configs/camera_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create RealSense config
    rs_cfg = config_dict['realsense']
    config = RealSenseConfig(
        rgb_width=rs_cfg['rgb_width'],
        rgb_height=rs_cfg['rgb_height'],
        rgb_fps=rs_cfg['rgb_fps'],
        depth_width=rs_cfg['depth_width'],
        depth_height=rs_cfg['depth_height'],
        depth_fps=rs_cfg['depth_fps'],
        enable_rgb=rs_cfg['enable_rgb'],
        enable_depth=rs_cfg['enable_depth'],
        enable_decimation=rs_cfg['enable_decimation'],
        enable_spatial_filter=rs_cfg['enable_spatial_filter'],
        enable_temporal_filter=rs_cfg['enable_temporal_filter'],
        enable_hole_filling=rs_cfg['enable_hole_filling'],
        align_depth_to_color=rs_cfg['align_depth_to_color']
    )

    print("\n[CONTROLS]")
    print("  q - Quit")
    print("  i - Show camera info")
    print("  s - Save current frame")

    # Initialize camera
    camera = RealSenseCamera(config)
    camera.start()

    frame_count = 0

    try:
        while True:
            # Get frame
            frame = camera.get_frame()

            if frame is None:
                continue

            # Display RGB
            if frame.rgb is not None:
                cv2.imshow("RealSense RGB", frame.rgb)

            # Display depth colormap
            if frame.depth_colormap is not None:
                cv2.imshow("RealSense Depth", frame.depth_colormap)

            # Show frame info
            if frame_count % 30 == 0:
                print(f"\r[INFO] Frame {frame_count} | FPS: ~{config.rgb_fps}", end='')

            frame_count += 1

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('i'):
                print("\n\n=== Camera Info ===")
                info = camera.get_camera_info()
                for k, v in info.items():
                    print(f"  {k}: {v}")
                print()

            elif key == ord('s'):
                # Save frame
                if frame.rgb is not None:
                    cv2.imwrite(f"frame_{frame_count}_rgb.png", frame.rgb)
                if frame.depth_colormap is not None:
                    cv2.imwrite(f"frame_{frame_count}_depth.png", frame.depth_colormap)
                print(f"\n[INFO] Saved frame {frame_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
