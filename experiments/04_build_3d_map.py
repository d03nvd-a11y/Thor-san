#!/usr/bin/env python3
"""
Experiment 04: Build 3D Map (SIMPLIFIED)

Build 3D octree map from RealSense depth data.
MUCH simpler than the old dual-camera version!
"""

import cv2
import yaml
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.realsense import RealSenseCamera, RealSenseConfig
from vision.depth import PointCloudGenerator
from spatial_memory import OctreeMap


def main():
    print("=" * 60)
    print("EXPERIMENT 04: Build 3D Map (Simple!)")
    print("=" * 60)

    # Load config
    with open('configs/camera_config.yaml', 'r') as f:
        cam_config = yaml.safe_load(f)

    with open('configs/vision_config.yaml', 'r') as f:
        vis_config = yaml.safe_load(f)

    # Initialize
    print("\n[INFO] Starting RealSense...")
    config = RealSenseConfig(enable_rgb=True, enable_depth=True, align_depth_to_color=True)
    camera = RealSenseCamera(config)
    camera.start()

    pcg = PointCloudGenerator()

    octree_map = OctreeMap(
        resolution=vis_config['spatial_memory']['octree_resolution']
    )

    print("\n[CONTROLS]")
    print("  q - Quit and save map")
    print("  c - Clear map")
    print("  v - Visualize current map")
    print("  SPACE - Add current frame to map")

    frame_count = 0
    map_frames = 0

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            # Display
            cv2.imshow("RGB", frame.rgb)
            cv2.imshow("Depth", frame.depth_colormap)

            # Info overlay
            info = f"Frames: {frame_count} | Map updates: {map_frames}"
            frame_copy = frame.rgb.copy()
            cv2.putText(frame_copy, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("RGB", frame_copy)

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                print("\n[INFO] Adding frame to map...")

                # Generate point cloud
                intr = frame.color_intrinsics
                camera_matrix = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]
                ])

                depth_m = frame.depth * frame.depth_scale
                pcd = pcg.depth_to_point_cloud(depth_m, camera_matrix, frame.rgb)
                pcd = pcg.filter_point_cloud(pcd, voxel_size=0.005)

                # Add to map
                octree_map.add_point_cloud(pcd)
                map_frames += 1

                print(f"[INFO] Map now has {octree_map.num_points_added} points")

            elif key == ord('c'):
                octree_map.clear()
                map_frames = 0
                print("\n[INFO] Map cleared")

            elif key == ord('v'):
                print("\n[INFO] Visualizing map...")
                try:
                    octree_map.visualize()
                except Exception as e:
                    print(f"[WARN] Visualization failed: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

        # Save map
        if map_frames > 0:
            print("\n[INFO] Saving map...")
            octree_map.save("data/maps/octree_map.pcd")

            stats = octree_map.get_statistics()
            print("\n=== Map Statistics ===")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    print("\n✓ Experiment complete!")
    print("💡 Notice: No calibration, no sync, just works!")


if __name__ == "__main__":
    main()
