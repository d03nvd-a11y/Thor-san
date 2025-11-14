#!/usr/bin/env python3
"""
Experiment 02: RGB-D Point Cloud Visualization

Generate and visualize 3D point clouds from RealSense depth data.
"""

import cv2
import yaml
import sys
import os
import open3d as o3d
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.realsense import RealSenseCamera, RealSenseConfig
from vision.depth import PointCloudGenerator


def main():
    print("=" * 60)
    print("EXPERIMENT 02: RGB-D Point Cloud")
    print("=" * 60)

    # Load config
    with open('configs/camera_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)

    rs_cfg = config_dict['realsense']
    config = RealSenseConfig(
        rgb_width=rs_cfg['rgb_width'],
        rgb_height=rs_cfg['rgb_height'],
        depth_width=rs_cfg['depth_width'],
        depth_height=rs_cfg['depth_height'],
        enable_rgb=True,
        enable_depth=True,
        align_depth_to_color=True
    )

    print("\n[CONTROLS]")
    print("  q - Quit")
    print("  SPACE - Capture and visualize point cloud")
    print("  s - Save point cloud")

    # Initialize
    camera = RealSenseCamera(config)
    camera.start()

    pcg = PointCloudGenerator()

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                continue

            # Display
            if frame.rgb is not None:
                cv2.imshow("RGB", frame.rgb)
            if frame.depth_colormap is not None:
                cv2.imshow("Depth", frame.depth_colormap)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                print("\n[INFO] Generating point cloud...")

                # Create camera matrix from intrinsics
                intr = frame.color_intrinsics
                camera_matrix = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]
                ])

                # Generate point cloud
                depth_m = frame.depth * frame.depth_scale
                pcd = pcg.depth_to_point_cloud(
                    depth_m,
                    camera_matrix,
                    frame.rgb
                )

                print(f"[INFO] Point cloud has {len(pcd.points)} points")

                # Filter
                pcd = pcg.filter_point_cloud(pcd, voxel_size=0.005)
                print(f"[INFO] After filtering: {len(pcd.points)} points")

                # Visualize
                print("[INFO] Visualizing (close window to continue)...")
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name="Point Cloud",
                    width=800,
                    height=600
                )

            elif key == ord('s'):
                print("\n[INFO] Saving point cloud...")

                intr = frame.color_intrinsics
                camera_matrix = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]
                ])

                depth_m = frame.depth * frame.depth_scale
                pcd = pcg.depth_to_point_cloud(depth_m, camera_matrix, frame.rgb)
                pcd = pcg.filter_point_cloud(pcd)

                filename = "point_cloud.pcd"
                o3d.io.write_point_cloud(filename, pcd)
                print(f"[INFO] Saved to {filename}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()
