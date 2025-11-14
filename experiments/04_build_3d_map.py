#!/usr/bin/env python3
"""
Experiment 04: Build 3D Spatial Map

Builds a 3D octree map from binocular vision depth data.
Demonstrates spatial memory system.
"""

import cv2
import yaml
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.capture import MultiCameraCapture, FrameSynchronizer
from vision.binocular import BinocularVision, BinocularConfig
from spatial_memory import OctreeMap
from vision.depth import PointCloudGenerator


def main():
    print("=" * 60)
    print("EXPERIMENT 04: Build 3D Spatial Map")
    print("=" * 60)

    # Load configurations
    with open('configs/camera_config.yaml', 'r') as f:
        cam_config = yaml.safe_load(f)

    with open('configs/vision_config.yaml', 'r') as f:
        vis_config = yaml.safe_load(f)

    # Initialize systems
    print("\n[INFO] Starting cameras...")
    multi_cam = MultiCameraCapture(cam_config)
    multi_cam.start()

    synchronizer = FrameSynchronizer(
        max_time_diff_ms=cam_config['sync']['max_time_diff_ms']
    )

    # Binocular vision
    binocular_config = BinocularConfig(
        baseline_m=cam_config['binocular']['baseline_m'],
        focal_length_px=cam_config['binocular']['focal_length_px'],
        feature_type=cam_config['binocular']['feature_type'],
        max_features=cam_config['binocular']['max_features'],
        temporal_window=cam_config['binocular']['temporal_window']
    )
    binocular = BinocularVision(binocular_config)

    # 3D map
    octree_map = OctreeMap(
        resolution=vis_config['spatial_memory']['octree_resolution'],
        max_depth=vis_config['spatial_memory']['max_depth']
    )

    # Point cloud generator
    pcg = PointCloudGenerator()

    # Camera intrinsic matrix (approximate)
    focal_length = binocular_config.focal_length_px
    camera_matrix = np.array([
        [focal_length, 0, 320],
        [0, focal_length, 240],
        [0, 0, 1]
    ])

    print("[INFO] Systems ready!")
    print("\n[CONTROLS]")
    print("  q - Quit and save map")
    print("  c - Clear map")
    print("  v - Visualize current map")
    print("  s - Show statistics")

    frame_count = 0
    map_update_interval = 5  # Update map every N frames

    try:
        while True:
            # Get synchronized frames
            frames = multi_cam.get_frames()

            if not frames:
                time.sleep(0.01)
                continue

            synchronizer.add_frames(frames)
            sync_frames = synchronizer.get_synchronized_frames(['left', 'right'])

            if not sync_frames:
                continue

            left_frame = sync_frames.frames['left'].frame
            right_frame = sync_frames.frames['right'].frame

            # Process with binocular vision
            output = binocular.process_stereo_pair(left_frame, right_frame)

            frame_count += 1

            # Update map periodically
            if frame_count % map_update_interval == 0:
                # Generate point cloud from depth
                pcd = pcg.depth_to_point_cloud(
                    output.depth_map,
                    camera_matrix,
                    left_frame
                )

                # Filter point cloud
                if len(pcd.points) > 0:
                    pcd = pcg.filter_point_cloud(
                        pcd,
                        voxel_size=vis_config['spatial_memory']['voxel_downsample'],
                        remove_outliers=True
                    )

                    # Add to map
                    octree_map.add_point_cloud(pcd)

                    print(f"\r[INFO] Frame {frame_count} | Map points: {octree_map.num_points_added} | "
                          f"Updates: {octree_map.num_updates}", end='')

            # Visualize depth
            vis_depth = binocular.visualize_output(left_frame, output, show_features=False)
            if 'depth' in vis_depth:
                cv2.imshow("Depth Map", vis_depth['depth'])

            cv2.imshow("Left Camera", left_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                octree_map.clear()
                print("\n[INFO] Map cleared")

            elif key == ord('v'):
                print("\n[INFO] Visualizing map...")
                try:
                    octree_map.visualize()
                except Exception as e:
                    print(f"[WARN] Visualization failed: {e}")

            elif key == ord('s'):
                print("\n\n=== Map Statistics ===")
                stats = octree_map.get_statistics()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        print("\n[INFO] Shutting down...")
        multi_cam.stop()
        cv2.destroyAllWindows()

        # Save map
        print("\n[INFO] Saving 3D map...")
        map_path = "data/maps/octree_map.pcd"
        os.makedirs("data/maps", exist_ok=True)
        octree_map.save(map_path)

        # Final statistics
        print("\n=== Final Map Statistics ===")
        stats = octree_map.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n✓ Experiment complete!")
    print(f"✓ Map saved to: {map_path}")


if __name__ == "__main__":
    main()
