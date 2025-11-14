#!/usr/bin/env python3
"""
Experiment 05: Scene Visualization and Analysis

Loads a 3D map and performs scene analysis:
- Surface detection
- Object clustering
- Workspace analysis
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spatial_memory import OctreeMap
from intelligence import SceneAnalyzer
import open3d as o3d


def main():
    print("=" * 60)
    print("EXPERIMENT 05: Scene Visualization and Analysis")
    print("=" * 60)

    # Check if map exists
    map_path = "data/maps/octree_map.pcd"

    if not os.path.exists(map_path):
        print(f"\n[ERROR] No map found at {map_path}")
        print("[INFO] Run experiment 04 first to build a 3D map")
        return

    # Load map
    print(f"\n[INFO] Loading map from {map_path}...")
    octree_map = OctreeMap()
    octree_map.load(map_path)

    # Convert to point cloud
    pcd = octree_map.to_point_cloud()
    print(f"[INFO] Loaded {len(pcd.points)} points")

    # Initialize scene analyzer
    analyzer = SceneAnalyzer(
        plane_threshold=0.01,
        min_plane_points=100
    )

    print("\n[INFO] Analyzing scene...")

    # Comprehensive scene analysis
    scene_analysis = analyzer.analyze_scene(pcd)

    # Print results
    print("\n" + "=" * 60)
    print("SCENE ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nSurfaces detected: {scene_analysis['num_surfaces']}")
    print(f"Support surfaces: {scene_analysis['num_support_surfaces']}")

    if scene_analysis['support_surfaces']:
        print("\nSupport Surface Details:")
        for i, surface in enumerate(scene_analysis['support_surfaces'], 1):
            print(f"  Surface {i}:")
            print(f"    Type: {surface.surface_type}")
            print(f"    Center: ({surface.center[0]:.3f}, {surface.center[1]:.3f}, {surface.center[2]:.3f})")
            print(f"    Area: {surface.area:.3f} m²")
            print(f"    Height: {surface.center[2]:.3f} m")

    print(f"\nGround height: {scene_analysis['ground_height']:.3f} m")

    print(f"\nWorkspace bounds:")
    wb = scene_analysis['workspace_bounds']
    print(f"  Min: ({wb.min_bound[0]:.3f}, {wb.min_bound[1]:.3f}, {wb.min_bound[2]:.3f})")
    print(f"  Max: ({wb.max_bound[0]:.3f}, {wb.max_bound[1]:.3f}, {wb.max_bound[2]:.3f})")
    print(f"  Volume: {wb.reachable_volume:.3f} m³")

    print(f"\nObject clusters: {scene_analysis['num_object_clusters']}")
    print(f"Obstacle points: {scene_analysis['num_obstacles']}")

    # Visualize
    print("\n[INFO] Visualizing scene...")
    print("[INFO] Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Close window to continue")

    # Create visualization with detected surfaces
    geometries = [pcd]

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,
        origin=[0, 0, 0]
    )
    geometries.append(coord_frame)

    # Visualize support surfaces
    for i, surface in enumerate(scene_analysis['support_surfaces']):
        # Create a plane mesh at surface location
        plane = o3d.geometry.TriangleMesh.create_box(
            width=0.3,
            height=0.3,
            depth=0.01
        )

        # Move to surface center
        plane.translate(surface.center - np.array([0.15, 0.15, 0.005]))

        # Color based on height
        color = [0, 1, 0] if i == 0 else [0, 0.5, 1]  # Green for lowest, blue for others
        plane.paint_uniform_color(color)

        geometries.append(plane)

    # Visualize object clusters
    clusters = scene_analysis['object_clusters']
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    for i, cluster in enumerate(clusters[:6]):  # Show first 6 clusters
        cluster.paint_uniform_color(colors[i])
        geometries.append(cluster)

    # Show visualization
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Scene Analysis",
        width=1024,
        height=768
    )

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()
