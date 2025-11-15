#!/usr/bin/env python3
"""
Experiment 00: Test Bag File Playback

Tests BagFilePlayer to verify it works exactly like live camera.
Perfect for developing without physical RealSense!
"""

import cv2
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.realsense import BagFilePlayer


def find_bag_files():
    """Find available bag files."""
    data_dir = Path(__file__).parent.parent / 'data' / 'bags'
    if not data_dir.exists():
        return []

    bag_files = list(data_dir.glob('*.bag'))
    return bag_files


def main():
    print("=" * 60)
    print("EXPERIMENT 00: Bag File Playback Test")
    print("=" * 60)

    # Find available bag files
    bag_files = find_bag_files()

    if not bag_files:
        print("\n[ERROR] No bag files found!")
        print("\nTo download sample bag files:")
        print("  python scripts/download_sample_bags.py")
        print("\nOr place your own .bag files in: data/bags/")
        return

    # Show available files
    print("\nAvailable bag files:")
    for i, bag_file in enumerate(bag_files, 1):
        size_mb = bag_file.stat().st_size / 1024 / 1024
        print(f"  {i}. {bag_file.name} ({size_mb:.1f} MB)")

    # Select file
    if len(bag_files) == 1:
        selected = bag_files[0]
        print(f"\nUsing: {selected.name}")
    else:
        choice = input(f"\nSelect bag file (1-{len(bag_files)}): ").strip()
        try:
            idx = int(choice) - 1
            selected = bag_files[idx]
        except:
            print("[ERROR] Invalid selection")
            return

    print("\n[CONTROLS]")
    print("  q - Quit")
    print("  i - Show camera info")
    print("  SPACE - Pause/Resume")
    print("  s - Save current frame")

    # Initialize bag player
    camera = BagFilePlayer(
        str(selected),
        loop=True,        # Loop when reaching end
        real_time=False   # Process as fast as possible
    )
    camera.start()

    frame_count = 0
    paused = False

    try:
        while True:
            if not paused:
                # Get frame (works exactly like RealSenseCamera!)
                frame = camera.get_frame()

                if frame is None:
                    print("\n[INFO] End of bag file reached")
                    break

                # Display RGB
                if frame.rgb is not None:
                    cv2.imshow("Bag File - RGB", frame.rgb)

                # Display depth colormap
                if frame.depth_colormap is not None:
                    cv2.imshow("Bag File - Depth", frame.depth_colormap)

                # Show frame info
                if frame_count % 30 == 0:
                    print(f"\r[INFO] Frame {frame_count}", end='')

                frame_count += 1

            # Keyboard controls
            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('i'):
                print("\n\n=== Camera Info ===")
                info = camera.get_camera_info()
                for k, v in info.items():
                    print(f"  {k}: {v}")
                print()

            elif key == ord(' '):  # SPACE
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n[INFO] Playback {status}")

            elif key == ord('s'):
                # Save frame
                if frame.rgb is not None:
                    filename = f"frame_{frame_count}_rgb.png"
                    cv2.imwrite(filename, frame.rgb)
                    print(f"\n[INFO] Saved {filename}")
                if frame.depth_colormap is not None:
                    filename = f"frame_{frame_count}_depth.png"
                    cv2.imwrite(filename, frame.depth_colormap)
                    print(f"[INFO] Saved {filename}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\n✓ Test complete!")
    print("\n💡 TIP: BagFilePlayer works exactly like RealSenseCamera!")
    print("   You can use it in ALL experiments by changing:")
    print("     camera = RealSenseCamera()")
    print("   to:")
    print("     camera = BagFilePlayer('data/bags/your_file.bag')")


if __name__ == "__main__":
    main()
