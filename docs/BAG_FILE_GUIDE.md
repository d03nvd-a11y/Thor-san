# Bag File Testing Guide

Testing Thor-San vision system without physical RealSense D435i hardware.

---

## What are Bag Files?

**Bag files** (.bag) are recordings from Intel RealSense cameras containing:
- RGB video stream
- Depth video stream
- Camera intrinsics
- Timestamps

They play back **exactly like a live camera** - perfect for development and testing!

---

## Quick Start

### 1. Download Sample Bag Files

```bash
python scripts/download_sample_bags.py
```

This downloads sample recordings to `data/bags/`:
- `outdoor_scene.bag` - Outdoor scene with depth (~50MB)
- `stairs.bag` - Indoor stairs navigation (~30MB)

### 2. Test Playback

```bash
python experiments/00_test_bag_file.py
```

This will:
- Show available bag files
- Play RGB and depth streams
- Demonstrate pause/resume controls

### 3. Run Full Pipeline

```bash
python experiments/03_test_detection_BAG.py
```

This runs the complete detection + tracking + 3D pipeline using bag files!

---

## Using Bag Files in Your Code

### Replace Live Camera

```python
# Before (live camera):
from vision.realsense import RealSenseCamera
camera = RealSenseCamera()

# After (bag file):
from vision.realsense import BagFilePlayer
camera = BagFilePlayer("data/bags/outdoor_scene.bag", loop=True)
```

**That's it!** Everything else works exactly the same.

### Example

```python
from vision.realsense import BagFilePlayer
from vision.detection import YOLODetector

# Load bag file
camera = BagFilePlayer("data/bags/outdoor_scene.bag")
camera.start()

detector = YOLODetector()

# Process frames (same as live camera!)
while True:
    frame = camera.get_frame()
    if frame is None:
        break

    detections = detector.detect(frame.rgb)
    print(f"Found {len(detections)} objects")

camera.stop()
```

---

## BagFilePlayer Options

```python
BagFilePlayer(
    bag_file_path="data/bags/scene.bag",
    loop=True,        # Loop playback when reaching end
    real_time=False   # False = as fast as possible
                      # True = play at recorded speed
)
```

---

## Finding More Bag Files

### Public Datasets

1. **TUM RGB-D Dataset**
   - https://vision.in.tum.de/data/datasets/rgbd-dataset
   - Indoor scenes with objects
   - Download `.bag` files directly

2. **Intel RealSense Samples**
   - https://github.com/IntelRealSense/librealsense
   - Official test data

3. **Record Your Own**
   - Use Intel RealSense Viewer
   - Tools → Record to .bag file
   - Share with team members!

### Recording Your Own

If you have access to a RealSense camera temporarily:

```bash
# Install RealSense Viewer
sudo apt install realsense-viewer

# Open viewer
realsense-viewer

# Click "Record" button
# Move camera around your workspace
# Stop recording → saves .bag file
```

---

## Advantages

✅ **Develop without hardware**
- No RealSense needed
- Faster iteration
- Portable development

✅ **Reproducible testing**
- Same scene every time
- Easier debugging
- Share test cases

✅ **Dataset creation**
- Record once, test forever
- Build benchmark datasets
- Version control your test data

---

## Limitations

⚠️ **What bag files CAN'T do:**
- Can't move the camera (pre-recorded)
- Can't test different lighting (fixed)
- Can't test with new objects (unless recorded)

💡 **Solution:** Record multiple bag files for different scenarios!

---

## Converting to Bag Files

### From Kinect/Other Sensors

Use ROS bag format, then convert:

```bash
# Install tools
pip install pyrealsense2 rosbag

# Convert (example script)
python scripts/convert_to_realsense_bag.py input.bag output.bag
```

### From Video + Depth Images

Create synthetic bag files from image sequences:

```python
# Coming soon: scripts/create_bag_from_images.py
```

---

## Troubleshooting

### "No bag files found"

```bash
# Download samples
python scripts/download_sample_bags.py

# Or place your own .bag files in:
mkdir -p data/bags
cp /path/to/your/file.bag data/bags/
```

### "Playback finished immediately"

- Bag file might be corrupted
- Try downloading again
- Check file size (should be >1MB)

### "No depth stream in bag file"

- Some bag files only have RGB
- Download a different sample
- Record your own with depth enabled

---

## Best Practices

1. **Keep bag files small** (<100MB for quick testing)
2. **Name descriptively** (`office_desk_01.bag`, not `test.bag`)
3. **Record variety** (different angles, lighting, object arrangements)
4. **Version control metadata** (not the .bag itself, too large)
5. **Document contents** (what objects, what scene)

---

## Example Workflow

```
Day 1: Record bag file at office
       ↓
Day 2-10: Develop algorithms at home using bag file
       ↓
Day 11: Test with live camera at office
       ↓
Day 12: Record new bag files with edge cases
       ↓
Day 13+: Continue development with new test cases
```

---

## Next Steps

Once your algorithms work with bag files:

1. Test with live RealSense camera
2. Record bag files of failures for debugging
3. Build benchmark dataset
4. Deploy to robot!

---

**The bag file system lets you develop 90% of the vision pipeline without hardware!**
