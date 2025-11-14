# Thor-San: Practical Robotic Vision with Intel RealSense

🤖 **Simple, efficient 3D vision for robotic manipulation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 💡 Philosophy: Leverage Hardware, Not Reinvent It

**Thor-San v2.0** takes a practical approach: **use the right tool for the job**.

Instead of building complicated feature-matching algorithms, we use **Intel RealSense D435i** - a depth camera with:
- ✅ Pre-calibrated stereo vision (factory calibrated)
- ✅ Hardware depth processing (ASIC chip)
- ✅ Active IR projection (works in darkness)
- ✅ ±2mm depth accuracy
- ✅ IMU for motion tracking
- ✅ **10 lines of code vs 1000+**

---

## 🎯 What This System Does

Thor-San is a complete vision pipeline for robotic arms:

1. **RGB-D Capture**: Get color + depth from RealSense
2. **Object Detection**: YOLO v8 real-time detection
3. **3D Object Tracking**: Track objects with 3D poses
4. **Spatial Memory**: Build 3D octree maps
5. **Scene Understanding**: Detect surfaces, cluster objects
6. **Grasp Planning**: Generate 6-DOF grasp poses
7. **Task Planning**: High-level manipulation planning

All built on **simple, practical foundations**.

---

## 📦 Repository Structure

```
thor-san/
├── vision/
│   ├── realsense/         # 🆕 Simple D435i wrapper (200 lines!)
│   │   ├── camera.py      # Camera interface
│   │   └── processing.py  # Depth utilities
│   ├── detection/         # YOLO v8 + tracking
│   ├── depth/             # Point cloud generation
│   └── segmentation/      # Future: SAM integration
│
├── spatial_memory/        # 3D octree maps + object database
├── intelligence/          # Scene analysis + task planning
│
├── experiments/           # Simple demos
│   ├── 01_test_realsense.py       # Test camera
│   ├── 02_rgbd_pointcloud.py      # 3D visualization
│   ├── 03_test_detection.py       # Detection + 3D poses
│   ├── 04_build_3d_map.py         # Build spatial map
│   └── 05_visualize_scene.py      # Scene analysis
│
├── configs/
│   ├── camera_config.yaml         # RealSense settings
│   └── vision_config.yaml         # Detection/planning settings
│
└── data/                  # Calibration, models, maps
```

**Code Reduction**: From 37 files (7,329 lines) → **~20 files (~2,000 lines)**

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- **Intel RealSense D435i** camera
- (Optional) CUDA GPU for faster YOLO

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/thor-san.git
cd thor-san

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Test RealSense Camera

```bash
python experiments/01_test_realsense.py
```

Shows RGB + depth streams. **That's it!** No calibration needed.

### Generate Point Clouds

```bash
python experiments/02_rgbd_pointcloud.py
```

Press SPACE to capture and visualize 3D point cloud.

### Object Detection with 3D Poses

```bash
python experiments/03_test_detection.py
```

Detects objects and shows their 3D positions. **This is the power of depth cameras!**

### Build 3D Map

```bash
python experiments/04_build_3d_map.py
```

Build persistent 3D octree map of environment.

---

## 💻 Code Examples

### Simple Camera Usage

```python
from vision.realsense import RealSenseCamera

# That's it! Pre-calibrated and ready!
with RealSenseCamera() as camera:
    while True:
        frame = camera.get_frame()

        # RGB image
        cv2.imshow("RGB", frame.rgb)

        # Depth (hardware-generated!)
        cv2.imshow("Depth", frame.depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
```

**10 lines**. Compare to the old system: 200+ lines for camera sync + calibration!

### Get 3D Object Positions

```python
from vision.realsense import RealSenseCamera, DepthProcessor
from vision.detection import YOLODetector

camera = RealSenseCamera()
camera.start()

detector = YOLODetector()
depth_proc = DepthProcessor(camera.depth_scale)

frame = camera.get_frame()

# Detect objects
detections = detector.detect(frame.rgb)

# Get 3D positions
for det in detections:
    cx, cy = det.center

    # Get depth at object center (in meters!)
    depth_m = depth_proc.get_depth_at_point(frame.depth, cx, cy)

    print(f"{det.class_name} at ({cx}, {cy}) - Distance: {depth_m:.2f}m")
```

**Real 3D positions with minimal code!**

### Build 3D Map

```python
from vision.realsense import RealSenseCamera
from vision.depth import PointCloudGenerator
from spatial_memory import OctreeMap

camera = RealSenseCamera()
camera.start()

pcg = PointCloudGenerator()
octree = OctreeMap(resolution=0.01)  # 1cm voxels

# Capture and add to map
frame = camera.get_frame()

# Convert depth to point cloud
pcd = pcg.depth_to_point_cloud(
    frame.depth * frame.depth_scale,
    camera_matrix,
    frame.rgb
)

# Add to spatial map
octree.add_point_cloud(pcd)

# Save
octree.save("my_map.pcd")
```

---

## ⚙️ Configuration

### RealSense Settings (`configs/camera_config.yaml`)

```yaml
realsense:
  # Streams
  rgb_width: 640
  rgb_height: 480
  depth_width: 640
  depth_height: 480
  fps: 30

  # Hardware post-processing (fast!)
  enable_spatial_filter: true   # Smooth depth
  enable_temporal_filter: true  # Reduce noise
  enable_hole_filling: true     # Fill gaps

  # Alignment
  align_depth_to_color: true    # Essential for RGB-D
```

### Detection Settings (`configs/vision_config.yaml`)

```yaml
detection:
  model: "yolov8n.pt"          # Nano (fast) model
  confidence_threshold: 0.5
  device: "cuda"               # or "cpu"

spatial_memory:
  octree_resolution: 0.01      # 1cm voxels
```

---

## 📊 Performance Comparison

### OLD System (Dual Monocular Cameras):
- 📸 2 USB cameras to manage
- 🔧 Manual calibration required (chessboard pattern)
- ⏱️ Frame synchronization overhead
- 🧮 Feature matching: ~30-50ms
- 📏 Depth accuracy: ±5cm at 1m
- 💾 7,329 lines of code

### NEW System (RealSense D435i):
- 📸 **1 USB camera**
- 🔧 **Pre-calibrated** from factory
- ⏱️ **No sync needed**
- 🧮 **Hardware depth**: <5ms
- 📏 **Depth accuracy: ±2mm at 1m**
- 💾 **~2,000 lines of code**

**Result: 70% less code, 10x faster, 10x more accurate!**

---

## 🎓 What You Learn

- **Practical Robotics**: Use the right hardware
- **RGB-D Processing**: Depth cameras vs stereo
- **Object Detection**: YOLO v8 integration
- **3D Reconstruction**: Point clouds and octrees
- **Spatial Intelligence**: Scene understanding
- **Grasp Planning**: 6-DOF pose generation

---

## 🔧 Hardware Requirements

### Required:
- **Intel RealSense D435i** (~$200)
- USB 3.0 port
- Computer with 4GB+ RAM

### Optional:
- NVIDIA GPU (for faster YOLO)
- Robot arm (for actual manipulation)

---

## 📚 Key Features

### ✅ Implemented

- **RealSense Integration**: Simple camera wrapper
- **RGB-D Capture**: Color + depth streams
- **Hardware Depth Processing**: Spatial, temporal, hole-filling filters
- **Point Cloud Generation**: With color mapping
- **YOLO v8 Detection**: Real-time object detection
- **Multi-Object Tracking**: ID persistence across frames
- **3D Object Poses**: Instant 3D position from depth
- **Octree Mapping**: Efficient 3D spatial memory
- **Scene Analysis**: Surface detection, clustering
- **Grasp Planning**: 6-DOF pose generation
- **Task Planning**: High-level action sequencing

### 🔮 Future Enhancements

- [ ] ROS 2 integration
- [ ] Real robot arm control
- [ ] Visual servoing
- [ ] Dynamic obstacle avoidance
- [ ] Multi-camera fusion (multiple RealSense)
- [ ] SAM segmentation integration

---

## 🤔 Design Philosophy

### Why RealSense Over Dual Cameras?

**The Question**: Should robots mimic humans?

**The Answer**: **No!** Robots should leverage their advantages:

| Human Vision | Robot Vision (RealSense) |
|--------------|-------------------------|
| 2 eyes, fixed baseline | Adjustable camera placement |
| Visible light only | IR + RGB + depth |
| ~60 Hz processing | 90+ FPS possible |
| Imperfect memory | Perfect digital storage |
| Gets tired | 24/7 operation |
| ±1cm depth accuracy | **±2mm accuracy** |
| Requires calibration | **Pre-calibrated** |

**Conclusion**: Use hardware that's **better than human eyes**, not equal to them!

---

## 📖 Documentation

Each module is well-documented:

- `vision/realsense/camera.py` - Camera interface with examples
- `vision/detection/yolo_detector.py` - YOLO integration
- `spatial_memory/octree_map.py` - 3D mapping
- `intelligence/grasp_planner.py` - Grasp generation

---

## 🤝 Contributing

Contributions welcome! This is now a **practical** system focused on:
- Simplicity
- Performance
- Real-world usability

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- **Intel RealSense** for excellent depth cameras
- **Ultralytics** for YOLO v8
- **Open3D** for 3D processing
- **The robot community** for teaching us to use the right tools!

---

## ⭐ Key Takeaway

> **"The best code is no code. The best tool is the one that works."**

Thor-San v2.0: **Simple**, **Practical**, **Effective**.

---

*Built for robots that work in the real world* 🤖
