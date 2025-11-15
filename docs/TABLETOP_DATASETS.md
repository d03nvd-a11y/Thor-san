# Tabletop Manipulation Datasets

**For robot arms on tables viewing objects** - this is what you need!

---

## Best Datasets for Your Use Case

### 🥇 RECOMMENDED: TUM RGB-D Desk Scene

**Perfect for tabletop robot arm testing!**

```bash
# Download via script
python scripts/download_sample_bags.py
# Select: 1 (tabletop_manipulation)
```

**What's in it:**
- Desk with various objects (keyboard, mouse, monitor, cups, etc.)
- Multiple viewing angles
- Good depth quality
- Indoor lighting
- ~180MB

**Why it's perfect:**
- Same scenario as robot arm on table
- Detectable objects (YOLO recognizes keyboards, mice, cups, etc.)
- Realistic workspace clutter
- Multiple object types

---

## Dataset Comparison for Tabletop Arms

| Dataset | Scene Type | Objects | Size | Best For |
|---------|-----------|---------|------|----------|
| **TUM Desk** ✅ | Office desk | Keyboard, mouse, monitor, cups | 180MB | **Robot arm development** |
| TUM Cabinet | Storage cabinet | Books, boxes on shelves | 240MB | Shelf manipulation |
| TUM Room | Full room | Furniture, multiple objects | 150MB | Scene understanding |
| Test Small | Stairs | Navigation | 30MB | Quick testing only |

---

## What These Datasets Contain

### TUM Desk (tabletop_manipulation.bag)

**Camera view:** Looking down at desk from ~1m height (similar to robot arm camera!)

**Visible objects:**
- Computer keyboard ✓ (YOLO detects)
- Computer mouse ✓ (YOLO detects)
- Monitor ✓ (YOLO detects)
- Cup/mug ✓ (YOLO detects)
- Books ✓ (YOLO detects)
- Various desk items

**Scene characteristics:**
- Indoor office lighting
- Cluttered workspace (realistic!)
- Objects at various distances (0.3m - 2m)
- Good for testing grasp planning

**Download directly:**
```bash
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.bag -P data/bags/
```

---

## Additional Tabletop Datasets

### YCB Video Dataset (Advanced)

**Best for:** Training manipulation algorithms

- 92 objects from YCB benchmark
- Tabletop scenes with pose annotations
- RGB-D video sequences
- Download: https://rse-lab.cs.washington.edu/projects/posecnn/

**Note:** Large dataset (~100GB), needs conversion to .bag format

### Cornell Grasping Dataset

**Best for:** Grasp detection training

- RGB-D images of objects on table
- Grasp rectangle annotations
- 885 images of 240 objects
- Download: http://pr.cs.cornell.edu/grasping/rect_data/data.php

### Princeton ModelNet (3D Models)

**Best for:** Object recognition

- 3D CAD models of objects
- Can render synthetic scenes
- Good for training

---

## Creating Your Own Tabletop Dataset

### Option 1: Use Phone Camera + MiDaS (No Hardware)

```python
# Record video with phone
# Use MiDaS depth estimation
# Create pseudo RGB-D dataset
# (See docs/BAG_FILE_GUIDE.md)
```

### Option 2: Borrow RealSense Temporarily

```bash
# 1. Get access to RealSense (friend, lab, etc.)
# 2. Set up your actual workspace
# 3. Record multiple scenarios:

# Scenario 1: Empty table
realsense-viewer → record → empty_table.bag

# Scenario 2: Single object
realsense-viewer → record → single_cup.bag

# Scenario 3: Multiple objects
realsense-viewer → record → cluttered_desk.bag

# Scenario 4: Different arrangements
realsense-viewer → record → workspace_01.bag
```

**Tips for recording:**
- Mount camera where robot camera will be
- Use actual objects robot will manipulate
- Record different lighting conditions
- Capture various object arrangements
- 30-60 seconds per scenario is enough

### Option 3: Simulation

```python
# PyBullet simulation
import pybullet as p
import pybullet_data

# Create table + objects
# Render RGB-D from robot viewpoint
# Save as bag file
# (More complex, but reproducible)
```

---

## Recommended Workflow for Your Robot Arm

### Phase 1: Development (Now - No Hardware)

```bash
# Download TUM desk scene
python scripts/download_sample_bags.py
# Select: tabletop_manipulation

# Develop algorithms
python experiments/03_test_detection_BAG.py
```

**What to test:**
- ✓ Object detection (keyboard, mouse, cup, etc.)
- ✓ Object tracking (persistent IDs)
- ✓ 3D pose estimation
- ✓ Grasp planning on detected objects
- ✓ Collision checking with table

### Phase 2: Real Data Collection (When you get RealSense)

```bash
# Set up your actual workspace
# Record 5-10 bag files:
1. empty_workspace.bag
2. single_object_*.bag (different objects)
3. multi_object_*.bag (2-3 objects)
4. cluttered_*.bag (many objects)
5. edge_cases_*.bag (occlusions, etc.)
```

### Phase 3: Integration (With robot)

```python
# Switch from bag file to live camera
# camera = BagFilePlayer(...)  # Development
camera = RealSenseCamera()     # Production

# Use your tested algorithms!
# Everything works the same
```

---

## Object Detection Expectations

### Objects YOLO v8 Can Detect on Tables:

✅ **Common desk/table objects:**
- cup, bottle, bowl
- keyboard, mouse, laptop
- book, cell phone
- scissors, pen (sometimes)
- vase, clock

✅ **Kitchen table objects:**
- cup, wine glass, bowl, fork, knife, spoon
- bottle, banana, apple, orange
- cake, donut, pizza (if present)

❌ **NOT detected (need custom training):**
- Screws, bolts, small parts
- Custom tools
- Unmarked boxes
- Novel objects

**Solution for custom objects:**
- Train YOLO on your specific objects
- Use segmentation (SAM) instead
- Color-based detection for simple cases

---

## Performance on Tabletop Data

### Expected Results with TUM Desk:

```
Detection:
  Keyboard: 90-95% confidence
  Mouse: 70-85% confidence
  Cup: 85-95% confidence
  Monitor: 80-90% confidence

Tracking:
  Stable IDs: ✓
  Motion tracking: ✓ (if camera moves)

3D Poses:
  Accuracy: ±2cm with RealSense
  Depth range: 0.3m - 2.0m (perfect for table)

Grasp Planning:
  Grasps generated: 5-10 per object
  Quality scores: 0.6-0.9
```

---

## Troubleshooting

### "No objects detected"

- Bag file might not contain YOLO-recognizable objects
- Try `tabletop_manipulation.bag` (guaranteed to have objects)
- Lower confidence threshold in config

### "Depth is noisy on objects"

- Normal for some bag files (older sensors)
- RealSense D435i has better quality
- Use depth filters (already enabled)

### "Objects too far away"

- TUM datasets have 1-2m distances (fine)
- Robot arm usually works at 0.5-1.5m
- Perfect range overlap

---

## Next Steps

1. **Download recommended dataset:**
   ```bash
   python scripts/download_sample_bags.py
   # Select: 1 (tabletop_manipulation)
   ```

2. **Test full pipeline:**
   ```bash
   python experiments/03_test_detection_BAG.py
   ```

3. **Develop your algorithms** using this data

4. **When you get RealSense:** Record your own workspace

5. **Switch to live camera** with 1 line change!

---

**The TUM desk dataset is the closest thing to your robot arm use case!** 🎯
