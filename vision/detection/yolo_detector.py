"""
YOLO v8 Object Detection

Real-time object detection using YOLOv8.
Provides bounding boxes, class labels, and confidence scores.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not available. Install with: pip install ultralytics")


@dataclass
class Detection:
    """Single object detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[int, int]  # (cx, cy)


class YOLODetector:
    """
    YOLO v8 object detector.

    Provides real-time object detection for robotic vision.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        target_classes: Optional[List[str]] = None
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLO model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            device: "cuda" or "cpu"
            target_classes: List of class names to detect (None = all classes)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes

        # Load model
        print(f"[INFO] Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

        # Move to device
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                else:
                    print("[WARN] CUDA not available, using CPU")
                    self.device = "cpu"
            except ImportError:
                print("[WARN] PyTorch not available, using CPU")
                self.device = "cpu"

        # Get class names
        self.class_names = self.model.names

        # Filter target class IDs
        if target_classes is not None:
            self.target_class_ids = self._get_class_ids(target_classes)
        else:
            self.target_class_ids = None

        print(f"[INFO] YOLO detector ready (device={self.device})")

    def _get_class_ids(self, class_names: List[str]) -> List[int]:
        """Get class IDs from class names."""
        class_ids = []
        for name in class_names:
            for class_id, class_name in self.class_names.items():
                if class_name.lower() == name.lower():
                    class_ids.append(class_id)
                    break
        return class_ids

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in image.

        Args:
            image: Input image (BGR)

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        # Parse results
        detections = []

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    # Filter by target classes
                    if self.target_class_ids is not None:
                        if class_id not in self.target_class_ids:
                            continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=self.class_names[class_id],
                        confidence=float(conf),
                        center=(cx, cy)
                    )
                    detections.append(detection)

        return detections

    def detect_and_visualize(
        self,
        image: np.ndarray,
        show_conf: bool = True,
        show_labels: bool = True
    ) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect objects and create visualization.

        Args:
            image: Input image
            show_conf: Show confidence scores
            show_labels: Show class labels

        Returns:
            (detections, visualization_image)
        """
        detections = self.detect(image)
        vis_image = self.visualize_detections(image, detections, show_conf, show_labels)
        return detections, vis_image

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_conf: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw detections on image.

        Args:
            image: Input image
            detections: List of detections
            show_conf: Show confidence scores
            show_labels: Show class labels

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Define colors for different classes
        np.random.seed(42)
        colors = {
            class_id: tuple(map(int, np.random.randint(0, 255, 3)))
            for class_id in self.class_names.keys()
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_id, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if show_labels or show_conf:
                label_parts = []
                if show_labels:
                    label_parts.append(det.class_name)
                if show_conf:
                    label_parts.append(f"{det.confidence:.2f}")

                label = " ".join(label_parts)

                # Get text size for background
                (text_w, text_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    1
                )

                # Draw background
                cv2.rectangle(
                    vis,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    vis,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

            # Draw center point
            cv2.circle(vis, det.center, 4, color, -1)

        return vis

    def filter_by_class(
        self,
        detections: List[Detection],
        class_names: List[str]
    ) -> List[Detection]:
        """Filter detections by class names."""
        return [
            det for det in detections
            if det.class_name in class_names
        ]

    def get_largest_detection(
        self,
        detections: List[Detection]
    ) -> Optional[Detection]:
        """Get detection with largest bounding box area."""
        if not detections:
            return None

        largest = max(
            detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        )
        return largest

    def get_detection_by_center(
        self,
        detections: List[Detection],
        point: Tuple[int, int]
    ) -> Optional[Detection]:
        """Get detection whose bounding box contains point."""
        x, y = point

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return det

        return None
