"""
detector.py — YOLOv8 object detection wrapper with smart filtering.

Encapsulates model loading, inference, result parsing, class filtering,
minimum-area filtering, and temporal smoothing so the rest of the system
only works with clean Python data structures.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class Detection:
    """A single detected object in one frame."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class ObjectDetector:
    """Thin wrapper around Ultralytics YOLOv8 with smart filtering."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence: float = config.CONFIDENCE_THRESHOLD,
    ):
        print(f"[Detector] Loading model: {model_path} (conf ≥ {confidence:.0%}) …")
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Map class IDs → human-readable names from the loaded model
        self.class_names: dict[int, str] = self.model.names  # e.g. {0: 'person', …}

        # Build the set of allowed class IDs from config
        self._allowed_ids: set[int] | None = None
        if config.TRACKED_CLASSES is not None:
            self._allowed_ids = set()
            for cid, cname in self.class_names.items():
                if cname in config.TRACKED_CLASSES:
                    self._allowed_ids.add(cid)
            print(f"[Detector] Tracking {len(self._allowed_ids)} classes: "
                  f"{[self.class_names[i] for i in sorted(self._allowed_ids)]}")
        else:
            print(f"[Detector] Ready — {len(self.class_names)} classes loaded (no filter).")

        # ── Temporal smoothing state ──────────────
        # Tracks how many consecutive frames each (class_id, grid_cell) has been seen.
        self._persistence: defaultdict[tuple, int] = defaultdict(int)
        self._prev_keys: set[tuple] = set()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Returns a list of `Detection` objects (may be empty).
        Applies: class filtering → confidence filtering → min-area → temporal smoothing.
        """
        # `verbose=False` silences per-frame console output from Ultralytics
        results = self.model(
            frame,
            conf=self.confidence,
            iou=config.IOU_THRESHOLD,
            half=config.HALF_PRECISION,
            imgsz=config.IMGSZ,
            verbose=False,
        )

        raw_detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])

                # ── Class filter ──────────────────
                if self._allowed_ids is not None and cls_id not in self._allowed_ids:
                    continue

                conf = float(box.conf[0])

                # ── Per-class confidence ──────────
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                min_conf = config.CONFIDENCE_PER_CLASS.get(class_name, self.confidence)
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # ── Minimum area filter ───────────
                area = (x2 - x1) * (y2 - y1)
                if area < config.MIN_BOX_AREA:
                    continue

                raw_detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        # ── Temporal smoothing ────────────────
        if config.DETECTION_PERSISTENCE_FRAMES <= 1:
            return raw_detections

        return self._apply_temporal_smoothing(raw_detections)

    def _apply_temporal_smoothing(self, detections: list[Detection]) -> list[Detection]:
        """
        Only emit a detection if the same class has been detected in roughly
        the same location for N consecutive frames. This eliminates flickering
        false positives without adding latency for real detections.
        """
        grid_size = 80  # pixels — quantise position to a coarse grid
        current_keys: set[tuple] = set()
        smoothed: list[Detection] = []

        for det in detections:
            # Create a spatial key: (class_id, grid_x, grid_y)
            cx = (det.x1 + det.x2) // 2
            cy = (det.y1 + det.y2) // 2
            key = (det.class_id, cx // grid_size, cy // grid_size)
            current_keys.add(key)

            self._persistence[key] += 1

            if self._persistence[key] >= config.DETECTION_PERSISTENCE_FRAMES:
                smoothed.append(det)

        # Decay keys that were not seen this frame
        stale = self._prev_keys - current_keys
        for key in stale:
            self._persistence.pop(key, None)

        self._prev_keys = current_keys
        return smoothed
