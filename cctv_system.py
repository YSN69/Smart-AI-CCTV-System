"""
cctv_system.py — Core orchestrator for the Smart AI CCTV System.

Manages the camera feed, drives detection + tracking + suspicious-activity
analysis, draws the annotated frame, and exposes a clean start/stop lifecycle.

Pipeline: capture → detect → filter → track → analyse → draw → display
"""

import time

import cv2
import numpy as np

import config
from detector import ObjectDetector, Detection
from logger import EventLogger
from suspicious_activity import SuspiciousActivityDetector
from tracker import PersonTracker, TrackedPerson
from utils import draw_detection, draw_tracked_person, draw_overlay, get_colour_for_class


class CCTVSystem:
    """High-level CCTV pipeline: capture → detect → track → analyse → display → log."""

    def __init__(self):
        # Core components
        self.logger = EventLogger()
        self.detector = ObjectDetector()
        self.tracker = PersonTracker()
        self.suspicious = SuspiciousActivityDetector(self.logger)

        # Video capture handle (initialised in `start()`)
        self.cap: cv2.VideoCapture | None = None

        # Runtime state
        self._running = False
        self._fps = 0.0
        self._active_alerts: list = []
        self._frame_count = 0
        self._alert_expire_time = 0.0

        # Cache for frame-skip: reuse previous results on skipped frames
        self._cached_detections: list[Detection] = []
        self._cached_tracked: list[TrackedPerson] = []

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────
    def start(self) -> None:
        """Open the video source and enter the main processing loop."""
        self._open_camera()
        self.logger.log_system("CCTV system started")
        self._running = True

        tracking_status = "ON" if self.tracker.enabled else "OFF"

        print(f"\n{'='*55}")
        print(f"   {config.WINDOW_NAME}")
        print(f"   Tracking: {tracking_status}")
        print(f"   Press 'q' to quit  |  Press 's' to save snapshot")
        print(f"{'='*55}\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[System] Interrupted by user.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Release resources gracefully."""
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.flush()  # Ensure all buffered log entries are written
        self.logger.log_system("CCTV system stopped")
        self.logger.flush()  # Flush the stop event too
        print("[System] Shutdown complete.")

    # ──────────────────────────────────────────
    # Camera management
    # ──────────────────────────────────────────
    def _open_camera(self) -> None:
        """Open the configured camera / video file."""
        source = config.CAMERA_SOURCE
        print(f"[System] Opening video source: {source}")
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source '{source}'. "
                "Check your camera index or file path in config.py."
            )

        # Apply resolution settings
        if config.FRAME_WIDTH:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        if config.FRAME_HEIGHT:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[System] Capture resolution: {actual_w}×{actual_h}")

    # ──────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────
    def _main_loop(self) -> None:
        """Read → detect → track → analyse → draw → show, frame by frame."""
        prev_time = time.time()
        detection_log_interval = 5.0       # log detections every N seconds
        last_detection_log = 0.0

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("[System] End of video stream (or read error).")
                break

            self._frame_count += 1

            # ── 1. Detect (with optional frame skip) ──
            run_detection = (
                config.DETECTION_FRAME_SKIP <= 1 or
                self._frame_count % config.DETECTION_FRAME_SKIP == 0
            )

            if run_detection:
                detections = self.detector.detect(frame)
                self._cached_detections = detections
            else:
                detections = self._cached_detections

            # ── 2. Track persons ──────────────────
            if run_detection:
                tracked_persons = self.tracker.update(detections, frame)
                self._cached_tracked = tracked_persons
            else:
                tracked_persons = self._cached_tracked

            # ── 3. Analyse suspicious activity ───
            new_alerts = self.suspicious.analyse(detections, tracked_persons, self.tracker)
            if new_alerts:
                self._active_alerts = new_alerts
                self._alert_expire_time = time.time() + 5   # show for 5 s

            # Clear expired alerts
            if time.time() > self._alert_expire_time:
                self._active_alerts = []

            # ── 4. Periodic detection logging ────
            now = time.time()
            if detections and (now - last_detection_log >= detection_log_interval):
                for d in detections:
                    self.logger.log_detection(d.class_name, d.confidence)
                last_detection_log = now

            # ── 5. Draw annotations ──────────────
            self._annotate_frame(frame, detections, tracked_persons)

            # ── 6. Compute FPS (exponential moving average) ──
            current_time = time.time()
            raw_fps = 1.0 / max(current_time - prev_time, 1e-6)
            self._fps = self._fps * (1 - config.FPS_SMOOTH_ALPHA) + raw_fps * config.FPS_SMOOTH_ALPHA
            prev_time = current_time

            # ── 7. Draw HUD overlay ──────────────
            person_count = len(tracked_persons)
            tracked_ids = [tp.track_id for tp in tracked_persons if tp.track_id >= 0]

            draw_overlay(
                frame, self._fps, len(detections),
                person_count=person_count,
                alerts=self._active_alerts,
                tracked_ids=tracked_ids if tracked_ids else None,
            )

            # ── 8. Display ───────────────────────
            cv2.imshow(config.WINDOW_NAME, frame)

            # ── 9. Handle keyboard input ─────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[System] Quit requested.")
                break
            elif key == ord("s"):
                self._save_snapshot(frame)

    # ──────────────────────────────────────────
    # Annotation
    # ──────────────────────────────────────────
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracked_persons: list[TrackedPerson],
    ) -> None:
        """Draw bounding boxes for all detections and tracked persons."""
        # Build a set of person detection boxes that are covered by tracking
        tracked_boxes = set()
        if self.tracker.enabled:
            for tp in tracked_persons:
                draw_tracked_person(frame, tp)
                # Mark approximate region as tracked (to avoid double-drawing)
                tracked_boxes.add((tp.x1 // 10, tp.y1 // 10, tp.x2 // 10, tp.y2 // 10))

        # Draw non-person detections (or person detections if tracking is off)
        for det in detections:
            if det.class_name == "person" and self.tracker.enabled:
                # Skip — already drawn by tracker above
                continue
            draw_detection(frame, det)

    # ──────────────────────────────────────────
    # Snapshot
    # ──────────────────────────────────────────
    def _save_snapshot(self, frame: np.ndarray) -> None:
        """Save the current annotated frame as an image file."""
        import os
        snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(snapshot_dir, filename)
        cv2.imwrite(path, frame)
        self.logger.log_system(f"Snapshot saved: {path}")
        print(f"[System] Snapshot saved → {path}")
