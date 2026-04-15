"""
config.py — Central configuration for the Smart AI CCTV System.

All tuneable settings live here so you never need to dig into other modules.
"""

import os

# Fix OpenMP duplicate library crash (common with Anaconda + PyTorch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────
# Camera / Video Source
# ──────────────────────────────────────────────
# Use 0 for default webcam, or provide a path to a video file.
# Examples:
#   CAMERA_SOURCE = 0                       # webcam
#   CAMERA_SOURCE = "sample_videos/demo.mp4" # local video file
CAMERA_SOURCE = "test.mp4"

# Desired capture resolution (width, height). Set to None for camera default.
# Lower = faster. 640×480 is ideal for CPU-only; use 1280×720 if you have a GPU.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Frames per second cap (0 = unlimited, uses camera native FPS)
FPS_CAP = 30

# ──────────────────────────────────────────────
# YOLO Model
# ──────────────────────────────────────────────
# Ultralytics YOLOv8 model name or path to a custom .pt file.
# Standard models: "yolov8n.pt" (fastest) or "yolov8s.pt" (more accurate)
# Detects 80 COCO classes (person, car, laptop, cell phone, etc.)
YOLO_MODEL = "yolov8m.pt"

# Minimum confidence threshold for a detection to be drawn / logged.
CONFIDENCE_THRESHOLD = 0.40

# IoU (Intersection over Union) threshold for Non-Max Suppression.
# Lower = fewer overlapping boxes. 0.50 is the sweet spot.
# YOLO default is 0.7 which is too permissive for surveillance.
IOU_THRESHOLD = 0.50

# Enable half-precision (FP16) inference for ~2x GPU speedup.
# MUST be False on CPU-only machines (FP16 requires CUDA GPU).
HALF_PRECISION = True

# Inference input resolution. YOLO resizes frames internally to this.
# 640 = default accuracy. 416 = faster on CPU. 320 = fastest but less accurate.
IMGSZ = 640

# ──────────────────────────────────────────────
# Class Filtering & Detection Accuracy
# ──────────────────────────────────────────────
# Only detect these classes. Set to None to detect ALL 80 COCO classes.
# (Includes person, car, bike, laptop, cell phone, backpack, bottle, chair, etc.)
TRACKED_CLASSES = None

# Per-class confidence overrides (higher threshold for noisy classes).
# Classes not listed here use the global CONFIDENCE_THRESHOLD.
CONFIDENCE_PER_CLASS = {
    "backpack": 0.55,
    "handbag": 0.55,
    "suitcase": 0.55,
}

# Minimum bounding box area (in pixels²) to accept a detection.
# Rejects tiny false-positive boxes.
MIN_BOX_AREA = 2500

# Temporal smoothing: require detection to persist for N consecutive frames
# before showing it. Higher = more stable boxes, less flicker. Set to 1 to disable.
DETECTION_PERSISTENCE_FRAMES = 3

# ──────────────────────────────────────────────
# Person Tracking (DeepSORT)
# ──────────────────────────────────────────────
ENABLE_TRACKING = True

# Maximum frames a track can be "lost" before it is deleted.
TRACKER_MAX_AGE = 30

# Minimum consecutive detections to confirm a new track.
TRACKER_N_INIT = 3

# Maximum cosine distance for re-identification.
TRACKER_MAX_COSINE_DIST = 0.3

# ──────────────────────────────────────────────
# Suspicious Activity Detection
# ──────────────────────────────────────────────
# If a *person* is detected continuously for this many seconds → WARNING alert.
PERSON_LOITER_SECONDS = 60

# If a person stays for THIS many seconds → CRITICAL alert (the "300-second rule").
PERSON_LOITER_SECONDS_CRITICAL = 300

# If the total person count jumps by this amount between frames → suspicious.
SUDDEN_CROWD_THRESHOLD = 5

# Maximum allowed persons in frame before crowding alert.
CROWD_DENSITY_THRESHOLD = 10

# Repeated motion: if a person ID disappears & reappears this many times
# within the time window → suspicious.
REPEATED_MOTION_COUNT = 3
REPEATED_MOTION_WINDOW_SECONDS = 60

# Cooldown (seconds) between repeated suspicious‑activity logs for the same
# rule so the log file doesn't explode.
SUSPICIOUS_LOG_COOLDOWN = 30

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "events.json")

# Maximum number of events to keep in the JSON log (oldest trimmed first).
MAX_LOG_ENTRIES = 10_000

# Batch flushing: reduce disk I/O by buffering writes.
# Flush to disk every N events OR every T seconds (whichever comes first).
LOG_FLUSH_INTERVAL = 10       # events
LOG_FLUSH_SECONDS = 5.0       # seconds

# ──────────────────────────────────────────────
# Display / UI
# ──────────────────────────────────────────────
WINDOW_NAME = "Smart AI CCTV System"

# Show FPS overlay on the video feed
SHOW_FPS = True

# Show detection count overlay
SHOW_DETECTION_COUNT = True

# Show person count overlay
SHOW_PERSON_COUNT = True

# Show timestamp on the video feed
SHOW_TIMESTAMP = True

# Show person tracking IDs on bounding boxes
SHOW_TRACK_IDS = True

# Bounding‑box line thickness (pixels)
BOX_THICKNESS = 2

# Corner bracket length (pixels) for the modern bounding box style
CORNER_LENGTH = 20

# Font scale for labels
LABEL_FONT_SCALE = 0.55

# ──────────────────────────────────────────────
# Curated Colour Palette (BGR) — priority classes
# ──────────────────────────────────────────────
# Hand-picked vibrant colours for the most important classes.
# Other classes fall back to auto-generated HSV colours.
CLASS_COLOURS = {
    "person":     (230, 180, 30),    # Cyan-ish blue
    "car":        (40, 160, 255),    # Warm orange
    "bicycle":    (80, 255, 80),     # Lime green
    "motorcycle": (255, 140, 40),    # Deep blue
    "bus":        (60, 200, 235),    # Gold
    "truck":      (180, 105, 255),   # Pink-violet
    "backpack":   (255, 50, 200),    # Magenta
    "handbag":    (100, 220, 255),   # Light orange
    "suitcase":   (255, 200, 100),   # Light blue
    "laptop":     (200, 100, 255),   # Purple
    "cell phone": (100, 255, 200),   # Mint
    "bottle":     (255, 150, 150),   # Light blue 
    "chair":      (150, 150, 150),   # Grey
}

# ──────────────────────────────────────────────
# Performance
# ──────────────────────────────────────────────
# Process detection every N frames (1 = every frame, 2 = every other, etc.)
# Intermediate frames reuse the previous detection results.
# Set to 2 for a near-2x FPS boost with minimal accuracy loss.
# Set to 3 on CPU-only machines for smoother performance.
DETECTION_FRAME_SKIP = 2

# FPS display smoothing factor (exponential moving average).
# Lower = smoother display, higher = more responsive. Range: 0.01 – 1.0
FPS_SMOOTH_ALPHA = 0.1

# ──────────────────────────────────────────────
# COCO class names (80 classes) — used by YOLOv8 pretrained on COCO
# ──────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ──────────────────────────────────────────────
# Debugging & Optimization Tips
# ──────────────────────────────────────────────
# Common lag causes:
#   1. DETECTION_FRAME_SKIP = 1 → set to 2 or 3 for real-time feeds
#   2. Full-res inference → lower IMGSZ (e.g. 416) for speed
#   3. Flushing logs every frame → use LOG_FLUSH_INTERVAL
#   4. No GPU → set HALF_PRECISION = False on CPU-only machines
#
# OpenCV pitfalls:
#   - cv2.waitKey(1) is REQUIRED in the loop, waitKey(0) will freeze
#   - Always release VideoCapture in a finally block
#   - Use cv2.CAP_DSHOW on Windows if camera init is slow
#
# GPU check:
#   import torch; print(torch.cuda.is_available())
#   If True, HALF_PRECISION = True gives ~2x speedup
#
# Memory leaks:
#   - Always call cap.release() and cv2.destroyAllWindows()
#   - Avoid accumulating frame copies — use in-place drawing
