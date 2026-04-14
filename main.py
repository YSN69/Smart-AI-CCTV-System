"""
main.py — Entry point for the Smart AI CCTV System.

Usage:
    python main.py                     # use defaults from config.py
    python main.py --source 1          # use webcam index 1
    python main.py --source video.mp4  # use a video file
    python main.py --model yolov8s.pt  # use a different YOLO model
    python main.py --conf 0.5          # set confidence threshold
    python main.py --no-tracking       # disable person tracking
    python main.py --classes person car # only detect specific classes
"""

import argparse
import sys

import config
from cctv_system import CCTVSystem


def parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides for config values."""
    parser = argparse.ArgumentParser(
        description="Smart AI CCTV System — real-time multi-class object detection with suspicious activity alerts.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index (integer) or path to a video file. Overrides config.CAMERA_SOURCE.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLOv8 model name or .pt path. Overrides config.YOLO_MODEL.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (0-1). Overrides config.CONFIDENCE_THRESHOLD.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable person tracking (DeepSORT).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of class names to detect. Overrides config.TRACKED_CLASSES.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Run detection every N frames for better performance. Overrides config.DETECTION_FRAME_SKIP.",
    )
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    """Apply CLI arguments to the global config module."""
    if args.source is not None:
        # If it looks like an integer, treat it as a camera index
        try:
            config.CAMERA_SOURCE = int(args.source)
        except ValueError:
            config.CAMERA_SOURCE = args.source

    if args.model is not None:
        config.YOLO_MODEL = args.model

    if args.conf is not None:
        config.CONFIDENCE_THRESHOLD = args.conf

    if args.no_tracking:
        config.ENABLE_TRACKING = False

    if args.classes is not None:
        config.TRACKED_CLASSES = args.classes

    if args.frame_skip is not None:
        config.DETECTION_FRAME_SKIP = max(1, args.frame_skip)


def main() -> None:
    """Initialise and run the CCTV system."""
    args = parse_args()
    apply_overrides(args)

    tracking_status = "ON" if config.ENABLE_TRACKING else "OFF"
    classes_display = ", ".join(config.TRACKED_CLASSES) if config.TRACKED_CLASSES else "ALL (80 classes)"

    print("┌─────────────────────────────────────────────────────┐")
    print("│          Smart AI CCTV System v2.0                  │")
    print("│   Real-time Detection • Tracking • Surveillance     │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print(f"  Source     : {config.CAMERA_SOURCE}")
    print(f"  Model      : {config.YOLO_MODEL}")
    print(f"  Confidence : {config.CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Tracking   : {tracking_status}")
    print(f"  Classes    : {classes_display}")
    print(f"  Frame skip : {config.DETECTION_FRAME_SKIP}")
    print(f"  Log file   : {config.LOG_FILE}")
    print()

    system = CCTVSystem()
    system.start()


if __name__ == "__main__":
    main()
