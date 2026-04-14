# 🎯 Smart AI CCTV System v2.0

Real-time multi-class object detection, person tracking, and intelligent suspicious activity surveillance — built with Python, OpenCV, YOLOv8, and DeepSORT.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)
![DeepSORT](https://img.shields.io/badge/DeepSORT-Tracking-purple)

---

## ✨ Features

### Detection & Tracking
- **Multi-class object detection** — YOLOv8 with smart class filtering (person, car, bike, bag, etc.)
- **Person tracking** — DeepSORT-based persistent ID assignment across frames
- **Dwell-time monitoring** — tracks how long each person has been in frame
- **Temporal smoothing** — eliminates flickering false positives
- **Per-class confidence** — higher thresholds for noisy classes (bags, etc.)
- **Min-area filtering** — rejects tiny false-positive boxes

### Suspicious Activity Intelligence
- **Loitering detection** — alerts when a tracked person stays too long (by ID)
- **Sudden crowd surge** — alerts on rapid person-count spikes
- **Repeated motion** — flags persons who appear-disappear-reappear suspiciously
- **Crowd density** — alerts when too many persons are in frame

### Premium UI
- **Corner-bracket bounding boxes** — modern security-cam aesthetic
- **Frosted glass HUD** — blurred, tinted status bar overlay
- **Color-coded FPS** — green/yellow/red based on performance
- **Real-time timestamp** — live date and time overlay
- **Person ID labels** — unique ID + confidence on each tracked person
- **Dwell-time badges** — shows seconds elapsed, turns red when loitering
- **Pulsing alert banners** — animated red/orange banners with severity icons
- **Curated color palette** — hand-picked vibrant colors for priority classes

### Infrastructure
- **Structured JSON logging** — thread-safe, rotating event log
- **Snapshot capture** — save annotated frames on demand
- **CLI overrides** — change any setting without editing code
- **Modular architecture** — clean separation across 8 focused modules
- **Frame-skip mode** — skip detection frames for performance boost

---

## 📁 Project Structure

```
smart_cctv_system/
├── main.py                  # Entry point — CLI args, launches system
├── cctv_system.py           # Core orchestrator — capture → detect → track → draw
├── detector.py              # YOLOv8 wrapper — inference, filtering, smoothing
├── tracker.py               # DeepSORT person tracker — IDs, dwell time
├── suspicious_activity.py   # 4-rule suspicious activity analyser
├── logger.py                # JSON event logger (thread-safe)
├── config.py                # All tuneable settings in one place
├── utils.py                 # Premium UI rendering — HUD, boxes, alerts
├── requirements.txt         # Python dependencies
├── logs/                    # Auto-generated event logs (JSON)
│   └── events.json
├── snapshots/               # Auto-generated saved frames
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+** installed
- A webcam (or a `.mp4` video file for testing)

### 2. Install dependencies

```bash
cd smart_cctv_system
pip install -r requirements.txt
```

> The first run will also auto-download the YOLOv8 model weights (~6 MB for `yolov8n`).

### 3. Run the system

```bash
# Default: webcam with tracking
python main.py

# Use a specific webcam
python main.py --source 1

# Use a video file
python main.py --source path/to/video.mp4

# Use a larger model for better accuracy
python main.py --model yolov8s.pt

# Customize confidence threshold
python main.py --conf 0.6

# Disable person tracking
python main.py --no-tracking

# Only detect specific classes
python main.py --classes person car bicycle

# Boost FPS by running detection every 2nd frame
python main.py --frame-skip 2
```

### 4. Controls

| Key | Action |
|-----|--------|
| `q` | Quit the system |
| `s` | Save a snapshot of the current frame |

---

## ⚙️ Configuration

All settings are in **`config.py`**:

### Detection
| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_SOURCE` | `0` | Webcam index or video file path |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `TRACKED_CLASSES` | person, car, bike, bag… | Classes to detect (None = all 80) |
| `MIN_BOX_AREA` | `2000` | Min bounding box area (pixels²) |
| `DETECTION_PERSISTENCE_FRAMES` | `2` | Frames before showing detection |

### Tracking
| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_TRACKING` | `True` | Enable/disable DeepSORT tracking |
| `TRACKER_MAX_AGE` | `30` | Frames before deleting lost tracks |
| `TRACKER_N_INIT` | `3` | Frames to confirm a new track |

### Suspicious Activity
| Setting | Default | Description |
|---------|---------|-------------|
| `PERSON_LOITER_SECONDS` | `15` | Seconds before loitering alert |
| `SUDDEN_CROWD_THRESHOLD` | `5` | Person-count jump to flag |
| `CROWD_DENSITY_THRESHOLD` | `10` | Max persons before alert |
| `REPEATED_MOTION_COUNT` | `3` | Re-appearances to flag |
| `SUSPICIOUS_LOG_COOLDOWN` | `30` | Seconds between repeated alerts |

### Performance
| Setting | Default | Description |
|---------|---------|-------------|
| `DETECTION_FRAME_SKIP` | `1` | Process every Nth frame (1=all) |
| `FRAME_WIDTH` | `1280` | Capture width |
| `FRAME_HEIGHT` | `720` | Capture height |

---

## 🔍 Suspicious Activity Rules

| Rule | Trigger | Severity |
|------|---------|----------|
| **Loitering** | Tracked person stays > 15s | ⚠️ WARNING |
| **Sudden Crowd** | Person count jumps by ≥ 5 | 🔴 CRITICAL |
| **Repeated Motion** | Same person reappears ≥ 3 times in 60s | ⚠️ WARNING |
| **Crowd Density** | ≥ 10 persons in frame at once | 🔴 CRITICAL |

---

## 📋 Event Log Format

Logs are saved to `logs/events.json` as a JSON array:

```json
[
  {
    "timestamp": "2025-01-15T14:23:05",
    "event_type": "detection",
    "object": "person",
    "confidence": 0.87
  },
  {
    "timestamp": "2025-01-15T14:23:21",
    "event_type": "suspicious_activity",
    "rule": "loitering",
    "details": "Person #3 loitering for 16s (threshold 15s)",
    "track_id": 3
  }
]
```

---

## 🏗️ Architecture

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌─────────────────────┐
│  Camera   │───▶│  Detector  │───▶│  Tracker   │───▶│  Suspicious Activity│
│  (OpenCV) │    │  (YOLOv8)  │    │ (DeepSORT) │    │  Analyser           │
└──────────┘    └────────────┘    └────────────┘    └──────────┬──────────┘
                      │                 │                       │
                      ▼                 ▼                       ▼
               ┌────────────┐                           ┌────────────┐
               │   Utils    │                           │   Logger   │
               │ (Premium   │                           │   (JSON)   │
               │    UI)     │                           └────────────┘
               └────────────┘
```

---

## 🚀 Performance Tips

1. **Use `yolov8n.pt`** (nano) for maximum FPS — it's the default
2. **Frame skip**: `python main.py --frame-skip 2` runs detection every 2nd frame
3. **Reduce resolution**: Lower `FRAME_WIDTH`/`FRAME_HEIGHT` in config.py
4. **Disable tracking**: `python main.py --no-tracking` saves ~5-10 FPS
5. **GPU acceleration**: Use `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` for CUDA

---

## 🎓 Fine-Tuning YOLOv8 (Advanced)

To improve accuracy for your specific use case:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolov8n.pt")

# Train on your custom dataset
results = model.train(
    data="your_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)

# Export and use the fine-tuned model
# python main.py --model runs/detect/train/weights/best.pt
```

Your dataset YAML should follow the [Ultralytics format](https://docs.ultralytics.com/datasets/).

---

## 📝 License

This project is open-source and available under the MIT License.
