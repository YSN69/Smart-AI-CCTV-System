# 🛡️ Smart AI CCTV System (SaaS Ready)

**Enterprise-Grade Video Surveillance & Analytics Engine**

A highly scalable, GPU-accelerated computer vision system designed to be the foundational backbone for a **Surveillance-as-a-Service (SaaS)** platform. Powered by YOLOv8 and DeepSORT, this system provides real-time object detection, persistent person tracking, and proactive suspicious activity alerts.

## 🚀 Key SaaS Capabilities

*   **⚡ Real-Time Pipeline**: Built on PyTorch and OpenCV, optimized for high framerate, low-latency edge computing using Half-Precision (FP16) inference and Exponential Moving Average smoothing.
*   **👁️ Universal Detection**: Natively recognizes 80 COCO classes out of the box (People, Vehicles, Luggage, Laptops, Mobile Phones, etc.) using `yolov8n.pt` or `yolov8s.pt`.
*   **🕵️ Persistent Tracking**: DeepSORT integration enables cross-frame person tracking and accurate dwell-time analysis without confusing overlapping subjects.
*   **🚨 Smart Threat Rules**: Rule-engine evaluates suspicious behaviors in real time:
    *   **The 300-Second Rule**: Actively tracks individuals and fires two-tier warnings (60s early warning, 300s Critical Threat) for loitering.
    *   **Crowd Density**: Monitors frame-over-frame crowd sudden surges and total density violations.
    *   **Repeated Motion**: Flags repetitive disappear/reappear behavior.
*   **📸 Beautiful HUD UI**: Premium UI featuring "frosted glass" data overlays, modern corner-bracket bounding boxes, progress bars, and threat-level badges (suitable for native apps and demonstrations).
*   **📊 Microservice-Ready Logging**: Features an asynchronous, lock-protected Batch Logger that dumps structured JSON events (reducing I/O by 10x). Perfect for ingestion into cloud dashboards, Kafka streams, or webhook alerting services.

## 🛠️ Technology Stack 
* **Core Language:** Python 3.13 
* **Computer Vision:** OpenCV
* **Artificial Intelligence Engine:** Ultralytics (YOLOv8) + PyTorch (CUDA 12.4 enabled)
* **Object Tracking:** DeepSORT
* **Data Flow / Persistence:** Batch JSON I/O

## ⚙️ Quick Start for Local Testing

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Smart-AI-CCTV-System.git
   cd Smart-AI-CCTV-System
   ```
2. **Setup your environment:**
   *(Ensure you have a CUDA-enabled GPU and the correct PyTorch wheels installed)*
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the core engine:**
   ```bash
   python main.py
   ```

## 🏗️ SaaS Architecture Roadmap

This codebase is specifically engineered to be scalable. The frontend UI overlays (`utils.py`) can be decoupled, allowing the core analysis loop to act as a headless edge-node microservice:

1.  **Edge Nodes (Clients):** Run `cctv_system.py` strictly for inference and tracking directly on the client's local camera subnet.
2.  **Telemetry (Cloud):** Refactor the `EventLogger` to push JSON events and critical threat alerts via WebSocket or HTTP POST to an AWS/GCP backend API.
3.  **Command Center (Dashboard):** Stream annotated frames via WebRTC or HLS to a central Next.js/React dashboard for human security operators.

## 🛡️ Centralized Configuration 
All business logic constraints are strictly decoupled from the code. Loitering thresholds, crowd limits, box styles, confidence thresholds, and logging rates are all adjustable in `config.py`.

---
*Developed by Yuvraj — Designed as the foundation for modern surveillance ecosystems.*
