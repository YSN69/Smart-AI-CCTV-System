"""
utils.py — Premium UI rendering for the Smart AI CCTV System.

Modern, professional visuals using OpenCV:
- Rounded corner bounding boxes with corner brackets
- Frosted glass HUD overlay
- Color-coded FPS indicator
- Pulsing alert banners
- Timestamp overlay
- Person tracking ID labels
"""

import time
from datetime import datetime

import cv2
import numpy as np

import config
from detector import Detection
from tracker import TrackedPerson


# ──────────────────────────────────────────────
# Colour palette — curated + fallback
# ──────────────────────────────────────────────
def _generate_colour_palette(n: int) -> list[tuple[int, int, int]]:
    """Return *n* visually distinct BGR colours using HSV spacing."""
    colours = []
    for i in range(n):
        hue = int(180 * i / n)           # OpenCV hue range is 0-179
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colours.append(tuple(int(c) for c in bgr))
    return colours


# Pre-compute fallback colours once
_FALLBACK_COLOURS = _generate_colour_palette(len(config.COCO_CLASSES))


def get_colour_for_class(class_name: str, class_id: int = 0) -> tuple[int, int, int]:
    """Return the BGR colour for a class. Uses curated palette first, fallback otherwise."""
    if class_name in config.CLASS_COLOURS:
        return config.CLASS_COLOURS[class_name]
    return _FALLBACK_COLOURS[class_id % len(_FALLBACK_COLOURS)]


# ──────────────────────────────────────────────
# Modern rounded bounding box with transparent fill
# ──────────────────────────────────────────────
def draw_rounded_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: tuple[int, int, int],
    thickness: int = 2,
    radius: int = 15,
) -> None:
    """Draw a modern rounded bounding box (border only)."""

    # ── Rounded border ──
    # Ensure radius isn't larger than box dimensions
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    
    if r <= 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
        return

    # Straight line segments
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), colour, thickness, cv2.LINE_AA) # Top
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), colour, thickness, cv2.LINE_AA) # Bottom
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), colour, thickness, cv2.LINE_AA) # Left
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), colour, thickness, cv2.LINE_AA) # Right

    # Corner arcs
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, colour, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, colour, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r),  90, 0, 90, colour, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r),   0, 0, 90, colour, thickness, cv2.LINE_AA)


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    colour: tuple[int, int, int],
    font_scale: float = 0.50,
) -> None:
    """Draw a pill-shaped label with semi-transparent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

    # Pill background
    pad_x, pad_y = 8, 5
    lx1 = x
    ly1 = y - th - 2 * pad_y
    lx2 = x + tw + 2 * pad_x
    ly2 = y

    # Semi-transparent rounded rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), colour, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Accent line on top
    cv2.line(frame, (lx1, ly1), (lx2, ly1), colour, 2, cv2.LINE_AA)

    # Text
    cv2.putText(
        frame, text,
        (x + pad_x, y - pad_y),
        font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )


# ──────────────────────────────────────────────
# Full bounding box drawing (box + label)
# ──────────────────────────────────────────────
def draw_detection(
    frame: np.ndarray,
    det: Detection,
    track_id: int = -1,
    dwell_time: float = 0.0,
) -> None:
    """Draw a complete detection: corner box + label + optional track info."""
    colour = get_colour_for_class(det.class_name, det.class_id)

    # Draw rounded transparent box
    draw_rounded_box(
        frame, det.x1, det.y1, det.x2, det.y2,
        colour, config.BOX_THICKNESS,
    )

    # Build label text
    label = f"{det.class_name.upper()} {det.confidence:.0%}"
    if track_id >= 0 and config.SHOW_TRACK_IDS:
        label = f"ID:{track_id} {label}"

    draw_label(frame, label, det.x1, det.y1, colour, config.LABEL_FONT_SCALE)

    # Draw dwell time for tracked persons
    if track_id >= 0 and dwell_time > 2.0:
        dwell_text = f"{dwell_time:.0f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(dwell_text, font, 0.45, 1)
        cx = (det.x1 + det.x2) // 2 - tw // 2
        cy = det.y2 + th + 8

        # Warning colour if loitering
        dwell_colour = (0, 200, 255) if dwell_time >= config.PERSON_LOITER_SECONDS else (200, 200, 200)
        cv2.putText(frame, dwell_text, (cx, cy), font, 0.45, dwell_colour, 1, cv2.LINE_AA)


def draw_tracked_person(frame: np.ndarray, tp: TrackedPerson) -> None:
    """Draw a tracked person with their tracking info and dwell-time progress bar."""
    colour = get_colour_for_class(tp.class_name, 0)

    draw_rounded_box(
        frame, tp.x1, tp.y1, tp.x2, tp.y2,
        colour, config.BOX_THICKNESS,
    )

    # Build label
    label = f"PERSON {tp.confidence:.0%}"
    if tp.track_id >= 0 and config.SHOW_TRACK_IDS:
        label = f"ID:{tp.track_id} {label}"

    draw_label(frame, label, tp.x1, tp.y1, colour, config.LABEL_FONT_SCALE)

    # Dwell time badge
    if tp.track_id >= 0 and tp.dwell_time > 2.0:
        dwell_text = f"{tp.dwell_time:.0f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(dwell_text, font, 0.45, 1)
        cx = (tp.x1 + tp.x2) // 2 - tw // 2
        cy = tp.y2 + th + 8
        dwell_colour = (0, 80, 255) if tp.dwell_time >= config.PERSON_LOITER_SECONDS else (200, 200, 200)
        cv2.putText(frame, dwell_text, (cx, cy), font, 0.45, dwell_colour, 1, cv2.LINE_AA)

    # ── Dwell-time progress bar ───────────────
    if tp.track_id >= 0 and tp.dwell_time > 5.0:
        _draw_dwell_progress_bar(frame, tp)


def _draw_dwell_progress_bar(
    frame: np.ndarray,
    tp: TrackedPerson,
) -> None:
    """Draw a thin progress bar under a tracked person's bounding box.

    Fills from green → yellow → red as dwell time approaches PERSON_LOITER_SECONDS_CRITICAL.
    """
    bar_width = tp.x2 - tp.x1
    bar_height = 4
    bar_x = tp.x1
    bar_y = tp.y2 + 22  # below the dwell-time text

    # Clamp progress to 0..1 relative to the CRITICAL threshold (300s)
    progress = min(tp.dwell_time / config.PERSON_LOITER_SECONDS_CRITICAL, 1.0)
    fill_width = int(bar_width * progress)

    # Colour gradient: green (0%) → yellow (50%) → red (100%)
    if progress < 0.5:
        # Green to Yellow
        t = progress * 2  # 0..1
        r = int(0 + t * 255)
        g = 255
    else:
        # Yellow to Red
        t = (progress - 0.5) * 2  # 0..1
        r = 255
        g = int(255 * (1 - t))
    bar_colour_bgr = (0, g, r)  # BGR

    # Background (dark)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (40, 40, 40), -1)
    # Filled portion
    if fill_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                      bar_colour_bgr, -1)
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (80, 80, 80), 1)


# ──────────────────────────────────────────────
# HUD Overlay
# ──────────────────────────────────────────────
def draw_overlay(
    frame: np.ndarray,
    fps: float,
    detection_count: int,
    person_count: int = 0,
    alerts: list | None = None,
    tracked_ids: list[int] | None = None,
) -> None:
    """Draw a premium frosted-glass HUD overlay on the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top status bar (frosted glass) ─────────
    bar_height = 48
    _draw_frosted_bar(frame, 0, 0, w, bar_height)

    x_cursor = 15

    # FPS with colour coding
    if config.SHOW_FPS:
        if fps >= 20:
            fps_colour = (80, 255, 80)     # Green
        elif fps >= 10:
            fps_colour = (0, 220, 255)     # Yellow
        else:
            fps_colour = (60, 60, 255)     # Red

        fps_text = f"FPS {fps:.0f}"
        cv2.putText(frame, fps_text, (x_cursor, 33), font, 0.60, fps_colour, 2, cv2.LINE_AA)
        x_cursor += 110

        # Separator dot
        cv2.circle(frame, (x_cursor, 25), 3, (100, 100, 100), -1)
        x_cursor += 20

    # Detection count
    if config.SHOW_DETECTION_COUNT:
        obj_text = f"Objects: {detection_count}"
        cv2.putText(frame, obj_text, (x_cursor, 33), font, 0.55, (200, 220, 240), 1, cv2.LINE_AA)
        x_cursor += 130

        cv2.circle(frame, (x_cursor, 25), 3, (100, 100, 100), -1)
        x_cursor += 20

    # Person count
    if config.SHOW_PERSON_COUNT:
        person_text = f"Persons: {person_count}"
        person_colour = (80, 255, 80) if person_count < config.CROWD_DENSITY_THRESHOLD else (60, 60, 255)
        cv2.putText(frame, person_text, (x_cursor, 33), font, 0.55, person_colour, 1, cv2.LINE_AA)
        x_cursor += 140

    # Timestamp on far right
    if config.SHOW_TIMESTAMP:
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, font, 0.48, 1)
        cv2.putText(frame, ts, (w - tw - 15, 20), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # System title - below timestamp
    title = config.WINDOW_NAME
    (tw, _), _ = cv2.getTextSize(title, font, 0.45, 1)
    cv2.putText(frame, title, (w - tw - 15, 40), font, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    # ── Recording indicator (pulsing red dot) ──
    pulse = int(abs(time.time() % 1.0 - 0.5) * 2 * 255)
    cv2.circle(frame, (w - tw - 30, 30), 5, (0, 0, max(100, pulse)), -1)

    # ── Bottom info bar ────────────────────────
    if tracked_ids:
        _draw_frosted_bar(frame, 0, h - 30, w, 30)
        ids_text = f"Tracking IDs: {', '.join(str(i) for i in sorted(tracked_ids)[:15])}"
        if len(tracked_ids) > 15:
            ids_text += f" +{len(tracked_ids) - 15} more"
        cv2.putText(frame, ids_text, (15, h - 8), font, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

    # ── Alert banners (bottom, pulsing) ────────
    if alerts:
        _draw_alerts(frame, alerts)

    # ── Threat level indicator ─────────────────
    if alerts:
        _draw_threat_level(frame, alerts)


def _draw_frosted_bar(
    frame: np.ndarray,
    x: int, y: int, w: int, bar_h: int,
) -> None:
    """Draw a frosted glass effect bar."""
    h_frame = frame.shape[0]

    # Clamp to frame bounds
    y_end = min(y + bar_h, h_frame)
    if y >= h_frame or y_end <= y:
        return

    # Extract the region
    roi = frame[y:y_end, x:x + w]
    if roi.size == 0:
        return

    # Blur the region for frosted glass effect
    blurred = cv2.GaussianBlur(roi, (15, 15), 10)

    # Dark tint
    tint = np.full_like(blurred, (25, 25, 30), dtype=np.uint8)
    frosted = cv2.addWeighted(blurred, 0.3, tint, 0.7, 0)

    frame[y:y_end, x:x + w] = frosted

    # Subtle bottom border line
    if y_end < h_frame:
        cv2.line(frame, (x, y_end - 1), (x + w, y_end - 1), (60, 60, 60), 1)


def _draw_alerts(frame: np.ndarray, alerts: list) -> None:
    """Draw pulsing alert banners at the bottom of the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Determine vertical position — above bottom bar if tracking IDs shown
    base_y = h - 40

    for i, alert in enumerate(reversed(alerts[-3:])):  # Show latest 3
        # Get alert text (support both string and Alert dataclass)
        if hasattr(alert, "message"):
            alert_text = alert.message
            severity = getattr(alert, "severity", "WARNING")
        else:
            alert_text = str(alert)
            severity = "WARNING"

        y_top = base_y - (i * 45) - 35
        y_bottom = base_y - (i * 45)

        if y_top < 50:
            break

        # Pulse effect
        pulse = abs(time.time() % 1.0 - 0.5) * 2
        alpha = 0.6 + pulse * 0.2

        # Colour based on severity — distinct visuals
        if severity == "CRITICAL":
            bar_colour = (0, 0, 220)      # Bright red
            border_colour = (0, 0, 255)   # Pure red border
            border_thickness = 2
            icon = "[!!] CRITICAL"
        else:
            bar_colour = (0, 120, 220)    # Amber/orange
            border_colour = (0, 160, 255) # Orange border
            border_thickness = 1
            icon = "[!] WARNING"

        # Draw alert bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_top), (w - 10, y_bottom), bar_colour, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Border
        cv2.rectangle(frame, (10, y_top), (w - 10, y_bottom),
                      border_colour, border_thickness, cv2.LINE_AA)

        # Alert text
        display_text = f"{icon}: {alert_text}"
        # Truncate if too long
        max_chars = w // 9
        if len(display_text) > max_chars:
            display_text = display_text[:max_chars - 3] + "..."

        cv2.putText(
            frame, display_text,
            (25, y_bottom - 10),
            font, 0.50, (255, 255, 255), 1, cv2.LINE_AA,
        )


def _draw_threat_level(frame: np.ndarray, alerts: list) -> None:
    """Draw a compact threat-level indicator in the top-right area of the HUD."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Determine highest severity
    max_severity = "WARNING"
    for alert in alerts:
        if hasattr(alert, "severity") and alert.severity == "CRITICAL":
            max_severity = "CRITICAL"
            break

    if max_severity == "CRITICAL":
        label = "THREAT: HIGH"
        colour = (0, 0, 255)     # Red
    else:
        label = "THREAT: LOW"
        colour = (0, 180, 255)   # Orange

    # Position: right side, below the top bar
    (tw, th), _ = cv2.getTextSize(label, font, 0.50, 2)
    x = w - tw - 20
    y = 65

    # Pulsing background pill
    pulse = abs(time.time() % 0.8 - 0.4) * 2.5
    alpha = 0.5 + pulse * 0.3
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, label, (x, y), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
