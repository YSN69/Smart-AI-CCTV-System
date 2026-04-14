"""
suspicious_activity.py — Intelligent rule-based suspicious activity detector.

Implements four rules using tracker data:
1. Loitering — a person (by track ID) stays too long
2. Sudden Crowd — person count spikes between frames
3. Repeated Motion — same person ID disappears & reappears repeatedly
4. Crowd Density — too many persons in frame at once

Easy to extend: add a new `_check_*` method and call it from `analyse()`.
"""

import time
from dataclasses import dataclass

import config
from detector import Detection
from logger import EventLogger
from tracker import TrackedPerson, PersonTracker


@dataclass
class Alert:
    """A structured alert with severity."""
    rule: str
    message: str
    severity: str  # "WARNING" or "CRITICAL"


class SuspiciousActivityDetector:
    """Stateful analyser that tracks detections across frames."""

    def __init__(self, logger: EventLogger):
        self.logger = logger

        # ── Sudden crowd state ──────────────────
        self._prev_person_count: int = 0

        # ── Cooldown tracking ───────────────────
        self._last_alert_time: dict[str, float] = {}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────
    def analyse(
        self,
        detections: list[Detection],
        tracked_persons: list[TrackedPerson],
        tracker: PersonTracker,
    ) -> list[Alert]:
        """
        Run all suspicious-activity rules against the current frame.

        Returns a list of Alert objects (empty means nothing suspicious).
        """
        alerts: list[Alert] = []

        alerts += self._check_loitering(tracked_persons)
        alerts += self._check_sudden_crowd(tracked_persons)
        alerts += self._check_repeated_motion(tracker)
        alerts += self._check_crowd_density(tracked_persons)

        # Update rolling state for next frame
        self._prev_person_count = len(tracked_persons)

        return alerts

    # ──────────────────────────────────────────
    # Rule implementations
    # ──────────────────────────────────────────
    def _check_loitering(self, tracked_persons: list[TrackedPerson]) -> list[Alert]:
        """Flag if any tracked person's dwell time exceeds the thresholds.

        Two-tier system:
          - WARNING  at PERSON_LOITER_SECONDS (default 60s)
          - CRITICAL at PERSON_LOITER_SECONDS_CRITICAL (default 300s — the "5-minute rule")
        """
        alerts: list[Alert] = []
        for tp in tracked_persons:
            if tp.track_id < 0:
                continue  # untracked

            # ── CRITICAL tier: 300-second rule ────
            if tp.dwell_time >= config.PERSON_LOITER_SECONDS_CRITICAL:
                rule_key = f"loitering_critical_{tp.track_id}"
                if self._can_alert(rule_key):
                    msg = (f"SUSPICIOUS — Person #{tp.track_id} in frame for "
                           f"{tp.dwell_time:.0f}s (>{config.PERSON_LOITER_SECONDS_CRITICAL}s)")
                    self.logger.log_suspicious("loitering_critical", msg, {"track_id": tp.track_id})
                    self.logger.log_tracking(tp.track_id, tp.dwell_time, "loitering_critical")
                    alerts.append(Alert(rule="loitering", message=msg, severity="CRITICAL"))

            # ── WARNING tier: early awareness ─────
            elif tp.dwell_time >= config.PERSON_LOITER_SECONDS:
                rule_key = f"loitering_{tp.track_id}"
                if self._can_alert(rule_key):
                    msg = (f"Person #{tp.track_id} lingering for "
                           f"{tp.dwell_time:.0f}s (threshold {config.PERSON_LOITER_SECONDS}s)")
                    self.logger.log_suspicious("loitering", msg, {"track_id": tp.track_id})
                    self.logger.log_tracking(tp.track_id, tp.dwell_time, "loitering_warning")
                    alerts.append(Alert(rule="loitering", message=msg, severity="WARNING"))

        return alerts

    def _check_sudden_crowd(self, tracked_persons: list[TrackedPerson]) -> list[Alert]:
        """Flag if the person count jumps dramatically between frames."""
        current_count = len(tracked_persons)
        delta = current_count - self._prev_person_count

        if delta >= config.SUDDEN_CROWD_THRESHOLD:
            if self._can_alert("sudden_crowd"):
                msg = (f"Person count surged: {self._prev_person_count} → {current_count} "
                       f"(+{delta}, threshold {config.SUDDEN_CROWD_THRESHOLD})")
                self.logger.log_suspicious("sudden_crowd", msg)
                return [Alert(rule="sudden_crowd", message=msg, severity="CRITICAL")]
        return []

    def _check_repeated_motion(self, tracker: PersonTracker) -> list[Alert]:
        """Flag if a person ID has appeared-disappeared-reappeared too many times."""
        if not tracker.enabled:
            return []

        alerts: list[Alert] = []
        now = time.time()
        window = config.REPEATED_MOTION_WINDOW_SECONDS

        for tid, hist in tracker.get_all_histories().items():
            # Count appearances within the time window
            recent = [t for t in hist.appearance_times if now - t <= window]
            if len(recent) >= config.REPEATED_MOTION_COUNT:
                rule_key = f"repeated_motion_{tid}"
                if self._can_alert(rule_key):
                    msg = (f"Person #{tid} appeared {len(recent)} times in "
                           f"{window}s (threshold {config.REPEATED_MOTION_COUNT})")
                    self.logger.log_suspicious("repeated_motion", msg, {"track_id": tid})
                    alerts.append(Alert(rule="repeated_motion", message=msg, severity="WARNING"))
        return alerts

    def _check_crowd_density(self, tracked_persons: list[TrackedPerson]) -> list[Alert]:
        """Flag if too many persons are in frame simultaneously."""
        count = len(tracked_persons)
        if count >= config.CROWD_DENSITY_THRESHOLD:
            if self._can_alert("crowd_density"):
                msg = (f"High crowd density: {count} persons detected "
                       f"(threshold {config.CROWD_DENSITY_THRESHOLD})")
                self.logger.log_suspicious("crowd_density", msg, {"person_count": count})
                return [Alert(rule="crowd_density", message=msg, severity="CRITICAL")]
        return []

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────
    def _can_alert(self, rule_name: str) -> bool:
        """Enforce per-rule cooldown to prevent log spam."""
        now = time.time()
        last = self._last_alert_time.get(rule_name, 0.0)
        if now - last >= config.SUSPICIOUS_LOG_COOLDOWN:
            self._last_alert_time[rule_name] = now
            return True
        return False
