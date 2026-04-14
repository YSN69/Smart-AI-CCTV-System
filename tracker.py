"""
tracker.py — Person tracking with DeepSORT for persistent ID assignment.

Provides unique IDs across frames, dwell-time tracking, and re-appearance
counting for the suspicious-activity analyser.

Falls back gracefully to simple pass-through if deep-sort-realtime is not
installed.
"""

import time
from dataclasses import dataclass, field

import numpy as np

import config
from detector import Detection

# Try to import DeepSORT — graceful fallback if not available
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("[Tracker] WARNING: deep-sort-realtime not installed. "
          "Tracking disabled — install with: pip install deep-sort-realtime")


@dataclass
class TrackedPerson:
    """A tracked person with persistent identity."""
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float
    dwell_time: float         # seconds since this ID was first seen
    is_confirmed: bool        # has the track been confirmed (past n_init)?


@dataclass
class _TrackHistory:
    """Internal state for a single track ID."""
    first_seen: float = 0.0
    last_seen: float = 0.0
    appearance_count: int = 0
    appearance_times: list[float] = field(default_factory=list)
    was_lost: bool = False


class PersonTracker:
    """
    Wraps DeepSORT to provide:
    - Unique ID assignment across frames
    - Dwell-time (how long each ID has been visible)
    - Appearance counting (how many times an ID has appeared/re-appeared)
    """

    def __init__(self):
        self._histories: dict[int, _TrackHistory] = {}
        self._total_unique_persons = 0

        if DEEPSORT_AVAILABLE and config.ENABLE_TRACKING:
            self._tracker = DeepSort(
                max_age=config.TRACKER_MAX_AGE,
                n_init=config.TRACKER_N_INIT,
                max_cosine_distance=config.TRACKER_MAX_COSINE_DIST,
            )
            self._enabled = True
            print("[Tracker] DeepSORT initialised "
                  f"(max_age={config.TRACKER_MAX_AGE}, "
                  f"n_init={config.TRACKER_N_INIT})")
        else:
            self._tracker = None
            self._enabled = False
            if config.ENABLE_TRACKING and not DEEPSORT_AVAILABLE:
                print("[Tracker] Tracking requested but DeepSORT unavailable. "
                      "Running without tracking.")
            else:
                print("[Tracker] Tracking disabled by config.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def total_unique_persons(self) -> int:
        return self._total_unique_persons

    def get_history(self, track_id: int) -> _TrackHistory | None:
        """Return the history for a given track ID, or None."""
        return self._histories.get(track_id)

    def get_all_histories(self) -> dict[int, _TrackHistory]:
        """Return all track histories (for suspicious-activity analysis)."""
        return self._histories

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[TrackedPerson]:
        """
        Update the tracker with new detections and return tracked persons.

        If tracking is disabled, returns TrackedPerson objects with track_id=-1
        and dwell_time=0 (simple pass-through).
        """
        # Filter to persons only for tracking
        person_dets = [d for d in detections if d.class_name == "person"]

        if not self._enabled or not person_dets:
            # Pass-through mode: wrap detections as untracked
            return [
                TrackedPerson(
                    track_id=-1,
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    dwell_time=0.0,
                    is_confirmed=True,
                )
                for d in person_dets
            ]

        # Prepare detections for DeepSORT: list of ([x, y, w, h], conf, class)
        raw_dets = []
        for d in person_dets:
            w = d.x2 - d.x1
            h = d.y2 - d.y1
            raw_dets.append(([d.x1, d.y1, w, h], d.confidence, d.class_name))

        # Update DeepSORT
        tracks = self._tracker.update_tracks(raw_dets, frame=frame)

        now = time.time()
        tracked: list[TrackedPerson] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            # DeepSORT may return str or int IDs — normalise to int
            try:
                tid = int(track.track_id)
            except (ValueError, TypeError):
                tid = hash(track.track_id) % 100000
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Update history
            if tid not in self._histories:
                self._histories[tid] = _TrackHistory(
                    first_seen=now,
                    last_seen=now,
                    appearance_count=1,
                    appearance_times=[now],
                )
                self._total_unique_persons += 1
            else:
                hist = self._histories[tid]
                # Check if this is a re-appearance (was lost for >1s)
                if now - hist.last_seen > 1.0:
                    hist.appearance_count += 1
                    hist.appearance_times.append(now)
                    hist.was_lost = True
                hist.last_seen = now

            hist = self._histories[tid]
            dwell = now - hist.first_seen

            # Get confidence from the detection data if available
            det_class = track.det_class if track.det_class else "person"
            det_conf = track.det_conf if track.det_conf else 0.0

            tracked.append(TrackedPerson(
                track_id=tid,
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_name=det_class,
                confidence=det_conf,
                dwell_time=dwell,
                is_confirmed=True,
            ))

        return tracked
