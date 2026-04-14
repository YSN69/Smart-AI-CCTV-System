"""
logger.py — Event logging system for the Smart AI CCTV System.

Writes structured JSON logs with timestamps to a persistent file.
Thread‑safe so it can be extended to async pipelines later.
Uses batch flushing to reduce disk I/O overhead.
"""

import json
import os
import time
import threading
from datetime import datetime

import config


class EventLogger:
    """Append‑only JSON‑lines logger with automatic log rotation."""

    def __init__(self, log_file: str = config.LOG_FILE, max_entries: int = config.MAX_LOG_ENTRIES):
        self.log_file = log_file
        self.max_entries = max_entries
        self._lock = threading.Lock()

        # Batch flushing state
        self._flush_counter = 0
        self._last_flush_time = time.time()

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Load existing entries (or start fresh)
        self._entries: list[dict] = self._load()
        print(f"[Logger] Initialised — {len(self._entries)} existing entries in {self.log_file}")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────
    def log_detection(self, object_name: str, confidence: float, extra: dict | None = None) -> None:
        """Log a single detected object."""
        entry = {
            "timestamp": self._now(),
            "event_type": "detection",
            "object": object_name,
            "confidence": round(confidence, 3),
        }
        if extra:
            entry.update(extra)
        self._append(entry)

    def log_suspicious(self, rule: str, details: str, extra: dict | None = None) -> None:
        """Log a suspicious‑activity event."""
        entry = {
            "timestamp": self._now(),
            "event_type": "suspicious_activity",
            "rule": rule,
            "details": details,
        }
        if extra:
            entry.update(extra)
        self._append(entry)
        print(f"[ALERT] {rule}: {details}")

    def log_system(self, message: str) -> None:
        """Log a system lifecycle event (start, stop, error)."""
        entry = {
            "timestamp": self._now(),
            "event_type": "system",
            "message": message,
        }
        self._append(entry)

    def log_tracking(self, track_id: int, dwell_time: float, event: str = "tracking") -> None:
        """Log a person tracking event with dwell duration."""
        entry = {
            "timestamp": self._now(),
            "event_type": "tracking",
            "track_id": track_id,
            "dwell_time_seconds": round(dwell_time, 1),
            "event": event,
        }
        self._append(entry)

    def get_recent(self, n: int = 20) -> list[dict]:
        """Return the *n* most recent log entries (newest last)."""
        with self._lock:
            return list(self._entries[-n:])

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────
    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _append(self, entry: dict) -> None:
        with self._lock:
            self._entries.append(entry)
            # Trim oldest entries if we exceed the cap
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
            self._flush_counter += 1

            # Batch flush: write to disk every N events or T seconds
            now = time.time()
            should_flush = (
                self._flush_counter >= config.LOG_FLUSH_INTERVAL or
                (now - self._last_flush_time) >= config.LOG_FLUSH_SECONDS
            )
            if should_flush:
                self._flush()
                self._flush_counter = 0
                self._last_flush_time = now

    def flush(self) -> None:
        """Force an immediate flush to disk (call on shutdown)."""
        with self._lock:
            self._flush()
            self._flush_counter = 0
            self._last_flush_time = time.time()

    def _flush(self) -> None:
        """Write the full log list to disk (atomic‑ish write)."""
        tmp_path = self.log_file + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.log_file)       # atomic on most OS's
        except OSError as exc:
            print(f"[Logger] Write error: {exc}")

    def _load(self) -> list[dict]:
        """Load existing JSON log from disk."""
        if not os.path.isfile(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return []
