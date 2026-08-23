"""
Persistent 1-5 star rating storage, backed by a small SQLite database.

Ratings are keyed by absolute file path. Used by the thumbnail grid/preview
window for quick keyboard rating, and by the "Analyze Ratings" tool, which
correlates ratings against embedded ComfyUI generation metadata (LoRAs
used, prompt keywords) to help identify what's driving good/bad results
across a rated batch.
"""

import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".img_sort", "ratings.db")


class RatingManager:
    """SQLite-backed store for per-image 1-5 star ratings."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    path TEXT PRIMARY KEY,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    rated_at TEXT NOT NULL
                )
                """
            )

    def set_rating(self, path, rating):
        """Set the rating (1-5) for a path, or clear it if rating is None."""
        path = os.path.abspath(path)
        with self._connect() as conn:
            if rating is None:
                conn.execute("DELETE FROM ratings WHERE path = ?", (path,))
            else:
                if not (1 <= rating <= 5):
                    raise ValueError(f"rating must be 1-5, got {rating}")
                conn.execute(
                    "INSERT INTO ratings (path, rating, rated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "rating = excluded.rating, rated_at = excluded.rated_at",
                    (path, rating, datetime.now(timezone.utc).isoformat()),
                )

    def get_rating(self, path):
        """Return the rating (1-5) for a path, or None if unrated."""
        path = os.path.abspath(path)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rating FROM ratings WHERE path = ?", (path,)
            ).fetchone()
        return row[0] if row else None

    def get_ratings_bulk(self, paths):
        """Return {path: rating} for whichever of `paths` are rated."""
        paths = [os.path.abspath(p) for p in paths if p]
        if not paths:
            return {}
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in paths)
            rows = conn.execute(
                f"SELECT path, rating FROM ratings WHERE path IN ({placeholders})",
                paths,
            ).fetchall()
        return dict(rows)


_manager = None


def get_rating_manager():
    """Get or create the shared RatingManager instance."""
    global _manager
    if _manager is None:
        _manager = RatingManager()
    return _manager


def format_stars(rating):
    """Return a 5-character filled/empty star string for a 1-5 rating."""
    return "★" * rating + "☆" * (5 - rating)


def combine_display_text(base_text, rating):
    """Append a rating's star string to an existing thumbnail label, if rated."""
    if not rating:
        return base_text
    stars = format_stars(rating)
    return f"{base_text} {stars}".strip()
