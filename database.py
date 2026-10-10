"""database.py - MySQL access layer (connection pool, queries, async writer).

* All SQL values go through parameter placeholders (%s) - never string-built.
  The only text spliced into SQL are fixed strings written in this file
  (column names, ORDER BY direction) or placeholder lists like "%s, %s".
* One small connection POOL is reused instead of reconnecting for every call.
* Inserts from the video loop go through EventWriter (background thread), so a
  slow or dead database can never freeze the live video.
"""
from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from config import DatabaseConfig
from logger import get_logger
from utils import escape_like

log = get_logger(__name__)

TABLE = "detection_events"
EVENT_COLUMNS = ("timestamp", "object_class", "confidence", "bbox_x", "bbox_y",
                 "bbox_w", "bbox_h", "source_type", "session_id")

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    log_id       BIGINT UNSIGNED  NOT NULL AUTO_INCREMENT,
    `timestamp`  DATETIME(3)      NOT NULL,
    object_class VARCHAR(64)      NOT NULL,
    confidence   DECIMAL(5,4)     NOT NULL,
    bbox_x       INT              NOT NULL,
    bbox_y       INT              NOT NULL,
    bbox_w       INT UNSIGNED     NOT NULL,
    bbox_h       INT UNSIGNED     NOT NULL,
    source_type  ENUM('webcam','video_file') NOT NULL DEFAULT 'webcam',
    session_id   CHAR(36)         NULL,
    created_at   TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (log_id),
    KEY idx_timestamp (`timestamp`),
    KEY idx_object_class (object_class),
    KEY idx_session (session_id),
    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

INSERT_SQL = (f"INSERT INTO {TABLE} (`timestamp`, object_class, confidence, bbox_x, bbox_y, "
              "bbox_w, bbox_h, source_type, session_id) "
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")

HISTORY_COLUMNS = ("log_id, `timestamp`, object_class, confidence, "
                   "bbox_x, bbox_y, bbox_w, bbox_h, source_type, session_id")

BUCKET_SECONDS = {"minute": 60, "hour": 3600, "day": 86400}


class DatabaseError(RuntimeError):
    """A database operation failed (message is safe to show to the user)."""


@dataclass
class HistoryFilter:
    search: str = ""
    classes: List[str] = field(default_factory=list)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    source_type: Optional[str] = None      # "webcam" | "video_file" | None


def build_where(f: Optional[HistoryFilter]) -> Tuple[str, List[Any]]:
    """Turn a HistoryFilter into ('WHERE ...' or '', [params]). Pure & testable."""
    if f is None:
        return "", []
    clauses: List[str] = []
    params: List[Any] = []
    if f.search and f.search.strip():
        clauses.append("object_class LIKE %s")
        params.append(f"%{escape_like(f.search.strip())}%")
    if f.classes:
        clauses.append("object_class IN (" + ", ".join(["%s"] * len(f.classes)) + ")")
        params.extend(f.classes)
    if f.date_from:
        clauses.append("`timestamp` >= %s")
        params.append(datetime.combine(f.date_from, datetime.min.time()))
    if f.date_to:                                   # inclusive end day
        clauses.append("`timestamp` < %s")
        params.append(datetime.combine(f.date_to + timedelta(days=1), datetime.min.time()))
    if f.source_type:
        clauses.append("source_type = %s")
        params.append(f.source_type)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def event_to_params(event: Dict[str, Any]) -> Tuple[Any, ...]:
    """Order an event dict into the INSERT parameter tuple (raises KeyError if a column is missing)."""
    return tuple(event[c] for c in EVENT_COLUMNS)


class Database:
    def __init__(self, cfg: DatabaseConfig, connector_module=None):
        self.cfg = cfg
        self._connector = connector_module          # injectable for tests
        self._pool = None
        self._lock = threading.Lock()

    # ---- connection management ------------------------------------
    def _mysql(self):
        if self._connector is None:
            try:
                import mysql.connector as connector
                import mysql.connector.pooling  # noqa: F401  (make sure submodule is loaded)
            except ImportError as exc:
                raise DatabaseError("Package 'mysql-connector-python' is not installed. "
                                    "Run: pip install -r requirements.txt") from exc
            self._connector = connector
        return self._connector

    def _get_pool(self):
        with self._lock:
            if self._pool is None:
                pool_cls = self._mysql().pooling.MySQLConnectionPool
                try:
                    self._pool = pool_cls(
                        pool_name="visiontrack_pool", pool_size=self.cfg.pool_size,
                        host=self.cfg.host, port=self.cfg.port, user=self.cfg.user,
                        password=self.cfg.password, database=self.cfg.name,
                        connection_timeout=5, autocommit=False, charset="utf8mb4")
                except Exception as exc:
                    raise DatabaseError(self._friendly(exc)) from exc
            return self._pool

    def _reset_pool(self) -> None:
        with self._lock:
            self._pool = None

    @staticmethod
    def _friendly(exc: Exception) -> str:
        text = str(exc)
        low = text.lower()
        if "access denied" in low:
            hint = "Wrong DB_USER/DB_PASSWORD in .env, or the user has no rights on the database."
        elif "unknown database" in low:
            hint = "The database does not exist yet. Run database_schema.sql as the MySQL admin."
        elif "can't connect" in low or "refused" in low or "2003" in low or "timed out" in low:
            hint = "MySQL is not running or DB_HOST/DB_PORT are wrong. Start the MySQL80 service."
        else:
            hint = "See the troubleshooting section of the README."
        return f"{text} -> {hint}"

    @contextmanager
    def connection(self) -> Iterator[Any]:
        pool = self._get_pool()
        try:
            conn = pool.get_connection()
        except Exception as exc:
            self._reset_pool()
            raise DatabaseError(self._friendly(exc)) from exc
        try:
            yield conn
        finally:
            try:
                conn.close()                         # returns it to the pool
            except Exception:                        # pragma: no cover
                log.debug("error returning connection to pool", exc_info=True)

    def _query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        try:
            with self.connection() as conn:
                cur = conn.cursor(dictionary=True)
                try:
                    cur.execute(sql, tuple(params))
                    return cur.fetchall()
                finally:
                    cur.close()
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(self._friendly(exc)) from exc

    def _frame(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        df = pd.DataFrame(self._query(sql, params))
        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].astype(float)
        return df

    # ---- status / setup -------------------------------------------
    def check_connection(self) -> Tuple[bool, str]:
        """Return (ok, message). Never raises."""
        try:
            self._query("SELECT 1 AS ok")
            return True, f"Connected to {self.cfg.name} on {self.cfg.host}:{self.cfg.port}"
        except DatabaseError as exc:
            return False, str(exc)

    def init_schema(self) -> None:
        """Create the table if missing (needs CREATE privilege; the SQL file grants it)."""
        try:
            with self.connection() as conn:
                cur = conn.cursor()
                try:
                    cur.execute(CREATE_TABLE_SQL)
                    conn.commit()
                finally:
                    cur.close()
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(self._friendly(exc)) from exc

    # ---- writes ----------------------------------------------------
    def insert_events(self, events: Sequence[Dict[str, Any]]) -> int:
        """Insert events with ONE parameterised executemany. Returns rows inserted.

        Raises DatabaseError on failure (nothing is reported as saved then).
        """
        if not events:
            return 0
        rows = [event_to_params(e) for e in events]
        try:
            with self.connection() as conn:
                cur = conn.cursor()
                try:
                    cur.executemany(INSERT_SQL, rows)
                    conn.commit()
                    return len(rows)
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    cur.close()
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(self._friendly(exc)) from exc

    def insert_event(self, event: Dict[str, Any]) -> int:
        return self.insert_events([event])

    # ---- reads -----------------------------------------------------
    def fetch_recent(self, limit: int = 20) -> pd.DataFrame:
        limit = max(1, min(int(limit), 1000))
        return self._frame(f"SELECT {HISTORY_COLUMNS} FROM {TABLE} "
                           "ORDER BY `timestamp` DESC, log_id DESC LIMIT %s", [limit])

    def fetch_history(self, f: Optional[HistoryFilter] = None, limit: int = 50,
                      offset: int = 0) -> pd.DataFrame:
        where, params = build_where(f)
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        return self._frame(f"SELECT {HISTORY_COLUMNS} FROM {TABLE} {where} "
                           "ORDER BY `timestamp` DESC, log_id DESC LIMIT %s OFFSET %s",
                           params + [limit, offset])

    def count_history(self, f: Optional[HistoryFilter] = None) -> int:
        where, params = build_where(f)
        rows = self._query(f"SELECT COUNT(*) AS n FROM {TABLE} {where}", params)
        return int(rows[0]["n"]) if rows else 0

    def export_history(self, f: Optional[HistoryFilter] = None,
                       max_rows: int = 100_000) -> pd.DataFrame:
        """All rows matching the filter (capped) - used for CSV export."""
        where, params = build_where(f)
        return self._frame(f"SELECT {HISTORY_COLUMNS} FROM {TABLE} {where} "
                           "ORDER BY `timestamp` DESC, log_id DESC LIMIT %s",
                           params + [int(max_rows)])

    def distinct_classes(self) -> List[str]:
        rows = self._query(f"SELECT DISTINCT object_class FROM {TABLE} ORDER BY object_class")
        return [r["object_class"] for r in rows]

    def summary(self, f: Optional[HistoryFilter] = None) -> Dict[str, Any]:
        """Real aggregate numbers from the table (nothing estimated)."""
        where, params = build_where(f)
        rows = self._query(
            f"SELECT COUNT(*) AS total, COUNT(DISTINCT object_class) AS unique_classes, "
            f"AVG(confidence) AS avg_confidence, MIN(`timestamp`) AS first_ts, "
            f"MAX(`timestamp`) AS last_ts FROM {TABLE} {where}", params)
        row = rows[0] if rows else {}
        top = self._query(f"SELECT object_class, COUNT(*) AS n FROM {TABLE} {where} "
                          "GROUP BY object_class ORDER BY n DESC, object_class LIMIT 1", params)
        return {
            "total": int(row.get("total") or 0),
            "unique_classes": int(row.get("unique_classes") or 0),
            "avg_confidence": float(row["avg_confidence"]) if row.get("avg_confidence") is not None else None,
            "first_ts": row.get("first_ts"), "last_ts": row.get("last_ts"),
            "top_class": top[0]["object_class"] if top else None,
            "top_class_count": int(top[0]["n"]) if top else 0,
        }

    def class_counts(self, f: Optional[HistoryFilter] = None, limit: int = 30) -> pd.DataFrame:
        where, params = build_where(f)
        return self._frame(f"SELECT object_class, COUNT(*) AS detections, AVG(confidence) AS avg_confidence "
                           f"FROM {TABLE} {where} GROUP BY object_class "
                           "ORDER BY detections DESC, object_class LIMIT %s", params + [int(limit)])

    def confidence_values(self, f: Optional[HistoryFilter] = None, limit: int = 20000) -> pd.DataFrame:
        where, params = build_where(f)
        return self._frame(f"SELECT object_class, confidence FROM {TABLE} {where} "
                           "ORDER BY log_id DESC LIMIT %s", params + [int(limit)])

    def activity_over_time(self, bucket: str = "minute",
                           f: Optional[HistoryFilter] = None) -> pd.DataFrame:
        """Events per time bucket. ``bucket`` is whitelisted; seconds go in as parameters."""
        seconds = BUCKET_SECONDS.get(bucket)
        if seconds is None:
            raise ValueError(f"bucket must be one of {sorted(BUCKET_SECONDS)}")
        where, params = build_where(f)
        sql = (f"SELECT FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(`timestamp`) / %s) * %s) AS bucket, "
               f"COUNT(*) AS detections FROM {TABLE} {where} GROUP BY bucket ORDER BY bucket")
        return self._frame(sql, [seconds, seconds] + params)

    def close(self) -> None:
        """Drop the pool reference; pooled connections close when garbage collected."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool._remove_connections()          # mysql-connector internal; best effort
            except Exception:
                log.debug("pool cleanup skipped", exc_info=True)


class EventWriter:
    """Background thread that batches events into MySQL.

    Honest accounting: ``inserted`` counts rows MySQL really accepted;
    ``failed`` counts rows lost because an insert raised; ``dropped`` counts
    rows rejected because the queue was full. There is NO automatic retry -
    failed events are skipped (and reported), not silently re-claimed as saved.
    """

    def __init__(self, db: Database, max_queue: int = 1000, batch_size: int = 50,
                 flush_interval: float = 0.5):
        self._db = db
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stats_lock = threading.Lock()
        self.submitted = 0
        self.inserted = 0
        self.failed = 0
        self.dropped = 0
        self.last_error: Optional[str] = None

    def start(self) -> "EventWriter":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
            self._thread.start()
        return self

    def submit(self, event: Dict[str, Any]) -> bool:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            with self._stats_lock:
                self.dropped += 1
            return False
        with self._stats_lock:
            self.submitted += 1
        return True

    def _drain(self) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        while len(batch) < self._batch_size:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            n = self._db.insert_events(batch)
            with self._stats_lock:
                self.inserted += n
                self.last_error = None
        except Exception as exc:                     # DatabaseError or anything unexpected
            log.error("Insert of %d events failed: %s", len(batch), exc)
            with self._stats_lock:
                self.failed += len(batch)
                self.last_error = str(exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._flush_interval)
            while True:
                batch = self._drain()
                if not batch:
                    break
                self._flush_batch(batch)

    def stop(self, timeout: float = 5.0) -> None:
        """Flush what is queued, then stop the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        while True:                                  # final flush if thread was slow/dead
            batch = self._drain()
            if not batch:
                break
            self._flush_batch(batch)

    def snapshot(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {"submitted": self.submitted, "inserted": self.inserted,
                    "failed": self.failed, "dropped": self.dropped,
                    "pending": self._q.qsize(), "last_error": self.last_error}
