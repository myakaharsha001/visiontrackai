import time
import unittest
from datetime import date, datetime

from config import DatabaseConfig
from database import (CREATE_TABLE_SQL, EVENT_COLUMNS, INSERT_SQL, Database, DatabaseError,
                      EventWriter, HistoryFilter, build_where, event_to_params)
from detector import detection_to_event
from tests.helpers import det

CFG = DatabaseConfig("localhost", 3306, "vision_platform", "u", "p", 2)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        self.conn.log.append(("execute", sql, params))
        if self.conn.fail:
            raise RuntimeError("Lost connection to MySQL server")

    def executemany(self, sql, rows):
        self.conn.log.append(("executemany", sql, list(rows)))
        if self.conn.fail:
            raise RuntimeError("Lost connection to MySQL server")

    def fetchall(self):
        return self.conn.rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, fail=False, rows=None):
        self.fail, self.rows, self.log = fail, rows or [], []
        self.commits = self.rollbacks = self.closes = 0

    def cursor(self, dictionary=False):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def get_connection(self):
        return self.conn


class FakeConnector:
    def __init__(self, conn):
        self.conn = conn
        self.pool_kwargs = None
        outer = self

        class _Pooling:
            @staticmethod
            def MySQLConnectionPool(**kw):
                outer.pool_kwargs = kw
                return FakePool(outer.conn)
        self.pooling = _Pooling


def make_db(conn):
    return Database(CFG, connector_module=FakeConnector(conn))


class WhereTests(unittest.TestCase):
    def test_no_filter(self):
        self.assertEqual(build_where(None), ("", []))
        self.assertEqual(build_where(HistoryFilter()), ("", []))

    def test_all_filters_are_placeholders_not_values(self):
        f = HistoryFilter(search="per", classes=["person", "car"], date_from=date(2026, 1, 1),
                          date_to=date(2026, 1, 31), source_type="webcam")
        sql, params = build_where(f)
        self.assertNotIn("person", sql)
        self.assertNotIn("2026", sql)
        self.assertEqual(sql.count("%s"), len(params))
        self.assertIn("IN (%s, %s)", sql)
        self.assertEqual(params[0], "%per%")
        self.assertEqual(params[-1], "webcam")
        # inclusive end date -> strictly less than the next midnight
        self.assertIn(datetime(2026, 2, 1), params)

    def test_injection_attempt_stays_in_params(self):
        evil = "x'; DROP TABLE detection_events; --"
        sql, params = build_where(HistoryFilter(search=evil, classes=[evil]))
        self.assertNotIn("DROP", sql)
        from utils import escape_like
        self.assertIn(f"%{escape_like(evil)}%", params)
        self.assertIn(evil, params)

    def test_like_wildcards_escaped(self):
        _, params = build_where(HistoryFilter(search="100%_"))
        self.assertEqual(params[0], "%100\\%\\_%")


class EventParamTests(unittest.TestCase):
    def test_param_order_matches_insert_columns(self):
        ev = detection_to_event(det(), "video_file", "sid")
        params = event_to_params(ev)
        self.assertEqual(len(params), len(EVENT_COLUMNS))
        self.assertEqual(INSERT_SQL.count("%s"), len(params))
        self.assertEqual(params[1], "person")

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            event_to_params({"object_class": "x"})

    def test_required_columns_in_schema_sql(self):
        for col in ("log_id", "`timestamp`", "object_class", "confidence",
                    "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            self.assertIn(col, CREATE_TABLE_SQL)
        with open("database_schema.sql", encoding="utf-8") as fh:
            sql_file = fh.read()
        for col in ("log_id", "object_class", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            self.assertIn(col, sql_file)


class DatabaseTests(unittest.TestCase):
    def test_insert_uses_executemany_and_commits(self):
        conn = FakeConn()
        db = make_db(conn)
        n = db.insert_events([detection_to_event(det(), "webcam", "s"),
                              detection_to_event(det("car", 0.8, cid=2), "webcam", "s")])
        self.assertEqual(n, 2)
        kind, sql, rows = conn.log[0]
        self.assertEqual(kind, "executemany")
        self.assertEqual(sql, INSERT_SQL)
        self.assertEqual(len(rows), 2)
        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.closes, 1)         # returned to pool

    def test_insert_failure_raises_and_rolls_back(self):
        conn = FakeConn(fail=True)
        db = make_db(conn)
        with self.assertRaises(DatabaseError):
            db.insert_events([detection_to_event(det(), "webcam", "s")])
        self.assertEqual(conn.rollbacks, 1)
        self.assertEqual(conn.commits, 0)

    def test_check_connection_never_raises(self):
        ok, msg = make_db(FakeConn(fail=True)).check_connection()
        self.assertFalse(ok)
        self.assertTrue(msg)
        ok, _ = make_db(FakeConn()).check_connection()
        self.assertTrue(ok)

    def test_pool_reused_between_calls(self):
        conn = FakeConn()
        fc = FakeConnector(conn)
        db = Database(CFG, connector_module=fc)
        db.check_connection()
        pool = db._pool
        db.check_connection()
        self.assertIs(db._pool, pool)
        self.assertEqual(fc.pool_kwargs["pool_size"], 2)

    def test_history_query_is_parameterised(self):
        conn = FakeConn(rows=[])
        db = make_db(conn)
        db.fetch_history(HistoryFilter(search="x'; --"), limit=10, offset=20)
        _, sql, params = conn.log[0]
        self.assertNotIn("--", sql)
        self.assertEqual(params[-2:], (10, 20))

    def test_limit_is_capped(self):
        conn = FakeConn(rows=[])
        make_db(conn).fetch_recent(10**9)
        self.assertEqual(conn.log[0][2][-1], 1000)

    def test_confidence_converted_to_float(self):
        from decimal import Decimal
        df = make_db(FakeConn(rows=[{"confidence": Decimal("0.8532")}])).fetch_recent(1)
        self.assertIsInstance(df["confidence"].iloc[0], float)

    def test_bad_bucket_rejected(self):
        with self.assertRaises(ValueError):
            make_db(FakeConn()).activity_over_time("year; DROP TABLE x")

    def test_summary_shapes(self):
        conn = FakeConn(rows=[{"total": 3, "unique_classes": 2, "avg_confidence": 0.8,
                               "first_ts": None, "last_ts": None, "object_class": "person", "n": 2}])
        s = make_db(conn).summary()
        self.assertEqual((s["total"], s["unique_classes"], s["top_class"]), (3, 2, "person"))


class FlakyDB:
    def __init__(self, fail=False, delay=0.0):
        self.fail, self.delay, self.saved = fail, delay, []

    def insert_events(self, events):
        time.sleep(self.delay)
        if self.fail:
            raise DatabaseError("MySQL is down")
        self.saved.extend(events)
        return len(events)


class EventWriterTests(unittest.TestCase):
    def test_events_are_flushed_on_stop(self):
        db = FlakyDB()
        w = EventWriter(db, flush_interval=0.05).start()
        for i in range(7):
            self.assertTrue(w.submit({"i": i}))
        w.stop()
        self.assertEqual(len(db.saved), 7)
        snap = w.snapshot()
        self.assertEqual((snap["submitted"], snap["inserted"], snap["failed"]), (7, 7, 0))

    def test_failures_are_counted_not_claimed_as_saved(self):
        db = FlakyDB(fail=True)
        w = EventWriter(db, flush_interval=0.05).start()
        for i in range(4):
            w.submit({"i": i})
        w.stop()
        snap = w.snapshot()
        self.assertEqual(snap["inserted"], 0)
        self.assertEqual(snap["failed"], 4)
        self.assertIn("down", snap["last_error"])

    def test_full_queue_drops_and_never_blocks(self):
        w = EventWriter(FlakyDB(), max_queue=2)          # thread not started
        results = [w.submit({"i": i}) for i in range(5)]
        self.assertEqual(results, [True, True, False, False, False])
        self.assertEqual(w.snapshot()["dropped"], 3)


if __name__ == "__main__":
    unittest.main()
