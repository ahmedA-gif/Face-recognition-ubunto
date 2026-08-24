from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool


_NEON_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://authenticator:npg_IaOx4mzXT2ib@ep-lively-hill-adqayuhc-pooler.c-2.us-east-1.aws.neon.tech/AI%20attendance%20system%20?sslmode=require",
)

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        ensure_database()
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            dsn=_NEON_URL,
        )
    return _pool


def get_conn():
    """Get a connection from the pool with RealDictCursor as default cursor."""
    pool = get_pool()
    return pool.getconn()


def put_conn(conn):
    """Return a connection to the pool."""
    pool = get_pool()
    if conn is not None:
        pool.putconn(conn)


def execute_query(sql: str, params: tuple = (), *, fetch: bool = True) -> List[Dict[str, Any]]:
    """Execute a query and return rows as list of dicts."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if fetch and cur.description is not None:
                return [dict(row) for row in cur.fetchall()]
            return []
    finally:
        put_conn(conn)


def execute_write(sql: str, params: tuple = ()) -> int:
    """Execute a write query (INSERT/UPDATE/DELETE) and return rowcount."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
    finally:
        put_conn(conn)


def execute_insert(sql: str, params: tuple = ()) -> Any:
    """Execute an INSERT and return the generated id."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.fetchone()[0] if cur.description else None
    finally:
        put_conn(conn)


def execute_many(sql: str, params_list: list) -> int:
    """Execute a batch INSERT/UPDATE and return rowcount."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)
            conn.commit()
            return cur.rowcount
    finally:
        put_conn(conn)


def ensure_database():
    """Verify the target database is reachable."""
    import psycopg2 as _pg
    conn = _pg.connect(_NEON_URL)
    conn.close()


def run_migrations():
    """Create all tables if they don't exist."""
    ensure_database()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    person TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    track_id INTEGER,
                    camera_id TEXT,
                    confidence REAL,
                    snapshot_path TEXT,
                    event_id TEXT,
                    created_at TEXT DEFAULT NOW()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_id_desc ON events(id DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_date_person ON events(date, person)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BYTEA NOT NULL,
                    dim INTEGER NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id SERIAL PRIMARY KEY,
                    date TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    person_name TEXT,
                    check_in_time TEXT,
                    check_out_time TEXT,
                    status TEXT,
                    work_hours REAL,
                    confidence REAL,
                    camera_id TEXT,
                    updated_at TEXT DEFAULT NOW(),
                    UNIQUE(date, person_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs(date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance_logs(person_id, date)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS boundary_calibration_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TEXT DEFAULT NOW(),
                    line_coords TEXT,
                    cluster_confidence REAL
                )
            """)
        conn.commit()
    finally:
        put_conn(conn)


def create_readonly_user(username: str = "readonly_dev", password: str = "readonly_pass_2026"):
    """Create a read-only user for other developers."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE USER {username} WITH PASSWORD '{password}'")
            cur.execute(f"GRANT CONNECT ON DATABASE \"AI attendance system\" TO {username}")
            cur.execute(f"GRANT USAGE ON SCHEMA public TO {username}")
            cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {username}")
            cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {username}")
        conn.commit()
        print(f"[DB] Read-only user '{username}' created successfully")
    except psycopg2.errors.DuplicateObject:
        print(f"[DB] User '{username}' already exists")
    finally:
        put_conn(conn)


if __name__ == "__main__":
    print("[DB] Testing Neon connection...")
    rows = execute_query("SELECT NOW() as time")
    print(f"[DB] Connected! Server time: {rows[0]['time']}")

    print("[DB] Running migrations...")
    run_migrations()
    print("[DB] Migrations complete")

    print("[DB] Creating read-only user...")
    create_readonly_user()
    print("[DB] Done!")
