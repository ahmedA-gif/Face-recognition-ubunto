"""Migrate data from local SQLite databases to Neon PostgreSQL.

Run: python scripts/migrate_to_neon.py
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.postgres import execute_insert, execute_query, run_migrations, create_readonly_user


SQLITE_FILES = {
    "events": ROOT / "data" / "db" / "events.db",
    "faces": ROOT / "data" / "db" / "faces.db",
    "attendance": ROOT / "data" / "db" / "attendance.db",
}


def migrate_events():
    path = SQLITE_FILES["events"]
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return 0
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT date, time, person, direction, track_id, camera_id, confidence, snapshot_path, event_id FROM events ORDER BY id"
    ).fetchall()
    conn.close()
    count = 0
    for r in rows:
        execute_insert(
            """INSERT INTO events(date, time, person, direction, track_id, camera_id, confidence, snapshot_path, event_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (r["date"], r["time"], r["person"], r["direction"], r["track_id"],
             r["camera_id"], r["confidence"], r["snapshot_path"], r["event_id"]),
        )
        count += 1
    print(f"  [OK] {count} events migrated")
    return count


def migrate_faces():
    path = SQLITE_FILES["faces"]
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return 0
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT name, embedding, dim FROM faces ORDER BY id").fetchall()
    conn.close()
    count = 0
    for r in rows:
        emb = r["embedding"]
        if isinstance(emb, str):
            emb = emb.encode("latin-1")
        execute_insert(
            "INSERT INTO faces(name, embedding, dim) VALUES (%s, %s, %s)",
            (r["name"], emb, r["dim"]),
        )
        count += 1
    print(f"  [OK] {count} face embeddings migrated")
    return count


def migrate_attendance():
    path = SQLITE_FILES["attendance"]
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return 0
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT date, person_id, person_name, check_in_time, check_out_time, status, work_hours, confidence, camera_id FROM attendance_logs ORDER BY id"
    ).fetchall()
    count = 0
    for r in rows:
        execute_insert(
            """INSERT INTO attendance_logs(date, person_id, person_name, check_in_time, check_out_time, status, work_hours, confidence, camera_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT(date, person_id) DO NOTHING""",
            (r["date"], r["person_id"], r["person_name"], r["check_in_time"],
             r["check_out_time"], r["status"], r["work_hours"], r["confidence"], r["camera_id"]),
        )
        count += 1

    rows = conn.execute(
        "SELECT line_coords, cluster_confidence FROM boundary_calibration_history ORDER BY id"
    ).fetchall()
    cal_count = 0
    for r in rows:
        execute_insert(
            "INSERT INTO boundary_calibration_history(line_coords, cluster_confidence) VALUES (%s, %s)",
            (r["line_coords"], r["cluster_confidence"]),
        )
        cal_count += 1

    conn.close()
    print(f"  [OK] {count} attendance logs, {cal_count} calibrations migrated")
    return count


if __name__ == "__main__":
    print("=" * 60)
    print("  SQLite -> Neon PostgreSQL Migration")
    print("=" * 60)

    print("\n1. Testing Neon connection...")
    rows = execute_query("SELECT NOW() as time")
    print(f"   Connected! Server time: {rows[0]['time']}")

    print("\n2. Creating tables...")
    run_migrations()
    print("   Tables created")

    print("\n3. Migrating data...")
    migrate_events()
    migrate_faces()
    migrate_attendance()

    print("\n4. Creating read-only user...")
    create_readonly_user("readonly_dev", "readonly_pass_2026")

    print("\n" + "=" * 60)
    print("  Migration complete!")
    print("=" * 60)

    print("\n5. Verifying counts...")
    for table in ["events", "faces", "attendance_logs"]:
        rows = execute_query(f"SELECT COUNT(*) AS n FROM {table}")
        print(f"   {table}: {rows[0]['n']} rows")
