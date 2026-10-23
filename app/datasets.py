"""학습에 쓸 파형 세트. 샘플은 Influx에 두고, 여기에는 시각과 센서만 저장한다."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .catalog import _connect


def init_datasets() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_items (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                machine TEXT NOT NULL,
                channel TEXT NOT NULL,
                ts TEXT NOT NULL,
                rpm REAL,
                n INTEGER,
                unit TEXT,
                preset TEXT,
                UNIQUE(dataset_id, machine, channel, ts)
            )
            """
        )


def list_datasets() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT id, name, created_at FROM datasets ORDER BY id DESC").fetchall()
        out = []
        for row in rows:
            items = _items(conn, row["id"])
            out.append({"id": row["id"], "name": row["name"], "createdAt": row["created_at"], "items": items})
        return out


def get_dataset(dataset_id: int) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT id, name, created_at FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return {"id": row["id"], "name": row["name"], "createdAt": row["created_at"], "items": _items(conn, row["id"])}


def create_dataset(name: str, items: list[dict]) -> dict:
    cleaned = name.strip()
    if not cleaned or len(cleaned) > 40:
        raise ValueError("세트 이름은 1~40자입니다.")
    if not items or len(items) > 100:
        raise ValueError("세트에는 파형을 1~100개 넣습니다.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        try:
            cur = conn.execute("INSERT INTO datasets (name, created_at) VALUES (?, ?)", (cleaned, now))
        except sqlite3.IntegrityError as e:
            raise ValueError("같은 이름의 세트가 있습니다.") from e
        dataset_id = int(cur.lastrowid)
        for item in items:
            conn.execute(
                """
                INSERT OR IGNORE INTO dataset_items (dataset_id, machine, channel, ts, rpm, n, unit, preset)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    item["machine"],
                    item["channel"],
                    item["ts"],
                    item.get("rpm"),
                    item.get("n"),
                    item.get("unit") or "",
                    item.get("preset") or "",
                ),
            )
    return get_dataset(dataset_id)


def delete_dataset(dataset_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM dataset_items WHERE dataset_id = ?", (dataset_id,))
        deleted = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,)).rowcount
    if not deleted:
        raise KeyError(dataset_id)


def _items(conn: sqlite3.Connection, dataset_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT machine, channel, ts, rpm, n, unit, preset
        FROM dataset_items WHERE dataset_id = ? ORDER BY ts
        """,
        (dataset_id,),
    ).fetchall()
    return [
        {
            "machine": row["machine"],
            "channel": row["channel"],
            "ts": row["ts"],
            "rpm": row["rpm"],
            "n": row["n"],
            "unit": row["unit"] or "",
            "preset": row["preset"] or "",
        }
        for row in rows
    ]
