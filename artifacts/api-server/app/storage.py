import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .config import settings


class Storage:
    def __init__(self) -> None:
        self.path = Path(settings.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, raw_message TEXT NOT NULL,
                sender TEXT, risk_score INTEGER NOT NULL, label TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id INTEGER NOT NULL, helpful INTEGER NOT NULL,
                note TEXT, created_at TEXT NOT NULL, FOREIGN KEY(analysis_id) REFERENCES analyses(id)
            );""")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save_analysis(self, payload: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            cur = db.execute("INSERT INTO analyses(created_at,raw_message,sender,risk_score,label,payload) VALUES(?,?,?,?,?,?)", (now, payload["raw_message"], payload.get("sender"), payload["risk_score"], payload["label"], json.dumps(payload)))
            return int(cur.lastrowid)

    def history(self, limit: int) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [json.loads(row["payload"]) | {"id": row["id"], "created_at": row["created_at"]} for row in db.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,))]

    def get(self, analysis_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            return (json.loads(row["payload"]) | {"id": row["id"], "created_at": row["created_at"]}) if row else None

    def feedback(self, analysis_id: int, helpful: bool, note: str | None) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO feedback(analysis_id,helpful,note,created_at) VALUES(?,?,?,?)", (analysis_id, int(helpful), note, datetime.now(timezone.utc).isoformat()))

    def stats(self) -> dict[str, Any]:
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            labels = {row["label"]: row["count"] for row in db.execute("SELECT label,COUNT(*) count FROM analyses GROUP BY label")}
            return {"total_analyses": total, "labels": labels}