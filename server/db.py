import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, Dict
from uuid import uuid4
import secrets


class Database:
    """Minimal SQLite wrapper for users, sessions, and file metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    token TEXT UNIQUE NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    prompt TEXT,
                    created_at INTEGER NOT NULL,
                    combined_md TEXT,
                    combined_txt TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT,
                    original_name TEXT,
                    page INTEGER,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()

    def create_user(self) -> Dict[str, str]:
        now = int(time.time())
        user_id = uuid4().hex
        token = secrets.token_hex(16)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, token, created_at) VALUES (?, ?, ?)",
                (user_id, token, now),
            )
            conn.commit()
        return {"user_id": user_id, "token": token}

    def get_user_by_token(self, token: str) -> Optional[Dict[str, str]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, token, created_at FROM users WHERE token = ?", (token,)
            ).fetchone()
        if not row:
            return None
        return {"user_id": row["id"], "token": row["token"], "created_at": row["created_at"]}

    def create_session(self, session_id: str, user_id: str, prompt: str) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, prompt, created_at) VALUES (?, ?, ?, ?)",
                (session_id, user_id, prompt, now),
            )
            conn.commit()

    def finalize_session(
        self, session_id: str, combined_md: str, combined_txt: str
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET combined_md = ?, combined_txt = ? WHERE id = ?",
                (combined_md, combined_txt, session_id),
            )
            conn.commit()

    def add_file_entry(
        self,
        session_id: str,
        kind: str,
        input_path: str,
        output_path: Optional[str],
        original_name: Optional[str],
        page: Optional[int] = None,
    ) -> None:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (session_id, kind, input_path, output_path, original_name, page, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, kind, input_path, output_path, original_name, page, now),
            )
            conn.commit()
