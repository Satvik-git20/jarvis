"""SQLite database lifecycle and rollback-safe schema migrations."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Migration:
    """One ordered, transactional database migration."""

    version: int
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool')),
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(conversation_id, sequence)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence ON messages(conversation_id, sequence)",
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                through_sequence INTEGER NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(conversation_id, through_sequence)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS provider_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                error_text TEXT NOT NULL DEFAULT '',
                cooldown_until REAL NOT NULL DEFAULT 0
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_provider_logs_provider_time ON provider_logs(provider, occurred_at)",
            """
            CREATE TABLE IF NOT EXISTS legacy_imports (
                source_path TEXT PRIMARY KEY,
                imported_at REAL NOT NULL
            )
            """,
        ),
    ),
)


def database_path_for(path: Path) -> Path:
    """Map legacy JSON paths to the shared SQLite database path.

    ``SessionStore(data / 'sessions.json')`` remains supported for callers
    and gives the importer the legacy source location, but persistence now
    happens in ``data / 'jarvis.db'``.
    """
    return path if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"} else path.parent / "jarvis.db"


class Database:
    """Owns connections and schema evolution for one JARVIS data directory."""

    def __init__(self, path: Path, *, migrations: tuple[Migration, ...] = MIGRATIONS):
        self.path = database_path_for(path)
        self._migrations = migrations
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        self._enable_wal(conn)
        return conn

    @staticmethod
    def _enable_wal(conn: sqlite3.Connection) -> None:
        """Switch the file to WAL, surviving other connections racing to do it.

        Unlike ordinary statements, ``PRAGMA journal_mode`` does not wait on
        ``busy_timeout`` while the database is still being created -- it fails
        immediately with "database is locked". So a daemon and a CLI starting
        on the same instant would otherwise crash one of them. WAL is a
        persistent property of the file, so whoever loses the race only has to
        wait out the winner's transition (milliseconds).
        """
        for attempt in range(100):
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError:
                if attempt == 99:
                    raise
                time.sleep(0.01)

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for queries and close it afterwards.

        ``with sqlite3.connect(...)`` only commits -- it never closes -- so a
        read path that used it would leak a file handle per call. Reads have
        no transaction to end, so closing here is safe and unconditional.
        """
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield an immediate transaction; rollback every failed write."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        """Apply pending migrations one at a time, atomically and in order."""
        conn = self.connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
            )
            for migration in self._migrations:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    already_applied = conn.execute(
                        "SELECT 1 FROM schema_migrations WHERE version = ?", (migration.version,)
                    ).fetchone()
                    if already_applied:
                        conn.commit()
                        continue
                    for statement in migration.statements:
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (migration.version, time.time()),
                    )
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
        finally:
            conn.close()

    def schema_version(self) -> int:
        with self.read() as conn:
            row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
        return int(row[0])
