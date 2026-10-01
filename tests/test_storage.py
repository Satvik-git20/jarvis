"""Storage migrations, repositories, legacy compatibility, and contention."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from jarvis.core.budget import BudgetLedger, Limits
from jarvis.core.conversation import SessionStore
from jarvis.storage.database import MIGRATIONS, Database, Migration
from jarvis.storage.repositories import ConversationRepository, ProviderLogRepository

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_database_creates_versioned_storage_schema(tmp_path):
    db = Database(tmp_path / "jarvis.db")

    assert db.schema_version() == 1
    with db.connect() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"conversations", "messages", "summaries", "provider_logs", "schema_migrations"} <= tables


def test_failed_migration_rolls_back_its_partial_schema(tmp_path):
    migrations = (
        Migration(1, ("CREATE TABLE stable (id INTEGER PRIMARY KEY)",)),
        Migration(2, ("CREATE TABLE partial (id INTEGER PRIMARY KEY)", "THIS IS NOT SQL")),
    )
    path = tmp_path / "rollback.db"

    with pytest.raises(sqlite3.OperationalError):
        Database(path, migrations=migrations)

    with sqlite3.connect(path) as conn:
        versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert versions == {1}
    assert "stable" in tables
    assert "partial" not in tables


def test_conversation_repository_persists_messages_and_summaries(tmp_path):
    repo = ConversationRepository(Database(tmp_path / "jarvis.db"))
    repo.get_or_create("c_1", now=1.0)
    repo.append_turn("c_1", "hello", "hi")
    repo.append_turn("c_1", "how are you?", "well")
    repo.upsert_summary("c_1", "Greeting completed", through_sequence=2)

    assert [(m.role, m.content) for m in repo.history("c_1")] == [
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "how are you?"),
        ("assistant", "well"),
    ]
    assert repo.latest_summary("c_1") == "Greeting completed"


def test_session_store_imports_legacy_json_once_and_uses_sqlite_afterwards(tmp_path):
    legacy = tmp_path / "sessions.json"
    legacy.write_text(json.dumps({
        "s_legacy": {
            "id": "s_legacy", "created": 1.0, "updated": 2.0,
            "messages": [{"role": "user", "content": "old question"}],
        }
    }), "utf-8")

    store = SessionStore(legacy)
    assert [message.content for message in store.history("s_legacy", system=None)] == ["old question"]
    store.append("s_legacy", "new question", "new answer")

    reopened = SessionStore(legacy)
    assert [message.content for message in reopened.history("s_legacy", system=None)] == [
        "old question", "new question", "new answer",
    ]
    assert (tmp_path / "jarvis.db").is_file()
    assert json.loads(legacy.read_text("utf-8"))["s_legacy"]["messages"][0]["content"] == "old question"


def test_provider_logs_are_durable_and_enforce_budget(tmp_path):
    path = tmp_path / "budget.json"
    first = BudgetLedger(path, limits={"p": Limits(rpm=3, rpd=10, tpd=20)})
    first.record("p", tokens=12)
    first.penalize("p", "temporary", seconds=60)
    assert first.available("p") is False

    second = BudgetLedger(path, limits={"p": Limits(rpm=3, rpd=10, tpd=20)})
    assert second.snapshot()["p"]["total_requests"] == 1
    assert second.snapshot()["p"]["total_tokens"] == 12
    assert second.snapshot()["p"]["errors"] == 1
    assert second.snapshot()["p"]["last_error"] == "temporary"
    assert second.available("p") is False
    second.record("p")
    assert second.available("p") is True


def test_budget_ledger_imports_existing_json_history(tmp_path):
    now = time.time()
    legacy = tmp_path / "budget.json"
    legacy.write_text(json.dumps({
        "providers": {
            "p": {
                "requests_day": [now], "tokens_day": [[now, 7]],
                "total_requests": 4, "total_tokens": 20, "errors": 2,
                "last_error": "old failure", "cooldown_until": 0, "last_used": now,
            }
        }
    }), "utf-8")

    ledger = BudgetLedger(legacy, limits={"p": Limits(rpm=10, rpd=10, tpd=100)})
    snapshot = ledger.snapshot()["p"]
    assert snapshot["total_requests"] == 4
    assert snapshot["total_tokens"] == 20
    assert snapshot["errors"] == 2
    assert snapshot["last_error"] == "old failure"


def test_concurrent_session_writers_do_not_lose_turns(tmp_path):
    """Separate store instances model daemon and voice processes sharing one DB."""
    legacy_path = tmp_path / "sessions.json"
    workers, turns_per_worker = 6, 8
    # A timeout turns "a worker died before the barrier" into a test failure
    # instead of a suite that hangs forever.
    barrier = threading.Barrier(workers, timeout=30)

    def write_turns(worker: int) -> None:
        store = SessionStore(legacy_path, max_turns=1_000)
        store.get_or_create("s_shared")
        barrier.wait()
        for turn in range(turns_per_worker):
            store.append("s_shared", f"u-{worker}-{turn}", f"a-{worker}-{turn}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write_turns, range(workers)))

    final = SessionStore(legacy_path, max_turns=1_000).history("s_shared", system=None)
    assert len(final) == workers * turns_per_worker * 2
    assert len({message.content for message in final}) == len(final)


def test_provider_log_repository_tracks_rolling_and_total_usage(tmp_path):
    logs = ProviderLogRepository(Database(tmp_path / "jarvis.db"))
    logs.record_completion("ollama", tokens=9)
    usage = logs.usage("ollama")

    assert usage.requests_minute == usage.requests_day == usage.total_requests == 1
    assert usage.tokens_minute == usage.tokens_day == usage.total_tokens == 9


# --- migrations ------------------------------------------------------------


def test_reopening_the_database_is_idempotent(tmp_path):
    path = tmp_path / "jarvis.db"
    Database(path)

    reopened = Database(path)
    assert reopened.schema_version() == 1
    with reopened.read() as conn:
        applied = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert applied == [1]


def test_appending_a_migration_upgrades_in_place_and_preserves_data(tmp_path):
    path = tmp_path / "jarvis.db"
    ConversationRepository(Database(path)).append_turn("c_1", "hi", "hello")
    upgraded_migrations = MIGRATIONS + (Migration(2, ("CREATE TABLE notes (id INTEGER PRIMARY KEY)",)),)

    upgraded = Database(path, migrations=upgraded_migrations)

    assert upgraded.schema_version() == 2
    with upgraded.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
    assert len(ConversationRepository(upgraded).history("c_1")) == 2


def test_failed_second_migration_leaves_version_one_intact(tmp_path):
    path = tmp_path / "v1_then_v2.db"
    Database(path)
    broken = MIGRATIONS + (Migration(2, ("CREATE TABLE half (id INTEGER PRIMARY KEY)", "NOT SQL AT ALL")),)

    with pytest.raises(sqlite3.OperationalError):
        Database(path, migrations=broken)

    assert Database(path).schema_version() == 1
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "half" not in tables


# --- legacy JSON compatibility --------------------------------------------


def test_corrupt_legacy_json_is_ignored_and_retried_next_open(tmp_path):
    legacy = tmp_path / "sessions.json"
    legacy.write_text("{ not json", "utf-8")

    assert SessionStore(legacy).ids() == []

    legacy.write_text(
        json.dumps({"s_ok": {"created": 1.0, "updated": 2.0,
                             "messages": [{"role": "user", "content": "recovered"}]}}),
        "utf-8",
    )
    assert [m.content for m in SessionStore(legacy).history("s_ok", system=None)] == ["recovered"]


def test_non_dict_legacy_payload_is_ignored(tmp_path):
    legacy = tmp_path / "sessions.json"
    legacy.write_text("[1, 2, 3]", "utf-8")

    assert SessionStore(legacy).ids() == []


def test_legacy_import_marker_blocks_a_second_import(tmp_path):
    legacy = tmp_path / "sessions.json"
    legacy.write_text(
        json.dumps({"s_1": {"created": 1.0, "updated": 2.0,
                            "messages": [{"role": "user", "content": "first"}]}}),
        "utf-8",
    )
    SessionStore(legacy)

    legacy.write_text(
        json.dumps({"s_1": {"created": 1.0, "updated": 9.0,
                            "messages": [{"role": "user", "content": "first"},
                                         {"role": "assistant", "content": "tampered"}]}}),
        "utf-8",
    )

    reopened = SessionStore(legacy)
    assert [m.content for m in reopened.history("s_1", system=None)] == ["first"]


# --- repositories ----------------------------------------------------------


def test_trim_keeps_newest_turns_and_clear_keeps_the_conversation(tmp_path):
    repo = ConversationRepository(Database(tmp_path / "jarvis.db"))
    for i in range(5):
        repo.append_turn("c_1", f"u{i}", f"a{i}")
    repo.upsert_summary("c_1", "early summary", through_sequence=2)

    repo.trim_to_turns("c_1", 2)
    assert [m.content for m in repo.history("c_1")] == ["u3", "a3", "u4", "a4"]

    repo.trim_to_turns("c_1", 0)
    assert repo.history("c_1") == []

    repo.append_turn("c_1", "again", "still here")
    repo.clear("c_1")
    assert repo.history("c_1") == []
    assert repo.latest_summary("c_1") is None
    assert "c_1" in repo.ids()


def test_upsert_summary_replaces_the_row_for_the_same_sequence(tmp_path):
    repo = ConversationRepository(Database(tmp_path / "jarvis.db"))
    repo.get_or_create("c_1", now=1.0)
    repo.upsert_summary("c_1", "old", through_sequence=4)
    repo.upsert_summary("c_1", "new", through_sequence=4)

    assert repo.latest_summary("c_1") == "new"
    with Database(tmp_path / "jarvis.db").read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 1


def test_read_paths_close_every_connection_they_open(tmp_path):
    class TrackingDatabase(Database):
        """Records every connection so the test can prove each one closed."""

        def __init__(self, path):
            self.created: list[sqlite3.Connection] = []
            super().__init__(path)

        def connect(self) -> sqlite3.Connection:
            conn = super().connect()
            self.created.append(conn)
            return conn

    db = TrackingDatabase(tmp_path / "jarvis.db")
    conversations = ConversationRepository(db)
    logs = ProviderLogRepository(db)
    conversations.append_turn("c_1", "hi", "hello")
    conversations.history("c_1")
    conversations.ids()
    conversations.latest_summary("c_1")
    logs.record_completion("p", tokens=1)
    logs.usage("p")
    logs.providers()
    db.schema_version()
    with db.read():
        pass

    leaked = []
    for conn in db.created:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue
        leaked.append(conn)
    for conn in leaked:
        conn.close()
    assert not leaked, f"{len(leaked)} of {len(db.created)} connections were never closed"


# --- cross-process concurrency --------------------------------------------


def test_concurrent_database_startup_never_fails_on_wal(tmp_path):
    """Racing Database() constructions must not raise "database is locked".

    ``PRAGMA journal_mode`` ignores busy_timeout during the initial WAL
    transition, so without the retry in ``Database._enable_wal`` a daemon and
    a CLI starting on the same instant could crash one of them. Reproduced
    intermittently as a dead worker hanging the barrier in the test below.
    """
    workers, rounds = 6, 8
    for round_index in range(rounds):
        path = tmp_path / f"round_{round_index}" / "jarvis.db"
        barrier = threading.Barrier(workers, timeout=15)
        errors: list[Exception] = []

        def build(_: int, *, _path=path, _barrier=barrier, _errors=errors) -> None:
            try:
                _barrier.wait()
                Database(_path)
            except Exception as exc:
                _errors.append(exc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(build, range(workers)))
        assert not errors, f"round {round_index}: {[repr(e) for e in errors]}"


def test_multi_process_writers_do_not_lose_turns(tmp_path):
    """Four real processes appending to one jarvis.db.

    The original bug was exactly this shape: the daemon and the voice CLI were
    separate processes racing on a shared file. Threads inside one interpreter
    would not have caught it, so these writers are true subprocesses.
    """
    legacy_path = tmp_path / "sessions.json"
    workers, turns_per_worker = 4, 5
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from jarvis.core.conversation import SessionStore\n"
        "path, worker, turns = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])\n"
        "store = SessionStore(path, max_turns=1_000)\n"
        "store.get_or_create('s_shared')\n"
        "for turn in range(turns):\n"
        "    store.append('s_shared', f'u-{worker}-{turn}', f'a-{worker}-{turn}')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(legacy_path), str(worker), str(turns_per_worker)],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for worker in range(workers)
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode("utf-8", "replace")

    final = SessionStore(legacy_path, max_turns=1_000).history("s_shared", system=None)
    assert len(final) == workers * turns_per_worker * 2
    assert len({message.content for message in final}) == len(final)
