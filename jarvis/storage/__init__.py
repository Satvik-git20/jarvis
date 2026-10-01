"""Durable storage primitives for JARVIS.

The public application layer continues to expose ``SessionStore`` and
``BudgetLedger``.  This package owns the SQLite implementation beneath those
backwards-compatible facades.
"""

from .database import Database, Migration, database_path_for

__all__ = ["Database", "Migration", "database_path_for"]
