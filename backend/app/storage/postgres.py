"""PostgreSQL schema migrations for SARAL's domain tables."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import atexit
import os
import threading
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool


DATABASE_URL_VARIABLES = ("SARAL_DATABASE_URL", "DATABASE_URL_UNPOOLED", "DATABASE_URL")


def database_url_from_env() -> str | None:
    """The first database URL set in the environment, or None."""
    for name in DATABASE_URL_VARIABLES:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


_POOLS: dict[str, ConnectionPool] = {}
_POOLS_LOCK = threading.Lock()


def pooled_connection(database_url: str):
    """A connection from one shared pool per database, as a context manager that commits on
    success and rolls back on error, like `psycopg.connect`."""
    pool = _POOLS.get(database_url)
    if pool is None:
        with _POOLS_LOCK:
            pool = _POOLS.get(database_url)
            if pool is None:
                _validate_database_url(database_url)
                pool = ConnectionPool(
                    database_url, min_size=0, max_size=8, max_idle=240, open=True,
                    check=ConnectionPool.check_connection, name="saral-storage",
                )
                _POOLS[database_url] = pool
    return pool.connection()


@atexit.register
def _close_pools() -> None:
    for pool in _POOLS.values():
        pool.close()


DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    sql: str
    checksum: str


def discover_migrations(path: Path = DEFAULT_MIGRATIONS_DIR) -> tuple[Migration, ...]:
    migrations = []
    for migration_path in sorted(path.glob("*.sql")):
        sql = migration_path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=migration_path.stem,
                sql=sql,
                checksum=sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    if not migrations:
        raise RuntimeError(f"no PostgreSQL migrations found in {path}")
    return tuple(migrations)


def apply_domain_migrations(database_url: str) -> tuple[str, ...]:
    """Apply immutable domain migrations and return newly applied versions."""
    _validate_database_url(database_url)
    applied: list[str] = []
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS saral_schema_migrations (
                version text PRIMARY KEY,
                checksum text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext('saral_domain_migrations'))"
        )
        for migration in discover_migrations():
            existing = connection.execute(
                "SELECT checksum FROM saral_schema_migrations WHERE version = %s",
                (migration.version,),
            ).fetchone()
            if existing:
                if existing[0] != migration.checksum:
                    raise RuntimeError(
                        f"migration {migration.version} changed after it was applied"
                    )
                continue
            connection.execute(migration.sql)
            connection.execute(
                "INSERT INTO saral_schema_migrations (version, checksum) VALUES (%s, %s)",
                (migration.version, migration.checksum),
            )
            applied.append(migration.version)
    return tuple(applied)


def _validate_database_url(database_url: str) -> None:
    if not database_url or not database_url.startswith(("postgresql://", "postgres://")):
        raise ValueError("SARAL_DATABASE_URL must be a PostgreSQL connection URL")
