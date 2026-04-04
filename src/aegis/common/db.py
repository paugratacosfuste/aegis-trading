"""PostgreSQL connection pool and query helpers."""

from contextlib import contextmanager
from typing import Any

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from aegis.common.exceptions import DatabaseError


class DatabasePool:
    """Thread-safe PostgreSQL connection pool."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aegis",
        user: str = "aegis",
        password: str = "aegis",
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        self._pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DatabasePool":
        """Create pool from a database config dict (Settings.database)."""
        return cls(
            host=config["host"],
            port=config["port"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            min_connections=config.get("min_connections", 2),
            max_connections=config.get("max_connections", 10),
        )

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool. Returns it on exit (even on error)."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def execute(self, sql: str, params: tuple | None = None) -> None:
        """Execute a write query (INSERT, UPDATE, DELETE) and commit."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a write query for multiple param sets and commit."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
            conn.commit()

    def fetch_one(self, sql: str, params: tuple | None = None) -> dict | None:
        """Execute a read query and return one row as a dict."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return dict(row) if row is not None else None

    def fetch_all(self, sql: str, params: tuple | None = None) -> list[dict]:
        """Execute a read query and return all rows as dicts."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()
