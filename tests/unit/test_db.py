"""Tests for database connection pool and helpers. Written FIRST per TDD.

These tests use a real PostgreSQL connection (marked as integration if DB unavailable).
For unit testing the pool logic, we mock psycopg2.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestDatabasePool:
    def test_create_pool(self):
        """Pool initializes with config dict."""
        from aegis.common.db import DatabasePool

        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
                min_connections=1,
                max_connections=5,
            )
            mock_pool.assert_called_once()
            db.close()

    def test_get_connection_context_manager(self):
        """get_connection() returns and releases connections properly."""
        from aegis.common.db import DatabasePool

        mock_conn = MagicMock()
        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool_instance

            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
            )

            with db.get_connection() as conn:
                assert conn is mock_conn

            mock_pool_instance.putconn.assert_called_once_with(mock_conn)
            db.close()

    def test_get_connection_returns_on_exception(self):
        """Connection is returned to pool even if exception occurs."""
        from aegis.common.db import DatabasePool

        mock_conn = MagicMock()
        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool_instance

            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
            )

            with pytest.raises(ValueError):
                with db.get_connection() as conn:
                    raise ValueError("test error")

            mock_pool_instance.putconn.assert_called_once_with(mock_conn)
            db.close()

    def test_execute(self):
        """execute() runs SQL and commits."""
        from aegis.common.db import DatabasePool

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool_instance

            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
            )

            db.execute("INSERT INTO foo (bar) VALUES (%s)", ("baz",))
            mock_cursor.execute.assert_called_once_with(
                "INSERT INTO foo (bar) VALUES (%s)", ("baz",)
            )
            mock_conn.commit.assert_called_once()
            db.close()

    def test_fetch_all(self):
        """fetch_all() returns list of rows."""
        from aegis.common.db import DatabasePool

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(1, "a"), (2, "b")]
        mock_cursor.description = [("id",), ("name",)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool_instance

            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
            )

            rows = db.fetch_all("SELECT * FROM foo")
            assert rows == [(1, "a"), (2, "b")]
            db.close()

    def test_fetch_one(self):
        """fetch_one() returns a single row or None."""
        from aegis.common.db import DatabasePool

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, "a")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool_instance

            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
            )

            row = db.fetch_one("SELECT * FROM foo WHERE id = %s", (1,))
            assert row == (1, "a")
            db.close()

    def test_execute_many(self):
        """execute_many() runs SQL for a list of param tuples."""
        from aegis.common.db import DatabasePool

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool_instance

            db = DatabasePool(
                host="localhost",
                port=5432,
                dbname="test",
                user="test",
                password="test",
            )

            params = [("a",), ("b",), ("c",)]
            db.execute_many("INSERT INTO foo (bar) VALUES (%s)", params)
            mock_cursor.executemany.assert_called_once_with(
                "INSERT INTO foo (bar) VALUES (%s)", params
            )
            mock_conn.commit.assert_called_once()
            db.close()

    def test_from_config(self):
        """DatabasePool.from_config() creates pool from Settings.database dict."""
        from aegis.common.db import DatabasePool

        config = {
            "host": "myhost",
            "port": 5433,
            "dbname": "mydb",
            "user": "myuser",
            "password": "mypass",
            "min_connections": 3,
            "max_connections": 8,
        }
        with patch("aegis.common.db.pool.ThreadedConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            db = DatabasePool.from_config(config)
            mock_pool.assert_called_once_with(
                3,
                8,
                host="myhost",
                port=5433,
                dbname="mydb",
                user="myuser",
                password="mypass",
            )
            db.close()
