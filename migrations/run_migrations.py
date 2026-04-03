"""Database migration runner. Executes .sql files in order, tracking applied migrations."""

import argparse
import sys
from pathlib import Path

import psycopg2


MIGRATIONS_DIR = Path(__file__).parent


def get_connection(db_url: str):
    return psycopg2.connect(db_url)


def ensure_migrations_table(conn):
    """Create the schema_migrations tracking table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    conn.commit()


def get_applied_migrations(conn) -> set[str]:
    """Return set of already-applied migration filenames."""
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def get_pending_migrations(applied: set[str]) -> list[Path]:
    """Return sorted list of .sql files not yet applied."""
    all_sql = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return [f for f in all_sql if f.name not in applied]


def apply_migration(conn, migration_file: Path):
    """Apply a single migration file in a transaction."""
    sql = migration_file.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)",
            (migration_file.name,),
        )
    conn.commit()


def run_migrations(db_url: str) -> int:
    """Run all pending migrations. Returns count of migrations applied."""
    conn = get_connection(db_url)
    try:
        ensure_migrations_table(conn)
        applied = get_applied_migrations(conn)
        pending = get_pending_migrations(applied)

        if not pending:
            print("No pending migrations.")
            return 0

        for migration in pending:
            print(f"Applying {migration.name}...")
            try:
                apply_migration(conn, migration)
                print(f"  OK")
            except Exception as e:
                conn.rollback()
                print(f"  FAILED: {e}")
                raise

        print(f"\n{len(pending)} migration(s) applied successfully.")
        return len(pending)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--db-url",
        default="postgresql://aegis:aegis@localhost:5432/aegis",
        help="PostgreSQL connection URL",
    )
    args = parser.parse_args()

    try:
        run_migrations(args.db_url)
    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
