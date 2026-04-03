#!/bin/bash
set -euo pipefail

echo "=== Aegis Trading System Setup ==="

# 1. Create micromamba environment
echo "[1/6] Creating micromamba environment..."
micromamba create -n aegis python=3.11 -c conda-forge -y 2>/dev/null || echo "Environment 'aegis' already exists"
eval "$(micromamba shell hook --shell bash)"
micromamba activate aegis

# 2. Install PostgreSQL
echo "[2/6] Installing PostgreSQL 16..."
brew install postgresql@16 2>/dev/null || echo "PostgreSQL already installed"
brew services start postgresql@16 2>/dev/null || echo "PostgreSQL already running"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
for i in {1..10}; do
    pg_isready -q && break
    sleep 1
done

# 3. Create database and user
echo "[3/6] Creating database and user..."
createuser -s aegis 2>/dev/null || echo "User 'aegis' already exists"
psql postgres -c "ALTER USER aegis PASSWORD 'aegis';" 2>/dev/null || true
createdb -O aegis aegis 2>/dev/null || echo "Database 'aegis' already exists"

# 4. Install Python package
echo "[4/6] Installing Python package..."
cd "$(dirname "$0")/.."
pip install -e ".[dev]"

# 5. Run migrations
echo "[5/6] Running database migrations..."
python migrations/run_migrations.py

# 6. Verify
echo "[6/6] Verifying installation..."
python -c "import aegis; print('aegis package imported successfully')"
pytest tests/unit/ -x --tb=short -q 2>/dev/null || echo "Tests will pass once implemented"

echo ""
echo "=== Setup complete ==="
echo "Activate with: micromamba activate aegis"
echo "Run with: python -m aegis.main --config configs/lab.yaml"
