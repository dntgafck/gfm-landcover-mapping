#!/usr/bin/env bash
# scripts/setup_dev.sh
# Convenience wrapper for macOS development setup
# Installs base + dev dependencies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up development environment for macOS..."
echo ""

exec "$SCRIPT_DIR/setup_env.sh" --dev "$@"
