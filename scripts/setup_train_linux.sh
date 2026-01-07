#!/usr/bin/env bash
# scripts/setup_train_linux.sh
# Convenience wrapper for Linux training setup
# Installs base + training + GPU dependencies
# Typically run with --frozen flag for reproducible remote training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up training environment for Linux..."
echo ""

exec "$SCRIPT_DIR/setup_env.sh" --train "$@"
