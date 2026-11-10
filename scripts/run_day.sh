#!/usr/bin/env bash
# Linux/macOS wrapper. On Windows use:  python scripts\run_day.py <day>
exec python3 "$(dirname "$0")/run_day.py" "$@"
