#!/bin/bash
# Run every examples/*/plot_*.py through the audit harness and summarize OFF-Y / OFF-X / FAIL.
set -u
OUT=/tmp/gallery_audit.txt
: > "$OUT"
for f in examples/*/plot_*.py; do
    echo "=== $f ===" >> "$OUT"
    .venv/bin/python scripts/_audit_examples.py "$f" 2>&1 \
        | grep -E "AUDIT|FAILED" \
        | awk '!seen[$0]++' \
        | head -8 >> "$OUT"
done
# Summary: list only problematic files
echo ""
echo "=== PROBLEMS ==="
awk '
  /^=== / { file=$0 }
  /SCRIPT FAILED/ { print file "\n  " $0 }
  /OFF-Y|OFF-X/   { print file "\n  " $0 }
' "$OUT"
