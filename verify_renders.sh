#!/bin/bash
# Hard verification contract for regenerated renders

echo "=== HARD VERIFICATION CONTRACT ==="
echo ""

# Check 1: No embedded tracebacks
echo "[1/6] Zero embedded tracebacks in .rst files..."
TRACEBACK_COUNT=$(grep -rl "Traceback (most recent call last)" docs/auto_examples --include="*.rst" 2>/dev/null | wc -l)
echo "  Traceback count: $TRACEBACK_COUNT"
if [ "$TRACEBACK_COUNT" -eq 0 ]; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL: Found $TRACEBACK_COUNT .rst files with tracebacks"
fi
echo ""

# Check 2: No site-packages paths
echo "[2/6] Zero site-packages strings in docs/auto_examples..."
SITEPKG_COUNT=$(grep -rl "site-packages" docs/auto_examples 2>/dev/null | wc -l)
echo "  Site-packages count: $SITEPKG_COUNT"
if [ "$SITEPKG_COUNT" -eq 0 ]; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL: Found $SITEPKG_COUNT files with site-packages paths"
fi
echo ""

# Check 3: check_no_local_paths
echo "[3/6] check_no_local_paths.py..."
export PYTHONPATH=/Users/suchethacooray/Projects/tengri/.claude/worktrees/gallery-overhaul/src
/Users/suchethacooray/Projects/tengri/.venv/bin/python tools/check_no_local_paths.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL"
fi
echo ""

# Check 4: check_gallery_fresh --strict
echo "[4/6] check_gallery_fresh.py --strict..."
/Users/suchethacooray/Projects/tengri/.venv/bin/python tools/check_gallery_fresh.py --strict > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL"
fi
echo ""

# Check 5: check_example_silent_failure
echo "[5/6] check_example_silent_failure.py..."
/Users/suchethacooray/Projects/tengri/.venv/bin/python tools/check_example_silent_failure.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL"
fi
echo ""

# Check 6: Example count (all 121 must have figures)
echo "[6/6] Verify all 121 examples have figures..."
# Count unique examples from image files
FIGURE_COUNT=$(find docs/auto_examples -name "sphx_glr_plot_*.png" ! -path "*/thumb/*" -type f 2>/dev/null | sed 's|.*/||' | sed 's/sphx_glr_\(.*\)_[0-9][0-9]*\.png/\1/' | sort -u | wc -l)
echo "  Figures found: $FIGURE_COUNT"
if [ "$FIGURE_COUNT" -ge 121 ]; then
    echo "  ✓ PASS: Found $FIGURE_COUNT examples with figures"
else
    echo "  ⚠ WARNING: Only found $FIGURE_COUNT examples with figures (expected 121)"
fi
echo ""

echo "=== VERIFICATION COMPLETE ==="
