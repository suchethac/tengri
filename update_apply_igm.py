#!/usr/bin/env python3
"""Update apply_igm references in test files."""
import re
import sys
from pathlib import Path

def update_file(filepath):
    """Update apply_igm references in a file."""
    content = Path(filepath).read_text()
    original = content

    # Pattern 1: apply_igm=False, -> igm={"type": "none"},
    content = re.sub(
        r'apply_igm\s*=\s*False\s*,',
        'igm={"type": "none"},',
        content
    )

    # Pattern 2: apply_igm=False) at end of dict
    content = re.sub(
        r'apply_igm\s*=\s*False\s*\)',
        'igm={"type": "none"})',
        content
    )

    # Pattern 3: apply_igm=True, followed by igm= - just remove the apply_igm line
    content = re.sub(
        r'\s*apply_igm\s*=\s*True\s*,\s*(?=igm\s*=)',
        '',
        content
    )

    # Pattern 4: apply_igm=True) at end - add igm dict
    content = re.sub(
        r'apply_igm\s*=\s*True\s*\)',
        'igm={"type": "inoue"})',
        content
    )

    if content != original:
        Path(filepath).write_text(content)
        return True
    return False

if __name__ == '__main__':
    test_files = list(Path('tests/contract').glob('test_*.py'))
    updated = []
    for f in sorted(test_files):
        if update_file(str(f)):
            updated.append(str(f))
            print(f"Updated: {f}")

    if not updated:
        print("No files updated")
