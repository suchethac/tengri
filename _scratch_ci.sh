#!/bin/zsh
# Wait for CI on PR #1766 to settle, then print the verdict table.
while true; do
  pending=$(gh pr checks 1766 --json bucket --jq 'map(select(.bucket=="pending")) | length' 2>/dev/null)
  if [ "$pending" = "0" ]; then
    break
  fi
  sleep 45
done
gh pr checks 1766 --json name,bucket --jq '.[] | select(.bucket!="skipping") | "\(.bucket)\t\(.name)"' | sort
