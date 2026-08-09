#!/usr/bin/env bash
# Install python_total_oom_guard.sh as a macOS LaunchAgent so it is always
# running: started at login, and restarted by launchd if it ever dies.
#
# Why this exists: a `nohup ... &` daemon is invisible infrastructure. It does
# not survive logout or reboot, nothing restarts it if it dies, and its /tmp
# log is periodically reaped — so there is no evidence it was ever gone. On
# 2026-08-09 a machine reached ~120 GB of summed python RSS with the guard
# installed but dead for weeks; the absence was only noticed because the log
# file was missing. An always-on supervisor removes that whole failure class.
#
# Usage:
#   scripts/install_oom_guard_agent.sh            # install and start
#   scripts/install_oom_guard_agent.sh --uninstall
#
# Config is baked into the plist at install time; re-run to change it.
#   TOTAL_LIMIT_GB=32 AVAIL_PCT_MIN=15 scripts/install_oom_guard_agent.sh

set -euo pipefail

LABEL="com.tengri.oomguard"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
GUARD_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/python_total_oom_guard.sh"
GUARD_DST="$HOME/.local/bin/python_total_oom_guard.sh"
LOG_DIR="$HOME/Library/Logs"
LOG="$LOG_DIR/tengri-oomguard.log"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "error: this installer is macOS-only (launchd). On Linux use a systemd user unit." >&2
    exit 64
fi

uninstall() {
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "uninstalled ${LABEL} (removed $PLIST)"
    echo "the guard script remains at $GUARD_DST — delete it by hand if you want it gone"
}

if [[ "${1:-}" == "--uninstall" ]]; then
    uninstall
    exit 0
fi

# Tunables. Defaults chosen for a machine whose memory lands in MANY python
# processes rather than a few fat ones: no MIN_KILL floor, and the pressure
# triggers doing the real work, since a sum-RSS limit can sit above what the
# workload ever reaches while the box thrashes.
TOTAL_LIMIT_GB="${TOTAL_LIMIT_GB:-32}"
AVAIL_PCT_MIN="${AVAIL_PCT_MIN:-10}"
AVAIL_PCT_SOFT="${AVAIL_PCT_SOFT:-20}"
# NOT off. The claim that macOS "grows its swap file on demand and never
# shrinks it" -- which is why this used to be pinned to 0 -- is false: measured
# 2026-08-10, swap went 8.47 -> 43.23 -> 8.47 GB across one incident. Pinning
# it to 0 here also silently overrode the guard's own default, so the script
# could be fixed and the installed agent still miss. Every threshold this file
# writes into the plist is an override; keep them equal to the script defaults
# unless there is a reason, or fixing one layer will not fix the machine.
SWAP_MAX_GB="${SWAP_MAX_GB:-20}"
SWAP_GROWTH_GB="${SWAP_GROWTH_GB:-3}"
SWAP_GROWTH_WINDOW_SEC="${SWAP_GROWTH_WINDOW_SEC:-120}"
SHED_GB="${SHED_GB:-8}"
PRESSURE_MIN_PYTHON_GB="${PRESSURE_MIN_PYTHON_GB:-8}"
COOLDOWN_SEC="${COOLDOWN_SEC:-60}"
INTERVAL_SEC="${INTERVAL_SEC:-2}"
MIN_KILL_MB="${MIN_KILL_MB:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$HOME/.local/bin" "$HOME/Library/LaunchAgents" "$LOG_DIR"

# Install a stable copy: the repo may live in a worktree that gets deleted.
install -m 0755 "$GUARD_SRC" "$GUARD_DST"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${GUARD_DST}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>TOTAL_LIMIT_GB</key><string>${TOTAL_LIMIT_GB}</string>
        <key>AVAIL_PCT_MIN</key><string>${AVAIL_PCT_MIN}</string>
        <key>AVAIL_PCT_SOFT</key><string>${AVAIL_PCT_SOFT}</string>
        <key>SWAP_MAX_GB</key><string>${SWAP_MAX_GB}</string>
        <key>SWAP_GROWTH_GB</key><string>${SWAP_GROWTH_GB}</string>
        <key>SWAP_GROWTH_WINDOW_SEC</key><string>${SWAP_GROWTH_WINDOW_SEC}</string>
        <key>SHED_GB</key><string>${SHED_GB}</string>
        <key>PRESSURE_MIN_PYTHON_GB</key><string>${PRESSURE_MIN_PYTHON_GB}</string>
        <key>COOLDOWN_SEC</key><string>${COOLDOWN_SEC}</string>
        <key>INTERVAL_SEC</key><string>${INTERVAL_SEC}</string>
        <key>MIN_KILL_MB</key><string>${MIN_KILL_MB}</string>
        <key>DRY_RUN</key><string>${DRY_RUN}</string>
        <key>LOG</key><string>${LOG}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/tengri-oomguard.out</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/tengri-oomguard.err</string>
</dict>
</plist>
PLIST_EOF

# Replace any running instance, then start.
#
# `launchctl bootout` returns before the job is fully torn down, so an
# immediate bootstrap races it and fails with "5: Input/output error" — which
# leaves the machine with NO guard at all, the worst possible outcome for a
# reinstall. Wait for the job to actually disappear, then retry.
launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
    launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || break
    sleep 1
done

bootstrapped=0
for _ in 1 2 3 4 5; do
    if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then
        bootstrapped=1
        break
    fi
    sleep 1
done
if (( bootstrapped == 0 )); then
    echo "error: launchctl bootstrap failed after 5 attempts — the machine is UNGUARDED." >&2
    echo "       retry with: launchctl bootstrap gui/$(id -u) $PLIST" >&2
    exit 70
fi
launchctl kickstart -k "gui/$(id -u)/${LABEL}"

echo "installed ${LABEL}"
echo "  guard   : $GUARD_DST"
echo "  plist   : $PLIST"
echo "  log     : $LOG"
swap_desc="off — the box is UNGUARDED against swap thrash"
(( SWAP_MAX_GB > 0 )) && swap_desc=">${SWAP_MAX_GB}GB with avail<${AVAIL_PCT_SOFT}%"
growth_desc="off"
(( SWAP_GROWTH_GB > 0 )) && growth_desc="+${SWAP_GROWTH_GB}GB/${SWAP_GROWTH_WINDOW_SEC}s"
echo "  limits  : sum-rss>${TOTAL_LIMIT_GB}GB (weak: RSS shrinks while swapping)"
echo "            avail<${AVAIL_PCT_MIN}% hard | swap ${swap_desc} | growth ${growth_desc}"
echo "  shed    : ${SHED_GB}GB per trip, ${COOLDOWN_SEC}s cooldown, min_kill=${MIN_KILL_MB}MB, dry_run=${DRY_RUN}"
echo "  pressure trips only while python holds >= ${PRESSURE_MIN_PYTHON_GB}GB"
echo
echo "check   : launchctl print gui/$(id -u)/${LABEL} | head -20"
echo "follow  : tail -f $LOG"
echo "remove  : $0 --uninstall"
