#!/usr/bin/env python3
"""Analyze a tengri compile event log.

Usage:
    python scripts/analyze_compile_log.py [--log PATH]

Reads a JSON lines compile log (default ~/.cache/tengri_jax_cache/compile.log)
and prints:
  - Total event count and aggregate wall time
  - Per-method breakdown (count, total, mean, max)
  - Consecutive events with differing signatures (spurious recompiles)
  - Inferred cache-hit ratio

Requires only stdlib: json, argparse, statistics, collections.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_log(path: Path) -> list[dict]:
    """Load JSON lines compile log.

    Parameters
    ----------
    path : Path
        Path to compile log file.

    Returns
    -------
    list[dict]
        List of parsed JSON objects, in order.

    Raises
    ------
    FileNotFoundError
        If the log file does not exist.
    json.JSONDecodeError
        If a line cannot be parsed as JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Compile log not found: {path}")

    events = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Failed to parse line {lineno}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
    return events


def analyze_events(events: list[dict]) -> dict:
    """Compute aggregate statistics from compile events.

    Parameters
    ----------
    events : list[dict]
        Parsed compile events.

    Returns
    -------
    dict
        Dictionary with keys:
        - total_events: int
        - total_time_s: float
        - per_method: dict mapping method → {count, total, mean, max}
        - cache_hits: int
        - cache_misses: int
        - hit_ratio: float
        - spurious_recompiles: list of dicts describing signature changes
    """
    total_time = 0.0
    per_method = defaultdict(list)
    cache_hits = 0
    cache_misses = 0
    spurious = []

    for event in events:
        duration = event.get("duration_s", 0.0)
        method = event.get("method") or "unknown"
        is_hit = event.get("inferred_cache_hit", False)

        total_time += duration
        per_method[method].append(duration)

        if is_hit:
            cache_hits += 1
        else:
            cache_misses += 1

    # Compute per-method stats
    method_stats = {}
    for method, durations in per_method.items():
        method_stats[method] = {
            "count": len(durations),
            "total": sum(durations),
            "mean": statistics.mean(durations),
            "max": max(durations),
        }

    # Find spurious recompiles (consecutive events with different signatures)
    for i in range(len(events) - 1):
        curr = events[i]
        next_ev = events[i + 1]
        if curr.get("signature") != next_ev.get("signature"):
            spurious.append(
                {
                    "index": i,
                    "event_1": {
                        "name": curr.get("name"),
                        "method": curr.get("method"),
                        "signature": curr.get("signature")[:80] + "..."
                        if len(curr.get("signature", "")) > 80
                        else curr.get("signature"),
                    },
                    "event_2": {
                        "name": next_ev.get("name"),
                        "method": next_ev.get("method"),
                        "signature": next_ev.get("signature")[:80] + "..."
                        if len(next_ev.get("signature", "")) > 80
                        else next_ev.get("signature"),
                    },
                }
            )

    total_events_counted = cache_hits + cache_misses
    hit_ratio = cache_hits / total_events_counted if total_events_counted > 0 else 0.0

    return {
        "total_events": len(events),
        "total_time_s": total_time,
        "per_method": dict(method_stats),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_ratio": hit_ratio,
        "spurious_recompiles": spurious,
    }


def format_report(stats: dict) -> str:
    """Format analysis results as a human-readable report.

    Parameters
    ----------
    stats : dict
        Output from analyze_events().

    Returns
    -------
    str
        Formatted report text.
    """
    lines = []

    lines.append("=" * 70)
    lines.append("TENGRI COMPILE LOG ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total compile events:        {stats['total_events']}")
    lines.append(f"Total compile wall time:     {stats['total_time_s']:.2f} s")
    lines.append(f"Cache hits (inferred):       {stats['cache_hits']}")
    lines.append(f"Cache misses (inferred):     {stats['cache_misses']}")
    lines.append(f"Hit ratio:                   {stats['hit_ratio'] * 100:.1f}%")
    lines.append("")

    if stats["per_method"]:
        lines.append("PER-METHOD BREAKDOWN")
        lines.append("-" * 70)
        header = f"{'Method':<20} {'Count':>6} {'Total (s)':>12} {'Mean (s)':>12} {'Max (s)':>12}"
        lines.append(header)
        lines.append("-" * 70)
        for method, stats_dict in sorted(stats["per_method"].items()):
            row = (
                f"{method:<20} {stats_dict['count']:>6} "
                f"{stats_dict['total']:>12.2f} {stats_dict['mean']:>12.2f} "
                f"{stats_dict['max']:>12.2f}"
            )
            lines.append(row)
        lines.append("")

    if stats["spurious_recompiles"]:
        lines.append("SPURIOUS RECOMPILES (consecutive events with different signatures)")
        lines.append("-" * 70)
        for spg in stats["spurious_recompiles"][:20]:  # limit output
            idx = spg["index"]
            ev1 = spg["event_1"]
            ev2 = spg["event_2"]
            lines.append(
                f"[{idx}→{idx + 1}] {ev1['name']} ({ev1['method']}) → "
                f"{ev2['name']} ({ev2['method']})"
            )
            lines.append(f"  sig[{idx}]: {ev1['signature']}")
            lines.append(f"  sig[{idx + 1}]: {ev2['signature']}")
            lines.append("")

        if len(stats["spurious_recompiles"]) > 20:
            lines.append(f"  ... and {len(stats['spurious_recompiles']) - 20} more")
            lines.append("")
    else:
        lines.append("SPURIOUS RECOMPILES")
        lines.append("-" * 70)
        lines.append("None detected (all consecutive events share the same signature).")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze a tengri compile event log.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to compile log (default: ~/.cache/tengri_jax_cache/compile.log)",
    )

    args = parser.parse_args()

    log_file = args.log
    if log_file is None:
        log_file = Path.home() / ".cache" / "tengri_jax_cache" / "compile.log"

    try:
        events = load_log(log_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not events:
        print("No events found in log.")
        return

    stats = analyze_events(events)
    report = format_report(stats)
    print(report)


if __name__ == "__main__":
    main()
