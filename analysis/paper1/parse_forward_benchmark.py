#!/usr/bin/env python3
"""Parse the stdout of bench/scripts/benchmark_forward_model.py into the JSON read by fig03_precompute.py.

Usage:
  python analysis/paper1/parse_forward_benchmark.py --log <benchmark stdout> --out <json>

The log must start with a line ``START <date> <time> host=<h> commit=<sha> jax=<v>`` and end
with ``EXIT=0 <date> <time>`` (see REPRODUCTION_COMMANDS.md, Figure 3). Panel (a) rows are the
forward configurations of the double-power-law section; every section is kept under "all_sections".
"""
import argparse
import json
import pathlib
import platform
import re
import subprocess

PANEL_ROWS = [
    ("Stellar", "Stellar only"),
    ("+ nebular (SSP)", "+ nebular (baked-in SSP)"),
    ("+ dust IR (DL07)", "+ dust IR (DL07)"),
    ("+ AGN (QSOgen)", "+ AGN (QSOgen)"),
    ("+ AGN torus (Kubota-Done)", "+ AGN (K&D 3-zone full)"),
    ("+ radio", "+ radio (SF + AGN)"),
    ("+ X-ray", "+ X-ray (XRB + corona)"),
    ("Panchromatic (nebular, THEMIS, radio, X-ray)", "Typical: neb+THEMIS+radio+xray"),
    ("All components", "Kitchen sink (all components)"),
]
PANEL_SECTION = "Forward: DPL (parametric, D=6)"
ROW_RE = re.compile(
    r"^\s{2}(\S.*?)\s{2,}exact=\s*(\d+) µs\s+precomp=\s*(\d+) µs\s+speedup=\s*([\d.]+)×(?:\s+err=\s*([\d.]+)%)?"
)
SKIP_RE = re.compile(r"^\s{2}(\S.*?)\s{2,}SKIPPED \((.*)$")
HEAD_RE = re.compile(r"^\s+(Forward|Gradient): (.+)$")


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse(log_text: str) -> dict:
    sections: dict[str, list[dict]] = {}
    current = None
    for line in log_text.splitlines():
        head = HEAD_RE.match(line)
        if head:
            current = f"{head.group(1)}: {head.group(2).strip()}"
            sections[current] = []
            continue
        row = ROW_RE.match(line)
        if row and current is not None:
            entry = {
                "label": row.group(1).strip(),
                "exact_us": int(row.group(2)),
                "precomp_us": int(row.group(3)),
                "speedup": float(row.group(4)),
            }
            if row.group(5):
                entry["max_rel_err_pct"] = float(row.group(5))
            sections[current].append(entry)
            continue
        skip = SKIP_RE.match(line)
        if skip and current is not None:
            sections[current].append({"label": skip.group(1).strip(), "skipped": skip.group(2).rstrip(")")})
    if PANEL_SECTION not in sections:
        raise SystemExit(f"section {PANEL_SECTION!r} not found in log")
    forward = {e["label"]: e for e in sections[PANEL_SECTION] if "exact_us" in e}
    missing = [src for _, src in PANEL_ROWS if src not in forward]
    if missing:
        raise SystemExit(f"panel rows missing from the log: {missing}")
    panel = [
        {"label": label, "source_label": src, **{k: forward[src][k] for k in ("exact_us", "precomp_us", "speedup")}}
        for label, src in PANEL_ROWS
    ]
    header = dict(re.findall(r"^\s{2}(Platform|Precision|Filters|Redshift|Runs|SSP): (.+)$", log_text, re.M))
    start = re.search(r"^START (\S+ \S+) host=(\S+) commit=(\S+) jax=(\S+)", log_text, re.M)
    end = re.search(r"^EXIT=0 (\S+ \S+)", log_text, re.M)
    if start is None or end is None:
        raise SystemExit("log lacks the START/EXIT=0 stamp lines; the run is not a complete, successful run")
    other = re.search(r"^fit_procs_at_(?:start|end)=(\d+)", log_text, re.M)
    metadata = {
        "script": "bench/scripts/benchmark_forward_model.py",
        "started": start.group(1),
        "finished": end.group(1),
        "host": start.group(2),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "n_cpu": _sysctl("hw.ncpu"),
        "os": platform.platform(),
        "commit": start.group(3),
        "jax": start.group(4),
        "python": platform.python_version(),
        "platform": header.get("Platform"),
        "precision": header.get("Precision"),
        "filters": header.get("Filters"),
        "redshift": header.get("Redshift"),
        "runs": header.get("Runs"),
        "ssp": header.get("SSP"),
        "sfh_for_panel": PANEL_SECTION.split(": ", 1)[1],
        "other_fit_processes_during_run": int(other.group(1)) if other else None,
    }
    return {"metadata": metadata, "panel_a": panel, "all_sections": sections}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    result = parse(args.log.read_text())
    result["metadata"]["log"] = str(args.log)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for row in result["panel_a"]:
        print(f"{row['label']:<46} exact={row['exact_us']:>6} µs  precomp={row['precomp_us']:>6} µs  {row['speedup']}x")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
