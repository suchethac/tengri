#!/usr/bin/env python3
"""Download all key photometric filters from the SVO Filter Profile Service.

Fetches every filter in ``tengri.observation.filters.FILTER_REGISTRY``
and caches them as two-column text files under ``data/filters/``.  Subsequent
calls to ``load_filter()`` / ``load_filter_set()`` will use the cache and
never hit the network.

Source: Spanish Virtual Observatory (SVO) Filter Profile Service
    https://svo2.cab.inta-csic.es/theory/fps/

Usage
-----
    python scripts/download_filters.py                  # download all
    python scripts/download_filters.py --dry-run        # show what would be fetched
    python scripts/download_filters.py --filter sdss_g  # single filter
    python scripts/download_filters.py --force          # re-download cached files
    python scripts/download_filters.py --cache-dir /tmp/filters  # custom cache dir
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline helpers (avoid importing tengri so the script works in bare envs)
# ---------------------------------------------------------------------------

_SVO_BASE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
_VOT_NS = "{http://www.ivoa.net/xml/VOTable/v1.2}"

#: The single filter registry, read from the package data file.
#:
#: This was a hand-maintained copy of ``FILTER_REGISTRY``, carrying a "keep in
#: sync" comment and 250 of the registry's 431 entries -- ALHAMBRA, J-PAS,
#: J-PLUS, SHARDS, HAWK-I and SkyMapper were all absent, so the script this
#: loader's own offline error message recommends could not download them. A
#: comment cannot fail a build, so nothing reported the drift.
#:
#: Reading the JSON keeps the constraint the copy existed for. The comment
#: above says "avoid importing tengri so the script works in bare envs", and
#: that still holds: this is ``json`` from the standard library reading a data
#: file, not an import of the package.
_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "tengri"
    / "observation"
    / "data"
    / "filters_registry.json"
)

if not _REGISTRY_PATH.is_file():
    raise SystemExit(
        f"Filter registry not found at {_REGISTRY_PATH}.\n"
        "This script reads the registry from the package data file and must be "
        "run from a tengri source checkout."
    )

#: ``short alias -> SVO filter identifier``. Synthetic bands (ALMA top-hats,
#: X-ray energy ranges) are not in this file at all -- they are constructed in
#: tengri.observation.filters.synthetic and have nothing to fetch -- so no
#: exclusion step is needed here.
_FILTER_REGISTRY: dict[str, str] = json.loads(_REGISTRY_PATH.read_text())


def _svo_id_to_filename(svo_id: str) -> str:
    return svo_id.replace("/", "_").replace(".", "_") + ".dat"


def _parse_votable(xml_bytes: bytes) -> tuple[list[float], list[float]]:
    root = ET.fromstring(xml_bytes)
    rows = None
    for ns in [_VOT_NS, "{http://www.ivoa.net/xml/VOTable/v1.1}", ""]:
        tabledata = root.find(f".//{ns}TABLEDATA")
        if tabledata is not None:
            rows = tabledata.findall(f"{ns}TR")
            if rows:
                break

    if not rows:
        raise ValueError("No TABLEDATA rows in SVO VOTable response — filter ID may be invalid.")

    wave_list: list[float] = []
    trans_list: list[float] = []
    for row in rows:
        cells = row.findall(f"{_VOT_NS}TD")
        if not cells:
            cells = row.findall("{http://www.ivoa.net/xml/VOTable/v1.1}TD")
        if not cells:
            cells = row.findall("TD")
        if len(cells) >= 2:
            wave_list.append(float(cells[0].text))
            trans_list.append(float(cells[1].text))

    if not wave_list:
        raise ValueError("Zero data points parsed from SVO VOTable response.")

    return wave_list, trans_list


def _download_one(svo_id: str, dest: Path, retries: int = 3) -> str:
    """Fetch one filter from SVO and write to *dest*.

    Returns
    -------
    str
        ``"ok"``, ``"cached"``, or an error message.
    """
    if dest.exists():
        return "cached"

    url = f"{_SVO_BASE_URL}?ID={svo_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "tengri/1.0"})

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                xml_bytes = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(1)
            else:
                return f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return str(exc)

    try:
        wave, trans = _parse_votable(xml_bytes)
    except ValueError as exc:
        return str(exc)

    # Sort by wavelength
    pairs = sorted(zip(wave, trans))
    wave_sorted = [p[0] for p in pairs]
    trans_sorted = [p[1] for p in pairs]

    header = "# Wavelength(Angstrom)  Transmission"
    with open(dest, "w") as fh:
        fh.write(header + "\n")
        for w, t in zip(wave_sorted, trans_sorted):
            fh.write(f"{w:.6e}  {t:.6e}\n")

    return "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    names: list[str],
    cache_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    # De-duplicate SVO IDs (e.g. johnson_j and 2mass_j share one SVO ID)
    seen_svo: dict[str, str] = {}  # svo_id -> first short name
    tasks: list[tuple[str, str]] = []  # (short_name, svo_id), deduplicated
    for name in names:
        svo_id = _FILTER_REGISTRY[name]
        if svo_id not in seen_svo:
            seen_svo[svo_id] = name
            tasks.append((name, svo_id))
        # else: alias — will use the cached file from the first occurrence

    print(f"Filters to fetch : {len(tasks)} unique SVO IDs ({len(names)} names requested)")
    print(f"Cache directory  : {cache_dir}")
    print(f"Force re-download: {force}")
    print()

    if dry_run:
        print("DRY RUN — nothing will be written.\n")
        col_w = max(len(n) for n, _ in tasks) + 2
        print(f"{'Short name':<{col_w}} SVO ID")
        print("-" * (col_w + 35))
        for name, svo_id in tasks:
            fname = _svo_id_to_filename(svo_id)
            dest = cache_dir / fname
            status = "cached" if dest.exists() else "would download"
            print(f"{name:<{col_w}} {svo_id:<35s}  [{status}]")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for i, (name, svo_id) in enumerate(tasks, 1):
        fname = _svo_id_to_filename(svo_id)
        dest = cache_dir / fname

        if force and dest.exists():
            dest.unlink()

        status = _download_one(svo_id, dest)

        if status == "ok":
            downloaded += 1
            tag = "downloaded"
        elif status == "cached":
            skipped += 1
            tag = "cached   "
        else:
            failed.append((name, status))
            tag = "FAILED   "

        frac = i / len(tasks)
        bar = "#" * int(frac * 20) + "." * (20 - int(frac * 20))
        print(
            f"\r[{bar}] {i}/{len(tasks)}  {tag}  {name:<20s}",
            end="",
            flush=True,
        )

    print()  # newline after progress bar
    print()
    print(f"Downloaded : {downloaded}")
    print(f"Cached     : {skipped}")
    print(f"Failed     : {len(failed)}")

    if failed:
        print("\nFailed filters:")
        for name, reason in failed:
            print(f"  {name:<20s} {reason}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download key photometric filters from the SVO Filter Profile Service."
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        metavar="NAME",
        nargs="+",
        help=(
            "Short name(s) to download (e.g. sdss_g jwst_f200w). "
            "Defaults to all filters in the registry."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: data/filters/ relative to repo root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file is already cached.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without fetching anything.",
    )
    args = parser.parse_args()

    if args.filters:
        unknown = [n for n in args.filters if n not in _FILTER_REGISTRY]
        if unknown:
            print(f"Unknown filter name(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(_FILTER_REGISTRY))}")
            sys.exit(1)
        names = args.filters
    else:
        names = list(_FILTER_REGISTRY.keys())

    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        cache_dir = repo_root / "data" / "filters"

    run(names, cache_dir, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
