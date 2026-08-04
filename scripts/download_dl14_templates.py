#!/usr/bin/env python3
"""Download DL14 dust emission templates from CIGALE's GitLab repository.

The DL14 templates extend Draine & Li 2007 with:
- Variable alpha parameter (power-law slope, 1.0-3.0)
- Extended q_PAH range (0.47-7.32%)
- Extended U_min range (0.1-50)
- U_max raised to 10^7

Source: CIGALE project (Boquien et al. 2019)
    https://gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder/dl2014

Usage:
    python scripts/download_dl14_templates.py [--output-dir data/dl14_raw]
    python scripts/download_dl14_templates.py --dry-run  # show what would be downloaded
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# -----------------------------------------------------------------------
# DL14 grid specification (from CIGALE's database_builder/dl2014/__init__.py)
# -----------------------------------------------------------------------

QPAH_MODELS = [
    "000",
    "010",
    "020",
    "030",
    "040",
    "050",
    "060",
    "070",
    "080",
    "090",
    "100",
]

UMIN_VALUES = [
    "0.100",
    "0.120",
    "0.150",
    "0.170",
    "0.200",
    "0.250",
    "0.300",
    "0.350",
    "0.400",
    "0.500",
    "0.600",
    "0.700",
    "0.800",
    "1.000",
    "1.200",
    "1.500",
    "1.700",
    "2.000",
    "2.500",
    "3.000",
    "3.500",
    "4.000",
    "5.000",
    "6.000",
    "7.000",
    "8.000",
    "10.00",
    "12.00",
    "15.00",
    "17.00",
    "20.00",
    "25.00",
    "30.00",
    "35.00",
    "40.00",
    "50.00",
]

ALPHA_VALUES = [
    "1.0",
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.6",
    "1.7",
    "1.8",
    "1.9",
    "2.0",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "2.7",
    "2.8",
    "2.9",
    "3.0",
]

GITLAB_RAW_BASE = (
    "https://gitlab.lam.fr/api/v4/projects/cigale%2Fcigale/repository/files/{path}/raw?ref=master"
)


def _url_encode_path(path: str) -> str:
    """URL-encode a file path for GitLab API (/ -> %2F)."""
    return path.replace("/", "%2F")


def _download_file(url: str, dest: str, retries: int = 3) -> bool:
    """Download a single file with retry logic."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  Failed: {e}")
                return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  Failed: {e}")
                return False
    return False


def download_templates(output_dir: str, dry_run: bool = False) -> None:
    """Download all DL14 template files from CIGALE GitLab."""
    os.makedirs(output_dir, exist_ok=True)

    # Count total files
    n_single = len(QPAH_MODELS) * len(UMIN_VALUES)  # 396
    n_powerlaw = len(QPAH_MODELS) * len(UMIN_VALUES) * len(ALPHA_VALUES)  # 8316
    total = n_single + n_powerlaw
    print(f"DL14 template grid:")
    print(f"  q_PAH models: {len(QPAH_MODELS)}")
    print(f"  U_min values: {len(UMIN_VALUES)}")
    print(f"  alpha values: {len(ALPHA_VALUES)}")
    print(f"  Single-U files: {n_single}")
    print(f"  Power-law files: {n_powerlaw}")
    print(f"  Total files: {total}")

    if dry_run:
        print("\n[dry-run] Would download to:", output_dir)
        # Show a few example paths
        for model in QPAH_MODELS[:2]:
            umin = UMIN_VALUES[0]
            print(f"  U{umin}_{umin}_MW3.1_{model}/spec_1.0.dat")
            print(f"  U{umin}_1e7_MW3.1_{model}/spec_1.0.dat")
            print(f"  U{umin}_1e7_MW3.1_{model}/spec_2.0.dat")
        return

    downloaded = 0
    skipped = 0
    failed = 0

    for im, model in enumerate(QPAH_MODELS):
        for iu, umin in enumerate(UMIN_VALUES):
            # --- Single-U template ---
            single_dir_name = f"U{umin}_{umin}_MW3.1_{model}"
            single_fname = "spec_1.0.dat"
            remote_path = f"database_builder/dl2014/data/{single_dir_name}/{single_fname}"

            local_dir = os.path.join(output_dir, single_dir_name)
            local_path = os.path.join(local_dir, single_fname)

            if os.path.exists(local_path):
                skipped += 1
            else:
                os.makedirs(local_dir, exist_ok=True)
                url = GITLAB_RAW_BASE.format(path=_url_encode_path(remote_path))
                ok = _download_file(url, local_path)
                if ok:
                    downloaded += 1
                else:
                    failed += 1

            # --- Power-law templates (one per alpha) ---
            pl_dir_name = f"U{umin}_1e7_MW3.1_{model}"
            pl_local_dir = os.path.join(output_dir, pl_dir_name)

            for alpha in ALPHA_VALUES:
                pl_fname = f"spec_{alpha}.dat"
                remote_path = f"database_builder/dl2014/data/{pl_dir_name}/{pl_fname}"
                local_path = os.path.join(pl_local_dir, pl_fname)

                if os.path.exists(local_path):
                    skipped += 1
                else:
                    os.makedirs(pl_local_dir, exist_ok=True)
                    url = GITLAB_RAW_BASE.format(path=_url_encode_path(remote_path))
                    ok = _download_file(url, local_path)
                    if ok:
                        downloaded += 1
                    else:
                        failed += 1

            # Progress
            done = im * len(UMIN_VALUES) + iu + 1
            frac = done / (len(QPAH_MODELS) * len(UMIN_VALUES))
            print(
                f"\r  [{frac:.0%}] model={model} umin={umin} "
                f"(downloaded={downloaded}, skipped={skipped}, failed={failed})",
                end="",
                flush=True,
            )

    print(f"\n\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download DL14 dust emission templates from CIGALE GitLab"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/dl14_raw/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        args.output_dir = str(repo_root / "data" / "dl14_raw")

    download_templates(args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
