#!/usr/bin/env python3
"""Build a Draine, Li, Hensley et al. 2021 PAHspec HDF5 grid.

Downloads the canonical PAHspec ASCII tarballs from B.T. Draine's Princeton
site and repacks them into a single HDF5 file for fast loading by tengri's
``Draine2021PAHSEDComponent``.

Source
------
    https://www.astro.princeton.edu/~draine/PAHspec/<config>.tgz
    Mirror: https://doi.org/10.7910/DVN/LPUHIQ

Reference
---------
    Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K., Sandstrom, K.,
    Smith, J.-D.T. 2021, ApJ, 917, 3 (arXiv:2011.07046).

Each tarball contains 135 ASCII spectra named
``pahspec.out_<isrf>_<lgU>_<ion>_<size>.gz`` covering the 15 lgU x 3 ion
x 3 size grid for one starlight spectrum.  The ASCII files have a 7-line
header followed by 5 columns:

    wave [um]    nu*P_nu_total    nu*P_nu_Astrodust    nu*P_nu_PAH+
    nu*P_nu_PAH0  (all in erg s^-1 H^-1)

Usage
-----
    python scripts/build_pahspec_hdf5.py \\
        --raw-dir ~/.cache/tengri/pahspec_raw \\
        --output  data/pahspec_draine2021.h5 \\
        --download

    # Smoke fixture (single starlight, all axes; ~6 MB):
    python scripts/build_pahspec_hdf5.py \\
        --raw-dir ~/.cache/tengri/pahspec_raw \\
        --output  tests/fixtures/pahspec_smoke.h5 \\
        --starlights mMMP \\
        --no-slab

The script is idempotent: existing tarballs are reused unless
``--force-download`` is given.
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Grid axes
# ---------------------------------------------------------------------------

LGU_VALUES: tuple[float, ...] = tuple(0.5 * i for i in range(15))  # 0.0..7.0
ION_LEVELS: tuple[str, ...] = ("lo", "st", "hi")
SIZE_DISTS: tuple[str, ...] = ("sma", "std", "lrg")


# Starlight spectra exposed by the PAHspec library.  Names match the
# tarball filenames (without ``.tgz``).  The ASCII files inside use a
# slightly different ISRF tag (the ``isrf_tag`` below).
@dataclass(frozen=True)
class StarlightSpec:
    name: str  # tarball name (e.g. "BC03_Z0.02_3Myr")
    isrf_tag: str  # token used inside pahspec.out_<isrf_tag>_... filenames


STARLIGHTS: tuple[StarlightSpec, ...] = (
    # ``isrf_tag`` is the token used inside ``pahspec.out_<isrf_tag>_*``
    # filenames within each tarball.  The convention differs from the
    # tarball name: lowercase family + metallicity, age in years with
    # e-notation (3e6 = 3 Myr, 1e9 = 1 Gyr).
    StarlightSpec("mMMP", "mmpisrf"),
    StarlightSpec("m31bulge", "m31blge"),
    StarlightSpec("BC03_Z0.0004_10Myr", "bc03_z0.0004_1e7"),
    StarlightSpec("BC03_Z0.02_3Myr", "bc03_z0.02_3e6"),
    StarlightSpec("BC03_Z0.02_10Myr", "bc03_z0.02_1e7"),
    StarlightSpec("BC03_Z0.02_100Myr", "bc03_z0.02_1e8"),
    StarlightSpec("BC03_Z0.02_300Myr", "bc03_z0.02_3e8"),
    StarlightSpec("BC03_Z0.02_1Gyr", "bc03_z0.02_1e9"),
    StarlightSpec("BPASS_Z0.001_10Myr", "bpass_z0.001_1e7"),
    StarlightSpec("BPASS_Z0.02_3Myr", "bpass_z0.02_3e6"),
    StarlightSpec("BPASS_Z0.02_10Myr", "bpass_z0.02_1e7"),
    StarlightSpec("BPASS_Z0.02_100Myr", "bpass_z0.02_1e8"),
    StarlightSpec("BPASS_Z0.02_300Myr", "bpass_z0.02_3e8"),
    StarlightSpec("BPASS_Z0.02_1Gyr", "bpass_z0.02_1e9"),
)

PAHSPEC_BASE_URL = "https://www.astro.princeton.edu/~draine/PAHspec"


# ---------------------------------------------------------------------------
# ASCII parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PAHSpec:
    """Parsed contents of a single ``pahspec.out_*`` file."""

    wavelength_um: np.ndarray  # (n_wave,)
    nu_pnu_total: np.ndarray  # (n_wave,) erg/s/H
    nu_pnu_astrodust: np.ndarray
    nu_pnu_pah_plus: np.ndarray
    nu_pnu_pah_neutral: np.ndarray
    tir_total: float  # erg/s/H
    radfld: str  # ISRF tag from header


def parse_pahspec_ascii(text: str) -> PAHSpec:
    """Parse a Draine 2021 ``pahspec.out_*`` ASCII string.

    Parameters
    ----------
    text : str
        The full (already-decompressed) file contents.

    Returns
    -------
    PAHSpec
        Frozen dataclass with arrays of length ``n_wave``.

    Raises
    ------
    ValueError
        If the header is malformed or any column has a non-finite value.
    """
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty ASCII file")

    radfld = ""
    tir_total: float | None = None
    data_start: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("RADFLD="):
            radfld = stripped.split("=", 1)[1].strip()
        elif "total TIR power" in stripped:
            tir_total = float(stripped.split("=", 1)[0].strip())
        elif stripped.startswith("(um)"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("could not find data section ('(um)' header line)")
    if tir_total is None:
        raise ValueError("missing 'total TIR power' header line")

    rows: list[list[float]] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"expected 5 columns, got {len(parts)}: {stripped!r}")
        rows.append([float(p) for p in parts])

    if not rows:
        raise ValueError("no data rows found")

    arr = np.asarray(rows, dtype=np.float64)
    wave_um = arr[:, 0]
    if not np.all(np.diff(wave_um) > 0):
        raise ValueError("wavelength column is not strictly increasing")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in spectrum")

    return PAHSpec(
        wavelength_um=wave_um,
        nu_pnu_total=arr[:, 1],
        nu_pnu_astrodust=arr[:, 2],
        nu_pnu_pah_plus=arr[:, 3],
        nu_pnu_pah_neutral=arr[:, 4],
        tir_total=tir_total,
        radfld=radfld,
    )


def _read_pahspec_file(path: Path) -> PAHSpec:
    """Read a ``pahspec.out_*`` file (gzipped or plain)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return parse_pahspec_ascii(f.read())
    return parse_pahspec_ascii(path.read_text())


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------


def _lgU_token(lgU: float) -> str:
    """Return the Draine-style lgU token used in pahspec.out_* filenames.

    Examples: 0.0 -> "0.00", 1.5 -> "1.50", 7.0 -> "7.00".
    """
    return f"{lgU:.2f}"


def pahspec_filename(
    starlight: StarlightSpec, lgU: float, ion: str, size: str, *, slab: bool
) -> str:
    """Build the canonical filename inside a Draine PAHspec tarball.

    Non-slab convention:
        ``pahspec.out_<isrf>_<lgU>_<ion>_<size>.gz``

    Slab convention (A_V=2):
        ``pahspec.out_<isrf>_<lgU>_slab_<ion>_<size>.gz``
    """
    slab_token = "_slab" if slab else ""
    return f"pahspec.out_{starlight.isrf_tag}_{_lgU_token(lgU)}{slab_token}_{ion}_{size}.gz"


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------


def _expected_size_via_head(url: str) -> int | None:
    """Return Content-Length from HEAD or None if unavailable."""
    try:
        result = subprocess.run(
            ["curl", "-sIL", url],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def _curl_download(url: str, target: Path, max_attempts: int = 5) -> None:
    """Download ``url`` -> ``target`` using curl with resume + retry.

    Verifies the final file size against the server's Content-Length
    when available; otherwise validates the tarball by attempting a
    short ``tar -tzf`` listing.
    """
    expected = _expected_size_via_head(url)
    for attempt in range(1, max_attempts + 1):
        if target.exists() and expected is not None and target.stat().st_size == expected:
            return
        cmd = [
            "curl",
            "-fL",  # follow redirects, fail on 4xx/5xx
            "--retry",
            "5",
            "--retry-connrefused",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "-C",
            "-",  # resume partial download
            "-o",
            str(target),
            url,
        ]
        print(f"[download] (attempt {attempt}/{max_attempts}) {url}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] curl exit={exc.returncode}; retrying after 5s")
            continue
        # Validate.
        if expected is not None and target.stat().st_size != expected:
            print(
                f"[warn] size mismatch: got {target.stat().st_size}, expected {expected}; retrying"
            )
            continue
        # Final check: tarball must be openable end-to-end.
        try:
            with tarfile.open(target, "r:gz") as tf:
                tf.next()
            return
        except (tarfile.ReadError, EOFError, OSError) as exc:
            print(f"[warn] tarball corrupt: {exc!r}; deleting and retrying")
            target.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} after {max_attempts} attempts")


def download_tarball(
    starlight: StarlightSpec, slab: bool, raw_dir: Path, force: bool = False
) -> Path:
    """Download a single tarball from the Princeton PAHspec site.

    Uses ``curl`` with resume, retry, and end-to-end tarball validation
    to be robust to network blips on the ~250 MB-per-tarball downloads.

    Returns
    -------
    Path
        Local path of the downloaded ``.tgz``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_slab" if slab else ""
    fname = f"{starlight.name}{suffix}.tgz"
    target = raw_dir / fname
    if target.exists() and not force:
        # Verify integrity even on cached files.
        try:
            with tarfile.open(target, "r:gz") as tf:
                tf.next()
            return target
        except (tarfile.ReadError, EOFError, OSError):
            print(f"[warn] cached tarball {target} is corrupt; redownloading")
            target.unlink()
    url = f"{PAHSPEC_BASE_URL}/{fname}"
    _curl_download(url, target)
    return target


def extract_tarball(tgz_path: Path, dest: Path) -> Path:
    """Extract a tarball if not already extracted; return the inner dir."""
    inner = dest / tgz_path.stem  # e.g., dest/mMMP or dest/mMMP_slab
    if inner.is_dir() and any(inner.iterdir()):
        return inner
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[extract]  {tgz_path}  ->  {inner}")
    with tarfile.open(tgz_path, "r:gz") as tf:
        # Allow the tarball to dictate its own top-level dir name.
        tf.extractall(dest, filter="data")
    if not inner.is_dir():
        # Fallback: tarball used a different top-level name.
        for p in dest.iterdir():
            if p.is_dir() and p.name.startswith(tgz_path.stem.split("_slab")[0]):
                return p
        raise FileNotFoundError(f"could not locate extracted dir for {tgz_path}")
    return inner


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------


def _assemble_grid(
    raw_dir: Path,
    starlights: list[StarlightSpec],
    slab_variants: list[bool],
) -> dict:
    """Walk extracted PAHspec dirs and pack into ndarrays.

    Returns a dict ready to write to HDF5.
    """
    # Read one reference file to size the wavelength axis.
    first_starlight = starlights[0]
    first_slab = slab_variants[0]
    suffix = "_slab" if first_slab else ""
    inner = raw_dir / f"{first_starlight.name}{suffix}"
    if not inner.is_dir():
        raise FileNotFoundError(f"missing extracted dir {inner}; run with --download")
    # Use the canonical (lgU=0, st, std) sample as the wavelength reference.
    ref_path = inner / pahspec_filename(
        first_starlight,
        0.0,
        "st",
        "std",
        slab=bool(first_slab),
    )
    if not ref_path.exists():
        raise FileNotFoundError(f"reference file missing: {ref_path}")
    ref = _read_pahspec_file(ref_path)
    wave_um = ref.wavelength_um
    n_wave = wave_um.size

    n_sl = len(starlights)
    n_slab = len(slab_variants)
    n_lgU = len(LGU_VALUES)
    n_ion = len(ION_LEVELS)
    n_size = len(SIZE_DISTS)
    shape = (n_sl, n_slab, n_lgU, n_ion, n_size, n_wave)

    total = np.zeros(shape, dtype=np.float32)
    astro = np.zeros(shape, dtype=np.float32)
    pah_p = np.zeros(shape, dtype=np.float32)
    pah_0 = np.zeros(shape, dtype=np.float32)
    tir = np.zeros(shape[:-1], dtype=np.float64)
    present = np.zeros(shape[:-1], dtype=bool)

    for i_sl, sl in enumerate(starlights):
        for i_slab, is_slab in enumerate(slab_variants):
            sub_suffix = "_slab" if is_slab else ""
            inner = raw_dir / f"{sl.name}{sub_suffix}"
            if not inner.is_dir():
                print(f"[skip] missing dir {inner}")
                continue
            for i_u, lgU in enumerate(LGU_VALUES):
                for i_i, ion in enumerate(ION_LEVELS):
                    for i_s, sz in enumerate(SIZE_DISTS):
                        path = inner / pahspec_filename(
                            sl,
                            lgU,
                            ion,
                            sz,
                            slab=bool(is_slab),
                        )
                        if not path.exists():
                            continue
                        spec = _read_pahspec_file(path)
                        if spec.wavelength_um.size != n_wave:
                            raise ValueError(
                                f"wavelength size mismatch in {path}: "
                                f"{spec.wavelength_um.size} != {n_wave}"
                            )
                        if not np.allclose(spec.wavelength_um, wave_um, rtol=1e-6):
                            raise ValueError(f"wavelength grid mismatch in {path}")
                        idx = (i_sl, i_slab, i_u, i_i, i_s)
                        total[idx] = spec.nu_pnu_total
                        astro[idx] = spec.nu_pnu_astrodust
                        pah_p[idx] = spec.nu_pnu_pah_plus
                        pah_0[idx] = spec.nu_pnu_pah_neutral
                        tir[idx] = spec.tir_total
                        present[idx] = True

    n_present = int(present.sum())
    n_expected = n_sl * n_slab * n_lgU * n_ion * n_size
    print(f"[grid]    {n_present}/{n_expected} cells filled")

    return {
        "wavelength_um": wave_um.astype(np.float32),
        "lgU": np.asarray(LGU_VALUES, dtype=np.float32),
        "starlight_names": np.asarray([s.name for s in starlights], dtype="S64"),
        "ion_names": np.asarray(ION_LEVELS, dtype="S4"),
        "size_names": np.asarray(SIZE_DISTS, dtype="S4"),
        "slab": np.asarray(slab_variants, dtype=bool),
        "nu_pnu_total": total,
        "nu_pnu_astrodust": astro,
        "nu_pnu_pah_plus": pah_p,
        "nu_pnu_pah_neutral": pah_0,
        "tir_total": tir.astype(np.float32),
        "present": present,
    }


def _write_hdf5(out_path: Path, grid: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write]   {out_path}")
    with h5py.File(out_path, "w") as f:
        f.attrs["paper"] = "Draine, Li, Hensley et al. 2021 ApJ 917 3"
        f.attrs["arxiv"] = "2011.07046"
        f.attrs["doi"] = "10.7910/DVN/LPUHIQ"
        f.attrs["columns"] = "nu*P_nu erg s^-1 H^-1; per-component"
        for k, v in grid.items():
            f.create_dataset(
                k,
                data=v,
                compression="gzip",
                compression_opts=4,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _select_starlights(names: list[str] | None) -> list[StarlightSpec]:
    if not names:
        return list(STARLIGHTS)
    by_name = {s.name: s for s in STARLIGHTS}
    out: list[StarlightSpec] = []
    for n in names:
        if n not in by_name:
            raise SystemExit(f"unknown starlight {n!r}; pick from {sorted(by_name)}")
        out.append(by_name[n])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--raw-dir", type=Path, default=Path("~/.cache/tengri/pahspec_raw").expanduser()
    )
    p.add_argument(
        "--output", type=Path, required=True, help="output HDF5 file (created/overwritten)"
    )
    p.add_argument(
        "--starlights", nargs="*", default=None, help="subset of starlight names; default = all 15"
    )
    p.add_argument("--no-slab", action="store_true", help="skip the A_V=2 slab variants")
    p.add_argument("--download", action="store_true", help="download tarballs if missing")
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args(argv)

    starlights = _select_starlights(args.starlights)
    slab_variants = [False] if args.no_slab else [False, True]

    raw_dir: Path = args.raw_dir.expanduser().resolve()

    for sl in starlights:
        for slab in slab_variants:
            if args.download or args.force_download:
                tgz = download_tarball(sl, slab, raw_dir, force=args.force_download)
            else:
                suffix = "_slab" if slab else ""
                tgz = raw_dir / f"{sl.name}{suffix}.tgz"
                if not tgz.exists():
                    print(f"[skip] {tgz} not present (use --download)")
                    continue
            extract_tarball(tgz, raw_dir)

    grid = _assemble_grid(raw_dir, starlights, slab_variants)
    _write_hdf5(args.output, grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
