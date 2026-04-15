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

# Mirrors FILTER_REGISTRY in filters.py — keep in sync.
# Note: ALMA bands are synthetic top-hats (no SVO entry) and are excluded here.
_FILTER_REGISTRY: dict[str, str] = {
    # ------------------------------------------------------------------
    # UV / optical — ground-based survey imagers
    # ------------------------------------------------------------------
    # SDSS
    "sdss_u": "SLOAN/SDSS.u",
    "sdss_g": "SLOAN/SDSS.g",
    "sdss_r": "SLOAN/SDSS.r",
    "sdss_i": "SLOAN/SDSS.i",
    "sdss_z": "SLOAN/SDSS.z",
    # LSST / Rubin Observatory
    "lsst_u": "LSST/LSST.u",
    "lsst_g": "LSST/LSST.g",
    "lsst_r": "LSST/LSST.r",
    "lsst_i": "LSST/LSST.i",
    "lsst_z": "LSST/LSST.z",
    "lsst_y": "LSST/LSST.y",
    # Pan-STARRS PS1
    "ps1_g": "PAN-STARRS/PS1.g",
    "ps1_r": "PAN-STARRS/PS1.r",
    "ps1_i": "PAN-STARRS/PS1.i",
    "ps1_z": "PAN-STARRS/PS1.z",
    "ps1_y": "PAN-STARRS/PS1.y",
    # DES / DECam (CTIO)
    "des_g": "CTIO/DECam.g_filter",
    "des_r": "CTIO/DECam.r_filter",
    "des_i": "CTIO/DECam.i_filter",
    "des_z": "CTIO/DECam.z_filter",
    "des_y": "CTIO/DECam.Y",
    # CFHT MegaCam
    "megacam_u": "CFHT/MegaCam.u",
    "megacam_u2": "CFHT/MegaCam.u_1",
    "megacam_g": "CFHT/MegaCam.g",
    "megacam_g2": "CFHT/MegaCam.g_1",
    "megacam_r": "CFHT/MegaCam.r",
    "megacam_r2": "CFHT/MegaCam.r_1",
    "megacam_i": "CFHT/MegaCam.i",
    "megacam_i2": "CFHT/MegaCam.i_1",
    "megacam_i3": "CFHT/MegaCam.i_2",
    "megacam_z": "CFHT/MegaCam.z",
    "megacam_z2": "CFHT/MegaCam.z_1",
    # Subaru HSC (r2/i2 = second-gen filters used in HSC-SSP PDR2+)
    "hsc_g": "Subaru/HSC.g",
    "hsc_r": "Subaru/HSC.r2_filter",
    "hsc_i": "Subaru/HSC.i2_filter",
    "hsc_z": "Subaru/HSC.z",
    "hsc_y": "Subaru/HSC.Y",
    # Subaru SuprimeCam — broadband
    "suprime_b": "Subaru/Suprime.B",
    "suprime_g": "Subaru/Suprime.g",
    "suprime_v": "Subaru/Suprime.V",
    "suprime_r": "Subaru/Suprime.r",
    "suprime_rc": "Subaru/Suprime.Rc_filter",
    "suprime_i": "Subaru/Suprime.i",
    "suprime_ic": "Subaru/Suprime.Ic_filter",
    "suprime_z": "Subaru/Suprime.z",
    "suprime_y": "Subaru/Suprime.Y_filter",
    # Subaru SuprimeCam — intermediate-band
    "suprime_ib427": "Subaru/Suprime.IB427",
    "suprime_ib464": "Subaru/Suprime.IB464",
    "suprime_ib484": "Subaru/Suprime.IB484",
    "suprime_ib505": "Subaru/Suprime.IB505",
    "suprime_ib527": "Subaru/Suprime.IB527",
    "suprime_ib574": "Subaru/Suprime.IB574",
    "suprime_ib624": "Subaru/Suprime.IB624",
    "suprime_ib679": "Subaru/Suprime.IB679",
    "suprime_ib709": "Subaru/Suprime.IB709",
    "suprime_ib738": "Subaru/Suprime.IB738",
    "suprime_ib767": "Subaru/Suprime.IB767",
    "suprime_ib827": "Subaru/Suprime.IB827",
    # Subaru SuprimeCam — narrow-band
    "suprime_na656": "Subaru/Suprime.NA656_filter",
    "suprime_nb711": "Subaru/Suprime.NB711",
    "suprime_nb816": "Subaru/Suprime.NB816",
    "suprime_nb921": "Subaru/Suprime.NB921_filter",
    # ------------------------------------------------------------------
    # UV / optical — space telescopes
    # ------------------------------------------------------------------
    # GALEX
    "galex_fuv": "GALEX/GALEX.FUV",
    "galex_nuv": "GALEX/GALEX.NUV",
    # XMM-Newton OM
    "xmm_uvw2": "XMM/OM.UVW2",
    "xmm_uvm2": "XMM/OM.UVM2",
    "xmm_uvw1": "XMM/OM.UVW1",
    "xmm_u": "XMM/OM.U",
    "xmm_b": "XMM/OM.B",
    "xmm_v": "XMM/OM.V",
    # Swift UVOT
    "uvot_uvw2": "Swift/UVOT.UVW2",
    "uvot_uvm2": "Swift/UVOT.UVM2",
    "uvot_uvw1": "Swift/UVOT.UVW1",
    "uvot_u": "Swift/UVOT.U",
    "uvot_b": "Swift/UVOT.B",
    "uvot_v": "Swift/UVOT.V",
    "uvot_white": "Swift/UVOT.white",
    # ------------------------------------------------------------------
    # NIR — ground-based
    # ------------------------------------------------------------------
    # 2MASS
    "2mass_j": "2MASS/2MASS.J",
    "2mass_h": "2MASS/2MASS.H",
    "2mass_ks": "2MASS/2MASS.Ks",
    # VISTA / VIRCAM
    "vista_z": "Paranal/VISTA.Z",
    "vista_y": "Paranal/VISTA.Y",
    "vista_j": "Paranal/VISTA.J",
    "vista_h": "Paranal/VISTA.H",
    "vista_ks": "Paranal/VISTA.Ks",
    # UKIRT WFCAM / UKIDSS
    "ukidss_z": "UKIRT/UKIDSS.Z",
    "ukidss_y": "UKIRT/UKIDSS.Y",
    "ukidss_j": "UKIRT/UKIDSS.J",
    "ukidss_h": "UKIRT/UKIDSS.H",
    "ukidss_k": "UKIRT/UKIDSS.K",
    # ------------------------------------------------------------------
    # HST
    # ------------------------------------------------------------------
    "hst_f435w": "HST/ACS_WFC.F435W",
    "hst_f606w": "HST/ACS_WFC.F606W",
    "hst_f775w": "HST/ACS_WFC.F775W",
    "hst_f814w": "HST/ACS_WFC.F814W",
    "hst_f850lp": "HST/ACS_WFC.F850LP",
    "hst_f105w": "HST/WFC3_IR.F105W",
    "hst_f125w": "HST/WFC3_IR.F125W",
    "hst_f140w": "HST/WFC3_IR.F140W",
    "hst_f160w": "HST/WFC3_IR.F160W",
    # ------------------------------------------------------------------
    # JWST NIRCam — original throughput curves
    # ------------------------------------------------------------------
    "jwst_f070w": "JWST/NIRCam.F070W",
    "jwst_f090w": "JWST/NIRCam.F090W",
    "jwst_f115w": "JWST/NIRCam.F115W",
    "jwst_f140m": "JWST/NIRCam.F140M",
    "jwst_f150w": "JWST/NIRCam.F150W",
    "jwst_f150w2": "JWST/NIRCam.F150W2",
    "jwst_f162m": "JWST/NIRCam.F162M",
    "jwst_f164n": "JWST/NIRCam.F164N",
    "jwst_f182m": "JWST/NIRCam.F182M",
    "jwst_f187n": "JWST/NIRCam.F187N",
    "jwst_f200w": "JWST/NIRCam.F200W",
    "jwst_f210m": "JWST/NIRCam.F210M",
    "jwst_f212n": "JWST/NIRCam.F212N",
    "jwst_f250m": "JWST/NIRCam.F250M",
    "jwst_f277w": "JWST/NIRCam.F277W",
    "jwst_f300m": "JWST/NIRCam.F300M",
    "jwst_f322w2": "JWST/NIRCam.F322W2",
    "jwst_f323n": "JWST/NIRCam.F323N",
    "jwst_f335m": "JWST/NIRCam.F335M",
    "jwst_f356w": "JWST/NIRCam.F356W",
    "jwst_f360m": "JWST/NIRCam.F360M",
    "jwst_f405n": "JWST/NIRCam.F405N",
    "jwst_f410m": "JWST/NIRCam.F410M",
    "jwst_f430m": "JWST/NIRCam.F430M",
    "jwst_f444w": "JWST/NIRCam.F444W",
    "jwst_f460m": "JWST/NIRCam.F460M",
    "jwst_f466n": "JWST/NIRCam.F466N",
    "jwst_f470n": "JWST/NIRCam.F470N",
    "jwst_f480m": "JWST/NIRCam.F480M",
    # ------------------------------------------------------------------
    # JWST NIRCam2025 — recalibrated throughputs (use for data after 2025)
    # ------------------------------------------------------------------
    "nircam25_f070w": "JWST/NIRCam2025.F070W",
    "nircam25_f090w": "JWST/NIRCam2025.F090W",
    "nircam25_f115w": "JWST/NIRCam2025.F115W",
    "nircam25_f140m": "JWST/NIRCam2025.F140M",
    "nircam25_f150w": "JWST/NIRCam2025.F150W",
    "nircam25_f150w2": "JWST/NIRCam2025.F150W2",
    "nircam25_f162m": "JWST/NIRCam2025.F162M",
    "nircam25_f164n": "JWST/NIRCam2025.F164N",
    "nircam25_f182m": "JWST/NIRCam2025.F182M",
    "nircam25_f187n": "JWST/NIRCam2025.F187N",
    "nircam25_f200w": "JWST/NIRCam2025.F200W",
    "nircam25_f210m": "JWST/NIRCam2025.F210M",
    "nircam25_f212n": "JWST/NIRCam2025.F212N",
    "nircam25_f250m": "JWST/NIRCam2025.F250M",
    "nircam25_f277w": "JWST/NIRCam2025.F277W",
    "nircam25_f300m": "JWST/NIRCam2025.F300M",
    "nircam25_f322w2": "JWST/NIRCam2025.F322W2",
    "nircam25_f323n": "JWST/NIRCam2025.F323N",
    "nircam25_f335m": "JWST/NIRCam2025.F335M",
    "nircam25_f356w": "JWST/NIRCam2025.F356W",
    "nircam25_f360m": "JWST/NIRCam2025.F360M",
    "nircam25_f405n": "JWST/NIRCam2025.F405N",
    "nircam25_f410m": "JWST/NIRCam2025.F410M",
    "nircam25_f430m": "JWST/NIRCam2025.F430M",
    "nircam25_f444w": "JWST/NIRCam2025.F444W",
    "nircam25_f460m": "JWST/NIRCam2025.F460M",
    "nircam25_f466n": "JWST/NIRCam2025.F466N",
    "nircam25_f470n": "JWST/NIRCam2025.F470N",
    "nircam25_f480m": "JWST/NIRCam2025.F480M",
    # ------------------------------------------------------------------
    # JWST NIRISS
    # ------------------------------------------------------------------
    "niriss_f090w": "JWST/NIRISS.F090W",
    "niriss_f115w": "JWST/NIRISS.F115W",
    "niriss_f140m": "JWST/NIRISS.F140M",
    "niriss_f150w": "JWST/NIRISS.F150W",
    "niriss_f158m": "JWST/NIRISS.F158M",
    "niriss_f200w": "JWST/NIRISS.F200W",
    "niriss_f277w": "JWST/NIRISS.F277W",
    "niriss_f356w": "JWST/NIRISS.F356W",
    "niriss_f380m": "JWST/NIRISS.F380M",
    "niriss_f430m": "JWST/NIRISS.F430M",
    "niriss_f444w": "JWST/NIRISS.F444W",
    "niriss_f480m": "JWST/NIRISS.F480M",
    # ------------------------------------------------------------------
    # JWST NIRSpec
    # ------------------------------------------------------------------
    "nirspec_prism": "JWST/NIRSpec.Prism",
    "nirspec_g235m": "JWST/NIRSpec.G235M_F170LP",
    "nirspec_g235h": "JWST/NIRSpec.G235H_F170LP",
    "nirspec_g395m": "JWST/NIRSpec.G395M_F290LP",
    "nirspec_g395h": "JWST/NIRSpec.G395H_F290LP",
    # ------------------------------------------------------------------
    # JWST MIRI (5.6–25.5 μm)
    # ------------------------------------------------------------------
    "miri_f560w": "JWST/MIRI.F560W",
    "miri_f770w": "JWST/MIRI.F770W",
    "miri_f1000w": "JWST/MIRI.F1000W",
    "miri_f1065c": "JWST/MIRI.F1065C",
    "miri_f1130w": "JWST/MIRI.F1130W",
    "miri_f1140c": "JWST/MIRI.F1140C",
    "miri_f1280w": "JWST/MIRI.F1280W",
    "miri_f1500w": "JWST/MIRI.F1500W",
    "miri_f1550c": "JWST/MIRI.F1550C",
    "miri_f1800w": "JWST/MIRI.F1800W",
    "miri_f2100w": "JWST/MIRI.F2100W",
    "miri_f2300c": "JWST/MIRI.F2300C",
    "miri_f2550w": "JWST/MIRI.F2550W",
    # ------------------------------------------------------------------
    # Roman Space Telescope / WFI
    # ------------------------------------------------------------------
    "roman_f062": "Roman/WFI.F062",
    "roman_f087": "Roman/WFI.F087",
    "roman_f106": "Roman/WFI.F106",
    "roman_f129": "Roman/WFI.F129",
    "roman_f158": "Roman/WFI.F158",
    "roman_f184": "Roman/WFI.F184",
    "roman_f213": "Roman/WFI.F213",
    # ------------------------------------------------------------------
    # Euclid
    # ------------------------------------------------------------------
    "euclid_vis": "Euclid/VIS.vis",
    "euclid_y": "Euclid/NISP.Y",
    "euclid_j": "Euclid/NISP.J",
    "euclid_h": "Euclid/NISP.H",
    # ------------------------------------------------------------------
    # Spitzer
    # ------------------------------------------------------------------
    "irac_36": "Spitzer/IRAC.I1",
    "irac_45": "Spitzer/IRAC.I2",
    "irac_58": "Spitzer/IRAC.I3",
    "irac_80": "Spitzer/IRAC.I4",
    "mips_24": "Spitzer/MIPS.24mu",
    "mips_70": "Spitzer/MIPS.70mu",
    "mips_160": "Spitzer/MIPS.160mu",
    # ------------------------------------------------------------------
    # WISE
    # ------------------------------------------------------------------
    "wise_w1": "WISE/WISE.W1",
    "wise_w2": "WISE/WISE.W2",
    "wise_w3": "WISE/WISE.W3",
    "wise_w4": "WISE/WISE.W4",
    # ------------------------------------------------------------------
    # AKARI
    # ------------------------------------------------------------------
    "akari_n2": "AKARI/IRC.N2",
    "akari_n3": "AKARI/IRC.N3",
    "akari_n4": "AKARI/IRC.N4",
    "akari_s7": "AKARI/IRC.S7",
    "akari_s9w": "AKARI/IRC.S9W",
    "akari_s11": "AKARI/IRC.S11",
    "akari_l15": "AKARI/IRC.L15",
    "akari_l18w": "AKARI/IRC.L18W",
    "akari_l24": "AKARI/IRC.L24",
    "akari_n60": "AKARI/FIS.N60",
    "akari_wides": "AKARI/FIS.WIDE-S",
    "akari_widel": "AKARI/FIS.WIDE-L",
    "akari_n160": "AKARI/FIS.N160",
    # ------------------------------------------------------------------
    # Herschel
    # ------------------------------------------------------------------
    "herschel_70": "Herschel/Pacs.blue",
    "herschel_100": "Herschel/Pacs.green",
    "herschel_160": "Herschel/Pacs.red",
    "herschel_250": "Herschel/SPIRE.PSW",
    "herschel_350": "Herschel/SPIRE.PMW",
    "herschel_500": "Herschel/SPIRE.PLW",
    "herschel_250_ext": "Herschel/SPIRE.PSW_ext",
    "herschel_350_ext": "Herschel/SPIRE.PMW_ext",
    "herschel_500_ext": "Herschel/SPIRE.PLW_ext",
    # ------------------------------------------------------------------
    # (Sub)millimeter
    # ------------------------------------------------------------------
    "scuba2_450": "JCMT/SCUBA2.450GHz",
    "scuba2_850": "JCMT/SCUBA2.850GHz",
    "laboca_870": "APEX/LABOCA.345GHz",
    "saboca_350": "APEX/SABOCA.852GHz",
    # ------------------------------------------------------------------
    # Generic / standard photometric systems
    # ------------------------------------------------------------------
    "johnson_u": "Generic/Johnson.U",
    "johnson_b": "Generic/Johnson.B",
    "johnson_v": "Generic/Johnson.V",
    "johnson_r": "Generic/Johnson.R",
    "johnson_i": "Generic/Johnson.I",
    "johnson_j": "2MASS/2MASS.J",
    "cousins_r": "Generic/Cousins.R",
    "cousins_i": "Generic/Cousins.I",
}


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
