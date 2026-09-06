# SPDX-License-Identifier: BSD-3-Clause
"""Shared test fixtures for tengri test suite."""

import os
from collections import Counter
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ── Import source verification guard ────────────────────────────────────────
#
# Guard against importing tengri from a different checkout or worktree than
# the one running tests. This catches misconfigurations of the shared venv
# (.venv/lib/python3.12/site-packages/__editable__.astro_tengri-*.pth files
# that point to a stale worktree), which silently run tests against incorrect
# source code (#2170).
#
# Mechanism: when tests exist in src/tengri/, verify that the imported module
# lives in the same tree. The check runs at module load time (before any test)
# so failures block collection instead of silently running the wrong code.
# Stays silent for installed (non-editable) releases, since there is no
# src/tengri/ to compare.


def _check_tengri_source_tree_match(
    repo_root: Path | None = None,
    imported_file: Path | None = None,
) -> None:
    """Verify imported tengri lives in the same tree as the tests.

    Parameters
    ----------
    repo_root : Path, optional
        Repository root to check. Defaults to the parent of this file's
        directory (the real repository). Injectable for testing.
    imported_file : Path, optional
        Path of the imported ``tengri/__init__.py``. Defaults to importing
        tengri and reading ``__file__``. Injectable for testing.

    Raises
    ------
    RuntimeError
        If tests exist in src/tengri/ but the imported tengri module
        comes from a different location.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    tengri_src = repo_root / "src" / "tengri"

    # Only check if this is a development tree with local source.
    if not tengri_src.exists():
        return

    if imported_file is None:
        import tengri

        imported_file = Path(tengri.__file__)

    # Resolve both sides so symlinked checkouts (e.g. /tmp vs /private/tmp
    # on macOS) compare by real location, not spelling.
    imported_file = Path(imported_file).resolve()
    expected_file = (tengri_src / "__init__.py").resolve()

    if imported_file != expected_file:
        raise RuntimeError(
            f"Imported tengri from wrong source tree (#2170).\n"
            f"Tests are in:    {repo_root}\n"
            f"Expected source: {expected_file}\n"
            f"Actual source:   {imported_file}\n\n"
            f"This usually means the shared .venv has a stale editable install "
            f"pth file pointing to a different worktree. Fix by running:\n"
            f"  pip install -e . --force-reinstall --no-deps\n"
            f"from {repo_root}, or by setting:\n"
            f"  PYTHONPATH={repo_root / 'src'}"
        )


# Run the import verification at collection time.
_check_tengri_source_tree_match()

# ── Network guard ────────────────────────────────────────────────────────
#
# No test may reach the network. This exists because two tests that did
# reddened main (#1546): they called ``load_ssp_data`` on a gitignored 114 MB
# path, so the loader fell through to downloading it. On runners where DNS
# worked they passed *by fetching 114 MB per job*; on the one where it did not
# they failed with ``Temporary failure in name resolution``.
#
# That is the shape worth blocking. A network-dependent test is not reliably
# red — it is *occasionally green*, so it survives review, passes CI, and then
# breaks main when a runner happens to have no DNS. The same defect reached
# main once before through the gallery (#1486).
#
# Blocking makes it fail immediately and locally instead, with a message that
# names the fixture to use. Opt out with ``@pytest.mark.network`` for a test
# whose subject genuinely IS the download path — though note every existing
# such test mocks the transport rather than needing this marker.

#: Addresses a test may still reach: loopback (a local fixture server) and
#: AF_UNIX paths. Everything else is refused.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", ""})


class NetworkAccessDuringTest(RuntimeError):
    """Raised when a test tries to reach the network."""


def _network_refusal(target: str) -> NetworkAccessDuringTest:
    return NetworkAccessDuringTest(
        f"This test tried to reach the network ({target}).\n\n"
        "Tests must be hermetic: a networked test is not reliably red, it is "
        "occasionally green, so it passes review and CI and then breaks main "
        "on a runner without DNS (#1546, and #1486 before it).\n\n"
        "If you need an SSP grid, use the `synthetic_ssp_wide` fixture (no "
        "file, no network, 100 A - 1 mm). For a real library, use the "
        "`ssp_data_fsps` / `ssp_data_wne` fixtures, which skip when the data "
        "is absent instead of downloading it.\n\n"
        "If the download path genuinely IS the subject, mark the test "
        "`@pytest.mark.network` — but prefer mocking the transport, which is "
        "what every existing download test does."
    )


@pytest.fixture(autouse=True)
def _forbid_network(request, monkeypatch):
    """Refuse outbound connections for the duration of each test.

    Guards both chokepoints: ``getaddrinfo`` catches anything addressed by
    hostname (the CI failure mode exactly), and ``socket.connect`` catches a
    bare IP that never resolves a name.
    """
    if request.node.get_closest_marker("network"):
        return

    import socket

    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in _LOOPBACK:
            raise _network_refusal(f"DNS lookup for {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        # AF_UNIX addresses are str paths and stay allowed.
        if isinstance(host, str) and host not in _LOOPBACK and not host.startswith("/"):
            raise _network_refusal(f"connect to {host!r}")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ─────────────────────────────────────────────────────────────────────
# Skip-tally hook: surface how many parity-sweep tests actually ran
# vs. skipped due to missing data. Without this, the test report says
# "X passed, Y skipped" without itemizing whether the skipped ones
# were tests the session was supposed to verify.
# ─────────────────────────────────────────────────────────────────────

_SKIPPED_PARITY_TESTS: Counter[str] = Counter()


def pytest_runtest_makereport(item, call):  # pragma: no cover — pytest hook
    """Track skips on the parity-sweep tests so the terminal summary
    can flag low coverage."""
    if call.when != "setup" and call.when != "call":
        return
    rep = getattr(call, "excinfo", None)
    if rep is None:
        return
    nodeid = item.nodeid
    if "test_sedmodelcomponent_e2e" not in nodeid and "test_spectrum_lut" not in nodeid:
        return
    if call.excinfo.typename == "Skipped":
        _SKIPPED_PARITY_TESTS[nodeid] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # pragma: no cover
    """Surface parity-sweep skip count, and any file that tested nothing."""
    inert = inert_test_files()
    if inert:
        terminalreporter.write_sep("=", "FILES THAT TESTED NOTHING")
        for path, rec in inert:
            terminalreporter.write_line(
                f"  {path}: 0 ran, {rec['skip_call']} skipped from inside a test body"
            )
        terminalreporter.write_line(
            "Every test in the file(s) above skipped from inside a test body, so "
            "the file is green while verifying nothing. That is worth a look, not "
            "an automatic defect: a runtime `pytest.skip()` guarding genuinely "
            "absent data is legitimate (prefer a `skipif` marker, so the gate is "
            "visible at collection). The failure it catches is a skip whose "
            "*reason is wrong* — an `except` handler reporting 'data not "
            "available' when what actually raised was a stale API. Check which "
            "one this is. See #1615."
        )

    n_skipped = sum(_SKIPPED_PARITY_TESTS.values())
    if n_skipped > 0:
        terminalreporter.write_sep("=", "PARITY-SWEEP SKIP TALLY")
        terminalreporter.write_line(
            f"{n_skipped} parity-sweep test(s) skipped — usually data missing on CI runner."
        )
        terminalreporter.write_line(
            "If this count is high (>50%), the coverage claim in the README is misleading. "
            "Either ship tiny synthetic test fixtures or accept the coverage gap honestly."
        )

    # Class-killer guard: surface files where EVERY test skipped, whether
    # from markers (setup phase) or from test-body logic (call phase).
    # This guard is opt-in via TENGRI_GUARD_FULLY_SKIP env var to avoid
    # false-positives on local partial runs (-k filters, single-file invocations).
    if os.environ.get("TENGRI_GUARD_FULLY_SKIP"):
        fully_skipped_files = fully_skipped_test_files(_FILE_OUTCOMES)
        allowlist = _SKIP_GUARD_ALLOWLIST
        violations = [
            f for f, _ in fully_skipped_files if f not in allowlist and not _is_excluded_tree(f)
        ]
        if violations:
            terminalreporter.write_sep("!", "FULLY-SKIPPING TEST FILES (CLASS-KILLER GUARD)")
            for f in violations:
                terminalreporter.write_line(f"  {f}")
            terminalreporter.write_line(
                "Every test in the file(s) above was skipped, rendering the file inert "
                "coverage. Files on the guard allowlist or in excluded trees (integration, "
                "crossval) are expected; others are bugs. See #1615/#1946."
            )
            if exitstatus == 0:
                # Only fail the session if the test run itself succeeded but coverage is inert.
                # If tests failed for other reasons, that takes precedence.
                terminalreporter.write_sep("!", "FAILING: INERT COVERAGE DETECTED")
                raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────
# Inert-file detector: a file where every test skipped *from inside a
# test body* has stopped testing anything, and says so with a green tick.
#
# `tests/components/dust/test_dust_emission_traceable.py` did exactly that
# for an unknown stretch: 6 of 6 skipping, five on
# `SEDModel.__init__() got an unexpected keyword argument 'filter_waves'`
# — a stale-API TypeError caught by `except Exception: pytest.skip(...)`.
# It was the only thing exercising the dust template-threading seam, so it
# also stood in as the evidence that the seam worked (#1615).
#
# The discriminator is *when* the skip happened, which is what makes this
# rule usable on CI at all:
#
#   * `@pytest.mark.skipif(...)` is evaluated before the test runs, so the
#     report arrives with ``when == "setup"``. This is the honest
#     data-gate — "CLOUDY grid not present" — and CI is full of them
#     because the grids are not shipped. Never flagged.
#   * `pytest.skip()` reached *inside* the test body arrives with
#     ``when == "call"``. The test started, something went wrong, and the
#     handler converted it to a skip. One of those is ordinary; a file
#     where they are the ONLY outcome is a file testing nothing.
#
# So: flag a file iff nothing ran AND at least one skip came from a test
# body. A file gated entirely by markers stays silent no matter how many
# of its tests skip.
# ─────────────────────────────────────────────────────────────────────

_FILE_OUTCOMES: dict[str, dict[str, int]] = {}
_FILE_COLLECTED: dict[str, int] = {}  # Track collected test count per file


def pytest_runtest_logreport(report):  # pragma: no cover — pytest hook
    """Tally, per file, what actually executed versus what skipped and when."""
    path = report.nodeid.split("::")[0]
    rec = _FILE_OUTCOMES.setdefault(path, {"ran": 0, "skip_setup": 0, "skip_call": 0})

    # xfail arrives as a call-phase "skip" with ``wasxfail`` attached. It is a
    # test that ran and behaved as declared, not an absence of testing.
    if getattr(report, "wasxfail", None) is not None:
        if report.when == "call":
            rec["ran"] += 1
        return

    if report.when == "setup" and report.skipped:
        rec["skip_setup"] += 1
    elif report.when == "call":
        if report.skipped:
            rec["skip_call"] += 1
        else:
            rec["ran"] += 1


def pytest_collection_finish(session):  # pragma: no cover — pytest hook
    """Record how many tests were collected per file."""
    for item in session.items:
        path = item.nodeid.split("::")[0]
        _FILE_COLLECTED[path] = _FILE_COLLECTED.get(path, 0) + 1


def inert_test_files() -> list[tuple[str, dict[str, int]]]:
    """Files where nothing ran and at least one test skipped from its own body.

    Returns
    -------
    list of (path, counts)
        Sorted by path. Empty when every collected file either ran something
        or was skipped entirely by markers.
    """
    return sorted(
        (path, rec)
        for path, rec in _FILE_OUTCOMES.items()
        if rec["ran"] == 0 and rec["skip_call"] > 0
    )


def fully_skipped_test_files(
    outcomes: dict[str, dict[str, int]],
) -> list[tuple[str, dict[str, int]]]:
    """Files whose every executed test skipped (ran == 0, skips > 0).

    Derived purely from runtest reports: under pytest-xdist the workers'
    reports reach the controller while ``session.items`` does not, so a
    collection-based count is empty exactly where CI runs in parallel and
    would make this guard vacuous there. Deselected tests (-k/-m) produce
    no reports, so partial runs cannot make an innocent file look fully
    skipped; a file whose only outcomes are errors has ran == 0 and zero
    skips and is left to the ordinary failure reporting.
    """
    return sorted(
        (path, rec)
        for path, rec in outcomes.items()
        if rec["ran"] == 0 and (rec["skip_setup"] + rec["skip_call"]) > 0
    )


# ─────────────────────────────────────────────────────────────────────
# Guard allowlist: files that legitimately skip all collected tests.
#
# Entries here are expected to fully-skip due to missing optional data,
# platform gates, or design (e.g., integration tests without SSP grids).
# Keep justifications short: each entry documents *why* this file can
# skip everything and still be considered correct coverage.
#
# EXCLUDED TREES (checked via path, not allowlisted):
#   - tests/integration/: needs real SSP grids (data/ssp_*.h5), shipped to
#     developers only. BUG-NSS-01 regression test (test_user_scenarios.py)
#     is here; it skips if BPASS grid missing — that skip is expected.
#   - tests/crossval/: regression against external code (bagpipes, FSPS),
#     not run by default (explicitly gated with -m crossval).
#
# Per-file allowlist (entries are literal paths as pytest reports them):
#   - tests/components/agn/grahsp/test_template_parity.py: upstream GRAHSP repo
#   - tests/components/dust/test_synthesizer_line_parity.py: optional Synthesizer grids
#   - tests/components/nebular/test_cue_hybrid_diagnostic.py: optional BC03 SSP
#   - tests/contract/test_build_resolver_sedmodelcomponent.py: BC03 SSP not shipped
#   - tests/contract/test_phase4d_c_agn_threading.py: bare-stellar SSP not shipped
#   - tests/contract/test_synthesizer_nlr_grammar.py: optional Synthesizer grids
# ─────────────────────────────────────────────────────────────────────

_SKIP_GUARD_ALLOWLIST: set[str] = {
    "tests/components/agn/grahsp/test_template_parity.py",
    "tests/components/dust/test_synthesizer_line_parity.py",
    "tests/components/nebular/test_cue_hybrid_diagnostic.py",
    "tests/contract/test_build_resolver_sedmodelcomponent.py",
    "tests/contract/test_phase4d_c_agn_threading.py",
    "tests/contract/test_synthesizer_nlr_grammar.py",
}

_EXCLUDED_TREES = ("tests/integration/", "tests/crossval/")


def _is_excluded_tree(file_path: str) -> bool:
    """Check if a test file is in an excluded tree (integration, crossval)."""
    return any(file_path.startswith(tree) for tree in _EXCLUDED_TREES)


# Suppress background JIT compilation in the test suite.  Without this,
# every Fitter() instantiation spawns a compilation thread; with many test
# files each creating multiple Fitters the process floods with concurrent
# XLA compilations and exhausts memory.  Individual tests that exercise
# the compilation machinery clear this env var themselves.
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

# TENGRI_DISABLE_SSP_AUTODOWNLOAD used to be set here, to stop a test that
# named an absent grid from silently fetching it (#1528 reddened main that
# way).  Do not put it back.  ``load_ssp_data`` no longer fetches unless asked
# — ``download=False`` is the library default (#1553) — so the suite is
# protected by the API rather than by an env var this file has to remember to
# set, and the network guard above catches anything that still tries.
#
# Re-adding it as a third layer would make things worse, not safer: it trips
# before the network guard does, so *its* message would win, and the one
# ``load_ssp_data`` raised said "tengri.download_ssp() fetches the default
# FSPS grid to data/" — right for a user at a REPL, exactly backwards for the
# test author who actually reads it, since downloading is the defect there.
# When guards stack, the narrowest one owns the error message.

from tengri.components.stellar.sfh.gp_sfh import compute_sqrt_power_drw
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.utils.grid import grid_spacing, make_log_age_grid

# Enable 64-bit for numerical precision in tests
jax.config.update("jax_enable_x64", True)

# ── Paths for real SSP data ──────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SSP_FILE_WNE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_FILE_FSPS = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_SSP_FILE_BC03 = _DATA_DIR / "bc03_pdva_stelib_chabrier.h5"


# ── Session-scoped real SSP fixtures ─────────────────────────────
# Loading HDF5 SSP data is expensive (~0.5-1s per call).  Session scope
# ensures a single load shared across all test files that need it.


@pytest.fixture(scope="session")
def ssp_data_wne():
    """Load the wNE SSP data once per session.  Skip if file missing."""
    if not _SSP_FILE_WNE.is_file():
        pytest.skip(f"SSP data not found: {_SSP_FILE_WNE}")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_FILE_WNE))


@pytest.fixture(scope="session")
def ssp_data_fsps():
    """Load the FSPS SSP data once per session.  Skip if file missing."""
    if not _SSP_FILE_FSPS.is_file():
        pytest.skip(f"SSP data not found: {_SSP_FILE_FSPS}")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_FILE_FSPS))


@pytest.fixture(scope="session")
def ssp_data_bc03():
    """Bare-stellar BC03 SSP for Cue / CloudyGrid backends.

    Cue's NN ionizing-spectrum fit requires *bare-stellar* SSPs: it predicts
    line luminosities by reading max log10(Q_H) ≳ 45 for ages < 10 Myr from
    the template. wNE (post-nebular) SSPs have log10(Q_H) ≲ 43 because the
    ionizing photons were already absorbed during the original Cloudy run.
    Feeding wNE to Cue under-predicts line fluxes by 4-7 dex.

    BC03 PARSEC + STELIB Chabrier is the canonical bare-stellar SSP shipped
    by ``tengri.download_ssp("bc03_pdva_stelib_chabrier")``.
    """
    if not _SSP_FILE_BC03.is_file():
        pytest.skip(
            f"BC03 SSP not found: {_SSP_FILE_BC03} — run "
            f"`tengri.download_ssp('bc03_pdva_stelib_chabrier')` to fetch it."
        )
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_FILE_BC03))


@pytest.fixture
def real_ssp_only():
    """Skip a physics-value test when only the synthetic SSP fixture is present.

    #613 writes a synthetic SSP at ``_SSP_FILE_WNE`` so *structural* tests run
    on CI. Tests that assert calibrated physical values (magnitudes, SED ratios,
    energy balance) must NOT run on that uncalibrated grid — request this fixture
    to skip cleanly when the real grid is absent (file missing or tagged
    ``synthetic``).
    """
    if not _SSP_FILE_WNE.is_file():
        pytest.skip(f"real SSP grid not found: {_SSP_FILE_WNE}")
    with h5py.File(_SSP_FILE_WNE, "r") as f:
        if f.attrs.get("synthetic", False):
            pytest.skip("only the synthetic SSP fixture is present (#613); needs the real grid")


@pytest.fixture(scope="session")
def sdss_filters():
    """Load SDSS ugriz filters once per session."""
    from tengri.observation.filters import load_filter_set

    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


# ── Session-scoped synthetic SSP fixture ─────────────────────────
# Many unit tests create identical (3, 20, 100) synthetic SSPs.  Sharing
# a single instance avoids redundant array allocation and — more
# importantly — ensures all tests hit the same JIT-compiled code paths.


@pytest.fixture(scope="session")
def synthetic_ssp():
    """Minimal synthetic SSP: 3 Z × 20 ages × 100 wavelengths."""
    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="session")
def synthetic_ssp_wide():
    """Realistic synthetic SSP on a UV→far-IR grid with a SMOOTH continuum.

    Unlike :func:`synthetic_ssp` (narrow optical grid, noisy flux), this fixture
    spans ~100 Å – 1 mm so it drives the dust energy balance (L_absorbed in the
    UV/optical) and gives dust IR re-emission a grid to live on — and its
    continuum is smooth, so the SSP × filter Φ-tensor LUT is near machine-exact.

    Purpose (#613): let *structural* precompute/contract tests run on CI without
    the gitignored ``data/ssp_*.h5`` grids, instead of silently skipping (which
    is how #629/#617 regressions reached main). Use with :func:`synthetic_tophat_obs`.
    Physics-value tests (crossval, regression_paper) still need real SSPs.
    """
    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)  # 100 Å – 1 mm (1e7 Å)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)  # log10(age/Gyr): ~1 Myr – 13.8 Gyr
    # Absolute log10(Z) spanning a realistic FSPS-like range: with
    # LOG10_ZSUN = -1.848 this is logzsol ≈ [-2.15, 0.55], so priors/Fixed values
    # inside ~[-2, 0.5] are genuinely in-grid (keeps the metallicity-bounds
    # contract tests meaningful) while ±5/-10 fall outside.
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2  # bright in the UV/optical, ~0 in the far-IR
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="session")
def synthetic_tophat_obs():
    """5-band synthetic top-hat photometry Observation — no filter-data files.

    Companion to :func:`synthetic_ssp_wide` for CI-runnable structural tests.
    Edges taper to 0 (like real filters) so the padded filter integral behaves.
    """
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
    return Observation(photometry=Photometry(filters=curves))


@pytest.fixture(scope="session")
def simple_observation():
    """Synthetic 3-band observation matching the synthetic SSP wavelength range."""
    from tengri.observation.photometry import FilterCurve

    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.5 for _ in range(3)]
    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )
    from tengri import Observation, Photometry

    return Observation(photometry=Photometry(filters=curves))


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference gradient for scalar functions.

    Provides O(eps^2) accurate gradient estimate. Use to verify JAX autodiff:

        grad_jax = float(jax.grad(f)(jnp.array(x)))
        grad_fd  = fd_grad(f, x)
        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-3)

    Parameters
    ----------
    f : callable
        Scalar function float -> float (or jnp.array scalar -> scalar).
    x : float
        Point at which to estimate the gradient.
    eps : float
        Step size. 1e-4 is appropriate for float64; use 1e-3 for float32.

    Returns
    -------
    float
        Finite-difference gradient estimate (f(x+eps) - f(x-eps)) / (2*eps).
    """
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


#: Trees whose tests are sampler-driven end-to-end fits. They are auto-marked
#: `slow` so the default run is the PR-gating fast tier; CI runs them as a
#: separate schedule/label-gated job.
_SLOW_TREES = ("inference", "integration")


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test under the heavy trees as ``slow``.

    Marking by path rather than by decorator keeps the two trees exhaustively
    covered: a new file dropped into tests/inference cannot silently rejoin the
    fast tier because someone forgot the decorator.
    """
    tests_root = Path(__file__).parent
    for item in items:
        try:
            tree = Path(item.path).relative_to(tests_root).parts[0]
        except ValueError:
            continue
        if tree in _SLOW_TREES:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _isolate_component_registry():
    """Undo global ``SEDModelComponent`` registrations a test makes.

    ``SEDModelComponent.__init_subclass__`` records every subclass in a module-level
    ``_REGISTRY``, so a test that declares a throwaway component (``test_mbb``,
    ``test_minimal``, ...) leaks it into the registry for the rest of the session.
    Any test that WALKS the registry — ``test_param_defaults`` enumerates it and
    demands an in-bounds ``default=`` on every declaration — then passes or fails on
    nothing but file order: alphabetically ``test_param_defaults`` ran first and saw
    a clean registry, but under any other ordering it inherited the throwaways and
    reported them as missing defaults.

    Snapshot and restore so a registration cannot cross a test boundary. Registry
    entries created at import time (module-level component classes, including every
    real component) are already present when the first test starts, so they survive.
    """
    from tengri.components.sed_model_component import _REGISTRY

    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def pytest_configure(config):
    """Create minimal synthetic CB19 grid fixture before test collection.

    @pytest.mark.skipif(not path.exists(), ...) is evaluated at collection
    time when the test module is imported, so the file must already exist.
    This hook runs when the root conftest is imported — before collection —
    so any skipif that checks for cb19_templates.h5 will see the file.

    When the real file is present (built by scripts/download_cb19_templates.py
    while that route still worked, or supplied by hand -- see
    docs/internal/advanced/cb19_grid.md for the current status), this hook is
    a no-op.
    """
    # Keep the suite hermetic: never read or write the user's on-disk
    # z-table cache (~/.cache/tengri_precomp). The cache's own contract
    # tests opt back in with monkeypatch.delenv + a tmp_path cache dir.
    os.environ.setdefault("TENGRI_DISABLE_PRECOMP_CACHE", "1")

    # The synthetic SSP fixtures are unphysical by construction (young-bin
    # log Q_H ≈ 62 vs the physical ~47), which trips the CueBackend wNE
    # sanity band on every structural Cue test. Downgrade to the guard's
    # warning path suite-wide; the guard's raise contract is pinned by
    # tests/contract/test_new_user_errors.py, which delenv's this.
    #
    # Since #1579 this reaches the Q_H *heuristic* only -- which is all this
    # comment ever claimed. It also used to disable CueBackend's metadata
    # check, so a test could pair a *real* wNE grid with Cue and silently fit
    # a double-counted nebular model; test_population_psd_pilot.py did. A
    # heuristic needs an escape hatch because its false positives are routine
    # (see above); a declaration read from nebular_included has no
    # false-positive mode, so bypassing it can only hide a real error.
    os.environ.setdefault("TENGRI_ALLOW_WNE_CUE", "1")

    cb19_path = Path(__file__).parent.parent / "data" / "cb19_templates.h5"
    _create_cb19_fixture_if_missing(cb19_path)
    _create_silva04_fixture_if_missing(
        Path(__file__).parent.parent / "data" / "silva04_torus_grid.h5"
    )
    # #613: a schema-faithful synthetic SSP at the canonical default path so the
    # ~20 SSP-gated *structural* contract tests RUN in CI instead of skipping
    # (the masking that let #608/#768 regressions reach main). Physics-value
    # tests still gate themselves on the real grid via ``real_ssp_only``.
    _create_synthetic_ssp_if_missing(_SSP_FILE_WNE)


def _cb19_grid_is_degenerate(path: Path) -> bool:
    """Whether a CB_19 file on disk carries a single repeated line ratio.

    Parameters
    ----------
    path : Path
        Candidate ``cb19_templates.h5``.

    Returns
    -------
    bool
        ``True`` when the file is absent, unreadable, or every finite entry of
        its line-ratio slab is the same number.
    """
    if not path.is_file():
        return True
    try:
        with h5py.File(path, "r") as f:
            key = "grids/SSP/Kroupa01/mu100/line_ratios"
            if key not in f:
                return True
            ratios = f[key][:]
    except (OSError, KeyError):
        return True
    finite = ratios[np.isfinite(ratios)]
    return bool(finite.size == 0 or finite.min() == finite.max())


@pytest.fixture(scope="session")
def usable_cb19_grid_path(tmp_path_factory) -> Path:
    """A CB_19 grid file that loads: the packaged one, or a synthetic stand-in.

    ``data/cb19_templates.h5`` is untracked and, on a machine that ran an early
    build script, is the flat placeholder of #924: one repeated ratio, which
    :class:`~tengri.components.nebular.CB19DegenerateGridError` now refuses.
    Any test that needs *a grid to load* rather than *the packaged file
    specifically* should take this fixture.

    Returns
    -------
    Path
        The packaged file when it is usable, else a synthetic grid varying
        along every interpolation axis, written under a session tmp dir.
    """
    from tests._cb19_grid import write_synthetic_cb19_grid

    packaged = Path(__file__).parent.parent / "data" / "cb19_templates.h5"
    if not _cb19_grid_is_degenerate(packaged):
        return packaged
    return write_synthetic_cb19_grid(tmp_path_factory.mktemp("cb19_grid") / "cb19_templates.h5")


@pytest.fixture(scope="session", autouse=True)
def _usable_cb19_default_path(usable_cb19_grid_path):
    """Point CB_19's default path at a grid that loads, when the packaged one does not.

    Refusing the placeholder is the point of #2181, but it must not turn every
    CB_19 test into an error about the developer's data directory. The file on
    disk is left untouched; the refusal itself is asserted by
    ``tests/regression/bug/test_bug_2181_cb19_placeholder_grid.py``, which
    writes its own placeholder.

    Inert wherever the packaged file is usable, which includes CI (where
    ``pytest_configure`` writes the synthetic grid) and any machine carrying
    the real download.
    """
    packaged = Path(__file__).parent.parent / "data" / "cb19_templates.h5"
    if usable_cb19_grid_path == packaged:
        yield
        return

    patch = pytest.MonkeyPatch()
    patch.setattr("tengri.components.nebular.cloudy_cb19._DEFAULT_PATH", usable_cb19_grid_path)
    yield
    patch.undo()


def _create_cb19_fixture_if_missing(cb19_path: Path) -> None:
    """Write a synthetic CB_19 grid at the packaged path when none is present.

    The stand-in varies along every interpolation axis. Before #2181 it was ten
    Case B ratios *broadcast* across all seven axes: distinguishable per line,
    constant along every physical axis, so ``neb_logU``, ``neb_logZ_gas``,
    ``neb_log_nH``, ``neb_co`` and ``neb_dno`` were bit-exactly inert on it and
    no test in the suite could observe that.
    """
    if cb19_path.exists():
        return

    from tests._cb19_grid import write_synthetic_cb19_grid

    write_synthetic_cb19_grid(cb19_path)

    # Tag as synthetic so developers know it's a test fixture, not the real data.
    # pytest_configure output goes before any test output; warn() is the right channel.
    import warnings

    warnings.warn(
        f"Created synthetic CB19 fixture at {cb19_path} for tests. "
        "scripts/download_cb19_templates.py cannot currently replace it with "
        "the real grid: the upstream 3MdB CB_19 table is unpopulated "
        "pending an erratum (#2198); see docs/internal/advanced/cb19_grid.md.",
        UserWarning,
        stacklevel=1,
    )


def _create_silva04_fixture_if_missing(silva04_path: Path) -> None:
    """Synthesize a minimal Silva+04 cold-torus grid if absent.

    The Silva+04 loader raises FileNotFoundError when its HDF5 grid is
    missing, breaking ~ 20 test modules at import time in CI. We
    synthesize a minimal grid with a zero template so the orchestration
    code path runs; real physics tests can guard themselves with
    ``@pytest.mark.skipif`` against the actual grid file.

    The grid is keyed on ``silva04/{log_nh_axis, wavelength, template}``
    per ``_load_silva04_arrays`` in ``components/agn/silva04.py``.
    """
    if silva04_path.exists():
        return
    import warnings

    silva04_path.parent.mkdir(parents=True, exist_ok=True)
    n_nh, n_wave = 8, 64
    log_nh_axis = np.linspace(22.0, 25.0, n_nh).astype(np.float64)
    wavelength = np.logspace(np.log10(1.0), np.log10(1e7), n_wave).astype(np.float64)
    # Template: zero everywhere — gives a defensibly null Silva+04
    # spectrum so tests exercise the orchestration without accidentally
    # claiming numerical agreement with Silva+04 physics.
    template = np.zeros((n_nh, n_wave), dtype=np.float64)
    with h5py.File(silva04_path, "w") as f:
        g = f.create_group("silva04")
        g.create_dataset("log_nh_axis", data=log_nh_axis)
        g.create_dataset("wavelength", data=wavelength)
        g.create_dataset("template", data=template)
    warnings.warn(
        f"Created synthetic Silva+04 grid at {silva04_path} for tests. "
        "Run scripts/build_silva04_grid.py to replace with the real grid.",
        UserWarning,
        stacklevel=1,
    )


def _create_synthetic_ssp_if_missing(ssp_path: Path) -> None:
    """Synthesize a schema-faithful SSP HDF5 grid at the default path if absent.

    Purpose (#613): CI ships no real ``data/ssp_*.h5`` grids, so every
    ``skipif(not _SSP_FILE_WNE.is_file())`` contract test silently skips — the
    exact masking that let the spectroscopy/joint break (#608) and the
    ``xray_log_nh`` KeyError (#768) reach main while CI stayed green. Writing a
    tiny synthetic SSP at the canonical default path lets the *structural*
    contract tests (API surface, cross-component threading, builder grammar)
    actually run on CI. Mirrors ``_create_cb19_fixture_if_missing`` /
    ``_create_silva04_fixture_if_missing``.

    The grid is smooth and broad (91 Å – 1 mm) so it drives the full forward
    chain: ionizing photons below the Lyman limit (nebular), UV/optical
    luminosity for dust energy balance, and a far-IR tail for dust re-emission.
    It is **not** physically calibrated — physics-value tests (``regression_paper``,
    ``crossval``) must guard on the real grid via the ``real_ssp_only`` fixture.

    When the real file is present (dev machine after ``tengri.download_ssp``),
    this is a no-op.
    """
    if ssp_path.exists():
        return
    import warnings

    ssp_path.parent.mkdir(parents=True, exist_ok=True)

    n_met, n_age, n_wave = 5, 22, 1200
    # 91 Å (Lyman limit) – 1 mm (1e7 Å): spans ionizing → far-IR.
    wave = np.logspace(np.log10(91.0), np.log10(1.0e7), n_wave).astype(np.float64)
    # log10(age/Gyr): ~0.3 Myr – 13.8 Gyr.
    lg_age_gyr = np.linspace(-3.5, 1.14, n_age).astype(np.float64)
    # Absolute log10(Z); with LOG10_ZSUN = -1.848 this is logzsol ≈ [-2.15, 0.85].
    lgmet = np.linspace(-4.0, -1.0, n_met).astype(np.float64)

    # Smooth continuum: bright in the UV/optical, ~0 in the far-IR (∝ λ^-1.5),
    # with mild monotonic age (younger = bluer/brighter) and metallicity trends.
    base = (5000.0 / wave) ** 1.5  # (n_wave,)
    age_factor = 1.0 + 0.20 * (lg_age_gyr.mean() - lg_age_gyr)  # younger → larger
    met_factor = 1.0 + 0.10 * (lgmet - lgmet.mean())
    flux = base[None, None, :] * age_factor[None, :, None] * met_factor[:, None, None]
    flux = np.abs(flux).astype(np.float64) + 1e-12  # strictly positive

    # Surviving stellar mass fraction: ~0.97 (young) → ~0.55 (old), monotone.
    frac = np.linspace(0.97, 0.55, n_age)
    mass_remaining = np.broadcast_to(frac[None, :], (n_met, n_age)).astype(np.float64)

    with h5py.File(ssp_path, "w") as f:
        f.create_dataset("ssp_wave", data=wave)
        f.create_dataset("ssp_flux", data=flux)
        f.create_dataset("ssp_lg_age_gyr", data=lg_age_gyr)
        f.create_dataset("ssp_lgmet", data=lgmet)
        f.create_dataset("ssp_mass_remaining", data=mass_remaining)
        f.attrs["synthetic"] = True
        f.attrs["imf"] = "chabrier"

    warnings.warn(
        f"Created synthetic SSP grid at {ssp_path} for tests (#613). "
        "It is NOT physically calibrated — run tengri.download_ssp(...) for the "
        "real grid. Physics-value tests guard on it via the real_ssp_only fixture.",
        UserWarning,
        stacklevel=1,
    )


@pytest.fixture
def rng_key():
    """Default PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def log_age_grid():
    """Standard 256-point log-age grid."""
    return make_log_age_grid(256)


@pytest.fixture
def d_log_age(log_age_grid):
    """Grid spacing."""
    return grid_spacing(log_age_grid)


@pytest.fixture
def drw_params_moderate():
    """Moderate burstiness DRW parameters."""
    return {"psd_sigma": 1.0, "psd_tau_yr": 50e6}  # 50 Myr


@pytest.fixture
def drw_params_smooth():
    """Smooth (low burstiness) DRW parameters."""
    return {"psd_sigma": 0.5, "psd_tau_yr": 200e6}  # 200 Myr


@pytest.fixture
def drw_params_bursty():
    """Highly bursty DRW parameters."""
    return {"psd_sigma": 3.0, "psd_tau_yr": 5e6}  # 5 Myr


@pytest.fixture
def sqrt_power_moderate(d_log_age, drw_params_moderate):
    """Pre-computed amplitude operator for moderate regime."""
    return compute_sqrt_power_drw(
        256,
        float(d_log_age),
        drw_params_moderate["psd_sigma"],
        drw_params_moderate["psd_tau_yr"],
    )


@pytest.fixture
def missing_data_in(monkeypatch):
    """Simulate missing data files for ONE module's lookups (#1693).

    Usage: missing_data_in("tengri.components.dust.emission") patches
    that module's `find_data_str` to return None, isolating the module
    from the real data-lookup path while keeping all other modules intact.
    """
    import importlib

    def _apply(module_path: str):
        """Patch find_data_str in a specific module to return None."""
        mod = importlib.import_module(module_path)
        monkeypatch.setattr(mod, "find_data_str", lambda *a, **k: None)
        return mod

    return _apply
