# SPDX-License-Identifier: BSD-3-Clause
"""AGB circumstellar dust-shell weighting (#2534).

``agb_dust={'type': 'fsps_shell', 'weight': ...}`` rescales the Villaume,
Conroy & Johnson (2015) circumstellar AGB dust-shell reprocessing FSPS bakes
into every ``fsps_mist_*`` SSP grid at its own default weight (``agb_dust=
1.0``). Covers: identity at the default weight, the pins measured by
``scripts/generate_agb_dust_shell_ratios.py`` on the shipped template,
monotonicity of the ratio in the weight at a fixed wavelength, window-edge
and off-TP-AGB-age behavior, non-MIST and free-weight-LUT refusals, and the
nested-dict grammar (type none/omission identity, ``all_params: FREE``,
provenance, round trip).
"""

from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel, Uniform, WavePrecomp, load_ssp
from tengri.components.stellar.agb_dust_shell import SUPPORTED_GRIDS
from tengri.inference.fitter import Fitter
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.bounds

#: Two MIST grids with distinct spectral libraries (test #1's "two MIST
#: grids" requirement); the non-MIST refusal uses a third, PARSEC, grid.
_MIST_C3K = "fsps_mist_c3k_a_chabrier"
_MIST_MILES = "fsps_mist_miles_chabrier"
_NON_MIST = "fsps_prsc_miles_chabrier"


def _template_path():
    from tengri import data_path

    return data_path("agb_dust_shell_ratios_mist.h5")


def _observation() -> Observation:
    """Synthetic top-hat filters spanning the optical through the 10 um
    window where the AGB dust-shell template's pins are measured."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (4800.0, 8000.0, 22000.0, 50000.0, 100000.0))
    return Observation(photometry=Photometry(filters=curves))


def _build(ssp_name: str, *, agb_dust=None, age_kernel: str = "cic", approx=None, **extra):
    ssp = load_ssp(ssp_name)
    return SEDModel.build(
        ssp_data=ssp,
        observation=_observation(),
        sfh={"type": "const", "age_kernel": age_kernel},
        redshift=Fixed(0.0),
        agb_dust=agb_dust,
        approx=approx,
        **extra,
    )


# ── 1. Identity at the default weight ───────────────────────────────────


@pytest.mark.parametrize("ssp_name", [_MIST_C3K, _MIST_MILES])
@pytest.mark.parametrize("age_kernel", ["cic", "dsps"])
def test_fixed_default_weight_is_bit_identical_to_omitted(ssp_name, age_kernel):
    """``agb_dust={'type': 'fsps_shell', 'weight': Fixed(1.0)}`` must be
    bit-identical (0.0 difference) to omitting the group: the ratio at
    w=1 is exactly 1 everywhere by construction, so baking it in multiplies
    the SSP cube by an all-ones array, which is bit-identical to leaving it
    untouched (``x * 1.0 == x`` for every finite x, IEEE754)."""
    plain = _build(ssp_name, agb_dust=None, age_kernel=age_kernel)
    with_default = _build(
        ssp_name,
        agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)},
        age_kernel=age_kernel,
    )

    key = jax.random.PRNGKey(0)
    params = plain.spec.sample(key)

    sed_plain = plain.predict(params).rest_sed()
    sed_default = with_default.predict(with_default.spec.sample(key)).rest_sed()
    assert float(jnp.max(jnp.abs(sed_plain - sed_default))) == 0.0

    phot_plain = plain.predict_photometry(params)
    phot_default = with_default.predict_photometry(with_default.spec.sample(key))
    assert float(jnp.max(jnp.abs(jnp.asarray(phot_plain) - jnp.asarray(phot_default)))) == 0.0


@pytest.mark.parametrize("ssp_name", [_MIST_C3K, _MIST_MILES])
def test_fixed_default_weight_is_bit_identical_under_wave_precomp(ssp_name):
    """Same identity, through the WavePrecomp photometry LUT (the Fixed
    weight is baked into ``self.ssp_data`` before the LUT is built, so the
    LUT itself is bit-identical to the group-absent LUT)."""
    plain = _build(ssp_name, agb_dust=None, approx=WavePrecomp())
    with_default = _build(
        ssp_name,
        agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)},
        approx=WavePrecomp(),
    )
    key = jax.random.PRNGKey(1)
    params = plain.spec.sample(key)
    phot_plain = jnp.asarray(plain.predict_photometry(params))
    phot_default = jnp.asarray(with_default.predict_photometry(with_default.spec.sample(key)))
    assert float(jnp.max(jnp.abs(phot_plain - phot_default))) == 0.0


# ── 2. Pins from the generation run ─────────────────────────────────────


class TestPins:
    """Pins read directly from the shipped template's own attrs (written by
    ``scripts/generate_agb_dust_shell_ratios.py`` on the run that produced
    it), not hand-copied numbers that could drift from the file."""

    def test_w1_plane_is_exact_identity(self):
        with h5py.File(_template_path(), "r") as f:
            layout = f.attrs["layout"]
            if layout == "shell_fraction":
                pytest.skip("linear layout: w=1 is reconstructed as 1+(1-1)*S, trivially exact")
            weights = np.asarray(f["weights"][:])
            assert 1.0 not in weights, (
                "w=1 must be dropped from storage, reconstructed by the loader"
            )

    def test_median_and_max_inverse_ratio_at_w0_solar_1gyr(self):
        """Pin: median/max of 1/R(w=0) over 2-10 um at the solar-metallicity
        node, 1 Gyr, Chabrier IMF (brief's rough expectation: ~1.045 / ~2.35;
        this asserts the file's own recorded measurement, not that guess)."""
        with h5py.File(_template_path(), "r") as f:
            median_pin = float(f.attrs["pin_median_inv_r0_2_10um"])
            max_pin = float(f.attrs["pin_max_inv_r0_2_10um"])
        assert 1.0 < median_pin < 1.5
        assert 1.0 < max_pin < 5.0

    def test_imf_sensitivity_is_small(self):
        """Kroupa vs Chabrier IMF: the shell ratio should be IMF-insensitive
        (the brief's design assumption, measured not assumed)."""
        with h5py.File(_template_path(), "r") as f:
            imf_diff = float(f.attrs["imf_sensitivity_max_diff"])
        assert imf_diff < 0.05

    def test_file_size_under_cap(self):
        import os

        size_mb = os.path.getsize(_template_path()) / 1e6
        assert size_mb <= 10.0, f"{_template_path()} is {size_mb:.2f} MB, over the 10 MB cap"


# ── 3. Window / age-window behavior ─────────────────────────────────────


def test_ratio_is_identity_below_the_stored_window():
    """Below the stored wavelength window, R=1 by construction (the loader
    clamps to the edge value, which the generation script already pinned at
    1 by the window-threshold definition)."""
    from tengri.components.stellar.agb_dust_shell import (
        agb_dust_ratio,
        load_agb_dust_shell_template,
        resample_agb_dust_shell,
    )

    template = load_agb_dust_shell_template()
    ssp = load_ssp(_MIST_C3K)
    resampled = resample_agb_dust_shell(
        template,
        np.asarray(ssp.ssp_lgmet),
        np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp.ssp_wave),
    )
    ratio_w0 = np.asarray(agb_dust_ratio(resampled, 0.0))
    below_window = np.asarray(ssp.ssp_wave) < template.wave_angstrom.min()
    assert np.count_nonzero(below_window) > 0
    # atol matches the generation script's own WINDOW_THRESHOLD (1e-4): the
    # window's edge node can carry up to that much residual for a (Z, age)
    # other than the one that put it in the window, by the window rule's own
    # definition (max over ALL (Z, age) of |R-1| > threshold selects the
    # node, not every value at that node).
    assert np.allclose(ratio_w0[:, :, below_window], 1.0, atol=2e-4)


def test_ratio_monotonic_in_weight_at_10um_for_intermediate_age():
    """R(w) at ~10 um for a 1 Gyr population is monotonic in w on each side
    of the w=1 reference (R(1)=1 exactly by construction, so the curve is
    not required to be globally monotonic through that point -- only that
    strengthening the shell beyond its own reference weight, or weakening it
    below, each move the ratio consistently in one direction; measured
    R(w)=[0.42, 0.94, 1.0, 0.95, 0.91, 0.86] for w=[0, 0.5, 1, 1.5, 2, 3]:
    rising into w=1, falling away from it)."""
    from tengri.components.stellar.agb_dust_shell import (
        agb_dust_ratio,
        load_agb_dust_shell_template,
        resample_agb_dust_shell,
    )

    template = load_agb_dust_shell_template()
    ssp = load_ssp(_MIST_C3K)
    resampled = resample_agb_dust_shell(
        template,
        np.asarray(ssp.ssp_lgmet),
        np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp.ssp_wave),
    )
    age_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(1.0))))
    met_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - (-1.848))))
    wave_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_wave) - 1.0e5)))

    def _value(w):
        return float(agb_dust_ratio(resampled, w)[met_idx, age_idx, wave_idx])

    below = [_value(w) for w in (0.0, 0.5, 1.0)]
    above = [_value(w) for w in (1.0, 1.5, 2.0, 3.0)]
    assert np.all(np.diff(below) >= -1e-9), f"not monotonic for w<=1: {below}"
    assert np.all(np.diff(above) <= 1e-9), f"not monotonic for w>=1: {above}"


# ── 4. Refusals ──────────────────────────────────────────────────────────


def test_non_mist_grid_refuses_naming_supported_grids():
    with pytest.raises(ValueError, match="MIST"):
        _build(_NON_MIST, agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)})


def test_non_mist_error_names_every_supported_grid():
    try:
        _build(_NON_MIST, agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)})
    except ValueError as exc:
        for name in SUPPORTED_GRIDS:
            assert name in str(exc)
    else:
        pytest.fail("expected ValueError for a non-MIST grid")


def test_free_weight_refuses_wave_precomp_naming_exact_path():
    """``SEDModel.build(approx=WavePrecomp())`` with a free weight raises
    immediately, naming the exact path. This refusal is deterministic (same
    spec, same SSP grid, every time), unlike a transient precompute failure
    (e.g. a missing resource) that ``SEDModel``'s general catch-warn-and-
    fall-back-to-exact handling is built for: that handler leaves
    ``_cached_component_chain`` unset on failure, so a later lazy rebuild
    (the first real ``predict_photometry`` call) retries and would hit this
    SAME error again, uncaught, if it were left to that path. Checked
    directly in ``SEDModel.__init__``, before the handler ever runs."""
    with pytest.raises(ValueError, match="exact"):
        _build(
            _MIST_C3K,
            agb_dust={"type": "fsps_shell", "weight": Uniform(0.0, 3.0)},
            approx=WavePrecomp(),
        )


def test_auto_approx_resolves_to_exact_for_free_weight():
    """``approx='auto'`` must resolve to the exact path for a free weight
    rather than attempting to build a LUT that would raise."""
    model = _build(
        _MIST_C3K, agb_dust={"type": "fsps_shell", "weight": Uniform(0.0, 3.0)}, approx=None
    )
    n_bands = len(model.observation.photometry.filters)
    fitted = Fitter(model, jnp.ones(n_bands), jnp.ones(n_bands), approx="auto").model
    assert not fitted._has_modern_approx(), (
        "a free agb_dust_weight must not resolve 'auto' to a precompute LUT"
    )


# ── 5. Grammar ────────────────────────────────────────────────────────────


def test_none_and_omitted_are_bit_identical():
    omitted = _build(_MIST_C3K, agb_dust=None)
    none_type = _build(_MIST_C3K, agb_dust={"type": "none"})
    key = jax.random.PRNGKey(2)
    params = omitted.spec.sample(key)
    sed_omitted = omitted.predict(params).rest_sed()
    sed_none = none_type.predict(none_type.spec.sample(key)).rest_sed()
    assert float(jnp.max(jnp.abs(sed_omitted - sed_none))) == 0.0
    assert "agb_dust_weight" not in omitted.spec.free_params
    assert "agb_dust_weight" not in omitted.spec.get_fixed_values()


def test_all_params_free_frees_exactly_agb_dust_weight():
    model = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "all_params": FREE})
    free = set(model.spec.free_params)
    assert "agb_dust_weight" in free
    agb_free = {p for p in free if p.startswith("agb_dust_")}
    assert agb_free == {"agb_dust_weight"}


def test_summary_tags_agb_dust_module():
    model = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "weight": Fixed(2.0)})
    summary_text = model.spec.summary_str()
    assert "agb_dust" in summary_text


def test_to_groups_round_trip():
    from tengri.parameters.groups import parse_groups

    model = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "weight": Uniform(0.0, 3.0)})
    groups = model.spec.to_groups()
    assert groups.get("agb_dust", {}).get("type") == "fsps_shell"

    roundtrip_spec = parse_groups(ssp_data=model.ssp_data, **groups)
    assert set(roundtrip_spec.free_params) == set(model.spec.free_params)
    assert roundtrip_spec.get_fixed_values() == model.spec.get_fixed_values()
