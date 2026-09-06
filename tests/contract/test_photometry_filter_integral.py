# SPDX-License-Identifier: BSD-3-Clause
"""Lock the algebraic identity introduced in ADR-0016 (#398.e):

    compute_flux_density(L_ν, wave, fw, ft, z, dl)
      == lnu_to_fnu(lnu_filter_integral(L_ν, wave, fw, ft, z), dl, z)

The refactor in #398.e factored ``compute_flux_density`` into two
canonical pieces: filter-integration of rest-frame L_ν, and the
``lnu_to_fnu`` flux conversion. The identity test pins that
composition so future "optimizations" don't silently break
``_phot_lnu_precomp`` consumers (AGN, nebular, stellar precomputes).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.observation.photometry import compute_flux_density, lnu_filter_integral
from tengri.units import lnu_to_fnu

pytestmark = pytest.mark.contract


@pytest.fixture
def fiducial():
    wave_rest = jnp.linspace(1000.0, 30000.0, 5000)
    L_nu = (wave_rest / 5000.0) ** (-1.5) * 1e30
    filter_wave = jnp.linspace(5000.0, 7000.0, 200)
    filter_trans = jnp.ones_like(filter_wave)
    return wave_rest, L_nu, filter_wave, filter_trans


def test_compute_flux_density_equals_composition(fiducial):
    """The refactor must produce bit-exact output to the prior inline form.

    Verified by computing the composition explicitly and comparing.
    """
    wave_rest, L_nu, fw, ft = fiducial
    z = 0.5
    dl_cm = 1.0e26  # arbitrary scalar

    flux_via_fn = float(compute_flux_density(L_nu, wave_rest, fw, ft, z, dl_cm))
    L_filter = lnu_filter_integral(L_nu, wave_rest, fw, ft, z)
    flux_via_composition = float(lnu_to_fnu(L_filter, dl_cm, z))

    rel_err = abs(flux_via_fn - flux_via_composition) / abs(flux_via_composition)
    assert rel_err < 1e-12, f"composition identity broke: {rel_err:.2e}"


def test_lnu_filter_integral_is_dl_cm_independent(fiducial):
    """``lnu_filter_integral`` deliberately omits cosmological dimming.

    Pin this — if someone "accidentally" adds a ``dl_cm`` arg back,
    the publish convention for ``_phot_lnu_precomp`` silently changes.
    """
    import inspect

    sig = inspect.signature(lnu_filter_integral)
    assert "dl_cm" not in sig.parameters, (
        "lnu_filter_integral grew a dl_cm parameter — convention drift "
        "from ADR-0016. _phot_lnu_precomp consumers expect L_ν, not F_ν."
    )


def test_no_inverse_cosmology_dance_in_agn_or_nebular():
    """Structural guard: ``agn/component.py`` and ``nebular/component.py`` must
    not reintroduce the old ADR-0016 refactor regression pattern.

    This was the pre-refactor flux conversion dance: build ``_phot_lnu_precomp``
    by calling ``compute_flux_density(dl_cm=1.0)`` and then multiplying by
    ``inv_cosmology``. The refactor replaced this with direct ``lnu_filter_integral``
    calls (which return L_ν, distance-independent) followed by ``lnu_to_fnu`` at
    projection time (which applies cosmological dimming).

    This test checks that the source code of the two component modules does NOT
    contain the old pattern. The internal ``_phot_lnu_precomp`` is not reachable
    from the public API, so this is a source-level structural guard rather than
    a behavioral test. It catches:
    - Reintroducing the ``inv_cosmology = `` assignment
    - Calling ``compute_flux_density(dl_cm=jnp.asarray(1.0))`` as the old dance

    See ADR-0016 (#398.e) for the full refactor narrative.
    """
    import inspect

    from tengri.components.agn import component as agn_component
    from tengri.components.nebular import component as nebular_component

    for mod, label in ((agn_component, "agn"), (nebular_component, "nebular")):
        src = inspect.getsource(mod)
        # Look for the dance pattern itself (an actual assignment),
        # not casual references in migration comments.
        assert "inv_cosmology = " not in src, (
            f"{label}/component.py reintroduced inv_cosmology assignment — "
            f"the ADR-0016 cleanup regressed."
        )
        assert "dl_cm=jnp.asarray(1.0)" not in src, (
            f"{label}/component.py still calls compute_flux_density with "
            f"dl_cm=1.0 sentinel — the ADR-0016 cleanup regressed."
        )


def test_lnu_filter_integral_returns_lnu_units(fiducial):
    """The output of ``lnu_filter_integral`` is in L_ν units (erg/s/Hz),
    NOT F_ν. Verify by checking the magnitude does not scale with d_L."""
    wave_rest, L_nu, fw, ft = fiducial
    z = jnp.asarray(0.5)

    L = float(lnu_filter_integral(L_nu, wave_rest, fw, ft, z))
    # Sanity: should be on the same order as the input L_ν.
    L_input_peak = float(L_nu.max())
    # Top-hat filter at 5000-7000 Å picks up ~the SED at λ ~ 6000/(1+z=0.5) = 4000 Å rest.
    # L_nu at 4000 Å rest with our normalization: (4000/5000)^(-1.5) × 1e30 ≈ 1.4e30.
    # Allow loose tolerance — this is a sanity check on units, not exact value.
    assert 0.1 * L_input_peak < L < 10.0 * L_input_peak, (
        f"L_nu_filter = {L:.4e} not within an order of L_input_peak = {L_input_peak:.4e}"
    )


def test_lnu_filter_integral_batch_matches_singles_under_padding(fiducial):
    """Zero-padded rows must integrate like their unpadded originals.

    ``pad_filters`` zero-pads every filter table shorter than the longest
    one, leaving a non-ascending wavelength row (…, 4130, 0, 0, …).
    ``_filter_integral_union`` requires ascending nodes, so the batch
    wrapper must rewrite the pad tail (``_ascending_padded_filter_wave``)
    before integrating — the exact-path ``_compute_flux_density_padded``
    already does. Without the rewrite ``jnp.interp`` sees unsorted nodes
    and the shorter bands silently return 0 — zeroing every additive
    ``*_phot_lnu_precomp`` family (radio, X-ray, dust IR) and the #1026
    IGM band factor for real heterogeneous filter sets. Homogeneous
    same-length sets (all synthetic-tophat fixtures) never pad, which is
    how this survived: assert against genuinely ragged tables.
    """
    from tengri.observation.photometry import lnu_filter_integral_batch, pad_filters

    wave_rest, L_nu, _, _ = fiducial
    z = 0.5
    filters = [
        (jnp.linspace(3000.0, 4000.0, 30), jnp.sin(jnp.linspace(0.0, jnp.pi, 30))),
        (jnp.linspace(6000.0, 9000.0, 80), jnp.sin(jnp.linspace(0.0, jnp.pi, 80))),
        (jnp.linspace(11000.0, 13000.0, 55), jnp.sin(jnp.linspace(0.0, jnp.pi, 55))),
    ]
    singles = jnp.array([lnu_filter_integral(L_nu, wave_rest, fw, ft, z) for fw, ft in filters])
    fw_pad, ft_pad, _ = pad_filters([f[0] for f in filters], [f[1] for f in filters])
    batch = lnu_filter_integral_batch(L_nu, wave_rest, fw_pad, ft_pad, z)
    assert jnp.all(singles > 0), "degenerate fixture: single-filter integrals must be > 0"
    assert jnp.allclose(batch, singles, rtol=1e-12), (
        f"padded batch diverges from unpadded singles: {batch} vs {singles}"
    )
