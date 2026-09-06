# SPDX-License-Identifier: BSD-3-Clause
r"""Every dust-law parameter the grammar admits must reach the curve (#2185).

The attenuation laws each took a ``**_kwargs`` catch-all, so a parameter the
grammar accepted and routed to the law was absorbed at the function signature
and discarded. Nothing raised at any layer: the grammar accepted the key, the
spec declared it free, the sampler explored it, and ``k(lambda)`` never moved.

Six of the seven pairs the issue names are worse than plain inertness, because
the *working* spelling sits beside the inert one in the same group:

===============  ============================  ==================
law              ``slope`` -> ``dust_slope``   ``delta``
===============  ============================  ==================
``noll09``       inert, exactly 0.0            live, 1.79e+02
``salim_sbl18``  inert, exactly 0.0            live, 1.79e+02
``tea``          inert, exactly 0.0            live, 1.80e+02
``narayanan_z``  inert, exactly 0.0            live, 1.89e+02
===============  ============================  ==================

plus ``vw07_bc`` / ``vw07_diff`` accepting ``slope`` against a hardcoded
n = -1.3 / n = -0.7, and ``calzetti`` accepting ``Rv`` against a hardcoded
R_V = 4.05. Those three values are not free parameters of their papers:
Calzetti et al. (2000) [1]_ measure R_V = 4.05 +/- 0.80 for the starburst
sample and the piecewise polynomial is fitted at that value, and Wild et al.
(2007) [2]_ give the two power-law slopes as fitted constants of the
birth-cloud and diffuse screens. So the decision at the declaration is that the
laws keep the constants and stop accepting the parameter, not that the constant
becomes free.

The shared spellings are refused by the group's variant scope. The per-screen
spellings (``slope_bc`` / ``slope_diff`` / ``slope_neb`` and friends) route
past the parameter partition as structural keys carrying a static float, and
were not scoped at all: measured across the 22 registered laws, **72 of the 88
(law, per-screen key) pairs left the photometry bit-identical to omitting the
key**. All 72 now raise.

What this file pins, per law, through the public API:

1. every parameter ``all_params: FREE`` frees moves the photometry;
2. every per-screen key the grammar admits moves the photometry;
3. every per-screen key it refuses raises :class:`ParameterError`;
4. the issue's seven (law, parameter) pairs raise on build;
5. ``dust_tau_diff`` moves in the same model -- the in-model control that says
   a null result is the parameter's, not the fixture's.

References
----------
.. [1] D. Calzetti, L. Armus, R. C. Bohlin, A. L. Kinney, J. Koornneef, and
   T. Storchi-Bergmann, "The Dust Content and Opacity of Actively Star-forming
   Galaxies," ApJ, 533, 682 (2000). https://doi.org/10.1086/308692
.. [2] V. Wild, G. Kauffmann, T. Heckman, S. Charlot, G. Lemson, J. Brinchmann,
   T. Reichard, and A. Pasquali, "Bursty stellar populations and obscured
   active galactic nuclei in galaxy bulges," MNRAS, 381, 543 (2007).
   https://doi.org/10.1111/j.1365-2966.2007.12253.x
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.components.dust.attenuation import TWO_COMPONENT_OVERRIDE_KEYS
from tengri.components.dust.laws._registry import law_kwarg_names, list_laws
from tengri.config.exceptions import ParameterError
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Relative photometry change below which a parameter counts as inert. Measured
#: on this fixture, the 74 live (law, parameter) measurements span
#: 5.70e-03 (``conroy2010`` ``slope_bc``/``slope_diff``) to 4.40e+04
#: (``kriek_conroy`` ``dust_delta``), and the inert ones were exactly 0.0. The
#: floor sits three orders of magnitude below the weakest live effect rather
#: than at the edge of it.
_LIVE_FLOOR = 1e-6

#: Per-screen key stem -> two values well inside the parameter's declared prior.
#: ``dust_slope`` Uniform(-1.5, -0.3), ``dust_Rv`` Uniform(2.0, 6.0),
#: ``dust_delta`` Uniform(-1.0, 0.4), ``dust_bump_strength`` Uniform(0.0, 2.0).
_SCREEN_SWEEP: dict[str, tuple[float, float]] = {
    "slope": (-1.4, -0.4),
    "Rv": (2.2, 5.8),
    "delta": (-0.9, 0.3),
    "bump_strength": (0.1, 1.9),
}

#: The seven (law, shared key) pairs the issue measured as exactly inert.
_ISSUE_PAIRS = (
    ("noll09", "slope"),
    ("salim_sbl18", "slope"),
    ("tea", "slope"),
    ("narayanan_z", "slope"),
    ("vw07_bc", "slope"),
    ("vw07_diff", "slope"),
    ("calzetti", "Rv"),
)


def _all_laws() -> tuple[str, ...]:
    """Every registered law, read from the live registry.

    A hardcoded menu is how #1482 and #1833 were missed; a law registered later
    is covered here without editing this file.
    """
    return tuple(sorted(row["name"] for row in list_laws(headline=False)))


@pytest.fixture(scope="module")
def dust_ssp() -> SSPData:
    """SSP with a Lyman break, so the UV is not an unphysical tail."""
    ages = jnp.linspace(-3.0, 1.14, 20)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 700)
    base = ((5000.0 / wave) ** 2 * jnp.where(wave < 912.0, 1e-6, 1.0))[None, None, :]
    flux = (
        base
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def uv_obs() -> Observation:
    """Bands straddling the 2175 A bump, where the shape parameters act."""

    def _tophat(center: float, n: int = 24) -> FilterCurve:
        wave = jnp.linspace(center * 0.8, center * 1.2, n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (1500.0, 2800.0, 3500.0, 6200.0)))
    )


def _build(ssp, obs, atten):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=atten,
            dust_emission={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )


def _two(law, **extra):
    return {"type": "two_component", "law": law, **extra}


def _phot(model, params):
    return np.asarray(model.predict_photometry(params))


def _rel(a, b):
    return float(np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-300)))


def _sweep_free(model, name):
    """Max relative photometry change from moving one free parameter."""
    free = list(model.spec.free_params)
    base = {}
    for other in free:
        lo, hi = (float(v) for v in model.spec.get_distribution(other).bounds)
        base[other] = jnp.asarray(0.5 * (lo + hi))
    lo, hi = (float(v) for v in model.spec.get_distribution(name).bounds)
    span = hi - lo
    low = dict(base, **{name: jnp.asarray(lo + 0.05 * span)})
    high = dict(base, **{name: jnp.asarray(hi - 0.05 * span)})
    return _rel(_phot(model, low), _phot(model, high))


def _admitted_screen_keys(law: str) -> tuple[str, ...]:
    """Per-screen stems the grammar admits for ``law`` on both screens."""
    reads = law_kwarg_names(law)
    return tuple(stem for stem, law_kw in TWO_COMPONENT_OVERRIDE_KEYS.items() if law_kw in reads)


class TestTheIssuePairs:
    """The seven (law, shared key) pairs the issue measured as exactly inert."""

    @pytest.mark.parametrize("law,key", _ISSUE_PAIRS)
    def test_shared_key_raises(self, dust_ssp, uv_obs, law, key):
        with pytest.raises(ParameterError) as exc:
            _build(dust_ssp, uv_obs, _two(law, all_params=FREE, **{key: FREE}))
        message = str(exc.value)
        assert f"'{key}'" in message
        assert law in message

    def test_noll09_names_the_parameter_it_does_read(self, dust_ssp, uv_obs):
        """The near miss is the point: ``delta`` is three characters away."""
        with pytest.raises(ParameterError) as exc:
            _build(dust_ssp, uv_obs, _two("noll09", all_params=FREE, slope=FREE))
        assert "delta" in str(exc.value)

    @pytest.mark.parametrize("law,key", _ISSUE_PAIRS)
    def test_per_screen_spelling_raises_too(self, dust_ssp, uv_obs, law, key):
        """``slope_bc``/``slope_diff`` was the half the group scope did not reach."""
        lo, hi = _SCREEN_SWEEP[key]
        with pytest.raises(ParameterError) as exc:
            _build(
                dust_ssp,
                uv_obs,
                _two(law, all_params=Fixed(DEFAULT), **{f"{key}_bc": lo, f"{key}_diff": hi}),
            )
        assert f"'{key}_bc'" in str(exc.value)
        assert law in str(exc.value)


class TestNoAdmittedParameterIsInert:
    """Whatever the grammar admits for a law must move that law's photometry."""

    @pytest.mark.parametrize("law", _all_laws())
    def test_the_control_moves(self, dust_ssp, uv_obs, law):
        """``dust_tau_diff`` is the in-model control for the two tests below."""
        model = _build(dust_ssp, uv_obs, _two(law, all_params=FREE))
        assert _sweep_free(model, "dust_tau_diff") > _LIVE_FLOOR

    @pytest.mark.parametrize("law", _all_laws())
    def test_every_wildcard_freed_parameter_moves(self, dust_ssp, uv_obs, law):
        model = _build(dust_ssp, uv_obs, _two(law, all_params=FREE))
        inert = [
            name for name in model.spec.free_params if _sweep_free(model, name) <= _LIVE_FLOOR
        ]
        assert not inert, (
            f"law {law!r}: 'all_params: FREE' freed {inert}, and moving them across "
            f"their declared prior left predict_photometry unchanged. A freed "
            f"parameter the curve never reads is a flat direction the sampler pays "
            f"for (#2185)."
        )

    @pytest.mark.parametrize("law", _all_laws())
    def test_every_admitted_per_screen_key_moves(self, dust_ssp, uv_obs, law):
        admitted = _admitted_screen_keys(law)
        base_model = _build(dust_ssp, uv_obs, _two(law, all_params=Fixed(DEFAULT)))
        pinned = {"dust_tau_bc": jnp.asarray(1.0), "dust_tau_diff": jnp.asarray(0.5)}
        base = _phot(base_model, pinned)
        for stem in admitted:
            lo, hi = _SCREEN_SWEEP[stem]
            model = _build(
                dust_ssp,
                uv_obs,
                _two(law, all_params=Fixed(DEFAULT), **{f"{stem}_bc": lo, f"{stem}_diff": hi}),
            )
            moved = _rel(base, _phot(model, pinned))
            assert moved > _LIVE_FLOOR, (
                f"law {law!r}: the grammar admits {stem}_bc/{stem}_diff, but setting "
                f"them to ({lo}, {hi}) left predict_photometry bit-identical to "
                f"omitting them (rel {moved:.3e}). Admitted and inert is #2185."
            )

    @pytest.mark.parametrize("law", _all_laws())
    def test_every_refused_per_screen_key_raises(self, dust_ssp, uv_obs, law):
        refused = set(TWO_COMPONENT_OVERRIDE_KEYS) - set(_admitted_screen_keys(law))
        for stem in sorted(refused):
            lo, hi = _SCREEN_SWEEP[stem]
            with pytest.raises(ParameterError, match=rf"{stem}_bc"):
                _build(
                    dust_ssp,
                    uv_obs,
                    _two(law, all_params=Fixed(DEFAULT), **{f"{stem}_bc": lo, f"{stem}_diff": hi}),
                )


class TestTheLawsRefuseWhatTheyCannotRead:
    """The mechanism, one layer below the grammar."""

    @pytest.mark.parametrize("law", _all_laws())
    def test_no_law_declares_a_kwargs_catchall(self, law):
        """A catch-all is what made every layer above it silent."""
        import inspect

        from tengri.components.dust.laws._registry import _law_callable

        sig = inspect.signature(_law_callable(law))
        catchalls = [p.name for p in sig.parameters.values() if p.kind is p.VAR_KEYWORD]
        assert not catchalls, (
            f"law {law!r} declares {catchalls}: it will ACCEPT any parameter and "
            f"read none of them, which is the #2185 mechanism."
        )

    def test_calzetti_refuses_the_rv_it_hardcodes(self):
        """R_V = 4.05 is fitted into the polynomial, not a free knob."""
        from tengri.components.dust.attenuation import calzetti

        with pytest.raises(TypeError, match="dust_Rv"):
            calzetti(jnp.asarray([5500.0]), dust_Rv=3.1)

    @pytest.mark.parametrize("law", ("vw07_bc", "vw07_diff"))
    def test_vw07_refuses_the_slope_it_hardcodes(self, law):
        """Wild et al. (2007) give n = -1.3 / -0.7 as fitted constants."""
        from tengri.components.dust.laws._registry import resolve_dust_law

        with pytest.raises(TypeError, match="n_slope"):
            resolve_dust_law(law)(jnp.asarray([5500.0]), n_slope=-1.0)


class TestMixedLawScreensStillWork:
    """Narrowing per screen is what lets the laws drop the catch-all."""

    def test_each_screen_gets_only_its_own_law_parameters(self, dust_ssp, uv_obs):
        """``calzetti`` + ``cardelli``: ``Rv`` is the diffuse screen's alone."""
        model = _build(
            dust_ssp,
            uv_obs,
            {
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "cardelli",
                "all_params": FREE,
            },
        )
        assert "dust_Rv" in model.spec.free_params
        assert _sweep_free(model, "dust_Rv") > _LIVE_FLOOR

    def test_a_lone_screen_key_is_complete_when_the_partner_cannot_read_it(self, dust_ssp, uv_obs):
        """``power_law`` reads a slope; ``noll09`` has none to give."""
        model = _build(
            dust_ssp,
            uv_obs,
            {
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "all_params": Fixed(DEFAULT),
                "slope_bc": -1.4,
            },
        )
        base = _build(
            dust_ssp,
            uv_obs,
            {
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "all_params": Fixed(DEFAULT),
            },
        )
        pinned = {"dust_tau_bc": jnp.asarray(1.0), "dust_tau_diff": jnp.asarray(0.5)}
        assert _rel(_phot(base, pinned), _phot(model, pinned)) > _LIVE_FLOOR


def test_per_screen_keys_are_refused_on_a_single_screen_model(dust_ssp, uv_obs):
    """``single_component`` never routes ``dust_law_overrides``; it dropped them."""
    with pytest.raises(ParameterError, match="slope_bc"):
        _build(
            dust_ssp,
            uv_obs,
            {
                "type": "single_component",
                "law": "power_law",
                "all_params": Fixed(DEFAULT),
                "slope_bc": -1.4,
            },
        )
