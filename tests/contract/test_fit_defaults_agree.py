# SPDX-License-Identifier: BSD-3-Clause
"""The fit surfaces agree about what a fit is (#1289).

Five surfaces started a fit and three of them disagreed about the default:

    ForwardModel.fit   'vi'                  <- canonical
    Fitter.run         'vi_nonlinear_fast'   <- the engine ForwardModel.fit calls
    SEDModel.fit       'vi'                  <- sugar over ForwardModel.fit
    fit_batch          'vi'
    Galaxy.fit         'map'                 <- and a different kwarg name

plus a sixth answer in ``defaults.toml`` (``method = "auto"``). The sharpest
edge was that ``ForwardModel.fit`` *delegates to* ``Fitter``:

    forward.fit(data, noise)                # ran "vi"
    Fitter(forward, data, noise).run()      # ran "vi_nonlinear_fast"

Same objects, same data, two backends, no warning. ``'vi'`` and
``'vi_nonlinear_fast'`` are the same geoVI algorithm — both pass
``sample_mode="nonlinear_resample"``, differing only in Python logging — so
aligning them preserves the posterior.

``Galaxy.fit`` keeps ``"map"`` on purpose; the test below pins that as a
*documented* difference rather than an accident.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import tengri
from tengri.inference._backend_registry import DEFAULT_METHOD, get_backend

pytestmark = pytest.mark.contract

#: Surfaces that must share the one default.
SHARED_DEFAULT = {
    "ForwardModel.fit": lambda: tengri.ForwardModel.fit,
    "Fitter.run": lambda: tengri.Fitter.run,
    "SEDModel.fit": lambda: tengri.SEDModel.fit,
    "fit_batch": lambda: tengri.fit_batch,
}


@pytest.mark.parametrize("label", sorted(SHARED_DEFAULT))
def test_every_fit_surface_shares_one_default(label):
    fn = SHARED_DEFAULT[label]()
    param = inspect.signature(fn).parameters.get("method")
    assert param is not None, f"{label} has no 'method' argument"
    assert param.default == DEFAULT_METHOD, (
        f"{label} defaults to {param.default!r}, not DEFAULT_METHOD "
        f"({DEFAULT_METHOD!r}). Which doc a user follows must not determine "
        "their inference."
    )


def test_the_default_is_a_real_working_backend():
    """Guard the guard: a DEFAULT_METHOD naming a phantom would pass above."""
    entry = get_backend(DEFAULT_METHOD)
    assert entry.tier != "broken", f"DEFAULT_METHOD={DEFAULT_METHOD!r} is tier='broken' (#1287)"


def test_galaxy_fit_takes_method_not_backend():
    sig = inspect.signature(tengri.Galaxy.fit)
    assert "method" in sig.parameters, "Galaxy.fit must accept 'method'"
    assert "backend" in sig.parameters, "'backend' must remain as a deprecated alias"


class _RecordingFitter:
    """Captures the method Galaxy.fit dispatches, without running inference."""

    seen: str | None = None

    def __init__(self, *a, **k):
        pass

    def run(self, method, **kw):
        _RecordingFitter.seen = method
        return "RESULT"


class _RecordingBibliography:
    def __init__(self):
        self.cited: list = []

    def add_backend(self, m):
        self.cited.append(m)


def _stub_galaxy():
    """A Galaxy with just enough state for ``fit`` to reach dispatch."""
    g = tengri.Galaxy.__new__(tengri.Galaxy)
    g.build_model = lambda: None
    g._flux_obs = [1.0]
    g._noise = [0.1]
    g.model = None
    g.bibliography = _RecordingBibliography()
    return g


def _run_fit(**kwargs):
    """Dispatch a fit against the recording fitter; return (method, galaxy)."""
    from unittest.mock import patch

    import tengri.facade as facade

    g = _stub_galaxy()
    with patch.object(facade, "Fitter", _RecordingFitter):
        tengri.Galaxy.fit(g, **kwargs)
    return _RecordingFitter.seen, g


def test_galaxy_bare_fit_still_defaults_to_map():
    """The facade's documented, deliberate divergence — pinned, not accidental."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method, _ = _run_fit()
    assert method == "map"
    assert not caught, "the ordinary path must be silent"


def test_galaxy_backend_alias_warns_and_is_honored():
    """The alias must keep working, loudly — asserted by behavior, not source."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method, _ = _run_fit(backend="vi")
    assert method == "vi", "the deprecated alias must still select the backend"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "Galaxy.fit(backend=) must warn, not silently accept"
    )


def test_galaxy_method_kwarg_is_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method, _ = _run_fit(method="mcmc_nuts")
    assert method == "mcmc_nuts"
    assert not caught, "the new spelling must not warn"


def test_passing_both_method_and_backend_is_an_error():
    """Two names for one concept must not be allowed to disagree silently."""
    with pytest.raises(TypeError, match="backend"):
        _run_fit(method="vi", backend="map")


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, "map"), ({"backend": "vi"}, "vi"), ({"method": "mcmc_nuts"}, "mcmc_nuts")],
)
def test_the_citation_records_the_method_actually_run(kwargs, expected):
    """Regression: the citation hook read the raw ``backend`` alias.

    After ``backend`` became an optional alias defaulting to ``None``, that
    line recorded ``None`` for every call that used the new spelling — three
    of four paths — silently dropping the inference citation. Inspecting the
    source would not have caught it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _, g = _run_fit(**kwargs)
    assert g.bibliography.cited == [expected], (
        f"citation recorded {g.bibliography.cited} for {kwargs}, expected [{expected!r}]"
    )


def test_defaults_toml_advertises_only_real_methods():
    """The user-facing config file must not name backends that raise.

    It listed "vi_nifty_fast", "vi_nifty_fast_linear", "vi_native" and
    "vi_native_linear" — four of fifteen options, none registered.
    """
    import re
    from pathlib import Path

    import tengri.inference._backend_registry as reg

    toml = Path(tengri.__file__).parent / "defaults.toml"
    text = toml.read_text()
    block = text.split("[inference]")[1].split("[inference.")[0]

    quoted = set(re.findall(r'"([a-z0-9_]+)"', block))
    # "auto" is resolved by a dispatch shim in Fitter.run, not a registry entry.
    candidates = {q for q in quoted if q.startswith(("vi", "mcmc", "map", "laplace", "nss"))}
    candidates -= {"auto"}

    registered = set(reg._BACKENDS)
    phantom = sorted(candidates - registered)
    assert not phantom, (
        f"defaults.toml advertises inference methods that are not registered "
        f"and raise KeyError if set: {phantom}"
    )


def test_the_phantom_names_really_do_raise():
    """Guard the guard: proves the test above is checking something real."""
    for name in ("vi_native", "vi_nifty_fast", "vi_native_linear"):
        with pytest.raises((ValueError, KeyError)):
            get_backend(name)


def test_forward_fit_and_fitter_run_no_longer_diverge():
    """The specific pair that made the divergence dangerous."""
    fwd = inspect.signature(tengri.ForwardModel.fit).parameters["method"].default
    fit = inspect.signature(tengri.Fitter.run).parameters["method"].default
    assert fwd == fit, (
        f"ForwardModel.fit defaults to {fwd!r} but delegates to Fitter.run, "
        f"which defaults to {fit!r}. forward.fit(d, n) and "
        "Fitter(forward, d, n).run() would run different backends."
    )


def test_no_deprecation_warning_on_the_canonical_path():
    """Aligning the defaults must not make ordinary use noisy."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        inspect.signature(tengri.ForwardModel.fit)
        inspect.signature(tengri.Fitter.run)


def test_sedmodel_fit_emits_no_deprecation_warning(ssp_data_wne, simple_observation):
    """SEDModel.fit is un-deprecated sugar — it must not warn.

    Regression for #1322 Wave 3: SEDModel.fit was un-deprecated as the
    Bagpipes one-liner. Verify it emits NO DeprecationWarning on actual use.
    """
    from tengri import FIXED

    sed = tengri.SEDModel.build(
        ssp_data=ssp_data_wne,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )
    # Generate a trivial mock for fitting
    params = {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 5.0,
        "sfh_dpl_age_gyr": 10.0,
        "sfh_dpl_log_total_mass": 0.5,
        "met_logzsol": 0.0,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.2,
        "dust_slope": -0.7,
        "redshift": 0.05,
    }
    mock = sed.mock(params, snr=5.0, key=__import__("jax").random.PRNGKey(0))

    # Verify sed.fit(method="map") emits NO DeprecationWarning (excluding unrelated JAX warnings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sed.fit(mock.flux_obs, mock.noise, method="map")

    # sed.fit is un-deprecated sugar, so it must emit NO tengri DeprecationWarning
    # — neither its own (removed) warning nor the internal Fitter(sed_model) one
    # (suppressed in fit_model). Any non-third-party DeprecationWarning is a
    # regression. We exclude only JAX's own version-deprecation chatter, matched by
    # its message (never by a tengri-specific substring, so re-introducing ANY
    # tengri deprecation text — e.g. the original "SEDModel.fit is deprecated" —
    # fails this guard rather than slipping past a tailored string match).
    tengri_deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "jax" not in str(w.message).lower()
    ]
    assert not tengri_deprecations, (
        "SEDModel.fit (un-deprecated sugar) must emit no tengri DeprecationWarning; "
        f"got: {[str(w.message) for w in tengri_deprecations]}"
    )

    # Sanity check: the fit did run and returned a Posterior
    from tengri.inference.posterior import Posterior

    assert isinstance(result, Posterior)
