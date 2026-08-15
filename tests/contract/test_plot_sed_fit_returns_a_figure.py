# SPDX-License-Identifier: BSD-3-Clause
"""``plot_sed_fit`` returned a Figure or an Axes depending on the data given.

The branch was::

    if show_residuals and posterior_draws is not None:
        return fig  # Figure
    else:
        return ax_main  # Axes

so what the caller held depended on whether they had passed
``posterior_draws`` — an *optional* argument for uncertainty shading. Measured
across the four combinations:

======================================  =======  ====  =====
call                                    returns  axes  empty
======================================  =======  ====  =====
default (residuals on, no draws)        Axes     2     **1**
residuals on + draws                    Figure   2     none
show_residuals=False                    Axes     1     none
show_residuals=False + draws            Axes     1     none
======================================  =======  ====  =====

Two consequences, both on the path a first-time user takes:

* ``fig = plot_sed_fit(w, f, n); fig.savefig(...)`` raised
  ``AttributeError: 'Axes' object has no attribute 'savefig'`` — and that call
  is written three times in the library's own documentation (the function's
  Returns section says ``fig : matplotlib Figure``, its Examples block binds
  ``fig``, and ``tengri/plot/__init__.py`` does too). The sibling
  ``plot_spectrum_fit`` returns a real Figure.
* the **default** call built a residual panel it never drew into, so plotting
  photometry before you have a posterior produced an empty box under the SED.

The residual panel is now created only when it will be populated, and the
function has one return type.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tengri

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_N = 6


@pytest.fixture
def photometry():
    rng = np.random.default_rng(0)
    wave = np.linspace(3000.0, 20000.0, _N)
    flux = np.abs(rng.normal(1e-28, 1e-29, _N))
    return wave, flux, 0.05 * flux


@pytest.fixture
def draws():
    rng = np.random.default_rng(1)
    return np.abs(rng.normal(1e-28, 1e-29, (20, _N)))


def _calls(draws):
    return {
        "default": dict(),
        "with draws": dict(posterior_draws=draws),
        "no residuals": dict(show_residuals=False),
        "no residuals + draws": dict(show_residuals=False, posterior_draws=draws),
    }


@pytest.mark.parametrize(
    "label", ["default", "with draws", "no residuals", "no residuals + draws"]
)
def test_it_always_returns_a_figure(label, photometry, draws):
    """One return type, whatever optional data was supplied."""
    wave, flux, noise = photometry
    out = tengri.plot.plot_sed_fit(wave, flux, noise, **_calls(draws)[label])
    assert isinstance(out, plt.Figure), (
        f"{label}: returned {type(out).__name__}; the caller cannot know what "
        f"they hold if an optional data argument changes the type."
    )
    plt.close(out)


def test_the_documented_savefig_call_works(photometry, tmp_path):
    """The Returns section, the Examples block and tengri.plot's own docstring
    all bind ``fig`` and this is what they promise."""
    wave, flux, noise = photometry
    fig = tengri.plot.plot_sed_fit(wave, flux, noise)
    fig.savefig(tmp_path / "sed.png")
    assert (tmp_path / "sed.png").exists()
    plt.close(fig)


def test_the_default_call_leaves_no_empty_panel(photometry):
    """A residual panel with nothing in it is worse than no panel."""
    wave, flux, noise = photometry
    fig = tengri.plot.plot_sed_fit(wave, flux, noise)
    empty = [
        i
        for i, a in enumerate(fig.axes)
        if not (list(a.lines) or list(a.collections) or list(a.patches))
    ]
    assert not empty, (
        f"axes {empty} are empty on the default call: the residual panel is "
        f"built but only populated when posterior_draws is given."
    )
    plt.close(fig)


def test_the_residual_panel_still_appears_when_it_has_data(photometry, draws):
    """The converse — the fix must not delete the feature."""
    wave, flux, noise = photometry
    fig = tengri.plot.plot_sed_fit(wave, flux, noise, posterior_draws=draws)
    assert len(fig.axes) == 2, (
        f"expected a residual panel with draws supplied, got {len(fig.axes)} axes"
    )
    plt.close(fig)


def test_a_caller_supplied_axes_is_still_used(photometry, draws):
    """``ax=`` is documented as 'Axes to plot on'."""
    wave, flux, noise = photometry
    _, ax = plt.subplots()
    fig = tengri.plot.plot_sed_fit(wave, flux, noise, ax=ax, posterior_draws=draws)
    assert fig is ax.figure, "the caller's Axes was ignored"
    assert list(ax.lines) or list(ax.collections), "nothing was drawn on the given Axes"
    plt.close(fig)


def test_the_sibling_agrees(photometry):
    """plot_spectrum_fit already returned a Figure; the two should match."""
    wave, flux, noise = photometry
    out = tengri.plot.plot_spectrum_fit(wave, flux, noise)
    assert isinstance(out, plt.Figure)
    plt.close(out)
