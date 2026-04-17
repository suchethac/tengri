"""Smoke tests for Scope B: fitter.py split into vi.py, mcmc.py, evidence.py, map_dispatch.py."""


def test_vi_module_importable():
    """After split, vi backend must be importable and expose run_nifty_fast_vi."""
    from tengri.inference.backends.vi.nifty import run_nifty_fast_vi

    assert callable(run_nifty_fast_vi)


def test_vi_module_all_functions():
    from tengri.inference.backends.vi.native import run_native_vi
    from tengri.inference.backends.vi.nifty import run_nifty_fast_vi, run_nifty_vi

    assert callable(run_nifty_fast_vi)
    assert callable(run_native_vi)
    assert callable(run_nifty_vi)


def test_mcmc_module_importable():
    from tengri.inference.backends.mcmc.common import run_elliptical_slice, run_nuts, run_raytrace

    assert callable(run_raytrace)
    assert callable(run_nuts)
    assert callable(run_elliptical_slice)


def test_evidence_module_importable():
    from tengri.inference.backends.evidence import run_nss

    assert callable(run_nss)


def test_map_dispatch_importable():
    from tengri.inference.backends.laplace import run_laplace
    from tengri.inference.backends.map_dispatch import run_map
    from tengri.inference.backends.pathfinder import run_pathfinder

    assert callable(run_map)
    assert callable(run_laplace)
    assert callable(run_pathfinder)


def test_fitter_still_importable():
    """Fitter class must still be importable after the split."""
    from tengri.inference.fitter import Fitter

    assert Fitter is not None
