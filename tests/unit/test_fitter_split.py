"""Smoke tests for Scope B: fitter.py split into vi.py, mcmc.py, evidence.py, map_dispatch.py."""


def test_vi_module_importable():
    """After split, vi.py must be importable and expose run_nifty_fast_vi."""
    from tengri.inference.vi import run_nifty_fast_vi

    assert callable(run_nifty_fast_vi)


def test_vi_module_all_functions():
    from tengri.inference.vi import run_native_vi, run_nifty_fast_vi, run_nifty_vi

    assert callable(run_nifty_fast_vi)
    assert callable(run_native_vi)
    assert callable(run_nifty_vi)


def test_mcmc_module_importable():
    from tengri.inference.mcmc import run_elliptical_slice, run_nuts, run_raytrace

    assert callable(run_raytrace)
    assert callable(run_nuts)
    assert callable(run_elliptical_slice)


def test_evidence_module_importable():
    from tengri.inference.evidence import run_nss

    assert callable(run_nss)


def test_map_dispatch_importable():
    from tengri.inference.map_dispatch import run_laplace, run_map, run_pathfinder

    assert callable(run_map)
    assert callable(run_laplace)
    assert callable(run_pathfinder)


def test_fitter_still_importable():
    """Fitter class must still be importable after the split."""
    from tengri.inference.fitter import Fitter

    assert Fitter is not None
