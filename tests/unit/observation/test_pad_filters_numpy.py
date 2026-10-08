"""pad_filters assembles in numpy and must reproduce the per-filter jnp scatter loop exactly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.photometry import pad_filters


def _pad_filters_scatter_reference(filter_waves, filter_trans):
    """The pre-change jnp implementation, kept verbatim as the bitwise reference."""
    max_len = max(len(fw) for fw in filter_waves)
    fw_padded = jnp.zeros((len(filter_waves), max_len))
    ft_padded = jnp.zeros((len(filter_trans), max_len))
    n_valid = jnp.array([len(fw) for fw in filter_waves])
    for i, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        n = len(fw)
        fw_padded = fw_padded.at[i, :n].set(fw)
        ft_padded = ft_padded.at[i, :n].set(ft)
    return fw_padded, ft_padded, n_valid


def _heterogeneous_filters():
    rng = np.random.default_rng(3)
    lengths = [5, 17, 1, 40, 17, 9]
    waves = [np.sort(rng.uniform(1e3, 3e4, n)) for n in lengths]
    trans = [rng.uniform(0.0, 1.0, n) for n in lengths]
    return waves, trans


def test_pad_filters_matches_scatter_loop_bitwise():
    waves, trans = _heterogeneous_filters()
    got = pad_filters(waves, trans)
    ref = _pad_filters_scatter_reference(waves, trans)
    for g, r in zip(got, ref):
        assert g.dtype == r.dtype
        assert g.shape == r.shape
        np.testing.assert_array_equal(np.asarray(g), np.asarray(r))


def test_pad_filters_zero_pads_beyond_each_filter():
    waves, trans = _heterogeneous_filters()
    fw, ft, n_valid = pad_filters(waves, trans)
    fw, ft, n_valid = np.asarray(fw), np.asarray(ft), np.asarray(n_valid)
    for i, (w, t) in enumerate(zip(waves, trans)):
        n = len(w)
        assert n_valid[i] == n
        np.testing.assert_array_equal(fw[i, :n], w)
        np.testing.assert_array_equal(ft[i, :n], t)
        assert np.all(fw[i, n:] == 0.0)
        assert np.all(ft[i, n:] == 0.0)


@pytest.mark.parametrize("x64", [True, False])
def test_pad_filters_dtypes_match_jnp_path(x64):
    waves, trans = _heterogeneous_filters()
    with jax.enable_x64(x64):
        got = pad_filters(waves, trans)
        ref = _pad_filters_scatter_reference(waves, trans)
        assert [g.dtype for g in got] == [r.dtype for r in ref]
