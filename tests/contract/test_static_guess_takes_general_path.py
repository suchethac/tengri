# SPDX-License-Identifier: BSD-3-Clause
"""Every site that guesses about a traced or runtime input must say which path it takes (#2847).

A function that inspects an input which may be a JAX tracer, or a redshift that may be
runtime under catalog mode, can only answer from the static structure it can see. When
it cannot know the answer it must take the general path, or raise at build. It must
never default to the convenient fast path.

This contract scans ``src/tengri`` with the ``ast`` module for two kinds of guess:

* ``isinstance(x, jax.core.Tracer)`` (and tuples containing it), and calls to the known
  concreteness helpers;
* ``.is_fixed`` read off a redshift distribution.

Each function that does either must appear in :data:`ALLOWLIST`, mapped to a sentence
saying which path the unknown takes. A new unlisted site fails the test, so a new guess
is reviewed when it is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tengri"

#: Helpers whose result is a concreteness guess. Calling one is a guess site.
_CONCRETENESS_HELPERS = frozenset(
    {
        "_is_log_uniform",
        "_require_log_uniform_grid",
        "_fe2_is_concrete_narrow",
        "_is_concrete_nonpositive",
        "_is_traced",
        "redshift_is_runtime",
    }
)

#: ``module::qualname`` -> which path an unknown (traced or runtime) input takes.
#: One entry per site that branches on a tracer or a runtime redshift; a new entry needs
#: the same justification.
ALLOWLIST: dict[str, str] = {
    # ---- answered: an unknown takes the general path, or raises at build ----
    "_cache_keys.py::baked": "traced value raises TypeError: a traced value cannot key a cache",
    "components/agn/blocks/runner.py::compose_l_nu": (
        "traced fracAGN goes through jax.lax.cond over both polar branches"
    ),
    "components/agn/blocks/torus_library_transmission.py::_below_library_index": (
        "tracer → None; the caller (library sightline) then evaluates the X-ray absorption on "
        "the whole grid with a where-select below the library's first node — the original dense, "
        "exact general path; the static index set is only an optimization for a concrete build "
        "grid."
    ),
    "components/agn/blr.py::_fe2_is_concrete_narrow": (
        "traced FWHM takes the band-limited path, valid for FWHM >= 500 km/s only; a traced "
        "narrower width returns NaN by design (#2768, pinned by its regression test), and the "
        "grammar refuses a free BLR width, so the model path never traces one"
    ),
    "components/agn/blr.py::_fe2_pseudo_continuum": (
        "concreteness via _fe2_is_concrete_narrow: traced width -> band-limited, NaN below 500 "
        "km/s (#2768, loud)"
    ),
    "components/agn/blr.py::_fe2_total_power": (
        "concreteness via _fe2_is_concrete_narrow: traced width -> band-limited, NaN below 500 "
        "km/s (#2768, loud)"
    ),
    "components/nebular/nebular_grid_precompute.py::nebular_z_request": (
        "None only when redshift is Fixed and not catalog; catalog or free -> z grid (runtime)"
    ),
    "components/nebular/nebular_grid_precompute.py::reference_redshift": (
        "catalog -> catalog floor; Fixed -> its value; free -> prior floor (no fixed value used "
        "for a runtime z)"
    ),
    "components/stellar/component.py::StellarSEDComponent._spectrum_lut_families": (
        "a traced draw dispersion only skips the equality check against the table's build-time "
        "sigma_v; the pre-broadened table exists only for a fixed sigma_v (a free one leaves "
        "lut.sigma_v_kms None and broadens the summed spectrum on the full model grid every "
        "draw), so no traced dispersion reaches the table route (#2958)"
    ),
    "forward/sed_model.py::SEDModel.__init__": (
        "catalog or free redshift -> z-table enabled in both branches (#2858)"
    ),
    "forward/sed_model.py::SEDModel._build_component_chain": (
        "via redshift_is_runtime: catalog or free -> runtime z-table spec, never the placeholder"
    ),
    "forward/sed_model.py::SEDModel._get_or_build_predict_spectrum_on_grid_jit": (
        "traced wave_obs -> runtime argument to the kernel, which takes the general LSF path"
    ),
    "forward/sed_model.py::SEDModel._init_cosmology": (
        "fixed distance only for Fixed and not catalog; otherwise runtime distance (#2847)"
    ),
    "forward/sed_model.py::SEDModel._init_instrument": (
        "runtime redshift -> no value to check the LSF against, so the sigma-lib warning is "
        "skipped (documented, #2847); a Fixed z is checked"
    ),
    "forward/sed_model.py::SEDModel._predict_spectrum_on_grid": (
        "traced wave_obs -> grid passed at runtime, general path"
    ),
    "forward/sed_model.py::SEDModel._response_z_nodes": (
        "z-table nodes built unless Fixed and not catalog"
    ),
    "forward/sed_model.py::SEDModel.filter_coverage_redshifts": (
        "fixed redshift → checks band coverage at that z; free (or catalog_z_range) → checks at "
        "every finite prior/catalog end, never a single guessed z. Coverage is monotone in z "
        "because the observed axis shifts as (1+z), so both ends bound the whole range."
    ),
    "forward/sed_model.py::SEDModel.predict_line_fluxes": (
        "traced min_deltas skips only the eager line-matching loud check; the structural match "
        "is static and the eager call on the same model runs the check"
    ),
    "forward/sed_model.py::redshift_is_runtime": "unknown distribution -> runtime (#2858)",
    "observation/spectroscopy.py::Spectroscopy.resolve_conserving": (
        "traced grid -> resample mode resolved on the configured grid (see gap below)"
    ),
    "observation/spectroscopy.py::_is_traced": "predicate: tracer -> True",
    "observation/spectrum.py::_is_concrete_nonpositive": (
        "tracer -> False: not known to be zero, so the general (non-skipping) path runs"
    ),
    "observation/spectrum.py::_is_log_uniform": (
        "tracer -> False: unknown, takes the general piecewise LSF path (#2847)"
    ),
    "observation/spectrum.py::_is_traced": "predicate: tracer -> True",
    "observation/spectrum.py::apply_lsf": (
        "log-uniform FFT only for a concrete log grid; traced or linear -> piecewise (#2847)"
    ),
    "observation/spectrum.py::broaden_velocity_only": (
        "traced sigma is not skipped (general path runs)"
    ),
    "observation/spectrum.py::broaden_velocity_only_window": (
        "traced sigma is not skipped (general path runs)"
    ),
    "observation/spectrum.py::rest_grid_observed_window": (
        "traced grid -> None: the caller uses the full grid (general)"
    ),
    "observation/spectrum.py::velocity_broaden": (
        "traced grid -> piecewise general path of broaden_velocity_only (#2847)"
    ),
    "utils/interpolation.py::concrete_canonical_numpy": (
        "tracer -> None: the caller takes its original jnp formula (same arithmetic, general path)"
    ),
    "utils/sed_quantities.py::_uv_anchor_plan": (
        "tracer → 'traced': in-grid interpolation with NaN wherever the grid does not span 1600 Å "
        "(loud), never the extrapolate/edge plans, which need a concrete grid."
    ),
    # ---- recorded gaps: the unknown is not the general path; owner to decide ----
    "observation/spectrum.py::_lsf_sigma_bound_kms": (
        "GAP (documented #2832): a traced resolution or sigma counts as 2000 km/s, so a traced "
        "sigma above 2000 km/s is under-padded; pass pad_pixels"
    ),
    "observation/spectrum.py::_lsf_pad_pixels": (
        "GAP (documented #2832): traced components count as 2000 km/s via the bound; a traced "
        "grid pads by n - 1 (general)"
    ),
    "components/agn/fritz.py::_refuse_off_grid": (
        "GAP: traced value skips the off-grid check; the builder's declared bounds refuse "
        "traced values at build (documented)"
    ),
    "parameters/groups.py::_warn_free_redshift_onset_ceiling": (
        "GAP (warning only): Fixed redshift skips the onset-ceiling warning, including a catalog "
        "placeholder whose real range is free"
    ),
}


def _rel(path: Path) -> str:
    return path.relative_to(_SRC).as_posix()


def _is_tracer_ref(node: ast.AST) -> bool:
    """True for ``jax.core.Tracer`` or a name/tuple that ends in ``Tracer``."""
    if isinstance(node, ast.Attribute):
        return node.attr == "Tracer"
    if isinstance(node, ast.Name):
        return node.id == "Tracer"
    if isinstance(node, ast.Tuple):
        return any(_is_tracer_ref(elt) for elt in node.elts)
    return False


def _function_sites(tree: ast.Module):
    """Yield ``(qualname, function node)`` for every function, including methods."""

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                yield qual, child
                yield from walk(child, qual + ".")
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, f"{prefix}{child.name}.")
            else:
                yield from walk(child, prefix)

    yield from walk(tree, "")


def _guess_kinds(func: ast.AST) -> set[str]:
    """Which guess kinds a function body contains, ignoring nested functions' bodies."""
    kinds: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            is_isinstance = isinstance(node.func, ast.Name) and node.func.id == "isinstance"
            if is_isinstance and len(node.args) >= 2 and _is_tracer_ref(node.args[1]):
                kinds.add("tracer")
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in _CONCRETENESS_HELPERS:
                kinds.add("concreteness")
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "is_fixed"
            and _mentions_redshift(func, node)
        ):
            kinds.add("redshift_is_fixed")
    return kinds


def _mentions_redshift(func: ast.AST, attr: ast.Attribute) -> bool:
    """Whether the ``.is_fixed`` read is on a redshift distribution."""
    base = attr.value
    if isinstance(base, ast.Name):
        if "redshift" in base.id:
            return True
        # ``dist = spec.get_distribution("redshift")`` then ``dist.is_fixed``
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == base.id for t in node.targets)
                and isinstance(node.value, ast.Call)
                and any(
                    isinstance(a, ast.Constant) and a.value == "redshift" for a in node.value.args
                )
            ):
                return True
    return False


def collect_guess_sites() -> dict[str, set[str]]:
    """Map ``module::qualname`` to the guess kinds found in that function."""
    sites: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for qual, func in _function_sites(tree):
            kinds = _guess_kinds(func)
            if kinds:
                sites[f"{_rel(path)}::{qual}"] = kinds
    return sites


def test_every_guess_site_is_allowlisted_with_its_general_path():
    sites = collect_guess_sites()
    unlisted = sorted(set(sites) - set(ALLOWLIST))
    assert not unlisted, (
        "new traced/runtime guess site(s) without a recorded general-path answer; "
        "add each to ALLOWLIST with the path an unknown input takes:\n  "
        + "\n  ".join(f"{s} {sorted(sites[s])}" for s in unlisted)
    )


def test_allowlist_has_no_stale_entries():
    sites = collect_guess_sites()
    stale = sorted(set(ALLOWLIST) - set(sites))
    assert not stale, f"ALLOWLIST entries no longer match a guess site: {stale}"


def test_allowlist_entries_state_a_path():
    for key, answer in ALLOWLIST.items():
        assert answer.strip(), f"{key} has no recorded path"
