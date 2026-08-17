# SpatialComponent Protocol + SpatialModelComponent + Concrete Profiles Plan

> Item #4 of the post-tracer-bullet architecture sequence. Independent of items #1 (PR #169) and #2 (PR #171) — does not touch SED or observation surfaces.

**Goal:** Land the mirror twin of the SED-side architecture for the spatial side. `SpatialComponent` Protocol (substrate) + `SpatialModelComponent` astronomer-facing base + three concrete profiles (`Sersic`, `Exponential`, `FlatSlab`). Bulge-disk composer and GP-spatial field are deferred.

**Architecture:** All new files. No existing class is renamed or restructured. The mirror is line-for-line: `SpatialComponent` mirrors `SEDComponent`, `SpatialModelComponent` mirrors `SEDModelComponent`, concrete profiles mirror physics adapters under `components/<domain>/`.

**Spatial state lives in `state.derived`.** Per architecture spec §3.3, today's A-path key is `state.derived["spatial_profile_2d"]`. Reserved keys for B-path: `spatial_profile_per_age`, `spatial_profile_per_wave`. The spatial grid is conveyed via `state.derived["spatial_grid_xy_kpc"]` (an `(x_grid, y_grid)` pair of 2D arrays in kpc).

**Out of scope:**
- `SpatialModel` and `SpatialSEDModel` sub-models (item #5)
- `BulgeDisk` composer (later)
- `GPSpatialField` (correlated-field IFT prior; later)
- Per-age / per-wavelength B-path (reserved but not exercised)
- ObservationModel adapters for resolved imaging / fiber aperture (item #6)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/tengri/protocols/spatial.py` | Create | `SpatialComponent` Protocol — mirror of `SEDComponent`, runtime-checkable. |
| `src/tengri/protocols/__init__.py` | Modify | Re-export `SpatialComponent`. |
| `src/tengri/components/spatial_model_component.py` | Create | `SpatialModelComponent` — astronomer-facing base; mirror of `SEDModelComponent`. |
| `src/tengri/components/spatial/__init__.py` | Create | Public exports (Sersic, Exponential, FlatSlab). |
| `src/tengri/components/spatial/sersic.py` | Create | `Sersic` concrete profile. |
| `src/tengri/components/spatial/exponential.py` | Create | `Exponential` concrete profile. |
| `src/tengri/components/spatial/flat_slab.py` | Create | `FlatSlab` concrete profile. |
| `tests/unit/protocols/test_spatial_protocol.py` | Create | Protocol shape checks. |
| `tests/unit/components/spatial/test_spatial_model_component.py` | Create | Base class behavior (param discovery, default apply). |
| `tests/unit/components/spatial/test_sersic.py` | Create | Sersic numeric smoke test. |
| `tests/unit/components/spatial/test_exponential.py` | Create | Exponential numeric smoke test. |
| `tests/unit/components/spatial/test_flat_slab.py` | Create | FlatSlab numeric smoke test. |

All new files. No deletions. No modifications to existing source files except `protocols/__init__.py` (one export line).

---

## Task 1: SpatialComponent Protocol

Mirror of `tengri.protocols.component.SEDComponent`. Runtime-checkable. Same methods (`declared_parameters`, `precompute`, `apply`), same role of `parameter_prefix` ("spatial_"), same role of `inputs`/`outputs`/`optional_inputs`.

**The Protocol signature** lives at `src/tengri/protocols/spatial.py`:

```python
"""SpatialComponent Protocol: contract for spatial-physics blocks.

Mirror of :class:`tengri.protocols.component.SEDComponent` on the spatial
side of the forward model. Each spatial component owns one piece of the
2D surface-brightness profile — a Sérsic envelope, an exponential disk,
a flat aperture, etc. — plus the parameters and precomputed tensors that
go with it.

See architecture spec ``docs/dev/archive/forward-model-architecture.md`` §3.2
for the astronomer-facing convenience base ``SpatialModelComponent``
that satisfies this Protocol with auto-discovery and a default apply().
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp

from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "SpatialComponent",
    "SpatialComponentConfig",
    "SpatialComponentState",
]


# Spatial components reuse the frozen-dataclass machinery from SEDComponent.
# These aliases let astronomer-facing imports read symmetrically with the
# SED side (``from tengri.protocols.spatial import SpatialComponentConfig``)
# without duplicating the type definitions.
SpatialComponentConfig = SEDComponentConfig
SpatialComponentState = SEDComponentState


@runtime_checkable
class SpatialComponent(Protocol):
    """Contract for one block of the spatial forward model.

    Concrete subclasses (``Sersic``, ``Exponential``, ``FlatSlab``,
    eventually ``BulgeDisk`` and ``GPSpatialField``) live in
    :mod:`tengri.components.spatial.<name>`. They publish a 2D
    surface-brightness profile into :attr:`ForwardState.derived` under
    the ``"spatial_profile_2d"`` key.

    Required attributes mirror :class:`SEDComponent` exactly:

    - ``name: str`` — stable identifier
    - ``parameter_prefix: str`` — always ``"spatial_"``
    - ``config: SpatialComponentConfig`` — frozen structural knobs

    Required methods are identical in shape to :class:`SEDComponent`:

    - ``declared_parameters()`` → list of :class:`ParamDeclaration`
    - ``precompute(...)`` → :class:`SpatialComponentState` (eager)
    - ``apply(state, params)`` → :class:`ForwardState` (pure JAX)
    """

    name: str
    parameter_prefix: str
    config: SpatialComponentConfig

    def declared_parameters(self) -> list[ParamDeclaration]: ...

    def precompute(self, **kwargs: Any) -> SpatialComponentState: ...

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState: ...
```

### Tests for Task 1

`tests/unit/protocols/test_spatial_protocol.py`:

```python
"""Smoke tests for the SpatialComponent Protocol."""

from __future__ import annotations

import jax.numpy as jnp

from tengri.protocols.spatial import SpatialComponent
from tengri.protocols.component import (
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
)


class _MinimalSpatialComponent:
    name = "minimal"
    parameter_prefix = "spatial_"
    config = SEDComponentConfig()

    def declared_parameters(self) -> list[ParamDeclaration]:
        return []

    def precompute(self, **kwargs):
        from tengri.protocols.spatial import SpatialComponentState
        return SpatialComponentState()

    def apply(self, state, params):
        return state


def test_spatial_component_is_runtime_checkable() -> None:
    assert isinstance(_MinimalSpatialComponent(), SpatialComponent)


def test_spatial_component_rejects_missing_apply() -> None:
    class Broken:
        name = "broken"
        parameter_prefix = "spatial_"
        config = SEDComponentConfig()

        def declared_parameters(self): return []
        def precompute(self, **kwargs): pass
    assert not isinstance(Broken(), SpatialComponent)


def test_spatial_component_config_and_state_alias_sed_versions() -> None:
    """Spatial components reuse the SED frozen-dataclass machinery."""
    from tengri.protocols.spatial import SpatialComponentConfig, SpatialComponentState
    from tengri.protocols.component import SEDComponentConfig, SEDComponentState
    assert SpatialComponentConfig is SEDComponentConfig
    assert SpatialComponentState is SEDComponentState
```

### Steps

- [ ] **Step 1: write the failing test** — create the test file above.
- [ ] **Step 2: run; expect import error** — `pytest tests/unit/protocols/test_spatial_protocol.py -v`
- [ ] **Step 3: implement** the protocol file.
- [ ] **Step 4: export** in `src/tengri/protocols/__init__.py` — add `from tengri.protocols.spatial import SpatialComponent`; add `"SpatialComponent"` to `__all__` alphabetically.
- [ ] **Step 5: run; expect 3 passed.**
- [ ] **Step 6: ruff check + format.**
- [ ] **Step 7: commit** `feat(protocols): SpatialComponent Protocol (mirror of SEDComponent)`

---

## Task 2: SpatialModelComponent base

Mirror of `tengri.components.sed_model_component.SEDModelComponent`. Same `__init_subclass__` auto-discovery of class-level `Distribution` attrs. Same `inputs`/`outputs` dict pattern. Default `apply()` handles param slicing, input lookup, predict call, state update — but updates `state.derived["spatial_profile_2d"]` instead of `state.sed_intrinsic`.

**The astronomer-facing contract** (the only thing subclass authors touch) is:

```python
class MyProfile(SpatialModelComponent):
    name = "my_profile"
    parameter_prefix = "spatial_"

    # Class-level Distribution attrs — auto-discovered
    re_kpc = Uniform(0.1, 20.0, description="Effective radius", units="kpc")

    # Cross-component contract (optional)
    reads = {}                              # most spatial blocks read nothing
    publishes = {"spatial_profile_2d": ""}  # the 2D profile (unitless)

    def predict(self, p, profile_in, grid_kpc):
        '''Pure JAX. Returns (profile_out, published).'''
        ...
```

### File: `src/tengri/components/spatial_model_component.py`

```python
"""SpatialModelComponent: astronomer-facing base for spatial physics blocks.

Mirror of :class:`tengri.components.sed_model_component.SEDModelComponent`
on the spatial side. Provides the same auto-discovery of free parameters
(class-level :class:`Distribution` attrs), the same ``inputs``/``outputs``
contract, and a sensible default :meth:`apply` orchestration that:

1. Slices ``params`` by ``parameter_prefix`` (always ``"spatial_"``).
2. Pulls cross-component reads from ``state.derived``.
3. Reads the spatial grid from ``state.derived["spatial_grid_xy_kpc"]``
   (a tuple of ``(x_grid_kpc, y_grid_kpc)`` 2D arrays).
4. Calls subclass :meth:`predict`.
5. Writes the resulting profile to ``state.derived["spatial_profile_2d"]``
   and any publishes the subclass declared.

Subclasses MUST override :meth:`predict` with signature
``predict(p, profile_in, grid_kpc) → (profile_out, published)``.

See architecture spec ``docs/dev/archive/forward-model-architecture.md`` §3.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from tengri.parameters.priors import Distribution
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["SpatialModelComponent"]

# Module-level registry for type-based lookup. Matches the SED-side
# pattern in tengri.components.sed_model_component._REGISTRY.
_SPATIAL_REGISTRY: dict[str, type[SpatialModelComponent]] = {}


class SpatialModelComponent:
    """Astronomer-facing base for spatial physics blocks.

    Subclasses declare:
      * ``name`` (str) and ``parameter_prefix`` (always ``"spatial_"``)
      * Class-level :class:`Distribution` attributes → free parameters
      * Optional ``reads`` (dict of name → units) and ``publishes``
        (dict of name → units) for the cross-component contract
      * A ``predict(p, profile_in, grid_kpc, **reads_kwargs)`` method

    Mirror of :class:`SEDModelComponent`. Behavior is identical except:
      * Default :attr:`parameter_prefix` is ``"spatial_"`` (not ``"sed_"``)
      * The default :meth:`apply` updates
        ``state.derived["spatial_profile_2d"]`` (not ``state.sed_intrinsic``)
      * :meth:`predict` is called with ``grid_kpc`` (not ``wave``)
    """

    # Defaults — subclasses override
    name: str = "spatial_component"
    parameter_prefix: str = "spatial_"
    reads: dict[str, str] = {}
    publishes: dict[str, str] = {"spatial_profile_2d": ""}
    config: SEDComponentConfig

    # Auto-discovered at class-creation time
    _free_param_attrs: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Discover class-level Distribution attrs → free parameters.
        # MRO walk so subclass overrides parent attrs cleanly.
        free_attrs: list[str] = []
        for attr_name in vars(cls):
            if attr_name.startswith("_"):
                continue
            attr = getattr(cls, attr_name, None)
            if isinstance(attr, Distribution):
                free_attrs.append(attr_name)
        cls._free_param_attrs = tuple(free_attrs)

        # Register the subclass for type-based lookup.
        component_name = getattr(cls, "name", None)
        if component_name and component_name != "spatial_component":
            if component_name in _SPATIAL_REGISTRY and _SPATIAL_REGISTRY[component_name] is not cls:
                existing = _SPATIAL_REGISTRY[component_name]
                raise ValueError(
                    f"SpatialModelComponent name collision: {component_name!r} is "
                    f"already registered to {existing.__module__}.{existing.__name__}; "
                    f"{cls.__module__}.{cls.__name__} cannot also claim it."
                )
            _SPATIAL_REGISTRY[component_name] = cls

    def __init__(self) -> None:
        self.config = SEDComponentConfig(name=self.name)

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Lift class-level Distribution attrs into ParamDeclaration tuples."""
        decls: list[ParamDeclaration] = []
        for attr_name in self._free_param_attrs:
            dist = getattr(type(self), attr_name)
            decls.append(
                ParamDeclaration(
                    name=f"{self.parameter_prefix}{attr_name}",
                    prior=dist,
                    description=getattr(dist, "description", ""),
                    units=getattr(dist, "units", ""),
                )
            )
        return decls

    def inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component reads, derived from the ``reads`` dict."""
        return tuple(
            DerivedKey(name=k, units=u, description="") for k, u in self.reads.items()
        )

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component publishes, derived from the ``publishes`` dict."""
        return tuple(
            DerivedKey(name=k, units=u, description="") for k, u in self.publishes.items()
        )

    def precompute(self, **kwargs: Any):
        """No-op by default. Subclasses override if they need cached tensors."""
        from tengri.protocols.spatial import SpatialComponentState
        return SpatialComponentState(name=self.name)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Default orchestration: slice params, lookup grid, call predict, write state."""
        prefix_len = len(self.parameter_prefix)
        p_sliced = {
            k[prefix_len:]: v for k, v in params.items() if k.startswith(self.parameter_prefix)
        }

        # Read cross-component inputs
        input_kwargs = {}
        for input_key in self.inputs():
            key_name = input_key.name
            if key_name not in state.derived:
                raise KeyError(
                    f"SpatialModelComponent {self.name!r} declares required input "
                    f"{key_name!r}, but it is not in state.derived. "
                    f"Available: {list(state.derived.keys())}"
                )
            input_kwargs[key_name] = state.derived[key_name]

        # Read the spatial grid from state.derived (set up by SpatialModel)
        grid_kpc = state.derived.get("spatial_grid_xy_kpc")
        if grid_kpc is None:
            raise KeyError(
                f"SpatialModelComponent {self.name!r}: state.derived does not contain "
                f"'spatial_grid_xy_kpc'. The grid must be set up by SpatialModel or "
                f"the caller before running spatial components. "
                f"Available: {list(state.derived.keys())}"
            )

        # Read the running profile (zeros if no upstream component published one)
        profile_in = state.derived.get("spatial_profile_2d")
        if profile_in is None:
            x_grid, _ = grid_kpc
            profile_in = jnp.zeros_like(x_grid)

        # Call predict
        profile_out, published = self.predict(p_sliced, profile_in, grid_kpc, **input_kwargs)

        # Always publish the running profile; merge subclass publishes
        new_derived = state.derived.with_(
            spatial_profile_2d=profile_out,
            **published,
        )
        return state.with_(derived=new_derived)

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        profile_in: jnp.ndarray,
        grid_kpc: Any,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX prediction step. MUST be implemented by subclasses.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped (``p["re_kpc"]``, not
            ``p["spatial_re_kpc"]``).
        profile_in : ndarray, shape (ny, nx)
            Running 2D surface-brightness profile from upstream components,
            or zeros if this component is first.
        grid_kpc : tuple of (ndarray, ndarray), each shape (ny, nx)
            ``(x_grid_kpc, y_grid_kpc)`` — the 2D spatial coordinate grids.
        **inputs : ndarray
            Cross-component reads keyed by names in :attr:`reads`.

        Returns
        -------
        tuple[ndarray, mapping[str, ndarray]]
            ``(profile_out, published)``. ``profile_out`` has shape
            ``(ny, nx)`` and replaces (or extends) ``profile_in``.
            ``published`` is a dict matching :attr:`publishes` keys
            other than ``spatial_profile_2d``, which the base class
            handles automatically.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.predict() must be implemented by subclass"
        )
```

### Tests for Task 2

`tests/unit/components/spatial/test_spatial_model_component.py`:

```python
"""Tests for SpatialModelComponent base class behavior."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import ForwardState
from tengri.protocols.derived_bundle import DerivedBundle


class _DummyProfile(SpatialModelComponent):
    name = "dummy_profile"
    parameter_prefix = "spatial_"

    radius = Uniform(0.1, 10.0, description="radius", units="kpc")

    reads = {}
    publishes = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        r = jnp.sqrt(x**2 + y**2)
        profile = jnp.exp(-r / p["radius"])
        return profile, {}


@pytest.fixture
def dummy_profile():
    return _DummyProfile()


@pytest.fixture
def state_with_grid():
    x = jnp.linspace(-5, 5, 10)
    y = jnp.linspace(-5, 5, 10)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    return state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))


def test_auto_discovers_distribution_attrs_as_free_params(dummy_profile) -> None:
    declared = dummy_profile.declared_parameters()
    assert {d.name for d in declared} == {"spatial_radius"}
    assert declared[0].units == "kpc"


def test_apply_writes_profile_to_state_derived(dummy_profile, state_with_grid) -> None:
    params = {"spatial_radius": jnp.float64(2.0)}
    out = dummy_profile.apply(state_with_grid, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (10, 10)
    # Profile should peak at the center
    assert profile[5, 5] > profile[0, 0]


def test_apply_strips_prefix_before_calling_predict(dummy_profile, state_with_grid) -> None:
    # If prefix isn't stripped, predict would error on KeyError("radius").
    params = {"spatial_radius": jnp.float64(2.0)}
    dummy_profile.apply(state_with_grid, params)  # would raise otherwise


def test_apply_raises_when_grid_missing(dummy_profile) -> None:
    state = ForwardState(wave=jnp.zeros(1))
    params = {"spatial_radius": jnp.float64(2.0)}
    with pytest.raises(KeyError, match="spatial_grid_xy_kpc"):
        dummy_profile.apply(state, params)
```

### Steps

- [ ] Write the failing test, run, expect import error.
- [ ] Implement the base class.
- [ ] Run; expect 4 passed.
- [ ] Lint + format.
- [ ] Commit `feat(components): SpatialModelComponent astronomer-facing base`

---

## Task 3: Sersic concrete profile

The first concrete spatial physics. Sersic profile with effective radius `r_e`, Sersic index `n`, axis ratio `b/a`, position angle `pa_deg`.

### File: `src/tengri/components/spatial/sersic.py`

```python
"""Sersic spatial profile.

Surface-brightness profile of the form

    I(r) ∝ exp(-b_n * ((r / r_e)^{1/n} - 1))

where ``r_e`` is the effective (half-light) radius, ``n`` is the
Sérsic index (n=1 ↔ exponential disk, n=4 ↔ de Vaucouleurs bulge),
and ``b_n`` is the Sérsic normalization that makes ``r_e`` enclose
half the total flux (Sérsic 1968).

References
----------
.. [Sersic1968] Sérsic, J. L. 1968, *Atlas de Galaxias Australes*,
   Cordoba, Argentina: Observatorio Astronomico.
.. [Ciotti1991] Ciotti, L. & Bertin, G. 1999, A&A, 352, 447 —
   asymptotic expansion for b_n.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["Sersic"]


def _b_n(n: jnp.ndarray) -> jnp.ndarray:
    """Sérsic normalization b_n via the Ciotti & Bertin (1999) expansion.

    Valid for n > 0.36; the analytic expansion is accurate to 10^-3 over
    the n ∈ [0.5, 10] range encountered in galaxy fits. See Ciotti &
    Bertin 1999 eq. 18 [Ciotti1991]_.
    """
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


class Sersic(SpatialModelComponent):
    """Sérsic surface-brightness profile.

    Free parameters
    ---------------
    re_kpc : Uniform(0.1, 20.0)
        Effective (half-light) radius [kpc].
    n : Uniform(0.5, 8.0)
        Sérsic index.
    axis_ratio : Uniform(0.1, 1.0)
        Minor-to-major axis ratio (b/a). 1.0 ↔ circular.
    pa_deg : Uniform(-90.0, 90.0)
        Position angle of the major axis [deg], measured east of north.
    """

    name = "sersic"
    parameter_prefix = "spatial_"

    re_kpc = Uniform(0.1, 20.0, description="Effective radius", units="kpc")
    n = Uniform(0.5, 8.0, description="Sersic index", units="")
    axis_ratio = Uniform(0.1, 1.0, description="Axis ratio b/a", units="")
    pa_deg = Uniform(-90.0, 90.0, description="Position angle", units="deg")

    reads: dict[str, str] = {}
    publishes = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        pa_rad = p["pa_deg"] * jnp.pi / 180.0
        cos_pa, sin_pa = jnp.cos(pa_rad), jnp.sin(pa_rad)

        # Rotate the grid into the profile's principal axes.
        x_rot = cos_pa * x + sin_pa * y
        y_rot = -sin_pa * x + cos_pa * y

        # Elliptical radius: x along major axis, y along minor (scaled by b/a).
        r_ell = jnp.sqrt(x_rot**2 + (y_rot / p["axis_ratio"]) ** 2)

        b_n = _b_n(p["n"])
        intensity = jnp.exp(-b_n * ((r_ell / p["re_kpc"]) ** (1.0 / p["n"]) - 1.0))
        return intensity, {}
```

### Tests

`tests/unit/components/spatial/test_sersic.py`:

```python
"""Numeric smoke tests for the Sersic spatial profile."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial.sersic import Sersic
from tengri.protocols.component import ForwardState
from tengri.protocols.spatial import SpatialComponent


@pytest.fixture
def sersic():
    return Sersic()


@pytest.fixture
def grid_5kpc():
    x = jnp.linspace(-5, 5, 20)
    y = jnp.linspace(-5, 5, 20)
    return jnp.meshgrid(x, y)


@pytest.fixture
def state_with_grid(grid_5kpc):
    state = ForwardState(wave=jnp.zeros(1))
    return state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=tuple(grid_5kpc)))


def test_sersic_satisfies_spatial_component_protocol(sersic) -> None:
    assert isinstance(sersic, SpatialComponent)


def test_sersic_publishes_profile_at_origin_peak(sersic, state_with_grid) -> None:
    params = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_n": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    out = sersic.apply(state_with_grid, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (20, 20)
    # Central pixel should be the brightest
    center = profile[10, 10]
    edge = profile[0, 0]
    assert center > edge


def test_sersic_n_4_is_more_centrally_concentrated_than_n_1(sersic, state_with_grid) -> None:
    """n=4 (de Vaucouleurs) profile is more concentrated than n=1 (exponential)."""
    base = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    p_dv = {**base, "spatial_n": jnp.float64(4.0)}
    p_exp = {**base, "spatial_n": jnp.float64(1.0)}

    profile_dv = sersic.apply(state_with_grid, p_dv).derived["spatial_profile_2d"]
    profile_exp = sersic.apply(state_with_grid, p_exp).derived["spatial_profile_2d"]

    # Center-to-edge contrast is steeper for de Vaucouleurs
    ratio_dv = profile_dv[10, 10] / profile_dv[0, 0]
    ratio_exp = profile_exp[10, 10] / profile_exp[0, 0]
    assert ratio_dv > ratio_exp


def test_sersic_circular_axis_ratio_gives_axisymmetric_profile(sersic, state_with_grid) -> None:
    params = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_n": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    profile = sersic.apply(state_with_grid, params).derived["spatial_profile_2d"]
    # Reflectional symmetry: profile[i, j] == profile[-i-1, -j-1]
    assert jnp.allclose(profile, profile[::-1, ::-1], rtol=1e-10)
```

### Steps

- [ ] Test → fail → implement → 4 passed → lint → commit `feat(components): Sersic spatial profile`

---

## Task 4: Exponential and FlatSlab concrete profiles

Two more concrete profiles, same shape as Sersic. Smaller tests (one numeric smoke each + protocol check).

`Exponential` is mathematically the n=1 special case of Sersic but expressed plainly for users who want explicit disk physics. `FlatSlab` is a uniform top-hat aperture (Heaviside cut at `radius_kpc`) — the "flat-slab" baseline the architecture spec calls out as the existing-codes' implicit approximation.

### Files

`src/tengri/components/spatial/exponential.py`:
```python
"""Exponential disk surface-brightness profile.

    I(r) ∝ exp(-r / r_d)

where ``r_d`` is the disk scale length. Equivalent to a Sérsic profile
with n=1, expressed as a standalone block for users who want explicit
disk physics in their parameter naming.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["Exponential"]


class Exponential(SpatialModelComponent):
    """Exponential disk profile.

    Free parameters
    ---------------
    rd_kpc : Uniform(0.1, 20.0)
        Disk scale length [kpc].
    axis_ratio : Uniform(0.1, 1.0)
        Minor-to-major axis ratio.
    pa_deg : Uniform(-90.0, 90.0)
        Position angle of the major axis [deg].
    """

    name = "exponential"
    parameter_prefix = "spatial_"

    rd_kpc = Uniform(0.1, 20.0, description="Disk scale length", units="kpc")
    axis_ratio = Uniform(0.1, 1.0, description="Axis ratio b/a", units="")
    pa_deg = Uniform(-90.0, 90.0, description="Position angle", units="deg")

    reads: dict[str, str] = {}
    publishes = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        pa_rad = p["pa_deg"] * jnp.pi / 180.0
        cos_pa, sin_pa = jnp.cos(pa_rad), jnp.sin(pa_rad)
        x_rot = cos_pa * x + sin_pa * y
        y_rot = -sin_pa * x + cos_pa * y
        r_ell = jnp.sqrt(x_rot**2 + (y_rot / p["axis_ratio"]) ** 2)
        intensity = jnp.exp(-r_ell / p["rd_kpc"])
        return intensity, {}
```

`src/tengri/components/spatial/flat_slab.py`:
```python
"""Flat-slab (uniform aperture) spatial profile.

A uniform disk: intensity = 1 inside ``radius_kpc``, 0 outside.
This is the implicit "all-sky-is-the-galaxy" model that classical
SED-fitting codes use when they scale a spectrum by an aperture factor.
Architecturally explicit here so users can verify the assumption.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["FlatSlab"]


class FlatSlab(SpatialModelComponent):
    """Uniform-disk (flat-slab) profile.

    Free parameters
    ---------------
    radius_kpc : Uniform(0.1, 50.0)
        Disk radius [kpc]. Intensity = 1 inside, 0 outside.

    Notes
    -----
    The discontinuous edge is smoothed with a small ``softness`` factor
    so the function remains differentiable — important for
    gradient-based inference. The softened profile is

        I(r) = sigmoid((R - r) / softness)

    where ``softness`` defaults to 1% of ``R``. This is a numerical
    convenience, not a physical claim.
    """

    name = "flat_slab"
    parameter_prefix = "spatial_"

    radius_kpc = Uniform(0.1, 50.0, description="Disk radius", units="kpc")

    reads: dict[str, str] = {}
    publishes = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        r = jnp.sqrt(x**2 + y**2)
        R = p["radius_kpc"]
        softness = R * 0.01
        intensity = jax.nn.sigmoid((R - r) / softness)
        return intensity, {}
```

(Note: requires `import jax` for `jax.nn.sigmoid`. Add at top.)

### Tests

`tests/unit/components/spatial/test_exponential.py`:
```python
"""Numeric smoke tests for the Exponential disk profile."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial.exponential import Exponential
from tengri.protocols.component import ForwardState
from tengri.protocols.spatial import SpatialComponent


def test_exponential_satisfies_protocol() -> None:
    assert isinstance(Exponential(), SpatialComponent)


def test_exponential_publishes_profile_falling_with_radius() -> None:
    x = jnp.linspace(-5, 5, 20)
    y = jnp.linspace(-5, 5, 20)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    state = state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))

    params = {
        "spatial_rd_kpc": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    out = Exponential().apply(state, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (20, 20)
    assert profile[10, 10] > profile[0, 0]
```

`tests/unit/components/spatial/test_flat_slab.py`:
```python
"""Numeric smoke tests for the FlatSlab uniform-aperture profile."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial.flat_slab import FlatSlab
from tengri.protocols.component import ForwardState
from tengri.protocols.spatial import SpatialComponent


def test_flat_slab_satisfies_protocol() -> None:
    assert isinstance(FlatSlab(), SpatialComponent)


def test_flat_slab_inside_is_near_unity_outside_is_near_zero() -> None:
    x = jnp.linspace(-5, 5, 50)
    y = jnp.linspace(-5, 5, 50)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    state = state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))

    params = {"spatial_radius_kpc": jnp.float64(2.0)}
    profile = FlatSlab().apply(state, params).derived["spatial_profile_2d"]

    # At the center (r=0), should be ~1 (inside the disk)
    assert profile[25, 25] > 0.9
    # At a corner (r ~ 7 kpc), should be ~0 (well outside)
    assert profile[0, 0] < 0.01
```

### Steps

- [ ] Implement both files + tests, lint, commit each separately:
  - `feat(components): Exponential disk spatial profile`
  - `feat(components): FlatSlab uniform-aperture spatial profile`

---

## Task 5: Public-API exports + docs

- [ ] **Step 1: create `src/tengri/components/spatial/__init__.py`**

```python
"""Spatial physics blocks: surface-brightness profiles.

See :doc:`docs/dev/archive/forward-model-architecture.md` §3.2.
"""

from tengri.components.spatial.exponential import Exponential
from tengri.components.spatial.flat_slab import FlatSlab
from tengri.components.spatial.sersic import Sersic

__all__ = ["Exponential", "FlatSlab", "Sersic"]
```

- [ ] **Step 2: update `docs/dev/where-things-live.md`**

In the "Forward model assembly" table, add a row:
```markdown
| Edit / write a spatial profile (Sérsic, exponential, flat slab) | `components/spatial/<name>.py` — subclass `SpatialModelComponent` |
```

- [ ] **Step 3: update CHANGELOG**

Add an entry under `### Added` in `## [Unreleased]`:

```markdown
- **`tengri.protocols.SpatialComponent`** Protocol — mirror of `SEDComponent` on the spatial side. Runtime-checkable.
- **`SpatialModelComponent`** astronomer-facing base — mirror of `SEDModelComponent`. Auto-discovers class-level `Distribution` attrs, default `apply()` handles param slicing and `state.derived["spatial_profile_2d"]` updates.
- **`tengri.components.spatial.{Sersic, Exponential, FlatSlab}`** concrete spatial profiles. Sérsic (n free), pure exponential disk, and the uniform-disk "flat slab" baseline (explicit form of the implicit aperture model classical codes use).
```

- [ ] **Step 4: commit** `docs: spatial components — where-things-live + CHANGELOG entries`

---

## Task 6: Regression check + push + PR

- [ ] Run full new test suite:
  ```bash
  PYTHONPATH=/tengri/src \
      /tengri/.venv/bin/pytest \
      tests/unit/protocols/test_spatial_protocol.py \
      tests/unit/components/spatial/ -q
  ```

- [ ] Project-wide lint:
  ```bash
  /tengri/.venv/bin/ruff check src/ tests/
  ```

- [ ] Public import smoke:
  ```bash
  PYTHONPATH=/tengri/src \
      /tengri/.venv/bin/python -c "
  from tengri.protocols.spatial import SpatialComponent
  from tengri.components.spatial import Sersic, Exponential, FlatSlab
  print('OK')
  "
  ```

- [ ] Push:
  ```bash
  git push -u origin HEAD
  ```

- [ ] Open PR with base `feat/forward-model-observation-predict` (stacked).

---

## Self-review checklist

- [ ] All new files have module docstrings.
- [ ] `SpatialComponent` is exported from `tengri.protocols`.
- [ ] `SpatialModelComponent` is documented as the astronomer-facing base.
- [ ] All 3 concrete profiles satisfy `SpatialComponent` (isinstance test passes).
- [ ] `Sersic` references Sérsic 1968 + Ciotti & Bertin 1999.
- [ ] No file is renamed or deleted.
- [ ] PR is stacked on PR #171 (or main if #171 merges first).
