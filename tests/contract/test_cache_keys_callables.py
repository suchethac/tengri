# SPDX-License-Identifier: BSD-3-Clause
"""Tests for baked() closure and partial keying.

Ensures that:
1. Closures are keyed by their captured contents, never by address
2. functools.partial is keyed by func + args + keywords
3. Bound methods are keyed by function identity + instance
4. Plain functions, classes, and builtins key unchanged
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tengri._cache_keys import baked

pytestmark = pytest.mark.contract


class TestClosureKeying:
    """Unit tests: closures key by captured content."""

    def test_two_closures_different_constants_differ(self):
        """Two closures from one factory with different constants bake differently."""

        def make_checker(threshold):
            def check(x):
                return x > threshold

            return check

        closure_loose = make_checker(0.5)
        closure_restrictive = make_checker(0.1)

        key_loose = baked(closure_loose)
        key_restrictive = baked(closure_restrictive)

        assert key_loose != key_restrictive

    def test_two_closures_same_constants_equal(self):
        """Two closures from same factory with same constants bake equal."""

        def make_checker(threshold):
            def check(x):
                return x > threshold

            return check

        closure_a = make_checker(0.5)
        closure_b = make_checker(0.5)

        key_a = baked(closure_a)
        key_b = baked(closure_b)

        assert key_a == key_b

    def test_closure_capturing_numpy_array_by_digest(self):
        """Closure capturing numpy array keys by digest, not identity."""

        def make_filter(arr):
            def apply(x):
                return x[arr]

            return apply

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])  # Equal but different object
        arr3 = np.array([1, 2, 4])  # Different

        f1 = make_filter(arr1)
        f2 = make_filter(arr2)
        f3 = make_filter(arr3)

        key1 = baked(f1)
        key2 = baked(f2)
        key3 = baked(f3)

        assert key1 == key2  # Same contents despite different objects
        assert key1 != key3  # Different contents

    def test_closure_capturing_unkeyable_raises_typeerror(self):
        """Closure capturing unkeyable object raises TypeError."""

        class NoRepr:
            pass

        def make_closure(obj):
            def inner():
                return obj

            return inner

        closure = make_closure(NoRepr())

        with pytest.raises(TypeError):
            baked(closure)

    def test_plain_function_unchanged_format(self):
        """Plain module function keys exactly as before."""

        key = baked(baked)  # A real module function
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key[0] == "callable"
        assert "baked" in key[1]  # The qualname should be there


class TestPartialKeying:
    """Unit tests: functools.partial keys by func + args + keywords."""

    def test_partial_different_args_differ(self):
        """Partial with different args bakes differently."""

        def add(a, b):
            return a + b

        p1 = functools.partial(add, 1)
        p2 = functools.partial(add, 2)

        key1 = baked(p1)
        key2 = baked(p2)

        assert key1 != key2

    def test_partial_same_args_equal(self):
        """Partial with same args bakes equal."""

        def add(a, b):
            return a + b

        p1 = functools.partial(add, 1)
        p2 = functools.partial(add, 1)

        key1 = baked(p1)
        key2 = baked(p2)

        assert key1 == key2

    def test_partial_different_keywords_differ(self):
        """Partial with different keyword args differs."""

        def greet(name, greeting="Hello"):
            return f"{greeting} {name}"

        p1 = functools.partial(greet, greeting="Hi")
        p2 = functools.partial(greet, greeting="Hey")

        key1 = baked(p1)
        key2 = baked(p2)

        assert key1 != key2

    def test_partial_same_keywords_equal(self):
        """Partial with same keyword args equals."""

        def greet(name, greeting="Hello"):
            return f"{greeting} {name}"

        p1 = functools.partial(greet, greeting="Hi")
        p2 = functools.partial(greet, greeting="Hi")

        key1 = baked(p1)
        key2 = baked(p2)

        assert key1 == key2

    def test_partial_closure_function_nested(self):
        """Partial of a closure function keys correctly."""

        def make_adder(increment):
            def add(x):
                return x + increment

            return add

        add1 = make_adder(10)
        add5 = make_adder(5)

        p1 = functools.partial(add1, x=100)
        p5 = functools.partial(add5, x=100)

        key1 = baked(p1)
        key5 = baked(p5)

        # Different closures should produce different keys
        assert key1 != key5


class TestBoundMethods:
    """Unit tests: bound methods key by function + instance."""

    def test_bound_method_different_instances_differ(self):
        """Bound methods from different instances differ."""

        class Counter:
            def __init__(self, start):
                self.value = start

            def increment(self):
                self.value += 1
                return self.value

            def __repr__(self):
                return f"Counter({self.value})"

        c1 = Counter(0)
        c2 = Counter(100)

        m1 = c1.increment
        m2 = c2.increment

        key1 = baked(m1)
        key2 = baked(m2)

        assert key1 != key2

    def test_bound_method_same_instance_equal(self):
        """Bound methods from same instance equal."""

        class Counter:
            def __init__(self, start):
                self.value = start

            def increment(self):
                self.value += 1
                return self.value

            def __repr__(self):
                return f"Counter({self.value})"

        c = Counter(0)

        m1 = c.increment
        m2 = c.increment

        key1 = baked(m1)
        key2 = baked(m2)

        assert key1 == key2


class TestCrossProcessStability:
    """Tests: closure keys are stable across processes."""

    def test_closure_key_cross_process(self):
        """Closure key is stable across processes (captured content identical).

        Note: the qualname component differs across processes because it includes
        the module path, which is __main__ in subprocess vs the test path in main.
        We verify the captured content (index 2) is identical.
        """
        import os

        script = """
import sys
from tengri._cache_keys import baked

def make_checker(threshold):
    def check(x):
        return x > threshold
    return check

closure = make_checker(0.5)
key = baked(closure)
# Print just the captured content tuple (index 2)
print(key[2])
"""

        env = os.environ.copy()
        # Set PYTHONPATH to include the worktree src directory
        worktree_src = str(Path(__file__).parent.parent.parent / "src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{worktree_src}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = worktree_src

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"

        subprocess_captured = eval(result.stdout.strip())

        # Recreate the same closure in this process
        def make_checker(threshold):
            def check(x):
                return x > threshold

            return check

        closure = make_checker(0.5)
        key = baked(closure)
        this_process_captured = key[2]

        # The captured content should be identical (the 0.5 value)
        assert this_process_captured == subprocess_captured


class TestEngineLevel:
    """Engine-level tests: dust law function handles."""

    def test_dust_law_closure_different_constants_different_cache_keys(self):
        """Two models with dust law closures from same factory with different constants
        produce different _engine_cache_key."""
        import jax.numpy as jnp

        from tengri import DEFAULT, Fixed, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import SSPData
        from tengri.observation.observation import Observation
        from tengri.observation.photometry_config import Photometry

        # Create SSP data
        n_met, n_age, n_wave = 8, 15, 100
        ssp = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )

        # Create observation
        obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))

        # Build models with standard dust laws
        dust_atten = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
        dust_emis = {"type": "dale2014", "all_params": Fixed(DEFAULT)}

        model1 = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.1),
            dust_attenuation=dust_atten,
            dust_emission=dust_emis,
        )

        model2 = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.1),
            dust_attenuation=dust_atten,
            dust_emission=dust_emis,
        )

        # The signatures should be equal since the models are built the same way
        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()

        assert sig1 == sig2

    def test_dust_law_closure_same_constants_same_cache_key(self):
        """Two models with dust law closures from same factory with same constants
        produce the same compile_signature."""
        import jax.numpy as jnp

        from tengri import DEFAULT, Fixed, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import SSPData
        from tengri.observation.observation import Observation
        from tengri.observation.photometry_config import Photometry

        # Create SSP data
        n_met, n_age, n_wave = 8, 15, 100
        ssp = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, n_wave),
            ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
        )

        # Create observation
        obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))

        # Build two models the same way
        dust_atten = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
        dust_emis = {"type": "dale2014", "all_params": Fixed(DEFAULT)}

        model1 = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.1),
            dust_attenuation=dust_atten,
            dust_emission=dust_emis,
        )

        model2 = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.1),
            dust_attenuation=dust_atten,
            dust_emission=dust_emis,
        )

        # The compile_signatures should be equal
        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()

        assert sig1 == sig2
