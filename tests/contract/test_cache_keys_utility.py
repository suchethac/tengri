# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._cache_keys import (
    MODES,
    KeyPolicy,
    PolicyRow,
    array_key,
    assert_policy_complete,
    baked,
    classify,
    content,
    derive_key,
    exclude,
    frozen_dataclass_key,
    shape,
    stable_digest,
)

pytestmark = pytest.mark.contract


class TestStableDigest:
    """Test stable_digest for determinism and correctness."""

    def test_deterministic_across_calls(self):
        """stable_digest produces the same result on repeated calls."""
        d1 = stable_digest(b"tengri")
        d2 = stable_digest(b"tengri")
        assert d1 == d2

    def test_known_value(self):
        """stable_digest returns expected value for known input."""
        # Compute once and verify: blake2b(b"tengri", digest_size=16).hexdigest()
        expected = "563dfa597c7c1bdb38f39047fec2f3de"
        result = stable_digest(b"tengri")
        assert result == expected

    def test_differs_for_different_input(self):
        """stable_digest produces different results for different inputs."""
        d1 = stable_digest(b"tengri")
        d2 = stable_digest(b"tengri2")
        assert d1 != d2


class TestBakedScalars:
    """Test baked() on scalar types."""

    def test_none_unchanged(self):
        """baked(None) is None."""
        assert baked(None) is None

    def test_bool_unchanged(self):
        """baked(True) is True and baked(False) is False."""
        assert baked(True) is True
        assert baked(False) is False

    def test_int_unchanged(self):
        """baked(42) is 42."""
        assert baked(42) == 42

    def test_float_full_precision(self):
        """baked(1e-30) differs from baked(1e-20) (full precision, not rounded)."""
        d1 = baked(1e-30)
        d2 = baked(1e-20)
        assert d1 != d2
        assert d1 == 1e-30
        assert d2 == 1e-20

    def test_string_unchanged(self):
        """baked("x") == "x"."""
        assert baked("x") == "x"

    def test_bytes_unchanged(self):
        """baked(b"x") == b"x"."""
        assert baked(b"x") == b"x"


class TestBakedContainers:
    """Test baked() on containers."""

    def test_list_bakes_to_tuple(self):
        """baked([1, 2, 3]) produces a tuple."""
        result = baked([1, 2, 3])
        assert result == (1, 2, 3)

    def test_list_vs_tuple_same_items_equal(self):
        """list vs tuple with the same items bake equal."""
        assert baked([1, 2, 3]) == baked((1, 2, 3))

    def test_dict_order_independent(self):
        """dict key order does not matter; both bake equal."""
        d1 = baked({"a": 1, "b": 2})
        d2 = baked({"b": 2, "a": 1})
        assert d1 == d2

    def test_mapping_proxy_bakes_like_dict(self):
        """MappingProxyType bakes like its dict."""
        d = {"a": 1, "b": 2}
        proxy = types.MappingProxyType(d)
        assert baked(proxy) == baked(d)

    def test_set_order_independent(self):
        """sets bake order-independently."""
        s1 = baked({1, 2, 3})
        s2 = baked({3, 2, 1})
        assert s1 == s2

    def test_frozenset_order_independent(self):
        """frozensets bake order-independently."""
        fs1 = baked(frozenset({1, 2, 3}))
        fs2 = baked(frozenset({3, 2, 1}))
        assert fs1 == fs2


class TestBakedCallables:
    """Test baked() on callables."""

    def test_two_functions_bake_differently(self):
        """two module-level functions bake differently."""

        def func1():
            pass

        def func2():
            pass

        assert baked(func1) != baked(func2)

    def test_same_function_twice_equal(self):
        """the same function twice bakes equal."""

        def myfunc():
            pass

        assert baked(myfunc) == baked(myfunc)

    def test_bound_method_includes_class_qualname(self):
        """a bound method includes the class qualname."""

        class MyClass:
            def method(self):
                pass

        obj = MyClass()
        result = baked(obj.method)
        # Should contain ("callable", "<something>MyClass.method>")
        assert isinstance(result, tuple)
        assert result[0] == "callable"
        assert "MyClass" in result[1]

    def test_lambda_bakes(self):
        """lambda functions are baked as callables."""

        def lam(x):
            return x

        result = baked(lam)
        assert isinstance(result, tuple)
        assert result[0] == "callable"


class TestBakedDelegation:
    """Test baked() delegation to cache_key/jit_cache_key."""

    def test_object_with_cache_key(self):
        """an object with cache_key() returns ("cache_key", <qualname>, ...)."""

        class HasCacheKey:
            def cache_key(self):
                return ("k", 1)

        obj = HasCacheKey()
        result = baked(obj)
        assert result[0] == "cache_key"
        assert result[1].endswith("HasCacheKey")
        assert result[2] == ("k", 1)

    def test_object_with_jit_cache_key(self):
        """an object with jit_cache_key() likewise."""

        class HasJitCacheKey:
            def jit_cache_key(self):
                return ("jit_k", 2)

        obj = HasJitCacheKey()
        result = baked(obj)
        assert result[0] == "jit_cache_key"
        assert result[1].endswith("HasJitCacheKey")
        assert result[2] == ("jit_k", 2)

    def test_object_default_repr_raises(self):
        """an object with only default object.__repr__ raises TypeError naming type."""

        class NoRepr:
            pass

        obj = NoRepr()
        with pytest.raises(TypeError) as excinfo:
            baked(obj)
        assert "NoRepr" in str(excinfo.value)
        assert "cache_key" in str(excinfo.value).lower()

    def test_object_custom_repr_bakes_via_repr(self):
        """an object with its own __repr__ bakes via repr."""

        class CustomRepr:
            def __repr__(self):
                return "<CustomRepr:42>"

        obj = CustomRepr()
        result = baked(obj)
        assert result[0] == "repr"
        assert result[1].endswith("CustomRepr")
        assert result[2] == "<CustomRepr:42>"


class TestBakedArrays:
    """Test baked() on arrays."""

    def test_numpy_arrays_equal_content_equal_key(self):
        """two distinct np arrays with equal content bake equal."""
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([1.0, 2.0, 3.0])
        assert baked(a1) == baked(a2)

    def test_numpy_arrays_different_content_differ(self):
        """different content gives different keys."""
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([1.0, 2.0, 4.0])
        assert baked(a1) != baked(a2)

    def test_numpy_arrays_different_dtype_differ(self):
        """different dtype (float32 vs float64) differs."""
        a1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert baked(a1) != baked(a2)

    def test_numpy_arrays_different_shape_differ(self):
        """different shape differs."""
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([[1.0, 2.0, 3.0]])
        assert baked(a1) != baked(a2)

    def test_jax_array_and_numpy_array_identical_content_equal(self):
        """a JAX array and numpy array with identical content/dtype/shape bake EQUAL."""
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        jax_arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        assert baked(np_arr) == baked(jax_arr)

    def test_jax_tracer_raises_type_error(self):
        """a JAX tracer raises TypeError."""

        def traced_baked():
            return baked(jnp.ones(3))

        with pytest.raises(TypeError) as excinfo:
            jax.jit(traced_baked)()
        assert "traced" in str(excinfo.value).lower()

    def test_array_key_memo_same_jax_array(self):
        """calling array_key twice on the same jax array returns identical keys."""
        jax_arr = jnp.array([1.0, 2.0, 3.0])
        k1 = array_key(jax_arr)
        k2 = array_key(jax_arr)
        assert k1 == k2

    def test_array_key_numpy_mutated_gives_different_keys(self):
        """Numpy array mutated in place yields DIFFERENT keys (no stale memo)."""
        np_arr = np.array([1.0, 2.0, 3.0])
        k1 = array_key(np_arr)
        np_arr[0] = 99.0  # mutate in place
        k2 = array_key(np_arr)
        assert k1 != k2


class TestFrozenDataclassKey:
    """Test frozen_dataclass_key()."""

    def test_equal_instances_equal_keys(self):
        """two instances with equal fields bake equal."""

        @dataclasses.dataclass(frozen=True)
        class Cfg:
            a: int
            b: str

        c1 = Cfg(1, "x")
        c2 = Cfg(1, "x")
        assert frozen_dataclass_key(c1) == frozen_dataclass_key(c2)

    def test_changing_field_changes_key(self):
        """changing one field changes the key."""

        @dataclasses.dataclass(frozen=True)
        class Cfg:
            a: int
            b: str

        c1 = Cfg(1, "x")
        c2 = Cfg(2, "x")
        assert frozen_dataclass_key(c1) != frozen_dataclass_key(c2)

    def test_extra_field_changes_key(self):
        """a dataclass with one extra field yields different key even when shared fields agree."""

        @dataclasses.dataclass(frozen=True)
        class Cfg1:
            a: int
            b: str

        @dataclasses.dataclass(frozen=True)
        class Cfg2:
            a: int
            b: str
            c: float

        c1 = Cfg1(1, "x")
        c2 = Cfg2(1, "x", 3.14)
        assert frozen_dataclass_key(c1) != frozen_dataclass_key(c2)

    def test_non_dataclass_raises_type_error(self):
        """a non-dataclass raises TypeError."""

        class NotDataclass:
            pass

        obj = NotDataclass()
        with pytest.raises(TypeError):
            frozen_dataclass_key(obj)


class TestClassify:
    """Test classify() function."""

    def test_classify_small_class(self):
        """classify on a small class with attributes a, b, c, d; policy covering some."""

        class SmallClass:
            def __init__(self):
                self.a = 1
                self.b = np.array([1.0, 2.0])
                self.c = [1, 2]
                self.d = lambda: None

        policy = {
            "a": content("structural"),
            "b": shape("data array"),
            "d": exclude("handle"),
        }

        modes, unclassified = classify(SmallClass(), policy)
        assert modes["a"] == "content"
        assert modes["b"] == "shape"
        assert modes["c"] == "content"
        assert "d" not in modes  # excluded
        assert unclassified == ("c",)

    def test_classify_unclassified_attributes(self):
        """classify reports unclassified attributes in sorted order."""

        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2
                self.z = 3

        policy = {"x": content("x")}
        _, unclassified = classify(Obj(), policy)
        assert unclassified == ("y", "z")

    def test_classify_invalid_mode_raises(self):
        """a policy row with mode not in MODES raises ValueError."""

        class Obj:
            def __init__(self):
                self.a = 1

        policy = {"a": ("bogus", "reason")}
        with pytest.raises(ValueError) as excinfo:
            classify(Obj(), policy)
        assert "bogus" in str(excinfo.value)


class TestDeriveKey:
    """Test derive_key() function."""

    def test_derive_key_full_example(self):
        """test derive_key with content, shape, exclude modes and version/tail."""

        class SmallClass:
            def __init__(self):
                self.a = 1
                self.b = np.array([1.0, 2.0])
                self.c = [1, 2]
                self.d = lambda: None

        policy = {
            "a": content("structural"),
            "b": shape("data array"),
            "d": exclude("handle"),
        }

        key = derive_key(SmallClass(), policy, version=3, tail=(True, "cpu"))
        # Key structure: (class_name, version, entries, tail_baked)
        assert key[0].endswith("SmallClass")
        assert key[1] == 3
        # entries should include a (content) and c (unclassified, defaulting to content)
        # b should be ("shape", ...)
        # d should be omitted
        assert isinstance(key[2], tuple)  # entries
        assert key[3] == (True, "cpu")  # tail

    def test_derive_key_changing_shape_changes_key(self):
        """changing b's shape changes the key, but changing values does not."""

        class Obj:
            def __init__(self, arr):
                self.a = 1
                self.b = arr

        policy = {"a": content("structural"), "b": shape("data array")}

        arr1 = np.array([1.0, 2.0])
        obj1 = Obj(arr1)
        key1 = derive_key(obj1, policy)

        arr2 = np.array([1.0, 2.0])  # same shape, different values
        obj2 = Obj(arr2)
        key2 = derive_key(obj2, policy)
        assert key1 == key2  # same key because shape mode ignores values

        arr3 = np.array([1.0, 2.0, 3.0])  # different shape
        obj3 = Obj(arr3)
        key3 = derive_key(obj3, policy)
        assert key1 != key3

    def test_derive_key_exclude_ignored(self):
        """changing d leaves the key unchanged when d is excluded."""

        class Obj:
            def __init__(self, func):
                self.a = 1
                self.func = func

        policy = {"a": content("structural"), "func": exclude("handle")}

        def f1():
            pass

        def f2():
            pass

        obj1 = Obj(f1)
        obj2 = Obj(f2)
        key1 = derive_key(obj1, policy)
        key2 = derive_key(obj2, policy)
        assert key1 == key2  # same key because func is excluded

    def test_derive_key_unclassified_defaults_to_content(self):
        """changing an unclassified attribute changes the key."""

        class Obj:
            def __init__(self, c):
                self.a = 1
                self.c = c

        policy = {"a": content("structural")}

        obj1 = Obj([1, 2])
        obj2 = Obj([1, 2, 3])
        key1 = derive_key(obj1, policy)
        key2 = derive_key(obj2, policy)
        assert key1 != key2

    def test_derive_key_version_in_key(self):
        """version parameter appears in the key and changing it changes the key."""

        class Obj:
            def __init__(self):
                self.a = 1

        policy = {"a": content("structural")}
        obj = Obj()

        key1 = derive_key(obj, policy, version=1)
        key2 = derive_key(obj, policy, version=2)
        assert key1 != key2

    def test_derive_key_tail_in_key(self):
        """tail parameter appears in the key and changing it changes the key."""

        class Obj:
            def __init__(self):
                self.a = 1

        policy = {"a": content("structural")}
        obj = Obj()

        key1 = derive_key(obj, policy, tail=(True, "cpu"))
        key2 = derive_key(obj, policy, tail=(False, "cpu"))
        assert key1 != key2

    def test_derive_key_shape_mode_non_array_raises(self):
        """a non-array under shape mode raises TypeError naming the attribute."""

        class Obj:
            def __init__(self):
                self.a = 1

        policy = {"a": shape("should be array")}

        with pytest.raises(TypeError) as excinfo:
            derive_key(Obj(), policy)
        assert "a" in str(excinfo.value)


class TestAssertPolicyComplete:
    """Test assert_policy_complete() function."""

    def test_complete_policy_passes(self):
        """passes for the full policy."""

        class Obj:
            def __init__(self):
                self.a = 1
                self.b = 2

        policy = {"a": content("a"), "b": content("b")}
        objs = [Obj()]
        assert_policy_complete(objs, policy)  # should not raise

    def test_unclassified_attribute_raises(self):
        """raises naming 'c' when c is unclassified."""

        class Obj:
            def __init__(self):
                self.a = 1
                self.b = 2
                self.c = 3

        policy = {"a": content("a"), "b": content("b")}
        objs = [Obj()]

        with pytest.raises(AssertionError) as excinfo:
            assert_policy_complete(objs, policy)
        assert "c" in str(excinfo.value)
        assert "unclassified" in str(excinfo.value).lower()

    def test_stale_policy_entry_raises(self):
        """raises naming 'zz' when the policy has a row for an attribute no object has."""

        class Obj:
            def __init__(self):
                self.a = 1
                self.b = 2

        policy = {"a": content("a"), "b": content("b"), "zz": exclude("unused")}
        objs = [Obj()]

        with pytest.raises(AssertionError) as excinfo:
            assert_policy_complete(objs, policy)
        assert "zz" in str(excinfo.value)
        assert "stale" in str(excinfo.value).lower()

    def test_multiple_objects_union_of_vars(self):
        """over an iterable of objects, the union of vars() names must match the policy."""

        class ObjA:
            def __init__(self):
                self.a = 1
                self.b = 2

        class ObjB:
            def __init__(self):
                self.b = 2
                self.c = 3

        policy = {"a": content("a"), "b": content("b"), "c": content("c")}
        objs = [ObjA(), ObjB()]
        assert_policy_complete(objs, policy)  # should not raise


class TestHelperFunctions:
    """Test helper functions content(), shape(), exclude()."""

    def test_content_helper(self):
        """content(reason) returns ("content", reason)."""
        result = content("test reason")
        assert result == ("content", "test reason")

    def test_shape_helper(self):
        """shape(reason) returns ("shape", reason)."""
        result = shape("test reason")
        assert result == ("shape", "test reason")

    def test_exclude_helper(self):
        """exclude(reason) returns ("exclude", reason)."""
        result = exclude("test reason")
        assert result == ("exclude", "test reason")


class TestModes:
    """Test MODES constant."""

    def test_modes_constant(self):
        """MODES contains the three expected strings."""
        assert MODES == ("content", "shape", "exclude")


class TestTypeAliases:
    """Test type aliases are defined."""

    def test_policy_row_alias_exists(self):
        """PolicyRow type alias is defined."""
        assert PolicyRow is not None

    def test_key_policy_alias_exists(self):
        """KeyPolicy type alias is defined."""
        assert KeyPolicy is not None


class TestClassObjects:
    """A class object is keyed by name, even when it defines cache_key."""

    def test_plain_class_is_keyed_as_callable(self):
        """A NamedTuple type stored on a model (the _Observables pattern) bakes by name."""

        class Observables:
            pass

        key = baked(Observables)
        assert key[0] == "callable"
        assert key[1].endswith("Observables")

    def test_class_defining_cache_key_is_not_called_unbound(self):
        """getattr(cls, "cache_key") is the unbound function; baked must not call it."""

        class WithKey:
            def cache_key(self):
                return ("k", 1)

        assert baked(WithKey)[0] == "callable"
        tag, qualname, key = baked(WithKey())
        assert tag == "cache_key"
        assert qualname.endswith("WithKey")  # a local class carries <locals> in its qualname
        assert key == ("k", 1)

    def test_dead_jax_arrays_are_evicted_from_the_memo(self):
        """A memo miss drops entries whose array was garbage-collected."""
        import gc

        from tengri._cache_keys import _ARRAY_KEY_CACHE

        arr = jnp.arange(5.0)
        array_key(arr)
        n_live = len(_ARRAY_KEY_CACHE)
        del arr
        gc.collect()
        array_key(jnp.arange(6.0))  # a miss triggers eviction
        assert len(_ARRAY_KEY_CACHE) <= n_live
