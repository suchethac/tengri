"""Tests for tengri.runtime.deprecation helpers.

Covers:
- deprecated_alias: DeprecationWarning emitted, return value preserved
- deprecated_class_alias: warning on instantiation, isinstance/issubclass delegation
"""

import warnings

import pytest

from tengri.runtime.deprecation import deprecated_alias, deprecated_class_alias

# ── deprecated_alias ──────────────────────────────────────────────


class TestDeprecatedAlias:
    def test_warning_emitted(self):
        """Calling an aliased function emits DeprecationWarning."""

        @deprecated_alias("new_name")
        def old_fn():
            return 42

        with pytest.warns(DeprecationWarning, match="old_fn"):
            old_fn()

    def test_canonical_name_in_warning(self):
        """Warning message includes the canonical name."""

        @deprecated_alias("new_canonical")
        def old_func():
            return 1

        with pytest.warns(DeprecationWarning, match="new_canonical"):
            old_func()

    def test_remove_in_in_warning(self):
        """Warning message includes the removal version."""

        @deprecated_alias("better_fn", remove_in="2.0")
        def legacy():
            return 0

        with pytest.warns(DeprecationWarning, match="2.0"):
            legacy()

    def test_return_value_preserved(self):
        """Deprecated alias returns the wrapped function's value."""

        @deprecated_alias("new_fn")
        def old():
            return {"key": 99}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = old()

        assert result == {"key": 99}

    def test_args_forwarded(self):
        """Arguments are forwarded correctly to the underlying function."""

        @deprecated_alias("add_new")
        def add_old(a, b):
            return a + b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = add_old(3, 4)

        assert result == 7

    def test_kwargs_forwarded(self):
        """Keyword arguments are forwarded correctly."""

        @deprecated_alias("greet_new")
        def greet(name="world"):
            return f"hello {name}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = greet(name="tengri")

        assert result == "hello tengri"

    def test_functools_wraps_preserves_name(self):
        """functools.wraps keeps the original function name and docstring."""

        @deprecated_alias("new_fn")
        def my_function():
            """Original docstring."""
            return 0

        assert my_function.__name__ == "my_function"
        assert "Original docstring" in (my_function.__doc__ or "")


# ── deprecated_class_alias ────────────────────────────────────────


class _RealClass:
    """Stand-in for the canonical new class."""

    def __init__(self, value=0):
        self.value = value


class TestDeprecatedClassAlias:
    def test_instantiation_emits_warning(self):
        """Instantiating the alias class emits DeprecationWarning."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        with pytest.warns(DeprecationWarning, match="OldClass"):
            OldClass()

    def test_new_class_name_in_warning(self):
        """Warning message mentions the canonical class name."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        with pytest.warns(DeprecationWarning, match="_RealClass"):
            OldClass()

    def test_returns_real_instance(self):
        """Alias instantiation returns an instance of the real class."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj = OldClass(value=7)
        assert obj.value == 7

    def test_isinstance_against_alias(self):
        """isinstance(real_instance, alias_class) returns True via __instancecheck__."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        real_obj = _RealClass(value=3)
        assert isinstance(real_obj, OldClass)

    def test_isinstance_against_real_class(self):
        """Instance from alias is also an instance of the real class."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj = OldClass()
        assert isinstance(obj, _RealClass)

    def test_issubclass_delegation(self):
        """issubclass(_RealClass, alias_class) returns True via __subclasscheck__."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        assert issubclass(_RealClass, OldClass)

    def test_args_forwarded_to_real_class(self):
        """Constructor arguments are forwarded to the real class."""
        OldClass = deprecated_class_alias("OldClass", _RealClass)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj = OldClass(value=42)
        assert obj.value == 42
