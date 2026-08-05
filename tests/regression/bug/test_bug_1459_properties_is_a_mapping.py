# SPDX-License-Identifier: BSD-3-Clause
"""``posterior.properties`` must satisfy the mapping protocol it implies (#1459).

``PosteriorProperties`` implemented ``__getitem__``, ``__iter__``,
``__contains__`` and ``keys`` but not ``__len__``, so it read as a dict right
up to ``len()`` and ``.items()`` — the two calls one reaches for first when
inspecting a fit result — which raised a bare ``TypeError`` /
``AttributeError`` naming neither the object nor the supported surface.

It is on the path every user takes after a fit, being the documented
replacement for the deprecated ``Posterior.derived``.

Asserted against the class rather than a completed fit: the protocol is a
property of the type, and running a sampler to check ``len()`` would make a
cheap contract depend on an expensive one.
"""

import collections.abc as abc

import pytest

from tengri.inference.posterior import PosteriorProperties

pytestmark = pytest.mark.regression_bug


def test_it_is_a_mapping():
    assert issubclass(PosteriorProperties, abc.Mapping)


@pytest.mark.parametrize(
    "method",
    ["__getitem__", "__iter__", "__len__", "__contains__", "keys", "items", "values", "get"],
)
def test_the_full_mapping_surface_is_present(method):
    """``items``/``values``/``get`` come from the ABC once ``__len__`` exists."""
    assert hasattr(PosteriorProperties, method), f"{method} missing"


def test_len_is_defined_on_the_class_not_inherited_as_a_stub():
    """``__len__`` must be real — ``Mapping`` declares it abstract, not free.

    Subclassing ``Mapping`` without defining ``__len__`` would make the class
    abstract and fail at construction instead, so this pins the actual fix
    rather than the registration.
    """
    assert "__len__" in PosteriorProperties.__dict__


def test_it_is_not_accidentally_mutable():
    """The read-only contract must survive becoming a Mapping.

    ``Mapping`` is the immutable half of the protocol; picking up
    ``MutableMapping`` by mistake would silently add ``__setitem__``.
    """
    assert not issubclass(PosteriorProperties, abc.MutableMapping)
    assert not hasattr(PosteriorProperties, "__setitem__")
