# SPDX-License-Identifier: BSD-3-Clause
"""Census: every short key a sub-block accepts is read by that sub-block's type.

For every ``(category, type)`` in the composable AGN registry, every short key
the grammar accepts inside that sub-block must either be a parameter the selected
type declares as a read (:func:`_agn_subblock_declared_params`), or be refused by
:func:`_reject_inert_agn_subblock_keys`. A key the grammar accepts and the
refusal lets through is an inert key: written, validated, and then dropped.

Keys owned by the shared ``agn`` group (``agn_log_lbol``, ``agn_cos_inc``,
``agn_lum_ratio``) are exempt: the top level reads them for every block, so they
are not inert per type.

Grid-free and SSP-free: the refusal is pure Python over the consumes table.

Tier: contract.
"""

import pytest

from tengri import Fixed
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.config.exceptions import ParameterError
from tengri.parameters.agn_ownership import (
    _AGN_CONSUMES_CATEGORY,
    _AGN_PARTITION,
    _agn_param_group,
    _agn_subblock_declared_params,
)
from tengri.parameters.groups import (
    _name_spellings,
    _reject_inert_agn_subblock_keys,
    _short_names_for_group,
    _short_names_for_registered_type,
)

pytestmark = pytest.mark.contract

PARTITION = {name: _agn_param_group(name) for name in _AGN_PARTITION}
_CATEGORY_OF_CONSUMES = {v: k for k, v in _AGN_CONSUMES_CATEGORY.items()}
# Every spelling (short, full, legacy) -> the canonical partition name it resolves to.
_CANONICAL = {}
for _name in PARTITION:
    for _spelling in _name_spellings([_name]):
        _CANONICAL.setdefault(_spelling, _name)


def _census_cases():
    """Every (grammar category, block type) the composable registry holds."""
    cases = []
    for consumes_category, types in AGN_BLOCKS.items():
        category = _CATEGORY_OF_CONSUMES.get(consumes_category)
        if category is None:
            continue
        for block_type in sorted(types):
            cases.append((category, block_type))
    return cases


def _inert_triples():
    """(category, type, key) for every accepted, owned, undeclared key the refusal lets through."""
    offending = []
    for category, block_type in _census_cases():
        group = f"agn.{category}"
        declared = _agn_subblock_declared_params(
            category, block_type, selection={category: block_type}
        )
        assert declared is not None, f"no declared reads for {category!r} {block_type!r}"
        accepted = _short_names_for_group(group, PARTITION) | _short_names_for_registered_type(
            block_type
        )
        for key in sorted(accepted):
            full = _CANONICAL.get(key, key)
            if PARTITION.get(full) == "agn":
                continue
            try:
                _reject_inert_agn_subblock_keys(
                    category,
                    {"type": block_type, key: Fixed(0.0)},
                    {category: block_type},
                    PARTITION,
                )
                refused = False
            except ParameterError:
                refused = True
            if full in declared and refused:
                offending.append((category, block_type, key, "refused but read"))
            elif full not in declared and not refused:
                offending.append((category, block_type, key, "accepted but not read"))
    return offending


def test_every_accepted_key_is_read_by_its_selected_type():
    """Census over the whole composable registry. Prints each offending triple."""
    offending = _inert_triples()
    assert not offending, (
        f"{len(offending)} (category, type, key) triples accepted but not read, "
        f"or refused but read:\n"
        + "\n".join(f"  {c!r} {t!r} {k!r}: {why}" for c, t, k, why in offending)
    )
