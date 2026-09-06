# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1120: components reachable from the build grammar.

Components registered in _REGISTRY but absent from their axis menu are
silently unreachable to users — describe(), search(), and list_*() all fail
to name them, and SEDModel.build(xray={'type': 'xray_aird'}) is the only
path to instantiate them. This mirrors #1273 (dust structural axis) and
#1276 (radio sub-block axis invisible to discovery).
"""

import pytest

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize(
    "axis,expected",
    [
        ("xray", {"xray_aird", "agn_xray_corona"}),
        ("radio", {"radio_powerlaw", "radio_dpl"}),
    ],
)
def test_registered_components_reachable_from_grammar(axis, expected):
    """#1120: every component registered for an axis must appear in its
    grammar menu, or it is silently unreachable.

    This inverts the T5 guard (menu subset of builder), and tests the
    builder subset of menu — every type the grammar accepts must be
    named by the corresponding discovery menu.
    """
    from tengri.parameters.groups import _valid_radio_types, _valid_xray_types

    if axis == "xray":
        menu = _valid_xray_types()
    elif axis == "radio":
        menu = _valid_radio_types()
    else:
        raise ValueError(f"Unknown axis: {axis}")

    assert expected <= menu, f"unreachable on {axis}: {expected - menu}"


@pytest.mark.parametrize(
    "axis,expected",
    [
        ("xray", [("xray_aird", "Fixed(DEFAULT)"), ("agn_xray_corona", "Fixed(DEFAULT)")]),
        ("radio", [("radio_powerlaw", "Fixed(DEFAULT)"), ("radio_dpl", "Fixed(DEFAULT)")]),
    ],
)
def test_newly_reachable_components_build_with_fixed_defaults(
    axis, expected, synthetic_ssp_wide, synthetic_tophat_obs
):
    """Each newly reachable component must actually construct with default
    parameters ('all_params': Fixed(DEFAULT)).

    A name appearing in a menu but failing to build would be the #1279
    (T5) mirror image — moving the lie, not fixing it. These components
    need grid coverage beyond the optical (UV and FIR for X-ray, radio);
    use synthetic_ssp_wide + synthetic_tophat_obs (session fixtures) which
    span 100 Angstrom – 1 mm.
    """
    from tengri import DEFAULT, Fixed, SEDModel

    for comp_name, _ in expected:
        # #1980: the radio menu's {'type': name} spelling is retired — a menu
        # name is reachable through its composable resolution instead.
        if axis == "radio":
            from tengri.parameters.groups import _legacy_radio_type_to_blocks

            sf_variant, agn_variant = _legacy_radio_type_to_blocks(comp_name)
            cfg = {
                "sf": {"type": sf_variant},
                "agn": {"type": agn_variant},
                "all_params": Fixed(DEFAULT),
            }
        else:
            cfg = {"type": comp_name, "all_params": Fixed(DEFAULT)}
        extra = {}
        if axis == "radio":
            # The legacy mapping leaves the SF arm on its FIRRC default
            # (bell2003), which normalizes against L_ir and is refused at
            # build time without a dust component (#2106). Reachability is
            # the subject here, so supply the dust the FIRRC arm requires.
            extra = {
                "dust_attenuation": {
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                "dust_emission": {"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            }
        try:
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                redshift=0.1,
                **{axis: cfg},
                **extra,
            )
            # If we reach here, the model built successfully.
            assert model is not None
        except Exception as exc:
            pytest.fail(f"Component {comp_name} on {axis} failed to build: {exc}")
