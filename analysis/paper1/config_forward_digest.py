# SPDX-License-Identifier: BSD-3-Clause
"""Does a grid configuration predict the same photometry on two source trees?

The 20x6 grid's Configuration III row is complete on pinned code. If the pin
moves to main mid-grid, the remaining five rows are computed by different
forward-model code, and Section 7's configuration-to-configuration spread would
carry a code change inside it with nothing in the cells to separate the two.
This answers whether that matters, per configuration, before anyone re-runs
anything.

It is deliberately *not* :mod:`paper1.mock_forward_digest`. That module asks the
same question of the kitchen-sink mock, whose filter set includes GALEX. The
CANDELS set does not, and the bluest band decides whether the Lyman-continuum
fixes can fire at all, so the mock's answer does not transfer.

**The two arms must evaluate the same parameter vector.** ``spec.sample()``
does not produce one: PR #2296 made it return free keys only, so the same call
yields a different dict on either side of that commit, and a comparison built
on two sampled dicts would report a parameter difference as a physics
difference. So one arm samples and writes the vector with ``--params-out``, the
other reads it with ``--params-in``, and both restrict to the free set.

The observation is taken from a real completed cell rather than rebuilt, so the
filters and the redshift are the ones the grid actually fitted.

CLI::

    # arm A, on the pinned tree
    python -m paper1.config_forward_digest III --cell <cell>.json \\
        --params-out p.json --out pin.npz
    # arm B, on a main snapshot (see mock_forward_digest's recipe for the
    # TENGRI_DATA_DIR and TENGRI_DISABLE_PRECOMP_CACHE requirements)
    python -m paper1.config_forward_digest III --cell <cell>.json \\
        --params-in p.json --out main.npz
    python -m paper1.config_forward_digest --compare pin.npz main.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for entry in (str(HERE.parent), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

#: Sampling key. Any fixed value works; it is pinned so a rerun of one arm
#: reproduces its own vector rather than silently drawing a new one.
SAMPLE_SEED = 0


def _config_fn(key: str):
    import configs

    fn = getattr(configs, f"config_{key}", None)
    if fn is None:
        raise SystemExit(f"no configuration {key!r}; expected one of I II III IV V VI")
    return fn, configs


def digest(
    config_key: str,
    cell_path: Path,
    params_in: Path | None,
    params_out: Path | None,
    borrow_observation: bool = False,
):
    """One tree's predicted photometry for one configuration on one galaxy."""
    import jax

    import tengri
    from tengri.observation import Observation, Photometry

    jax.config.update("jax_enable_x64", True)

    cell = json.loads(cell_path.read_text())
    for required in ("gal_id", "z", "filter_names", "config"):
        if cell.get(required) is None:
            raise SystemExit(f"{cell_path} records no {required!r}")
    if cell["config"] != config_key and not borrow_observation:
        raise SystemExit(
            f"{cell_path} is configuration {cell['config']}, not {config_key}.\n"
            "The observation it carries -- filters, redshift -- belongs to the "
            "GALAXY and is shared by every configuration, so borrowing it is "
            "legitimate; only Configuration III has completed cells, so the "
            "other five have no cell of their own to read. Pass "
            "--borrow-observation to say so deliberately. The refusal is here "
            "because a borrowed cell must not be mistaken later for a cell of "
            "the configuration named on the digest."
        )
    if cell["config"] != config_key:
        print(
            f"borrowing the observation of a configuration {cell['config']} cell "
            f"for configuration {config_key}: galaxy {cell['gal_id']}, z "
            f"{float(cell['z']):.4f}, {len(cell['filter_names'])} bands"
        )

    fn, configs_mod = _config_fn(config_key)
    # load_ssp_for takes the CONFIGURATION key and does the SSP_FOR_CONFIG
    # lookup itself; passing the SSP name double-indexes and raises KeyError.
    ssp = configs_mod.load_ssp_for(config_key)
    filters = [str(f) for f in cell["filter_names"]]
    obs = Observation(photometry=Photometry.from_names(filters))
    model = fn(ssp, obs, float(cell["z"]))

    free = sorted(str(n) for n in model.spec.free_params)

    if params_in is not None:
        stored = json.loads(params_in.read_text())
        if sorted(stored) != free:
            raise SystemExit(
                "the two trees declare different free parameters for "
                f"configuration {config_key}, so the same vector cannot be "
                f"evaluated on both.\n  only in the file: {sorted(set(stored) - set(free))}"
                f"\n  only in this tree: {sorted(set(free) - set(stored))}\n"
                "That difference is itself the answer: the model changed, not "
                "only its prediction."
            )
        params = {k: float(v) for k, v in stored.items()}
    else:
        drawn = model.spec.sample(key=jax.random.PRNGKey(SAMPLE_SEED))
        # Free keys only: a Fixed key in the dict is refused outright on trees
        # carrying #2296 and honored on trees without it, which is exactly the
        # divergence this comparison must not import into its inputs.
        params = {k: float(np.asarray(drawn[k])) for k in free}
        if params_out is not None:
            params_out.write_text(json.dumps(params, indent=2, sort_keys=True))

    phot = np.asarray(model.predict_photometry(params), dtype=np.float64)
    return {
        "photometry": phot,
        "filter_names": np.array(filters),
        "free_params": np.array(free),
        "params_values": np.array([params[k] for k in free]),
        "config": np.array(config_key),
        "gal_id": np.array(int(cell["gal_id"])),
        "z": np.array(float(cell["z"])),
        "tengri_path": np.array(str(Path(tengri.__file__).resolve().parent)),
    }


def compare(a: Path, b: Path) -> int:
    """Per-band fractional change between two trees' predictions."""
    left = dict(np.load(a, allow_pickle=True))
    right = dict(np.load(b, allow_pickle=True))

    for key, label in (("free_params", "free parameters"), ("filter_names", "filters")):
        if list(left[key]) != list(right[key]):
            print(f"REFUSED: the two trees declare different {label}.")
            print(f"  {a.name}: {list(left[key])}")
            print(f"  {b.name}: {list(right[key])}")
            return 1
    if not np.array_equal(left["params_values"], right["params_values"]):
        print(
            "REFUSED: the two digests were evaluated at different parameter "
            "values, so any difference below would be that, not the code. Use "
            "--params-out on one arm and --params-in on the other."
        )
        return 1

    lp, rp = left["photometry"], right["photometry"]
    if lp.shape != rp.shape:
        print(f"REFUSED: photometry shapes differ, {lp.shape} vs {rp.shape}")
        return 1

    identical = int(np.sum(lp == rp))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(lp != 0, (rp - lp) / lp, 0.0)

    print(f"configuration {left['config']}  galaxy {left['gal_id']}  z {float(left['z']):.4f}")
    print(f"  A {a.name}: {left['tengri_path']}")
    print(f"  B {b.name}: {right['tengri_path']}")
    print(f"\n  bit-identical bands: {identical}/{lp.size}")
    print(f"\n{'band':<14}{'A':>14}{'B':>14}{'frac change':>14}")
    print("-" * 56)
    for name, x, y, f in zip(left["filter_names"], lp, rp, frac):
        print(f"{name!s:<14}{x:>14.6e}{y:>14.6e}{f:>13.3%}")
    worst = int(np.argmax(np.abs(frac)))
    print(
        f"\n  worst band: {left['filter_names'][worst]} at {frac[worst]:.4%}"
        if identical != lp.size
        else "\n  every band bit-identical: this configuration does not move between the trees."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="configuration key, I through VI")
    parser.add_argument("--cell", type=Path, help="a completed cell JSON for this configuration")
    parser.add_argument("--out", type=Path, help="where to write this tree's digest")
    parser.add_argument("--params-out", type=Path, help="write the sampled vector here")
    parser.add_argument("--params-in", type=Path, help="evaluate this vector instead of sampling")
    parser.add_argument(
        "--borrow-observation",
        action="store_true",
        help="use a cell of another configuration for its filters and redshift",
    )
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare)
    if not (args.config and args.cell and args.out):
        parser.error("need config, --cell and --out (or --compare A B)")

    payload = digest(
        args.config, args.cell, args.params_in, args.params_out, args.borrow_observation
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(
        f"wrote {args.out}  ({payload['photometry'].size} bands, {len(payload['free_params'])} free)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
