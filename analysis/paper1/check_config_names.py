"""Check that every model name the configurations declare is actually registered.

Configuration III declared ``law="charlot_fall2000"`` for most of a day. No such
law exists; that string is a *citation key* which the framework maps to
``power_law``. It survived because building the row needs a stellar library that
was absent from the machine where it was written, so the row was never executed
there -- and it failed on the first machine that had the library.

The gap this closes is that validating a name should not require the data. A
name is either in a registry or it is not, and that question is answerable in
milliseconds without touching an SSP grid, a Cloudy grid, or JAX. This guard
therefore runs everywhere, including a machine that cannot build half the suite.

It reads the declarations rather than the built models, for the same reason: a
build needs data, and needing data is what let the bad name hide.

What it cannot catch, stated so nobody trusts it too far:

* A registered name that is wrong for the science -- ``calzetti`` where
  ``kriek_conroy`` was meant. Both exist; only a reader can tell.
* A name valid in some registry but wrong for its group, except for dust laws,
  which are checked against the dust-law registry specifically.
* Anything about the values. ``tau_v=Uniform(0, 300)`` is nonsense and passes.

Run it directly, or import ``check`` from a test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import tengri

CONFIGS_PY = Path(__file__).with_name("configs.py")

#: Keys whose string value names a dust attenuation law. Checked against the
#: dust-law registry specifically, because that is where the defect was.
LAW_KEYS = ("law", "law_bc", "law_diff", "law_neb")

#: Registries a ``"type"`` value may legitimately name. Checked as a union: the
#: AST does not reliably say which build group a nested dict belongs to once
#: helper functions are involved, and an unregistered name is worth catching
#: even without knowing which registry it should have been in.
TYPE_REGISTRIES = (
    "list_sfh_models",
    "list_dust_emission_models",
    "list_dust_laws",
    "list_nebular_backends",
    "list_agn_models",
    "list_agn_blocks",
    "list_xray_models",
    "list_radio_models",
    "list_radio_blocks",
    "list_igm_models",
    "list_metallicity_modes",
    "list_shock_models",
)

#: Structural values that are not registry names.
STRUCTURAL = frozenset(
    {"single_component", "two_component", "composable", "none", "table", "ramp"}
)


def _names(list_fn: str) -> set[str]:
    """Registered names from one ``tengri.list_*`` helper."""
    try:
        entries = getattr(tengri, list_fn)()
    except Exception:
        return set()
    out: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                out.add(name)
        elif isinstance(entry, str):
            out.add(entry)
    return out


def _citation_keys() -> dict[str, str]:
    """Map citation key -> the law that cites it, for a better message.

    ``charlot_fall2000`` is not a law but it is what the framework cites for
    ``power_law``, so a user reaching for it has the right idea and the wrong
    string. Saying which law it cites turns a dead end into a fix.
    """
    from tengri.citations.resolve import NAME_TO_BIBKEY

    # NAME_TO_BIBKEY maps registry name -> bibliography key. Inverting it answers
    # "what did the user mean by this citation key", which is the question a
    # failed lookup actually poses. Several names can share one key (both smc
    # and lmc cite gordon2003_smc); keep the first, since the message is a hint
    # rather than a resolution.
    inverted: dict[str, str] = {}
    for name, bibkey in NAME_TO_BIBKEY.items():
        inverted.setdefault(bibkey, name)
    return inverted


def declared_names(source: str) -> list[tuple[int, str, str]]:
    """Every ``(lineno, key, value)`` naming a model in a dict literal."""
    tree = ast.parse(source)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, str):
                continue
            key = key_node.value
            if key in LAW_KEYS or key == "type":
                found.append((key_node.lineno, key, value_node.value))
    return found


def check(path: Path = CONFIGS_PY) -> list[str]:
    """Return a problem per unregistered name; empty means clean."""
    source = path.read_text()
    laws = _names("list_dust_laws")
    all_types: set[str] = set()
    for fn in TYPE_REGISTRIES:
        all_types |= _names(fn)
    citations = _citation_keys()

    problems: list[str] = []
    for lineno, key, value in declared_names(source):
        if value in STRUCTURAL:
            continue
        if key in LAW_KEYS:
            if value not in laws:
                hint = ""
                if value in citations:
                    hint = f" That is a citation key; the law it cites is {citations[value]!r}."
                problems.append(
                    f"{path.name}:{lineno}: {key}={value!r} is not a registered "
                    f"dust law.{hint} Registered: {sorted(laws)}"
                )
        elif value not in all_types:
            problems.append(f"{path.name}:{lineno}: type={value!r} is not in any model registry.")

    ssp_known = _names("list_known_ssps")
    if ssp_known:
        from paper1.configs import SSP_FOR_CONFIG

        for cfg, grid in SSP_FOR_CONFIG.items():
            if grid not in ssp_known:
                problems.append(
                    f"Configuration {cfg} names SSP grid {grid!r}, which is not a "
                    f"known library. Known names are listed by tengri.list_known_ssps()."
                )
    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"FAIL: {len(problems)} unregistered name(s)\n")
        for p in problems:
            print(f"  {p}\n")
        return 1
    print("OK: every model name declared in configs.py is registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
