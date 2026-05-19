Cross-component contract
========================

Tengri's forward model is built from a list of :class:`SEDComponent`
adapters. Each component declares — at construction time, in pure
metadata — the named cross-component quantities it *publishes* into
:attr:`PipelineState.derived` and the keys it *requires* from upstream
producers. The orchestrator validates this contract once at
:func:`tengri.forward.build_components` time, before any JIT trace,
catching renames, units drift, missing publishers, duplicate
publishers, and out-of-order dependencies.

See ``docs/dev/component_architecture.md`` for a tutorial; this page is
the API reference for the types involved.

Derived-data types
------------------

DerivedKey
~~~~~~~~~~

.. autoclass:: tengri.DerivedKey
   :members:
   :show-inheritance:

DerivedBundle
~~~~~~~~~~~~~

.. autoclass:: tengri.DerivedBundle
   :members:
   :show-inheritance:

PipelineContractError
~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: tengri.PipelineContractError
   :members:
   :show-inheritance:

Orchestrator helpers
--------------------

These run once at :func:`build_components` time. They are not on the
hot JIT-traced path and are intended for advanced users who construct
their own component lists or write new physics modules.

validate_pipeline
~~~~~~~~~~~~~~~~~

.. autofunction:: tengri.forward.orchestrator.validate_pipeline

topological_sort
~~~~~~~~~~~~~~~~

.. autofunction:: tengri.forward.orchestrator.topological_sort

merge_declared_parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tengri.forward.orchestrator.merge_declared_parameters

run_components
~~~~~~~~~~~~~~

.. autofunction:: tengri.forward.orchestrator.run_components

Related ADRs
------------

- :file:`docs/adr/0006-topological-component-ordering.md` — derived
  ordering from declared dependencies.
- :file:`docs/adr/0007-typed-derived-bundle.md` — replacing the
  free-form ``Mapping[str, Any]`` with a typed container.
- :file:`docs/adr/0009-typed-pipeline-contract.md` — the typed
  ``publishes`` / ``requires`` / ``requires_optional`` decision
  (originally authored as ADR-0004; renumbered).
