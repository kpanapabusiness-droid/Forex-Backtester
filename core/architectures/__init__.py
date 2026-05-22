"""L_PROTOCOL v3.0 Step 5 architectures A1..A6.

Each architecture exposes a uniform ``run(...)`` interface returning
:class:`StrategyResult`. See :mod:`core.architectures._protocol` for the
contract and :mod:`core.runners.arc_fold_runner` for the WFO driver
that invokes them.

Architectures:

  - A1 :mod:`core.architectures.a1_system_level_filter`
  - A2 :mod:`core.architectures.a2_classifier_filter`
  - A3 :mod:`core.architectures.a3_pipeline_de`
  - A4 :mod:`core.architectures.a4_pipeline_d_exits`
  - A5 :mod:`core.architectures.a5_portfolio_composition`
  - A6 :mod:`core.architectures.a6_meta_labeling`
"""
