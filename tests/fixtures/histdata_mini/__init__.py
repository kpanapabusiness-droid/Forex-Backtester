"""Synthetic HistData mini-fixture for the data-layer tests.

A pytest fixture lives in ``tests/conftest.py`` (or per-test) that constructs
this fixture inside a ``tmp_path``: two pairs × two months × bid+ask CSVs plus
an ``m1_manifest.json`` referencing them. See ``tests/test_histdata_loader.py``
for usage.
"""
