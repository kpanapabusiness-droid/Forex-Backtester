"""HistData M1 bid+ask data layer.

Public API:
    load_m1(pair, ...)             # core.data.histdata_loader
    aggregate(pair, tf, ...)       # core.data.aggregator
    compute_cache_key(...)         # core.data.cache_keys

The data layer reads HistData M1 derived CSVs from
``data/histdata/<PAIR>/m1/{bid,ask}/<YYYY>/<PAIR>_M1_{BID,ASK}_<YYYYMM>.csv``,
joins bid+ask into a single per-pair DataFrame, and caches the result as
parquet under ``data/cache/m1/<PAIR>.parquet``. Higher TFs (M5..W1) are
deterministically aggregated from the M1 cache and cached at
``data/cache/<TF>/<PAIR>.parquet``.

Cache invalidation is keyed on ``data/histdata/m1_manifest.json``: a per-pair
cache_key is the sha256 of the sorted ``(relpath, sha256)`` pairs for that pair
in the M1 manifest. Higher-TF cache keys derive from the upstream M1 cache_key.
"""
