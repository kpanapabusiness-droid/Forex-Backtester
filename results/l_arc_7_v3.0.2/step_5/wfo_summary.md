# Step 5 — WFO Summary (v3.0.2)

**Total configs evaluated:** 45 (thin)

## Architecture-map override applied

- c0 (bimodal, AUC 0.6192): ('A1', 'A4') 
- c1 (unclassified, AUC 0.6642): ('A1', 'A2', 'A6') (augmented per override)

## Top-K Amendment 3 results

| Rank | Config | Verdict (Amended) | Worst ratio | r_safe | r_hard | Chained DD base | Daily breaches @r_safe |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `A6::A6::cl1::sl2.0::thr0.5-0.7::exp2` | fail | 4.36 | 3.1634% | 3.9542% | 2.0914% | 3 |
| 2 | `A6::A6::cl1::sl1.5::thr0.5-0.7::exp2` | fail | 3.85 | 2.1936% | 2.7421% | 2.7631% | 2 |
| 3 | `A6::A6::cl1::sl2.0::thr0.4-0.6::exp2` | fail | 3.51 | 1.4268% | 1.7835% | 3.3550% | 2 |
