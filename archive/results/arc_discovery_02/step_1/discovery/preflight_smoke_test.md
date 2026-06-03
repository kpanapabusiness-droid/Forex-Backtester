# Pre-flight smoke test — arc_discovery_02

> 50-rule smoke per dispatch §5 + Amendment D (density filter). First 50 density-passed rules from the same seed=42 generator the full 10k run uses.

## Verdict: **HALT**

| Metric | Value |
|---|---|
| Total rules evaluated | 50 |
| Pool-floor pass count | 41 (82.0%) |
| Evaluation timeouts (iter-cap) | 4 (8.0%) |
| Clean fail-floor (below 500 trades) | 5 |
| Wall-clock per rule: p50 / p90 / max | 13.39s / 66.40s / 80.80s |
| Setup wall-clock (panels + features + grid) | 50.0s |
| Density-filter attempts (to find first 50) | 839 |
| Density-filter accept rate | 6.0% |
| Density-filter rejected | 789 |
| Density-filter cap hit | False |
| Density-filter wall-clock | 1.1s |
| 50-rule loop wall-clock | 1108.6s |
| **Extrapolated 10k wall-clock (incl. density-filter pass)** | **61.6h** |

## Chat go/no-go criteria (Amendment F — wall-clock informational only)

| Criterion | Threshold | Observed | Blocking? |
|---|---|---|---|
| Pool-floor pass rate | ≥ 70% | 82.0% | YES |
| Iter-cap fire rate | ≤ 5% | 8.0% | YES |
| 10k wall-clock projection | (informational) | 61.6h | NO — checkpointed |

## Per-rule timing (first 50)

| Rule ID | n_atoms | Pool size | Iters consumed | Eval timeout | Floor pass | Time-exit % | Wall-clock (s) |
|---|---|---|---|---|---|---|---|
| 29 | 5 | 23499 | 600789 | False | True | 14.6% | 8.91 |
| 34 | 2 | 7 | 80 | False | False | 0.0% | 0.01 |
| 36 | 4 | 46721 | 1184672 | False | True | 12.9% | 18.65 |
| 60 | 3 | 1228 | 31185 | False | True | 12.0% | 0.49 |
| 61 | 2 | 38405 | 974034 | False | True | 13.1% | 15.38 |
| 77 | 4 | 9712 | 272097 | False | True | 20.7% | 4.47 |
| 94 | 5 | 308 | 8836 | False | False | 21.4% | 0.19 |
| 98 | 5 | 29245 | 699360 | False | True | 11.4% | 13.95 |
| 110 | 3 | 102247 | 2573290 | False | True | 12.8% | 42.15 |
| 113 | 2 | 67391 | 1775672 | False | True | 14.8% | 27.99 |
| 127 | 2 | 23622 | 586048 | False | True | 12.9% | 8.81 |
| 130 | 1 | 119772 | 3023959 | False | True | 12.8% | 48.11 |
| 211 | 1 | 199053 | 5000005 | True | False | 12.7% | 78.33 |
| 218 | 1 | 8010 | 174232 | False | True | 7.7% | 3.06 |
| 289 | 2 | 123697 | 3152166 | False | True | 13.7% | 48.52 |
| 301 | 5 | 23671 | 593335 | False | True | 13.0% | 8.89 |
| 320 | 5 | 29624 | 692750 | False | True | 10.3% | 11.00 |
| 332 | 1 | 8 | 91 | False | False | 0.0% | 0.02 |
| 335 | 3 | 63489 | 1658503 | False | True | 14.5% | 25.96 |
| 338 | 4 | 37418 | 818325 | False | True | 8.5% | 13.78 |
| 355 | 1 | 48584 | 1231269 | False | True | 12.9% | 19.25 |
| 391 | 3 | 199422 | 5000023 | True | False | 12.6% | 80.80 |
| 396 | 4 | 198742 | 5000030 | True | False | 12.6% | 77.17 |
| 428 | 5 | 12048 | 305631 | False | True | 13.0% | 4.59 |
| 430 | 4 | 25814 | 643961 | False | True | 12.6% | 10.00 |
| 446 | 3 | 7444 | 186461 | False | True | 10.7% | 3.06 |
| 448 | 5 | 168860 | 4063204 | False | True | 10.9% | 67.35 |
| 452 | 1 | 8 | 91 | False | False | 0.0% | 0.01 |
| 461 | 2 | 123442 | 3117025 | False | True | 12.8% | 48.82 |
| 475 | 3 | 38272 | 930830 | False | True | 11.3% | 14.72 |
| 507 | 2 | 15567 | 430935 | False | True | 18.8% | 6.98 |
| 521 | 3 | 137766 | 3544478 | False | True | 13.4% | 57.83 |
| 522 | 4 | 15103 | 349374 | False | True | 11.1% | 6.81 |
| 540 | 2 | 38976 | 939663 | False | True | 11.1% | 16.03 |
| 588 | 4 | 6414 | 188237 | False | True | 23.3% | 2.89 |
| 627 | 3 | 96570 | 2415369 | False | True | 12.6% | 39.49 |
| 628 | 3 | 48732 | 1234205 | False | True | 13.5% | 20.42 |
| 643 | 4 | 431 | 10416 | False | False | 12.1% | 0.18 |
| 649 | 3 | 17458 | 426854 | False | True | 12.1% | 7.53 |
| 652 | 5 | 1070 | 21523 | False | True | 6.5% | 0.46 |
| 683 | 1 | 4815 | 139070 | False | True | 19.1% | 2.14 |
| 684 | 1 | 199309 | 5000002 | True | False | 12.6% | 80.31 |
| 755 | 4 | 52867 | 1348569 | False | True | 13.1% | 21.10 |
| 764 | 2 | 852 | 21232 | False | True | 11.7% | 0.33 |
| 768 | 3 | 124917 | 3183023 | False | True | 13.6% | 49.76 |
| 800 | 2 | 35356 | 793266 | False | True | 9.0% | 13.00 |
| 804 | 1 | 123442 | 3117025 | False | True | 12.8% | 52.77 |
| 812 | 4 | 8010 | 174232 | False | True | 7.7% | 3.22 |
| 828 | 5 | 9713 | 215571 | False | True | 7.8% | 4.31 |
| 838 | 3 | 39684 | 968184 | False | True | 11.9% | 16.59 |
