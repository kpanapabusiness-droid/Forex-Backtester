# Arc 7 v3.0 exit-extraction - best candidate

**Best Stage-B candidate (least-bad; still FAIL): `A3::cl0::sl_partial_close_1r_runner_trail::sl3.5::n5::exp2`**

- Verdict: **FAIL** - primary_failure_mode `step5_chained_dd_above_gate`
- Worst-fold ratio -0.614; worst-fold ROI -3.93% (r_base 0.5%); worst-fold DD 7.89% (DD@0.40% 6.31%)
- Chained IS+holdout DD 28.49% (gate <=10%) - the binding failure
- Negative folds 5/11; min trades/fold 0 (F1 2010 untrainable for A3 - empty IS)
- Holdout 2021: ROI -12.57%, DD 18.23%, n=438 - negative OOS
- Amendment-3: r_safe 0.5071%, scalable_to_safe=True

**Primary hypothesis (A1 full-pool + runner-trail, the Arc 10 frame) - all 6 FAIL:**

- `A1::full::sl_partial_close_1r_runner_trail::sl2.5::expinf`: worst-fold ROI -24.30%, worst-fold DD 30.52%, chained DD 77.0%, 8 neg folds, holdout -24.68%
- `A1::full::sl_partial_close_1r_runner_trail::sl3.5::exp2`: worst-fold ROI -12.25%, worst-fold DD 18.19%, chained DD 53.2%, 8 neg folds, holdout -1.00%
- `A1::full::sl_partial_close_1r_runner_trail::sl2.5::exp2`: worst-fold ROI -21.08%, worst-fold DD 25.77%, chained DD 62.9%, 7 neg folds, holdout -11.08%
- `A1::full::sl_partial_close_1r_runner_trail::sl3.5::expinf`: worst-fold ROI -22.52%, worst-fold DD 31.04%, chained DD 73.8%, 9 neg folds, holdout -21.90%
- `A1::full::sl_partial_close_1r_runner_trail::sl3.0::exp2`: worst-fold ROI -14.89%, worst-fold DD 16.41%, chained DD 64.8%, 7 neg folds, holdout -17.93%
- `A1::full::sl_partial_close_1r_runner_trail::sl3.0::expinf`: worst-fold ROI -26.38%, worst-fold DD 33.48%, chained DD 78.6%, 7 neg folds, holdout -19.92%
