## Summary
Why + what changed (1-2 lines).

## Checks (paste results)
- [ ] ruff check .
- [ ] pytest -q
- [ ] python scripts/smoke_test_selfcontained_v198.py -q --mode fast
- [ ] Indicator contracts respected (C1/C2/baseline/volume/exit)
- [ ] Config-driven only (no hardcoded params)
- [ ] Results unchanged unless intended (trade counts stable)
