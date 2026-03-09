
# Scenario 1 Run Summary
**Solver Status:** ok (optimal)
**Objective Value:** €6573.69

## Verification Tests
- `PASS` A: Time-series integrity
- `PASS` B: PV sanity
- `PASS` C: Feasibility + boundaries
- `PASS` D: Balance residual
- `PASS` E: No export constraint
- `PASS` F: Cost Monotonicity (C_S1 <= C_PV <= C_0)
- `PASS` H: Ledger Check (Sum of parts == whole)

## Results Summary
| Metric | Baseline (S0) | PV-only (S1) | PV+BESS (S1) |
|---|---|---|---|
| Grid Import (kWh) | 87886 | 61018 | 56334 |
| Grid Cost (€)     | 10589 | 7229 | 6574 |
| PV Used (kWh)     | 0 | 26868 | 32459 |
| PV Curtailed (kWh)| 0 | 16042 | 10451 |
| Battery Cycles    | 0 | 0 | 420.1 |
| Total Proc. Cost (€)| 15828 | 12469 | 11813 |

Outputs cleanly saved to: `Scenario1/results/scenario1`.

---
## Fix Report
**Issue A - Grid charge bleeding:**
Before Fix: Counted 119 hours with grid import == 0 and grid charge > 0. Total false grid charging energy logged: 518 kWh.
After Fix: Added constraint `P_ch_grid <= G_imp` to ensure grid_charge variable strictly represents literal grid consumption boundaries. Current count of instances with grid import == 0 alongside charge_grid > 0 is exactly: 0 instances at 0.0 total kWh.

**Issue B - Archetype Allocation Budget:**
Before Fix: Allocation mixed `C_0` per apartment costs against `C_1` Total Building Cost leading to non-logical "Negative Savings".
After Fix: De-multiplexed explicitly into `_total` properties ensuring scale uniformity (`C_0` total mapped to `C_1` total), alongside optional `_per_apt` derivations. 

Both fixes applied properly leaving verification tests intact.
