# Scenario 2 Summary
- **Shiftable Set**: MAIN
- **Shift Mode**: SAME_DAY (Comfort Hours: 06:00 to 23:00)

## Verification Checks
- `PASS` S1-A: Time-series integrity
- `PASS` S1-C: SOC Feasibility
- `PASS` S2-1: Comfort Bounds Compliance (Violations: 0)
- `PASS` S2-2: EV Window Compliance (Out-of-bound kWh: 0.00)
- `PASS` S2-3: Grid Cap Enforced (<= 50kW)
- `PASS` S2-4: Appliance Energy Conservation (diff=0.0000 kWh)
- `PASS` S2-5: Power Balance Week Check (max residual=0.000000 kW)

## Cost Summary
| Scenario | Annual Electricity Cost | Grid Import [kWh] |
|----------|------------------------|-------------------|
| Baseline (S0) | €10,589 | 87,886 |
| PV Only (S1a) | €7,229 | 61,018 |
| PV + BESS (S1b) | €6,574 | 56,334 |
| PV + BESS + Flex (S2) | €6,040 | 53,475 |
| **ΔFlex (S1b → S2 savings)** | **€534** | **2,859** |

## Appliance Energy Conservation (Service Conservation)
| Archetype | Baseline Appliance [kWh] | Scheduled Appliance [kWh] | Δ [kWh] | Status |
|-----------|--------------------------|---------------------------|----------|--------|
| CoupleWorking | 2888.2 | 2888.2 | 0.000 | PASS |
| Family1Child | 2021.9 | 2021.9 | 0.000 | PASS |
| Family3Children | 3055.1 | 3055.1 | 0.000 | PASS |
| RetiredCouple | 1492.9 | 1492.9 | 0.000 | PASS |
| **TOTAL** | **9458.0** | **9458.0** | **0.000** | **PASS** |

## Constraint Compliance
- Comfort violations: **0** (all tasks within 06:00–23:00)
- EV out-of-window: **0**
- Power balance max residual: **0.000000 kW**

## Note on Load Shifting
Flex shifts appliance cycles toward PV-rich hours (midday) and EV charging toward
off-peak hours where allowed by presence windows. Given SAME_DAY shifting with comfort
window 06:00–23:00, weekday F3 (23:00–07:00) is mostly unavailable for noisy appliances.

Full KPI object dumped to `scenario2_kpis_building.json`.
