# Scenario 3 – Thermal Retrofit Comparison Report

## 1. Objective
Compare the **annualized total cost** of supplying the same hourly thermal demand
(space heating + domestic hot water) with:
1. **Natural gas condensing boiler**
2. **Air-source heat pump (ASHP)**

## 2. Modeling Assumptions

| Parameter | Gas Boiler | ASHP |
|-----------|-----------|------|
| CAPEX | 400.0 €/kW_th | 1500.0 €/kW_th |
| Lifetime | 20 years | 19 years |
| Discount rate | 6% | 6% |
| CRF | 0.08718 | 0.08962 |
| O&M | 2% of CAPEX/year | 1% of CAPEX/year |
| Efficiency / COP | η = 0.9 | COP = 3.25 (fixed) |
| Energy source | Natural gas (ARERA 2025 prices) | Electricity (ARERA 2025 prices) |

- **COP mode**: Fixed fallback (3.25). No dynamic COP model was found in the repository.
- **Sizing**: Design capacity = peak hourly thermal demand (kW_th).
- **Gas price conversion**: PCS = 10.6944 kWh/Smc (ARERA standard).
- **Archetype weighting**: 1 dwelling per archetype (no multiplicity applied).
- **DHW resolution note**: The RetiredCouple DHW file was provided at minute resolution
  and was aggregated to hourly resolution by summation before alignment with the
  Scenario 3 hourly index.

## 3. Results by Archetype

| Archetype | Q_SH [kWh] | Q_DHW [kWh] | Q_total [kWh] | Peak [kW] | Boiler Total [€] | ASHP Total [€] | Selected | Savings [€] | Savings [%] |
|-----------|-----------|------------|--------------|----------|-----------------|---------------|----------|------------|------------|
| CoupleWorking | 4,300 | 3,915 | 8,216 | 9.71 | €770 | €1,751 | Gas Boiler | €981 | 56.0% |
| Family1Child | 5,376 | 6,259 | 11,634 | 14.37 | €1,117 | €2,568 | Gas Boiler | €1,451 | 56.5% |
| Family3Children | 6,182 | 8,918 | 15,100 | 18.15 | €1,429 | €3,261 | Gas Boiler | €1,831 | 56.2% |
| RetiredCouple | 4,838 | 3,759 | 8,597 | 10.58 | €824 | €1,895 | Gas Boiler | €1,070 | 56.5% |

## 4. Building Aggregate

| Metric | Value |
|--------|-------|
| Total thermal demand | 43,546 kWh_th |
| Peak thermal load | 34.85 kW_th |
| Boiler total annual cost | €4,141 |
| ASHP total annual cost | €9,475 |
| **Selected technology** | **Gas Boiler** |
| **Annual savings** | **€5,334** (56.3%) |

## 5. Conclusion

The **gas boiler** is the more cost-effective technology for this building, saving **€5,334/year** (56.3%) compared to an ASHP. The boiler's lower CAPEX (€21,125 vs €79,219) outweighs the ASHP's operational efficiency advantage.

## 6. Output Files

- `scenario3_archetype_summary.csv` – Per-archetype cost comparison
- `scenario3_building_summary.csv` – Building-level aggregate
- `scenario3_hourly_*.csv` – Hourly demand and energy inputs per archetype
- `scenario3_kpis.json` – All KPIs in machine-readable format
- `plot1_cost_comparison.png` – Annual cost bar chart
- `plot2_energy_input_comparison.png` – Energy input bar chart
- `plot3_cost_breakdown.png` – Stacked cost breakdown
