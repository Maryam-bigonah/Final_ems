#!/usr/bin/env python3
"""
run_scenario3.py – Scenario 3: Thermal Retrofit Comparison (Gas Boiler vs ASHP)

Compares the annualized total cost of supplying space heating (SH) and domestic
hot water (DHW) using either a condensing gas boiler or an air-source heat pump.

No PV, no BESS, no appliance scheduling.  Pure deterministic cost comparison.
"""

import sys
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project wiring ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE0_DIR   = PROJECT_ROOT / "Scenario 0"
DATASET_DIR  = PROJECT_ROOT / "Dataset"
sys.path.insert(0, str(SCENE0_DIR))

from src.loader import load_electricity_price, load_gas_price

ARCHETYPES = ["CoupleWorking", "Family1Child", "Family3Children", "RetiredCouple"]

# Gas PCS (Primary Calorific Standard) for Italy – kWh per Smc
# ARERA standard: PCS = 0.038520 GJ/Smc → 10.7 kWh/Smc (approx)
KWH_PER_SMC = 10.6944  # 0.038520 GJ/Smc * 1000/3.6

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config():
    """Load Scenario 3 config from YAML."""
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return raw["SCENARIO_3"]


def crf(r: float, n: int) -> float:
    """Capital Recovery Factor (annuity factor)."""
    return r * (1 + r)**n / ((1 + r)**n - 1)


def load_sh(arch: str, cfg: dict) -> pd.Series:
    """Load hourly SH thermal demand for one archetype (kWh_th)."""
    rel = cfg["inputs"]["space_heating_files"][arch]
    path = DATASET_DIR / rel
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df["Q_SH_kWh_th"]


def load_dhw(arch: str, cfg: dict, ref_idx: pd.DatetimeIndex) -> pd.Series:
    """Load hourly DHW thermal demand for one archetype (kWh_th).

    The original files are from 2016 and semicolon-delimited.
    Auto-detects sub-hourly resolution and aggregates to hourly by SUM.
    Year-shifts to 2025 and aligns to ref_idx.
    """
    fname = cfg["inputs"]["dhw_files"][arch]
    col   = cfg["inputs"]["dhw_thermal_column"]     # "Q_th_kWh"
    path  = DATASET_DIR / fname

    df = pd.read_csv(path, sep=";", low_memory=False)

    # Identify timestamp column
    ts_col = "Time" if "Time" in df.columns else df.columns[1]
    df["_ts"] = pd.to_datetime(df[ts_col], dayfirst=False)
    df = df.set_index("_ts").sort_index()

    # Detect timestep resolution
    median_dt = pd.Series(df.index).diff().dropna().median()
    is_subhourly = median_dt <= pd.Timedelta(minutes=2)
    if is_subhourly:
        print(f"    ⚠ {fname}: detected sub-hourly resolution "
              f"(median Δt = {median_dt}). Aggregating to hourly by SUM.")
        # Sum energy values within each clock-hour (Q_th_kWh is energy per timestep)
        df = df[[col]].resample("h").sum()

    # Year-shift: replace year with 2025
    new_idx = df.index.map(
        lambda ts: ts.replace(year=2025)
        if not (ts.month == 2 and ts.day == 29)
        else ts.replace(year=2025, month=2, day=28)
    )
    df.index = new_idx

    # Drop duplicates from leap-year folding
    df = df[~df.index.duplicated(keep="first")]

    # Reindex to reference hourly index, fill gaps with 0
    series = df[col].reindex(ref_idx).fillna(0.0)
    series = series.clip(lower=0.0)           # safety: no negative DHW
    series.name = "Q_DHW_kWh_th"
    return series


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SCENARIO 3 – Thermal Retrofit: Gas Boiler vs ASHP")
    print("=" * 60)

    cfg = load_config()
    out_dir = Path(__file__).resolve().parent / cfg["outputs"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load tariffs ─────────────────────────────────────────────────
    el_prices = load_electricity_price(DATASET_DIR / cfg["inputs"]["electricity_price_file"])
    ref_idx   = el_prices.index                      # 8760 DateTimeIndex

    gas_price_smc = load_gas_price(DATASET_DIR / cfg["inputs"]["gas_price_file"])
    # Convert gas price from €/Smc → €/kWh_gas
    gas_price_kwh = gas_price_smc / KWH_PER_SMC
    gas_price_kwh.name = "gas_price_eur_per_kwh"

    # ── 2. Technology parameters from config.yaml ───────────────────────
    boiler = cfg["gas_boiler"]
    ashp   = cfg["ashp"]
    r      = cfg["annualization"]["discount_rate"]

    eta_boiler   = boiler["efficiency"]
    cop_fixed    = ashp["fixed_cop_fallback"]

    capex_boiler_per_kw = boiler["capex_eur_per_kw_th"]
    capex_ashp_per_kw   = ashp["capex_eur_per_kw_th"]

    om_frac_boiler = boiler["om_fraction_of_capex_per_year"]
    om_frac_ashp   = ashp["om_fraction_of_capex_per_year"]

    n_boiler = boiler["lifetime_years"]
    n_ashp   = ashp["lifetime_years"]

    crf_boiler = crf(r, n_boiler)
    crf_ashp   = crf(r, n_ashp)

    print(f"\n  Boiler: η={eta_boiler}, CAPEX={capex_boiler_per_kw} €/kW, "
          f"lifetime={n_boiler}y, O&M={om_frac_boiler*100:.0f}%, CRF={crf_boiler:.5f}")
    print(f"  ASHP:   COP={cop_fixed} (fixed fallback), CAPEX={capex_ashp_per_kw} €/kW, "
          f"lifetime={n_ashp}y, O&M={om_frac_ashp*100:.0f}%, CRF={crf_ashp:.5f}")

    # ── 3. Load demands & compute per-archetype ─────────────────────────
    arch_results = []          # summary rows
    hourly_dfs   = {}          # per-archetype hourly DataFrames

    for arch in ARCHETYPES:
        print(f"\n── {arch} ──")

        q_sh  = load_sh(arch, cfg)
        q_dhw = load_dhw(arch, cfg, ref_idx)

        # Align SH to ref_idx (should already be aligned, but be safe)
        q_sh = q_sh.reindex(ref_idx).fillna(0.0).clip(lower=0.0)

        q_total = q_sh + q_dhw

        # ── Validation ──
        assert len(q_sh) == 8760, f"{arch} SH has {len(q_sh)} rows, expected 8760"
        assert len(q_dhw) == 8760, f"{arch} DHW has {len(q_dhw)} rows, expected 8760"
        assert (q_sh >= 0).all(), f"{arch} SH has negative values"
        assert (q_dhw >= 0).all(), f"{arch} DHW has negative values"

        ann_sh  = float(q_sh.sum())
        ann_dhw = float(q_dhw.sum())
        ann_tot = float(q_total.sum())
        peak    = float(q_total.max())

        print(f"  SH annual  = {ann_sh:,.1f} kWh_th")
        print(f"  DHW annual = {ann_dhw:,.1f} kWh_th")
        print(f"  Total      = {ann_tot:,.1f} kWh_th")
        print(f"  Peak load  = {peak:.2f} kW_th")

        # ── Sizing (same peak for both technologies) ──
        p_design = peak

        # ── Boiler ──
        capex_boiler_total = capex_boiler_per_kw * p_design
        ann_capex_boiler   = capex_boiler_total * crf_boiler
        ann_om_boiler      = om_frac_boiler * capex_boiler_total

        gas_input_kwh = q_total / eta_boiler                     # hourly kWh_gas
        gas_input_smc = gas_input_kwh / KWH_PER_SMC              # hourly Smc
        boiler_hourly_cost = gas_input_smc * gas_price_smc        # €/h
        boiler_opex = float(boiler_hourly_cost.sum())
        boiler_total = ann_capex_boiler + ann_om_boiler + boiler_opex

        # ── ASHP ──
        capex_ashp_total = capex_ashp_per_kw * p_design
        ann_capex_ashp   = capex_ashp_total * crf_ashp
        ann_om_ashp      = om_frac_ashp * capex_ashp_total

        hp_el_kwh = q_total / cop_fixed                           # hourly kWh_el
        hp_hourly_cost = hp_el_kwh * el_prices["price_eur_per_kwh"]
        hp_opex = float(hp_hourly_cost.sum())
        hp_total = ann_capex_ashp + ann_om_ashp + hp_opex

        # ── Selection ──
        if hp_total < boiler_total:
            selected = "ASHP"
            savings  = boiler_total - hp_total
        else:
            selected = "Gas Boiler"
            savings  = hp_total - boiler_total

        savings_pct = savings / max(boiler_total, hp_total) * 100

        print(f"  Boiler annual cost = €{boiler_total:,.0f}  "
              f"(OPEX €{boiler_opex:,.0f} + CAPEX_ann €{ann_capex_boiler:,.0f} + O&M €{ann_om_boiler:,.0f})")
        print(f"  ASHP   annual cost = €{hp_total:,.0f}  "
              f"(OPEX €{hp_opex:,.0f} + CAPEX_ann €{ann_capex_ashp:,.0f} + O&M €{ann_om_ashp:,.0f})")
        print(f"  → Selected: {selected}  (saves €{savings:,.0f}, {savings_pct:.1f}%)")

        arch_results.append({
            "archetype": arch,
            "annual_Q_SH_kWh_th": round(ann_sh, 2),
            "annual_Q_DHW_kWh_th": round(ann_dhw, 2),
            "annual_Q_total_kWh_th": round(ann_tot, 2),
            "peak_Q_total_kW_th": round(peak, 4),
            "boiler_gas_input_kWh": round(float(gas_input_kwh.sum()), 2),
            "boiler_opex_eur": round(boiler_opex, 2),
            "boiler_capex_total_eur": round(capex_boiler_total, 2),
            "boiler_capex_annualized_eur": round(ann_capex_boiler, 2),
            "boiler_om_annual_eur": round(ann_om_boiler, 2),
            "boiler_total_annual_cost_eur": round(boiler_total, 2),
            "hp_electricity_input_kWh": round(float(hp_el_kwh.sum()), 2),
            "hp_opex_eur": round(hp_opex, 2),
            "hp_capex_total_eur": round(capex_ashp_total, 2),
            "hp_capex_annualized_eur": round(ann_capex_ashp, 2),
            "hp_om_annual_eur": round(ann_om_ashp, 2),
            "hp_total_annual_cost_eur": round(hp_total, 2),
            "selected_technology": selected,
            "annual_savings_eur": round(savings, 2),
            "annual_savings_percent": round(savings_pct, 2),
        })

        # ── Hourly output ──
        hourly_dfs[arch] = pd.DataFrame({
            "timestamp": ref_idx,
            "Q_SH_kWh_th": q_sh.values,
            "Q_DHW_kWh_th": q_dhw.values,
            "Q_total_kWh_th": q_total.values,
            "boiler_gas_input_kWh": gas_input_kwh.values,
            "hp_electricity_input_kWh": hp_el_kwh.values,
            "electricity_price_eur_per_kwh": el_prices["price_eur_per_kwh"].values,
            "gas_price_eur_per_kwh": gas_price_kwh.values,
        })

    # ── 4. Building aggregate ───────────────────────────────────────────
    print("\n── BUILDING AGGREGATE ──")
    bld = {k: 0.0 for k in arch_results[0] if k not in ("archetype", "selected_technology")}
    for row in arch_results:
        for k in bld:
            if k in ("annual_savings_percent", "peak_Q_total_kW_th"):
                continue
            bld[k] += row[k]

    # Recompute peak as max of total building hourly demand
    bld_hourly_total = sum(hourly_dfs[a]["Q_total_kWh_th"] for a in ARCHETYPES)
    bld["peak_Q_total_kW_th"] = round(float(bld_hourly_total.max()), 4)

    if bld["hp_total_annual_cost_eur"] < bld["boiler_total_annual_cost_eur"]:
        bld_selected = "ASHP"
        bld_savings  = bld["boiler_total_annual_cost_eur"] - bld["hp_total_annual_cost_eur"]
    else:
        bld_selected = "Gas Boiler"
        bld_savings  = bld["hp_total_annual_cost_eur"] - bld["boiler_total_annual_cost_eur"]

    bld["annual_savings_eur"] = round(bld_savings, 2)
    bld["annual_savings_percent"] = round(
        bld_savings / max(bld["boiler_total_annual_cost_eur"], bld["hp_total_annual_cost_eur"]) * 100, 2
    )
    bld["archetype"] = "BUILDING"
    bld["selected_technology"] = bld_selected

    print(f"  Total thermal demand = {bld['annual_Q_total_kWh_th']:,.0f} kWh_th")
    print(f"  Boiler annual cost   = €{bld['boiler_total_annual_cost_eur']:,.0f}")
    print(f"  ASHP   annual cost   = €{bld['hp_total_annual_cost_eur']:,.0f}")
    print(f"  → Selected: {bld_selected}  (saves €{bld_savings:,.0f})")

    # ── 5. Save outputs ─────────────────────────────────────────────────
    # 5a. Archetype summary CSV
    summary_df = pd.DataFrame(arch_results)
    summary_df.to_csv(out_dir / "scenario3_archetype_summary.csv", index=False)

    # 5b. Building summary CSV
    bld_df = pd.DataFrame([bld])
    bld_df.to_csv(out_dir / "scenario3_building_summary.csv", index=False)

    # 5c. Hourly files per archetype
    for arch in ARCHETYPES:
        hourly_dfs[arch].to_csv(
            out_dir / f"scenario3_hourly_{arch}.csv", index=False
        )

    # 5d. JSON KPIs (combined)
    kpi_json = {
        "archetypes": arch_results,
        "building": bld,
        "parameters": {
            "boiler_efficiency": eta_boiler,
            "ashp_cop_mode": "fixed_fallback",
            "ashp_cop_value": cop_fixed,
            "discount_rate": r,
            "boiler_lifetime": n_boiler,
            "ashp_lifetime": n_ashp,
            "crf_boiler": round(crf_boiler, 6),
            "crf_ashp": round(crf_ashp, 6),
            "kwh_per_smc": KWH_PER_SMC,
        },
    }
    with open(out_dir / "scenario3_kpis.json", "w") as f:
        json.dump(kpi_json, f, indent=4)

    # ── 6. Plots ────────────────────────────────────────────────────────
    _generate_plots(arch_results, bld, out_dir)

    # ── 7. Markdown Report ──────────────────────────────────────────────
    _write_report(arch_results, bld, cfg, crf_boiler, crf_ashp, cop_fixed, out_dir)

    print(f"\n✓ All outputs saved to: {out_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


# ── Plotting ────────────────────────────────────────────────────────────────

def _generate_plots(arch_results, bld, out_dir):
    """Create comparison bar charts."""
    labels = [r["archetype"] for r in arch_results]

    # Plot 1 – Annual total cost comparison by archetype
    boiler_costs = [r["boiler_total_annual_cost_eur"] for r in arch_results]
    hp_costs     = [r["hp_total_annual_cost_eur"] for r in arch_results]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, boiler_costs, w, label="Gas Boiler", color="#e63946", edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x + w/2, hp_costs,     w, label="ASHP",       color="#2a9d8f", edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Annual Total Cost [€/year]")
    ax.set_title("Scenario 3 – Gas Boiler vs ASHP: Annual Total Cost by Archetype")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"€{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"€{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "plot1_cost_comparison.png", dpi=150)
    fig.savefig(out_dir / "plot1_cost_comparison.pdf")
    plt.close(fig)

    # Plot 2 – Annual energy input comparison
    boiler_gas  = [r["boiler_gas_input_kWh"] for r in arch_results]
    hp_el       = [r["hp_electricity_input_kWh"] for r in arch_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, boiler_gas, w, label="Gas Input (Boiler)", color="#f4a261", edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x + w/2, hp_el,      w, label="Electricity Input (ASHP)", color="#457b9d", edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Annual Energy Input [kWh]")
    ax.set_title("Scenario 3 – Energy Input: Gas (Boiler) vs Electricity (ASHP) by Archetype")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "plot2_energy_input_comparison.png", dpi=150)
    fig.savefig(out_dir / "plot2_energy_input_comparison.pdf")
    plt.close(fig)

    # Plot 3 – Cost breakdown stacked bar (CAPEX_ann + O&M + OPEX)
    fig, ax = plt.subplots(figsize=(10, 6))
    boiler_opex_vals  = [r["boiler_opex_eur"] for r in arch_results]
    boiler_capex_vals = [r["boiler_capex_annualized_eur"] for r in arch_results]
    boiler_om_vals    = [r["boiler_om_annual_eur"] for r in arch_results]
    hp_opex_vals      = [r["hp_opex_eur"] for r in arch_results]
    hp_capex_vals     = [r["hp_capex_annualized_eur"] for r in arch_results]
    hp_om_vals        = [r["hp_om_annual_eur"] for r in arch_results]

    x2 = np.arange(len(labels) * 2)
    bar_labels = []
    opex_all, capex_all, om_all, colors_all = [], [], [], []
    for i, arch in enumerate(labels):
        bar_labels.extend([f"{arch}\nBoiler", f"{arch}\nASHP"])
        opex_all.extend([boiler_opex_vals[i], hp_opex_vals[i]])
        capex_all.extend([boiler_capex_vals[i], hp_capex_vals[i]])
        om_all.extend([boiler_om_vals[i], hp_om_vals[i]])
        colors_all.extend(["#e63946", "#2a9d8f"])

    ax.bar(x2, opex_all, 0.6, label="Energy OPEX", color=[c + "99" for c in colors_all], edgecolor="k", linewidth=0.3)
    ax.bar(x2, capex_all, 0.6, bottom=opex_all, label="Annualized CAPEX", color=colors_all, edgecolor="k", linewidth=0.3)
    bottom2 = [a + b for a, b in zip(opex_all, capex_all)]
    ax.bar(x2, om_all, 0.6, bottom=bottom2, label="Annual O&M", color=[c + "55" for c in colors_all], edgecolor="k", linewidth=0.3)

    ax.set_ylabel("€/year")
    ax.set_title("Scenario 3 – Cost Breakdown by Component")
    ax.set_xticks(x2)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "plot3_cost_breakdown.png", dpi=150)
    fig.savefig(out_dir / "plot3_cost_breakdown.pdf")
    plt.close(fig)

    print("  ✓ Plots saved")


# ── Markdown report ─────────────────────────────────────────────────────────

def _write_report(arch_results, bld, cfg, crf_b, crf_a, cop, out_dir):
    boiler = cfg["gas_boiler"]
    ashp   = cfg["ashp"]
    r      = cfg["annualization"]["discount_rate"]

    md = f"""# Scenario 3 – Thermal Retrofit Comparison Report

## 1. Objective
Compare the **annualized total cost** of supplying the same hourly thermal demand
(space heating + domestic hot water) with:
1. **Natural gas condensing boiler**
2. **Air-source heat pump (ASHP)**

## 2. Modeling Assumptions

| Parameter | Gas Boiler | ASHP |
|-----------|-----------|------|
| CAPEX | {boiler['capex_eur_per_kw_th']} €/kW_th | {ashp['capex_eur_per_kw_th']} €/kW_th |
| Lifetime | {boiler['lifetime_years']} years | {ashp['lifetime_years']} years |
| Discount rate | {r*100:.0f}% | {r*100:.0f}% |
| CRF | {crf_b:.5f} | {crf_a:.5f} |
| O&M | {boiler['om_fraction_of_capex_per_year']*100:.0f}% of CAPEX/year | {ashp['om_fraction_of_capex_per_year']*100:.0f}% of CAPEX/year |
| Efficiency / COP | η = {boiler['efficiency']} | COP = {cop} (fixed) |
| Energy source | Natural gas (ARERA 2025 prices) | Electricity (ARERA 2025 prices) |

- **COP mode**: Fixed fallback ({cop}). No dynamic COP model was found in the repository.
- **Sizing**: Design capacity = peak hourly thermal demand (kW_th).
- **Gas price conversion**: PCS = {KWH_PER_SMC:.4f} kWh/Smc (ARERA standard).
- **Archetype weighting**: 1 dwelling per archetype (no multiplicity applied).
- **DHW resolution note**: The RetiredCouple DHW file was provided at minute resolution
  and was aggregated to hourly resolution by summation before alignment with the
  Scenario 3 hourly index.

## 3. Results by Archetype

| Archetype | Q_SH [kWh] | Q_DHW [kWh] | Q_total [kWh] | Peak [kW] | Boiler Total [€] | ASHP Total [€] | Selected | Savings [€] | Savings [%] |
|-----------|-----------|------------|--------------|----------|-----------------|---------------|----------|------------|------------|
"""
    for r_row in arch_results:
        md += (
            f"| {r_row['archetype']} "
            f"| {r_row['annual_Q_SH_kWh_th']:,.0f} "
            f"| {r_row['annual_Q_DHW_kWh_th']:,.0f} "
            f"| {r_row['annual_Q_total_kWh_th']:,.0f} "
            f"| {r_row['peak_Q_total_kW_th']:.2f} "
            f"| €{r_row['boiler_total_annual_cost_eur']:,.0f} "
            f"| €{r_row['hp_total_annual_cost_eur']:,.0f} "
            f"| {r_row['selected_technology']} "
            f"| €{r_row['annual_savings_eur']:,.0f} "
            f"| {r_row['annual_savings_percent']:.1f}% |\n"
        )

    md += f"""
## 4. Building Aggregate

| Metric | Value |
|--------|-------|
| Total thermal demand | {bld['annual_Q_total_kWh_th']:,.0f} kWh_th |
| Peak thermal load | {bld['peak_Q_total_kW_th']:.2f} kW_th |
| Boiler total annual cost | €{bld['boiler_total_annual_cost_eur']:,.0f} |
| ASHP total annual cost | €{bld['hp_total_annual_cost_eur']:,.0f} |
| **Selected technology** | **{bld['selected_technology']}** |
| **Annual savings** | **€{bld['annual_savings_eur']:,.0f}** ({bld['annual_savings_percent']:.1f}%) |

## 5. Conclusion

"""
    if bld["selected_technology"] == "ASHP":
        md += (
            f"The **air-source heat pump** is the more cost-effective technology for this building, "
            f"saving **€{bld['annual_savings_eur']:,.0f}/year** ({bld['annual_savings_percent']:.1f}%) "
            f"compared to a gas boiler. Despite higher upfront costs "
            f"(€{bld['hp_capex_total_eur']:,.0f} vs €{bld['boiler_capex_total_eur']:,.0f}), "
            f"the ASHP's superior performance (COP = {cop}) offsets the CAPEX difference through "
            f"lower annual electricity procurement costs.\n"
        )
    else:
        md += (
            f"The **gas boiler** is the more cost-effective technology for this building, "
            f"saving **€{bld['annual_savings_eur']:,.0f}/year** ({bld['annual_savings_percent']:.1f}%) "
            f"compared to an ASHP. The boiler's lower CAPEX "
            f"(€{bld['boiler_capex_total_eur']:,.0f} vs €{bld['hp_capex_total_eur']:,.0f}) "
            f"outweighs the ASHP's operational efficiency advantage.\n"
        )

    md += f"""
## 6. Output Files

- `scenario3_archetype_summary.csv` – Per-archetype cost comparison
- `scenario3_building_summary.csv` – Building-level aggregate
- `scenario3_hourly_*.csv` – Hourly demand and energy inputs per archetype
- `scenario3_kpis.json` – All KPIs in machine-readable format
- `plot1_cost_comparison.png` – Annual cost bar chart
- `plot2_energy_input_comparison.png` – Energy input bar chart
- `plot3_cost_breakdown.png` – Stacked cost breakdown
"""

    report_path = out_dir / "report_scenario3.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"  ✓ Report written: {report_path.name}")


if __name__ == "__main__":
    main()
