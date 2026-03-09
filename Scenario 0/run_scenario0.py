#!/usr/bin/env python3
"""
run_scenario0.py – Scenario 0 (Baseline) entry point.

Usage:
    python run_scenario0.py

Reads config.yaml, loads all data, computes KPIs, generates plots,
saves outputs, and runs verification checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import (
    load_electricity,
    load_gas,
    load_ev,
    load_electricity_price,
    load_gas_price,
)
from src.demand import build_electricity_demand, build_gas_demand
from src.tariff import apply_electricity_tariff, apply_gas_tariff, tou_split
from src.kpi import (
    compute_archetype_kpis,
    compute_building_kpis,
    compute_monthly_breakdown,
    compute_seasonal_breakdown,
)
from src.plots import generate_all_plots, generate_comparison_plots
from src.output import (
    save_csv_and_collect,
    save_excel_workbook,
    save_timeseries,
    save_verification_report,
)
from src.verification import run_all_checks


def load_config(path: Path) -> dict:
    """Load and validate the YAML configuration."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = raw["SCENARIO_0"]
    return cfg


def main() -> int:
    print("=" * 60)
    print("SCENARIO 0 – BASELINE COMPUTATION")
    print("=" * 60)

    # ── Load config ──────────────────────────────────────────────
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: config.yaml not found at {config_path}")
        return 1
    cfg = load_config(config_path)

    year = cfg["year"]
    Nk = cfg["building_composition"]
    include_ev = cfg["include_ev"]
    dataset_dir = (PROJECT_ROOT / cfg["dataset_dir"]).resolve()
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    archetype_keys = list(cfg["electricity_files"].keys())
    ev_col_map = cfg.get("ev_column_mapping", {})
    season_map = cfg["seasons"]
    rep_weeks = cfg["representative_weeks"]

    print(f"\nYear:            {year}")
    print(f"Archetypes:      {archetype_keys}")
    print(f"Building Nk:     {Nk}")
    print(f"Include EV:      {include_ev}")
    print(f"Dataset dir:     {dataset_dir}")
    print(f"Output dir:      {output_dir}")
    print()

    # ── Step A: Load all data ────────────────────────────────────
    print("─" * 40)
    print("STEP A: Loading data")
    print("─" * 40)

    # Electricity base
    base_el = {}
    for arch in archetype_keys:
        fpath = dataset_dir / cfg["electricity_files"][arch]
        base_el[arch] = load_electricity(fpath, arch, target_year=year)

    # Gas
    gas_data = {}
    for arch in archetype_keys:
        fpath = dataset_dir / cfg["gas_files"][arch]
        gas_data[arch] = load_gas(fpath, arch, target_year=year)

    # EV
    ev_path = dataset_dir / cfg["ev_file"]
    ev_df = load_ev(ev_path, ev_col_map, archetype_keys)

    # Prices
    el_prices = load_electricity_price(dataset_dir / cfg["price_files"]["electricity"])
    gas_prices = load_gas_price(dataset_dir / cfg["price_files"]["gas"], target_year=year)

    # ── Align all indices to the electricity price index ─────────
    ref_idx = el_prices.index
    for arch in archetype_keys:
        base_el[arch] = base_el[arch].reindex(ref_idx)
        gas_data[arch] = gas_data[arch].reindex(ref_idx)
    ev_df = ev_df.reindex(ref_idx).fillna(0.0)
    gas_prices = gas_prices.reindex(ref_idx)

    # Post-alignment NaN check
    for arch in archetype_keys:
        if base_el[arch].isna().any():
            raise ValueError(f"NaN in {arch} electricity after alignment.")
        if gas_data[arch].isna().any().any():
            raise ValueError(f"NaN in {arch} gas after alignment.")
    if el_prices.isna().any().any():
        raise ValueError("NaN in electricity prices after alignment.")
    if gas_prices.isna().any():
        raise ValueError("NaN in gas prices after alignment.")

    print("\n✓ All data loaded and aligned.\n")

    # ── Step B & C: Build demand ─────────────────────────────────
    print("─" * 40)
    print("STEP B & C: Building demand")
    print("─" * 40)

    e_el = {}
    for arch in archetype_keys:
        e_el[arch] = build_electricity_demand(
            base_el[arch],
            ev_df[arch],
            include_ev,
            arch,
        )

    gas_demand = {}
    for arch in archetype_keys:
        gas_demand[arch] = build_gas_demand(gas_data[arch], arch)

    print()

    # ── Step D & E: Tariffs, KPIs, TOU ───────────────────────────
    print("─" * 40)
    print("STEP D & E: Computing KPIs and TOU split")
    print("─" * 40)

    arch_kpi_list = []
    arch_results = {}  # for verification and plots

    for arch in archetype_keys:
        kpi, c_el, c_gas, tou = compute_archetype_kpis(
            arch, e_el[arch], gas_demand[arch], el_prices, gas_prices,
        )
        arch_kpi_list.append(kpi)
        arch_results[arch] = {
            "e_el": e_el[arch],
            "ev": ev_df[arch],
            "gas_df": gas_demand[arch],
            "c_el": c_el,
            "c_gas": c_gas,
            "tou": tou,
            "kpi": kpi,
        }

    # Building KPIs
    bld_kpi = compute_building_kpis(arch_kpi_list, Nk)

    print()

    # ── Step F: Building time series ─────────────────────────────
    print("─" * 40)
    print("STEP F: Building aggregate time series")
    print("─" * 40)

    bld_e_el = pd.Series(0.0, index=ref_idx, name="E_el_bld")
    bld_gas_kWh = pd.Series(0.0, index=ref_idx, name="G_bld_kWh")
    bld_gas_Smc = pd.Series(0.0, index=ref_idx, name="G_bld_Smc")
    bld_c_el = pd.Series(0.0, index=ref_idx, name="C_el_bld")
    bld_c_gas = pd.Series(0.0, index=ref_idx, name="C_gas_bld")

    for arch in archetype_keys:
        n = Nk[arch]
        bld_e_el += n * e_el[arch]
        bld_gas_kWh += n * gas_demand[arch]["gas_kWh"]
        bld_gas_Smc += n * gas_demand[arch]["gas_Smc"]
        bld_c_el += n * arch_results[arch]["c_el"]
        bld_c_gas += n * arch_results[arch]["c_gas"]

    bld_gas_df = pd.DataFrame(
        {"gas_kWh": bld_gas_kWh, "gas_Smc": bld_gas_Smc},
        index=ref_idx,
    )

    # Building TOU
    bld_tou = tou_split(bld_e_el, bld_c_el, el_prices["arera_band"])

    building_ts = pd.DataFrame(
        {
            "E_el_bld": bld_e_el,
            "G_kWh_bld": bld_gas_kWh,
            "G_Smc_bld": bld_gas_Smc,
            "C_el_bld": bld_c_el,
            "C_gas_bld": bld_c_gas,
            "p_el": el_prices["price_eur_per_kwh"],
            "p_gas_smc": gas_prices,
            "arera_band": el_prices["arera_band"],
        },
        index=ref_idx,
    )

    print(f"  ✓ Building aggregate: E_el={bld_e_el.sum():.0f} kWh, "
          f"G={bld_gas_kWh.sum():.0f} kWh, "
          f"C0={bld_c_el.sum() + bld_c_gas.sum():.2f} €")
    print()

    # ── Step G: Plots ────────────────────────────────────────────
    print("─" * 40)
    print("STEP G: Generating thesis-ready plots")
    print("─" * 40)

    for arch in archetype_keys:
        d = arch_results[arch]
        ev_series = d["ev"] if d["ev"].sum() > 0 else None
        generate_all_plots(
            entity_name=arch,
            e_el=d["e_el"],
            ev=ev_series,
            gas_df=d["gas_df"],
            c_el=d["c_el"],
            c_gas=d["c_gas"],
            tou=d["tou"],
            season_map=season_map,
            representative_weeks=rep_weeks,
            out_dir=output_dir,
        )

    # Building aggregate plots
    generate_all_plots(
        entity_name="Building",
        e_el=bld_e_el,
        ev=None,
        gas_df=bld_gas_df,
        c_el=bld_c_el,
        c_gas=bld_c_gas,
        tou=bld_tou,
        season_map=season_map,
        representative_weeks=rep_weeks,
        out_dir=output_dir,
    )

    # Cross-archetype comparison plots
    generate_comparison_plots(
        arch_data=arch_results,
        kpi_list=arch_kpi_list,
        representative_weeks=rep_weeks,
        out_dir=output_dir,
    )
    print()

    # ── Step H: Save outputs ─────────────────────────────────────
    print("─" * 40)
    print("STEP H: Saving outputs")
    print("─" * 40)

    sheets: dict[str, pd.DataFrame] = {}

    # KPI tables
    kpi_arch_df = pd.DataFrame(arch_kpi_list)
    save_csv_and_collect(kpi_arch_df, "kpi_archetype_s0", output_dir, sheets)

    kpi_bld_df = pd.DataFrame([bld_kpi])
    save_csv_and_collect(kpi_bld_df, "kpi_building_s0", output_dir, sheets)

    # Monthly breakdown
    monthly_frames = []
    for arch in archetype_keys:
        d = arch_results[arch]
        mb = compute_monthly_breakdown(
            arch, d["e_el"], d["c_el"], d["gas_df"], d["c_gas"],
        )
        monthly_frames.append(mb)

    # Building monthly
    bld_monthly = compute_monthly_breakdown(
        "BUILDING", bld_e_el, bld_c_el, bld_gas_df, bld_c_gas,
    )
    monthly_frames.append(bld_monthly)
    monthly_all = pd.concat(monthly_frames)
    save_csv_and_collect(monthly_all, "monthly_breakdown_s0", output_dir, sheets)

    # Seasonal breakdown
    seasonal_frames = []
    for arch in archetype_keys:
        d = arch_results[arch]
        sb = compute_seasonal_breakdown(
            arch, d["e_el"], d["c_el"], d["gas_df"], d["c_gas"], season_map,
        )
        seasonal_frames.append(sb)
    bld_seasonal = compute_seasonal_breakdown(
        "BUILDING", bld_e_el, bld_c_el, bld_gas_df, bld_c_gas, season_map,
    )
    seasonal_frames.append(bld_seasonal)
    seasonal_all = pd.concat(seasonal_frames)
    save_csv_and_collect(seasonal_all, "seasonal_breakdown_s0", output_dir, sheets)

    # TOU split table
    tou_frames = []
    for arch in archetype_keys:
        t = arch_results[arch]["tou"].copy()
        t.insert(0, "archetype", arch)
        tou_frames.append(t)
    bld_tou_df = bld_tou.copy()
    bld_tou_df.insert(0, "archetype", "BUILDING")
    tou_frames.append(bld_tou_df)
    tou_all = pd.concat(tou_frames)
    save_csv_and_collect(tou_all, "tou_split_s0", output_dir, sheets)

    # Excel workbook
    save_excel_workbook(sheets, output_dir)

    # Time series
    arch_ts_dict = {}
    for arch in archetype_keys:
        d = arch_results[arch]
        arch_ts_dict[arch] = pd.DataFrame(
            {
                "E_el": d["e_el"].values,
                "EV": d["ev"].values,
                "G_kWh": d["gas_df"]["gas_kWh"].values,
                "G_Smc": d["gas_df"]["gas_Smc"].values,
                "p_el": el_prices["price_eur_per_kwh"].values,
                "p_gas_smc": gas_prices.values,
                "arera_band": el_prices["arera_band"].values,
            },
            index=ref_idx,
        )
    save_timeseries(arch_ts_dict, building_ts, output_dir)

    # Per-archetype hourly total electricity demand CSVs
    ARCH_CSV_NAMES = {
        "CoupleWorking":   "sum_couple_electricity_demand",
        "Family1Child":    "sum_family1child_electricity_demand",
        "Family3Children": "sum_family3children_electricity_demand",
        "RetiredCouple":   "sum_retired_electricity_demand",
    }
    for arch in archetype_keys:
        d = arch_results[arch]
        demand_df = pd.DataFrame(
            {
                "timestamp": d["e_el"].index,
                "total_electricity_kWh": d["e_el"].values,
            }
        )
        csv_name = ARCH_CSV_NAMES.get(arch, f"sum_{arch}_electricity_demand")
        csv_path = output_dir / f"{csv_name}.csv"
        demand_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {csv_path.name}")

    print()

    # ── Step I: Verification ─────────────────────────────────────
    print("─" * 40)
    print("STEP I: Running verification checks")
    print("─" * 40)

    check_results = run_all_checks(
        arch_results, building_ts, el_prices, gas_prices, Nk, ev_col_map,
    )
    all_pass = save_verification_report(check_results, output_dir)

    # Print check summary
    n_pass = sum(1 for _, ok, _ in check_results if ok)
    n_fail = sum(1 for _, ok, _ in check_results if not ok)
    print(f"\n  {n_pass} PASSED, {n_fail} FAILED")

    if n_fail > 0:
        print("\n  ⚠ FAILED CHECKS:")
        for name, ok, detail in check_results:
            if not ok:
                print(f"    ✗ {name}: {detail}")

    # ── Console summary ──────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("SCENARIO 0 – KPI SUMMARY")
    print("=" * 60)
    print(f"\n{'Archetype':<20} {'E_el(kWh)':>12} {'C_el(€)':>10} "
          f"{'G(kWh)':>10} {'G(Smc)':>10} {'C_gas(€)':>10} {'C0(€)':>10}")
    print("-" * 92)
    for kpi in arch_kpi_list:
        print(
            f"{kpi['archetype']:<20} "
            f"{kpi['E_el_kWh']:>12,.1f} "
            f"{kpi['C_el_eur']:>10,.2f} "
            f"{kpi['G_kWh']:>10,.1f} "
            f"{kpi['G_Smc']:>10,.2f} "
            f"{kpi['C_gas_eur']:>10,.2f} "
            f"{kpi['C0_eur']:>10,.2f}"
        )
    print("-" * 92)
    print(
        f"{'BUILDING':<20} "
        f"{bld_kpi['E_el_kWh']:>12,.1f} "
        f"{bld_kpi['C_el_eur']:>10,.2f} "
        f"{bld_kpi['G_kWh']:>10,.1f} "
        f"{bld_kpi['G_Smc']:>10,.2f} "
        f"{bld_kpi['C_gas_eur']:>10,.2f} "
        f"{bld_kpi['C0_eur']:>10,.2f}"
    )
    print(f"\nOutputs saved to: {output_dir.resolve()}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
