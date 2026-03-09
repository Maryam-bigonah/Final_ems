"""
src/kpi.py – Compute annual KPIs per archetype and for the building aggregate.
"""
from __future__ import annotations

import pandas as pd

from .tariff import apply_electricity_tariff, apply_gas_tariff, tou_split


# ── per-archetype KPIs ───────────────────────────────────────────────────────

def compute_archetype_kpis(
    arch_name: str,
    e_el: pd.Series,
    gas_df: pd.DataFrame,
    el_prices: pd.DataFrame,
    gas_prices: pd.Series,
) -> dict:
    """
    Compute annual KPIs for one archetype.
    Returns a dict with all KPI fields.
    """
    # Electricity cost
    c_el = apply_electricity_tariff(e_el, el_prices)

    # Gas cost (via Smc)
    c_gas = apply_gas_tariff(gas_df["gas_Smc"], gas_prices)

    # TOU split
    tou = tou_split(e_el, c_el, el_prices["arera_band"])

    # Assemble
    kpi = {
        "archetype": arch_name,
        "E_el_kWh": e_el.sum(),
        "C_el_eur": c_el.sum(),
        "G_kWh": gas_df["gas_kWh"].sum(),
        "G_Smc": gas_df["gas_Smc"].sum(),
        "C_gas_eur": c_gas.sum(),
        "C0_eur": c_el.sum() + c_gas.sum(),
    }

    # TOU detail
    for band in ["F1", "F2", "F3"]:
        if band in tou.index:
            kpi[f"E_el_{band}_kWh"] = tou.loc[band, "energy_kWh"]
            kpi[f"C_el_{band}_eur"] = tou.loc[band, "cost_eur"]
        else:
            kpi[f"E_el_{band}_kWh"] = 0.0
            kpi[f"C_el_{band}_eur"] = 0.0

    return kpi, c_el, c_gas, tou


def compute_building_kpis(
    archetype_kpis: list[dict],
    Nk: dict[str, int],
) -> dict:
    """
    Scale each archetype by Nk and sum to get building totals.
    """
    bld = {
        "archetype": "BUILDING",
        "E_el_kWh": 0.0,
        "C_el_eur": 0.0,
        "G_kWh": 0.0,
        "G_Smc": 0.0,
        "C_gas_eur": 0.0,
        "C0_eur": 0.0,
    }
    for band in ["F1", "F2", "F3"]:
        bld[f"E_el_{band}_kWh"] = 0.0
        bld[f"C_el_{band}_eur"] = 0.0

    for kpi in archetype_kpis:
        arch = kpi["archetype"]
        n = Nk.get(arch, 1)
        for key in bld:
            if key == "archetype":
                continue
            bld[key] += kpi[key] * n

    return bld


# ── monthly & seasonal breakdowns ────────────────────────────────────────────

def compute_monthly_breakdown(
    arch_name: str,
    e_el: pd.Series,
    c_el: pd.Series,
    gas_df: pd.DataFrame,
    c_gas: pd.Series,
) -> pd.DataFrame:
    """Monthly energy/cost breakdown for one archetype."""
    monthly = pd.DataFrame(
        {
            "E_el_kWh": e_el,
            "C_el_eur": c_el,
            "G_kWh": gas_df["gas_kWh"].values,
            "G_Smc": gas_df["gas_Smc"].values,
            "C_gas_eur": c_gas.values,
        },
        index=e_el.index,
    )
    monthly = monthly.resample("ME").sum()
    monthly["C0_eur"] = monthly["C_el_eur"] + monthly["C_gas_eur"]
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.index.name = "month"
    monthly.insert(0, "archetype", arch_name)
    return monthly


def compute_seasonal_breakdown(
    arch_name: str,
    e_el: pd.Series,
    c_el: pd.Series,
    gas_df: pd.DataFrame,
    c_gas: pd.Series,
    season_map: dict[str, list[int]],
) -> pd.DataFrame:
    """Seasonal energy/cost breakdown for one archetype."""
    df = pd.DataFrame(
        {
            "E_el_kWh": e_el.values,
            "C_el_eur": c_el.values,
            "G_kWh": gas_df["gas_kWh"].values,
            "G_Smc": gas_df["gas_Smc"].values,
            "C_gas_eur": c_gas.values,
        },
        index=e_el.index,
    )
    month_to_season = {}
    for season, months in season_map.items():
        for m in months:
            month_to_season[m] = season

    df["season"] = df.index.month.map(month_to_season)

    seasonal = df.groupby("season")[
        ["E_el_kWh", "C_el_eur", "G_kWh", "G_Smc", "C_gas_eur"]
    ].sum()
    seasonal["C0_eur"] = seasonal["C_el_eur"] + seasonal["C_gas_eur"]

    # Reorder seasons
    season_order = list(season_map.keys())
    seasonal = seasonal.reindex(season_order)
    seasonal.insert(0, "archetype", arch_name)
    return seasonal
