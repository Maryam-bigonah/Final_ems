"""
src/tariff.py – Apply hourly tariffs and compute TOU splits.
"""
from __future__ import annotations

import pandas as pd


def apply_electricity_tariff(
    demand: pd.Series,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Hourly electricity cost: c_el(t) = p_el(t) × E_el(t).
    *prices* must have column 'price_eur_per_kwh' and same index as demand.
    Returns pd.Series of hourly cost [€].
    """
    cost = demand * prices["price_eur_per_kwh"]
    return cost.rename(demand.name.replace("E_el", "C_el") if demand.name else "C_el")


def apply_gas_tariff(
    gas_smc: pd.Series,
    gas_price_smc: pd.Series,
) -> pd.Series:
    """
    Hourly gas cost: c_gas(t) = p_gas(t) × gas_Smc(t).
    Returns pd.Series of hourly gas cost [€].
    """
    cost = gas_smc * gas_price_smc
    return cost.rename("C_gas")


def tou_split(
    demand: pd.Series,
    cost: pd.Series,
    band_series: pd.Series,
) -> pd.DataFrame:
    """
    Split electricity energy and cost by TOU band (F1/F2/F3).
    Returns a DataFrame with index = band, columns = [energy_kWh, cost_eur].

    Also verifies that splits sum to annual totals.
    """
    df = pd.DataFrame(
        {
            "energy_kWh": demand.values,
            "cost_eur": cost.values,
            "band": band_series.values,
        },
        index=demand.index,
    )

    split = df.groupby("band")[["energy_kWh", "cost_eur"]].sum()

    # Hard check: sums must equal totals
    total_energy = demand.sum()
    total_cost = cost.sum()
    split_energy = split["energy_kWh"].sum()
    split_cost = split["cost_eur"].sum()

    tol = 1e-6
    if abs(split_energy - total_energy) > tol:
        raise ValueError(
            f"TOU energy split mismatch: "
            f"sum(bands)={split_energy:.6f} vs total={total_energy:.6f}"
        )
    if abs(split_cost - total_cost) > tol:
        raise ValueError(
            f"TOU cost split mismatch: "
            f"sum(bands)={split_cost:.6f} vs total={total_cost:.6f}"
        )

    return split
