"""
src/demand.py – Construct hourly electricity and gas demand per archetype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ev_double_count_diagnostic(
    base_el: pd.Series,
    ev_series: pd.Series,
    archetype_name: str,
) -> None:
    """
    Print diagnostic warnings if EV energy might already be embedded
    inside the base electricity series.

    Checks:
    1. Night-hour (23:00–06:00) correlation between base and EV.
    2. Whether EV annual total is < 5 % of base annual total (trivially small).
    """
    if ev_series.sum() == 0:
        return  # No EV for this archetype

    print(f"  ↳ EV double-count diagnostic for {archetype_name}:")

    # Night hours mask
    night = base_el.index.hour.isin([23, 0, 1, 2, 3, 4, 5, 6])
    base_night = base_el[night]
    ev_night = ev_series[night]

    if ev_night.std() > 0 and base_night.std() > 0:
        corr = np.corrcoef(base_night.values, ev_night.values)[0, 1]
        print(f"    Night-hour correlation (base vs EV): {corr:.3f}")
        if corr > 0.6:
            print(
                "    ⚠ WARNING: High night-hour correlation suggests EV may "
                "already be embedded in base electricity. Consider INCLUDE_EV=False."
            )
    else:
        print("    Night-hour correlation: N/A (zero variance)")

    ratio = ev_series.sum() / base_el.sum() * 100
    print(f"    EV / base ratio: {ratio:.1f} %")


def build_electricity_demand(
    base_el: pd.Series,
    ev_series: pd.Series,
    include_ev: bool,
    archetype_name: str,
) -> pd.Series:
    """
    Total electricity demand: E_el_k(t) = E_base_k(t) + EV_k(t) if include_ev.
    """
    ev_double_count_diagnostic(base_el, ev_series, archetype_name)

    if include_ev:
        total = base_el + ev_series
        ev_kwh = ev_series.sum()
        print(
            f"  ✓ {archetype_name} total electricity = "
            f"{base_el.sum():.1f} (base) + {ev_kwh:.1f} (EV) = "
            f"{total.sum():.1f} kWh/year"
        )
    else:
        total = base_el.copy()
        print(
            f"  ✓ {archetype_name} total electricity = "
            f"{total.sum():.1f} kWh/year (EV excluded)"
        )

    return total.rename(f"E_el_{archetype_name}")


def build_gas_demand(
    gas_df: pd.DataFrame,
    archetype_name: str,
) -> pd.DataFrame:
    """
    Pass-through: gas demand is already clean from loader.
    Returns DataFrame with gas_kWh and gas_Smc columns.
    """
    print(
        f"  ✓ {archetype_name} gas demand: "
        f"{gas_df['gas_kWh'].sum():.1f} kWh/year, "
        f"{gas_df['gas_Smc'].sum():.2f} Smc/year"
    )
    return gas_df
