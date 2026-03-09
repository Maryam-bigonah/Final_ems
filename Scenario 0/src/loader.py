"""
src/loader.py – Robust data loading, parsing, validation, and year-shifting.

All loaders return pandas objects indexed by a 2025 hourly DateTimeIndex
(8 760 steps: 2025-01-01 00:00 … 2025-12-31 23:00).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

_TS_CANDIDATES = ["timestamp", "time", "datetime", "date_time", "Time"]

EXPECTED_HOURS = 8_760  # non-leap year


def _find_ts_col(columns: list[str]) -> str:
    """Return the first column name that looks like a timestamp."""
    for c in columns:
        if c.strip().lower() in [t.lower() for t in _TS_CANDIDATES]:
            return c
    raise ValueError(
        f"Could not auto-detect a timestamp column among {columns}.\n"
        f"Expected one of: {_TS_CANDIDATES}"
    )


def _parse_time(series: pd.Series) -> pd.DatetimeIndex:
    """Parse a time column, trying common date formats."""
    for fmt in ["%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", None]:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError("Unable to parse timestamps. Provide an explicit format.")


def _year_shift_and_trim(
    df: pd.DataFrame, target_year: int
) -> pd.DataFrame:
    """
    Drop Feb-29 rows (if source is leap year),
    then replace the year in the index to *target_year*.
    """
    idx = df.index
    # Drop Feb 29
    mask_feb29 = (idx.month == 2) & (idx.day == 29)
    n_dropped = mask_feb29.sum()
    if n_dropped > 0:
        print(f"  ↳ Dropping {n_dropped} rows on Feb 29 (leap year → non-leap).")
        df = df.loc[~mask_feb29].copy()

    # Shift year
    new_index = df.index.map(lambda ts: ts.replace(year=target_year))
    df.index = pd.DatetimeIndex(new_index, name="timestamp")
    return df


def _resample_minute_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-resolution data to hourly by summing.
    Assumes values represent energy consumed in each minute-interval.
    """
    # Floor to hour and groupby-sum
    df = df.copy()
    df.index = df.index.floor("h")
    hourly = df.groupby(df.index).sum()
    hourly.index.name = "timestamp"
    return hourly


# ── electricity loader ───────────────────────────────────────────────────────

def load_electricity(
    path: str | Path,
    archetype_name: str,
    target_year: int = 2025,
) -> pd.Series:
    """
    Load a sub-metered electricity CSV (semicolon-delimited).
    Returns an hourly pd.Series of total electricity demand [kWh/h]
    indexed by 2025 timestamps.
    """
    path = Path(path)
    print(f"Loading electricity: {path.name} [{archetype_name}]")

    df = pd.read_csv(path, sep=";", low_memory=False)

    # Find and parse timestamp
    ts_col = _find_ts_col(df.columns.tolist())
    df["_ts"] = _parse_time(df[ts_col])
    df = df.set_index("_ts").sort_index()
    df.index.name = "timestamp"

    # Drop non-numeric and helper columns
    cols_to_drop = [
        c
        for c in df.columns
        if not np.issubdtype(df[c].dtype, np.number)
    ]
    # Also drop the Timestep counter column if present
    for c in list(df.columns):
        if "timestep" in c.lower():
            cols_to_drop.append(c)
    cols_to_drop = list(set(cols_to_drop))
    numeric_df = df.drop(columns=cols_to_drop, errors="ignore")

    # Detect resolution
    deltas = pd.Series(numeric_df.index).diff().dropna()
    median_delta = deltas.median()
    is_minute = median_delta <= pd.Timedelta(minutes=2)

    if is_minute:
        print(f"  ↳ Minute-level data detected ({len(numeric_df)} rows). Resampling to hourly …")
        numeric_df = _resample_minute_to_hourly(numeric_df)
        print(f"  ↳ After resampling: {len(numeric_df)} rows.")

    # Sum all appliance columns → total demand per hour
    total = numeric_df.sum(axis=1).rename(f"E_el_{archetype_name}")

    # Year-shift
    total_df = total.to_frame()
    total_df = _year_shift_and_trim(total_df, target_year)
    total = total_df.iloc[:, 0]

    # Validate
    if len(total) != EXPECTED_HOURS:
        raise ValueError(
            f"[{archetype_name}] Expected {EXPECTED_HOURS} hourly rows after "
            f"year-shift, got {len(total)}."
        )
    if total.isna().any():
        raise ValueError(f"[{archetype_name}] NaN values found after loading electricity.")
    if (total < 0).any():
        raise ValueError(f"[{archetype_name}] Negative electricity demand found.")

    print(f"  ✓ {archetype_name} electricity loaded: {total.sum():.1f} kWh/year")
    return total


# ── gas loader ───────────────────────────────────────────────────────────────

def load_gas(
    path: str | Path,
    archetype_name: str,
    target_year: int = 2025,
) -> pd.DataFrame:
    """
    Load a gas CSV (semicolon-delimited).
    Returns an hourly pd.DataFrame with columns [gas_kWh, gas_Smc]
    indexed by 2025 timestamps.
    """
    path = Path(path)
    print(f"Loading gas: {path.name} [{archetype_name}]")

    df = pd.read_csv(path, sep=";", low_memory=False)

    ts_col = _find_ts_col(df.columns.tolist())
    df["_ts"] = _parse_time(df[ts_col])
    df = df.set_index("_ts").sort_index()
    df.index.name = "timestamp"

    # Extract the two columns we need
    gas_kwh_col = [c for c in df.columns if "gas_kwh" in c.lower()]
    gas_smc_col = [c for c in df.columns if "gas_smc" in c.lower() or "smc" in c.lower()]

    if not gas_kwh_col:
        raise ValueError(f"[{archetype_name}] Cannot find 'gas_kWh' column in {path.name}")
    if not gas_smc_col:
        raise ValueError(f"[{archetype_name}] Cannot find 'gas_Smc' column in {path.name}")

    result = pd.DataFrame(
        {
            "gas_kWh": pd.to_numeric(df[gas_kwh_col[0]], errors="coerce"),
            "gas_Smc": pd.to_numeric(df[gas_smc_col[0]], errors="coerce"),
        },
        index=df.index,
    )

    # Detect resolution
    deltas = pd.Series(result.index).diff().dropna()
    median_delta = deltas.median()
    is_minute = median_delta <= pd.Timedelta(minutes=2)

    if is_minute:
        print(f"  ↳ Minute-level data detected ({len(result)} rows). Resampling to hourly …")
        result = _resample_minute_to_hourly(result)
        print(f"  ↳ After resampling: {len(result)} rows.")

    # Year-shift
    result = _year_shift_and_trim(result, target_year)

    # Validate
    if len(result) != EXPECTED_HOURS:
        raise ValueError(
            f"[{archetype_name}] Expected {EXPECTED_HOURS} hourly gas rows, "
            f"got {len(result)}."
        )
    if result.isna().any().any():
        raise ValueError(f"[{archetype_name}] NaN values found in gas data.")
    if (result < 0).any().any():
        raise ValueError(f"[{archetype_name}] Negative gas demand found.")

    print(
        f"  ✓ {archetype_name} gas loaded: "
        f"{result['gas_kWh'].sum():.1f} kWh/year, "
        f"{result['gas_Smc'].sum():.2f} Smc/year"
    )
    return result


# ── EV loader ────────────────────────────────────────────────────────────────

def load_ev(
    path: str | Path,
    ev_column_mapping: dict[str, str],
    archetype_keys: list[str],
) -> pd.DataFrame:
    """
    Load EV.csv (comma-delimited, already 2025 hourly).
    Returns a DataFrame with one column per archetype (kWh/h),
    filling zero for archetypes without EV.
    """
    path = Path(path)
    print(f"Loading EV: {path.name}")

    df = pd.read_csv(path, low_memory=False)

    ts_col = _find_ts_col(df.columns.tolist())
    df["_ts"] = _parse_time(df[ts_col])
    df = df.set_index("_ts").sort_index()
    df.index.name = "timestamp"

    result = pd.DataFrame(index=df.index)

    for arch in archetype_keys:
        if arch in ev_column_mapping:
            fragment = ev_column_mapping[arch]
            matched = [c for c in df.columns if fragment in c]
            if len(matched) == 0:
                raise ValueError(
                    f"EV column mapping for '{arch}' → '{fragment}' "
                    f"matched no columns in EV.csv.\n"
                    f"Available columns: {df.columns.tolist()}"
                )
            if len(matched) > 1:
                raise ValueError(
                    f"EV column mapping for '{arch}' → '{fragment}' "
                    f"matched multiple columns: {matched}. "
                    f"Please specify a more precise mapping."
                )
            col = matched[0]
            result[arch] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            print(f"  ↳ {arch} → '{col}': {result[arch].sum():.1f} kWh/year")
        else:
            result[arch] = 0.0
            print(f"  ↳ {arch} → no EV (set to 0)")

    # Validate
    if len(result) != EXPECTED_HOURS:
        raise ValueError(
            f"EV data: expected {EXPECTED_HOURS} rows, got {len(result)}."
        )
    if result.isna().any().any():
        raise ValueError("NaN values found in EV data after loading.")

    return result


# ── electricity price loader ─────────────────────────────────────────────────

def load_electricity_price(path: str | Path) -> pd.DataFrame:
    """
    Load hourly electricity price CSV (comma-delimited, 2025).
    Returns DataFrame with columns [price_eur_per_kwh, arera_band].
    """
    path = Path(path)
    print(f"Loading electricity price: {path.name}")

    df = pd.read_csv(path)

    # Detect timestamp
    ts_col = _find_ts_col(df.columns.tolist())
    df["_ts"] = pd.to_datetime(df[ts_col])
    df = df.set_index("_ts").sort_index()
    df.index.name = "timestamp"

    # Validate required columns
    if "price_eur_per_kwh" not in df.columns:
        raise ValueError("Electricity price file missing 'price_eur_per_kwh' column.")
    if "arera_band" not in df.columns:
        raise ValueError(
            "Electricity price file missing 'arera_band' column. "
            "Cannot determine TOU bands (F1/F2/F3)."
        )

    result = df[["price_eur_per_kwh", "arera_band"]].copy()
    result["price_eur_per_kwh"] = pd.to_numeric(
        result["price_eur_per_kwh"], errors="coerce"
    )

    # Validate
    if len(result) != EXPECTED_HOURS:
        raise ValueError(
            f"Electricity price: expected {EXPECTED_HOURS} rows, got {len(result)}."
        )
    if result["price_eur_per_kwh"].isna().any():
        raise ValueError("NaN values in electricity prices.")
    if (result["price_eur_per_kwh"] < 0).any():
        raise ValueError("Negative electricity prices found.")

    bands = set(result["arera_band"].unique())
    expected_bands = {"F1", "F2", "F3"}
    if bands != expected_bands:
        raise ValueError(
            f"Unexpected TOU bands: {bands}. Expected: {expected_bands}"
        )

    print(
        f"  ✓ Electricity price loaded: "
        f"mean={result['price_eur_per_kwh'].mean():.4f} €/kWh, "
        f"bands={sorted(bands)}"
    )
    return result


# ── gas price loader ─────────────────────────────────────────────────────────

def load_gas_price(
    path: str | Path,
    target_year: int = 2025,
) -> pd.Series:
    """
    Load monthly gas price CSV and broadcast to 8 760 hourly values.
    Returns pd.Series of price in €/Smc indexed by 2025 DateTimeIndex.
    """
    path = Path(path)
    print(f"Loading gas price: {path.name}")

    df = pd.read_csv(path)

    # Find the price column
    price_col = [c for c in df.columns if "€/smc" in c.lower() or "eur" in c.lower() or "c_mem" in c.lower()]
    if not price_col:
        raise ValueError(
            f"Cannot find gas price column in {path.name}. "
            f"Columns: {df.columns.tolist()}"
        )
    price_col = price_col[0]

    # Parse month names
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }

    month_col = [c for c in df.columns if "month" in c.lower()]
    if not month_col:
        raise ValueError("Gas price file missing 'Month' column.")
    month_col = month_col[0]

    monthly_prices = {}
    for _, row in df.iterrows():
        m_str = str(row[month_col]).strip()
        m_num = month_map.get(m_str)
        if m_num is None:
            raise ValueError(f"Unrecognised month string: '{m_str}'")
        monthly_prices[m_num] = float(row[price_col])

    if len(monthly_prices) != 12:
        raise ValueError(
            f"Expected 12 monthly gas prices, got {len(monthly_prices)}."
        )

    # Build hourly index for 2025
    hourly_idx = pd.date_range(
        start=f"{target_year}-01-01",
        end=f"{target_year}-12-31 23:00:00",
        freq="h",
    )
    assert len(hourly_idx) == EXPECTED_HOURS, (
        f"Hourly index length {len(hourly_idx)} != {EXPECTED_HOURS}"
    )

    prices = pd.Series(
        [monthly_prices[ts.month] for ts in hourly_idx],
        index=hourly_idx,
        name="gas_price_eur_per_smc",
    )
    prices.index.name = "timestamp"

    print(f"  ✓ Gas price loaded: mean={prices.mean():.4f} €/Smc")
    return prices
