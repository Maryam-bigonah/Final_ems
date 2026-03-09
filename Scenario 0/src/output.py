"""
src/output.py – Save tables, time series, and Excel workbook.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_csv_and_collect(
    df: pd.DataFrame,
    name: str,
    out_dir: Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    """Save a DataFrame as CSV and collect it for the Excel workbook."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path)
    sheets[name] = df
    print(f"  ✓ Saved {csv_path.name}")


def save_excel_workbook(
    sheets: dict[str, pd.DataFrame],
    out_dir: Path,
    filename: str = "scenario0_outputs.xlsx",
) -> None:
    """Write all collected DataFrames into a single Excel workbook."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / filename
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet names max 31 chars
            short_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=short_name)
    print(f"  ✓ Saved Excel workbook: {xlsx_path.name}")


def save_timeseries(
    arch_ts: dict[str, pd.DataFrame],
    building_ts: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save aligned hourly time series as Parquet (or CSV fallback)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Archetype time series
    frames = []
    for arch_name, ts_df in arch_ts.items():
        ts_copy = ts_df.copy()
        ts_copy.insert(0, "archetype", arch_name)
        frames.append(ts_copy)

    all_arch = pd.concat(frames, axis=0)

    try:
        arch_path = out_dir / "timeseries_archetypes_s0.parquet"
        all_arch.to_parquet(arch_path)
        print(f"  ✓ Saved {arch_path.name}")
    except Exception:
        arch_path = out_dir / "timeseries_archetypes_s0.csv"
        all_arch.to_csv(arch_path)
        print(f"  ✓ Saved {arch_path.name} (CSV fallback)")

    # Building time series
    try:
        bld_path = out_dir / "timeseries_building_s0.parquet"
        building_ts.to_parquet(bld_path)
        print(f"  ✓ Saved {bld_path.name}")
    except Exception:
        bld_path = out_dir / "timeseries_building_s0.csv"
        building_ts.to_csv(bld_path)
        print(f"  ✓ Saved {bld_path.name} (CSV fallback)")


def save_verification_report(
    results: list[tuple[str, bool, str]],
    out_dir: Path,
) -> None:
    """Write verification_report.txt with PASS/FAIL for each check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "verification_report.txt"

    lines = ["=" * 60, "SCENARIO 0 – VERIFICATION REPORT", "=" * 60, ""]

    all_pass = True
    for check_name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        lines.append(f"[{status}] {check_name}")
        if detail:
            lines.append(f"       {detail}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    lines.append("=" * 60)

    report_path.write_text("\n".join(lines))
    print(f"  ✓ Verification report: {report_path.name}")
    return all_pass
