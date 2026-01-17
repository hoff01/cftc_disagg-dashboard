"""Europe energy dataset helpers (Spain CORES + UK Energy Trends)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "europe_energy"

SPAIN_OIL_PRODUCTS = DATA_DIR / "Spain_content_oil_products_all.csv"
SPAIN_CRUDE_BALANCE = DATA_DIR / "Spain_content_crude_oil_balance_refinery_output.csv"
SPAIN_GAS_MARKET = DATA_DIR / "Spain_content_gas_consumption_by_market.csv"
SPAIN_GAS_PRESSURE = DATA_DIR / "Spain_content_gas_consumption_by_pressure_bracket.csv"

UK_ET_312 = DATA_DIR / "UK_content_et_3_12_month.csv"
UK_ET_313 = DATA_DIR / "UK_content_et_3_13_month.csv"


class EuropeEnergyError(Exception):
    """Raised when Europe energy parsing fails."""


def _dedupe_columns(columns: Iterable[object]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for idx, col in enumerate(columns):
        name = "" if pd.isna(col) else str(col).strip()
        if not name:
            name = f"col_{idx}"
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            name = f"{name}_{counts[name]}"
        result.append(name)
    return result


def _parse_month_year(year: pd.Series, month: pd.Series) -> pd.Series:
    year_num = pd.to_numeric(year, errors="coerce")
    month_num = pd.to_numeric(month, errors="coerce")
    month_str = month.astype(str).str.strip()
    month_abbr = month_str.str[:3].str.title()
    date = pd.to_datetime(month_abbr + " " + year_num.astype("Int64").astype(str), format="%b %Y", errors="coerce")
    fallback_month = month_num.fillna(0).astype("Int64").astype(str).str.zfill(2)
    fallback_year = year_num.fillna(0).astype("Int64").astype(str)
    fallback = pd.to_datetime(
        fallback_year + "-" + fallback_month + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )
    return date.fillna(fallback)


def _find_header_row(df: pd.DataFrame, first: str, second: str) -> int:
    target_first = first.strip().lower()
    target_second = second.strip().lower()
    for idx, row in df.iterrows():
        if str(row.iloc[0]).strip().lower() == target_first and str(row.iloc[1]).strip().lower() == target_second:
            return idx
    raise EuropeEnergyError(f"Header row with {first}/{second} not found")


def _parse_year_month_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    header_idx = _find_header_row(df, "Year", "Month")
    header = _dedupe_columns(df.iloc[header_idx].tolist())
    data = df.iloc[header_idx + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")

    data = data.dropna(subset=["Year"]) if "Year" in data.columns else data
    if "Year" in data.columns:
        data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
        data = data[data["Year"].notna()]
    if "Month" in data.columns:
        data["Month"] = data["Month"].astype(str).str.strip()

    data["date"] = _parse_month_year(data.get("Year"), data.get("Month"))
    data = data[data["date"].notna()]

    drop_cols = [col for col in data.columns if col.startswith("col_") and data[col].isna().all()]
    data = data.drop(columns=drop_cols, errors="ignore")
    return data


def _parse_grouped_year_month_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    header_idx = _find_header_row(df, "Year", "Month")
    group_row = df.iloc[header_idx - 1].copy()
    sub_row = df.iloc[header_idx].copy()

    group_row.iloc[0:2] = None
    group_row = group_row.ffill()

    headers: list[str] = []
    for idx, sub in enumerate(sub_row.tolist()):
        if idx == 0:
            headers.append("Year")
            continue
        if idx == 1:
            headers.append("Month")
            continue
        group = group_row.iloc[idx]
        group_name = "" if pd.isna(group) else str(group).strip()
        sub_name = "" if pd.isna(sub) else str(sub).strip()
        if not group_name and not sub_name:
            headers.append(f"col_{idx}")
            continue
        if group_name and sub_name:
            headers.append(f"{group_name} - {sub_name}")
        else:
            headers.append(group_name or sub_name)

    data = df.iloc[header_idx + 1 :].copy()
    data.columns = _dedupe_columns(headers)
    data = data.dropna(how="all")
    data = data.dropna(subset=["Year"]) if "Year" in data.columns else data
    if "Year" in data.columns:
        data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
        data = data[data["Year"].notna()]
    if "Month" in data.columns:
        data["Month"] = data["Month"].astype(str).str.strip()
    data["date"] = _parse_month_year(data.get("Year"), data.get("Month"))
    data = data[data["date"].notna()]

    drop_cols = [col for col in data.columns if col.startswith("col_") and data[col].isna().all()]
    data = data.drop(columns=drop_cols, errors="ignore")
    return data


def _parse_uk_month_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    header_idx = None
    for idx, row in df.iterrows():
        if str(row.iloc[0]).strip().lower() == "date":
            header_idx = idx
            break
    if header_idx is None:
        raise EuropeEnergyError(f"Could not locate Date header in {path.name}")
    header = _dedupe_columns(df.iloc[header_idx].tolist())
    data = df.iloc[header_idx + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")
    data = data.dropna(subset=["Date"]) if "Date" in data.columns else data
    raw_dates = data.get("Date")
    data["date"] = pd.to_datetime(raw_dates, format="%B %Y", errors="coerce")
    data["date"] = data["date"].fillna(pd.to_datetime(raw_dates, format="%b %Y", errors="coerce"))
    data["date"] = data["date"].fillna(pd.to_datetime(raw_dates, format="%Y-%m", errors="coerce"))
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    data = data[data["date"].notna()]

    drop_cols = [col for col in data.columns if col.startswith("col_") and data[col].isna().all()]
    data = data.drop(columns=drop_cols, errors="ignore")
    return data


def _melt_dataset(df: pd.DataFrame, *, dataset: str, region: str, unit: str) -> pd.DataFrame:
    id_cols = ["date"]
    value_cols = [col for col in df.columns if col not in {"Year", "Month", "Date", "date"}]
    melted = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="series", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value", "date"])
    melted["dataset"] = dataset
    melted["region"] = region
    melted["unit"] = unit
    melted["year"] = melted["date"].dt.year
    melted["month"] = melted["date"].dt.month
    return melted


def build_europe_energy_long(*, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    base = Path(data_dir)

    datasets = []
    if (base / SPAIN_OIL_PRODUCTS.name).exists():
        df = _parse_year_month_table(base / SPAIN_OIL_PRODUCTS.name)
        datasets.append(
            _melt_dataset(
                df,
                dataset="Spain - Oil Products Consumption",
                region="Spain",
                unit="tonnes",
            )
        )
    if (base / SPAIN_CRUDE_BALANCE.name).exists():
        df = _parse_year_month_table(base / SPAIN_CRUDE_BALANCE.name)
        datasets.append(
            _melt_dataset(
                df,
                dataset="Spain - Crude Balance & Refinery Output",
                region="Spain",
                unit="thousand tonnes",
            )
        )
    if (base / SPAIN_GAS_MARKET.name).exists():
        df = _parse_grouped_year_month_table(base / SPAIN_GAS_MARKET.name)
        datasets.append(
            _melt_dataset(
                df,
                dataset="Spain - Gas Consumption (Market)",
                region="Spain",
                unit="GWh",
            )
        )
    if (base / SPAIN_GAS_PRESSURE.name).exists():
        df = _parse_grouped_year_month_table(base / SPAIN_GAS_PRESSURE.name)
        datasets.append(
            _melt_dataset(
                df,
                dataset="Spain - Gas Consumption (Pressure)",
                region="Spain",
                unit="GWh",
            )
        )

    if (base / UK_ET_312.name).exists():
        df = _parse_uk_month_table(base / UK_ET_312.name)
        datasets.append(
            _melt_dataset(
                df,
                dataset="UK - Refinery Throughput & Output (ET 3.12)",
                region="United Kingdom",
                unit="thousand tonnes",
            )
        )
    if (base / UK_ET_313.name).exists():
        df = _parse_uk_month_table(base / UK_ET_313.name)
        datasets.append(
            _melt_dataset(
                df,
                dataset="UK - Deliveries for Inland Consumption (ET 3.13)",
                region="United Kingdom",
                unit="thousand tonnes",
            )
        )

    if not datasets:
        raise EuropeEnergyError("No Europe energy datasets found to parse.")

    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.drop_duplicates(subset=["dataset", "series", "date"], keep="last")
    combined = combined.sort_values(["dataset", "series", "date"])
    return combined


__all__ = ["build_europe_energy_long", "EuropeEnergyError"]
