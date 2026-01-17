"""Distillate fuel oil (DFO) generator analytics from EIA 860/923."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "dfo_generators"

EIA860_OPERABLE = DATA_DIR / "eia860_2024_generator_operable.csv"
EIA860_PROPOSED = DATA_DIR / "eia860_2024_generator_proposed.csv"
EIA860_RETIRED = DATA_DIR / "eia860_2024_generator_retired_canceled.csv"

EIA923_FUEL_COSTS = DATA_DIR / "eia923_2024_page5_fuel_receipts_costs.csv"

DFO_CODES = {"DFO"}


class DFOGeneratorError(Exception):
    """Raised when DFO generator parsing fails."""


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


def _find_header_row(df: pd.DataFrame, first_token: str, second_token: str) -> int:
    first = first_token.strip().lower()
    second = second_token.strip().lower()
    for idx, row in df.iterrows():
        row_vals = [str(x).strip().lower() for x in row.tolist()]
        if row_vals and row_vals[0] == first and second in row_vals:
            return idx
    raise DFOGeneratorError(f"Header row not found for {first_token}/{second_token}.")


def _load_eia860(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, low_memory=False)
    header_idx = _find_header_row(df, "Utility ID", "Plant Code")
    header = _dedupe_columns(df.iloc[header_idx].tolist())
    data = df.iloc[header_idx + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")
    data = data.dropna(subset=["Utility ID"]) if "Utility ID" in data.columns else data
    drop_cols = [col for col in data.columns if col.startswith("Unnamed") or col.startswith("col_")]
    data = data.drop(columns=drop_cols, errors="ignore")
    return data


def _extract_dfo_generators(df: pd.DataFrame, status: str) -> pd.DataFrame:
    energy_cols = [col for col in df.columns if str(col).startswith("Energy Source")]
    if not energy_cols:
        return pd.DataFrame()
    mask = False
    for col in energy_cols:
        mask = mask | df[col].astype(str).str.strip().isin(DFO_CODES)
    filtered = df[mask].copy()
    if filtered.empty:
        return filtered

    filtered["status"] = status
    filtered["nameplate_mw"] = pd.to_numeric(filtered.get("Nameplate Capacity (MW)"), errors="coerce")
    filtered["summer_mw"] = pd.to_numeric(filtered.get("Summer Capacity (MW)"), errors="coerce")
    filtered["winter_mw"] = pd.to_numeric(filtered.get("Winter Capacity (MW)"), errors="coerce")
    filtered["plant_id"] = pd.to_numeric(filtered.get("Plant Code"), errors="coerce")
    filtered["utility_id"] = pd.to_numeric(filtered.get("Utility ID"), errors="coerce")
    filtered["fuel_sources"] = (
        filtered[energy_cols]
        .astype(str)
        .apply(lambda row: ", ".join(sorted({v for v in row if v and v != "nan"})), axis=1)
    )

    keep_cols = [
        "plant_id",
        "plant_name",
        "state",
        "county",
        "utility_name",
        "utility_id",
        "generator_id",
        "technology",
        "prime_mover",
        "status",
        "nameplate_mw",
        "summer_mw",
        "winter_mw",
        "fuel_sources",
    ]
    rename_map = {
        "Plant Name": "plant_name",
        "Plant Code": "plant_id",
        "State": "state",
        "County": "county",
        "Utility Name": "utility_name",
        "Utility ID": "utility_id",
        "Generator ID": "generator_id",
        "Technology": "technology",
        "Prime Mover": "prime_mover",
    }
    filtered = filtered.rename(columns=rename_map)
    filtered = filtered[[col for col in keep_cols if col in filtered.columns]].copy()
    filtered.columns = _dedupe_columns(filtered.columns)
    return filtered


def _state_region(state: str) -> str:
    if not state or state == "nan":
        return "Unknown"
    state = state.strip().upper()
    northeast = {"CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"}
    midwest = {"IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"}
    south = {
        "AL",
        "AR",
        "DE",
        "DC",
        "FL",
        "GA",
        "KY",
        "LA",
        "MD",
        "MS",
        "NC",
        "OK",
        "SC",
        "TN",
        "TX",
        "VA",
        "WV",
    }
    west = {"AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"}
    if state in northeast:
        return "Northeast"
    if state in midwest:
        return "Midwest"
    if state in south:
        return "South"
    if state in west:
        return "West"
    return "Other"


def build_dfo_inventory(*, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    base = Path(data_dir)
    frames = []
    if (base / EIA860_OPERABLE.name).exists():
        frames.append(_extract_dfo_generators(_load_eia860(base / EIA860_OPERABLE.name), "Operable"))
    if (base / EIA860_PROPOSED.name).exists():
        frames.append(_extract_dfo_generators(_load_eia860(base / EIA860_PROPOSED.name), "Proposed"))
    if (base / EIA860_RETIRED.name).exists():
        frames.append(_extract_dfo_generators(_load_eia860(base / EIA860_RETIRED.name), "Retired/Canceled"))

    if not frames:
        raise DFOGeneratorError("No EIA860 generator data found.")

    inventory = pd.concat(frames, ignore_index=True)
    inventory["region"] = inventory["state"].astype(str).apply(_state_region)
    return inventory


def build_dfo_fuel_costs(*, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    base = Path(data_dir)
    path = base / EIA923_FUEL_COSTS.name
    if not path.exists():
        raise DFOGeneratorError("EIA923 fuel costs CSV not found.")
    df = pd.read_csv(path, header=None, low_memory=False)
    header_idx = _find_header_row(df, "YEAR", "ENERGY_SOURCE")
    header = _dedupe_columns(df.iloc[header_idx].tolist())
    data = df.iloc[header_idx + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")

    data = data[data["ENERGY_SOURCE"].astype(str).str.strip().isin(DFO_CODES)]
    if data.empty:
        return data

    data["YEAR"] = pd.to_numeric(data["YEAR"], errors="coerce")
    data["MONTH"] = pd.to_numeric(data["MONTH"], errors="coerce")
    data["Plant Id"] = pd.to_numeric(data["Plant Id"], errors="coerce")
    data["QUANTITY"] = pd.to_numeric(data.get("QUANTITY"), errors="coerce")
    data["FUEL_COST"] = pd.to_numeric(data.get("FUEL_COST"), errors="coerce")

    data["date"] = pd.to_datetime(
        data["YEAR"].astype("Int64").astype(str) + "-" + data["MONTH"].astype("Int64").astype(str).str.zfill(2) + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )
    data = data[data["date"].notna()]
    data = data.rename(columns={"Plant Id": "plant_id", "Plant State": "state", "Plant Name": "plant_name"})
    data["region"] = data["state"].astype(str).apply(_state_region)

    return data[[
        "date",
        "plant_id",
        "plant_name",
        "state",
        "region",
        "FUEL_COST",
        "QUANTITY",
    ]].dropna(subset=["plant_id"])


__all__ = ["build_dfo_inventory", "build_dfo_fuel_costs", "DFOGeneratorError"]
