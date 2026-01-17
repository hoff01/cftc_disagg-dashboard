"""Port of Los Angeles container statistics (TEUs)."""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from typing import Optional

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data" / "port_teu"
PORT_LA_CACHE_CSV = DATA_DIR / "port_of_la_container_stats_2021_present.csv"
PORT_LA_URL_TEMPLATE = (
    "https://portoflosangeles.org/business/statistics/container-statistics/"
    "historical-teu-statistics-{year}"
)

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_TO_INDEX = {month: idx for idx, month in enumerate(MONTHS, start=1)}


def _normalize_header(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("\u2014", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace({"nan": "", "": None, "-": None})
    return pd.to_numeric(cleaned, errors="coerce")


def fetch_port_la_teu_table(year: int, *, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch a single year of Port of LA container statistics."""
    url = PORT_LA_URL_TEMPLATE.format(year=year)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    
    if response.status_code == 404:
        current_year = datetime.utcnow().year
        if year >= current_year:
            print(f"Warning: Port of LA data for {year} not found (404). Skipping.")
            return pd.DataFrame()
    
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError(f"No tables found for {year}.")

    raw = tables[0]
    if raw.empty:
        raise ValueError(f"Empty table for {year}.")

    header = [_normalize_header(value) for value in raw.iloc[0].tolist()]
    data = raw.iloc[1:].copy()
    if header and not header[0]:
        header[0] = "Month"
    data.columns = header
    data = data.loc[:, ~data.columns.duplicated()]

    if "Month" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Month"})

    data["Month"] = data["Month"].astype(str).str.strip().str.title()
    data = data[data["Month"].isin(MONTHS)]

    data.insert(0, "Year", year)

    for column in data.columns:
        if column in {"Year", "Month"}:
            continue
        data[column] = _coerce_numeric(data[column])

    ordered_columns = ["Year", "Month"] + [col for col in header[1:] if col]
    ordered_columns = [col for col in ordered_columns if col in data.columns]
    data = data[ordered_columns]

    return data.reset_index(drop=True)


def load_port_la_teu_cache(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the cached Port of LA TEU dataset if available."""
    csv_path = Path(path) if path else PORT_LA_CACHE_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def update_port_la_teu_cache(
    *,
    start_year: int = 2021,
    end_year: Optional[int] = None,
    refresh_years: int = 2,
    output_csv: Path | str = PORT_LA_CACHE_CSV,
    load_existing: bool = True,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Update Port of LA TEU dataset, refreshing the most recent years."""
    now = datetime.utcnow()
    resolved_end_year = end_year or now.year
    output_path = Path(output_csv)

    existing = pd.DataFrame()
    if load_existing and output_path.exists():
        existing = pd.read_csv(output_path)

    if not existing.empty:
        years_to_refresh = [
            year
            for year in range(resolved_end_year - refresh_years + 1, resolved_end_year + 1)
            if year >= start_year
        ]
        existing = existing[~existing["Year"].isin(years_to_refresh)]
    else:
        years_to_refresh = list(range(start_year, resolved_end_year + 1))

    frames = [existing] if not existing.empty else []
    for year in years_to_refresh:
        frames.append(fetch_port_la_teu_table(year, timeout=timeout))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates().reset_index(drop=True)
        combined["MonthIndex"] = combined["Month"].map(MONTH_TO_INDEX)
        combined = combined.sort_values(["Year", "MonthIndex"]).drop(columns=["MonthIndex"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return combined


__all__ = [
    "fetch_port_la_teu_table",
    "load_port_la_teu_cache",
    "update_port_la_teu_cache",
]
