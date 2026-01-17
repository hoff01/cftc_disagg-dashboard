"""Port of NY/NJ monthly TEU throughput (PANYNJ)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

PORT_NY_PAGE_URL = "https://www.panynj.gov/port/en/our-port/facts-and-figures.html"
PORT_NY_MODEL_URL = f"{PORT_NY_PAGE_URL}.model.json"
PORT_NY_SECTION_TITLE = "Monthly Cargo Volumes"

DATA_DIR = Path(__file__).resolve().parent / "data" / "port_teu"
PORT_NY_CACHE_CSV = DATA_DIR / "port_ny_teu.csv"

MONTHS_FULL = [
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
MONTH_ABBR = [m[:3] for m in MONTHS_FULL]
MONTH_MAP = dict(zip(MONTHS_FULL, MONTH_ABBR))
MONTH_INDEX = {month: idx for idx, month in enumerate(MONTH_ABBR, start=1)}


class PortNYError(Exception):
    """Raised when Port of NY/NJ extraction fails."""


def _coerce_number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in {"-", "--", "nan", "NaN"}:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -number if negative else number


def _iter_dicts(obj: object) -> Iterable[dict]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_dicts(value)


def _extract_table_html(obj: object, tables: list[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _extract_table_html(value, tables)
    elif isinstance(obj, list):
        for value in obj:
            _extract_table_html(value, tables)
    elif isinstance(obj, str) and "<table" in obj.lower():
        tables.append(obj)


def _find_monthly_accordion(model: dict) -> dict:
    for entry in _iter_dicts(model):
        if entry.get(":type", "").endswith("AccordionList"):
            title = str(entry.get("title", "")).strip().lower()
            if PORT_NY_SECTION_TITLE.lower() in title:
                return entry
    raise PortNYError("Monthly Cargo Volumes section not found in page model JSON.")


def _parse_year_rows_from_table(table: BeautifulSoup, month: str, *, min_year: int) -> list[dict]:
    records = []
    for row in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", "") for c in row.find_all(["th", "td"])]
        if not cells:
            continue
        first = cells[0].strip()
        if not re.fullmatch(r"\d{4}", first):
            continue
        year = int(first)
        if year < min_year:
            continue
        values = cells[1:]
        if len(values) < 7:
            continue
        import_loads = _coerce_number(values[0])
        import_empties = _coerce_number(values[1])
        export_loads = _coerce_number(values[2])
        export_empties = _coerce_number(values[3])
        total_loads = _coerce_number(values[4])
        total_empties = _coerce_number(values[5])
        total = _coerce_number(values[6])
        if (
            import_loads == 0
            and import_empties == 0
            and export_loads == 0
            and export_empties == 0
            and total_loads == 0
            and total_empties == 0
            and total == 0
        ):
            continue
        records.append(
            {
                "Month": MONTH_MAP[month],
                "Year": year,
                "Import Loads": import_loads,
                "Import Empties": import_empties,
                "Export Loads": export_loads,
                "Export Empties": export_empties,
                "Total Loads": total_loads,
                "Total Empties": total_empties,
                "Total": total,
            }
        )
    return records


def _parse_months_with_headings(html: str, *, min_year: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []
    current_month: Optional[str] = None

    for element in soup.find_all(["h2", "table"]):
        if element.name == "h2":
            label = element.get_text(" ", strip=True)
            if label in MONTHS_FULL:
                current_month = label
            else:
                current_month = None
            continue
        if element.name == "table" and current_month:
            records.extend(_parse_year_rows_from_table(element, current_month, min_year=min_year))
            current_month = None

    return records


def _parse_months_in_rows(html: str, *, min_year: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    records = []
    current_month = None

    for row in rows:
        cells = [c.get_text(" ", strip=True).replace("\xa0", "") for c in row.find_all(["th", "td"])]
        if not cells:
            continue
        first = cells[0].strip()
        if first in MONTHS_FULL:
            current_month = first
            continue
        if not current_month:
            continue
        if re.fullmatch(r"\d{4}", first):
            year = int(first)
            if year < min_year:
                continue
            values = cells[1:]
            if len(values) < 7:
                continue
            import_loads = _coerce_number(values[0])
            import_empties = _coerce_number(values[1])
            export_loads = _coerce_number(values[2])
            export_empties = _coerce_number(values[3])
            total_loads = _coerce_number(values[4])
            total_empties = _coerce_number(values[5])
            total = _coerce_number(values[6])
            if (
                import_loads == 0
                and import_empties == 0
                and export_loads == 0
                and export_empties == 0
                and total_loads == 0
                and total_empties == 0
                and total == 0
            ):
                continue
            records.append(
                {
                    "Month": MONTH_MAP[current_month],
                    "Year": year,
                    "Import Loads": import_loads,
                    "Import Empties": import_empties,
                    "Export Loads": export_loads,
                    "Export Empties": export_empties,
                    "Total Loads": total_loads,
                    "Total Empties": total_empties,
                    "Total": total,
                }
            )

    return records


def fetch_port_ny_teu_table(
    *,
    page_url: str = PORT_NY_PAGE_URL,
    min_year: int = 2021,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch Port of NY/NJ monthly TEU table from the facts-and-figures page."""
    base_url = page_url[:-5] if page_url.endswith(".html") else page_url
    model_url = f"{base_url}.model.json"
    response = requests.get(model_url, timeout=timeout)
    response.raise_for_status()
    model = response.json()

    accordion = _find_monthly_accordion(model)
    tables: list[str] = []
    _extract_table_html(accordion, tables)

    records: list[dict] = []
    for html in tables:
        records.extend(_parse_months_with_headings(html, min_year=min_year))
        records.extend(_parse_months_in_rows(html, min_year=min_year))

    if not records:
        raise PortNYError("No monthly TEU rows found in Port of NY/NJ tables.")

    frame = pd.DataFrame(
        records,
        columns=[
            "Month",
            "Year",
            "Import Loads",
            "Import Empties",
            "Export Loads",
            "Export Empties",
            "Total Loads",
            "Total Empties",
            "Total",
        ],
    )

    frame = frame.drop_duplicates(subset=["Month", "Year"], keep="first")
    frame["MonthIndex"] = frame["Month"].map(MONTH_INDEX)
    frame = frame.sort_values(["Year", "MonthIndex"]).drop(columns=["MonthIndex"]).reset_index(drop=True)

    return frame


def load_port_ny_teu_cache(path: Optional[Path] = None) -> pd.DataFrame:
    """Load cached Port of NY/NJ TEU data if available."""
    csv_path = Path(path) if path else PORT_NY_CACHE_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def update_port_ny_teu_cache(
    *,
    output_csv: Path | str = PORT_NY_CACHE_CSV,
    min_year: int = 2021,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Refresh the Port of NY/NJ TEU CSV by appending new months."""
    output_path = Path(output_csv)
    frame = fetch_port_ny_teu_table(page_url=PORT_NY_PAGE_URL, min_year=min_year, timeout=timeout)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = load_port_ny_teu_cache(output_path)
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Month", "Year"], keep="first")
        combined["MonthIndex"] = combined["Month"].map(MONTH_INDEX)
        combined = (
            combined.sort_values(["Year", "MonthIndex"])
            .drop(columns=["MonthIndex"])
            .reset_index(drop=True)
        )
        combined.to_csv(output_path, index=False)
        return combined

    frame.to_csv(output_path, index=False)

    return frame


__all__ = [
    "fetch_port_ny_teu_table",
    "load_port_ny_teu_cache",
    "update_port_ny_teu_cache",
]
