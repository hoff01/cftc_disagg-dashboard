"""Port of Long Beach TEU archive (monthly)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent / "data" / "port_teu"
PORT_LB_CACHE_CSV = DATA_DIR / "port_long_beach_teu.csv"
PORT_LB_TABLE_URL = "https://thehelm.polb.com/stellar_custom_table/table100/"

MONTH_RE = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}$",
    re.IGNORECASE,
)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u2014", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace({"nan": "", "": None, "-": None})
    return pd.to_numeric(cleaned, errors="coerce")


def _fetch_table_html(
    *,
    url: str = PORT_LB_TABLE_URL,
    timeout: float = 30.0,
    engine: str = "requests",
) -> str:
    if engine == "requests":
        import requests

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text

    raise ValueError(f"Unknown engine: {engine}")


def _fetch_table_data_playwright(
    *, url: str = PORT_LB_TABLE_URL, timeout: float = 30.0
) -> tuple[list[str], list[list[str]]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required to fetch Port of Long Beach tables. "
            "Install with: pip install playwright && playwright install"
        ) from exc

    timeout_ms = int(timeout * 1000)
    with sync_playwright() as playwright:
        browser = playwright.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_function(
            "window.jQuery && window.jQuery.fn && window.jQuery.fn.DataTable && "
            "window.jQuery('#table_1').length",
            timeout=timeout_ms,
        )
        page.wait_for_timeout(2000)
        payload = page.evaluate(
            """
() => {
  const table = window.jQuery('#table_1').DataTable();
  return {
    columns: table.columns().header().toArray().map(h => h.textContent.trim()),
    data: table.data().toArray(),
  };
}
"""
        )
        browser.close()

    columns = payload.get("columns") if isinstance(payload, dict) else None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not columns or not data:
        raise ValueError("Failed to load Port of Long Beach data table.")
    return columns, data


def fetch_port_long_beach_teu_table(
    *,
    min_year: int = 1995,
    timeout: float = 30.0,
    engine: str = "playwright",
) -> pd.DataFrame:
    """Fetch Port of Long Beach TEU archive table."""
    if engine == "playwright":
        headers, rows = _fetch_table_data_playwright(timeout=timeout)
        frame = pd.DataFrame(rows, columns=headers)
    else:
        html = _fetch_table_html(timeout=timeout, engine=engine)
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", attrs={"data-wpdatatable_id": "100"})
        if not table:
            raise ValueError("Port of Long Beach table not found in HTML.")

        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            raise ValueError("Port of Long Beach table headers not found.")

        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cells:
                continue
            rows.append(cells)

        frame = pd.DataFrame(rows, columns=headers)
    if "Date" not in frame.columns:
        raise ValueError("Port of Long Beach table is missing the Date column.")

    frame = frame[frame["Date"].astype(str).str.match(MONTH_RE, na=False)].copy()
    frame["_year"] = (
        frame["Date"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    )
    frame = frame[frame["_year"] >= float(min_year)].drop(columns=["_year"])

    for column in frame.columns:
        if column in {"Actual Date", "Date"}:
            continue
        frame[column] = _coerce_numeric(frame[column])

    if "Actual Date" in frame.columns:
        sort_dates = pd.to_datetime(frame["Actual Date"], errors="coerce")
        frame = frame.assign(_sort_date=sort_dates)
        frame = frame.sort_values("_sort_date").drop(columns=["_sort_date"])
    elif "Date" in frame.columns:
        sort_dates = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.assign(_sort_date=sort_dates)
        frame = frame.sort_values("_sort_date").drop(columns=["_sort_date"])

    return frame.reset_index(drop=True)


def load_port_long_beach_teu_cache(path: Optional[Path] = None) -> pd.DataFrame:
    """Load cached Port of Long Beach TEU dataset if available."""
    csv_path = Path(path) if path else PORT_LB_CACHE_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def update_port_long_beach_teu_cache(
    *,
    min_year: int = 1995,
    timeout: float = 30.0,
    engine: str = "playwright",
    output_csv: Path | str = PORT_LB_CACHE_CSV,
) -> pd.DataFrame:
    """Refresh the Port of Long Beach TEU archive CSV."""
    output_path = Path(output_csv)
    frame = fetch_port_long_beach_teu_table(
        min_year=min_year, timeout=timeout, engine=engine
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    return frame


__all__ = [
    "fetch_port_long_beach_teu_table",
    "load_port_long_beach_teu_cache",
    "update_port_long_beach_teu_cache",
]
