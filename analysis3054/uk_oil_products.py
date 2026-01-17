"""UK oil products (Energy Trends Section 3) monthly data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import io
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

UK_OIL_PRODUCTS_PAGE_URL = (
    "https://www.gov.uk/government/statistics/oil-and-oil-products-section-3-energy-trends"
)
UK_PREFIX = "UK_content"

DATA_DIR = Path(__file__).resolve().parent / "data" / "europe_energy"


class UKOilProductsError(Exception):
    """Raised when UK oil products extraction fails."""


def _normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text.strip().lower())


def _download_excel(url: str, *, timeout: float = 60.0) -> pd.ExcelFile:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return pd.ExcelFile(io.BytesIO(response.content), engine="openpyxl")


def fetch_uk_oil_products_links(
    *,
    page_url: str = UK_OIL_PRODUCTS_PAGE_URL,
    timeout: float = 30.0,
) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(page_url, timeout=timeout, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    et_312 = None
    et_313 = None

    for anchor in soup.find_all("a", href=True):
        text = _normalize(anchor.get_text(" ", strip=True))
        href = anchor["href"]
        if "et 3.12" in text and "monthly" in text:
            et_312 = href
        if "et 3.13" in text and "monthly" in text:
            et_313 = href

    if not et_312 or not et_313:
        raise UKOilProductsError("Could not locate ET 3.12 or ET 3.13 monthly links.")

    et_312 = requests.compat.urljoin(page_url, et_312)
    et_313 = requests.compat.urljoin(page_url, et_313)
    return {"et_3_12": et_312, "et_3_13": et_313}


def fetch_uk_oil_products_tables(*, timeout: float = 60.0) -> dict[str, pd.DataFrame]:
    """Download and extract the 'Month' sheet for ET 3.12 and ET 3.13."""
    links = fetch_uk_oil_products_links(timeout=timeout)

    et_312_excel = _download_excel(links["et_3_12"], timeout=timeout)
    et_313_excel = _download_excel(links["et_3_13"], timeout=timeout)

    if "Month" not in et_312_excel.sheet_names:
        raise UKOilProductsError("ET 3.12 workbook missing 'Month' sheet.")
    if "Month" not in et_313_excel.sheet_names:
        raise UKOilProductsError("ET 3.13 workbook missing 'Month' sheet.")

    return {
        "et_3_12_month": et_312_excel.parse(sheet_name="Month"),
        "et_3_13_month": et_313_excel.parse(sheet_name="Month"),
    }


def update_uk_oil_products_cache(
    *,
    output_dir: Path | str = DATA_DIR,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """Refresh UK oil products monthly CSVs."""
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    tables = fetch_uk_oil_products_tables(timeout=timeout)
    outputs: dict[str, Path] = {}
    for key, df in tables.items():
        filename = f"{UK_PREFIX}_{key}.csv"
        path = output_base / filename
        df.to_csv(path, index=False)
        outputs[filename] = path
    return outputs


def load_uk_oil_products_csv(name: str, *, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a cached UK oil products CSV by filename."""
    base = Path(base_dir) if base_dir else DATA_DIR
    path = base / name
    if not path.exists():
        raise UKOilProductsError(f"UK oil products CSV not found: {path}")
    return pd.read_csv(path)


__all__ = [
    "fetch_uk_oil_products_links",
    "fetch_uk_oil_products_tables",
    "update_uk_oil_products_cache",
    "load_uk_oil_products_csv",
]
