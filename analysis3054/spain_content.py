"""Spain (CORES) consumption and balance data extracts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import io
import re

import pandas as pd
import requests

SPAIN_OIL_PRODUCTS_URL = (
    "https://www.cores.es/sites/default/files/archivos/estadisticas/oil-products-consumption.xlsx"
)
SPAIN_CRUDE_OIL_URL = (
    "https://www.cores.es/sites/default/files/archivos/estadisticas/crude-oil-balance-and-refinery-output.xlsx"
)
SPAIN_GAS_CONSUMPTION_URL = (
    "https://www.cores.es/sites/default/files/archivos/estadisticas/gas-consumption.xlsx"
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "europe_energy"
SPAIN_PREFIX = "Spain_content"


class SpainContentError(Exception):
    """Raised when Spain content extraction fails."""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _download_excel(url: str, *, timeout: float = 60.0) -> pd.ExcelFile:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return pd.ExcelFile(io.BytesIO(response.content), engine="openpyxl")


def _find_sheet(excel: pd.ExcelFile, *, required: list[str]) -> str:
    tokens = [_normalize(token) for token in required]
    for name in excel.sheet_names:
        normalized = _normalize(name)
        if all(token in normalized for token in tokens):
            return name
    raise SpainContentError(f"Sheet not found: {required}")


def fetch_spain_content_tables(*, timeout: float = 60.0) -> dict[str, pd.DataFrame]:
    """Download and extract CORES tables into dataframes."""
    oil_excel = _download_excel(SPAIN_OIL_PRODUCTS_URL, timeout=timeout)
    crude_excel = _download_excel(SPAIN_CRUDE_OIL_URL, timeout=timeout)
    gas_excel = _download_excel(SPAIN_GAS_CONSUMPTION_URL, timeout=timeout)

    oil_sheet = _find_sheet(oil_excel, required=["All"])
    crude_sheet = _find_sheet(crude_excel, required=["Crude", "Refinery", "output"])
    gas_market_sheet = _find_sheet(gas_excel, required=["Consumption", "market"])
    gas_pressure_sheet = _find_sheet(gas_excel, required=["pressure", "bracket"])

    tables = {
        "oil_products_all": oil_excel.parse(sheet_name=oil_sheet),
        "crude_oil_balance_refinery_output": crude_excel.parse(sheet_name=crude_sheet),
        "gas_consumption_by_market": gas_excel.parse(sheet_name=gas_market_sheet),
        "gas_consumption_by_pressure_bracket": gas_excel.parse(sheet_name=gas_pressure_sheet),
    }
    return tables


def update_spain_content_cache(
    *,
    output_dir: Path | str = DATA_DIR,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """Refresh Spain content CSVs."""
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    tables = fetch_spain_content_tables(timeout=timeout)
    outputs: dict[str, Path] = {}
    for key, df in tables.items():
        filename = f"{SPAIN_PREFIX}_{key}.csv"
        path = output_base / filename
        df.to_csv(path, index=False)
        outputs[filename] = path
    return outputs


def load_spain_content_csv(name: str, *, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a cached Spain content CSV by filename."""
    base = Path(base_dir) if base_dir else DATA_DIR
    path = base / name
    if not path.exists():
        raise SpainContentError(f"Spain content CSV not found: {path}")
    return pd.read_csv(path)


__all__ = [
    "fetch_spain_content_tables",
    "update_spain_content_cache",
    "load_spain_content_csv",
]
