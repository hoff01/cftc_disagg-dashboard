"""Fetch EIA distillate fuel oil product supplied series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .eia_series import EIADataError, EIASeries, fetch_eia_series, fetch_eia_series_bulk

DEFAULT_SERIES_ID = "PET.MDIUPUS2.M"

DISTILLATE_SERIES_MAP: Dict[str, str] = {
    "us": "PET.MDIUPUS2.M",
    "padd1": "PET.MDIUPP12.M",
    "padd2": "PET.MDIUPP22.M",
    "padd3": "PET.MDIUPP32.M",
    "padd4": "PET.MDIUPP42.M",
    "padd5": "PET.MDIUPP52.M",
}

DISTILLATE_SERIES_META: Dict[str, Dict[str, str]] = {
    "us": {"label": "Total US", "series_id": DISTILLATE_SERIES_MAP["us"]},
    "padd1": {"label": "PADD 1", "series_id": DISTILLATE_SERIES_MAP["padd1"]},
    "padd2": {"label": "PADD 2", "series_id": DISTILLATE_SERIES_MAP["padd2"]},
    "padd3": {"label": "PADD 3", "series_id": DISTILLATE_SERIES_MAP["padd3"]},
    "padd4": {"label": "PADD 4", "series_id": DISTILLATE_SERIES_MAP["padd4"]},
    "padd5": {"label": "PADD 5", "series_id": DISTILLATE_SERIES_MAP["padd5"]},
}

DISTILLATE_COVARIATES_MAP: Dict[str, str] = {
    "stocks_us": "PET.MDISTUS1.M",
    "production_us": "PET.MDIOPUS2.M",
    "imports_us": "PET.MDIIMUS2.M",
    "exports_us": "PET.MDIEXUS2.M",
    "refinery_util_us": "PET.MOPUEUS2.M",
    "retail_price_us": "PET.EMD_EPD2D_PTE_NUS_DPG.M",
    "jet_fuel_supplied_us": "PET.MJTUPUS2.M",
    "crude_runs_us": "PET.MCRRIUS2.M",
    "operable_capacity_us": "PET.MOCLEUS2.M",
    "residual_fuel_supplied_us": "PET.MFOUPUS2.M",
    "propane_supplied_us": "PET.MPPUPUS2.M",
    "kerosene_supplied_us": "PET.MKSUPUS2.M",
}

DISTILLATE_COVARIATES_META: Dict[str, Dict[str, str]] = {
    "stocks_us": {
        "label": "Distillate Stocks (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["stocks_us"],
    },
    "production_us": {
        "label": "Distillate Production (Refiner Net)",
        "series_id": DISTILLATE_COVARIATES_MAP["production_us"],
    },
    "imports_us": {
        "label": "Distillate Imports (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["imports_us"],
    },
    "exports_us": {
        "label": "Distillate Exports (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["exports_us"],
    },
    "refinery_util_us": {
        "label": "Refinery Utilization (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["refinery_util_us"],
    },
    "retail_price_us": {
        "label": "Diesel Retail Price (US Avg)",
        "series_id": DISTILLATE_COVARIATES_MAP["retail_price_us"],
    },
    "jet_fuel_supplied_us": {
        "label": "Jet Fuel Product Supplied (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["jet_fuel_supplied_us"],
    },
    "crude_runs_us": {
        "label": "Refinery Gross Crude Oil Inputs (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["crude_runs_us"],
    },
    "operable_capacity_us": {
        "label": "Refinery Operable Capacity (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["operable_capacity_us"],
    },
    "residual_fuel_supplied_us": {
        "label": "Residual Fuel Oil Product Supplied (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["residual_fuel_supplied_us"],
    },
    "propane_supplied_us": {
        "label": "Propane/Propylene Product Supplied (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["propane_supplied_us"],
    },
    "kerosene_supplied_us": {
        "label": "Kerosene Product Supplied (Total US)",
        "series_id": DISTILLATE_COVARIATES_MAP["kerosene_supplied_us"],
    },
}


@dataclass(frozen=True)
class DistillateSeries:
    series_id: str
    data: pd.DataFrame


def fetch_distillate_product_supplied(
    *,
    api_key: Optional[str] = None,
    series_id: Optional[str] = None,
    timeout: float = 20.0,
) -> DistillateSeries:
    """Fetch EIA distillate product supplied series via the EIA v2 API."""
    series_id = series_id or DEFAULT_SERIES_ID
    result = fetch_eia_series(series_id=series_id, api_key=api_key, timeout=timeout)
    return DistillateSeries(series_id=result.series_id, data=result.data)


def fetch_distillate_series_bulk(
    *,
    api_key: Optional[str] = None,
    series_map: Optional[Dict[str, str]] = None,
    timeout: float = 20.0,
) -> Tuple[pd.DataFrame, list[str]]:
    """Fetch multiple distillate series (US + PADDs)."""

    series_map = series_map or DISTILLATE_SERIES_MAP
    return fetch_eia_series_bulk(series_map, api_key=api_key, timeout=timeout)


__all__ = [
    "DISTILLATE_SERIES_MAP",
    "DISTILLATE_SERIES_META",
    "DISTILLATE_COVARIATES_MAP",
    "DISTILLATE_COVARIATES_META",
    "fetch_distillate_product_supplied",
    "fetch_distillate_series_bulk",
    "DistillateSeries",
    "EIADataError",
]
