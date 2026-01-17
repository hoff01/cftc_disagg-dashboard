"""Fetch EIA series data (generic helper)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import requests

DEFAULT_API_ENV = "EIA_API_KEY"


@dataclass(frozen=True)
class EIASeries:
    series_id: str
    data: pd.DataFrame


class EIADataError(Exception):
    """Raised when EIA data cannot be retrieved."""


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    env_key = os.getenv(DEFAULT_API_ENV, "").strip()
    return env_key or None


def fetch_eia_series(
    *,
    series_id: str,
    api_key: Optional[str] = None,
    timeout: float = 20.0,
) -> EIASeries:
    """Fetch a single EIA series via the v2 seriesid API."""

    api_key = _resolve_api_key(api_key)
    if not api_key:
        raise EIADataError("EIA_API_KEY not set; cannot fetch EIA series.")
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"api_key": api_key}
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise EIADataError(f"EIA API request failed: {exc}") from exc
    data = payload.get("response", {}).get("data")
    if not data:
        raise EIADataError(f"No data returned for {series_id}.")

    df = pd.DataFrame(data)
    if "period" not in df.columns or "value" not in df.columns:
        raise EIADataError(f"Unexpected response format for {series_id}.")
    df["date"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return EIASeries(series_id=series_id, data=df[["date", "value"]])


def fetch_eia_series_bulk(
    series_map: Dict[str, str],
    *,
    api_key: Optional[str] = None,
    timeout: float = 20.0,
) -> Tuple[pd.DataFrame, list[str]]:
    """Fetch multiple EIA series. Returns dataframe + list of missing series keys."""

    if not series_map:
        return pd.DataFrame(), []
    series: Dict[str, pd.Series] = {}
    missing: list[str] = []
    for label, series_id in series_map.items():
        try:
            result = fetch_eia_series(series_id=series_id, api_key=api_key, timeout=timeout)
            frame = result.data.set_index("date")["value"].sort_index()
            series[label] = frame
        except Exception:
            missing.append(label)

    if not series:
        return pd.DataFrame(), missing
    df = pd.concat(series, axis=1).sort_index()
    return df, missing


__all__ = ["EIASeries", "EIADataError", "fetch_eia_series", "fetch_eia_series_bulk"]
