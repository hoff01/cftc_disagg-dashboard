"""Refinery name normalization helpers."""

from __future__ import annotations

import re
from typing import Dict

import pandas as pd

LA_REFINERY_NAME_MAP_RAW: Dict[str, str] = {
    "ALON REF KROTZ SPRINGS-KROTZ SGS REFY": "Delek Krotz Springs",
    "CALCASIEU REFINING CO - CALCASIEU REFY": "Calcasieu Refining",
    "CALUMET LUBRICANTS CO-COTTON VALLEY REFY": "Calumet Cotton Valley",
    "CALUMET PRINCETON REFIN - PRINCETON REFY": "Calumet Princeton",
    "CALUMET SHREVEPORT REFINING-SHREV REFY": "Calumet Shreveport",
    "CANAL REFINING CO-CHURCH POINT REF.": "Canal Refining",
    "CHALMETTE REFINING LLC - CHALMETTE REFY": "PBF Chalmette",
    "CITGO PETRO CORP--LAKE CHARLES REFY": "CITGO Lake Charles",
    "EQUILON ENTEPRISES LLC-ST ROSE REFINERY": "Shell St. Rose",
    "EQUILON ENTERPRISES LLC - NORCO REFINERY": "Shell Norco",
    "EQUILON ENTERPRISES LLC-CONVENT REFINERY": "Shell Convent",
    "EXXONMOBIL FUELS & LUBRS-BATON ROUGE RFY": "ExxonMobil Baton Rouge",
    "MARATHON PETR CO LP - GARYVILLE REFINERY": "Marathon Garyville",
    "PHILLIPS 66 COMPANY - LAKE CHARLES REF": "Phillips 66 Lake Charles",
    "PHILLIPS 66 COMPANY-ALLIANCE REFINERY": "Phillips 66 Alliance",
    "PLACID REFINING CO - PORT ALLEN  REFY": "Placid Refining",
    "SHELL CHEMICAL LP- NORCO REFINERY": "Shell Norco",
    "VALERO REF'G N.O.-VALERO ST CHARLES RFY": "Valero St. Charles",
    "VALERO REFINING MERAUX LLC- MERAUX REFY": "Valero Meraux",
}


def normalize_la_refinery_name_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


LA_REFINERY_NAME_MAP: Dict[str, str] = {
    normalize_la_refinery_name_key(key): value
    for key, value in LA_REFINERY_NAME_MAP_RAW.items()
}


def apply_la_refinery_name_map(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    keys = series.fillna("").astype(str).map(normalize_la_refinery_name_key)
    mapped = keys.map(LA_REFINERY_NAME_MAP)
    return mapped.where(mapped.notna() & (mapped.str.len() > 0), series)


def apply_la_refinery_name_map_to_df(
    df: pd.DataFrame,
    *,
    name_col: str = "Refinery Name",
) -> pd.DataFrame:
    if df is None or df.empty or name_col not in df.columns:
        return df
    updated = df.copy()
    updated[name_col] = apply_la_refinery_name_map(updated[name_col])
    return updated


__all__ = [
    "LA_REFINERY_NAME_MAP_RAW",
    "LA_REFINERY_NAME_MAP",
    "normalize_la_refinery_name_key",
    "apply_la_refinery_name_map",
    "apply_la_refinery_name_map_to_df",
]
