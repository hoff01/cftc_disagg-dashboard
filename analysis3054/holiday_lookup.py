"""Country and financial market holiday detection utilities.

This module wraps :mod:`holidays` and :mod:`pycountry` to normalise free-form
country names into ISO 3166-1 alpha-2 codes and check whether given dates fall
on or near a public holiday. It also provides a helper for mapping countries to
financial market identifiers and checking market holidays.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional

import io

import holidays
import numpy as np
import pandas as pd
import pycountry
import requests

# 1. Static Overrides for common terms pycountry might miss or specific user preferences
MANUAL_OVERRIDES: Dict[str, str] = {
    "europe": "EU",
    "eurozone": "EU",
    "eu": "EU",
    "uk": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "england": "GB",
    "usa": "US",
    "us": "US",
    "america": "US",
    "united states": "US",
    "russia": "RU",
    "south korea": "KR",
    "north korea": "KP",
    "uae": "AE",
    "vietnam": "VN",
}

# 2. Financial Market Mapping (ISO 3166-1 alpha-2 -> ISO 10383 MIC)
ISO_TO_MARKET: Dict[str, str] = {
    "US": "XNYS",  # USA -> NYSE
    "GB": "IFEUA",  # UK -> ICE Futures Europe
    # Major European Economies -> ICE Futures Europe (Broad mapping)
    "DE": "IFEUA",
    "FR": "IFEUA",
    "IT": "IFEUA",
    "ES": "IFEUA",
    "NL": "IFEUA",
    "BE": "IFEUA",
    "CH": "IFEUA",
    "SE": "IFEUA",
    "NO": "IFEUA",
    "DK": "IFEUA",
    "FI": "IFEUA",
    "EU": "IFEUA",
    "AT": "IFEUA",
    "PT": "IFEUA",
    "GR": "IFEUA",
    "PL": "IFEUA",
    # Others
    "BR": "BVMF",  # Brazil -> B3
    "IN": "XNSE",  # India -> NSE
    "CA": "XTSE",  # Canada -> TSX
    "JP": "XJPX",  # Japan -> JPX
    "CN": "XSHG",  # China -> Shanghai
}

# Cache for S&P Global Platts holiday schedule downloads keyed by years tuple
_PLATTS_CACHE: Dict[tuple, pd.DataFrame] = {}


@lru_cache(maxsize=2048)
def resolve_iso_code(raw_input: Optional[str]) -> Optional[str]:
    """Resolve a country description to an ISO 3166-1 alpha-2 code.

    The resolver first applies manual overrides for common aliases, then tries
    :func:`pycountry.countries.lookup` for exact matches, and finally falls back
    to :func:`pycountry.countries.search_fuzzy` for fuzzy matching.

    Args:
        raw_input: Country name or code (e.g. "The US", "socialist republic of vietnam").

    Returns:
        Two-letter ISO code (e.g. "US") or ``None`` if no match is found.
    """

    if not raw_input or not isinstance(raw_input, str):
        return None

    clean = raw_input.strip().lower()

    if clean in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[clean]

    try:
        match = pycountry.countries.lookup(clean)
        return match.alpha_2
    except LookupError:
        pass

    try:
        matches = pycountry.countries.search_fuzzy(clean)
        if matches:
            return matches[0].alpha_2
    except (LookupError, AttributeError):
        pass

    return None


def get_market_code(iso_code: Optional[str]) -> Optional[str]:
    """Return the ISO 10383 Market Identifier Code (MIC) for a country code."""

    if not iso_code:
        return None
    return ISO_TO_MARKET.get(iso_code)


def _normalise_dates(series: pd.Series, date_col: str) -> pd.Series:
    dates = pd.to_datetime(series[date_col])
    return dates.dt.date


def _window_lookup(holiday_map, dt: date, window: int, specific: bool) -> str:
    if dt in holiday_map:
        return holiday_map.get(dt) if specific else "Yes"

    if window > 0:
        for offset in range(1, window + 1):
            for candidate in (dt - timedelta(days=offset), dt + timedelta(days=offset)):
                if candidate in holiday_map:
                    return holiday_map.get(candidate) if specific else "Yes"

    return "No"


def _normalise_identifiers(value) -> list[str]:
    """Normalise a potentially multi-valued identifier field to a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _looks_like_market_code(identifier: str) -> bool:
    """Heuristically determine if a string resembles a market MIC.

    This avoids treating arbitrary location names as market identifiers while
    still allowing direct MIC inputs (typically 3–6 uppercase alphanumeric
    characters, e.g. "XNYS", "XNAS", "ICE").
    """

    if not isinstance(identifier, str):
        return False

    stripped = identifier.strip()
    return stripped.isupper() and stripped.isalnum() and 3 <= len(stripped) <= 6


def is_holiday(
    df: pd.DataFrame,
    date_col: str,
    country_col: str,
    window: int = 3,
    specific: bool = True,
) -> pd.Series:
    """Check if each row's date falls on or near a public holiday.

    Args:
        df: Input dataframe.
        date_col: Column name containing dates.
        country_col: Column containing country names/codes or iterables of values
            to check (the first matching holiday wins).
        window: Number of days on either side to check for nearby holidays.
        specific: If ``True`` return the holiday name, otherwise "Yes".

    Returns:
        A :class:`pandas.Series` aligned to ``df`` with the holiday name, "Yes"/"No",
        or ``NaN`` when none of the identifiers resolve to a known country.
    """

    working_df = df.copy()
    working_df[date_col] = _normalise_dates(working_df, date_col)

    all_identifiers = working_df[country_col].apply(_normalise_identifiers)
    unique_countries = {
        identifier
        for identifiers in all_identifiers
        for identifier in identifiers
    }
    iso_map = {name: resolve_iso_code(name) for name in unique_countries}

    holiday_cache: Dict[str, Dict] = {}
    years = pd.to_datetime(working_df[date_col]).dt.year.unique()

    def _resolve_holiday(dt_val: date, identifiers) -> Optional[str]:
        iso_codes = [iso_map.get(name) for name in identifiers if iso_map.get(name)]
        if not iso_codes:
            return None

        for iso_code in iso_codes:
            if iso_code not in holiday_cache:
                try:
                    holiday_cache[iso_code] = holidays.country_holidays(
                        iso_code, years=years, language="en_US"
                    )
                except Exception:
                    holiday_cache[iso_code] = None

            holiday_map = holiday_cache.get(iso_code)
            if holiday_map:
                lookup = _window_lookup(holiday_map, dt_val, window=window, specific=specific)
                if lookup != "No":
                    return lookup
        return "No"

    results = pd.Series(np.nan, index=working_df.index, dtype=object)
    for idx, row in working_df.iterrows():
        identifiers = all_identifiers.loc[idx]
        lookup = _resolve_holiday(row[date_col], identifiers)
        if lookup is not None:
            results.at[idx] = lookup

    return results


def is_financial_holiday(
    df: pd.DataFrame,
    date_col: str,
    country_col: str,
    window: int = 3,
    specific: bool = True,
    include_half_days: bool = True,
) -> pd.Series:
    """Identify financial market holidays based on country information.

    Country labels are resolved to ISO codes and mapped to market MICs using
    :data:`ISO_TO_MARKET`. Direct market codes are accepted and multiple inputs
    per row are supported (the first matching holiday is returned). Rows where
    no market can be determined return ``NaN``.

    Args:
        df: Input dataframe.
        date_col: Column containing dates.
        country_col: Column containing country names, market codes, or iterables
            of those values to check in order.
        window: Number of days on either side to check for nearby holidays.
        specific: If ``True`` return the holiday name, otherwise "Yes".
        include_half_days: When ``False``, holidays containing "Half Day" or
            "Early Close" are treated as working days and return "No".

    Returns:
        A :class:`pandas.Series` aligned to ``df`` with the market holiday name
        or "No"/``NaN`` if not applicable.
    """

    working_df = df.copy()
    working_df[date_col] = _normalise_dates(working_df, date_col)

    all_identifiers = working_df[country_col].apply(_normalise_identifiers)
    unique_identifiers = {
        identifier
        for identifiers in all_identifiers
        for identifier in identifiers
    }

    market_map: Dict[str, str] = {}
    for name in unique_identifiers:
        iso = resolve_iso_code(name)
        mic = get_market_code(iso) if iso else None
        if mic:
            market_map[name] = mic
        elif _looks_like_market_code(name):
            # Accept direct market codes when they resemble MICs (e.g. XNYS)
            market_map[name] = name

    market_cache: Dict[str, Dict] = {}
    years = pd.to_datetime(working_df[date_col]).dt.year.unique()

    def _check_financial(dt_val: date, identifiers) -> Optional[str]:
        market_codes = [market_map.get(name) for name in identifiers if market_map.get(name)]
        if not market_codes:
            return None

        found_calendar = False

        for market_code in market_codes:
            if market_code not in market_cache:
                try:
                    market_cache[market_code] = holidays.financial_holidays(market_code, years=years)
                except Exception:
                    market_cache[market_code] = None

            fin_hols = market_cache.get(market_code)
            if not fin_hols:
                continue

            found_calendar = True
            hol_name = _window_lookup(fin_hols, dt_val, window=window, specific=True)
            if hol_name != "No":
                if not include_half_days and ("Half Day" in hol_name or "Early Close" in hol_name):
                    continue
                return hol_name if specific else "Yes"

        if not found_calendar:
            return None
        return "No"

    results = pd.Series(np.nan, index=working_df.index, dtype=object)
    for idx, row in working_df.iterrows():
        identifiers = all_identifiers.loc[idx]
        lookup = _check_financial(row[date_col], identifiers)
        if lookup is not None:
            results.at[idx] = lookup

    return results


def _download_and_parse_platts(years_to_check=None) -> pd.DataFrame:
    """Download and parse S&P Global Platts Excel schedules into a dataframe.

    Each Excel file follows a repeating structure that includes at least three
    columns: a date column (often in the third position), an ``Exchanges``
    column listing comma-separated exchange identifiers, and a ``Holiday``
    description. This helper tolerates missing years and parse failures and
    always returns a dataframe with the canonical ``Date``/``Exchanges``/
    ``Holiday`` columns for downstream consumption.

    Years outside the available range are skipped, and failures yield an empty
    dataframe so callers remain resilient to network issues.
    """

    if years_to_check is None:
        current_year = datetime.now().year
        years_to_check = range(current_year - 3, current_year + 5)

    all_dfs = []
    base_url = (
        "https://www.spglobal.com/content/dam/spglobal/ci/en/documents/platts/en/"
        "our-methodology/holiday-schedules/holsked{}.xlsx"
    )

    for year in years_to_check:
        url = base_url.format(year)
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue

            with io.BytesIO(response.content) as buffer:
                temp_df = pd.read_excel(buffer)

            cols = list(temp_df.columns)
            if len(cols) > 2:
                temp_df.rename(columns={cols[2]: "Date"}, inplace=True)

            col_map = {c: c for c in temp_df.columns}
            for col in temp_df.columns:
                if "exchange" in str(col).lower():
                    col_map[col] = "Exchanges"
                if "holiday" in str(col).lower():
                    col_map[col] = "Holiday"
            temp_df.rename(columns=col_map, inplace=True)

            if {"Date", "Exchanges", "Holiday"}.issubset(temp_df.columns):
                all_dfs.append(temp_df[["Date", "Exchanges", "Holiday"]])

        except Exception:
            continue

    if not all_dfs:
        return pd.DataFrame(columns=["Date", "Exchanges", "Holiday"])

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df["Date"] = pd.to_datetime(master_df["Date"], errors="coerce")
    master_df.dropna(subset=["Date"], inplace=True)
    master_df.sort_values("Date", inplace=True)
    master_df["Exchanges"] = master_df["Exchanges"].astype(str)
    master_df["Holiday"] = master_df["Holiday"].astype(str)
    return master_df


def _compute_platts_years(dates: pd.Series, window: int) -> list[int]:
    """Compute the minimal set of years to download for Platts lookups."""

    normalized_dates = pd.to_datetime(dates)
    if normalized_dates.empty:
        current_year = datetime.now().year
        return [current_year]

    min_date = (normalized_dates.min() - pd.Timedelta(days=window)).to_pydatetime()
    max_date = (normalized_dates.max() + pd.Timedelta(days=window)).to_pydatetime()
    return list(range(min_date.year, max_date.year + 1))


def is_platts_holiday(
    df: pd.DataFrame,
    date_col: str,
    window: int = 0,
    exchanges: Optional[list[str]] = None,
    refresh: bool = False,
) -> pd.Series:
    """Check dates against the S&P Global Platts holiday schedule.

    The lookup is designed to be robust and fast:

    * Downloads are scoped to the years required by ``df[date_col]`` plus the
      provided ``window`` to avoid unnecessary network requests.
    * Parsed schedules are cached per year tuple and reused across calls unless
      ``refresh`` is ``True``.
    * When required data is unavailable—because downloads fail, exchanges are
      unmatched, or queried years are missing—the function returns ``NaN`` to
      signal the absence of coverage instead of misreporting "No".
    * When data is available, the nearest holiday within ``window`` days is
      returned, with "No" indicating a confirmed non-holiday.

    Args:
        df: Input dataframe.
        date_col: Name of the column containing dates.
        window: Days +/- to check for a holiday match. ``0`` is an exact match.
        exchanges: Exchange filters (e.g. ``["CME", "ICE"]``). All listed
            exchanges are considered, and the first matching holiday is returned.
            Defaults to ``["CME", "ICE"]``.
        refresh: Force a re-download of the schedules even if cached.

    Returns:
        A :class:`pandas.Series` aligned to ``df`` containing the holiday name,
        "No" when the date is confirmed as a working day, or ``NaN`` when no
        schedule data is available for the requested years/exchanges.
    """

    global _PLATTS_CACHE

    if exchanges is None:
        exchanges = ["CME", "ICE"]

    # Compute the minimal set of years to query and leverage a keyed cache.
    years_to_check = tuple(_compute_platts_years(df[date_col], window))

    if refresh or years_to_check not in _PLATTS_CACHE:
        _PLATTS_CACHE[years_to_check] = _download_and_parse_platts(years_to_check)

    platts = _PLATTS_CACHE.get(years_to_check)
    if platts is None or platts.empty:
        return pd.Series(np.nan, index=df.index)

    pattern = "|".join(exchanges)
    mask = platts["Exchanges"].str.contains(pattern, case=False, na=False)
    filtered = platts.loc[mask].copy()

    if filtered.empty:
        return pd.Series(np.nan, index=df.index)

    df_temp = df[[date_col]].copy()
    df_temp["__orig_idx"] = df_temp.index
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp.sort_values(date_col, inplace=True)

    filtered.sort_values("Date", inplace=True)

    if window == 0:
        merged = pd.merge(
            df_temp,
            filtered[["Date", "Holiday"]],
            left_on=date_col,
            right_on="Date",
            how="left",
        )
    else:
        merged = pd.merge_asof(
            df_temp,
            filtered[["Date", "Holiday"]],
            left_on=date_col,
            right_on="Date",
            tolerance=pd.Timedelta(days=window),
            direction="nearest",
        )

    merged.index = merged["__orig_idx"]
    merged.sort_index(inplace=True)

    available_years = set(filtered["Date"].dt.year.unique())
    results = merged["Holiday"].copy()
    missing_year_mask = ~df_temp.set_index("__orig_idx")[date_col].dt.year.isin(available_years)

    results.fillna("No", inplace=True)
    # Mark rows where the queried year is outside available schedules as NaN to
    # indicate missing coverage instead of a confirmed working day.
    results.loc[missing_year_mask] = np.nan

    return results


__all__ = [
    "resolve_iso_code",
    "get_market_code",
    "is_holiday",
    "is_financial_holiday",
    "is_platts_holiday",
    "MANUAL_OVERRIDES",
    "ISO_TO_MARKET",
]
