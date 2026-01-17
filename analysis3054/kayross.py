"""Kayross and DTN data processing for dashboards."""

from __future__ import annotations

import pandas as pd
from analysis3054.utils import get_padd

DTN_VOLUME_DIVISOR = 42000
DEFAULT_HISTORY_EXCLUDE = {2020, 2021}
DEFAULT_HISTORY_COUNT = 5
DEFAULT_LINE_START_YEAR = 2018

DTN_DIESEL_GRADES = {
    "#1 diesel",
    "#2 diesel",
    "no. 1 diesel",
    "no. 1 diesel fuel",
    "no. 2 diesel fuel",
}
DTN_GAS_GRADES = {
    "premium",
    "regular",
}

STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
STATE_NAME_LOOKUP = {name.lower(): name for name in STATE_ABBR_TO_NAME.values()}
STATE_ABBR_LOOKUP = {abbr.lower(): name for abbr, name in STATE_ABBR_TO_NAME.items()}


def normalize_state_name(value: str) -> str:
    if value is None:
        return ""
    key = str(value).strip()
    if not key:
        return ""
    lower = key.lower()
    return STATE_ABBR_LOOKUP.get(lower) or STATE_NAME_LOOKUP.get(lower) or key


def _read_csv(path: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False)


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def load_kayross_us(path: str) -> pd.DataFrame:
    """Load and process Kayross US data."""
    columns = ["VALUE_DATE", "VALUE_DIESEL", "VALUE_GASOLINE", "REGION"]
    df = _read_csv(path, columns)
    if not _has_columns(df, ["VALUE_DATE", "REGION"]):
        return _empty_frame(["date", "VALUE_DIESEL", "VALUE_GASOLINE", "REGION", "padd_specific", "padd_general"])
    df["date"] = pd.to_datetime(df["VALUE_DATE"])
    
    # Map States to PADDs
    # Note: Kayross 'REGION' seems to map to US States based on description
    df["padd_specific"] = get_padd(df["REGION"], specific=True)
    df["padd_general"] = get_padd(df["REGION"], specific=False)
    
    return df


def load_kayross_eu(path: str) -> pd.DataFrame:
    """Load and process Kayross EU data."""
    columns = ["VALUE_DATE", "COUNTRY", "VALUE_DIESEL_CONSUMPTION", "VALUE_GASOLINE_CONSUMPTION"]
    df = _read_csv(path, columns)
    if not _has_columns(df, ["VALUE_DATE", "COUNTRY"]):
        return _empty_frame(["date", "VALUE_DIESEL_CONSUMPTION", "VALUE_GASOLINE_CONSUMPTION", "COUNTRY"])
    df["date"] = pd.to_datetime(df["VALUE_DATE"])
    return df

def load_dtn_padd_data(path: str) -> pd.DataFrame:
    """Load and process DTN PADD daily data."""
    columns = ["effectiveDateTime", "region", "grade", "sumNetVolume", "coverageFactor"]
    df = _read_csv(path, columns)
    if not _has_columns(df, ["effectiveDateTime", "region", "grade", "sumNetVolume"]):
        return _empty_frame(["date", "region", "grade", "sumNetVolume", "coverageFactor", "padd_specific", "padd_general"])
    df["date"] = pd.to_datetime(df["effectiveDateTime"]).dt.normalize()

    padd_map = {
        "1A": "PADD 1-Northeast",
        "1B": "PADD 1-Northeast",
        "1C": "PADD 1-Southeast",
        "2": "PADD 2",
        "3": "PADD 3",
        "4": "PADD 4",
        "5": "PADD 5",
        "USA": "US",
    }
    df["padd_specific"] = df["region"].map(padd_map)

    general_map = {
        "1A": "PADD 1",
        "1B": "PADD 1",
        "1C": "PADD 1",
        "2": "PADD 2",
        "3": "PADD 3",
        "4": "PADD 4",
        "5": "PADD 5",
        "USA": "US",
    }
    df["padd_general"] = df["region"].map(general_map)
    return df


def load_dtn_rack_data(path: str) -> pd.DataFrame:
    """Load and process DTN rack-level daily data."""
    columns = ["effectiveDateTime", "rackId", "rackName", "state", "region", "grade", "sumNetVolume"]
    df = _read_csv(path, columns)
    if not _has_columns(df, ["effectiveDateTime", "state", "grade", "sumNetVolume"]):
        return _empty_frame(
            [
                "date",
                "rackId",
                "rackName",
                "state",
                "region",
                "grade",
                "sumNetVolume",
                "state_name",
                "padd_specific",
                "padd_general",
            ]
        )
    df["date"] = pd.to_datetime(df["effectiveDateTime"]).dt.normalize()
    df["state"] = df["state"].astype(str).str.strip()
    df["state_name"] = df["state"].apply(normalize_state_name)
    df["padd_specific"] = get_padd(df["state"], specific=True)
    df["padd_general"] = get_padd(df["state"], specific=False)
    return df


def load_dtn_data(path: str) -> pd.DataFrame:
    """Backward-compatible loader for DTN PADD data."""
    return load_dtn_padd_data(path)

def process_consumption_data(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: dict[str, str],
    rolling_window: int = 7
) -> pd.DataFrame:
    """
    Aggregate and smooth consumption data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Columns to group by (e.g., ['padd_specific']).
    metric_cols : dict[str, str]
        Mapping of output metric name to input column name.
        e.g., {'diesel': 'VALUE_DIESEL', 'gasoline': 'VALUE_GASOLINE'}
    rolling_window : int
        Window size for rolling mean.
        
    Returns
    -------
    pd.DataFrame
        Processed dataframe with date index, group columns, and smoothed metrics.
    """
    required = ["date"] + group_cols + list(metric_cols.values())
    if not _has_columns(df, required):
        return _empty_frame(["date"] + group_cols + list(metric_cols.keys()))

    agg_dict = {col: "sum" for col in metric_cols.values()}
    daily = df.groupby(["date"] + group_cols)[list(metric_cols.values())].agg(agg_dict).reset_index()
    
    # Pivot to get full time series for each group to ensure continuity for rolling
    results = None
    
    for out_name, in_name in metric_cols.items():
        pivoted = daily.pivot(index="date", columns=group_cols, values=in_name)
        smoothed = pivoted.rolling(window=rolling_window, min_periods=1).mean()
        melted = smoothed.melt(ignore_index=False, value_name=out_name).reset_index()
        
        if results is None:
            results = melted
        else:
            results = pd.merge(results, melted, on=["date"] + group_cols, how="outer")
            
    return results

def process_dtn_metrics(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    *,
    apply_coverage: bool = True,
) -> pd.DataFrame:
    """Aggregate DTN data into diesel/gasoline metrics."""
    if not _has_columns(df, ["date", "grade", "sumNetVolume"]):
        base_cols = group_cols if group_cols is not None else ["date", "region", "padd_specific", "padd_general"]
        return _empty_frame(base_cols + ["value_diesel", "value_gasoline"])

    grade = df["grade"].astype(str).str.strip().str.lower()
    df["is_diesel"] = grade.isin(DTN_DIESEL_GRADES)
    df["is_gasoline"] = grade.isin(DTN_GAS_GRADES)

    volume = pd.to_numeric(df["sumNetVolume"], errors="coerce")
    if apply_coverage and "coverageFactor" in df.columns:
        coverage = pd.to_numeric(df["coverageFactor"], errors="coerce")
        if coverage.notna().any():
            volume = volume * coverage.fillna(1.0)
    volume = volume / DTN_VOLUME_DIVISOR
    df["volume_kbd"] = volume
    
    if group_cols is None:
        group_cols = ["date", "region", "padd_specific", "padd_general"]
    
    diesel = (
        df[df["is_diesel"]]
        .groupby(group_cols)["volume_kbd"]
        .sum()
        .reset_index()
        .rename(columns={"volume_kbd": "value_diesel"})
    )
    gas = (
        df[df["is_gasoline"]]
        .groupby(group_cols)["volume_kbd"]
        .sum()
        .reset_index()
        .rename(columns={"volume_kbd": "value_gasoline"})
    )
    
    merged = pd.merge(diesel, gas, on=group_cols, how="outer").fillna(0)
    return merged

def _select_history_years(
    all_years: list[int],
    current_year: int,
    *,
    count: int = DEFAULT_HISTORY_COUNT,
    exclude_years: set[int] | None = None,
) -> list[int]:
    exclude = exclude_years or set()
    filtered = [y for y in all_years if y < current_year and y not in exclude]
    return sorted(filtered)[-count:]


def compute_historical_stats(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    *,
    include_years: list[int] | None = None,
    exclude_years: list[int] | None = None,
) -> pd.DataFrame:
    """Compute min, max, avg for each day of year based on historical data."""
    if include_years is not None:
        hist = df[df[date_col].dt.year.isin(include_years)].copy()
    elif exclude_years is not None:
        hist = df[~df[date_col].dt.year.isin(exclude_years)].copy()
    else:
        hist = df.copy()
    if hist.empty:
        return pd.DataFrame()
        
    # Standardize to year 2000 for aggregation
    # Handle leap years: 2000 is a leap year. If history has no Feb 29, it will be NaN.
    def to_common_date(d):
        try:
            return d.replace(year=2000)
        except ValueError:
            return d.replace(year=2000, day=28) # Fallback

    hist["common_date"] = hist[date_col].apply(to_common_date)
    
    # Group by common_date
    stats = hist.groupby("common_date")[value_col].agg(["min", "max", "mean"]).reset_index()
    stats.columns = ["common_date", "hist_min", "hist_max", "hist_avg"]
    stats = stats.sort_values("common_date")
    leap_mask = (stats["common_date"].dt.month == 2) & (stats["common_date"].dt.day == 29)
    if leap_mask.any():
        feb_28 = stats[stats["common_date"] == pd.Timestamp(year=2000, month=2, day=28)]
        mar_01 = stats[stats["common_date"] == pd.Timestamp(year=2000, month=3, day=1)]
        if not feb_28.empty and not mar_01.empty:
            for col in ["hist_min", "hist_max", "hist_avg"]:
                stats.loc[leap_mask, col] = (float(feb_28.iloc[0][col]) + float(mar_01.iloc[0][col])) / 2
        elif not feb_28.empty:
            for col in ["hist_min", "hist_max", "hist_avg"]:
                stats.loc[leap_mask, col] = float(feb_28.iloc[0][col])
        elif not mar_01.empty:
            for col in ["hist_min", "hist_max", "hist_avg"]:
                stats.loc[leap_mask, col] = float(mar_01.iloc[0][col])
        else:
            stats = stats[~leap_mask]
    return stats

def prepare_seasonality_payload(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_val: str,
    country: str = "US",
    *,
    history_years: list[int] | None = None,
    history_count: int = DEFAULT_HISTORY_COUNT,
    exclude_history_years: set[int] | None = None,
    include_years: list[int] | None = None,
    line_start_year: int = DEFAULT_LINE_START_YEAR,
    include_yoy: bool = False,
) -> dict:
    """Prepare data for a seasonality plot with historical bands and holidays."""
    if df.empty:
        return {"lines": [], "bands": {}, "history_years": [], "current_year": None, "prior_year": None}

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[value_col].notna()]
    if df.empty:
        return {"lines": [], "bands": {}, "history_years": [], "current_year": None, "prior_year": None}

    df["year"] = df[date_col].dt.year

    def to_common_date(d: pd.Timestamp) -> pd.Timestamp:
        try:
            return d.replace(year=2000)
        except ValueError:
            return d.replace(year=2000, day=28)

    df["common_date"] = df[date_col].apply(to_common_date)

    current_year = int(df["year"].max())
    prior_year = current_year - 1 if (current_year - 1) in df["year"].unique() else None

    all_years = sorted(int(y) for y in df["year"].unique())
    history_exclude = set(exclude_history_years or DEFAULT_HISTORY_EXCLUDE)
    if history_years is None:
        history_years = _select_history_years(
            all_years,
            current_year,
            count=history_count,
            exclude_years=history_exclude,
        )

    bands_data = {}
    if history_years:
        stats = compute_historical_stats(df, value_col, date_col, include_years=history_years)
        if not stats.empty:
            def _series_to_list(series: pd.Series) -> list[float | None]:
                return [None if pd.isna(val) else float(val) for val in series.tolist()]

            bands_data = {
                "x": stats["common_date"].dt.strftime("%Y-%m-%d").tolist(),
                "min": _series_to_list(stats["hist_min"]),
                "max": _series_to_list(stats["hist_max"]),
                "avg": _series_to_list(stats["hist_avg"]),
            }

    fridays = df[df[date_col].dt.dayofweek == 4].copy()
    if fridays.empty:
        return {
            "lines": [],
            "bands": bands_data,
            "history_years": history_years,
            "current_year": current_year,
            "prior_year": prior_year,
        }

    fridays = fridays.sort_values(date_col)
    fridays["week_index"] = fridays.groupby("year").cumcount()

    import holidays
    from datetime import timedelta

    try:
        if country in ["US", "USA", "United States", "US Total"]:
            country_holidays = holidays.US(years=df["year"].unique())
        else:
            country_holidays = holidays.country_holidays(country, years=df["year"].unique())
    except Exception:
        country_holidays = {}

    def get_holiday_in_window(d: pd.Timestamp) -> str | None:
        for i in range(7):
            check_date = d - timedelta(days=i)
            name = country_holidays.get(check_date)
            if name:
                return name
        return None

    fridays["holiday"] = fridays[date_col].apply(get_holiday_in_window)

    payload = {
        "lines": [],
        "bands": bands_data,
        "history_years": history_years,
        "current_year": current_year,
        "prior_year": prior_year,
    }

    all_years = sorted(int(y) for y in fridays["year"].unique())
    if not all_years:
        return payload

    if include_years is None:
        include_years = [y for y in all_years if y >= line_start_year]
    else:
        include_years = [int(y) for y in include_years if int(y) in all_years]

    lines = []
    for year in include_years:
        year_data = fridays[fridays["year"] == year].sort_values(date_col)
        prev_year = year - 1
        yoy_values = []

        if include_yoy:
            if prev_year in all_years:
                prev_data = (
                    fridays[fridays["year"] == prev_year]
                    .set_index("week_index")[value_col]
                )
                for _, row in year_data.iterrows():
                    curr_val = row[value_col]
                    prev_val = prev_data.get(row["week_index"])
                    if pd.notna(curr_val) and pd.notna(prev_val) and prev_val != 0:
                        yoy_values.append(((curr_val - prev_val) / prev_val) * 100)
                    else:
                        yoy_values.append(None)
            else:
                yoy_values = [None] * len(year_data)

        line_payload = {
            "year": int(year),
            "x": year_data["common_date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": year_data[value_col].tolist(),
            "dates": year_data[date_col].dt.strftime("%Y-%m-%d").tolist(),
            "holidays": year_data["holiday"].fillna("").tolist(),
            "is_current": int(year) == current_year,
            "is_prior": prior_year is not None and int(year) == prior_year,
        }
        if include_yoy:
            line_payload["values_yoy"] = yoy_values

        lines.append(line_payload)

    payload["lines"] = lines
    return payload
