"""CFTC disaggregated COT dashboard utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import base64
import io
import json
import re
import zipfile
from typing import Iterable, Optional

import pandas as pd
import requests

CFTC_DISAGG_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"
ICE_COT_URL_TEMPLATE = "https://www.ice.com/publicdocs/futures/COTHist{year}.csv"
DEFAULT_MARKET_CODES = ("CBT", "NYME", "CME", "CMX")
ICE_MARKET_CODES = ("ICEU",)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
CFTC_DASHBOARD_DIR = DATA_DIR / "cftc_disagg"
CFTC_CACHE_DIR = CFTC_DASHBOARD_DIR / "cache"
CFTC_CACHE_META = CFTC_CACHE_DIR / "cache_meta.json"
ICE_CACHE_DIR = CFTC_DASHBOARD_DIR / "ice_cache"
ICE_CACHE_META = ICE_CACHE_DIR / "cache_meta.json"

CFTC_FILTERED_CSV = CFTC_DASHBOARD_DIR / "cftc_disagg_filtered.csv"
CFTC_DASHBOARD_JSON = CFTC_DASHBOARD_DIR / "cftc_disagg_dashboard_data.json"
CFTC_DASHBOARD_HTML = CFTC_DASHBOARD_DIR / "cftc_disagg_dashboard.html"
CFTC_DASHBOARD_EXPORT_HTML = CFTC_DASHBOARD_DIR / "cftc_disagg_dashboard_export.html"
CFTC_ELIGIBLE_COMMODITIES_CSV = CFTC_DASHBOARD_DIR / "cftc_disagg_eligible_commodities.csv"
CFTC_SEASONALITY_EXPORT_CSV = CFTC_DASHBOARD_DIR / "cftc_disagg_seasonality_export.csv"
CFTC_SCATTER_EXPORT_CSV = CFTC_DASHBOARD_DIR / "cftc_disagg_scatter_export.csv"

CFTC_MARKET_MAP = {
    "067651": "WTI Crude Oil (Physical)",
    "06765A": "WTI Crude Oil (Financial)",
    "06765T": "Brent Crude (Last Day Financial)",
    "ICEU-B": "ICE Brent Crude",
    "0676A5": "WTI Houston (Argus)",
    "067A71": "WTI Midland (Argus)",
    "06765L": "WTI-Brent Calendar Spread",
    "067A28": "WTI Crude Calendar Spread Option",
    "111659": "RBOB Gasoline",
    "022651": "NY Harbor ULSD (Heating Oil)",
    "ICEU-G": "ICE Gasoil",
    "06665O": "Propane",
    "023651": "Natural Gas (Physical)",
    "03565B": "Henry Hub Natural Gas (Swap/Financial)",
    "023A55": "Henry Hub Natural Gas (Last Day Financial)",
    "03565C": "Henry Hub Natural Gas (Penultimate Financial)",
    "023A56": "Henry Hub Natural Gas (Penultimate Fin)",
    "02365U": "Natural Gas (European Style Option)",
    "023A84": "Natural Gas Calendar Spread Option",
    "088691": "Gold",
    "084691": "Silver",
    "085692": "Copper",
    "002602": "Corn",
    "005602": "Soybeans",
    "026603": "Soybean Meal",
    "007601": "Soybean Oil",
    "001602": "SRW Wheat (Soft Red Winter)",
    "001612": "HRW Wheat (Hard Red Winter)",
    "ICEU-Cocoa": "Cocoa",
    "ICEU-W": "White Sugar",
    "057642": "Live Cattle",
    "061641": "Feeder Cattle",
    "054642": "Lean Hogs",
}

CFTC_CANONICAL_COLS = {
    "Market_and_Exchange_Names",
    "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code",
    "CFTC_Market_Code",
    "Open_Interest_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
    "Pct_of_OI_M_Money_Long_All",
    "Pct_of_OI_M_Money_Short_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
    "Pct_of_OI_Prod_Merc_Long_All",
    "Pct_of_OI_Prod_Merc_Short_All",
    "Swap_Positions_Long_All",
    "Swap_Positions_Short_All",
    "Pct_of_OI_Swap_Long_All",
    "Pct_of_OI_Swap_Short_All",
    "FutOnly_or_Combined",
}
ICE_CANONICAL_COLS = {
    "Market_and_Exchange_Names",
    "As_of_Date_Form_MM/DD/YYYY",
    "As_of_Date_In_Form_YYMMDD",
    "CFTC_Market_Code",
    "CFTC_Commodity_Code",
    "Open_Interest_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
    "Pct_of_OI_M_Money_Long_All",
    "Pct_of_OI_M_Money_Short_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
    "Pct_of_OI_Prod_Merc_Long_All",
    "Pct_of_OI_Prod_Merc_Short_All",
    "Swap_Positions_Long_All",
    "Swap_Positions_Short_All",
    "Pct_of_OI_Swap_Long_All",
    "Pct_of_OI_Swap_Short_All",
    "FutOnly_or_Combined",
}

NUMERIC_COLS = [
    "Open_Interest_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
    "Pct_of_OI_M_Money_Long_All",
    "Pct_of_OI_M_Money_Short_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
    "Pct_of_OI_Prod_Merc_Long_All",
    "Pct_of_OI_Prod_Merc_Short_All",
    "Swap_Positions_Long_All",
    "Swap_Positions_Short_All",
    "Pct_of_OI_Swap_Long_All",
    "Pct_of_OI_Swap_Short_All",
]


def _load_cache_meta(path: Path = CFTC_CACHE_META) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache_meta(meta: dict, path: Path = CFTC_CACHE_META) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _url_exists(year: int, *, session: requests.Session) -> bool:
    url = CFTC_DISAGG_URL_TEMPLATE.format(year=year)
    resp = session.head(url, timeout=20, allow_redirects=True)
    return resp.status_code == 200


def _url_exists_ice(year: int, *, session: requests.Session) -> bool:
    url = ICE_COT_URL_TEMPLATE.format(year=year)
    resp = session.head(url, timeout=20, allow_redirects=True)
    return resp.status_code == 200


def resolve_latest_cftc_year(base_year: Optional[int] = None) -> int:
    base = base_year or datetime.utcnow().year
    with requests.Session() as session:
        for year in range(base + 1, base - 6, -1):
            if _url_exists(year, session=session):
                return year
    return base


def resolve_latest_ice_year(base_year: Optional[int] = None) -> int:
    base = base_year or datetime.utcnow().year
    with requests.Session() as session:
        for year in range(base + 1, base - 6, -1):
            if _url_exists_ice(year, session=session):
                return year
    return base


def _fetch_head(year: int, *, session: requests.Session) -> dict:
    url = CFTC_DISAGG_URL_TEMPLATE.format(year=year)
    resp = session.head(url, timeout=20, allow_redirects=True)
    if resp.status_code != 200:
        return {}
    return {
        "etag": resp.headers.get("ETag") or resp.headers.get("Etag"),
        "last_modified": resp.headers.get("Last-Modified"),
    }


def _fetch_ice_head(year: int, *, session: requests.Session) -> dict:
    url = ICE_COT_URL_TEMPLATE.format(year=year)
    resp = session.head(url, timeout=20, allow_redirects=True)
    if resp.status_code != 200:
        return {}
    return {
        "etag": resp.headers.get("ETag") or resp.headers.get("Etag"),
        "last_modified": resp.headers.get("Last-Modified"),
    }


def _download_year(year: int, *, session: requests.Session, timeout: int = 60) -> bytes:
    url = CFTC_DISAGG_URL_TEMPLATE.format(year=year)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _download_ice_year(
    year: int, *, session: requests.Session, timeout: int = 60
) -> Optional[bytes]:
    url = ICE_COT_URL_TEMPLATE.format(year=year)
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def _extract_txt(content: bytes, target_path: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Empty CFTC zip archive.")
        with zf.open(names[0]) as src, target_path.open("wb") as dst:
            dst.write(src.read())


def _ensure_year_file(
    year: int,
    latest_year: int,
    *,
    session: requests.Session,
    timeout: int = 60,
) -> Path:
    CFTC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CFTC_CACHE_DIR / f"com_disagg_{year}.txt"
    meta = _load_cache_meta()
    year_key = str(year)

    if year != latest_year and target_path.exists():
        return target_path

    head = _fetch_head(year, session=session) if year == latest_year else {}
    cached = meta.get(year_key, {})
    if year == latest_year and target_path.exists():
        if head and head.get("etag") and head.get("etag") == cached.get("etag"):
            return target_path
        if head and head.get("last_modified") and head.get("last_modified") == cached.get("last_modified"):
            return target_path

    content = _download_year(year, session=session, timeout=timeout)
    _extract_txt(content, target_path)

    meta[year_key] = {
        "etag": head.get("etag"),
        "last_modified": head.get("last_modified"),
        "downloaded_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_cache_meta(meta)
    return target_path


def _ensure_ice_year_file(
    year: int,
    latest_year: int,
    *,
    session: requests.Session,
    timeout: int = 60,
) -> Optional[Path]:
    ICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ICE_CACHE_DIR / f"COTHist{year}.csv"
    meta = _load_cache_meta(ICE_CACHE_META)
    year_key = str(year)

    if year != latest_year and target_path.exists():
        return target_path

    head = _fetch_ice_head(year, session=session) if year == latest_year else {}
    cached = meta.get(year_key, {})
    if year == latest_year and target_path.exists():
        if head and head.get("etag") and head.get("etag") == cached.get("etag"):
            return target_path
        if head and head.get("last_modified") and head.get("last_modified") == cached.get("last_modified"):
            return target_path

    content = _download_ice_year(year, session=session, timeout=timeout)
    if content is None:
        return None
    target_path.write_bytes(content)

    meta[year_key] = {
        "etag": head.get("etag"),
        "last_modified": head.get("last_modified"),
        "downloaded_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_cache_meta(meta, ICE_CACHE_META)
    return target_path


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _normalize_col(name: str) -> str:
    clean = name.strip().lstrip("\ufeff")
    return re.sub(r"_+", "_", clean)


def _resolve_display_name(code: str, fallback: str) -> str:
    return CFTC_MARKET_MAP.get(code, fallback)


def _load_plotly_bundle_bytes() -> bytes:
    bundle_path = CFTC_DASHBOARD_DIR / "plotly-3.3.1.min.js"
    if bundle_path.exists():
        return bundle_path.read_bytes()
    resp = requests.get("https://cdn.plot.ly/plotly-3.3.1.min.js", timeout=30)
    resp.raise_for_status()
    return resp.content


def _inline_plotly_tag() -> str:
    encoded = base64.b64encode(_load_plotly_bundle_bytes()).decode("ascii")
    return f"""<script>
    (function() {{
      const b64 = "{encoded}";
      const raw = atob(b64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) {{
        bytes[i] = raw.charCodeAt(i);
      }}
      const blob = new Blob([bytes], {{ type: "text/javascript" }});
      const url = URL.createObjectURL(blob);
      const script = document.createElement("script");
      script.defer = true;
      script.src = url;
      script.onload = () => URL.revokeObjectURL(url);
      document.head.appendChild(script);
    }})();
  </script>"""


def _load_cftc_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda col: _normalize_col(col) in CFTC_CANONICAL_COLS,
        dtype=str,
        na_values=[".", ""],
        keep_default_na=True,
    )
    df.columns = [_normalize_col(col) for col in df.columns]
    for col in CFTC_CANONICAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df["Market_and_Exchange_Names"] = df["Market_and_Exchange_Names"].astype(str).str.strip()
    df["CFTC_Market_Code"] = df["CFTC_Market_Code"].astype(str).str.strip()
    df["CFTC_Contract_Market_Code"] = (
        df["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
    )

    for col in NUMERIC_COLS:
        df[col] = _clean_numeric(df[col])

    df["Report_Date_as_YYYY-MM-DD"] = pd.to_datetime(
        df["Report_Date_as_YYYY-MM-DD"], errors="coerce"
    )
    df = df.dropna(subset=["Report_Date_as_YYYY-MM-DD"])
    df["report_date"] = df["Report_Date_as_YYYY-MM-DD"].dt.strftime("%Y-%m-%d")
    df["report_year"] = df["Report_Date_as_YYYY-MM-DD"].dt.year
    df["report_week"] = df["Report_Date_as_YYYY-MM-DD"].dt.isocalendar().week.astype(int)
    return df


def _load_ice_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda col: _normalize_col(col) in ICE_CANONICAL_COLS,
        dtype=str,
        na_values=[".", ""],
        keep_default_na=True,
        encoding="utf-8-sig",
    )
    df.columns = [_normalize_col(col) for col in df.columns]
    for col in ICE_CANONICAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df["Market_and_Exchange_Names"] = df["Market_and_Exchange_Names"].astype(str).str.strip()
    df["CFTC_Market_Code"] = df["CFTC_Market_Code"].fillna("").astype(str).str.strip()
    df["CFTC_Commodity_Code"] = df["CFTC_Commodity_Code"].fillna("").astype(str).str.strip()

    for col in NUMERIC_COLS:
        df[col] = _clean_numeric(df[col])

    date_series = pd.to_datetime(
        df["As_of_Date_Form_MM/DD/YYYY"], format="%m/%d/%Y", errors="coerce"
    )
    if "As_of_Date_In_Form_YYMMDD" in df.columns:
        fallback = pd.to_datetime(
            df["As_of_Date_In_Form_YYMMDD"], format="%y%m%d", errors="coerce"
        )
        date_series = date_series.fillna(fallback)

    df["Report_Date_as_YYYY-MM-DD"] = date_series.dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Report_Date_as_YYYY-MM-DD"])
    df["report_date"] = df["Report_Date_as_YYYY-MM-DD"]
    df["report_year"] = date_series.dt.year
    df["report_week"] = date_series.dt.isocalendar().week.astype(int)

    market_code = df["CFTC_Market_Code"].replace("", "ICE")
    commodity_code = df["CFTC_Commodity_Code"].replace("", pd.NA)
    fallback_code = (
        df["Market_and_Exchange_Names"]
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    code = commodity_code.fillna(fallback_code)
    df["CFTC_Contract_Market_Code"] = market_code + "-" + code
    return df


def _select_history_years(
    latest_year: int,
    *,
    count: int = 5,
    exclude: Iterable[int] = (2020, 2021),
) -> list[int]:
    years = []
    year = latest_year - 1
    excluded = set(exclude)
    while len(years) < count and year >= 1990:
        if year not in excluded:
            years.append(year)
        year -= 1
    return sorted(years)


def update_cftc_disagg_cache(
    *,
    latest_year: Optional[int] = None,
    last_years: int = 7,
    history_years: Optional[Iterable[int]] = None,
    timeout_seconds: int = 60,
) -> tuple[pd.DataFrame, dict]:
    latest_cftc = latest_year or resolve_latest_cftc_year()
    latest_ice = resolve_latest_ice_year(latest_cftc)
    latest = max(latest_cftc, latest_ice)
    history = list(history_years or _select_history_years(latest))
    cftc_years = sorted(set(list(range(latest_cftc - last_years + 1, latest_cftc + 1)) + history))
    ice_years = sorted(set(list(range(latest_ice - last_years + 1, latest_ice + 1)) + history))

    cftc_frames: list[pd.DataFrame] = []
    ice_frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for year in cftc_years:
            path = _ensure_year_file(year, latest_cftc, session=session, timeout=timeout_seconds)
            cftc_frames.append(_load_cftc_txt(path))
        for year in ice_years:
            path = _ensure_ice_year_file(year, latest_ice, session=session, timeout=timeout_seconds)
            if path is None:
                continue
            ice_frames.append(_load_ice_csv(path))

    if not cftc_frames:
        raise ValueError("No CFTC data available for requested years.")

    cftc_data = pd.concat(cftc_frames, ignore_index=True)
    cftc_data = cftc_data[cftc_data["FutOnly_or_Combined"].astype(str).str.strip().eq("Combined")]
    ice_data = pd.concat(ice_frames, ignore_index=True) if ice_frames else pd.DataFrame()
    if not ice_data.empty:
        ice_data = ice_data[ice_data["FutOnly_or_Combined"].astype(str).str.strip().eq("Combined")]

    min_open_interest = 100000
    eligible_codes: set[str] = set()

    cftc_current = cftc_data[cftc_data["report_year"] == latest_cftc]
    if cftc_current.empty:
        cftc_current = cftc_data
    if cftc_current.empty:
        raise ValueError(f"No rows found for latest year {latest_cftc}.")
    latest_cftc_date = cftc_current["report_date"].max()
    cftc_slice = cftc_current[cftc_current["report_date"] == latest_cftc_date].copy()
    cftc_slice["CFTC_Market_Code"] = cftc_slice["CFTC_Market_Code"].astype(str).str.strip()
    eligible_cftc = cftc_slice[
        cftc_slice["CFTC_Market_Code"].isin(DEFAULT_MARKET_CODES)
        & (cftc_slice["Open_Interest_All"] >= min_open_interest)
    ]
    eligible_codes.update(eligible_cftc["CFTC_Contract_Market_Code"].astype(str))

    latest_ice_date = None
    if not ice_data.empty:
        ice_current = ice_data[ice_data["report_year"] == latest_ice]
        if ice_current.empty:
            ice_current = ice_data
        latest_ice_date = ice_current["report_date"].max()
        ice_slice = ice_current[ice_current["report_date"] == latest_ice_date].copy()
        ice_slice["CFTC_Market_Code"] = ice_slice["CFTC_Market_Code"].astype(str).str.strip()
        eligible_ice = ice_slice[
            ice_slice["CFTC_Market_Code"].isin(ICE_MARKET_CODES)
            & (ice_slice["Open_Interest_All"] >= min_open_interest)
        ]
        eligible_codes.update(eligible_ice["CFTC_Contract_Market_Code"].astype(str))

    if not eligible_codes:
        raise ValueError("No contracts matched the market code and open interest filters.")

    data_frames = [cftc_data]
    if not ice_data.empty:
        data_frames.append(ice_data)
    data = pd.concat(data_frames, ignore_index=True)
    filtered = data[data["CFTC_Contract_Market_Code"].isin(eligible_codes)].copy()
    if filtered.empty:
        raise ValueError("No rows available after filtering eligible contracts.")

    latest_report_date = filtered["report_date"].max()
    latest_year_value = int(filtered["report_year"].max())
    meta = {
        "latest_year": latest_year_value,
        "latest_report_date": latest_report_date,
        "history_years": history,
        "filter_market_codes": list(DEFAULT_MARKET_CODES) + list(ICE_MARKET_CODES),
        "min_open_interest": min_open_interest,
    }
    return filtered, meta


def build_cftc_disagg_dashboard_payload(
    data: pd.DataFrame,
    *,
    meta: dict,
) -> dict:
    latest_year = meta["latest_year"]
    latest_date = meta["latest_report_date"]
    label_map = (
        data.sort_values("report_date")
        .drop_duplicates("CFTC_Contract_Market_Code", keep="last")
        .set_index("CFTC_Contract_Market_Code")
    )

    payload = {
        "meta": {
            "latest_year": int(latest_year),
            "latest_report_date": latest_date,
            "latest_week": int(
                data.loc[data["report_date"] == latest_date, "report_week"].max()
            ),
            "start_date": data["report_date"].min(),
            "end_date": data["report_date"].max(),
            "history_years": [int(y) for y in meta.get("history_years", [])],
            "filter_market_codes": meta.get("filter_market_codes", []),
            "min_open_interest": meta.get("min_open_interest"),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "commodities": {},
    }

    for code, group in data.groupby("CFTC_Contract_Market_Code"):
        group = group.sort_values("report_date")
        label_row = label_map.loc[code]
        display_name = _resolve_display_name(str(code), str(label_row["Market_and_Exchange_Names"]))
        commodity = {
            "name": str(label_row["Market_and_Exchange_Names"]),
            "display_name": display_name,
            "market_code": str(label_row["CFTC_Market_Code"]).strip(),
            "dates": group["report_date"].tolist(),
            "year": group["report_year"].astype(int).tolist(),
            "week": group["report_week"].astype(int).tolist(),
            "open_interest": group["Open_Interest_All"].where(pd.notna(group["Open_Interest_All"]), None).tolist(),
            "m_money_long": group["M_Money_Positions_Long_All"]
            .where(pd.notna(group["M_Money_Positions_Long_All"]), None)
            .tolist(),
            "m_money_short": group["M_Money_Positions_Short_All"]
            .where(pd.notna(group["M_Money_Positions_Short_All"]), None)
            .tolist(),
            "pct_long": group["Pct_of_OI_M_Money_Long_All"]
            .where(pd.notna(group["Pct_of_OI_M_Money_Long_All"]), None)
            .tolist(),
            "pct_short": group["Pct_of_OI_M_Money_Short_All"]
            .where(pd.notna(group["Pct_of_OI_M_Money_Short_All"]), None)
            .tolist(),
            "prod_long": group["Prod_Merc_Positions_Long_All"]
            .where(pd.notna(group["Prod_Merc_Positions_Long_All"]), None)
            .tolist(),
            "prod_short": group["Prod_Merc_Positions_Short_All"]
            .where(pd.notna(group["Prod_Merc_Positions_Short_All"]), None)
            .tolist(),
            "pct_prod_long": group["Pct_of_OI_Prod_Merc_Long_All"]
            .where(pd.notna(group["Pct_of_OI_Prod_Merc_Long_All"]), None)
            .tolist(),
            "pct_prod_short": group["Pct_of_OI_Prod_Merc_Short_All"]
            .where(pd.notna(group["Pct_of_OI_Prod_Merc_Short_All"]), None)
            .tolist(),
            "swap_long": group["Swap_Positions_Long_All"]
            .where(pd.notna(group["Swap_Positions_Long_All"]), None)
            .tolist(),
            "swap_short": group["Swap_Positions_Short_All"]
            .where(pd.notna(group["Swap_Positions_Short_All"]), None)
            .tolist(),
            "pct_swap_long": group["Pct_of_OI_Swap_Long_All"]
            .where(pd.notna(group["Pct_of_OI_Swap_Long_All"]), None)
            .tolist(),
            "pct_swap_short": group["Pct_of_OI_Swap_Short_All"]
            .where(pd.notna(group["Pct_of_OI_Swap_Short_All"]), None)
            .tolist(),
        }
        payload["commodities"][str(code)] = commodity

    return payload


def _build_dashboard_html(payload: dict, *, inline_plotly: bool = False) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    plotly_tag = """<script defer src="plotly-3.3.1.min.js" charset="utf-8"
    onerror="var s=document.createElement('script');s.src='https://cdn.plot.ly/plotly-3.3.1.min.js';s.defer=true;document.head.appendChild(s);">
  </script>"""
    if inline_plotly:
        plotly_tag = _inline_plotly_tag()
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CFTC Disaggregated Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  __PLOTLY__
  <style>
    :root {
      --bg: #0f172a;
      --panel: #121826;
      --accent: #38bdf8;
      --text: #f8fafc;
      --muted: #94a3b8;
      --grid: #243042;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #182235, #0f172a 55%);
      color: var(--text);
    }
    header {
      padding: 24px 32px;
      border-bottom: 1px solid #1e293b;
      background: #0f172a;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px 16px;
    }
    .header-left {
      display: flex;
      flex-direction: column;
    }
    header h1 {
      margin: 0;
      font-size: 24px;
    }
    header p {
      margin: 6px 0 0;
      color: var(--muted);
    }
    .header-note {
      margin-left: auto;
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .header-updated {
      margin-top: 4px;
      font-size: 11px;
      color: var(--muted);
    }
    .container {
      padding: 24px 32px 48px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: flex-start;
      background: var(--panel);
      padding: 16px;
      border-radius: 12px;
      border: 1px solid #1e293b;
    }
    .controls label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    select {
      margin-left: 8px;
      padding: 6px 8px;
      background: #0f172a;
      color: var(--text);
      border: 1px solid #1e293b;
      border-radius: 6px;
    }
    .controls select[multiple] {
      min-width: 280px;
      height: 160px;
    }
    .control-hint {
      font-size: 12px;
      color: var(--muted);
      flex-basis: 100%;
    }
    .summary-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .summary-card {
      background: var(--panel);
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 12px;
    }
    .summary-card h4 {
      margin: 0 0 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .summary-card .value {
      font-size: 20px;
      font-weight: 600;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }
    @media (min-width: 1100px) {
      .chart-grid.two {
        grid-template-columns: 1fr 1fr;
      }
    }
    .chart-card {
      background: var(--panel);
      border-radius: 14px;
      border: 1px solid #1e293b;
      padding: 12px;
    }
    .chart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin: 6px 0 12px;
    }
    .chart-header h3 {
      margin: 0;
      font-size: 16px;
    }
    .chart-header select[multiple] {
      min-width: 240px;
      height: 120px;
    }
    .header-actions {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      background-color: var(--panel);
      border: 1px solid #1e293b;
      border-radius: 6px;
      text-decoration: none;
      transition: all 0.2s ease;
    }
    .btn:hover {
      background-color: #1e293b;
      border-color: var(--accent);
      color: var(--accent);
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <h1>CFTC Money Manager Seasonality</h1>
      <p>Weekly seasonality for money manager positions and percent of open interest.</p>
    </div>
    <div class="header-actions">
      <button class="btn" onclick="exportSeasonalityData()">Export Seasonality Data</button>
      <button class="btn" onclick="exportScatterData()">Export Scatter Data</button>
    </div>
    <div class="header-note">
      <div>Created and Maintained by Alex Hoffmann</div>
      <div class="header-updated" id="lastUpdated"></div>
    </div>
  </header>
  <div class="container">
    <div class="controls">
      <label>Commodities
        <select id="commoditySelect" multiple></select>
      </label>
      <label>Report
        <select id="reportSelect">
          <option value="managed">Managed Money</option>
          <option value="producer">Producer</option>
          <option value="swap">Swap Dealers</option>
        </select>
      </label>
      <div class="control-hint">Hold Command/Ctrl to select multiple contracts.</div>
    </div>
    <div class="summary-grid" id="summaryGrid"></div>
    <div class="chart-grid two">
      <div class="chart-card">
        <h3 id="longTitle">Managed Money Long (Contracts)</h3>
        <div id="longChart"></div>
      </div>
      <div class="chart-card">
        <h3 id="shortTitle">Managed Money Short (Contracts)</h3>
        <div id="shortChart"></div>
      </div>
    </div>
    <div class="chart-grid two">
      <div class="chart-card">
        <h3 id="pctLongTitle">Managed Money Long (% OI)</h3>
        <div id="pctLongChart"></div>
      </div>
      <div class="chart-card">
        <h3 id="pctShortTitle">Managed Money Short (% OI)</h3>
        <div id="pctShortChart"></div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-header">
          <h3 id="scatterTitle">Managed Money %OI Long-Short Scatter</h3>
          <label>X Commodity
            <select id="xCommoditySelect" multiple></select>
          </label>
          <label>Y Commodity
            <select id="yCommoditySelect" multiple></select>
          </label>
        </div>
        <div id="percentileChart"></div>
      </div>
    </div>
  </div>

  <script>
    const DATA = __DATA__;
    const chartConfig = {responsive: true, displaylogo: false};
    const excludedYears = [2020, 2021];

    const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const BASE_YEAR = 2024;

    const METRICS = [
      { key: "m_money_long", label: "Money Manager Long (Contracts)", chartId: "longChart" },
      { key: "m_money_short", label: "Money Manager Short (Contracts)", chartId: "shortChart" },
      { key: "pct_long", label: "Money Manager Long (% OI)", chartId: "pctLongChart" },
      { key: "pct_short", label: "Money Manager Short (% OI)", chartId: "pctShortChart" }
    ];

    const commoditySelect = document.getElementById("commoditySelect");
    const reportSelect = document.getElementById("reportSelect");
    const xCommoditySelect = document.getElementById("xCommoditySelect");
    const yCommoditySelect = document.getElementById("yCommoditySelect");
    const summaryGrid = document.getElementById("summaryGrid");

    const REPORT_CONFIGS = {
      managed: {
        label: "Managed Money",
        longKey: "m_money_long",
        shortKey: "m_money_short",
        pctLongKey: "pct_long",
        pctShortKey: "pct_short"
      },
      producer: {
        label: "Producer",
        longKey: "prod_long",
        shortKey: "prod_short",
        pctLongKey: "pct_prod_long",
        pctShortKey: "pct_prod_short"
      },
      swap: {
        label: "Swap Dealers",
        longKey: "swap_long",
        shortKey: "swap_short",
        pctLongKey: "pct_swap_long",
        pctShortKey: "pct_swap_short"
      }
    };

    function formatNumber(value, suffix) {
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      return `${value.toLocaleString(undefined, {maximumFractionDigits: 2})}${suffix || ""}`;
    }

    function isValidNumber(value) {
      return value !== null && value !== undefined && !Number.isNaN(value);
    }

    function currentReportKey() {
      const key = reportSelect?.value || "managed";
      return REPORT_CONFIGS[key] ? key : "managed";
    }

    function reportConfig(reportKey) {
      return REPORT_CONFIGS[reportKey] || REPORT_CONFIGS.managed;
    }

    function updateChartTitles(reportKey) {
      const label = reportConfig(reportKey).label;
      document.getElementById("longTitle").textContent = `${label} Long (Contracts)`;
      document.getElementById("shortTitle").textContent = `${label} Short (Contracts)`;
      document.getElementById("pctLongTitle").textContent = `${label} Long (% OI)`;
      document.getElementById("pctShortTitle").textContent = `${label} Short (% OI)`;
      document.getElementById("scatterTitle").textContent = `${label} %OI Long-Short Scatter`;
    }

    function selectedCommodities() {
      return Array.from(commoditySelect.selectedOptions).map(opt => opt.value);
    }

    function selectedMultiValues(selectEl, fallbackIndex) {
      const values = Array.from(selectEl.selectedOptions).map(opt => opt.value);
      if (values.length) return values;
      if (selectEl.options.length) {
        const idx = Math.min(fallbackIndex || 0, selectEl.options.length - 1);
        selectEl.options[idx].selected = true;
        return [selectEl.options[idx].value];
      }
      return [];
    }

    function formatCommodityLabel(codes) {
      const names = [];
      codes.forEach(code => {
        const info = DATA.commodities?.[code] || {};
        const name = info.display_name || info.name || code;
        if (!names.includes(name)) names.push(name);
      });
      return names.join(" + ");
    }

    function axisTitleFontSize(label, count) {
      const base = 16;
      const lengthPenalty = Math.max(0, label.length - 28);
      const countPenalty = Math.max(0, count - 1) * 2;
      const size = base - Math.ceil(lengthPenalty / 12) - countPenalty;
      return Math.max(10, Math.round(size));
    }

    function ensureSelection() {
      const selected = selectedCommodities();
      if (selected.length > 0) return selected;
      if (commoditySelect.options.length) {
        commoditySelect.options[0].selected = true;
        return [commoditySelect.options[0].value];
      }
      return [];
    }

    function isoWeekDate(year, week, weekday) {
      const simple = new Date(Date.UTC(year, 0, 1 + (week - 1) * 7));
      const dow = simple.getUTCDay() || 7;
      if (dow <= 4) {
        simple.setUTCDate(simple.getUTCDate() - dow + 1);
      } else {
        simple.setUTCDate(simple.getUTCDate() + 8 - dow);
      }
      simple.setUTCDate(simple.getUTCDate() + (weekday - 1));
      return simple;
    }

    function monthTickValues() {
      return MONTH_LABELS.map((_, idx) => {
        const dt = new Date(Date.UTC(BASE_YEAR, idx, 1));
        return dt.toISOString().slice(0, 10);
      });
    }

    function buildAggregates(codes, reportKey) {
      const config = reportConfig(reportKey);
      const agg = { long: {}, short: {}, oi: {}, pctLong: {}, pctShort: {} };
      const years = new Set();
      const weeks = new Set();
      const useRawPct = codes.length === 1;

      codes.forEach(code => {
        const series = (DATA.commodities || {})[code];
        if (!series) return;
        for (let i = 0; i < series.dates.length; i += 1) {
          const year = series.year[i];
          const week = series.week[i];
          if (!year || !week) continue;
          years.add(year);
          weeks.add(week);

          if (!agg.long[year]) {
            agg.long[year] = {};
            agg.short[year] = {};
            agg.oi[year] = {};
            if (useRawPct) {
              agg.pctLong[year] = {};
              agg.pctShort[year] = {};
            }
          }

          const longVal = series[config.longKey]?.[i];
          if (longVal !== null && longVal !== undefined && !Number.isNaN(longVal)) {
            agg.long[year][week] = (agg.long[year][week] || 0) + longVal;
          }

          const shortVal = series[config.shortKey]?.[i];
          if (shortVal !== null && shortVal !== undefined && !Number.isNaN(shortVal)) {
            agg.short[year][week] = (agg.short[year][week] || 0) + shortVal;
          }

          const oiVal = series.open_interest[i];
          if (oiVal !== null && oiVal !== undefined && !Number.isNaN(oiVal)) {
            agg.oi[year][week] = (agg.oi[year][week] || 0) + oiVal;
          }

          if (useRawPct) {
            const pctL = series[config.pctLongKey]?.[i];
            if (pctL !== null && pctL !== undefined && !Number.isNaN(pctL)) {
              agg.pctLong[year][week] = pctL;
            }
            const pctS = series[config.pctShortKey]?.[i];
            if (pctS !== null && pctS !== undefined && !Number.isNaN(pctS)) {
              agg.pctShort[year][week] = pctS;
            }
          }
        }
      });

      const yearList = Array.from(years).sort((a, b) => a - b);
      const maxWeek = Math.max(53, ...(weeks.size ? Array.from(weeks) : [53]));
      return { agg, years: yearList, maxWeek, useRawPct };
    }

    function buildSeries(metricKey, aggregates) {
      const series = {};
      const { agg, years, maxWeek, useRawPct } = aggregates;

      years.forEach(year => {
        const values = [];
        for (let w = 1; w <= maxWeek; w += 1) {
          let value = null;
          if (metricKey === "m_money_long") {
            value = agg.long[year]?.[w] ?? null;
          } else if (metricKey === "m_money_short") {
            value = agg.short[year]?.[w] ?? null;
          } else if (metricKey === "pct_long") {
            if (useRawPct) {
              value = agg.pctLong[year]?.[w] ?? null;
            } else {
              const longVal = agg.long[year]?.[w];
              const oiVal = agg.oi[year]?.[w];
              if (longVal !== undefined && oiVal) value = (longVal / oiVal) * 100.0;
            }
          } else if (metricKey === "pct_short") {
            if (useRawPct) {
              value = agg.pctShort[year]?.[w] ?? null;
            } else {
              const shortVal = agg.short[year]?.[w];
              const oiVal = agg.oi[year]?.[w];
              if (shortVal !== undefined && oiVal) value = (shortVal / oiVal) * 100.0;
            }
          }
          values.push(value ?? null);
        }
        series[year] = values;
      });
      return series;
    }

    function renderSeasonality(metricKey, chartId, aggregates) {
      const series = buildSeries(metricKey, aggregates);
      const { years, maxWeek } = aggregates;
      const isPercent = metricKey.startsWith("pct");
      if (!years.length) {
        Plotly.newPlot(chartId, [{text: ["No data"], type: "scatter"}], {}, chartConfig);
        return;
      }

      const displayYears = years.filter(y => !excludedYears.includes(y));
      const currentYear = DATA.meta?.latest_year || Math.max(...years);
      const limitWeek = DATA.meta?.latest_week || maxWeek;
      const statsYears = displayYears.filter(y => y < currentYear);

      const weekDates = Array.from({length: maxWeek}, (_, i) => {
        const dt = isoWeekDate(BASE_YEAR, i + 1, 2);
        return dt.toISOString().slice(0, 10);
      });
      const monthTicks = monthTickValues();
      const weeklyMin = [];
      const weeklyMax = [];
      const weeklyAvg = [];
      const hoverTemplate = metricKey.startsWith("pct")
        ? "%{customdata}<br>%{y:.1f}%<extra></extra>"
        : "%{customdata}<br>%{y:,.0f}<extra></extra>";
      const avgHoverTemplate = isPercent
        ? "%{x|%b %d}<br>%{y:.1f}%<extra></extra>"
        : "%{x|%b %d}<br>%{y:,.0f}<extra></extra>";

      for (let i = 0; i < maxWeek; i += 1) {
        const vals = statsYears
          .map(y => series[y]?.[i])
          .filter(v => v !== null && v !== undefined);
        if (!vals.length) {
          weeklyMin.push(null);
          weeklyMax.push(null);
          weeklyAvg.push(null);
          continue;
        }
        weeklyMin.push(Math.min(...vals));
        weeklyMax.push(Math.max(...vals));
        weeklyAvg.push(vals.reduce((a, b) => a + b, 0) / vals.length);
      }

      const traces = [];
      if (statsYears.length) {
        traces.push({
          x: weekDates,
          y: weeklyMax,
          mode: "lines",
          line: { width: 0 },
          showlegend: false,
          hoverinfo: "skip"
        });
        traces.push({
          x: weekDates,
          y: weeklyMin,
          mode: "lines",
          line: { width: 0 },
          fill: "tonexty",
          fillcolor: "rgba(100, 116, 139, 0.3)",
          name: "Historical Range",
          hoverinfo: "skip"
        });
        const rangeLabel = statsYears.length > 1
          ? `Avg (${Math.min(...statsYears)}-${Math.max(...statsYears)})`
          : `Avg (${statsYears[0]})`;
        traces.push({
          type: "scatter",
          mode: "lines",
          name: rangeLabel,
          x: weekDates,
          y: weeklyAvg,
          line: { color: "#cbd5e1", width: 2, dash: "dot" },
          hovertemplate: avgHoverTemplate
        });
      }

      const yearColors = ["#3b82f6", "#8b5cf6", "#d946ef", "#f97316", "#14b8a6"];
      displayYears.forEach((year, idx) => {
        const values = (series[year] || []).map((val, i) => {
          if (year === currentYear && i + 1 > limitWeek) return null;
          return val;
        });
        const actualDates = Array.from({length: maxWeek}, (_, i) => {
          const dt = isoWeekDate(year, i + 1, 2);
          return dt.toISOString().slice(0, 10);
        });
        const isCurrent = year === currentYear;
        const isPrior = year === currentYear - 1;
        const color = isCurrent ? "#22c55e" : yearColors[idx % yearColors.length];
        const width = isCurrent ? 3 : 2;
        const mode = isCurrent ? "lines+markers" : "lines";
        const visible = (isCurrent || isPrior) ? true : "legendonly";
        if (isCurrent) {
          const prevValues = series[year - 1] || [];
          let carry = null;
          for (let i = prevValues.length - 1; i >= 0; i -= 1) {
            if (prevValues[i] !== null && prevValues[i] !== undefined) {
              carry = prevValues[i];
              break;
            }
          }
          if (carry !== null) {
            const firstIdx = values.findIndex(val => val !== null && val !== undefined);
            if (firstIdx > 0) {
              for (let i = 0; i < firstIdx; i += 1) {
                values[i] = carry;
              }
            }
          }
        }
        const trace = {
          type: "scatter",
          mode,
          name: year.toString(),
          x: weekDates,
          y: values,
          customdata: actualDates,
          hovertemplate: hoverTemplate,
          line: { color, width },
          visible
        };
        if (isCurrent) {
          trace.marker = { symbol: "circle", size: 6 };
        }
        traces.push(trace);
      });

      Plotly.newPlot(chartId, traces, {
        hovermode: "closest",
        separators: ".,",
        plot_bgcolor: "#0f172a",
        paper_bgcolor: "#0f172a",
        font: { color: "#f8fafc" },
        hoverlabel: { namelength: -1 },
        xaxis: {
          gridcolor: "#1f2a44",
          title: { text: "Month", standoff: 10 },
          tickmode: "array",
          tickvals: monthTicks,
          ticktext: MONTH_LABELS,
          range: [`${BASE_YEAR}-01-01`, `${BASE_YEAR}-12-31`]
        },
        yaxis: {
          title: isPercent ? "Percent of Open Interest" : "Contracts",
          gridcolor: "#1f2a44",
          tickformat: isPercent ? ".1f" : ".2s",
          ticksuffix: isPercent ? "%" : ""
        },
        legend: { orientation: "h", y: 1.1, x: 0.5, xanchor: "center" }
      }, chartConfig);
    }

    function buildAxisNetPct(codes, reportKey) {
      const config = reportConfig(reportKey);
      const totals = {};
      const years = new Set();
      const useRawPct = codes.length === 1;

      codes.forEach(code => {
        const series = (DATA.commodities || {})[code];
        if (!series) return;
        for (let i = 0; i < series.dates.length; i += 1) {
          const year = series.year[i];
          const week = series.week[i];
          if (!year || !week) continue;

          years.add(year);
          if (!totals[year]) {
            totals[year] = { long: {}, short: {}, oi: {}, pctLong: {}, pctShort: {} };
          }

          const longVal = series[config.longKey]?.[i];
          if (isValidNumber(longVal)) {
            totals[year].long[week] = (totals[year].long[week] || 0) + longVal;
          }

          const shortVal = series[config.shortKey]?.[i];
          if (isValidNumber(shortVal)) {
            totals[year].short[week] = (totals[year].short[week] || 0) + shortVal;
          }

          const oiVal = series.open_interest[i];
          if (isValidNumber(oiVal)) {
            totals[year].oi[week] = (totals[year].oi[week] || 0) + oiVal;
          }

          if (useRawPct) {
            const pctL = series[config.pctLongKey]?.[i];
            if (isValidNumber(pctL)) {
              totals[year].pctLong[week] = pctL;
            }
            const pctS = series[config.pctShortKey]?.[i];
            if (isValidNumber(pctS)) {
              totals[year].pctShort[week] = pctS;
            }
          }
        }
      });

      const byYear = {};
      Object.keys(totals).forEach(yearKey => {
        const year = Number(yearKey);
        const yearTotals = totals[year];
        const weeks = new Set([
          ...Object.keys(yearTotals.long),
          ...Object.keys(yearTotals.short),
          ...Object.keys(yearTotals.oi),
          ...Object.keys(yearTotals.pctLong),
          ...Object.keys(yearTotals.pctShort),
        ]);
        byYear[year] = {};
        weeks.forEach(weekKey => {
          const week = Number(weekKey);
          const longVal = yearTotals.long[week];
          const shortVal = yearTotals.short[week];
          const oiVal = yearTotals.oi[week];
          let value = null;
          if (
            isValidNumber(oiVal) && oiVal !== 0 &&
            isValidNumber(longVal) && isValidNumber(shortVal)
          ) {
            value = ((longVal - shortVal) / oiVal) * 100.0;
          } else if (useRawPct) {
            const pctL = yearTotals.pctLong[week];
            const pctS = yearTotals.pctShort[week];
            if (isValidNumber(pctL) && isValidNumber(pctS)) {
              value = pctL - pctS;
            }
          }
          if (isValidNumber(value)) {
            byYear[year][week] = value;
          }
        });
      });
      return { byYear, years: Array.from(years).sort((a, b) => a - b) };
    }

    function renderScatter() {
      const reportKey = currentReportKey();
      const reportLabel = reportConfig(reportKey).label;
      const xCodes = selectedMultiValues(xCommoditySelect, 0);
      const yCodes = selectedMultiValues(yCommoditySelect, xCommoditySelect.options.length > 1 ? 1 : 0);
      if (!xCodes.length || !yCodes.length) {
        Plotly.newPlot("percentileChart", [{ text: ["Select X/Y commodities."], type: "scatter" }], {}, chartConfig);
        return;
      }

      const xSeries = buildAxisNetPct(xCodes, reportKey);
      const ySeries = buildAxisNetPct(yCodes, reportKey);
      const years = Array.from(new Set([...xSeries.years, ...ySeries.years])).sort((a, b) => a - b);
      if (!years.length) {
        Plotly.newPlot("percentileChart", [{ text: ["No data"], type: "scatter" }], {}, chartConfig);
        return;
      }

      const currentYear = DATA.meta?.latest_year || Math.max(...years);
      const yearColors = ["#3b82f6", "#8b5cf6", "#d946ef", "#f97316", "#14b8a6", "#f59e0b"];
      const xName = formatCommodityLabel(xCodes);
      const yName = formatCommodityLabel(yCodes);
      const xTitleText = `${xName} ${reportLabel} %OI Long-Short`;
      const yTitleText = `${yName} ${reportLabel} %OI Long-Short`;
      const xTitleSize = axisTitleFontSize(xTitleText, xCodes.length);
      const yTitleSize = axisTitleFontSize(yTitleText, yCodes.length);

      const traces = [];
      const visibleX = [];
      const visibleY = [];
      years.forEach((year, idx) => {
        const xs = [];
        const ys = [];
        const labels = [];
        for (let w = 1; w <= 53; w += 1) {
          const xVal = xSeries.byYear?.[year]?.[w];
          const yVal = ySeries.byYear?.[year]?.[w];
          if (xVal === null || xVal === undefined || yVal === null || yVal === undefined) continue;
          xs.push(xVal);
          ys.push(yVal);
          labels.push(isoWeekDate(year, w, 2).toISOString().slice(0, 10));
        }
        if (!xs.length) return;
        const isCurrent = year === currentYear;
        const isPrior = year === currentYear - 1;
        const color = isCurrent ? "#22c55e" : yearColors[idx % yearColors.length];
        const size = isCurrent ? 9 : 6;
        const visible = (isCurrent || isPrior) ? true : "legendonly";
        if (visible === true) {
          visibleX.push(...xs);
          visibleY.push(...ys);
        }
        traces.push({
          type: "scatter",
          mode: "markers",
          name: year.toString(),
          x: xs,
          y: ys,
          text: labels,
          marker: { color, size, symbol: "circle", opacity: isCurrent ? 0.9 : 0.7 },
          visible,
          hovertemplate: `Year ${year}<br>Date %{text}<br>${xName}: %{x:.2f}%<br>${yName}: %{y:.2f}%<extra></extra>`
        });
      });

      const symmetricRange = (values) => {
        if (!values.length) return [-1, 1];
        const maxVal = Math.max(...values.map(v => Math.abs(v)));
        const span = maxVal === 0 ? 1 : maxVal * 1.05;
        return [-span, span];
      };
      const initialXRange = symmetricRange(visibleX);
      const initialYRange = symmetricRange(visibleY);

      Plotly.newPlot("percentileChart", traces, {
        hovermode: "closest",
        separators: ".,",
        plot_bgcolor: "#0f172a",
        paper_bgcolor: "#0f172a",
        font: { color: "#f8fafc" },
        hoverlabel: { namelength: -1 },
        xaxis: {
          gridcolor: "#1f2a44",
          title: { text: xTitleText, standoff: 10, font: { size: xTitleSize } },
          range: initialXRange,
          zeroline: true,
          zerolinecolor: "#64748b",
          zerolinewidth: 1
        },
        yaxis: {
          gridcolor: "#1f2a44",
          title: { text: yTitleText, standoff: 10, font: { size: yTitleSize } },
          range: initialYRange,
          zeroline: true,
          zerolinecolor: "#64748b",
          zerolinewidth: 1
        },
        legend: { orientation: "h", y: 1.1, x: 0.5, xanchor: "center" }
      }, chartConfig).then((plotDiv) => {
        const updateAxisRange = () => {
          const active = (plotDiv.data || []).filter(trace => trace.visible !== "legendonly" && trace.visible !== false);
          const xs = [];
          const ys = [];
          active.forEach(trace => {
            if (Array.isArray(trace.x)) {
              trace.x.forEach(val => {
                if (val !== null && val !== undefined && !Number.isNaN(val)) xs.push(val);
              });
            }
            if (Array.isArray(trace.y)) {
              trace.y.forEach(val => {
                if (val !== null && val !== undefined && !Number.isNaN(val)) ys.push(val);
              });
            }
          });
          const xRange = symmetricRange(xs);
          const yRange = symmetricRange(ys);
          Plotly.relayout(plotDiv, {"xaxis.range": xRange, "yaxis.range": yRange});
        };

        plotDiv.on("plotly_legendclick", () => {
          setTimeout(updateAxisRange, 0);
          return true;
        });
        plotDiv.on("plotly_legenddoubleclick", () => {
          setTimeout(updateAxisRange, 0);
          return true;
        });
      });
    }

    function renderSummary() {
      summaryGrid.innerHTML = "";
      const meta = DATA.meta || {};
      const cards = [
        { label: "Data Range", value: meta.start_date && meta.end_date ? `${meta.start_date} -> ${meta.end_date}` : "n/a" },
        { label: "Latest Report Date", value: meta.latest_report_date || "n/a" },
        { label: "Eligible Commodities", value: Object.keys(DATA.commodities || {}).length.toLocaleString() },
        { label: "History Years", value: (meta.history_years || []).join(", ") || "n/a" },
      ];
      cards.forEach(card => {
        const div = document.createElement("div");
        div.className = "summary-card";
        div.innerHTML = `<h4>${card.label}</h4><div class="value">${card.value}</div>`;
        summaryGrid.appendChild(div);
      });
    }

    function populateControls() {
      const entries = Object.entries(DATA.commodities || {});
      entries.sort((a, b) => a[1].name.localeCompare(b[1].name));
      commoditySelect.innerHTML = entries.map(([code, info]) => {
        const label = `${info.display_name || info.name} (${info.market_code})`;
        return `<option value="${code}">${label}</option>`;
      }).join("");

      const optionsHtml = entries.map(([code, info]) => {
        const label = `${info.display_name || info.name} (${info.market_code})`;
        return `<option value="${code}">${label}</option>`;
      }).join("");
      xCommoditySelect.innerHTML = optionsHtml;
      yCommoditySelect.innerHTML = optionsHtml;
      if (entries.length) {
        xCommoditySelect.options[0].selected = true;
        const yIndex = entries.length > 1 ? 1 : 0;
        yCommoditySelect.options[yIndex].selected = true;
      }
    }

    function refreshCharts() {
      const reportKey = currentReportKey();
      updateChartTitles(reportKey);
      const codes = ensureSelection();
      const aggregates = buildAggregates(codes, reportKey);
      METRICS.forEach(metric => {
        renderSeasonality(metric.key, metric.chartId, aggregates);
      });
      renderScatter();
    }

    function init() {
      populateControls();
      renderSummary();
      const updated = DATA.meta?.generated_at ? `Last updated: ${DATA.meta.generated_at}` : "";
      document.getElementById("lastUpdated").textContent = updated;

      refreshCharts();
      commoditySelect.addEventListener("change", refreshCharts);
      reportSelect.addEventListener("change", refreshCharts);
      xCommoditySelect.addEventListener("change", renderScatter);
      yCommoditySelect.addEventListener("change", renderScatter);
    }

    function downloadCSV(csvContent, fileName) {
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", fileName);
      link.style.visibility = "hidden";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    function exportSeasonalityData() {
      const codes = ensureSelection();
      const reportKey = currentReportKey();
      const config = reportConfig(reportKey);
      
      const aggregates = buildAggregates(codes, reportKey);
      const { agg, years, maxWeek, useRawPct } = aggregates;
      
      const header = ["Year", "Week", "Date_Approx", "Report_Type", "Commodities", "Long_Contracts", "Short_Contracts", "Net_Contracts", "Open_Interest", "Long_Pct_OI", "Short_Pct_OI", "Net_Pct_OI"];
      const rows = [header.join(",")];
      
      const commodityLabel = formatCommodityLabel(codes).replace(/"/g, '""');
      
      years.forEach(year => {
        for (let w = 1; w <= maxWeek; w++) {
            const l = agg.long[year]?.[w];
            const s = agg.short[year]?.[w];
            const oi = agg.oi[year]?.[w];
            
            if (l == null && s == null) continue;
            
            let realDate = isoWeekDate(year, w, 2).toISOString().slice(0, 10);
            const firstCode = codes[0];
            if (firstCode && DATA.commodities[firstCode]) {
                const sData = DATA.commodities[firstCode];
                 for(let k=0; k<sData.year.length; k++){
                     if(sData.year[k] === year && sData.week[k] === w) {
                         realDate = sData.dates[k];
                         break;
                     }
                 }
            }
            
            const longVal = l || 0;
            const shortVal = s || 0;
            const netVal = longVal - shortVal;
            const oiVal = oi || 0;
            
            let pctL = 0;
            let pctS = 0;
            
            if (useRawPct) {
                pctL = agg.pctLong[year]?.[w] || 0;
                pctS = agg.pctShort[year]?.[w] || 0;
            } else if (oiVal > 0) {
                pctL = (longVal / oiVal) * 100;
                pctS = (shortVal / oiVal) * 100;
            }
            const netPct = pctL - pctS;

            const row = [
                year,
                w,
                realDate,
                config.label,
                `"${commodityLabel}"`,
                longVal,
                shortVal,
                netVal,
                oiVal,
                pctL.toFixed(2),
                pctS.toFixed(2),
                netPct.toFixed(2)
            ];
            rows.push(row.join(","));
        }
      });
      
      downloadCSV(rows.join("\\n"), `seasonality_export_${reportKey}.csv`);
    }

    function exportScatterData() {
       const reportKey = currentReportKey();
       const config = reportConfig(reportKey);
       
       const xCodes = selectedMultiValues(xCommoditySelect, 0);
       const yCodes = selectedMultiValues(yCommoditySelect, xCommoditySelect.options.length > 1 ? 1 : 0);
       
       const xSeries = buildAxisNetPct(xCodes, reportKey);
       const ySeries = buildAxisNetPct(yCodes, reportKey);
       
       const header = ["Year", "Week", "Date_Approx", "Report_Type", "X_Commodities", "Y_Commodities", "X_Net_Pct_OI", "Y_Net_Pct_OI"];
       const rows = [header.join(",")];
       
       const xLabel = formatCommodityLabel(xCodes).replace(/"/g, '""');
       const yLabel = formatCommodityLabel(yCodes).replace(/"/g, '""');

       const years = Array.from(new Set([...xSeries.years, ...ySeries.years])).sort((a, b) => a - b);
       
       years.forEach(year => {
           for(let w=1; w<=53; w++) {
               const xVal = xSeries.byYear?.[year]?.[w];
               const yVal = ySeries.byYear?.[year]?.[w];
               
               if (isValidNumber(xVal) && isValidNumber(yVal)) {
                    let realDate = isoWeekDate(year, w, 2).toISOString().slice(0, 10);
                    const firstCode = xCodes[0];
                    if (firstCode && DATA.commodities[firstCode]) {
                         const sData = DATA.commodities[firstCode];
                         for(let k=0; k<sData.year.length; k++){
                             if(sData.year[k] === year && sData.week[k] === w) {
                                 realDate = sData.dates[k];
                                 break;
                             }
                         }
                    }
                   
                   rows.push([
                       year,
                       w,
                       realDate,
                       config.label,
                       `"${xLabel}"`,
                       `"${yLabel}"`,
                       xVal.toFixed(4),
                       yVal.toFixed(4)
                   ].join(","));
               }
           }
       });
       
       downloadCSV(rows.join("\\n"), `scatter_export_${reportKey}.csv`);
    }

    function waitForPlotly(callback) {
      let attempts = 0;
      const timer = setInterval(() => {
        attempts += 1;
        if (window.Plotly) {
          clearInterval(timer);
          callback();
        } else if (attempts > 200) {
          clearInterval(timer);
          const notice = document.createElement("div");
          notice.className = "summary-card";
          notice.innerHTML = "<h4>Dashboard Error</h4><div class=\\"value\\">Plotly failed to load. Make sure <code>plotly-3.3.1.min.js</code> is next to this HTML file.</div>";
          summaryGrid.innerHTML = "";
          summaryGrid.appendChild(notice);
        }
      }, 50);
    }

    waitForPlotly(init);
  </script>
</body>
</html>
"""
    html = html.replace("__PLOTLY__", plotly_tag)
    return html.replace("__DATA__", data_json)


def build_cftc_disagg_dashboard(
    *,
    output_html: Path | str = CFTC_DASHBOARD_HTML,
    output_export_html: Path | str = CFTC_DASHBOARD_EXPORT_HTML,
    output_json: Path | str = CFTC_DASHBOARD_JSON,
    output_csv: Path | str = CFTC_FILTERED_CSV,
    timeout_seconds: int = 60,
) -> dict:
    data, meta = update_cftc_disagg_cache(timeout_seconds=timeout_seconds)
    payload = build_cftc_disagg_dashboard_payload(data, meta=meta)

    label_map = (
        data.sort_values("report_date")
        .drop_duplicates("CFTC_Contract_Market_Code", keep="last")
        .set_index("CFTC_Contract_Market_Code")
    )
    eligible_df = label_map.reset_index()[
        ["CFTC_Contract_Market_Code", "Market_and_Exchange_Names", "CFTC_Market_Code"]
    ].copy()
    eligible_df["Common_Name"] = eligible_df.apply(
        lambda row: _resolve_display_name(
            str(row["CFTC_Contract_Market_Code"]), str(row["Market_and_Exchange_Names"])
        ),
        axis=1,
    )
    eligible_df = eligible_df.sort_values("Market_and_Exchange_Names").reset_index(drop=True)
    CFTC_ELIGIBLE_COMMODITIES_CSV.parent.mkdir(parents=True, exist_ok=True)
    eligible_df.to_csv(CFTC_ELIGIBLE_COMMODITIES_CSV, index=False)

    display_map = {
        str(code): _resolve_display_name(str(code), str(row["Market_and_Exchange_Names"]))
        for code, row in label_map.iterrows()
    }

    report_configs = {
        "managed": {
            "label": "Managed Money",
            "long": "M_Money_Positions_Long_All",
            "short": "M_Money_Positions_Short_All",
            "pct_long": "Pct_of_OI_M_Money_Long_All",
            "pct_short": "Pct_of_OI_M_Money_Short_All",
        },
        "producer": {
            "label": "Producer",
            "long": "Prod_Merc_Positions_Long_All",
            "short": "Prod_Merc_Positions_Short_All",
            "pct_long": "Pct_of_OI_Prod_Merc_Long_All",
            "pct_short": "Pct_of_OI_Prod_Merc_Short_All",
        },
        "swap": {
            "label": "Swap Dealers",
            "long": "Swap_Positions_Long_All",
            "short": "Swap_Positions_Short_All",
            "pct_long": "Pct_of_OI_Swap_Long_All",
            "pct_short": "Pct_of_OI_Swap_Short_All",
        },
    }
    base_cols = [
        "report_date",
        "report_year",
        "report_week",
        "CFTC_Contract_Market_Code",
        "CFTC_Market_Code",
        "Market_and_Exchange_Names",
        "Open_Interest_All",
    ]
    seasonal_frames = []
    scatter_frames = []
    for key, cfg in report_configs.items():
        frame = data[base_cols + [cfg["long"], cfg["short"], cfg["pct_long"], cfg["pct_short"]]].copy()
        frame["report_type"] = key
        frame["report_label"] = cfg["label"]
        frame["Common_Name"] = frame["CFTC_Contract_Market_Code"].map(display_map)
        frame = frame.rename(
            columns={
                cfg["long"]: "positions_long",
                cfg["short"]: "positions_short",
                cfg["pct_long"]: "pct_long",
                cfg["pct_short"]: "pct_short",
            }
        )
        seasonal_frames.append(frame)

        scatter = frame[
            base_cols + ["positions_long", "positions_short", "report_type", "report_label", "Common_Name"]
        ].copy()
        oi = scatter["Open_Interest_All"].replace(0, pd.NA)
        scatter["net_pct"] = ((scatter["positions_long"] - scatter["positions_short"]) / oi) * 100.0
        scatter_frames.append(scatter)

    if seasonal_frames:
        seasonality_export = pd.concat(seasonal_frames, ignore_index=True)
        seasonality_export.to_csv(CFTC_SEASONALITY_EXPORT_CSV, index=False)
    if scatter_frames:
        scatter_export = pd.concat(scatter_frames, ignore_index=True)
        scatter_export.to_csv(CFTC_SCATTER_EXPORT_CSV, index=False)

    export_df = data[
        [
            "report_date",
            "report_year",
            "report_week",
            "CFTC_Contract_Market_Code",
            "CFTC_Market_Code",
            "Market_and_Exchange_Names",
            "Open_Interest_All",
            "M_Money_Positions_Long_All",
            "M_Money_Positions_Short_All",
            "Pct_of_OI_M_Money_Long_All",
            "Pct_of_OI_M_Money_Short_All",
            "Prod_Merc_Positions_Long_All",
            "Prod_Merc_Positions_Short_All",
            "Pct_of_OI_Prod_Merc_Long_All",
            "Pct_of_OI_Prod_Merc_Short_All",
            "Swap_Positions_Long_All",
            "Swap_Positions_Short_All",
            "Pct_of_OI_Swap_Long_All",
            "Pct_of_OI_Swap_Short_All",
        ]
    ].copy()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_csv, index=False)

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_build_dashboard_html(payload), encoding="utf-8", errors="ignore")
    output_export_html = Path(output_export_html)
    output_export_html.parent.mkdir(parents=True, exist_ok=True)
    output_export_html.write_text(
        _build_dashboard_html(payload, inline_plotly=True),
        encoding="utf-8",
        errors="ignore",
    )
    return payload


__all__ = [
    "resolve_latest_cftc_year",
    "update_cftc_disagg_cache",
    "build_cftc_disagg_dashboard_payload",
    "build_cftc_disagg_dashboard",
    "CFTC_DASHBOARD_HTML",
    "CFTC_DASHBOARD_EXPORT_HTML",
    "CFTC_DASHBOARD_JSON",
    "CFTC_FILTERED_CSV",
    "CFTC_ELIGIBLE_COMMODITIES_CSV",
    "CFTC_SEASONALITY_EXPORT_CSV",
    "CFTC_SCATTER_EXPORT_CSV",
]
