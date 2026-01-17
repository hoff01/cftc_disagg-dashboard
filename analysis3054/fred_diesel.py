"""FRED diesel-related macro dataset (monthly + quarterly)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
from typing import Iterable, Optional

import pandas as pd
import pandas_datareader.data as web

from .diesel_indicators import DIESEL_DEMAND_IDS

FRED_API_ENV = "FRED_API_KEY"

FRED_TICKERS_BASE = [
    # --- ALTERNATIVE TRANSFORMS & FREQUENCY ---
    "HTRUCKSSAAR",  # Heavy Truck Sales (Annual Rate)
    "DAUTOSAAR",    # Auto Sales (Annual Rate)
    "ALTSALES",     # Light Vehicle Sales
    "WPU057303",    # PPI: #2 Diesel Fuel
    "WPU0561",      # PPI: Crude Petroleum (Domestic)
    "PNRGINDEXM",   # PPI: Energy Index
    
    # --- ADDITIONAL MACRO DRIVERS ---
    "A191RL1Q225SBEA", # Real GDP (QoQ %)
    "GDPPOT",       # Real Potential GDP
    "NROU",         # Natural Rate of Unemployment
    "CFNAI",        # Chicago Fed National Activity Index
    "RECPROUSM156N",# Smoothed Recession Probabilities
    "AWHAEMAN",     # Avg Weekly Hours: All Employees Mfg
    "AWHMAN",       # Avg Weekly Hours: Manufacturing
    "ICSA",         # Initial Claims
    "CCSA",         # Continued Claims
]

FRED_TICKERS_EXTRA = [
    # --- ENERGY SPECIFIC ---
    "MCOILWTICO",   # Monthly Crude Oil Prices: WTI
    "MCOILBRENTEU", # Monthly Crude Oil Prices: Brent
    "POILWTIUSDM",  # Global Price of WTI Crude
    "POILBREUSDM",  # Global Price of Brent Crude
    "CPIENGSL",     # CPI: Energy
    "CPIUFDSL",     # CPI: Food
    "WPU055",       # PPI: Fuels and Related Products and Power

    # --- DETAILED MANUFACTURING & INDUSTRIAL ---
    "IPCONGD",      # IP: Consumer Goods
    "IPFINAL",      # IP: Final Products
    "IPUTIL",       # IP: Utilities
    "IPBUSEQ",      # IP: Business Equipment
    "IPDMAT",       # IP: Durable Materials
    "MCUMFN",       # Capacity Utilization: Manufacturing

    # --- FINANCIAL CONDITIONS & SPREADS (ML FEATURES) ---
    "T10Y3M",       # 10-Year Minus 3-Month Treasury Yield Spread
    "T5YIE",        # 5-Year Breakeven Inflation Rate
    "AAA",          # Moody's Seasoned Aaa Corporate Bond Yield
    "BAA",          # Moody's Seasoned Baa Corporate Bond Yield
    "BAA10Y",       # Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury
    "BAMLH0A0HYM2", # ICE BofA US High Yield Index Option-Adjusted Spread
    "VIXCLS",       # CBOE Volatility Index (VIX)
    "SP500",        # S&P 500
    "DJIA",         # Dow Jones Industrial Average
    "NFCI",         # National Financial Conditions Index
    "GVZCLS",       # CBOE Gold ETF Volatility Index
    "OVXCLS",       # CBOE Crude Oil ETF Volatility Index
    
    # --- HOUSING REGIONS & RATES ---
    "HOUSTMW",      # Housing Starts: Midwest
    "HOUSTNE",      # Housing Starts: Northeast
    "HOUSTS",       # Housing Starts: South
    "HOUSTW",       # Housing Starts: West
    "CSUSHPINSA",   # S&P/Case-Shiller U.S. National Home Price Index
    "MORTGAGE30US", # 30-Year Fixed Rate Mortgage Average in the United States
    
    # --- MONEY & CREDIT ---
    "M1SL",         # M1 Money Stock
    "WALCL",        # Total Assets of the Fed
    "REVOLSL",      # Total Revolving Credit
    "PSAVERT",      # Personal Saving Rate
]


class FREDDataError(Exception):
    """Raised when FRED data extraction fails."""


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    env_key = os.getenv(FRED_API_ENV, "").strip()
    return env_key or None


def build_fred_tickers(extra: Optional[Iterable[str]] = None) -> list[str]:
    tickers = list(FRED_TICKERS_BASE)
    tickers.extend(FRED_TICKERS_EXTRA)
    tickers.extend(DIESEL_DEMAND_IDS)
    if extra:
        tickers.extend(extra)
    unique = []
    seen = set()
    for ticker in tickers:
        if ticker not in seen:
            unique.append(ticker)
            seen.add(ticker)
    return unique


def fetch_fred_data(
    tickers: Iterable[str],
    *,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch FRED data. Returns dataframe + list of missing tickers."""
    end = pd.to_datetime(end_date) if end_date else datetime.now()
    tickers_list = list(tickers)
    missing: list[str] = []
    api_key = _resolve_api_key(api_key)

    try:
        df = web.DataReader(tickers_list, "fred", start_date, end, api_key=api_key)
        df = df.ffill()
        return df, missing
    except Exception:
        series = {}
        for ticker in tickers_list:
            try:
                data = web.DataReader(ticker, "fred", start_date, end, api_key=api_key)
                if isinstance(data, pd.DataFrame):
                    series[ticker] = data[ticker] if ticker in data.columns else data.iloc[:, 0]
                else:
                    series[ticker] = data
            except Exception:
                missing.append(ticker)
        if not series:
            raise FREDDataError("Failed to fetch any FRED series.")
        df = pd.concat(series, axis=1).sort_index().ffill()
        return df, missing


def resample_fred_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = df.resample("ME").mean()
    quarterly = df.resample("QE").mean()
    return monthly, quarterly


def update_fred_diesel_cache(
    *,
    output_monthly: str | Path,
    output_quarterly: str | Path,
    start_date: str = "2008-01-01",
    api_key: Optional[str] = None,
    extra_tickers: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tickers = build_fred_tickers(extra=extra_tickers)
    df, missing = fetch_fred_data(tickers, start_date=start_date, api_key=api_key)
    monthly, quarterly = resample_fred_data(df)

    Path(output_monthly).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_monthly)
    quarterly.to_csv(output_quarterly)
    return monthly, quarterly, missing


__all__ = [
    "build_fred_tickers",
    "fetch_fred_data",
    "resample_fred_data",
    "update_fred_diesel_cache",
]
