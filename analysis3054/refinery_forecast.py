"""Refinery forecast utilities built on Chronos2 with maintenance covariates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .forecasting import chronos2_multivariate_forecast, chronos2_quantile_forecast
from .refinery_name_maps import apply_la_refinery_name_map_to_df
from .tx_refineries import _normalize_refinery_name_key


@dataclass(frozen=True)
class ForecastOutputs:
    refinery_data: pd.DataFrame
    forecast: pd.DataFrame


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
REFINERY_DIR = DATA_DIR / "refinery"
TX_DIR = DATA_DIR / "tx_refineries_transfers"

LA_PATH = REFINERY_DIR / "la_refinery_latest.csv"
TX_PATH = TX_DIR / "tx_refineries_2021_present.csv"
MAINTENANCE_PATH = REFINERY_DIR / "refinery_maintenance.csv"
TX_NAME_MAP_PATH = TX_DIR / "tx_refinery_name_map.csv"

DEFAULT_OUTPUT_DATA = REFINERY_DIR / "refinery_data.csv"
DEFAULT_OUTPUT_FORECAST = REFINERY_DIR / "refinery_forecast.csv"
DEFAULT_OUTPUT_PRODUCT_FORECAST = REFINERY_DIR / "refinery_product_forecast.csv"
FORECAST_EXCLUDE_REFINERIES = {"lyondell houston"}

UNIT_KEYWORDS = {
    "coker": "Coker",
    "atmos": "Atmos Distillation",
    "distill": "Atmos Distillation",
    "hydrocracker": "Hydrocracker",
    "fcc": "FCC",
    "alky": "Alkylation",
    "reform": "Reforming",
}


def _clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
        .fillna(0.0)
    )


def _parse_date(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _is_crude(product: str) -> bool:
    text = str(product).lower()
    return "crude" in text and "total" not in text


def _is_total(product: str) -> bool:
    return "total" in str(product).lower()


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col in df.columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        mapping[col] = normalized
    return mapping


def _apply_tx_name_map(series: pd.Series) -> pd.Series:
    if not TX_NAME_MAP_PATH.exists():
        return series
    mapping_df = pd.read_csv(TX_NAME_MAP_PATH)
    if mapping_df.empty:
        return series
    mapping = dict(
        zip(
            mapping_df["refinery_name_key"].astype(str),
            mapping_df["refinery_name_canonical"].astype(str),
        )
    )
    keys = series.map(_normalize_refinery_name_key)
    mapped = keys.map(mapping)
    return mapped.where(mapped.notna() & (mapped.str.len() > 0), series)


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]
    return None


def _load_refinery_base() -> pd.DataFrame:
    la_df = pd.read_csv(LA_PATH)
    la_df = apply_la_refinery_name_map_to_df(la_df)
    tx_df = pd.read_csv(TX_PATH)

    def prep(df: pd.DataFrame, *, date_col: str, refinery_col: str, product_col: str,
             production_col: str, crude_col: str, date_fmt: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df["month"] = _parse_date(df[date_col], fmt=date_fmt)
        df = df[df["month"].notna()].copy()
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
        df["product"] = df[product_col].astype(str).str.strip()
        df["refinery_name"] = df[refinery_col].astype(str).str.strip()
        df = df[~df["product"].map(_is_total)].copy()
        df["is_crude"] = df["product"].map(_is_crude)
        df[production_col] = _clean_numeric(df[production_col])
        df[crude_col] = _clean_numeric(df[crude_col])
        df["days_in_month"] = df["month"].dt.daysinmonth
        df["production_kbd"] = df[production_col] / df["days_in_month"] / 1000.0
        df["crude_kbd"] = df[crude_col] / df["days_in_month"] / 1000.0

        crude = (
            df[df["is_crude"]]
            .groupby(["refinery_name", "month"])["crude_kbd"]
            .sum()
        )
        output = (
            df[~df["is_crude"]]
            .groupby(["refinery_name", "month"])["production_kbd"]
            .sum()
        )
        combined = (
            pd.concat([output, crude], axis=1)
            .rename(columns={"production_kbd": "total_output_kbd"})
            .reset_index()
        )
        combined["crude_kbd"] = combined["crude_kbd"].fillna(0.0)
        combined["yield_pct"] = np.where(
            combined["crude_kbd"] > 0,
            (combined["total_output_kbd"] / combined["crude_kbd"]) * 100.0,
            0.0,
        )
        return combined

    la_prepped = prep(
        la_df,
        date_col="Report Date",
        refinery_col="Refinery Name",
        product_col="R3 Product Code Description",
        production_col="Production",
        crude_col="Distillations",
        date_fmt="%d-%b-%Y",
    )
    tx_prepped = prep(
        tx_df,
        date_col="statement_date",
        refinery_col="refinery_name",
        product_col="Name of Material",
        production_col="Products Manufactured",
        crude_col="Input Runs to Stills and/or Blends",
    )
    return pd.concat([la_prepped, tx_prepped], ignore_index=True)


def _load_refinery_product_series() -> pd.DataFrame:
    la_df = pd.read_csv(LA_PATH)
    la_df = apply_la_refinery_name_map_to_df(la_df)
    tx_df = pd.read_csv(TX_PATH)

    def prep(
        df: pd.DataFrame,
        *,
        date_col: str,
        refinery_col: str,
        product_col: str,
        production_col: str,
        crude_col: str,
        date_fmt: Optional[str] = None,
        apply_tx_map: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        df["month"] = _parse_date(df[date_col], fmt=date_fmt)
        if df["month"].isna().any() and {"statement_year", "statement_month"}.issubset(df.columns):
            missing = df["month"].isna()
            df.loc[missing, "month"] = pd.to_datetime(
                {
                    "year": pd.to_numeric(df.loc[missing, "statement_year"], errors="coerce"),
                    "month": pd.to_numeric(df.loc[missing, "statement_month"], errors="coerce"),
                    "day": 1,
                },
                errors="coerce",
            )
        df = df[df["month"].notna()].copy()
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
        df["product"] = df[product_col].astype(str).str.strip()
        refinery_series = df[refinery_col].astype(str).str.strip()
        if apply_tx_map:
            refinery_series = _apply_tx_name_map(refinery_series)
        df["refinery_name"] = refinery_series
        df = df[~df["product"].map(_is_total)].copy()
        df["is_crude"] = df["product"].map(_is_crude)
        df[production_col] = _clean_numeric(df[production_col])
        df[crude_col] = _clean_numeric(df[crude_col])
        df["days_in_month"] = df["month"].dt.daysinmonth
        df["production_kbd"] = df[production_col] / df["days_in_month"] / 1000.0
        df["crude_kbd"] = df[crude_col] / df["days_in_month"] / 1000.0
        return df

    la_prepped = prep(
        la_df,
        date_col="Report Date",
        refinery_col="Refinery Name",
        product_col="R3 Product Code Description",
        production_col="Production",
        crude_col="Distillations",
        date_fmt="%d-%b-%Y",
    )
    tx_prepped = prep(
        tx_df,
        date_col="statement_date",
        refinery_col="refinery_name",
        product_col="Name of Material",
        production_col="Products Manufactured",
        crude_col="Input Runs to Stills and/or Blends",
        apply_tx_map=True,
    )

    combined = pd.concat([la_prepped, tx_prepped], ignore_index=True)
    product_rows = (
        combined[~combined["is_crude"]]
        .groupby(["refinery_name", "month", "product"])["production_kbd"]
        .sum()
        .reset_index()
        .rename(columns={"month": "date", "product": "series", "production_kbd": "value"})
    )
    crude_rows = (
        combined[combined["is_crude"]]
        .groupby(["refinery_name", "month"])["crude_kbd"]
        .sum()
        .reset_index()
        .rename(columns={"month": "date", "crude_kbd": "value"})
    )
    crude_rows["series"] = "crude_kbd"
    return pd.concat([product_rows, crude_rows], ignore_index=True)


def _load_maintenance() -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    if not MAINTENANCE_PATH.exists():
        return pd.DataFrame(), [], {}
    maint = pd.read_csv(MAINTENANCE_PATH)
    col_map = _normalize_columns(maint)
    maint = maint.rename(columns=col_map)

    date_col = _resolve_column(maint, ["date", "month", "statement_date", "report_date"])
    refinery_col = _resolve_column(maint, ["refinery_name", "refinery", "operator", "name"])
    if date_col is None or refinery_col is None:
        return pd.DataFrame(), [], {}

    maint = maint.rename(columns={date_col: "date", refinery_col: "refinery_name"})
    maint["date"] = _parse_date(maint["date"])
    maint = maint[maint["date"].notna()].copy()
    maint["date"] = maint["date"].dt.to_period("M").dt.to_timestamp()
    maint["refinery_name"] = maint["refinery_name"].astype(str).str.strip()

    covariate_cols = []
    label_map = {}
    for col in maint.columns:
        if col in {"date", "refinery_name"}:
            continue
        if not pd.api.types.is_numeric_dtype(maint[col]):
            maint[col] = _clean_numeric(maint[col])
        covariate_cols.append(col)

        label = None
        lower = col.lower()
        for key, title in UNIT_KEYWORDS.items():
            if key in lower:
                status = None
                if "planned" in lower:
                    status = "Planned"
                elif "unplanned" in lower:
                    status = "Unplanned"
                label = f"{title} ({status})" if status else title
                break
        if label:
            label_map[col] = label

    return maint, covariate_cols, label_map


def _build_refinery_dataset(
    base_df: pd.DataFrame,
    maint_df: pd.DataFrame,
    covariate_cols: List[str],
) -> pd.DataFrame:
    df = base_df.copy()
    df = df.rename(columns={"month": "date"})
    if maint_df.empty or not covariate_cols:
        for col in covariate_cols:
            df[col] = 0.0
        return df

    maint_grouped = (
        maint_df.groupby(["refinery_name", "date"])[covariate_cols]
        .sum()
        .reset_index()
    )
    merged = df.merge(maint_grouped, on=["refinery_name", "date"], how="left")
    for col in covariate_cols:
        merged[col] = merged[col].fillna(0.0)
    return merged


def _append_future_covariates(
    dataset: pd.DataFrame,
    maint_df: pd.DataFrame,
    covariate_cols: List[str],
) -> pd.DataFrame:
    if maint_df.empty or not covariate_cols:
        return dataset

    rows = []
    for refinery, group in dataset.groupby("refinery_name"):
        last_date = group["date"].max()
        future = maint_df[
            (maint_df["refinery_name"].str.lower() == refinery.lower())
            & (maint_df["date"] > last_date)
        ]
        if future.empty:
            continue
        future = future.copy()
        future["total_output_kbd"] = np.nan
        future["crude_kbd"] = np.nan
        future["yield_pct"] = np.nan
        cols = ["refinery_name", "date", "total_output_kbd", "crude_kbd", "yield_pct"] + covariate_cols
        rows.append(future[cols])
    if rows:
        dataset = pd.concat([dataset, *rows], ignore_index=True)
    return dataset


def _extract_quantiles(
    forecast_df: pd.DataFrame,
    quantiles: Tuple[float, float, float],
) -> pd.DataFrame:
    q20, q50, q80 = quantiles
    df = forecast_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [float(str(c)) for c in df.columns]
    return pd.DataFrame({
        "date": df.index,
        "p20": df.get(q20, pd.Series(index=df.index, dtype=float)),
        "p50": df.get(q50, pd.Series(index=df.index, dtype=float)),
        "p80": df.get(q80, pd.Series(index=df.index, dtype=float)),
    })


def _extract_median_forecast(forecast_df: pd.DataFrame, *, quantile: float = 0.5) -> pd.DataFrame:
    df = forecast_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        levels = [float(str(level)) for level in df.columns.get_level_values(-1)]
        unique_levels = sorted(set(levels))
        if not unique_levels:
            raise ValueError("No quantiles found in forecast output.")
        closest = min(unique_levels, key=lambda val: abs(val - quantile))
        df = df.xs(closest, level=-1, axis=1)
    df.columns = [str(col) for col in df.columns]
    df.index.name = "date"
    return df


def build_refinery_forecast(
    *,
    output_data: Path = DEFAULT_OUTPUT_DATA,
    output_forecast: Path = DEFAULT_OUTPUT_FORECAST,
    horizon_default: int = 6,
    quantiles: Tuple[float, float, float] = (0.2, 0.5, 0.8),
) -> ForecastOutputs:
    base_df = _load_refinery_base()
    maint_df, covariate_cols, _ = _load_maintenance()
    dataset = _build_refinery_dataset(base_df, maint_df, covariate_cols)
    dataset = _append_future_covariates(dataset, maint_df, covariate_cols)
    dataset = dataset.sort_values(["refinery_name", "date"]).reset_index(drop=True)
    output_data.write_text(dataset.to_csv(index=False), encoding="utf-8")

    forecast_rows = []
    use_covariates = not maint_df.empty and bool(covariate_cols)

    for refinery, group in dataset.groupby("refinery_name"):
        if str(refinery).strip().lower() in FORECAST_EXCLUDE_REFINERIES:
            continue
        history = group[group["total_output_kbd"].notna()].copy()
        if history.empty or len(history) < 12:
            continue

        prediction_length = horizon_default
        future_cov_df = None
        covariate_cols_used = covariate_cols if use_covariates else None
        if use_covariates:
            last_date = history["date"].max()
            future_dates = pd.date_range(
                start=last_date + pd.offsets.MonthBegin(1),
                periods=prediction_length,
                freq="MS",
            )
            future_cov = (
                maint_df[maint_df["refinery_name"].str.lower() == refinery.lower()]
                .groupby("date")[covariate_cols]
                .sum()
                .reindex(future_dates)
                .fillna(0.0)
                .reset_index()
            )
            future_cov_df = future_cov.rename(columns={"index": "date"})

        for metric in ["total_output_kbd", "yield_pct"]:
            metric_cols = ["date", metric] + (covariate_cols if use_covariates else [])
            df_metric = history[metric_cols].copy()
            df_metric = df_metric.rename(columns={metric: "target"})
            try:
                res = chronos2_quantile_forecast(
                    df_metric,
                    date_col="date",
                    target_col="target",
                    covariate_cols=covariate_cols_used,
                    future_cov_df=future_cov_df,
                    prediction_length=prediction_length,
                    quantile_levels=list(quantiles),
                    device_map="cpu",
                )
            except Exception:
                continue
            quant_df = _extract_quantiles(res.forecasts, quantiles)
            quant_df["refinery_name"] = refinery
            quant_df["metric"] = metric
            forecast_rows.append(quant_df)

    forecast = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    if forecast.empty and output_forecast.exists():
        try:
            existing = pd.read_csv(output_forecast)
        except Exception:
            existing = pd.DataFrame()
        return ForecastOutputs(refinery_data=dataset, forecast=existing)
    output_forecast.write_text(forecast.to_csv(index=False), encoding="utf-8")
    return ForecastOutputs(refinery_data=dataset, forecast=forecast)


def build_refinery_product_forecast(
    *,
    output_forecast: Path = DEFAULT_OUTPUT_PRODUCT_FORECAST,
    horizon_default: int = 6,
    quantile: float = 0.5,
) -> pd.DataFrame:
    series_df = _load_refinery_product_series()
    forecast_rows = []

    for refinery, group in series_df.groupby("refinery_name"):
        if str(refinery).strip().lower() in FORECAST_EXCLUDE_REFINERIES:
            continue
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"], errors="coerce")
        group = group[group["date"].notna()]
        if group.empty:
            continue

        pivot = (
            group.pivot_table(index="date", columns="series", values="value", aggfunc="sum")
            .sort_index()
        )
        if pivot.empty:
            continue
        months = pd.date_range(pivot.index.min(), pivot.index.max(), freq="MS")
        pivot = pivot.reindex(months, fill_value=0.0)
        if len(pivot) < 12:
            continue
        df_wide = pivot.reset_index().rename(columns={"index": "date"})
        target_cols = [col for col in df_wide.columns if col != "date"]
        if not target_cols:
            continue

        try:
            res = chronos2_multivariate_forecast(
                df_wide,
                date_col="date",
                target_cols=target_cols,
                prediction_length=horizon_default,
                quantile_levels=[quantile],
                device_map="cpu",
            )
            forecast_wide = _extract_median_forecast(res.forecasts, quantile=quantile)
        except Exception:
            last_values = df_wide[target_cols].iloc[-1].fillna(0.0)
            forecast_dates = pd.date_range(
                df_wide["date"].iloc[-1] + pd.offsets.MonthBegin(1),
                periods=horizon_default,
                freq="MS",
            )
            forecast_wide = pd.DataFrame(
                [last_values.values] * horizon_default,
                index=forecast_dates,
                columns=target_cols,
            )

        forecast_long = (
            forecast_wide.reset_index()
            .rename(columns={"index": "date"})
            .melt(id_vars="date", var_name="series", value_name="p50")
        )
        forecast_long["refinery_name"] = str(refinery)
        forecast_rows.append(forecast_long)

    forecast = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    output_forecast.write_text(forecast.to_csv(index=False), encoding="utf-8")
    return forecast


__all__ = ["build_refinery_forecast", "build_refinery_product_forecast", "ForecastOutputs"]
