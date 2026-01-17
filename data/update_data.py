#!/usr/bin/env python3
"""Update CFTC disaggregated data and rebuild the dashboard outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import analysis3054.cftc_disagg_dashboard as cftc

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _configure_paths() -> Path:
    data_dir = DATA_DIR
    cftc.DATA_DIR = data_dir
    cftc.CFTC_DASHBOARD_DIR = data_dir
    cftc.CFTC_CACHE_DIR = data_dir / "cache"
    cftc.CFTC_CACHE_META = cftc.CFTC_CACHE_DIR / "cache_meta.json"
    cftc.ICE_CACHE_DIR = data_dir / "ice_cache"
    cftc.ICE_CACHE_META = data_dir / "ice_cache" / "cache_meta.json"
    cftc.CFTC_FILTERED_CSV = data_dir / "cftc_disagg_filtered.csv"
    cftc.CFTC_DASHBOARD_JSON = data_dir / "cftc_disagg_dashboard_data.json"
    cftc.CFTC_DASHBOARD_HTML = data_dir / "cftc_disagg_dashboard.html"
    cftc.CFTC_DASHBOARD_EXPORT_HTML = data_dir / "cftc_disagg_dashboard_export.html"
    cftc.CFTC_ELIGIBLE_COMMODITIES_CSV = data_dir / "cftc_disagg_eligible_commodities.csv"
    cftc.CFTC_SEASONALITY_EXPORT_CSV = data_dir / "cftc_disagg_seasonality_export.csv"
    cftc.CFTC_SCATTER_EXPORT_CSV = data_dir / "cftc_disagg_scatter_export.csv"
    return data_dir


def main() -> None:
    _configure_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Update CFTC disaggregated data (downloads when cache is missing) and rebuild the dashboard."
        )
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=cftc.CFTC_DASHBOARD_HTML,
        help="Output path for the dashboard HTML.",
    )
    parser.add_argument(
        "--output-export-html",
        type=Path,
        default=cftc.CFTC_DASHBOARD_EXPORT_HTML,
        help="Output path for the exportable dashboard HTML.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=cftc.CFTC_DASHBOARD_JSON,
        help="Output path for the dashboard JSON payload.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=cftc.CFTC_FILTERED_CSV,
        help="Output path for the filtered CSV export.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds for data downloads.",
    )
    args = parser.parse_args()

    cftc.build_cftc_disagg_dashboard(
        output_html=args.output_html,
        output_export_html=args.output_export_html,
        output_json=args.output_json,
        output_csv=args.output_csv,
        timeout_seconds=args.timeout,
    )

    print(f"Dashboard built at: {args.output_html}")
    print(f"Exportable dashboard built at: {args.output_export_html}")
    print(f"Payload JSON written at: {args.output_json}")
    print(f"Filtered CSV written at: {args.output_csv}")


if __name__ == "__main__":
    main()
