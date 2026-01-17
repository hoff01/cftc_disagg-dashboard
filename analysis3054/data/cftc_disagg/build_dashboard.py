#!/usr/bin/env python3
"""Build the CFTC disaggregated dashboard with configurable outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from analysis3054.cftc_disagg_dashboard import (  # noqa: E402
    CFTC_DASHBOARD_EXPORT_HTML,
    CFTC_DASHBOARD_HTML,
    CFTC_DASHBOARD_JSON,
    CFTC_FILTERED_CSV,
    build_cftc_disagg_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the CFTC disaggregated dashboard. "
            "Data is downloaded automatically when cache files are missing."
        )
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=CFTC_DASHBOARD_HTML,
        help="Output path for the dashboard HTML.",
    )
    parser.add_argument(
        "--output-export-html",
        type=Path,
        default=CFTC_DASHBOARD_EXPORT_HTML,
        help="Output path for the exportable (inline) dashboard HTML.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=CFTC_DASHBOARD_JSON,
        help="Output path for the dashboard JSON payload.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=CFTC_FILTERED_CSV,
        help="Output path for the filtered CSV export.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds for data downloads.",
    )
    args = parser.parse_args()

    build_cftc_disagg_dashboard(
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
