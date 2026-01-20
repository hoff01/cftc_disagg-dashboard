# CFTC Disaggregated Dashboard

## Workspace integration

This repo is expected at `$ANALYSIS3054_DASHBOARD_HOME/dashboard_repos/cftc_disagg`. Automated update/build scripts live in `$ANALYSIS3054_DASHBOARD_HOME/scripts` and use the analysis code from `$ANALYSIS3054_CODEX_HOME` (default `$ANALYSIS3054_DASHBOARD_HOME`).

Builds the CFTC disaggregated Commitments of Traders dashboard (CFTC + ICE history).

## Layout

data/
  build_dashboard.py
  update_data.py
  requirements.txt
  *.csv (inputs)
  *.html (outputs)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r data/requirements.txt
```

## Build

```bash
python data/build_dashboard.py
```

## Inputs

- Cached CFTC text files in `data/cache/`
- ICE history in `data/ice_cache/`
- Optional filters in `data/cftc_disagg_eligible_commodities.csv`

## Outputs

- `data/cftc_disagg_dashboard.html`
- `data/cftc_disagg_dashboard_export.html`
- `data/cftc_disagg_dashboard_data.json`
- `data/cftc_disagg_filtered.csv`
- `data/cftc_disagg_scatter_export.csv`
- `data/cftc_disagg_seasonality_export.csv`

## Update data

```bash
python data/update_data.py
```

## Notes

All dashboards load Plotly 3.3.1 from the CDN. Export HTML files inline Plotly for offline use when present.
