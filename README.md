# CFTC Disaggregated Dashboard

Builds the CFTC disaggregated Commitments of Traders dashboard (CFTC + ICE history).

## Layout

analysis3054/data/cftc_disagg/
  build_dashboard.py
  update_data.py
  requirements.txt
  *.csv (inputs)
  *.html (outputs)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r analysis3054/data/cftc_disagg/requirements.txt
```

## Build

```bash
python analysis3054/data/cftc_disagg/build_dashboard.py
```

## Inputs

- Cached CFTC text files in `analysis3054/data/cftc_disagg/cache/`
- ICE history in `analysis3054/data/cftc_disagg/ice_cache/`
- Optional filters in `analysis3054/data/cftc_disagg/cftc_disagg_eligible_commodities.csv`

## Outputs

- `analysis3054/data/cftc_disagg/cftc_disagg_dashboard.html`
- `analysis3054/data/cftc_disagg/cftc_disagg_dashboard_export.html`
- `analysis3054/data/cftc_disagg/cftc_disagg_dashboard_data.json`
- `analysis3054/data/cftc_disagg/cftc_disagg_filtered.csv`
- `analysis3054/data/cftc_disagg/cftc_disagg_scatter_export.csv`
- `analysis3054/data/cftc_disagg/cftc_disagg_seasonality_export.csv`

## Update data

```bash
python analysis3054/data/cftc_disagg/update_data.py
```

## Notes

All dashboards load Plotly 3.3.1 from the CDN. Export HTML files inline Plotly for offline use when present.
