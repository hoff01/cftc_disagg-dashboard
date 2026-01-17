# Analysis3054

Analysis3054 is a data pipeline + dashboard toolkit for energy, commodities, and demand planning. Each dashboard lives in its own folder under `analysis3054/data/<dashboard>/` with a build script, update helper, requirements, and its input/output files.

## Repository layout

Each dashboard folder is self-contained so you can copy it elsewhere:

```
analysis3054/data/<dashboard>/
  build_dashboard.py
  update_data.py
  requirements.txt
  *.csv (inputs)
  *.html (outputs)
```

## Install (repo)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Per-dashboard dependencies live in each `requirements.txt`:

```bash
pip install -r analysis3054/data/kayross/requirements.txt
```

All dashboards load Plotly JS from the Plotly 3.3.1 CDN (export HTML inlines Plotly for offline use).

## Dashboards

### 1) Vehicle Miles & Fuel Usage (Kayross + DTN)

Folder: `analysis3054/data/kayross/`

Inputs (4 CSVs):
- `kayross_us.csv` columns: `VALUE_DATE`, `VALUE_DIESEL`, `VALUE_GASOLINE`, `REGION`
- `kayross_eu.csv` columns: `VALUE_DATE`, `COUNTRY`, `VALUE_DIESEL_CONSUMPTION`, `VALUE_GASOLINE_CONSUMPTION`
- `dtn_refined_fuels.csv` (PADD) columns: `effectiveDateTime`, `region`, `grade`, `sumNetVolume`, `coverageFactor`
- `dtn_refined_fuels_rack.csv` (rack/state) columns: `effectiveDateTime`, `rackId`, `rackName`, `state`, `region`, `grade`, `sumNetVolume`

Build:
```bash
python analysis3054/data/kayross/build_dashboard.py \
  --na-csv analysis3054/data/kayross/kayross_us.csv \
  --eu-csv analysis3054/data/kayross/kayross_eu.csv \
  --padd-csv analysis3054/data/kayross/dtn_refined_fuels.csv \
  --rack-csv analysis3054/data/kayross/dtn_refined_fuels_rack.csv
```

Outputs:
- `analysis3054/data/kayross/kayross_dashboard.html`
- `analysis3054/data/kayross/kayross_dashboard_export.html` (inline Plotly)

Chronos2 forecast:
```bash
KAYROSS_ENABLE_CHRONOS2=1 python analysis3054/data/kayross/build_dashboard.py
```

DTN updater (optional):
```bash
python analysis3054/data/kayross/update_data.py --update-dtn
```
DTN auth (any one):
- `DTN_API_KEY` (preferred)
- `DTN_ACCESS_TOKEN`
- or OAuth: `DTN_CLIENT_ID` + `DTN_CLIENT_SECRET` (+ optional `DTN_AUDIENCE`)

Pandas inputs:
- `build_dashboard.py` accepts DataFrames via `build_payload(...)`. If you want in-memory DataFrames, copy that script and call `build_payload`/`build_html` directly.

Notes:
- DTN diesel grades: `#1 Diesel`, `#2 Diesel` (plus common variants).
- DTN gasoline grades: `Premium`, `Regular`.
- PADD volumes apply coverage factor then divide by 42000 (KBD). Rack/state data skips coverage factor and still divides by 42000.

### 2) CFTC Disaggregated COT Dashboard

Folder: `analysis3054/data/cftc_disagg/`

Build (downloads data if cache missing):
```bash
python analysis3054/data/cftc_disagg/build_dashboard.py
```

Outputs:
- `analysis3054/data/cftc_disagg/cftc_disagg_dashboard.html`
- `analysis3054/data/cftc_disagg/cftc_disagg_dashboard_export.html`
- `analysis3054/data/cftc_disagg/cftc_disagg_dashboard_data.json`
- `analysis3054/data/cftc_disagg/cftc_disagg_filtered.csv`

### 3) Port TEU Dashboard

Folder: `analysis3054/data/port_teu/`

Inputs:
- `port_of_la_container_stats_2021_present.csv`
- `port_long_beach_teu.csv`
- `port_ny_teu.csv`
- `port_houston_teu.csv`
- `port_savannah_teu.csv`

Build:
```bash
python analysis3054/data/port_teu/build_dashboard.py --data-dir analysis3054/data/port_teu
```

Update (Gemini disabled; Houston/Savannah are manual):
```bash
python analysis3054/data/port_teu/update_data.py \
  --houston-csv /path/to/port_houston.csv \
  --savannah-csv /path/to/port_savannah.csv
```

### 4) Rail Traffic Dashboard

Folder: `analysis3054/data/rail_traffic/`

Inputs (manual):
- `rail_traffic_north_american.csv`
- `rail_traffic_us.csv`
- `rail_traffic_canada.csv`
- `rail_traffic_mexico.csv`

Build:
```bash
python analysis3054/data/rail_traffic/build_dashboard.py --data-dir analysis3054/data/rail_traffic
```

Update:
```bash
python analysis3054/data/rail_traffic/update_data.py \
  --north-american-csv /path/to/north_american.csv \
  --us-csv /path/to/us.csv \
  --canada-csv /path/to/canada.csv \
  --mexico-csv /path/to/mexico.csv
```

### 5) Macro Drivers Dashboard

Folder: `analysis3054/data/macro/`

Inputs:
- FRED snapshots: `fred_diesel_monthly.csv`, `fred_diesel_quarterly.csv`
- Macro focus: `fred_macro_focus_*`
- EIA snapshots: `eia_distillate_product_supplied.csv`, `eia_steo_monthly.csv`, `eia_steo_quarterly.csv`
- Forecast snapshots: `distillate_forecast*.csv`

Update (FRED + optional EIA refresh):
```bash
FRED_API_KEY=... python analysis3054/data/macro/update_data.py
```
Skip EIA refresh:
```bash
SKIP_EIA_FETCH=true python analysis3054/data/macro/update_data.py
```

Build:
```bash
python analysis3054/data/macro/build_dashboard.py
```

Forecast model override:
```bash
DISTILLATE_FORECAST_MODEL=chronos2 python analysis3054/data/macro/build_dashboard.py
```

### 6) Europe Energy Dashboard

Folder: `analysis3054/data/europe_energy/`

Update (Spain + UK sources):
```bash
python analysis3054/data/europe_energy/update_data.py
```

Build:
```bash
python analysis3054/data/europe_energy/build_dashboard.py
```

### 7) DFO Generator Dashboard

Folder: `analysis3054/data/dfo_generators/`

Inputs: EIA-860/923 raw tabs (place into this folder). The update helper rebuilds:
- `dfo_generators_inventory.csv`
- `dfo_generators_costs.csv`

Update + Build:
```bash
python analysis3054/data/dfo_generators/update_data.py
```

### 8) EIA Electricity Runner (EIA-860/923)

Folder: `analysis3054/data/eia_electricity/`

This runner refreshes the raw EIA-860/923 CSVs and can rebuild the DFO dashboard.

Run:
```bash
python analysis3054/data/eia_electricity/update_data.py
```

Options:
```bash
python analysis3054/data/eia_electricity/update_data.py --output-dir analysis3054/data/dfo_generators
python analysis3054/data/eia_electricity/update_data.py --year-860 2024 --year-923 2024
python analysis3054/data/eia_electricity/update_data.py --skip-dfo
```

### 9) Refinery Product Mix Dashboard

Folder: `analysis3054/data/refinery/`

Inputs:
- `la_refinery_latest.csv` (optional updater)
- `tx_refineries_2021_present.csv`
- `refinery_data.csv`
- `refinery_forecast.csv`
- `refinery_product_forecast.csv`
- `tx_refinery_name_map.csv`

Update:
```bash
python analysis3054/data/refinery/update_data.py --refresh-la
```

Build:
```bash
python analysis3054/data/refinery/build_dashboard.py
```

### 10) TX Refinery Receipts & Deliveries

Folder: `analysis3054/data/tx_refineries_transfers/`

Inputs:
- `tx_refineries_received_2021_present.csv`
- `tx_refineries_delivered_2021_present.csv`
- `tx_refineries_2021_present.csv`
- `tx_refinery_name_map.csv`

Build:
```bash
python analysis3054/data/tx_refineries_transfers/build_dashboard.py
```

Update:
```bash
python analysis3054/data/tx_refineries_transfers/update_data.py \
  --received-csv /path/to/tx_received.csv \
  --delivered-csv /path/to/tx_delivered.csv
```

## Notes

- Each dashboard folder is designed to be copied as-is.
- Update scripts avoid Gemini-based parsing; provide manual CSVs where noted.
- If you want offline HTML, use the `*_export.html` outputs where available.
