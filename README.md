# Heimdall

Sensor data hub for **INDRA** (hydro dam) and **ISSA** (plantation).

Heimdall started as a weather station analysis stack and now expands into a multi-domain telemetry hub. Weather prediction remains, but each project also carries domain sensors — simulated today, replaceable with live hardware later.

## Projects

| Project | Focus | Domains |
|---|---|---|
| **INDRA** | Hydro dam site | Weather, water level, water velocity, seepage, CCTV status |
| **ISSA** | Plantation | Weather, soil NPK (moisture, EC, pH, N, P, K) |

NPK uses chemistry-correct labels: **N** = nitrogen, **P** = phosphorus, **K** = potassium.

CCTV is status telemetry only (online / motion / frame age), not a live video stream.

## Hub pages

| Page | Purpose |
|---|---|
| **Overview** | What is happening now |
| **Sensors** | All domain readings (weather, hydro, soil, …) |
| **Forecast** | Next 7 days |
| **Map** | Project pins |
| **Progress** | How good past forecasts were |
| **Settings** | Display, models, thresholds |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Seed simulated weather + domain sensors for INDRA and ISSA
python scripts/run_heimdall_simulate.py

# Optional: restore older prediction history from data/weather.db
python scripts/migrate_weather_to_heimdall.py

# Hub UI
streamlit run app/dashboard.py

# Full weather analysis refresh (history, ensemble, verify)
python scripts/run_refresh_hub.py
```

## Layout

- `app/dashboard.py` — Heimdall Streamlit hub
- `src/telemetry/` — multi-domain sensors + simulators
- `src/hub_refresh.py` — weather analysis pipeline
- `config.yaml` — projects and settings
- `data/heimdall.db` — active database
- `data/weather.db` — optional backup of the older weather-only DB

## Notes

- Until hardware is installed, `run_heimdall_simulate.py` writes realistic synthetic series.
- Weather calibration / ensemble / Progress skill tracking still run through the existing analysis scripts.
- Default pins are in West Jaintia Hills, Meghalaya (edit in `config.yaml` or on the Map page).
