# Hardware ingest contract

This is the JSON payload both the microcontroller firmware and the analysis
pipeline must use. Units, UTC timestamps, and field names are locked so
calibration can align samples without unit conversion.

## Endpoint

```
POST /readings
Content-Type: application/json
```

## Schema

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `station_id` | string | yes | e.g. `WS-INDRA`, `WS-ISSA` |
| `timestamp` | ISO-8601 datetime | yes | **UTC** (`Z` suffix). Naive timestamps are treated as UTC. |
| `readings.temperature_c` | number or null | no | Air temperature, **°C** |
| `readings.humidity_pct` | number or null | no | Relative humidity, **%** (0–100 typical) |
| `readings.pressure_hpa` | number or null | no | Barometric pressure, **hPa** |
| `readings.rainfall_mm` | number or null | no | Rain **in this sample interval**, **mm** (not a running daily total) |
| `readings.wind_speed_ms` | number or null | no | Wind speed, **m/s** |
| `readings.wind_direction_deg` | number or null | no | Meteorological direction, **degrees from north**, `[0, 360)` |
| `meta.firmware` | string | no | Firmware version |
| `meta.battery_v` | number | no | Battery voltage |
| `source` | `"sensor"` \| `"simulator"` | no | Defaults to `"sensor"` |

Sampling interval (default): **every 5 minutes**. Keep it constant.

## Example

```json
{
  "station_id": "WS-INDRA",
  "timestamp": "2026-08-17T07:30:00Z",
  "readings": {
    "temperature_c": 28.4,
    "humidity_pct": 72.1,
    "pressure_hpa": 1008.2,
    "rainfall_mm": 0.0,
    "wind_speed_ms": 2.1,
    "wind_direction_deg": 180
  },
  "meta": {
    "firmware": "1.0.0",
    "battery_v": 3.7
  }
}
```

## curl

```bash
curl -X POST http://127.0.0.1:8000/readings \
  -H "Content-Type: application/json" \
  -d "{\"station_id\":\"WS-INDRA\",\"timestamp\":\"2026-08-17T07:30:00Z\",\"readings\":{\"temperature_c\":28.4,\"humidity_pct\":72.1,\"pressure_hpa\":1008.2,\"rainfall_mm\":0.0,\"wind_speed_ms\":2.1,\"wind_direction_deg\":180},\"meta\":{\"firmware\":\"1.0.0\",\"battery_v\":3.7}}"
```

## Clock

Send UTC. Flag samples whose clock is more than ~2 minutes off NTP once
hardware is live; the analysis side stores timestamps as received.
