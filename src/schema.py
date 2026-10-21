from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CHANNEL_NAMES = (
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "rainfall_mm",
    "wind_speed_ms",
    "wind_direction_deg",
)

QualityFlag = Literal["good", "suspect", "missing"]


class Readings(BaseModel):
    """Physical measurements in the units locked with the hardware team."""

    temperature_c: float | None = Field(None, description="Air temperature in °C")
    humidity_pct: float | None = Field(None, ge=0, le=110, description="Relative humidity in %")
    pressure_hpa: float | None = Field(None, description="Barometric pressure in hPa")
    rainfall_mm: float | None = Field(None, ge=0, description="Rain accumulated in this sample interval, mm")
    wind_speed_ms: float | None = Field(None, ge=0, description="Wind speed in m/s")
    wind_direction_deg: float | None = Field(None, ge=0, lt=360, description="Wind direction, degrees from north")


class Meta(BaseModel):
    firmware: str | None = None
    battery_v: float | None = Field(None, ge=0)


class SensorPayload(BaseModel):
    """Shared ingest contract for the microcontroller and the simulator."""

    station_id: str = Field(..., min_length=1, examples=["WS-INDRA", "WS-ISSA"])
    timestamp: datetime = Field(..., description="UTC instant of the sample")
    readings: Readings
    meta: Meta = Field(default_factory=Meta)
    source: Literal["sensor", "simulator"] = "sensor"

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def to_row(self) -> dict:
        row = {
            "station_id": self.station_id,
            "timestamp": self.timestamp.isoformat(),
            "firmware": self.meta.firmware,
            "battery_v": self.meta.battery_v,
            "source": self.source,
        }
        row.update(self.readings.model_dump())
        return row


class CalibrationCoefficients(BaseModel):
    station_id: str
    channel: str
    scale: float
    offset: float
    fitted_at: datetime
    n_points: int
    mae: float
    rmse: float
    bias: float
    mae_raw: float
    rmse_raw: float
    bias_raw: float
