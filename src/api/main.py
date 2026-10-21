from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.config import load_config
from src.ingestion.store import init_db, insert_payload, load_readings
from src.schema import SensorPayload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Weather Station Ingest",
    description="POST sensor readings from the microcontroller (or simulator).",
    version="0.1.0",
    lifespan=lifespan,
)


class IngestResponse(BaseModel):
    id: int
    station_id: str
    timestamp: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/readings", response_model=IngestResponse)
def post_reading(payload: SensorPayload) -> IngestResponse:
    row_id = insert_payload(payload)
    return IngestResponse(
        id=row_id,
        station_id=payload.station_id,
        timestamp=payload.timestamp.isoformat(),
    )


@app.get("/readings")
def get_readings(
    station_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=5000),
) -> list[dict]:
    cfg = load_config()
    station = station_id or cfg["station"]["id"]
    df = load_readings(station)
    if df.empty:
        return []
    return json.loads(df.tail(limit).to_json(orient="records", date_format="iso"))


@app.get("/contract")
def contract() -> dict:
    """Point hardware team at the locked JSON contract."""
    return {
        "docs": "docs/HARDWARE_CONTRACT.md",
        "example": SensorPayload.model_json_schema(),
    }
