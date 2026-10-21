"""Multi-domain sensor telemetry for Heimdall (beyond weather)."""

from src.telemetry.domains import DOMAIN_CHANNELS, domain_labels, project_domains
from src.telemetry.simulate import simulate_project_telemetry
from src.telemetry.store import (
    load_telemetry,
    load_telemetry_freshness,
    replace_telemetry_domain,
)

__all__ = [
    "DOMAIN_CHANNELS",
    "domain_labels",
    "project_domains",
    "simulate_project_telemetry",
    "load_telemetry",
    "load_telemetry_freshness",
    "replace_telemetry_domain",
]
