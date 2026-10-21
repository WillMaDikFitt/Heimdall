from __future__ import annotations

import html
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app import charts, maps, theme
from app.forecast_board import build_forecast_payload, render_forecast_board_html
from app.theme import normalize_theme_id
from src.calibration.qc import iqr_inlier_mask, window_metrics
from src.config import load_config, save_hub_settings, save_station_location
from src.hub import product_name, product_tagline, projects as hub_projects, resolve_project_id
from src.hub_refresh import load_refresh_status, run_hub_refresh
from src.telemetry.domains import DOMAIN_CHANNELS, domain_labels, project_domains
from src.telemetry.store import load_telemetry, load_telemetry_latest
from src.ingestion.store import (
    load_calibrated,
    load_ensemble_forecast,
    load_forecast_predictions,
    load_forecast_scores,
    load_forecast_verification_scores,
    load_imd_daily_rain,
    load_latest_coefficients,
    load_nwp_members,
    load_power_readings,
    load_readings,
    load_reference,
    load_source_validation_scores,
    load_walkforward_scores,
)
from src.forecast.ensemble import ensemble_as_operational, ensemble_to_daily
from src.reference.forecast import ForecastBundle, fetch_forecast, weather_label
from src.schema import CHANNEL_NAMES

RANGE_HOURS = {
    "24 hours": 24,
    "3 days": 72,
    "7 days": 168,
    "30 days": 720,
    "All data": None,
}

SOURCE_DETAILS = {
    "ecmwf": (
        "ECMWF",
        "Helps predict the next few days",
        "Future weather",
    ),
    "gfs": (
        "NOAA GFS",
        "Helps predict the next few days",
        "Future weather",
    ),
    "met_norway": (
        "MET Norway",
        "Helps predict the next few days",
        "Future weather",
    ),
    "Open-Meteo": (
        "Open-Meteo",
        "Past weather used for checking",
        "Past weather",
    ),
    "NASA POWER": (
        "NASA POWER",
        "Another record of past weather",
        "Past weather",
    ),
    "IMD grid": (
        "India Meteorological Department",
        "Official rainfall record for India",
        "Past rain",
    ),
    "Calibrated station": (
        "Local weather station",
        "Readings taken at the project pin",
        "On site",
    ),
}

CHANNEL_LABELS = {
    "temperature_c": "Temperature",
    "humidity_pct": "Humidity",
    "pressure_hpa": "Air pressure",
    "rainfall_mm": "Rainfall",
    "wind_speed_ms": "Wind speed",
    "wind_direction_deg": "Wind direction",
}

st.set_page_config(
    page_title="Heimdall",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _projects(cfg: dict) -> dict[str, dict]:
    return hub_projects(cfg)


@st.cache_data(ttl=30)
def _load(project_id: str = "indra") -> dict[str, object]:
    load_config.cache_clear()
    cfg = load_config()
    pid = resolve_project_id(cfg, project_id)
    project = _projects(cfg)[pid]
    runtime_cfg = {**cfg, "station": project["station"]}
    forecast_bundle: ForecastBundle | None = None
    forecast_error = ""
    try:
        forecast_bundle = fetch_forecast(
            cfg=runtime_cfg,
            forecast_days=int(cfg["forecast"].get("operational_days", 7)),
            cache_minutes=int(cfg["forecast"].get("cache_minutes", 30)),
        )
    except Exception as exc:  # keep diagnostics available during network outages
        forecast_error = str(exc)

    station_id = str(project["station"]["id"])
    dam_only = str(project["kind"]) == "dam"
    empty = pd.DataFrame()
    ensemble = load_ensemble_forecast(cfg=cfg, station_id=station_id)
    nwp_members = load_nwp_members(cfg=cfg, station_id=station_id)
    forecast_verification = load_forecast_verification_scores(cfg=cfg, station_id=station_id)
    scores = load_forecast_scores(cfg=cfg) if dam_only else empty
    walkforward = load_walkforward_scores(cfg=cfg) if dam_only else empty
    predictions = load_forecast_predictions(cfg=cfg) if dam_only else empty
    return {
        "cfg": runtime_cfg,
        "project": project,
        "readings": load_readings(station_id, cfg=cfg),
        "reference": load_reference(cfg=cfg, station_id=station_id),
        "calibrated": load_calibrated(cfg=cfg, station_id=station_id),
        "coeffs": load_latest_coefficients(cfg=cfg, station_id=station_id),
        "scores": scores,
        "walkforward": walkforward,
        "predictions": predictions,
        "operational_forecast": forecast_bundle,
        "ensemble": ensemble,
        "nwp_members": nwp_members,
        "power": load_power_readings(cfg=cfg) if dam_only else empty,
        "imd": load_imd_daily_rain(cfg=cfg) if dam_only else empty,
        "source_validation": load_source_validation_scores(cfg=cfg) if dam_only else empty,
        "forecast_verification": forecast_verification,
        "forecast_error": forecast_error,
    }


def _label(cfg: dict, channel: str) -> str:
    meta = cfg["channels"][channel]
    return f"{meta.get('label', channel)} ({meta['unit']})"


def _unit(cfg: dict, channel: str) -> str:
    return cfg["channels"][channel]["unit"]


def _filter_hours(frame: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    if frame.empty or hours is None or "timestamp" not in frame.columns:
        return frame
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[ts >= ts.max() - pd.Timedelta(hours=hours)].copy()


def _despike(frame: pd.DataFrame, enabled: bool, multiplier: float = 3.0) -> pd.DataFrame:
    """Blank IQR outliers per channel so injected spikes stop distorting the y-axis."""
    if not enabled or frame.empty:
        return frame
    out = frame.copy()
    for channel in CHANNEL_NAMES:
        if channel not in out.columns:
            continue
        values = pd.to_numeric(out[channel], errors="coerce").to_numpy(dtype=float)
        out.loc[~iqr_inlier_mask(values, multiplier=float(multiplier)), channel] = np.nan
    return out


def _recent(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[ts >= ts.max() - pd.Timedelta(hours=hours)]


def _compass(degrees: float | None) -> str:
    if degrees is None or pd.isna(degrees):
        return ""
    return charts.SECTORS[int((float(degrees) + 11.25) // 22.5) % 16]


def _num(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _source_card(
    name: str,
    description: str,
    role: str,
    status: str,
    active: bool = True,
) -> str:
    state_class = "source-ok" if active else "source-idle"
    return (
        f'<div class="source-card">'
        f'<div class="source-card-top"><span class="source-role">{html.escape(role)}</span>'
        f'<span class="source-state {state_class}">{html.escape(status)}</span></div>'
        f'<div class="source-name">{html.escape(name)}</div>'
        f'<div class="source-copy">{html.escape(description)}</div>'
        "</div>"
    )


def _data_sources_panel(
    nwp_members: pd.DataFrame,
    reference: pd.DataFrame,
    power: pd.DataFrame,
    imd: pd.DataFrame,
) -> None:
    """Show non-technical users where forecasts and checks come from."""
    cards: list[str] = []
    available_members = (
        set(nwp_members["member"].dropna().astype(str)) if not nwp_members.empty else set()
    )
    for member in ("ecmwf", "gfs", "met_norway"):
        name, description, role = SOURCE_DETAILS[member]
        active = member in available_members
        cards.append(
            _source_card(
                name,
                description,
                role,
                "Available" if active else "Not available",
                active,
            )
        )

    supporting = [
        ("Open-Meteo", not reference.empty, f"{len(reference):,} hours"),
        ("NASA POWER", not power.empty, f"{len(power):,} hours"),
        ("IMD grid", not imd.empty, f"{len(imd):,} days"),
    ]
    for key, active, status in supporting:
        name, description, role = SOURCE_DETAILS[key]
        cards.append(
            _source_card(name, description, role, status if active else "No data yet", active)
        )

    _page_intro("Sources", "Where the weather data comes from.")
    st.markdown(f'<div class="source-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption("Green means that information is available now.")


def _comparison_meaning(mae: float, channel: str) -> tuple[str, str]:
    """Give a cautious plain-language interpretation of inter-source disagreement."""
    limits = {
        "temperature_c": (2.0, 4.0),
        "humidity_pct": (10.0, 20.0),
        "pressure_hpa": (5.0, 15.0),
        "wind_speed_ms": (1.0, 2.5),
        "rainfall_mm": (2.0, 8.0),
    }
    good, high = limits.get(channel, (1.0, 3.0))
    if mae <= good:
        return "Close agreement", "agree-good"
    if mae <= high:
        return "Some difference", "agree-mid"
    return "Large difference", "agree-high"


def _friendly_forecast_source(source: str) -> str:
    if source.startswith("ensemble:"):
        members = source.removeprefix("ensemble:").split("+")
        names = [SOURCE_DETAILS.get(member, (member, "", ""))[0] for member in members]
        return "Combined forecast from " + ", ".join(names)
    return source.replace("·", "—")


def _range_foot(frame: pd.DataFrame, channel: str, digits: int = 1) -> str:
    if frame.empty or channel not in frame.columns:
        return ""
    values = pd.to_numeric(frame[channel], errors="coerce").dropna()
    if values.empty:
        return ""
    return f"24h range {values.min():.{digits}f} – {values.max():.{digits}f}"


def _page_intro(title: str, body: str = "") -> None:
    copy = (
        f'<div class="page-intro-copy">{html.escape(body)}</div>' if body.strip() else ""
    )
    st.markdown(
        f'<div class="page-intro">'
        f'<div class="page-intro-title">{html.escape(title)}</div>'
        f"{copy}"
        "</div>",
        unsafe_allow_html=True,
    )


def _ui_prefs(cfg: dict) -> dict:
    ui = dict((cfg.get("hub") or {}).get("ui") or {})
    window = str(ui.get("data_window", "7 days"))
    if window not in RANGE_HOURS:
        window = "7 days"
    auto_refresh = int(ui.get("auto_refresh_minutes", 0) or 0)
    if auto_refresh not in (0, 5, 15, 30):
        auto_refresh = 0
    return {
        "data_window": window,
        "show_calibrated": bool(ui.get("show_calibrated", True)),
        "show_reference": bool(ui.get("show_reference", True)),
        "show_raw": bool(ui.get("show_raw", False)),
        "despike": bool(ui.get("despike", True)),
        "color_theme": normalize_theme_id(ui.get("color_theme", "vision")),
        "auto_refresh_minutes": auto_refresh,
    }


def _skill_prefs(cfg: dict) -> dict[str, float]:
    ui = dict((cfg.get("hub") or {}).get("ui") or {})
    skill = dict(ui.get("skill") or {})
    return {
        "day_mae_good": float(skill.get("day_mae_good", 3.0)),
        "day_mae_fair": float(skill.get("day_mae_fair", 6.0)),
        "csi_target": float(skill.get("csi_target", 0.5)),
        "confidence_mae_good": float(skill.get("confidence_mae_good", 1.5)),
        "confidence_mae_fair": float(skill.get("confidence_mae_fair", 2.5)),
        "spread_steady": float(skill.get("spread_steady", 1.0)),
        "spread_mixed": float(skill.get("spread_mixed", 2.0)),
    }


def _format_analysis_stamp(status: dict | None = None) -> str:
    status = status if status is not None else load_refresh_status()
    finished = status.get("finished_at")
    if not finished:
        return "No analysis run yet"
    try:
        when = pd.Timestamp(finished)
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        local = when.tz_convert("Asia/Kolkata").strftime("%d %b %Y · %H:%M IST")
    except Exception:
        local = str(finished)
    if status.get("ok"):
        return f"Last analysis OK · {local}"
    return f"Last analysis failed · {local}"


def _active_theme_id(cfg: dict) -> str:
    saved = _ui_prefs(cfg)["color_theme"]
    preview = st.session_state.get("theme_preview")
    if preview:
        return normalize_theme_id(str(preview))
    return saved


def _render_settings_page(cfg: dict, projects: dict[str, dict]) -> None:
    """Hub display and forecast settings saved into config.yaml."""
    _page_intro("Settings", "Hub display and outlook options.")
    prefs = _ui_prefs(cfg)
    skill = _skill_prefs(cfg)
    forecast = cfg.get("forecast") or {}
    open_meteo = cfg.get("open_meteo") or {}
    calibration = cfg.get("calibration") or {}
    channels = cfg.get("channels") or {}
    hub = cfg.get("hub") or {}
    member_labels = {
        "ecmwf": "ECMWF",
        "gfs": "NOAA GFS",
        "met_norway": "MET Norway",
    }
    known_members = list(member_labels.keys())
    current_members = [
        m for m in list(forecast.get("ensemble_members") or known_members) if m in known_members
    ]
    if not current_members:
        current_members = known_members[:]
    drift_channels = [
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "rainfall_mm",
        "wind_speed_ms",
    ]

    if "theme_preview" not in st.session_state:
        st.session_state.theme_preview = prefs["color_theme"]
    color_theme = normalize_theme_id(str(st.session_state.theme_preview))

    st.markdown(theme.section("Colour theme", "Click a look to preview. Save to keep it."), unsafe_allow_html=True)
    st.markdown(
        f"""
        <style>
        .st-key-theme_pick_{color_theme} button {{
          box-shadow: 0 0 0 2px #ffffff, 0 10px 24px rgba(0,0,0,.28) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="theme_swatches"):
        cols = st.columns(len(theme.THEME_CHOICES), gap="small")
        for col, key in zip(cols, theme.THEME_CHOICES):
            with col:
                meta = theme.THEMES[key]
                if st.button(
                    meta["label"],
                    key=f"theme_pick_{key}",
                    type="secondary",
                    width="stretch",
                    help=f"{meta['label']}: {meta['blurb']}",
                ):
                    st.session_state.theme_preview = key
                    st.rerun()
    st.caption(f"{theme.THEMES[color_theme]['label']} — {theme.THEMES[color_theme]['blurb']}")
    if color_theme != prefs["color_theme"]:
        st.caption(
            f"Previewing {theme.THEMES[color_theme]['label']}. "
            f"Saved theme is still {theme.THEMES[prefs['color_theme']]['label']} until you press Save."
        )

    st.markdown(theme.section("Display", "Charts and tables on Data."), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        data_window = st.selectbox(
            "History window",
            list(RANGE_HOURS.keys()),
            index=list(RANGE_HOURS.keys()).index(prefs["data_window"]),
            key="set_data_window",
        )
        show_cal = st.checkbox(
            "Show checked station readings",
            value=prefs["show_calibrated"],
            key="set_show_cal",
        )
        show_ref = st.checkbox(
            "Show wider-area weather",
            value=prefs["show_reference"],
            key="set_show_ref",
        )
    with c2:
        show_raw = st.checkbox(
            "Show original sensor readings",
            value=prefs["show_raw"],
            key="set_show_raw",
        )
        despike = st.checkbox(
            "Hide sudden wrong readings",
            value=prefs["despike"],
            key="set_despike",
        )
        default_project = st.selectbox(
            "Default project on open",
            list(projects.keys()),
            index=list(projects.keys()).index(str(hub.get("default", "indra")))
            if str(hub.get("default", "indra")) in projects
            else 0,
            format_func=lambda key: projects[key]["label"],
            key="set_default_project",
        )
        auto_refresh_minutes = st.selectbox(
            "Auto-refresh charts",
            options=[0, 5, 15, 30],
            index=[0, 5, 15, 30].index(prefs["auto_refresh_minutes"]),
            format_func=lambda m: "Off" if m == 0 else f"Every {m} minutes",
            key="set_auto_refresh",
            help="Reloads cached charts only. Full NWP analysis still needs Run analysis.",
        )

    st.markdown(
        theme.section("Outlook", "Models and timing for the live rain outlook."),
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        operational_days = st.number_input(
            "Days ahead",
            min_value=3,
            max_value=10,
            value=int(forecast.get("operational_days", 7)),
            step=1,
            key="set_operational_days",
        )
    with f2:
        cache_minutes = st.number_input(
            "Cache minutes",
            min_value=5,
            max_value=180,
            value=int(forecast.get("cache_minutes", 30)),
            step=5,
            key="set_cache_minutes",
        )
    with f3:
        bias_window_days = st.number_input(
            "Bias window (days)",
            min_value=3,
            max_value=30,
            value=int(forecast.get("bias_window_days", 14)),
            step=1,
            key="set_bias_window",
        )
    ensemble_members = st.multiselect(
        "Forecast models",
        options=known_members,
        default=current_members,
        format_func=lambda key: member_labels.get(key, key),
        key="set_ensemble_members",
    )

    st.markdown(
        theme.section("Calibration", "How sensor readings are cleaned before fitting."),
        unsafe_allow_html=True,
    )
    cal1, cal2, cal3 = st.columns(3, gap="medium")
    with cal1:
        resample_minutes = st.number_input(
            "Resample minutes",
            min_value=5,
            max_value=180,
            value=int(calibration.get("resample_minutes", 60)),
            step=5,
            key="set_resample_minutes",
        )
    with cal2:
        iqr_multiplier = st.number_input(
            "IQR spike filter",
            min_value=1.0,
            max_value=5.0,
            value=float(calibration.get("iqr_multiplier", 1.5)),
            step=0.1,
            key="set_iqr_multiplier",
        )
    with cal3:
        holdout_fraction = st.number_input(
            "Holdout fraction",
            min_value=0.1,
            max_value=0.5,
            value=float(calibration.get("holdout_fraction", 0.3)),
            step=0.05,
            key="set_holdout_fraction",
        )

    st.markdown(
        theme.section("Drift alerts", "Warn when a sensor bias exceeds these limits."),
        unsafe_allow_html=True,
    )
    drift_cols = st.columns(len(drift_channels), gap="small")
    drift_thresholds: dict[str, float] = {}
    for col, channel in zip(drift_cols, drift_channels):
        meta = channels.get(channel) or {}
        with col:
            drift_thresholds[channel] = float(
                st.number_input(
                    str(meta.get("label", channel)),
                    min_value=0.1,
                    max_value=100.0,
                    value=float(meta.get("drift_alert_threshold", 2.0)),
                    step=0.1,
                    key=f"set_drift_{channel}",
                    help=f"Unit: {meta.get('unit', '')}",
                )
            )

    st.markdown(
        theme.section("Skill / confidence", "Bands used on Progress and Overview confidence."),
        unsafe_allow_html=True,
    )
    v1, v2, v3 = st.columns(3, gap="medium")
    with v1:
        verify_min_age = st.number_input(
            "Wait before scoring (hours)",
            min_value=12,
            max_value=168,
            value=int(forecast.get("verify_min_age_hours", 48)),
            step=6,
            key="set_verify_age",
            help="An outlook needs this much age before Progress can score it.",
        )
        day_mae_good = st.number_input(
            "Day MAE good (mm)",
            min_value=0.5,
            max_value=20.0,
            value=skill["day_mae_good"],
            step=0.5,
            key="set_day_mae_good",
        )
        confidence_mae_good = st.number_input(
            "Confidence MAE good (°C)",
            min_value=0.5,
            max_value=10.0,
            value=skill["confidence_mae_good"],
            step=0.1,
            key="set_conf_mae_good",
        )
    with v2:
        historical_days = st.number_input(
            "History days to keep",
            min_value=30,
            max_value=365,
            value=int(open_meteo.get("historical_days", 90)),
            step=10,
            key="set_historical_days",
        )
        day_mae_fair = st.number_input(
            "Day MAE fair (mm)",
            min_value=1.0,
            max_value=30.0,
            value=skill["day_mae_fair"],
            step=0.5,
            key="set_day_mae_fair",
        )
        confidence_mae_fair = st.number_input(
            "Confidence MAE fair (°C)",
            min_value=0.5,
            max_value=15.0,
            value=skill["confidence_mae_fair"],
            step=0.1,
            key="set_conf_mae_fair",
        )
    with v3:
        csi_target = st.number_input(
            "CSI timing target",
            min_value=0.1,
            max_value=1.0,
            value=skill["csi_target"],
            step=0.05,
            key="set_csi_target",
        )
        spread_steady = st.number_input(
            "Spread steady (°C)",
            min_value=0.2,
            max_value=5.0,
            value=skill["spread_steady"],
            step=0.1,
            key="set_spread_steady",
        )
        spread_mixed = st.number_input(
            "Spread mixed (°C)",
            min_value=0.5,
            max_value=10.0,
            value=skill["spread_mixed"],
            step=0.1,
            key="set_spread_mixed",
        )

    left, right = st.columns((1, 3), gap="medium")
    with left:
        save = st.button("Save settings", type="primary", width="stretch", key="save_hub_settings")
    with right:
        st.caption(
            "Saved to config.yaml. After model or day changes, press Run analysis "
            "(not only Reload UI)."
        )

    st.html(
        theme.paint_streamlit_controls(theme.theme_tokens(color_theme)["accent"]),
        unsafe_allow_javascript=True,
    )

    if save:
        try:
            save_hub_settings(
                {
                    "default": default_project,
                    "data_window": data_window,
                    "show_calibrated": show_cal,
                    "show_reference": show_ref,
                    "show_raw": show_raw,
                    "despike": despike,
                    "color_theme": color_theme,
                    "auto_refresh_minutes": int(auto_refresh_minutes),
                    "operational_days": int(operational_days),
                    "cache_minutes": int(cache_minutes),
                    "bias_window_days": int(bias_window_days),
                    "verify_min_age_hours": int(verify_min_age),
                    "ensemble_members": list(ensemble_members),
                    "historical_days": int(historical_days),
                    "resample_minutes": int(resample_minutes),
                    "iqr_multiplier": float(iqr_multiplier),
                    "holdout_fraction": float(holdout_fraction),
                    "drift_thresholds": drift_thresholds,
                    "skill": {
                        "day_mae_good": float(day_mae_good),
                        "day_mae_fair": float(day_mae_fair),
                        "csi_target": float(csi_target),
                        "confidence_mae_good": float(confidence_mae_good),
                        "confidence_mae_fair": float(confidence_mae_fair),
                        "spread_steady": float(spread_steady),
                        "spread_mixed": float(spread_mixed),
                    },
                }
            )
            st.session_state.theme_preview = color_theme
            load_config.cache_clear()
            _load.clear()
            st.success("Settings saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save settings: {exc}")


def _section_nav(
    options: list[str],
    *,
    key: str = "hub_section",
    default: str = "Overview",
) -> str:
    """Custom underline tab bar — replaces Streamlit segmented_control."""
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]

    current = str(st.session_state[key])
    with st.container(key="hub_nav"):
        cols = st.columns(len(options), gap="small")
        for col, name in zip(cols, options):
            with col:
                active = name == current
                if st.button(
                    name,
                    key=f"{key}_btn_{name}",
                    type="primary" if active else "secondary",
                    width="stretch",
                ):
                    if name != current:
                        st.session_state[key] = name
                        st.rerun()
    return str(st.session_state[key])


def _friendly_model_name(name: str) -> str:
    mapping = {
        "persistence": "Same as last hour",
        "persistence_fullhist": "Same as last hour (full history)",
        "seasonal_naive": "Same hour yesterday",
        "seasonal_naive_fullhist": "Same hour yesterday (full history)",
        "gradient_boosting": "Local learning model",
        "gradient_boosting_fullhist": "Local learning model (full history)",
        "bias_corrected_ref": "Bias-corrected area weather",
        "bias_corrected_ref_fullhist": "Bias-corrected area weather (full history)",
        "hybrid_rain": "Hybrid rain (timing + daily total)",
        "hybrid_rain_fullhist": "Hybrid rain (timing + daily total, full history)",
        "direct_rain_24h": "Direct 24-hour rain total",
        "direct_rain_24h_fullhist": "Direct 24-hour rain total (full history)",
        "direct_rain_6h": "Direct 6-hour rain total",
        "direct_rain_6h_fullhist": "Direct 6-hour rain total (full history)",
    }
    return mapping.get(name, name.replace("_", " "))


def _friendly_train_mode(mode: str) -> str:
    text = str(mode).replace("_", " ")
    if text.startswith("last "):
        return text.replace("d", " days").replace("  ", " ")
    if text == "full history":
        return "Using all available history"
    if text == "walkforward":
        return "Repeated day-by-day tests"
    return text


def _alerts(cfg: dict, coeffs: pd.DataFrame) -> list[str]:
    out: list[str] = []
    if coeffs.empty:
        return out
    for _, row in coeffs.iterrows():
        channel = row["channel"]
        meta = cfg["channels"].get(channel)
        if meta is None or pd.isna(row.get("bias")):
            continue
        threshold = float(meta["drift_alert_threshold"])
        bias = float(row["bias"])
        if abs(bias) > threshold:
            label = meta.get("label", channel)
            unit = meta["unit"]
            direction = "higher" if bias > 0 else "lower"
            out.append(
                f"{label} is reading about {abs(bias):.1f} {unit} {direction} than expected "
                f"(alert limit is ±{threshold:g} {unit})."
            )
    return out


def _masthead(station: dict, status: tuple[str, str], stamp: str, page: str) -> None:
    text, css_class = status
    st.markdown(
        f"""
        <div class="topbar">
          <div>
            <h1>{html.escape(page)}</h1>
            <p class="sub">{html.escape(str(station["name"]))} · {html.escape(str(station.get("site", "")))}</p>
          </div>
          <div>
            <div class="top-actions">
              <span class="pill {css_class}"><span class="dot"></span>{html.escape(text)}</span>
            </div>
            <div class="stamp">{html.escape(stamp)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _conditions(cfg: dict, readings: pd.DataFrame) -> None:
    last_24h = _recent(readings, 24)
    latest = readings.iloc[-1]

    def spark(channel: str) -> str:
        if channel not in last_24h.columns:
            return ""
        values = pd.to_numeric(last_24h[channel], errors="coerce").to_numpy(dtype=float)
        return theme.sparkline(values, theme.CHANNEL_ACCENT.get(channel, theme.ACCENT))

    rain_values = pd.to_numeric(
        last_24h.get("rainfall_mm", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    rain_total = float(np.clip(rain_values.to_numpy(), 0, None).sum())
    wind_dir = latest.get("wind_direction_deg")

    tiles = [
        theme.tile(
            "Temperature",
            _num(latest.get("temperature_c"), 1),
            "°C",
            _range_foot(last_24h, "temperature_c"),
            theme.CHANNEL_ACCENT["temperature_c"],
            spark("temperature_c"),
        ),
        theme.tile(
            "Humidity",
            _num(latest.get("humidity_pct"), 0),
            "%",
            _range_foot(last_24h, "humidity_pct", 0),
            theme.CHANNEL_ACCENT["humidity_pct"],
            spark("humidity_pct"),
        ),
        theme.tile(
            "Pressure",
            _num(latest.get("pressure_hpa"), 1),
            "hPa",
            _range_foot(last_24h, "pressure_hpa"),
            theme.CHANNEL_ACCENT["pressure_hpa"],
            spark("pressure_hpa"),
        ),
        theme.tile(
            "Rain, 24 h total",
            _num(rain_total, 1),
            "mm",
            "spike-filtered samples",
            theme.CHANNEL_ACCENT["rainfall_mm"],
            spark("rainfall_mm"),
        ),
        theme.tile(
            "Wind speed",
            _num(latest.get("wind_speed_ms"), 1),
            "m/s",
            _range_foot(last_24h, "wind_speed_ms"),
            theme.CHANNEL_ACCENT["wind_speed_ms"],
            spark("wind_speed_ms"),
        ),
        theme.tile(
            "Wind direction",
            _num(wind_dir, 0),
            "°",
            _compass(wind_dir),
            theme.CHANNEL_ACCENT["wind_direction_deg"],
            spark("wind_direction_deg"),
        ),
    ]
    st.markdown(theme.grid(tiles), unsafe_allow_html=True)


def _sun_time(value, tz: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(tz).strftime("%H:%M")


def _weather_icon(code: float | int | None) -> str:
    if code is None or pd.isna(code):
        return theme.icon("cloud")
    value = int(code)
    if value == 0:
        return theme.icon("temp")
    if value in {1, 2, 3, 45, 48}:
        return theme.icon("cloud")
    return theme.icon("drop")


def _forecast_risk(total_mm: float, wet_hours: int = 0) -> tuple[str, str]:
    """Rain-amount risk for the next 24 hours only."""
    if total_mm >= 40 or wet_hours >= 16:
        return "High", "A large rain total is expected"
    if total_mm >= 20 or wet_hours >= 10:
        return "High", "A heavy rain total is expected"
    if total_mm >= 10 or wet_hours >= 6:
        return "Medium", "A meaningful rain total is expected"
    if total_mm >= 2 or wet_hours >= 2:
        return "Low", "Only light rain is expected"
    return "Low", "Little or no rain is expected"


def _rain_summary_line(total_mm: float) -> str:
    if total_mm >= 40:
        return "Very heavy rain expected"
    if total_mm >= 20:
        return "Heavy rain expected"
    if total_mm >= 10:
        return "Rain expected"
    if total_mm >= 2:
        return "Light rain possible"
    return "Mostly dry"


def _wet_hours_by_date(hourly: pd.DataFrame, tz: str) -> dict:
    """Count hours with at least 0.1 mm of rain for each forecast day."""
    del tz  # Daily cards use the same UTC dates as the forecast feed.
    if hourly.empty or "timestamp" not in hourly.columns:
        return {}
    utc = pd.to_datetime(hourly["timestamp"], utc=True)
    rain = pd.to_numeric(hourly["rainfall_mm"], errors="coerce").fillna(0.0)
    counts = rain.ge(0.1).groupby(utc.dt.date).sum()
    return {key: int(val) for key, val in counts.items()}


def _with_wet_hours(daily: pd.DataFrame, hourly: pd.DataFrame, tz: str) -> pd.DataFrame:
    wet = _wet_hours_by_date(hourly, tz)
    out = daily.copy()
    out["wet_hours"] = [int(wet.get(pd.Timestamp(day).date(), 0)) for day in out["date"]]
    return out


def _hours_with_rain_label(hours: int) -> str:
    """Plain label for how many hours in a day are expected to have rain."""
    count = int(max(0, hours))
    if count <= 0:
        return "No rain expected"
    if count >= 22:
        return "Rain all day"
    if count == 1:
        return "Rain for 1 hour"
    if count >= 16:
        return "Rain most of the day"
    return f"Rain for {count} hours"


def _wet_hours_sentence(wet_hours: int) -> str:
    if wet_hours <= 0:
        return "No measurable rain is expected during the next 24 hours."
    if wet_hours == 1:
        return "Rain is expected for about 1 hour during the next 24 hours."
    if wet_hours >= 24:
        return "Rain is expected throughout the next 24 hours."
    if wet_hours >= 18:
        return f"Rain is expected during most of the next 24 hours (about {wet_hours} hours)."
    return f"Rain is expected for about {wet_hours} hours during the next 24 hours."


def _recent_rain(readings: pd.DataFrame, hours: int) -> float | None:
    if readings.empty or "rainfall_mm" not in readings.columns:
        return None
    recent = _recent(readings, hours)
    values = pd.to_numeric(recent["rainfall_mm"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(np.clip(values.to_numpy(dtype=float), 0, None).sum())


def _dam_rain_watch(
    forecast_24h: float,
    forecast_72h: float,
    recent_72h: float | None,
) -> tuple[str, str, str]:
    """Simple rainfall-only watch; never a reservoir-operation decision."""
    combined = forecast_72h + (recent_72h or 0.0)
    if forecast_24h >= 50 or forecast_72h >= 100 or combined >= 150:
        return (
            "High",
            "Increase monitoring of reservoir level, inflow, and spillway conditions.",
            "high",
        )
    if forecast_24h >= 25 or forecast_72h >= 60 or combined >= 100:
        return (
            "Raised",
            "Check reservoir level and inflow more often while the rain continues.",
            "moderate",
        )
    return (
        "Routine",
        "Continue normal monitoring and review the next update.",
        "low",
    )


def _crop_rain_watch(
    forecast_24h: float,
    wet_hours: int,
    humidity: float | None,
) -> tuple[str, str, str]:
    if forecast_24h >= 25 or wet_hours >= 14:
        return (
            "Wet spell",
            "Hold irrigation. Check the canopy for disease after this rain.",
            "high",
        )
    if forecast_24h >= 8 or wet_hours >= 6:
        return (
            "Rain likely",
            "Skip irrigation while the plants are wet.",
            "moderate",
        )
    if humidity is not None and not pd.isna(humidity) and float(humidity) >= 85:
        return (
            "Humid",
            "The air is moist. Only light rain is expected.",
            "moderate",
        )
    return (
        "Mostly dry",
        "Rain is unlikely. Check whether the plants need water.",
        "low",
    )


def _plain_confidence(
    nwp_verification: pd.DataFrame,
    ensemble_spread: float | None,
    skill: dict[str, float] | None = None,
) -> tuple[str, str]:
    bands = skill or {
        "confidence_mae_good": 1.5,
        "confidence_mae_fair": 2.5,
        "spread_steady": 1.0,
        "spread_mixed": 2.0,
    }
    if not nwp_verification.empty:
        temp_nwp = nwp_verification.loc[nwp_verification["channel"] == "temperature_c"]
        if not temp_nwp.empty:
            mae = float(temp_nwp["mae"].min())
            if mae <= bands["confidence_mae_good"]:
                return "Good", "Earlier forecasts were close to what happened"
            if mae <= bands["confidence_mae_fair"]:
                return "Fair", "Earlier forecasts were reasonably close"
            return "Limited", "Earlier forecasts were not very close"
    if ensemble_spread is not None and np.isfinite(ensemble_spread):
        if ensemble_spread <= bands["spread_steady"]:
            return "Steady", "The weather services mostly agree"
        if ensemble_spread <= bands["spread_mixed"]:
            return "Mixed", "The weather services give different answers"
        return "Uncertain", "The weather services disagree"
    return "Not checked yet", "There is not enough past information yet"


def _plain_local_check(walkforward: pd.DataFrame, local_scores: pd.DataFrame) -> tuple[str, str]:
    if not walkforward.empty:
        wf = walkforward.loc[
            (walkforward["channel"] == "temperature_c")
            & (walkforward["model"] == "gradient_boosting")
        ]
        if not wf.empty:
            best = wf.loc[wf["lead_hours"] == 24]
            row = best.iloc[0] if not best.empty else wf.iloc[0]
            mae = float(row["mae"])
            if mae <= 1.0:
                return "Good", f"Local model error about {mae:.1f} °C at 24 hours"
            if mae <= 2.0:
                return "Fair", f"Local model error about {mae:.1f} °C at 24 hours"
            return "Limited", f"Local model error about {mae:.1f} °C at 24 hours"
    if not local_scores.empty:
        temp_scores = local_scores.loc[local_scores["channel"] == "temperature_c"]
        if not temp_scores.empty:
            gb = temp_scores.loc[temp_scores["model"].str.contains("gradient")]
            subset = gb if not gb.empty else temp_scores
            mae = float(subset["mae"].min())
            return "Checked", f"Local model error about {mae:.1f} °C in the last test"
    return "Not ready", "Local sensor model has not been tested yet"


def _overview_alert_items(
    *,
    cfg: dict,
    project: dict,
    readings: pd.DataFrame,
    reference: pd.DataFrame,
    coeffs: pd.DataFrame,
    display_forecast: ForecastBundle | None,
    confidence_label: str,
    confidence_note: str,
) -> list[tuple[str, str]]:
    """Return (severity, message) pairs for the Overview alerts strip."""
    items: list[tuple[str, str]] = []
    kind = str(project.get("kind", "dam"))
    now = pd.Timestamp.now(tz="UTC")
    interval = float((cfg.get("sampling") or {}).get("interval_minutes", 5))
    stale_limit = max(interval * 6, 30.0)

    if display_forecast is not None:
        hourly = display_forecast.hourly
        future = hourly.loc[pd.to_datetime(hourly["timestamp"], utc=True) >= now]
        if future.empty:
            future = hourly
        rain_24 = float(
            pd.to_numeric(future.iloc[:24]["rainfall_mm"], errors="coerce").fillna(0).sum()
        )
        wet_hours = int(
            (pd.to_numeric(future.iloc[:24]["rainfall_mm"], errors="coerce").fillna(0) >= 0.1).sum()
        )
        if kind == "plantation":
            watch, _, level = _crop_rain_watch(rain_24, wet_hours, None)
        else:
            recent = _recent_rain(readings, 72)
            rain_72 = float(
                pd.to_numeric(future.iloc[:72]["rainfall_mm"], errors="coerce").fillna(0).sum()
            )
            watch, _, level = _dam_rain_watch(rain_24, rain_72, recent)
        severity = "ok" if level == "low" else ("danger" if level == "high" else "warn")
        items.append((severity, f"Rain watch · {watch}"))

    for alert in _alerts(cfg, coeffs):
        items.append(("warn", alert))

    if not readings.empty and "timestamp" in readings.columns:
        last_sensor = pd.to_datetime(readings["timestamp"], utc=True).max()
        age_min = (now - last_sensor).total_seconds() / 60.0
        if age_min > stale_limit:
            items.append(
                ("warn", f"Sensor data is {age_min / 60.0:.1f} h old (limit ~{stale_limit:.0f} min).")
            )
    elif not reference.empty and "timestamp" in reference.columns:
        last_ref = pd.to_datetime(reference["timestamp"], utc=True).max()
        age_h = (now - last_ref).total_seconds() / 3600.0
        if age_h > 6:
            items.append(("warn", f"Area weather history is {age_h:.0f} h old."))
    else:
        items.append(("warn", "No sensor or area history loaded yet."))

    status = load_refresh_status()
    finished = status.get("finished_at")
    if finished:
        try:
            when = pd.Timestamp(finished)
            if when.tzinfo is None:
                when = when.tz_localize("UTC")
            age_h = (now - when).total_seconds() / 3600.0
            if not status.get("ok"):
                items.append(("danger", f"Last analysis failed ({age_h:.0f} h ago)."))
            elif age_h > 36:
                items.append(("warn", f"Last analysis is {age_h:.0f} h old — consider Run analysis."))
            else:
                items.append(("ok", _format_analysis_stamp(status)))
        except Exception:
            items.append(("ok", _format_analysis_stamp(status)))
    else:
        items.append(("warn", "No analysis run yet — use Run analysis in the sidebar."))

    items.append(("ok", f"Confidence · {confidence_label}: {confidence_note}"))
    members = list((cfg.get("forecast") or {}).get("ensemble_members") or [])
    if members:
        labels = {"ecmwf": "ECMWF", "gfs": "GFS", "met_norway": "MET Norway"}
        chip = ", ".join(labels.get(m, m) for m in members)
        items.append(("ok", f"Models · {chip}"))
    return items


def _render_overview_alerts(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    chips = []
    for severity, message in items:
        chips.append(
            f'<span class="alert-chip alert-{html.escape(severity)}">'
            f"{html.escape(message)}</span>"
        )
    st.markdown(
        '<div class="overview-alerts">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def _forecast_overview(
    bundle: ForecastBundle,
    cfg: dict,
    station_readings: pd.DataFrame,
    local_scores: pd.DataFrame,
    walkforward: pd.DataFrame,
    nwp_verification: pd.DataFrame,
    tz: str,
    ensemble_spread: float | None = None,
    project: dict | None = None,
    reference: pd.DataFrame | None = None,
    coeffs: pd.DataFrame | None = None,
) -> None:
    hourly = bundle.hourly.copy()
    daily = _with_wet_hours(bundle.daily.copy(), hourly, tz)
    now = pd.Timestamp.now(tz="UTC")
    now_local = now.tz_convert(tz)
    future = hourly.loc[hourly["timestamp"] >= now].copy()
    if future.empty:
        future = hourly.copy()
    next_24h = future.iloc[:24]
    current = future.iloc[0]
    high = float(pd.to_numeric(next_24h["temperature_c"], errors="coerce").max())
    low = float(pd.to_numeric(next_24h["temperature_c"], errors="coerce").min())
    rain_total = float(
        pd.to_numeric(next_24h["rainfall_mm"], errors="coerce").fillna(0).sum()
    )
    rain_72h = float(
        pd.to_numeric(future.iloc[:72]["rainfall_mm"], errors="coerce").fillna(0).sum()
    )
    wet_hours = int(
        (pd.to_numeric(next_24h["rainfall_mm"], errors="coerce").fillna(0) >= 0.1).sum()
    )
    wind_max = float(pd.to_numeric(next_24h["wind_speed_ms"], errors="coerce").max())
    risk, _risk_note = _forecast_risk(rain_total, wet_hours)
    rain_headline = _rain_summary_line(rain_total)
    wet_hours_text = _wet_hours_sentence(wet_hours)
    skill = _skill_prefs(cfg)
    confidence_label, confidence_note = _plain_confidence(
        nwp_verification, ensemble_spread, skill
    )
    _render_overview_alerts(
        _overview_alert_items(
            cfg=cfg,
            project=project or {},
            readings=station_readings,
            reference=reference if reference is not None else pd.DataFrame(),
            coeffs=coeffs if coeffs is not None else pd.DataFrame(),
            display_forecast=bundle,
            confidence_label=confidence_label,
            confidence_note=confidence_note,
        )
    )
    recent_72h = _recent_rain(station_readings, 72)
    humidity_now = current.get("humidity_pct")
    kind = str((project or {}).get("kind", "dam"))
    station_is_live = (
        not station_readings.empty
        and "source" in station_readings.columns
        and station_readings["source"].eq("sensor").any()
    )
    if kind == "plantation":
        watch_label, watch_action, _watch_class = _crop_rain_watch(
            rain_total, wet_hours, None if humidity_now is None else float(pd.to_numeric(humidity_now, errors="coerce"))
        )
        watch_pct = {"Mostly dry": 28.0, "Humid": 48.0, "Rain likely": 68.0, "Wet spell": 90.0}.get(
            watch_label, 40.0
        )
        gauge_title = "Rain outlook"
        gauge_sub = "For irrigation and crop care"
        humidity_foot = "Plantation microclimate"
        week_note = "Rain total and how long the rain lasts at the plantation"
        if station_is_live:
            source_note = (
                "Rain outlook is area weather at this pin, checked against the solar station when available."
            )
        else:
            source_note = (
                "Rain millimetres are area estimates at this pin, not a plantation rain gauge. "
                "They will switch to the solar station when it is plugged in."
            )
    else:
        watch_label, watch_action, _watch_class = _dam_rain_watch(rain_total, rain_72h, recent_72h)
        watch_pct = {"Routine": 34.0, "Raised": 68.0, "High": 92.0}.get(watch_label, 34.0)
        gauge_title = "Rainfall watch"
        gauge_sub = "For reservoir monitoring only"
        humidity_foot = "Now at the dam area"
        week_note = "Rain total and how long the rain lasts at INDRA"
        if station_is_live:
            source_note = (
                "Rain outlook is area weather at this pin. "
                "Recent rain uses the station. Rainfall alone cannot decide gate operations."
            )
        else:
            source_note = (
                "Rain millimetres are area estimates at this pin, not a dam rain gauge. "
                "Rainfall alone cannot decide gate operations."
            )
    peak_hourly_rain = float(
        pd.to_numeric(future.iloc[:72]["rainfall_mm"], errors="coerce").fillna(0).max()
    )
    recent_rain_label = (
        f"{recent_72h:.1f} mm" if recent_72h is not None else "Not available"
    )
    recent_rain_kicker = "Recent 3-day rain · sensor" if station_is_live else "Recent 3-day rain"

    local_time = pd.Timestamp(current["timestamp"]).tz_convert(tz)
    current_temp = float(current["temperature_c"])
    humidity = current.get("humidity_pct")
    humidity_label = _num(humidity, 0) if humidity is not None else "—"
    pressure_now = current.get("pressure_hpa")

    fourth = (
        theme.stat_card(
            "Pressure",
            _num(pressure_now, 1),
            "hPa",
            "Used for rain outlook",
            "layers",
        )
        if kind == "plantation"
        else theme.stat_card(
            "Wind",
            f"{float(current['wind_speed_ms']):.1f}",
            "m/s",
            f"Peak {wind_max:.1f}",
            "wind",
        )
    )

    st.markdown(
        '<div class="v-stats">'
        + theme.stat_card("Temperature", f"{current_temp:.1f}", "°C", f"{low:.0f}–{high:.0f}°", "temp")
        + theme.stat_card("Rainfall (24h)", f"{rain_total:.1f}", "mm", rain_headline, "drop")
        + theme.stat_card("Humidity", humidity_label, "%", humidity_foot, "cloud")
        + fourth
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-row">'
        + theme.welcome_card(
            local_time.strftime("%A, %d %b"),
            rain_headline,
            wet_hours_text,
            f"{rain_total:.1f} mm expected in the next 24 hours",
        )
        + theme.semi_gauge(
            watch_pct,
            watch_label,
            watch_action,
            title=gauge_title,
            subtitle=gauge_sub,
        )
        + theme.ring_gauge(
            confidence_label,
            "Forecast check",
            confidence_note,
            {
                "Good": 90.0,
                "Fair": 72.0,
                "Steady": 82.0,
                "Mixed": 58.0,
                "Limited": 42.0,
                "Uncertain": 30.0,
                "Not checked yet": 22.0,
                "Checked": 68.0,
            }.get(confidence_label, 50.0),
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns((1.7, 1), gap="medium")
    with left:
        st.markdown(
            '<div class="chart-head"><h3>Rain outlook</h3>'
            f'<span class="{"warn" if risk == "High" else "note"}">'
            f"{html.escape(risk)} rain amount</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            charts.overview_trend_chart(future, tz),
            width="stretch",
            config={"displaylogo": False},
            key="chart_overview_trend",
        )
    with right:
        st.markdown(
            '<div class="chart-head"><h3>Rain by hour</h3>'
            f'<span class="note">{wet_hours} of 24 hours with rain</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            charts.activity_bar_chart(next_24h, tz),
            width="stretch",
            config={"displaylogo": False},
            key="chart_overview_activity",
        )
        st.markdown(
            f"""
            <div class="activity-stats">
              <div class="item"><div class="mini-icon">{theme.icon("drop")}</div>
                <div><p class="k">Next 24 hours</p><p class="v">{rain_total:.1f} mm</p></div></div>
              <div class="item"><div class="mini-icon">{theme.icon("cloud")}</div>
                <div><p class="k">Next 3 days</p><p class="v">{rain_72h:.1f} mm</p></div></div>
              <div class="item"><div class="mini-icon">{theme.icon("gauge")}</div>
                <div><p class="k">Heaviest hour</p><p class="v">{peak_hourly_rain:.1f} mm</p></div></div>
              <div class="item"><div class="mini-icon">{theme.icon("layers")}</div>
                <div><p class="k">{html.escape(recent_rain_kicker)}</p><p class="v">{html.escape(recent_rain_label)}</p></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        theme.section("This week", week_note),
        unsafe_allow_html=True,
    )
    day_cards: list[str] = []
    for _, row in daily.head(7).iterrows():
        date = pd.Timestamp(row["date"])
        is_today = date.date() == now_local.date()
        day_name = "Today" if is_today else date.strftime("%A")
        css = "day today" if is_today else "day"
        day_rain = float(row["rainfall_sum_mm"])
        day_wet = 0
        if "wet_hours" in row.index and pd.notna(row["wet_hours"]):
            day_wet = int(row["wet_hours"])
        day_hours = _hours_with_rain_label(day_wet)
        hours_class = "rain-hours all-day" if day_wet >= 22 else "rain-hours"
        day_cards.append(
            f'<div class="{css}">'
            f'<div class="d">{html.escape(day_name)}</div>'
            f'<div class="date">{html.escape(date.strftime("%d %b"))}</div>'
            f'<div class="icon">{_weather_icon(row.get("weather_code"))}</div>'
            f'<div class="rain-main">{day_rain:.1f}<span>mm</span></div>'
            f'<div class="{hours_class}">{html.escape(day_hours)}</div>'
            f'<div class="temps">H {float(row["temperature_max_c"]):.0f}° · '
            f'L {float(row["temperature_min_c"]):.0f}°</div>'
            "</div>"
        )
    st.markdown(f'<div class="days">{"".join(day_cards)}</div>', unsafe_allow_html=True)

    fetched_local = pd.Timestamp(bundle.fetched_at).tz_convert(tz)
    source_name = _friendly_forecast_source(str(bundle.source))
    st.markdown(
        f'<div class="source-line">'
        f"<strong>Source:</strong> {html.escape(source_name)} · "
        f"updated {fetched_local.strftime('%d %b %Y, %H:%M IST')}. "
        f"{html.escape(source_note)}</div>",
        unsafe_allow_html=True,
    )

    if kind == "plantation" and not station_is_live:
        st.markdown(
            theme.section(
                "Sensors that can still be added",
                "Temperature, humidity, and pressure are already in the kit plan.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kit-grid kit-grid-four">'
            + theme.tile("Rain gauge", "Optional", "", "Rain at the crop")
            + theme.tile("Soil moisture", "Optional", "", "For irrigation")
            + theme.tile("Leaf wetness", "Optional", "", "For disease risk")
            + theme.tile("Wireless link", "Ready", "", "Sends readings here")
            + "</div>",
            unsafe_allow_html=True,
        )


def _map_weather_snapshot(
    bundle: ForecastBundle | None,
    project: dict,
    readings: pd.DataFrame,
) -> dict:
    if bundle is None or bundle.hourly.empty:
        return {}
    hourly = bundle.hourly.copy()
    now = pd.Timestamp.now(tz="UTC")
    future = hourly.loc[hourly["timestamp"] >= now]
    if future.empty:
        future = hourly
    next_24h = future.iloc[:24]
    current = future.iloc[0]
    rain_total = float(pd.to_numeric(next_24h["rainfall_mm"], errors="coerce").fillna(0).sum())
    rain_72h = float(pd.to_numeric(future.iloc[:72]["rainfall_mm"], errors="coerce").fillna(0).sum())
    wet_hours = int((pd.to_numeric(next_24h["rainfall_mm"], errors="coerce").fillna(0) >= 0.1).sum())
    humidity = current.get("humidity_pct")
    if str(project.get("kind")) == "plantation":
        humid_val = None if humidity is None else float(pd.to_numeric(humidity, errors="coerce"))
        watch, _, _ = _crop_rain_watch(rain_total, wet_hours, humid_val)
    else:
        watch, _, _ = _dam_rain_watch(rain_total, rain_72h, _recent_rain(readings, 72))
    condition, _ = weather_label(current.get("weather_code"))
    return {
        "temp": float(current["temperature_c"]),
        "rain_24h": rain_total,
        "condition": condition,
        "watch": watch,
        "humidity": humidity,
    }


@st.cache_data(ttl=180, show_spinner=False)
def _lookup_places(query: str, lat: float, lon: float) -> list[tuple[float, float, str]]:
    return maps.search_places(query, lat, lon)


def _render_station_map(
    projects: dict[str, dict],
    project_id: str,
    weather: dict,
) -> None:
    project = projects[project_id]
    station = project["station"]
    lat = float(station["latitude"])
    lon = float(station["longitude"])
    pending_key = f"pending_pin_{project_id}"
    pending = st.session_state.get(pending_key)

    area_km = float(station.get("area_radius_km") or (6 if str(project.get("kind")) == "plantation" else 12))
    _page_intro("Map", "Set the pin and watch area.")
    st.markdown(
        '<div class="v-stats">'
        + theme.stat_card("Latitude", f"{lat:.5f}", "°N", str(station.get("site", "")), "layers")
        + theme.stat_card("Longitude", f"{lon:.5f}", "°E", str(project["label"]), "layers")
        + theme.stat_card("Watch area", f"{area_km:.0f}", "km", "Circle on the map", "layers")
        + theme.stat_card(
            "Next 24 h rain",
            "—" if weather.get("rain_24h") is None else f"{float(weather['rain_24h']):.1f}",
            "mm",
            str(weather.get("watch", "Forecast area")),
            "drop",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    area_col, save_area_col = st.columns((4, 1), gap="small")
    with area_col:
        area_km = st.number_input(
            "Watch area in kilometres",
            min_value=1.0,
            max_value=50.0,
            value=float(area_km),
            step=1.0,
            key=f"area_km_{project_id}",
            help="The coloured circle around the station. Rain outlook still uses the pin.",
        )
    with save_area_col:
        st.markdown("<div class='map-save-spacer'></div>", unsafe_allow_html=True)
        if st.button("Save area", width="stretch", key=f"save_area_{project_id}"):
            save_station_location(project_id, lat, lon, area_radius_km=float(area_km))
            load_config.cache_clear()
            _load.clear()
            st.rerun()

    components.html(
        maps.build_hub_map_html(
            projects,
            project_id,
            weather=weather,
            pending=pending,
            area_radius_km=float(area_km),
        ),
        height=560,
        scrolling=False,
    )

    query = st.text_input(
        "Find a place",
        placeholder="Type a dam, village, or town — then pick a match",
        key=f"place_query_{project_id}",
    )
    hits: list[tuple[float, float, str]] = []
    if len(query.strip()) >= 3:
        hits = _lookup_places(query.strip(), lat, lon)
        if not hits:
            st.warning("No matching places. Try a nearby town or the full dam name.")
        else:
            picked = st.selectbox(
                "Matching places",
                options=list(range(len(hits))),
                format_func=lambda index: hits[index][2],
                index=None,
                placeholder="Pick a match to move the pin",
                key=f"place_pick_{project_id}_{query.strip().casefold()}",
            )
            if picked is not None:
                hit = hits[picked]
                already = (
                    pending is not None
                    and abs(pending[0] - hit[0]) < 1e-5
                    and abs(pending[1] - hit[1]) < 1e-5
                )
                if not already:
                    st.session_state[pending_key] = (hit[0], hit[1])
                    st.session_state[f"place_name_{project_id}"] = hit[2]
                    st.rerun()
    else:
        st.caption("Type at least 3 letters to see place suggestions.")

    if pending is not None:
        plat, plon = pending
        place_name = st.session_state.get(f"place_name_{project_id}", "Selected place")
        st.info(
            f"New pin: {plat:.5f}°N, {plon:.5f}°E · {place_name}. "
            "Press Save location to use this place for forecasts."
        )
        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("Save location", type="primary", width="stretch"):
                save_station_location(project_id, plat, plon, area_radius_km=float(area_km))
                st.session_state.pop(pending_key, None)
                st.session_state.pop(f"place_name_{project_id}", None)
                load_config.cache_clear()
                _load.clear()
                st.success("Station location saved. Weather will refresh for this pin.")
                st.rerun()
        with clear_col:
            if st.button("Cancel", width="stretch"):
                st.session_state.pop(pending_key, None)
                st.session_state.pop(f"place_name_{project_id}", None)
                st.rerun()

    with st.expander("Type coordinates instead"):
        typed_lat = st.number_input("Latitude", value=float(lat), format="%.5f", key=f"lat_in_{project_id}")
        typed_lon = st.number_input("Longitude", value=float(lon), format="%.5f", key=f"lon_in_{project_id}")
        if st.button("Save typed location", key=f"save_typed_{project_id}"):
            save_station_location(
                project_id,
                float(typed_lat),
                float(typed_lon),
                area_radius_km=float(area_km),
            )
            st.session_state.pop(pending_key, None)
            load_config.cache_clear()
            _load.clear()
            st.rerun()
    st.caption(
        "The map starts on Detailed streets so it always loads. "
        "Use the layers button for satellite photos. Close-up satellite pictures are limited in this area."
    )


def _ensemble_rain_history(forecast_verification: pd.DataFrame) -> pd.DataFrame:
    """One rain row per issued Overview ensemble outlook (Open-Meteo referee)."""
    if forecast_verification.empty:
        return forecast_verification
    fv = forecast_verification.copy()
    rain = fv[
        (fv["channel"] == "rainfall_mm")
        & (fv["model"].astype(str).str.contains("ensemble", case=False, na=False))
        & (~fv["model"].astype(str).str.contains("imd", case=False, na=False))
    ].copy()
    if rain.empty:
        return rain
    rain["issued_at"] = pd.to_datetime(rain["issued_at"], utc=True)
    rain = rain.sort_values("issued_at")
    return rain.reset_index(drop=True)


def _skill_band(day_mae: float | None, skill: dict[str, float] | None = None) -> str:
    bands = skill or {"day_mae_good": 3.0, "day_mae_fair": 6.0}
    if day_mae is None or (isinstance(day_mae, float) and pd.isna(day_mae)):
        return "Not scored"
    if day_mae <= bands["day_mae_good"]:
        return "Good"
    if day_mae <= bands["day_mae_fair"]:
        return "Fair"
    if day_mae <= bands["day_mae_fair"] * 2:
        return "Weak"
    return "Poor"


def _fmt_mm(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.1f} mm"


def _fmt_csi(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.2f}"


def _render_sensors_page(cfg: dict, project: dict, weather_readings: pd.DataFrame) -> None:
    """Multi-domain sensor hub view for INDRA / ISSA."""
    label = str(project.get("label", "project"))
    domains = [d for d in project_domains(project) if d != "weather"]
    _page_intro(
        "Sensors",
        f"Live-style telemetry for {label}. Weather stays on Overview/Data; "
        "other domains are shown here (simulated until hardware is installed).",
    )
    station_id = str(project["station"]["id"])
    tz = str(project["station"].get("timezone", "Asia/Kolkata"))
    names = domain_labels()

    if not domains:
        st.info("No extra sensor domains configured for this project.")
        return

    domain = st.radio(
        "Domain",
        domains,
        format_func=lambda key: names.get(key, key),
        horizontal=True,
        key=f"sensor_domain_{project['id']}",
    )
    latest = load_telemetry_latest(station_id=station_id, domain=domain, cfg=cfg)
    series = load_telemetry(station_id=station_id, domain=domain, cfg=cfg)
    meta = DOMAIN_CHANNELS.get(domain, {})

    if latest.empty:
        st.warning(
            f"No {names.get(domain, domain)} data yet. "
            "Run `python scripts/run_heimdall_simulate.py` to seed simulated sensors."
        )
        return

    tiles = []
    for _, row in latest.iterrows():
        channel = str(row["channel"])
        info = meta.get(channel, {})
        value = row["value"]
        unit = str(row.get("unit") or info.get("unit") or "")
        if channel == "camera_online":
            text = "Online" if float(value or 0) >= 0.5 else "Offline"
            detail = "CCTV heartbeat"
        elif pd.isna(value):
            text = "—"
            detail = unit
        else:
            text = f"{float(value):.2f}" if abs(float(value)) < 1000 else f"{float(value):.1f}"
            detail = unit
        tiles.append(
            theme.tile(
                str(info.get("label", channel)),
                text,
                "" if channel == "camera_online" else unit,
                detail if channel != "camera_online" else "Simulated camera status",
                theme.ACCENT,
            )
        )
    st.markdown(theme.grid(tiles), unsafe_allow_html=True)

    if domain == "cctv":
        online_row = latest.loc[latest["channel"] == "camera_online"]
        online = bool(float(online_row["value"].iloc[0]) >= 0.5) if not online_row.empty else False
        st.caption(
            "CCTV pane shows status telemetry only (online, motion counts, frame age) — "
            "not live video. Replace with a camera stream URL when hardware is ready."
        )
        st.markdown(
            theme.panel(
                "Camera status",
                (
                    f'<p style="color:#A0AEC0;margin:0;line-height:1.5">'
                    f"{'Feed healthy' if online else 'Feed offline / stale'} · "
                    f"station <code>{html.escape(station_id)}</code></p>"
                ),
                note="Simulated until installation",
                tag="Online" if online else "Offline",
                tag_class="up" if online else "down",
            ),
            unsafe_allow_html=True,
        )

    channels = sorted(series["channel"].unique().tolist()) if not series.empty else []
    if not channels:
        return
    channel = st.selectbox(
        "Trend channel",
        channels,
        format_func=lambda key: str(meta.get(key, {}).get("label", key)),
        key=f"sensor_channel_{project['id']}_{domain}",
    )
    subset = series.loc[series["channel"] == channel].copy()
    subset = subset.sort_values("timestamp")
    unit = str(meta.get(channel, {}).get("unit") or "")
    fig = charts.telemetry_trend_chart(
        subset,
        title=str(meta.get(channel, {}).get("label", channel)),
        unit=unit,
        tz=tz,
    )
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"tel_{domain}_{channel}")

    # Weather quick link for context
    if not weather_readings.empty:
        last_wx = pd.Timestamp(weather_readings.iloc[-1]["timestamp"]).tz_convert(tz)
        st.caption(f"Weather last sample {last_wx.strftime('%d %b %Y · %H:%M IST')} (see Data / Overview).")


def _render_progress_page(
    forecast_verification: pd.DataFrame,
    tz: str,
    project: dict,
    cfg: dict | None = None,
) -> None:
    """Show rain-outlook skill from the first archived run to the latest."""
    kind = str(project.get("kind", "dam"))
    label = str(project.get("label", "this site"))
    skill = _skill_prefs(cfg or {})
    good_cut = skill["day_mae_good"]
    fair_cut = skill["day_mae_fair"]
    csi_target = skill["csi_target"]

    history = _ensemble_rain_history(forecast_verification)
    if history.empty:
        _page_intro("Progress", f"Scores appear after outlooks mature for {label}.")
        st.info("No scored outlooks yet for this project.")
        st.caption("Truth: Open-Meteo at the project pin. Run analysis to archive and verify.")
        return

    issued = pd.to_datetime(history["issued_at"], utc=True)
    show_bias_fix = kind == "dam"
    fix_at = pd.Timestamp("2026-09-17", tz="UTC")
    before = history.loc[issued < fix_at] if show_bias_fix else history.iloc[0:0]
    after = history.loc[issued >= fix_at] if show_bias_fix else history
    before_day = (
        pd.to_numeric(before["daily_mae"], errors="coerce") if not before.empty else pd.Series(dtype=float)
    )
    after_day = (
        pd.to_numeric(after["daily_mae"], errors="coerce") if not after.empty else pd.Series(dtype=float)
    )
    after_csi = (
        pd.to_numeric(after["csi"], errors="coerce") if not after.empty else pd.Series(dtype=float)
    )

    first = history.iloc[0]
    latest = history.iloc[-1]
    first_day = float(first["daily_mae"]) if pd.notna(first.get("daily_mae")) else None
    first_csi = float(first["csi"]) if pd.notna(first.get("csi")) else None
    latest_day = float(latest["daily_mae"]) if pd.notna(latest.get("daily_mae")) else None
    latest_csi = float(latest["csi"]) if pd.notna(latest.get("csi")) else None
    first_local = issued.iloc[0].tz_convert(tz).strftime("%d %b %Y")
    last_local = issued.iloc[-1].tz_convert(tz).strftime("%d %b %Y")
    start_mean = float(before_day.mean()) if before_day.notna().any() else first_day
    now_median = float(after_day.median()) if after_day.notna().any() else latest_day

    if (
        show_bias_fix
        and start_mean is not None
        and now_median is not None
        and start_mean > fair_cut * 2
        and now_median <= fair_cut
    ):
        journey_title = "Clear improvement from the start"
        journey = (
            f"We started near {_fmt_mm(start_mean)}/day error. Typical error now is about "
            f"{_fmt_mm(now_median)}/day after the bias fix. Amounts improved a lot; timing is still mixed."
        )
        tag, tag_cls = "Improved", "up"
    elif start_mean is not None and now_median is not None and now_median < start_mean - 1:
        journey_title = "Improving since tracking began"
        journey = (
            f"Day rain error moved from about {_fmt_mm(start_mean)} at the start "
            f"to about {_fmt_mm(now_median)} now."
        )
        tag, tag_cls = "Improving", "up"
    elif start_mean is not None and now_median is not None and now_median > start_mean + 1:
        journey_title = "Recent runs need attention"
        journey = (
            f"Day rain error was about {_fmt_mm(start_mean)} early on and is about "
            f"{_fmt_mm(now_median)} on recent typical runs."
        )
        tag, tag_cls = "Watch trend", "down"
    else:
        journey_title = "Skill from first run to now"
        journey = (
            f"Tracking {len(history)} verified outlooks for {label} "
            f"from {first_local} to {last_local}."
        )
        tag, tag_cls = "Tracked", "steady"

    if len(after_day.dropna()) >= 4:
        recent = float(after_day.dropna().iloc[-3:].mean())
        earlier = float(after_day.dropna().iloc[:-3].tail(3).mean())
        delta = recent - earlier
        if abs(delta) < 0.5:
            trend_note = "Latest runs are steady."
        elif delta < 0:
            trend_note = f"Latest runs are improving (~{abs(delta):.1f} mm better)."
        else:
            trend_note = f"Latest runs are a bit weaker (~{delta:.1f} mm higher error)."
    else:
        trend_note = "Need a few more scored runs to judge the latest trend."

    improvement = "Typical recent day error" if not show_bias_fix else "After bias fix (median)"
    if start_mean is not None and now_median is not None and start_mean > 0:
        pct = 100.0 * (start_mean - now_median) / start_mean
        if pct >= 5:
            improvement = f"About {pct:.0f}% lower day error than the start"
        elif pct <= -5:
            improvement = f"About {abs(pct):.0f}% higher day error than the start"
        else:
            improvement = "Similar day error to the start"

    good_n = int((after_day <= good_cut).sum()) if after_day.notna().any() else 0
    fair_n = (
        int(((after_day > good_cut) & (after_day <= fair_cut)).sum())
        if after_day.notna().any()
        else 0
    )
    weak_n = int((after_day > fair_cut).sum()) if after_day.notna().any() else 0
    timing_n = len(after_csi.dropna())
    useful_timing = int((after_csi >= csi_target).sum()) if after_csi.notna().any() else 0
    capability = (
        "Useful for a rough rain outlook and planning."
        if after_day.notna().any() and float(after_day.median()) <= fair_cut
        else "Amounts still need care on some days. Keep watching as new scores arrive."
    )

    _page_intro(
        "Progress",
        f"Rain skill for {label} from {first_local} to {last_local}.",
    )

    # 1. Journey summary
    journey_body = (
        f'<p style="color:#A0AEC0;font-size:.86rem;line-height:1.5;margin:0 0 10px">'
        f"{html.escape(journey)}</p>"
        f'<div class="cta" style="display:inline-block;margin:0;padding:7px 12px;'
        f'border-radius:10px;color:#fff;font-size:.74rem;font-weight:700;'
        f'background:rgba(255,255,255,.06);border:1px solid rgba(226,232,240,.12)">'
        f"{html.escape(trend_note)}</div>"
    )
    st.markdown(
        theme.panel(
            journey_title,
            journey_body,
            note=f"{len(history)} scored outlooks · Open-Meteo at pin",
            tag=tag,
            tag_class=tag_cls,
        ),
        unsafe_allow_html=True,
    )

    # 2. Then vs now
    left_val = (
        f'{_fmt_mm(start_mean).replace(" mm", "")}<span class="u">mm/day</span>'
        if start_mean is not None
        else "—"
    )
    right_val = (
        f'{_fmt_mm(now_median).replace(" mm", "")}<span class="u">mm/day</span>'
        if now_median is not None
        else "—"
    )
    st.markdown(
        theme.panel(
            "Where we started → where we are",
            theme.compare_pair(
                "At the start",
                left_val,
                f"{_skill_band(start_mean, skill)} · CSI {_fmt_csi(first_csi)} · {first_local}",
                "Now (typical)",
                right_val,
                f"{improvement} · latest {_skill_band(latest_day, skill)} ({_fmt_mm(latest_day)})",
            ),
            note="Issued rain vs later weather at the pin.",
        ),
        unsafe_allow_html=True,
    )

    # 3. Milestones
    if show_bias_fix:
        before_value = (
            _fmt_mm(float(before_day.mean())) if before_day.notna().any() else "No early runs"
        )
        after_value = _fmt_mm(float(after_day.mean())) if after_day.notna().any() else "—"
        rail = (
            '<div class="milestone-rail">'
            + theme.milestone(
                "Step 1",
                "First verified outlook",
                first_local,
                f"{_fmt_mm(first_day)} · CSI {_fmt_csi(first_csi)}",
            )
            + theme.milestone(
                "Step 2",
                "Before bias fix",
                before_value,
                f"{len(before)} run(s) · amounts too high" if not before.empty else "Tracking began at/after fix",
            )
            + theme.milestone(
                "Step 3",
                "Bias fix · 17 Sep",
                after_value,
                "Wet-hour scale · dry hours stay dry",
            )
            + theme.milestone(
                "Step 4",
                "Latest verified",
                last_local,
                f"{_skill_band(latest_day, skill)} · {_fmt_mm(latest_day)} · CSI {_fmt_csi(latest_csi)}",
            )
            + "</div>"
        )
        milestone_note = "Left to right track record."
        band_note = "After the 17 Sep bias fix."
        timing_label = f"CSI ≥ {csi_target:g} after fix"
    else:
        mid_idx = max(len(history) // 2, 0)
        mid = history.iloc[mid_idx]
        mid_day = float(mid["daily_mae"]) if pd.notna(mid.get("daily_mae")) else None
        mid_csi = float(mid["csi"]) if pd.notna(mid.get("csi")) else None
        mid_local = issued.iloc[mid_idx].tz_convert(tz).strftime("%d %b %Y")
        rail = (
            '<div class="milestone-rail">'
            + theme.milestone(
                "Step 1",
                "First verified outlook",
                first_local,
                f"{_fmt_mm(first_day)} · CSI {_fmt_csi(first_csi)}",
            )
            + theme.milestone(
                "Step 2",
                "Mid track",
                mid_local,
                f"{_skill_band(mid_day, skill)} · {_fmt_mm(mid_day)} · CSI {_fmt_csi(mid_csi)}",
            )
            + theme.milestone(
                "Step 3",
                "Latest verified",
                last_local,
                f"{_skill_band(latest_day, skill)} · {_fmt_mm(latest_day)} · CSI {_fmt_csi(latest_csi)}",
            )
            + "</div>"
        )
        milestone_note = f"Skill track for {label}."
        band_note = "Scored outlooks for this project."
        timing_label = f"CSI ≥ {csi_target:g}"

    st.markdown(
        theme.panel(
            "Milestones along the way",
            rail,
            note=milestone_note,
        ),
        unsafe_allow_html=True,
    )

    # 4. Capability bands
    bands = (
        '<div class="band-row">'
        + theme.band_card(
            "Good amounts",
            str(good_n),
            f"≤ {good_cut:g} mm day error",
            "good",
        )
        + theme.band_card(
            "Fair amounts",
            str(fair_n),
            f"{good_cut:g}–{fair_cut:g} mm day error",
            "fair",
        )
        + theme.band_card(
            "Weak amounts",
            str(weak_n),
            f"> {fair_cut:g} mm day error",
            "weak",
        )
        + theme.band_card(
            "Useful timing",
            f"{useful_timing}/{timing_n}",
            timing_label,
            "timing",
        )
        + "</div>"
        + f'<p style="color:#A0AEC0;font-size:.78rem;margin:8px 0 0;line-height:1.45">'
        f"{html.escape(capability)}</p>"
    )
    st.markdown(
        theme.panel(
            "What the system can do now",
            bands,
            note=band_note,
        ),
        unsafe_allow_html=True,
    )

    # 5. Charts
    st.markdown(
        theme.section("Charts", "Lower day error is better. Higher CSI is better."),
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            theme.panel(
                "Day rain error",
                '<p style="color:#A0AEC0;font-size:.76rem;margin:0">Lower is better</p>',
            ),
            unsafe_allow_html=True,
        )
        bars = charts.skill_amount_bars(history)
        if bars is not None:
            st.plotly_chart(
                bars, width="stretch", config={"displaylogo": False}, key="progress_daymae_bars"
            )
        line_amt = charts.skill_history_chart(
            history, "daily_mae", "Day error (mm)", better="lower"
        )
        if line_amt is not None:
            st.plotly_chart(
                line_amt, width="stretch", config={"displaylogo": False}, key="progress_daymae_line"
            )
    with right:
        st.markdown(
            theme.panel(
                "Rain timing CSI",
                '<p style="color:#A0AEC0;font-size:.76rem;margin:0">Higher is better</p>',
            ),
            unsafe_allow_html=True,
        )
        line = charts.skill_history_chart(history, "csi", "CSI", better="higher")
        if line is not None:
            st.plotly_chart(
                line, width="stretch", config={"displaylogo": False}, key="progress_csi_line"
            )

    # 6. Full history
    st.markdown(
        theme.section("History", "All scored outlooks, oldest first."),
        unsafe_allow_html=True,
    )
    table = history.copy()
    table["#"] = range(1, len(table) + 1)
    table["Date"] = pd.to_datetime(table["issued_at"], utc=True).dt.tz_convert(tz).dt.strftime(
        "%d %b %Y · %H:%M"
    )
    if show_bias_fix:
        table["Era"] = np.where(
            pd.to_datetime(table["issued_at"], utc=True) < fix_at,
            "Before fix",
            "After fix",
        )
    table["Day error (mm)"] = pd.to_numeric(table["daily_mae"], errors="coerce").map(
        lambda v: f"{v:.1f}" if pd.notna(v) else "—"
    )
    table["How close?"] = pd.to_numeric(table["daily_mae"], errors="coerce").map(
        lambda v: _skill_band(v, skill)
    )
    table["CSI"] = pd.to_numeric(table["csi"], errors="coerce").map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
    )
    table["CSI 6h"] = (
        pd.to_numeric(table["csi_6h"], errors="coerce").map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )
        if "csi_6h" in table.columns
        else "—"
    )
    table["Hit %"] = pd.to_numeric(table["hit_rate"], errors="coerce").map(
        lambda v: f"{100 * v:.0f}%" if pd.notna(v) else "—"
    )
    table["Hours checked"] = table["n"].astype(int)
    history_cols = [
        "#",
        "Date",
        *(["Era"] if show_bias_fix else []),
        "How close?",
        "Day error (mm)",
        "CSI",
        "CSI 6h",
        "Hit %",
        "Hours checked",
    ]
    st.dataframe(
        table[history_cols],
        width="stretch",
        hide_index=True,
    )
    st.caption("Scores update after each analysis once an outlook is old enough. Pin weather, not a gauge.")



def main() -> None:
    base_cfg = load_config()
    projects = _projects(base_cfg)
    boot_theme = _active_theme_id(base_cfg)
    st.markdown(theme.stylesheet(boot_theme), unsafe_allow_html=True)
    _accent = theme.theme_tokens(boot_theme)["accent"]
    st.html(theme.paint_streamlit_controls(_accent), unsafe_allow_javascript=True)
    try:
        from src.config import _sync_streamlit_primary

        _sync_streamlit_primary(_accent)
    except Exception:
        pass

    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand">
              <div class="brand-mark">H</div>
              <div><strong>{html.escape(product_name(base_cfg).upper())}</strong>
              <small>{html.escape(product_tagline(base_cfg).upper())}</small></div>
            </div>
            <div class="nav-kicker">PROJECTS</div>
            """,
            unsafe_allow_html=True,
        )
        project_id = st.radio(
            "Project",
            list(projects.keys()),
            index=list(projects.keys()).index(
                str((base_cfg.get("hub") or {}).get("default", "indra"))
            )
            if str((base_cfg.get("hub") or {}).get("default", "indra")) in projects
            else 0,
            format_func=lambda key: projects[key]["label"],
            label_visibility="collapsed",
            key="hub_project",
        )
        project = projects[project_id]
        station_preview = project["station"]
        st.markdown(
            f"""
            <div class="side-meta">
              <div class="k">Station</div>
              <div class="v">{html.escape(str(station_preview["name"]))}</div>
              <div class="s">{html.escape(str(station_preview.get("district", "")))} |
                {html.escape(str(project["blurb"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2, gap="small")
        with b1:
            reload_ui = st.button("Reload UI", width="stretch", help="Clear cached charts only.")
        with b2:
            run_analysis = st.button(
                "Run analysis",
                width="stretch",
                type="primary",
                help="Fetch history, rebuild ensemble, archive, and verify.",
            )
        st.markdown(
            f'<div class="side-analysis">{html.escape(_format_analysis_stamp())}</div>',
            unsafe_allow_html=True,
        )
        if reload_ui:
            load_config.cache_clear()
            _load.clear()
            st.rerun()
        if run_analysis:
            with st.spinner("Running hub analysis…"):
                try:
                    result = run_hub_refresh()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    result = None
            load_config.cache_clear()
            _load.clear()
            if result is not None:
                if result.get("ok"):
                    st.success("Analysis finished.")
                else:
                    st.error(result.get("error") or "Analysis finished with errors.")
            st.rerun()
        st.markdown(
            '<div class="help-card"><strong>Need help?</strong>'
            "<small>Use Settings for models and thresholds. "
            "After model changes, press Run analysis.</small></div>",
            unsafe_allow_html=True,
        )

    data = _load(project_id)
    cfg: dict = data["cfg"]  # type: ignore[assignment]
    project = data["project"]  # type: ignore[assignment]
    readings_all: pd.DataFrame = data["readings"]  # type: ignore[assignment]
    reference: pd.DataFrame = data["reference"]  # type: ignore[assignment]
    calibrated: pd.DataFrame = data["calibrated"]  # type: ignore[assignment]
    coeffs: pd.DataFrame = data["coeffs"]  # type: ignore[assignment]
    scores: pd.DataFrame = data["scores"]  # type: ignore[assignment]
    walkforward: pd.DataFrame = data["walkforward"]  # type: ignore[assignment]
    predictions: pd.DataFrame = data["predictions"]  # type: ignore[assignment]
    operational: ForecastBundle | None = data["operational_forecast"]  # type: ignore[assignment]
    ensemble: pd.DataFrame = data["ensemble"]  # type: ignore[assignment]
    nwp_members: pd.DataFrame = data["nwp_members"]  # type: ignore[assignment]
    source_validation: pd.DataFrame = data["source_validation"]  # type: ignore[assignment]
    forecast_verification: pd.DataFrame = data["forecast_verification"]  # type: ignore[assignment]
    power: pd.DataFrame = data["power"]  # type: ignore[assignment]
    imd: pd.DataFrame = data["imd"]  # type: ignore[assignment]
    forecast_error = str(data["forecast_error"])

    station = cfg["station"]
    tz = station.get("timezone", "Asia/Kolkata")
    alerts = _alerts(cfg, coeffs)
    prefs = _ui_prefs(cfg)
    window = prefs["data_window"]
    show_cal = prefs["show_calibrated"]
    show_ref = prefs["show_reference"]
    show_raw = prefs["show_raw"]
    despike = prefs["despike"]

    iqr_mult = float((cfg.get("calibration") or {}).get("iqr_multiplier", 3.0))
    readings = _despike(readings_all, despike, multiplier=max(iqr_mult, 1.5))
    hours = RANGE_HOURS[window]
    readings_w = _filter_hours(readings, hours)
    reference_w = _filter_hours(reference, hours)
    calibrated_w = _filter_hours(calibrated, hours)

    using_practice_data = (
        not readings.empty
        and "source" in readings.columns
        and not readings["source"].eq("sensor").any()
    )
    if str(project["kind"]) == "plantation" and readings.empty:
        status = ("Station not connected yet", "pill-idle")
    elif using_practice_data:
        status = ("Practice data only", "pill-idle")
    elif coeffs.empty:
        status = ("Sensor setup needed", "pill-idle")
    elif alerts:
        status = ("Sensor drift needs attention", "pill-warn")
    else:
        status = ("Sensors look healthy", "pill-ok")

    if readings.empty:
        stamp = f"Clock {pd.Timestamp.now(tz=tz).strftime('%d %b %Y · %H:%M IST')}"
    else:
        latest_ts = pd.Timestamp(readings.iloc[-1]["timestamp"]).tz_convert(tz)
        stamp = (
            f"Now {pd.Timestamp.now(tz=tz).strftime('%d %b %Y · %H:%M IST')} · "
            f"last reading {latest_ts.strftime('%d %b %Y · %H:%M IST')}"
        )

    _masthead(station, status, stamp, str(project["label"]))

    display_forecast = operational
    ensemble_spread: float | None = None
    if not ensemble.empty:
        hourly = ensemble_as_operational(
            ensemble,
            probability_source=operational.hourly if operational is not None else None,
        )
        daily = ensemble_to_daily(hourly)
        fetched = (
            hourly["fetched_at"].iloc[0]
            if "fetched_at" in hourly.columns
            else pd.Timestamp.now("UTC")
        )
        source = str(hourly["source"].iloc[0]) if "source" in hourly.columns else "NWP ensemble"
        display_forecast = ForecastBundle(
            hourly, daily, pd.Timestamp(fetched).to_pydatetime(), source
        )
        spread_col = pd.to_numeric(hourly.get("temperature_c_spread"), errors="coerce")
        if spread_col is not None and not spread_col.dropna().empty:
            ensemble_spread = float(spread_col.dropna().mean())

    if readings.empty and reference.empty and display_forecast is None:
        st.info(
            "No weather information is available yet."
        )
        return

    page = _section_nav(
        [
            "Overview",
            "Sensors",
            "Forecast",
            "Map",
            "Data",
            "Analytics",
            "Progress",
            "Sources",
            "Settings",
        ],
        key="hub_section",
        default="Overview",
    )

    auto_mins = int(prefs.get("auto_refresh_minutes") or 0)
    if auto_mins > 0:
        import time as _time

        if "_hub_ui_refresh_at" not in st.session_state:
            st.session_state["_hub_ui_refresh_at"] = _time.time()

        @st.fragment(run_every=timedelta(seconds=min(60, auto_mins * 60)))
        def _auto_ui_refresh() -> None:
            st.caption(
                f"Charts auto-reload every {auto_mins} min · "
                "full analysis still needs Run analysis"
            )
            elapsed = _time.time() - float(st.session_state.get("_hub_ui_refresh_at", _time.time()))
            if elapsed >= auto_mins * 60:
                st.session_state["_hub_ui_refresh_at"] = _time.time()
                load_config.cache_clear()
                _load.clear()
                st.rerun()

        _auto_ui_refresh()

    if page == "Overview":
        if display_forecast is None:
            st.error("Rain outlook unavailable. Check the network, then Run analysis or Reload UI.")
            if forecast_error:
                st.caption(forecast_error)
            _render_overview_alerts(
                _overview_alert_items(
                    cfg=cfg,
                    project=project,
                    readings=readings,
                    reference=reference,
                    coeffs=coeffs,
                    display_forecast=None,
                    confidence_label="Not checked yet",
                    confidence_note="Outlook unavailable",
                )
            )
        else:
            _forecast_overview(
                display_forecast,
                cfg,
                readings,
                scores,
                walkforward,
                forecast_verification,
                tz,
                ensemble_spread=ensemble_spread,
                project=project,
                reference=reference,
                coeffs=coeffs,
            )

    if page == "Sensors":
        _render_sensors_page(cfg, project, readings)

    if page == "Forecast":
        _page_intro(
            "7-day forecast",
            "Interactive rain, temperature, wind, and model agreement for this project.",
        )
        if display_forecast is None:
            st.info("Weekly forecast unavailable. Try Run analysis or Reload UI.")
            if forecast_error:
                st.caption(forecast_error)
        else:
            daily = _with_wet_hours(display_forecast.daily, display_forecast.hourly, tz)
            first_day = pd.Timestamp(daily.iloc[0]["date"]).strftime("%d %b")
            last_day = pd.Timestamp(daily.iloc[-1]["date"]).strftime("%d %b %Y")
            members = list((cfg.get("forecast") or {}).get("ensemble_members") or [])
            member_note = ", ".join(
                {"ecmwf": "ECMWF", "gfs": "GFS", "met_norway": "MET Norway"}.get(m, m)
                for m in members
            )
            st.caption(
                f"{first_day} – {last_day}"
                + (f" · models {member_note}" if member_note else "")
                + (
                    f" · ensemble spread {ensemble_spread:.1f} °C"
                    if ensemble_spread is not None and np.isfinite(ensemble_spread)
                    else ""
                )
            )
            payload = build_forecast_payload(
                daily=daily,
                hourly=display_forecast.hourly,
                nwp_members=nwp_members,
                tz=tz,
                theme_id=_active_theme_id(cfg),
                project_kind=str(project.get("kind", "dam")),
            )
            components.html(render_forecast_board_html(payload), height=1120, scrolling=False)

    if page == "Map":
        _render_station_map(
            projects,
            project_id,
            _map_weather_snapshot(display_forecast, project, readings),
        )

    if page == "Analytics":
        _page_intro(
            "Station checks",
            "Sensor setup quality and local prediction tests. "
            "For the interactive 7-day outlook, open Forecast.",
        )

    if page == "Data":
        _page_intro(
            "Station readings",
            (
                "Station offline. Overview still shows the area forecast."
                if str(project["kind"]) == "plantation" and readings.empty
                else "Practice data until the real station is online."
                if using_practice_data
                else "Latest values from the station."
            ),
        )
        if not readings.empty:
            st.markdown(theme.section("Now", f"Last {window.lower()}."), unsafe_allow_html=True)
            _conditions(cfg, readings)
        else:
            st.info(
                "No station readings yet."
                if str(project["kind"]) != "plantation"
                else "Waiting for the solar station."
            )
        if alerts:
            st.warning("  \n".join(f"- {a}" for a in alerts))

        st.markdown(
            theme.section("Temperature", f"Last {window.lower()}."),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            charts.series_chart(
                "temperature_c", "°C", _label(cfg, "temperature_c"),
                readings_w, reference_w, calibrated_w, tz, show_raw, show_cal, show_ref,
            ),
            width="stretch",
            config={"displaylogo": False},
            key="chart_live_temperature",
        )

        left, right = st.columns((1, 1))
        with left:
            st.markdown(
                theme.section("Wind direction", "Which way the wind usually comes from"),
                unsafe_allow_html=True,
            )
            rose = charts.wind_rose(readings_w)
            if rose is None:
                st.caption("Not enough wind data yet.")
            else:
                st.plotly_chart(
                    rose, width="stretch", config={"displaylogo": False}, key="chart_wind_rose"
                )
        with right:
            st.markdown(
                theme.section("Are the readings usable?", "A quick check for missing or unusual data"),
                unsafe_allow_html=True,
            )
            temp_cal = (
                calibrated.loc[calibrated["channel"] == "temperature_c"]
                if not calibrated.empty
                else calibrated
            )
            mae_7d = (
                window_metrics(temp_cal, hours=168)["mae"] if not temp_cal.empty else float("nan")
            )
            best_fc = "—"
            if not scores.empty:
                temp_scores = scores.loc[scores["channel"] == "temperature_c"]
                if not temp_scores.empty:
                    best_fc = f"{float(temp_scores['mae'].min()):.1f}"
            good_pct = float("nan")
            if not calibrated.empty:
                flags = calibrated["quality_flag"].value_counts()
                total = int(flags.sum())
                good_pct = 100.0 * int(flags.get("good", 0)) / total if total else float("nan")

            st.markdown(
                theme.grid(
                    [
                        theme.tile("Station readings", f"{len(readings_all):,}", "", "stored samples"),
                        theme.tile("Area weather readings", f"{len(reference):,}", "", "used as a check"),
                        theme.tile(
                            "Average temperature gap",
                            _num(mae_7d, 1),
                            "°C",
                            "station compared with area weather",
                        ),
                        theme.tile("Usable readings", _num(good_pct, 0), "%", "passed basic checks"),
                        theme.tile(
                            "Sensor setup",
                            "Ready" if not coeffs.empty else "Not ready",
                            "",
                            f"{len(coeffs)} measurements checked" if not coeffs.empty else "needs checking",
                        ),
                    ]
                ),
                unsafe_allow_html=True,
            )

    if page == "Data":
        st.markdown(
            theme.section(
                "How the weather changed",
                "Choose one item, such as rain or temperature, to see earlier readings.",
            ),
            unsafe_allow_html=True,
        )
        channel = st.selectbox(
            "Show",
            list(CHANNEL_NAMES),
            format_func=lambda c: cfg["channels"].get(c, {}).get("label", c),
            key="trend_channel",
        )
        unit = _unit(cfg, channel)
        channel_cal = (
            calibrated_w.loc[calibrated_w["channel"] == channel]
            if not calibrated_w.empty
            else calibrated_w
        )
        if not channel_cal.empty:
            stats = window_metrics(channel_cal, hours=hours or 24 * 365)
            tiles = [
                theme.tile("Average gap", _num(stats["mae"], 1), unit, "station vs area weather"),
                theme.tile(
                    "Usual direction",
                    _num(stats["bias"], 1),
                    unit,
                    "plus means station is higher",
                ),
            ]
            if not coeffs.empty and channel in set(coeffs["channel"]):
                row = coeffs.loc[coeffs["channel"] == channel].iloc[0]
                tiles.append(
                    theme.tile(
                        "Gap before and after setup",
                        f"{float(row['mae_raw']):.1f} → {float(row['mae']):.1f}",
                        unit,
                        "smaller is better",
                        theme.ACCENT,
                    )
                )
            st.markdown(theme.grid(tiles), unsafe_allow_html=True)

        st.plotly_chart(
            charts.series_chart(
                channel, unit, _label(cfg, channel),
                readings_w, reference_w, calibrated_w, tz, show_raw, show_cal, show_ref,
            ),
            width="stretch",
            config={"displaylogo": False},
            key=f"chart_trend_{channel}",
        )

        st.markdown(
            theme.section(
                "Typical day pattern",
                "Shows the usual change from morning to night",
            ),
            unsafe_allow_html=True,
        )
        heat = charts.diurnal_heatmap(calibrated_w, channel, unit, tz)
        if heat is None:
            st.caption("Need at least two days of adjusted station data.")
        else:
            st.plotly_chart(
                heat,
                width="stretch",
                config={"displaylogo": False},
                key=f"chart_heat_{channel}",
            )

    if page == "Analytics":
        st.markdown(
            theme.section(
                "Are the sensors set up correctly?",
                "A smaller gap after setup means the sensor reading improved. "
                "A larger percentage means a bigger improvement.",
            ),
            unsafe_allow_html=True,
        )
        if coeffs.empty:
            st.info("The sensors have not been checked yet.")
        else:
            improvements = [
                theme.tile(
                    cfg["channels"][row["channel"]].get("label", row["channel"]),
                    f"{100.0 * (1 - float(row['mae']) / float(row['mae_raw'])):.0f}%",
                    "",
                    f"Error fell from {float(row['mae_raw']):.1f} to {float(row['mae']):.1f} "
                    f"{cfg['channels'][row['channel']]['unit']}",
                    theme.CHANNEL_ACCENT.get(row["channel"], theme.ACCENT),
                )
                for _, row in coeffs.iterrows()
                if float(row["mae_raw"]) > 0
            ]
            st.markdown(theme.grid(improvements), unsafe_allow_html=True)

            with st.expander("Show setup details"):
                table = pd.DataFrame(
                    {
                        "Reading": coeffs["channel"].map(
                            lambda c: cfg["channels"].get(c, {}).get("label", c)
                        ),
                        "Readings used": coeffs["n_points"],
                        "Gap before": coeffs["mae_raw"].map("{:.2f}".format),
                        "Gap after": coeffs["mae"].map("{:.2f}".format),
                        "Gap on new data": coeffs["mae_holdout"].map(
                            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                        ),
                    }
                )
                st.dataframe(table, width="stretch", hide_index=True)
                st.caption("Smaller gaps are better.")

        if not calibrated.empty:
            flags = calibrated["quality_flag"].value_counts()
            total = int(flags.sum())
            st.markdown(
                theme.section("Stored readings", "How many readings look usable"),
                unsafe_allow_html=True,
            )
            st.markdown(
                theme.grid(
                    [
                        theme.tile(
                            name,
                            f"{int(flags.get(flag, 0)):,}",
                            "",
                            f"{100.0 * int(flags.get(flag, 0)) / total if total else 0:.0f}% of readings",
                            color,
                        )
                        for flag, name, color in (
                            ("good", "Good", theme.ACCENT),
                            ("suspect", "May be wrong", theme.WARN),
                            ("missing", "Missing", theme.DANGER),
                        )
                    ]
                ),
                unsafe_allow_html=True,
            )

    if page == "Analytics":
        st.markdown(
            theme.section(
                "Did our local prediction work?",
                "We compare an old prediction with what happened later. "
                "This does not change the rain forecast on Overview.",
            ),
            unsafe_allow_html=True,
        )
        if scores.empty or predictions.empty:
            st.info("No test result is available yet.")
        else:
            fc_channel = st.selectbox(
                "Show results for",
                sorted(scores["channel"].unique().tolist()),
                format_func=lambda c: cfg["channels"].get(c, {}).get("label", c),
                key="forecast_channel",
            )
            unit = _unit(cfg, fc_channel)
            subset = scores.loc[scores["channel"] == fc_channel].copy()
            if (
                fc_channel == "rainfall_mm"
                and "csi" in subset.columns
                and subset["csi"].notna().any()
            ):
                timing = subset.sort_values(["csi", "mae"], ascending=[False, True]).iloc[0]
                if "daily_mae" in subset.columns and subset["daily_mae"].notna().any():
                    amount = subset.sort_values(["daily_mae", "mae"], ascending=[True, True]).iloc[0]
                else:
                    amount = timing
                score_tiles = [
                    theme.tile(
                        "Best rain timing",
                        _friendly_model_name(str(timing["model"])),
                        "",
                        (
                            f"CSI {_num(float(timing['csi']), 2)}"
                            + (
                                f" · 6h {_num(float(timing['csi_6h']), 2)}"
                                if "csi_6h" in timing.index and pd.notna(timing.get("csi_6h"))
                                else ""
                            )
                            + (
                                f" · 24h {_num(float(timing['csi_24h']), 2)}"
                                if "csi_24h" in timing.index and pd.notna(timing.get("csi_24h"))
                                else ""
                            )
                        ),
                        theme.ACCENT,
                    ),
                    theme.tile(
                        "Best daily rain amount",
                        _friendly_model_name(str(amount["model"])),
                        "",
                        (
                            f"day gap {_num(float(amount['daily_mae']), 1)} mm"
                            if pd.notna(amount.get("daily_mae"))
                            else "daily score not ready"
                        ),
                    ),
                    theme.tile(
                        "Hours caught",
                        (
                            f"{100.0 * float(timing['hit_rate']):.0f}%"
                            if "hit_rate" in timing.index and pd.notna(timing.get("hit_rate"))
                            else "—"
                        ),
                        "",
                        "share of real wet hours predicted",
                    ),
                ]
                best = amount
            else:
                subset = subset.sort_values("mae")
                best = subset.iloc[0]
                naive = subset.loc[subset["model"].str.startswith("persistence")]
                gain = "—"
                if not naive.empty and float(naive["mae"].min()) > 0:
                    gain = f"{100.0 * (1 - float(best['mae']) / float(naive['mae'].min())):.0f}%"
                score_tiles = [
                    theme.tile(
                        "Best result came from",
                        _friendly_model_name(str(best["model"])),
                        "",
                        "tested on past weather",
                        theme.ACCENT,
                    ),
                    theme.tile(
                        "Average gap",
                        _num(float(best["mae"]), 1),
                        unit,
                        "smaller is better",
                    ),
                    theme.tile(
                        "Better than a simple guess",
                        gain,
                        "",
                        "higher is better",
                    ),
                ]

            st.markdown(theme.grid(score_tiles), unsafe_allow_html=True)

            st.markdown(
                theme.section(
                    "Prediction compared with real weather",
                    "Lines that stay close together mean a better result.",
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                charts.forecast_chart(predictions, fc_channel, unit, _label(cfg, fc_channel), tz),
                width="stretch",
                config={"displaylogo": False},
                key=f"chart_forecast_{fc_channel}",
            )

            with st.expander("Show more test details"):
                detail = {
                    "Method": subset["model"].map(_friendly_model_name),
                    "Hours predicted": subset["forecast_hours"],
                    "Average gap": subset["mae"].map("{:.2f}".format),
                }
                if fc_channel == "rainfall_mm" and "csi" in subset.columns:
                    detail["Wet-hour score"] = subset["csi"].map(
                        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                    )
                    if "hit_rate" in subset.columns:
                        detail["Hours caught"] = subset["hit_rate"].map(
                            lambda v: f"{100.0 * v:.0f}%" if pd.notna(v) else "—"
                        )
                    if "daily_mae" in subset.columns:
                        detail["Daily rain gap"] = subset["daily_mae"].map(
                            lambda v: f"{v:.1f} mm" if pd.notna(v) else "—"
                        )
                    if "csi_6h" in subset.columns:
                        detail["CSI 6h"] = subset["csi_6h"].map(
                            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                        )
                    if "csi_24h" in subset.columns:
                        detail["CSI 24h"] = subset["csi_24h"].map(
                            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                        )
                    if "block_24h_mae" in subset.columns:
                        detail["24h rain gap"] = subset["block_24h_mae"].map(
                            lambda v: f"{v:.1f} mm" if pd.notna(v) else "—"
                        )
                table = pd.DataFrame(detail)
                st.dataframe(table, width="stretch", hide_index=True)

                if not walkforward.empty:
                    wf_table = walkforward.copy()
                    wf_table["Reading"] = wf_table["channel"].map(
                        lambda value: CHANNEL_LABELS.get(str(value), str(value))
                    )
                    wf_table["Method"] = wf_table["model"].map(_friendly_model_name)
                    wf_table["Hours ahead"] = wf_table["lead_hours"]
                    wf_table["Average gap"] = wf_table["mae"].map("{:.2f}".format)
                    st.write("Repeated tests at different times:")
                    st.dataframe(
                        wf_table[
                            ["Reading", "Method", "Hours ahead", "Average gap"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    if page == "Progress":
        _render_progress_page(forecast_verification, tz, project, cfg=cfg)

    if page == "Settings":
        _render_settings_page(cfg, projects)

    if page == "Sources":
        _data_sources_panel(nwp_members, reference, power, imd)
        st.markdown(
            theme.section(
                "Do the weather services give similar answers?",
                "We compare free weather services for this project pin. "
                "They will not match exactly because each service looks at a wide area.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            theme.section(
                "How close are they?",
                "A smaller gap means the services agree more.",
            ),
            unsafe_allow_html=True,
        )
        if source_validation.empty:
            st.info(
                "No result is ready yet."
            )
        else:
            val_table = source_validation.copy()
            units = {
                "temperature_c": "°C",
                "humidity_pct": "%",
                "pressure_hpa": "hPa",
                "rainfall_mm": "mm",
                "wind_speed_ms": "m/s",
                "wind_direction_deg": "°",
            }
            val_table["Measurement"] = val_table["channel"].map(
                lambda value: CHANNEL_LABELS.get(str(value), str(value))
            )
            val_table["Compared sources"] = val_table.apply(
                lambda row: (
                    f"{SOURCE_DETAILS.get(str(row['source_a']), (str(row['source_a']), '', ''))[0]}"
                    " vs "
                    f"{SOURCE_DETAILS.get(str(row['source_b']), (str(row['source_b']), '', ''))[0]}"
                ),
                axis=1,
            )
            val_table["Average difference"] = val_table.apply(
                lambda row: f"{float(row['mae']):.1f} {units.get(str(row['channel']), '')}",
                axis=1,
            )
            val_table["How close?"] = val_table.apply(
                lambda row: _comparison_meaning(float(row["mae"]), str(row["channel"]))[0],
                axis=1,
            )
            st.dataframe(
                val_table[
                    [
                        "Measurement",
                        "Compared sources",
                        "How close?",
                        "Average difference",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "This only compares the services. It does not prove the exact weather at the dam."
            )
        if not forecast_verification.empty:
            st.markdown(
                theme.section(
                    "Were earlier forecasts close?",
                    "How the Overview rain outlook did once truth arrived.",
                ),
                unsafe_allow_html=True,
            )
            fv = forecast_verification.copy()
            fv["Measurement"] = fv["channel"].map(
                lambda value: CHANNEL_LABELS.get(str(value), str(value))
            )
            fv["Date"] = pd.to_datetime(fv["issued_at"], utc=True).dt.tz_convert(
                tz
            ).dt.strftime("%d %b %Y · %H:%M")
            fv["Source checked"] = fv["model"].map(
                {
                    "ensemble_operational": "Overview vs Open-Meteo",
                    "open_meteo_operational": "Open-Meteo only",
                    "ensemble_vs_imd": "Overview vs IMD daily",
                    "open_meteo_vs_imd": "Open-Meteo vs IMD daily",
                }
            ).fillna(fv["model"])
            units = {
                "temperature_c": "°C",
                "humidity_pct": "%",
                "rainfall_mm": "mm",
            }
            fv["Average difference"] = fv.apply(
                lambda row: (
                    f"{float(row['mae']):.1f} {units.get(str(row['channel']), '')}"
                ),
                axis=1,
            )

            rain_fv = fv[fv["channel"] == "rainfall_mm"].copy()
            if not rain_fv.empty and "csi" in rain_fv.columns and rain_fv["csi"].notna().any():
                def _fmt_opt(value, digits=2):
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        return "—"
                    return f"{float(value):.{digits}f}"

                rain_fv["Daily error (mm)"] = rain_fv["daily_mae"].map(lambda v: _fmt_opt(v, 1))
                rain_fv["CSI"] = rain_fv["csi"].map(lambda v: _fmt_opt(v, 2))
                rain_fv["CSI 6h"] = (
                    rain_fv["csi_6h"].map(lambda v: _fmt_opt(v, 2))
                    if "csi_6h" in rain_fv.columns
                    else "—"
                )
                rain_fv["CSI 24h"] = (
                    rain_fv["csi_24h"].map(lambda v: _fmt_opt(v, 2))
                    if "csi_24h" in rain_fv.columns
                    else "—"
                )
                st.caption(
                    "Rain checked two ways: Open-Meteo (hourly timing) and IMD (daily totals). "
                    "Neither is a dam rain gauge."
                )
                st.dataframe(
                    rain_fv[
                        [
                            "Date",
                            "Source checked",
                            "Daily error (mm)",
                            "CSI",
                            "CSI 6h",
                            "CSI 24h",
                            "Average difference",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.dataframe(
                    fv[["Date", "Measurement", "Source checked", "Average difference"]],
                    width="stretch",
                    hide_index=True,
                )
                if rain_fv.empty:
                    st.caption(
                        "Rain verification appears after an archived outlook is about 48 hours old."
                    )


main()
