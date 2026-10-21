"""Interactive 7-day forecast board (Chart.js) for the Hub Forecast page."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.theme import CHANNEL_ACCENT, theme_tokens

MEMBER_LABELS = {
    "ecmwf": "ECMWF",
    "gfs": "NOAA GFS",
    "met_norway": "MET Norway",
}


def _num(series: pd.Series) -> list[float | None]:
    values = pd.to_numeric(series, errors="coerce")
    out: list[float | None] = []
    for value in values:
        if pd.isna(value):
            out.append(None)
        else:
            out.append(round(float(value), 2))
    return out


def _member_daily_rain(nwp_members: pd.DataFrame, tz: str) -> dict[str, Any]:
    """Daily rain totals per NWP member for the next ~7 days."""
    if nwp_members.empty or "member" not in nwp_members.columns:
        return {"labels": [], "datasets": []}
    frame = nwp_members.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    future = frame.loc[frame["timestamp"] >= now - pd.Timedelta(hours=1)].copy()
    if future.empty:
        future = frame.copy()
    local = future["timestamp"].dt.tz_convert(tz)
    future["day"] = local.dt.strftime("%a %d")
    future["day_ord"] = local.dt.floor("D")
    rain = (
        future.groupby(["member", "day_ord", "day"], as_index=False)["rainfall_mm"]
        .sum()
        .sort_values("day_ord")
    )
    labels = list(dict.fromkeys(rain["day"].tolist()))
    colors = {
        "ecmwf": CHANNEL_ACCENT["temperature_c"],
        "gfs": CHANNEL_ACCENT["humidity_pct"],
        "met_norway": CHANNEL_ACCENT["rainfall_mm"],
    }
    datasets = []
    for member, group in rain.groupby("member"):
        by_day = {row["day"]: float(row["rainfall_mm"]) for _, row in group.iterrows()}
        key = str(member)
        datasets.append(
            {
                "label": MEMBER_LABELS.get(key, key),
                "data": [round(by_day.get(day, 0.0), 2) for day in labels],
                "backgroundColor": colors.get(key, "#9F7AEA"),
                "borderRadius": 6,
                "maxBarThickness": 22,
            }
        )
    return {"labels": labels, "datasets": datasets}


def build_week_summary(
    *,
    daily: pd.DataFrame,
    nwp_members: pd.DataFrame,
    project_kind: str = "dam",
) -> dict[str, Any]:
    """Plain-language week summary so charts are easier to read."""
    day = daily.head(7).copy()
    if day.empty:
        return {
            "headline": "No forecast yet",
            "level": "calm",
            "bullets": ["Run analysis to load the next 7 days."],
            "action": "Check back after the next analysis run.",
        }

    day["date"] = pd.to_datetime(day["date"])
    rain = pd.to_numeric(day.get("rainfall_sum_mm"), errors="coerce").fillna(0.0)
    wet = pd.to_numeric(day.get("wet_hours"), errors="coerce").fillna(0.0)
    tmax = pd.to_numeric(day.get("temperature_max_c"), errors="coerce")
    tmin = pd.to_numeric(day.get("temperature_min_c"), errors="coerce")
    wind = pd.to_numeric(day.get("wind_speed_max_ms"), errors="coerce").fillna(0.0)

    rain_total = float(rain.sum())
    wet_total = int(wet.sum())
    rainy_days = int((rain >= 1.0).sum())
    wettest_i = int(np.argmax(rain.to_numpy()))
    wettest_day = pd.Timestamp(day.iloc[wettest_i]["date"]).strftime("%A")
    wettest_mm = float(rain.iloc[wettest_i])
    warmest = float(tmax.max()) if tmax.notna().any() else None
    coolest = float(tmin.min()) if tmin.notna().any() else None
    wind_peak = float(wind.max()) if len(wind) else 0.0

    if rain_total >= 80 or rainy_days >= 5:
        level, headline = "wet", "A wet week ahead"
    elif rain_total >= 25 or rainy_days >= 3:
        level, headline = "mixed", "Mixed week - some rain expected"
    elif rain_total >= 5 or rainy_days >= 1:
        level, headline = "light", "Mostly settled with light rain possible"
    else:
        level, headline = "calm", "A mostly dry week ahead"

    bullets: list[str] = []
    if rain_total < 2:
        bullets.append("Total rain looks very low for the next 7 days.")
    else:
        bullets.append(
            f"About {rain_total:.0f} mm total rain across the week "
            f"({rainy_days} day{'s' if rainy_days != 1 else ''} with ≥1 mm)."
        )
    if wettest_mm >= 1:
        bullets.append(f"Wettest day looks like {wettest_day} (~{wettest_mm:.1f} mm).")
    else:
        bullets.append("No single day stands out as particularly wet.")
    if warmest is not None and coolest is not None:
        bullets.append(f"Temperatures roughly {coolest:.0f}-{warmest:.0f} C.")
    if wind_peak >= 8:
        bullets.append(f"Wind may pick up - peak around {wind_peak:.1f} m/s.")
    elif wet_total > 0:
        bullets.append(f"Rain may fall during about {wet_total} hours in total.")

    # Model agreement from daily member rain totals
    if not nwp_members.empty and "member" in nwp_members.columns:
        frame = nwp_members.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        future = frame.loc[frame["timestamp"] >= now - pd.Timedelta(hours=1)]
        if future.empty:
            future = frame
        totals = (
            future.groupby("member")["rainfall_mm"]
            .sum()
            .astype(float)
            .sort_values(ascending=False)
        )
        if len(totals) >= 2:
            spread = float(totals.max() - totals.min())
            if spread <= 5:
                bullets.append("Weather services largely agree on the week's rain.")
            elif spread <= 20:
                bullets.append(
                    "Services differ a bit on rain amounts - treat totals as a guide."
                )
            else:
                bullets.append(
                    "Services disagree on rain totals - confidence is lower this week."
                )

    if project_kind == "plantation":
        if level in {"wet", "mixed"}:
            action = "Plan irrigation carefully and watch for long wet spells on the canopy."
        elif level == "light":
            action = "Light rain only - check soil moisture before watering."
        else:
            action = "Dry stretch likely - check whether the plants need water."
    else:
        if level == "wet":
            action = "Watch reservoir level and inflow more closely on the wetter days."
        elif level == "mixed":
            action = "Normal monitoring, with extra attention around the wettest day."
        else:
            action = "Routine monitoring should be enough for rainfall this week."

    # Normalize remaining dashes used in earlier bullets
    bullets = [b.replace("\u2014", "-").replace("\u2013", "-") for b in bullets]

    return {
        "headline": headline,
        "level": level,
        "bullets": bullets[:5],
        "action": action,
    }


def build_forecast_payload(
    *,
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    nwp_members: pd.DataFrame,
    tz: str,
    theme_id: str | None = None,
    project_kind: str = "dam",
) -> dict[str, Any]:
    tokens = theme_tokens(theme_id)
    now = pd.Timestamp.now(tz="UTC")
    day = daily.head(7).copy()
    day["date"] = pd.to_datetime(day["date"])
    labels = [pd.Timestamp(d).strftime("%a %d") for d in day["date"]]
    full_labels = [pd.Timestamp(d).strftime("%A %d %b") for d in day["date"]]

    hours = hourly.copy()
    hours["timestamp"] = pd.to_datetime(hours["timestamp"], utc=True)
    future = hours.loc[hours["timestamp"] >= now].copy()
    if future.empty:
        future = hours.copy()
    future = future.iloc[: 24 * 7].copy()
    local_ts = future["timestamp"].dt.tz_convert(tz)
    hour_labels = [ts.strftime("%a %H:%M") for ts in local_ts]

    temp_spread = None
    if "temperature_c_spread" in future.columns:
        temp_spread = _num(future["temperature_c_spread"])

    rain_total = float(pd.to_numeric(day.get("rainfall_sum_mm"), errors="coerce").fillna(0).sum())
    wet_total = int(pd.to_numeric(day.get("wet_hours"), errors="coerce").fillna(0).sum())
    tmax = float(pd.to_numeric(day.get("temperature_max_c"), errors="coerce").max())
    tmin = float(pd.to_numeric(day.get("temperature_min_c"), errors="coerce").min())
    wind_peak = float(pd.to_numeric(day.get("wind_speed_max_ms"), errors="coerce").max())

    rain_vals = pd.to_numeric(day.get("rainfall_sum_mm"), errors="coerce").fillna(0.0)
    wettest = (
        full_labels[int(np.argmax(rain_vals.to_numpy()))] if len(full_labels) else "—"
    )

    summary = build_week_summary(
        daily=day, nwp_members=nwp_members, project_kind=project_kind
    )

    return {
        "theme": {
            "bg": tokens["bg"],
            "panel": tokens["panel"],
            "chip": tokens["chip"],
            "accent": tokens["accent"],
            "accent2": tokens["accent_2"],
            "blue": tokens["blue"],
            "cyan": tokens["cyan"],
            "green": tokens["green"],
            "warn": tokens["warn"],
            "text": "#FFFFFF",
            "muted": "#A0AEC0",
            "faint": "#718096",
            "border": tokens["border"],
            "rain": CHANNEL_ACCENT["rainfall_mm"],
            "temp": CHANNEL_ACCENT["temperature_c"],
            "wind": CHANNEL_ACCENT["wind_speed_ms"],
            "humidity": CHANNEL_ACCENT["humidity_pct"],
        },
        "summary": summary,
        "kpis": {
            "rain_total": round(rain_total, 1),
            "wet_hours": wet_total,
            "temp_high": round(tmax, 1) if np.isfinite(tmax) else None,
            "temp_low": round(tmin, 1) if np.isfinite(tmin) else None,
            "wind_peak": round(wind_peak, 1) if np.isfinite(wind_peak) else None,
            "wettest": wettest,
        },
        "daily": {
            "labels": labels,
            "fullLabels": full_labels,
            "rain": _num(day.get("rainfall_sum_mm", pd.Series(dtype=float))),
            "wetHours": _num(day.get("wet_hours", pd.Series(dtype=float))),
            "tempMax": _num(day.get("temperature_max_c", pd.Series(dtype=float))),
            "tempMin": _num(day.get("temperature_min_c", pd.Series(dtype=float))),
            "windMax": _num(day.get("wind_speed_max_ms", pd.Series(dtype=float))),
            "rainChance": _num(day.get("rain_probability_max_pct", pd.Series(dtype=float))),
        },
        "hourly": {
            "labels": hour_labels,
            "rain": _num(future.get("rainfall_mm", pd.Series(dtype=float))),
            "temp": _num(future.get("temperature_c", pd.Series(dtype=float))),
            "humidity": _num(future.get("humidity_pct", pd.Series(dtype=float))),
            "wind": _num(future.get("wind_speed_ms", pd.Series(dtype=float))),
            "tempSpread": temp_spread,
        },
        "members": _member_daily_rain(nwp_members, tz),
    }


def render_forecast_board_html(payload: dict[str, Any]) -> str:
    """Self-contained HTML board using Chart.js from CDN."""
    data = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: {payload["theme"]["bg"]};
      --panel: {payload["theme"]["panel"]};
      --chip: {payload["theme"]["chip"]};
      --accent: {payload["theme"]["accent"]};
      --accent2: {payload["theme"]["accent2"]};
      --cyan: {payload["theme"]["cyan"]};
      --blue: {payload["theme"]["blue"]};
      --green: {payload["theme"]["green"]};
      --warn: {payload["theme"]["warn"]};
      --text: #fff;
      --muted: #A0AEC0;
      --faint: #718096;
      --border: {payload["theme"]["border"]};
      --rain: {payload["theme"]["rain"]};
      --temp: {payload["theme"]["temp"]};
      --wind: {payload["theme"]["wind"]};
      --font: "Plus Jakarta Sans", Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 4px 2px 12px;
      background: transparent; color: var(--text);
      font-family: var(--font);
    }}
    .summary {{
      display: grid; grid-template-columns: 1.4fr 1fr; gap: 14px;
      padding: 16px 18px; border-radius: 18px; margin-bottom: 12px;
      background:
        radial-gradient(320px 160px at 92% 12%, color-mix(in srgb, var(--accent) 28%, transparent), transparent 60%),
        color-mix(in srgb, var(--panel) 94%, transparent);
      border: 1px solid var(--border);
    }}
    .summary .eyebrow {{
      color: var(--muted); font-size: .68rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase; margin: 0 0 6px;
    }}
    .summary h2 {{
      margin: 0 0 8px; font-size: 1.28rem; font-weight: 700;
      letter-spacing: -.02em; color: #fff; line-height: 1.25;
    }}
    .summary .tag {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 10px; border-radius: 999px; font-size: .7rem; font-weight: 700;
      border: 1px solid var(--border); margin-bottom: 10px;
      background: color-mix(in srgb, var(--chip) 80%, transparent);
    }}
    .summary .tag.calm {{ color: var(--green); }}
    .summary .tag.light {{ color: var(--cyan); }}
    .summary .tag.mixed {{ color: var(--warn); }}
    .summary .tag.wet {{ color: #FF8A8A; }}
    .summary ul {{
      margin: 0; padding: 0 0 0 16px; color: var(--muted);
      font-size: .8rem; line-height: 1.5;
    }}
    .summary ul li {{ margin: 0 0 4px; }}
    .summary .action {{
      padding: 14px 14px 12px; border-radius: 14px;
      background: color-mix(in srgb, var(--chip) 85%, transparent);
      border: 1px solid var(--border); align-self: stretch;
    }}
    .summary .action .k {{
      color: var(--muted); font-size: .68rem; font-weight: 700;
      letter-spacing: .06em; text-transform: uppercase; margin: 0 0 8px;
    }}
    .summary .action p {{
      margin: 0; color: #fff; font-size: .88rem; font-weight: 600; line-height: 1.45;
    }}
    @media (max-width: 900px) {{
      .summary {{ grid-template-columns: 1fr; }}
    }}
    .kpis {{
      display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px; margin-bottom: 12px;
    }}
    .kpi {{
      padding: 14px 14px 12px; border-radius: 16px;
      background: color-mix(in srgb, var(--chip) 88%, transparent);
      border: 1px solid var(--border);
      min-height: 78px;
    }}
    .kpi .k {{ color: var(--muted); font-size: .72rem; font-weight: 600; margin: 0 0 6px; }}
    .kpi .v {{ color: #fff; font-size: 1.2rem; font-weight: 700; margin: 0; letter-spacing: -.02em; }}
    .kpi .v .u {{ color: var(--muted); font-size: .75rem; font-weight: 500; margin-left: 3px; }}
    .kpi .s {{ color: var(--faint); font-size: .68rem; margin-top: 4px; line-height: 1.3; }}
    .grid {{
      display: grid; grid-template-columns: 1.15fr 1fr; gap: 12px; margin-bottom: 12px;
    }}
    .grid.wide {{ grid-template-columns: 1fr; }}
    .card {{
      padding: 14px 16px 10px; border-radius: 18px;
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      border: 1px solid var(--border);
      min-height: 280px;
    }}
    .card h3 {{
      margin: 0 0 2px; font-size: .92rem; font-weight: 700; color: #fff;
    }}
    .card p.note {{
      margin: 0 0 10px; color: var(--muted); font-size: .72rem; line-height: 1.35;
    }}
    .chart-wrap {{ position: relative; height: 220px; }}
    .chart-wrap.tall {{ height: 260px; }}
    .empty {{
      color: var(--muted); font-size: .8rem; padding: 24px 8px; text-align: center;
    }}
    @media (max-width: 900px) {{
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="summary" id="summary"></div>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="card">
      <h3>Daily rain &amp; wet hours</h3>
      <p class="note">Bars = expected rain (mm). Line = hours with measurable rain.</p>
      <div class="chart-wrap tall"><canvas id="rainDaily"></canvas></div>
    </div>
    <div class="card">
      <h3>Temperature range</h3>
      <p class="note">Daily high and low band for the next week.</p>
      <div class="chart-wrap tall"><canvas id="tempDaily"></canvas></div>
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Hourly rain timeline</h3>
      <p class="note">Next seven days, hour by hour.</p>
      <div class="chart-wrap"><canvas id="rainHourly"></canvas></div>
    </div>
    <div class="card">
      <h3>Wind &amp; humidity</h3>
      <p class="note">Hourly wind speed and relative humidity.</p>
      <div class="chart-wrap"><canvas id="windHum"></canvas></div>
    </div>
  </div>
  <div class="grid wide" id="memberCard" style="display:none">
    <div class="card">
      <h3>Model rain agreement</h3>
      <p class="note">Daily rain totals from each weather service — closer bars mean stronger agreement.</p>
      <div class="chart-wrap tall"><canvas id="memberRain"></canvas></div>
    </div>
  </div>
  <script>
    const DATA = {data};
    const T = DATA.theme;
    Chart.defaults.color = T.muted;
    Chart.defaults.borderColor = "rgba(226,232,240,0.12)";
    Chart.defaults.font.family = '"Plus Jakarta Sans", Helvetica, Arial, sans-serif';

    const S = DATA.summary || {{}};
    const level = S.level || "calm";
    const levelLabel = ({{
      calm: "Mostly dry",
      light: "Light rain",
      mixed: "Mixed week",
      wet: "Wet week",
    }})[level] || "Week outlook";
    const bullets = (S.bullets || []).map((b) => `<li>${{b}}</li>`).join("");
    document.getElementById("summary").innerHTML = `
      <div>
        <p class="eyebrow">Week in plain words</p>
        <span class="tag ${{level}}">${{levelLabel}}</span>
        <h2>${{S.headline || "Week ahead"}}</h2>
        <ul>${{bullets}}</ul>
      </div>
      <div class="action">
        <p class="k">What this means</p>
        <p>${{S.action || "Use the charts below for day-by-day detail."}}</p>
      </div>
    `;

    function kpi(label, value, unit, sub) {{
      const u = unit ? `<span class="u">${{unit}}</span>` : "";
      return `<div class="kpi"><p class="k">${{label}}</p><p class="v">${{value}}${{u}}</p><p class="s">${{sub || ""}}</p></div>`;
    }}
    const K = DATA.kpis;
    document.getElementById("kpis").innerHTML = [
      kpi("7-day rain", K.rain_total ?? "—", "mm", "Total expected"),
      kpi("Wet hours", K.wet_hours ?? "—", "", "Hours with rain ≥ 0.1 mm"),
      kpi("Warmest", K.temp_high ?? "—", "°C", "Highest daily max"),
      kpi("Coolest", K.temp_low ?? "—", "°C", "Lowest daily min"),
      kpi("Wettest day", K.wettest || "—", "", `Peak wind ${{K.wind_peak ?? "—"}} m/s`),
    ].join("");

    const tip = {{
      backgroundColor: "rgba(11,20,55,0.95)",
      titleColor: "#fff",
      bodyColor: "#A0AEC0",
      borderColor: "rgba(226,232,240,0.18)",
      borderWidth: 1,
      padding: 10,
      cornerRadius: 10,
    }};

    new Chart(document.getElementById("rainDaily"), {{
      type: "bar",
      data: {{
        labels: DATA.daily.labels,
        datasets: [
          {{
            type: "bar",
            label: "Rain (mm)",
            data: DATA.daily.rain,
            backgroundColor: T.rain,
            borderRadius: 8,
            yAxisID: "y",
            order: 2,
          }},
          {{
            type: "line",
            label: "Wet hours",
            data: DATA.daily.wetHours,
            borderColor: T.accent2,
            backgroundColor: "transparent",
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: T.accent2,
            yAxisID: "y1",
            order: 1,
          }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{ mode: "index", intersect: false }},
        plugins: {{
          legend: {{ labels: {{ boxWidth: 10, usePointStyle: true }} }},
          tooltip: {{
            ...tip,
            callbacks: {{
              title: (items) => DATA.daily.fullLabels[items[0].dataIndex] || items[0].label,
            }},
          }},
        }},
        scales: {{
          x: {{ grid: {{ display: false }} }},
          y: {{
            title: {{ display: true, text: "mm", color: T.faint }},
            beginAtZero: true,
            grid: {{ color: "rgba(226,232,240,0.06)" }},
          }},
          y1: {{
            position: "right",
            min: 0, max: 24,
            title: {{ display: true, text: "hours", color: T.faint }},
            grid: {{ drawOnChartArea: false }},
          }},
        }},
      }},
    }});

    new Chart(document.getElementById("tempDaily"), {{
      type: "line",
      data: {{
        labels: DATA.daily.labels,
        datasets: [
          {{
            label: "High",
            data: DATA.daily.tempMax,
            borderColor: T.temp,
            backgroundColor: "rgba(159,122,234,0.18)",
            fill: "+1",
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: T.temp,
          }},
          {{
            label: "Low",
            data: DATA.daily.tempMin,
            borderColor: T.blue,
            backgroundColor: "transparent",
            fill: false,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: T.blue,
          }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{ mode: "index", intersect: false }},
        plugins: {{
          legend: {{ labels: {{ boxWidth: 10, usePointStyle: true }} }},
          tooltip: tip,
        }},
        scales: {{
          x: {{ grid: {{ display: false }} }},
          y: {{
            title: {{ display: true, text: "°C", color: T.faint }},
            grid: {{ color: "rgba(226,232,240,0.06)" }},
          }},
        }},
      }},
    }});

    new Chart(document.getElementById("rainHourly"), {{
      type: "bar",
      data: {{
        labels: DATA.hourly.labels,
        datasets: [{{
          label: "Rain (mm)",
          data: DATA.hourly.rain,
          backgroundColor: (ctx) => {{
            const v = ctx.raw || 0;
            if (v >= 2) return T.warn;
            if (v >= 0.5) return T.rain;
            return "rgba(1,176,244,0.35)";
          }},
          borderRadius: 2,
          maxBarThickness: 6,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: tip,
        }},
        scales: {{
          x: {{
            ticks: {{
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 12,
              color: T.faint,
              font: {{ size: 10 }},
            }},
            grid: {{ display: false }},
          }},
          y: {{
            beginAtZero: true,
            title: {{ display: true, text: "mm", color: T.faint }},
            grid: {{ color: "rgba(226,232,240,0.06)" }},
          }},
        }},
      }},
    }});

    const windDatasets = [
      {{
        label: "Wind (m/s)",
        data: DATA.hourly.wind,
        borderColor: T.green,
        backgroundColor: "transparent",
        tension: 0.3,
        pointRadius: 0,
        yAxisID: "y",
      }},
      {{
        label: "Humidity (%)",
        data: DATA.hourly.humidity,
        borderColor: T.cyan,
        backgroundColor: "rgba(1,176,244,0.08)",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        yAxisID: "y1",
      }},
    ];
    if (DATA.hourly.tempSpread && DATA.hourly.tempSpread.some((v) => v != null)) {{
      windDatasets.push({{
        label: "Temp spread (°C)",
        data: DATA.hourly.tempSpread,
        borderColor: T.accent,
        borderDash: [4, 4],
        tension: 0.3,
        pointRadius: 0,
        yAxisID: "y",
      }});
    }}
    new Chart(document.getElementById("windHum"), {{
      type: "line",
      data: {{ labels: DATA.hourly.labels, datasets: windDatasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{ mode: "index", intersect: false }},
        plugins: {{
          legend: {{ labels: {{ boxWidth: 10, usePointStyle: true }} }},
          tooltip: tip,
        }},
        scales: {{
          x: {{
            ticks: {{ maxTicksLimit: 10, color: T.faint, font: {{ size: 10 }} }},
            grid: {{ display: false }},
          }},
          y: {{
            title: {{ display: true, text: "m/s", color: T.faint }},
            grid: {{ color: "rgba(226,232,240,0.06)" }},
          }},
          y1: {{
            position: "right",
            min: 0, max: 100,
            title: {{ display: true, text: "%", color: T.faint }},
            grid: {{ drawOnChartArea: false }},
          }},
        }},
      }},
    }});

    if (DATA.members && DATA.members.datasets && DATA.members.datasets.length) {{
      document.getElementById("memberCard").style.display = "grid";
      new Chart(document.getElementById("memberRain"), {{
        type: "bar",
        data: DATA.members,
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            legend: {{ labels: {{ boxWidth: 10, usePointStyle: true }} }},
            tooltip: tip,
          }},
          scales: {{
            x: {{ stacked: false, grid: {{ display: false }} }},
            y: {{
              beginAtZero: true,
              title: {{ display: true, text: "mm", color: T.faint }},
              grid: {{ color: "rgba(226,232,240,0.06)" }},
            }},
          }},
        }},
      }});
    }}
  </script>
</body>
</html>
"""
