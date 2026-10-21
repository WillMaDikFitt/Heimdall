"""Plotly figures used by the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import theme as theme
from app.theme import style_fig, to_local

SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def friendly_method(name: str) -> str:
    clean = name.replace("_fullhist", "")
    return {
        "persistence": "Same as last hour",
        "seasonal_naive": "Same time yesterday",
        "gradient_boosting": "Local prediction",
    }.get(clean, clean.replace("_", " "))


def series_chart(
    channel: str,
    unit: str,
    y_title: str,
    readings: pd.DataFrame,
    reference: pd.DataFrame,
    calibrated: pd.DataFrame,
    tz: str,
    show_raw: bool,
    show_cal: bool,
    show_ref: bool,
) -> go.Figure:
    fig = go.Figure()
    accent = theme.CHANNEL_ACCENT.get(channel, theme.ACCENT)

    if show_raw and not readings.empty and channel in readings.columns:
        fig.add_trace(
            go.Scatter(
                x=to_local(readings["timestamp"], tz),
                y=readings[channel],
                name="Original sensor",
                line=dict(width=1, color=theme.RAW),
                opacity=0.75,
                connectgaps=False,
                hovertemplate="Raw %{y:.2f} " + unit + "<extra></extra>",
            )
        )
    if show_ref and not reference.empty and channel in reference.columns:
        fig.add_trace(
            go.Scatter(
                x=to_local(reference["timestamp"], tz),
                y=reference[channel],
                name="Wider-area weather",
                line=dict(width=1.9, color=theme.REFERENCE, dash="dot"),
                hovertemplate="Wider-area weather: %{y:.2f} " + unit + "<extra></extra>",
            )
        )
    channel_cal = (
        calibrated.loc[calibrated["channel"] == channel] if not calibrated.empty else calibrated
    )
    if show_cal and not channel_cal.empty:
        ordered = channel_cal.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=to_local(ordered["timestamp"], tz),
                y=ordered["calibrated"],
                name="Checked station",
                line=dict(width=2.6, color=accent, shape="spline", smoothing=0.4),
                fill="tozeroy",
                fillcolor="rgba(67,24,255,0.12)",
                hovertemplate="Checked station: %{y:.2f} " + unit + "<extra></extra>",
            )
        )
    return style_fig(fig, y_title=y_title, height=400)


def operational_forecast_chart(hourly: pd.DataFrame, tz: str) -> go.Figure:
    """72-hour temperature outlook with rain probability and rainfall bars."""
    frame = hourly.sort_values("timestamp").copy()
    frame = frame.iloc[:72]
    local_time = to_local(frame["timestamp"], tz)
    now_local = pd.Timestamp.now(tz=tz)

    hover_x = local_time.dt.strftime("%a %d %b %Y<br>%H:%M IST")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=local_time,
            y=frame["rainfall_mm"],
            customdata=hover_x,
            name="Rain",
            marker=dict(color="rgba(1,176,244,.55)", line=dict(width=0)),
            yaxis="y2",
            hovertemplate="%{customdata}<br>Rain %{y:.1f} mm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=local_time,
            y=frame["rain_probability_pct"],
            customdata=hover_x,
            name="Chance of rain",
            line=dict(width=2, color=theme.REFERENCE, dash="dot"),
            yaxis="y3",
            hovertemplate="%{customdata}<br>Rain chance %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=local_time,
            y=frame["temperature_c"],
            customdata=hover_x,
            name="Temperature",
            line=dict(
                width=3,
                color=theme.CHANNEL_ACCENT["temperature_c"],
                shape="spline",
                smoothing=0.45,
            ),
            fill="tozeroy",
            fillcolor="rgba(67,24,255,.18)",
            hovertemplate="%{customdata}<br>Temperature %{y:.1f} °C<extra></extra>",
        )
    )
    if not frame.empty:
        start = local_time.iloc[0]
        end = local_time.iloc[-1]
        if start <= now_local <= end:
            fig.add_vline(
                x=now_local,
                line_width=1.4,
                line_dash="dash",
                line_color=theme.ACCENT,
                annotation_text=f"Now {now_local.strftime('%d %b %H:%M')}",
                annotation_position="top",
                annotation_font=dict(color=theme.ACCENT, size=11),
            )
    fig.update_layout(
        height=410,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.MUTED, size=12.5),
        barmode="overlay",
        hovermode="x unified",
        hoverlabel=dict(bgcolor=theme.PANEL, bordercolor=theme.BORDER, font=dict(color=theme.TEXT)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.MUTED),
        ),
        xaxis=dict(
            title=dict(text="Day and time", font=dict(color=theme.FAINT)),
            showgrid=True,
            gridcolor="rgba(255,255,255,.05)",
            zeroline=False,
            linecolor=theme.BORDER,
            tickfont=dict(color=theme.MUTED),
            tickformat="%d %b<br>%H:%M",
            hoverformat="%a %d %b %Y %H:%M IST",
        ),
        yaxis=dict(
            title=dict(text="Temperature (°C)", font=dict(color=theme.FAINT)),
            showgrid=True,
            gridcolor="rgba(255,255,255,.05)",
            zeroline=False,
            linecolor=theme.BORDER,
            tickfont=dict(color=theme.MUTED),
        ),
        yaxis2=dict(
            title=dict(text="Rain (mm)", font=dict(color=theme.FAINT)),
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
            tickfont=dict(color=theme.MUTED),
        ),
        yaxis3=dict(
            overlaying="y",
            side="right",
            position=0.94,
            range=[0, 100],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
    )
    return fig


def rain_outlook_chart(daily: pd.DataFrame) -> go.Figure:
    """Daily rain total, with hours of rain when that count is available."""
    dates = pd.to_datetime(daily["date"])
    labels = dates.dt.strftime("%a<br>%d %b")
    hover = dates.dt.strftime("%A, %d %b %Y")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=daily["rainfall_sum_mm"],
            customdata=hover,
            name="Rain",
            marker=dict(
                color="rgba(0,117,255,.72)",
                line=dict(width=0),
            ),
            hovertemplate="%{customdata}<br>%{y:.1f} mm<extra></extra>",
        )
    )
    has_hours = "wet_hours" in daily.columns
    if has_hours:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=daily["wet_hours"],
                customdata=hover,
                name="Hours with rain",
                mode="lines+markers",
                line=dict(width=2.2, color=theme.REFERENCE),
                marker=dict(size=7, color=theme.REFERENCE),
                yaxis="y2",
                hovertemplate="%{customdata}<br>%{y:.0f} hours with rain<extra></extra>",
            )
        )
    fig = style_fig(
        fig,
        y_title="Rain (mm)",
        x_title="Day",
        height=310,
    )
    if has_hours:
        fig.update_layout(
            yaxis2=dict(
                title=dict(text="Hours with rain", font=dict(color=theme.FAINT)),
                overlaying="y",
                side="right",
                range=[0, 24],
                showgrid=False,
                tickfont=dict(color=theme.MUTED),
            )
        )
    return fig


def temperature_range_chart(daily: pd.DataFrame) -> go.Figure:
    """Daily high/low temperature range."""
    dates = pd.to_datetime(daily["date"])
    labels = dates.dt.strftime("%a<br>%d %b")
    hover = dates.dt.strftime("%A, %d %b %Y")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=daily["temperature_min_c"],
            customdata=hover,
            name="Low",
            line=dict(width=2, color=theme.REFERENCE),
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate="%{customdata}<br>Low %{y:.1f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=daily["temperature_max_c"],
            customdata=hover,
            name="High",
            line=dict(width=2.4, color=theme.CHANNEL_ACCENT["temperature_c"]),
            mode="lines+markers",
            marker=dict(size=7),
            fill="tonexty",
            fillcolor="rgba(67,24,255,.16)",
            hovertemplate="%{customdata}<br>High %{y:.1f} °C<extra></extra>",
        )
    )
    return style_fig(
        fig,
        y_title="Temperature (°C)",
        x_title="Forecast date",
        height=310,
    )


def wind_rose(readings: pd.DataFrame) -> go.Figure | None:
    """Mean wind speed per 22.5 degree sector."""
    if readings.empty:
        return None
    needed = {"wind_direction_deg", "wind_speed_ms"}
    if not needed.issubset(readings.columns):
        return None

    frame = readings[["wind_direction_deg", "wind_speed_ms"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if frame.empty:
        return None

    sector_idx = ((frame["wind_direction_deg"] + 11.25) // 22.5).astype(int) % 16
    grouped = frame.groupby(sector_idx)["wind_speed_ms"].mean()
    values = [float(grouped.get(i, 0.0)) for i in range(16)]

    fig = go.Figure(
        go.Barpolar(
            r=values,
            theta=[i * 22.5 for i in range(16)],
            width=[20] * 16,
            marker=dict(
                color=values,
                colorscale=[[0, "rgba(96,165,250,.25)"], [1, theme.REFERENCE]],
                line=dict(color="rgba(226,232,240,.14)", width=1),
            ),
            hovertemplate="%{theta}°<br>Mean %{r:.2f} m/s<extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=30, r=30, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.MUTED, size=11.5),
        showlegend=False,
        hoverlabel=dict(bgcolor=theme.PANEL, bordercolor=theme.BORDER, font=dict(color=theme.TEXT)),
        polar=dict(
            bgcolor="rgba(255,255,255,.02)",
            radialaxis=dict(
                gridcolor="rgba(255,255,255,.07)",
                linecolor="rgba(255,255,255,.07)",
                tickfont=dict(color=theme.FAINT, size=10),
                angle=90,
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                gridcolor="rgba(255,255,255,.07)",
                linecolor="rgba(255,255,255,.07)",
                tickmode="array",
                tickvals=[i * 22.5 for i in range(16)],
                ticktext=SECTORS,
                tickfont=dict(color=theme.MUTED, size=10),
            ),
        ),
    )
    return fig


def diurnal_heatmap(calibrated: pd.DataFrame, channel: str, unit: str, tz: str) -> go.Figure | None:
    """Day (rows) by hour-of-day (columns) grid of calibrated values."""
    subset = calibrated.loc[calibrated["channel"] == channel] if not calibrated.empty else calibrated
    if subset.empty:
        return None

    frame = subset[["timestamp", "calibrated"]].copy()
    frame["local"] = to_local(frame["timestamp"], tz)
    frame["day"] = frame["local"].dt.strftime("%d %b")
    frame["hour"] = frame["local"].dt.hour
    pivot = frame.pivot_table(index="day", columns="hour", values="calibrated", aggfunc="mean")
    if pivot.empty or pivot.shape[0] < 2:
        return None

    order = (
        frame.groupby("day")["local"].min().sort_values().index.tolist()
    )
    pivot = pivot.reindex(order)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[f"{h:02d}" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Turbo",
            colorbar=dict(
                title=dict(text=unit, font=dict(color=theme.FAINT, size=11)),
                tickfont=dict(color=theme.MUTED, size=10),
                outlinewidth=0,
                thickness=12,
            ),
            hovertemplate="%{y} %{x}:00<br>%{z:.2f} " + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, 42 * pivot.shape[0]),
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.MUTED, size=11.5),
        hoverlabel=dict(bgcolor=theme.PANEL, bordercolor=theme.BORDER, font=dict(color=theme.TEXT)),
        xaxis=dict(
            title=dict(text="Hour of day (IST)", font=dict(color=theme.FAINT)),
            showgrid=False,
            tickfont=dict(color=theme.MUTED, size=10),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color=theme.MUTED, size=10), autorange="reversed"),
    )
    return fig


def forecast_chart(
    predictions: pd.DataFrame,
    channel: str,
    unit: str,
    y_title: str,
    tz: str,
) -> go.Figure:
    subset = predictions.loc[predictions["channel"] == channel]
    fig = go.Figure()
    if subset.empty:
        return style_fig(fig, y_title=y_title)

    actual = subset.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=to_local(actual["timestamp"], tz),
            y=actual["actual"],
            name="What happened",
            line=dict(width=3.2, color=theme.TEXT),
            hovertemplate="What happened: %{y:.2f} " + unit + "<extra></extra>",
        )
    )
    styles = {
        "persistence": (theme.RAW, "dot"),
        "seasonal_naive": (theme.REFERENCE, "dash"),
        "gradient_boosting": (theme.ACCENT, "solid"),
    }
    for model, group in subset.groupby("model"):
        base = str(model).replace("_fullhist", "")
        color, dash = styles.get(base, (theme.MUTED, "dot"))
        ordered = group.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=to_local(ordered["timestamp"], tz),
                y=ordered["predicted"],
                name=friendly_method(str(model)),
                line=dict(width=1.9, color=color, dash=dash),
                opacity=0.95 if base == "gradient_boosting" else 0.7,
                hovertemplate=f"{friendly_method(str(model))}: %{{y:.2f}} {unit}<extra></extra>",
            )
        )
    return style_fig(fig, y_title=y_title, height=420)


def error_by_lead_time(
    predictions: pd.DataFrame,
    channel: str,
    unit: str,
) -> go.Figure | None:
    """Absolute error against forecast lead time, per model."""
    subset = predictions.loc[predictions["channel"] == channel].copy()
    if subset.empty:
        return None

    styles = {
        "persistence": (theme.RAW, "dot"),
        "seasonal_naive": (theme.REFERENCE, "dash"),
        "gradient_boosting": (theme.ACCENT, "solid"),
    }
    fig = go.Figure()
    plotted = False
    for model, group in subset.groupby("model"):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        err = (ordered["predicted"] - ordered["actual"]).abs()
        if err.dropna().empty:
            continue
        base = str(model).replace("_fullhist", "")
        color, dash = styles.get(base, (theme.MUTED, "dot"))
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(ordered) + 1),
                y=err.rolling(3, min_periods=1).mean(),
                name=friendly_method(str(model)),
                line=dict(width=1.9, color=color, dash=dash),
                hovertemplate="Hours ahead: %{x}<br>Gap: %{y:.2f} " + unit + "<extra></extra>",
            )
        )
        plotted = True

    if not plotted:
        return None
    return style_fig(
        fig,
        y_title=f"Gap ({unit})",
        x_title="Hours ahead",
        height=330,
    )


def model_mae_bars(scores: pd.DataFrame, channel: str, unit: str) -> go.Figure | None:
    subset = scores.loc[scores["channel"] == channel].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("mae", ascending=True)
    colors = [
        theme.ACCENT if "gradient_boosting" in str(m) else theme.RAW for m in subset["model"]
    ]
    fig = go.Figure(
        go.Bar(
            x=subset["mae"],
            y=[friendly_method(str(m)) for m in subset["model"]],
            orientation="h",
            marker=dict(color=colors, line=dict(color=theme.BORDER, width=1)),
            text=[f"{v:.2f}" for v in subset["mae"]],
            textposition="outside",
            textfont=dict(color=theme.MUTED, size=11),
            hovertemplate="%{y}<br>Average gap %{x:.3f} " + unit + "<extra></extra>",
        )
    )
    fig = style_fig(
        fig,
        y_title="",
        x_title=f"Average gap ({unit}) — smaller is better",
        height=max(230, 34 * len(subset) + 90),
        show_legend=False,
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, float(subset["mae"].max()) * 1.25])
    return fig


def overview_trend_chart(hourly: pd.DataFrame, tz: str) -> go.Figure:
    """Vision-style dual series: rainfall and temperature over the next 72 hours."""
    frame = hourly.sort_values("timestamp").copy().iloc[:72]
    local_time = to_local(frame["timestamp"], tz)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=local_time,
            y=pd.to_numeric(frame["rainfall_mm"], errors="coerce").fillna(0),
            name="Rain",
            line=dict(width=3.2, color=theme.REFERENCE, shape="spline", smoothing=0.5),
            fill="tozeroy",
            fillcolor="rgba(0,117,255,.16)",
            hovertemplate="Rain %{y:.1f} mm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=local_time,
            y=pd.to_numeric(frame["temperature_c"], errors="coerce"),
            name="Temperature",
            line=dict(width=3.2, color=theme.PURPLE, shape="spline", smoothing=0.5),
            yaxis="y2",
            hovertemplate="Temperature %{y:.1f} °C<extra></extra>",
        )
    )
    fig = style_fig(fig, y_title="Rain (mm)", x_title="", height=320)
    fig.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            showline=False,
            tickfont=dict(color=theme.MUTED, size=11),
            title=dict(text="°C", font=dict(color=theme.FAINT, size=11)),
        )
    )
    return fig


def activity_bar_chart(hourly: pd.DataFrame, tz: str) -> go.Figure:
    """Thin white bars in the Vision UI Active Users style."""
    frame = hourly.sort_values("timestamp").copy().iloc[:24]
    labels = to_local(frame["timestamp"], tz).dt.strftime("%H")
    values = pd.to_numeric(frame["rainfall_mm"], errors="coerce").fillna(0)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color="#FFFFFF", line=dict(width=0)),
            width=0.38,
            hovertemplate="%{x}:00 · %{y:.1f} mm<extra></extra>",
            name="Rain",
        )
    )
    fig = style_fig(fig, y_title="", x_title="", height=180, show_legend=False)
    fig.update_layout(bargap=0.55)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    return fig


def skill_history_chart(
    history: pd.DataFrame,
    value_col: str,
    y_title: str,
    better: str = "lower",
) -> go.Figure | None:
    """Line chart of verification skill over issued outlook dates."""
    if history.empty or value_col not in history.columns:
        return None
    frame = history.dropna(subset=[value_col]).sort_values("issued_at").copy()
    if frame.empty:
        return None
    y = pd.to_numeric(frame[value_col], errors="coerce")
    x = pd.to_datetime(frame["issued_at"], utc=True)
    color = theme.REFERENCE if better == "lower" else theme.SUCCESS
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=y_title,
            line=dict(width=3, color=color, shape="spline", smoothing=0.4),
            marker=dict(size=8, color=color),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
        )
    )
    # Mark bias-fix boundary if present in the series.
    fix_at = pd.Timestamp("2026-09-17", tz="UTC")
    if x.min() < fix_at < x.max():
        fig.add_vline(
            x=fix_at.to_pydatetime(),
            line_width=1,
            line_dash="dash",
            line_color=theme.FAINT,
        )
        fig.add_annotation(
            x=fix_at.to_pydatetime(),
            y=1.02,
            yref="paper",
            text="Bias fix",
            showarrow=False,
            font=dict(size=11, color=theme.MUTED),
            xanchor="left",
        )
    fig = style_fig(fig, y_title=y_title, x_title="", height=300)
    return fig


def skill_amount_bars(history: pd.DataFrame) -> go.Figure | None:
    """Bar chart of daily rain error for each issued outlook."""
    if history.empty or "daily_mae" not in history.columns:
        return None
    frame = history.dropna(subset=["daily_mae"]).sort_values("issued_at").copy()
    if frame.empty:
        return None
    labels = pd.to_datetime(frame["issued_at"], utc=True).dt.strftime("%d %b")
    values = pd.to_numeric(frame["daily_mae"], errors="coerce")
    colors = [theme.DANGER if v > 12 else theme.WARN if v > 6 else theme.SUCCESS if v <= 3 else theme.REFERENCE for v in values]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{x}<br>Day error %{y:.1f} mm<extra></extra>",
            name="Day error",
        )
    )
    fig = style_fig(fig, y_title="Day error (mm)", x_title="", height=300, show_legend=False)
    return fig


def telemetry_trend_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    unit: str,
    tz: str,
) -> go.Figure | None:
    """Simple trend for Heimdall multi-domain telemetry."""
    if frame.empty or "timestamp" not in frame.columns or "value" not in frame.columns:
        return None
    subset = frame.sort_values("timestamp").copy()
    y = pd.to_numeric(subset["value"], errors="coerce")
    fig = go.Figure(
        go.Scatter(
            x=to_local(subset["timestamp"], tz),
            y=y,
            name=title,
            mode="lines",
            line=dict(width=2.4, color=theme.ACCENT, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor="rgba(67,24,255,.12)",
            hovertemplate=f"{title} %{{y:.2f}} {unit}<extra></extra>",
        )
    )
    return style_fig(
        fig,
        y_title=f"{title} ({unit})" if unit else title,
        x_title="",
        height=320,
        show_legend=False,
    )
