"""Vision UI visual system for the Streamlit dashboard."""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BG = "#030C1B"
PANEL = "#0B1437"
PANEL_SOFT = "#111C44"
BORDER = "rgba(226, 232, 240, 0.14)"
TEXT = "#FFFFFF"
MUTED = "#A0AEC0"
FAINT = "#718096"

ACCENT = "#4318FF"
REFERENCE = "#0075FF"
RAW = "#56577A"
WARN = "#FFB547"
DANGER = "#E31A1A"
SUCCESS = "#01B574"
PURPLE = "#9F7AEA"
CYAN = "#01B0F4"

CHANNEL_ACCENT = {
    "temperature_c": "#9F7AEA",
    "humidity_pct": "#0075FF",
    "pressure_hpa": "#4318FF",
    "rainfall_mm": "#01B0F4",
    "wind_speed_ms": "#01B574",
    "wind_direction_deg": "#FFB547",
}


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def _theme(
    *,
    label: str,
    blurb: str,
    accent: str,
    accent_2: str,
    blue: str,
    cyan: str,
    bg: str = "#030C1B",
    panel: str = "#0B1437",
    panel_soft: str = "#111C44",
    chip: str = "#1A2555",
    green: str = "#01B574",
    warn: str = "#FFB547",
    danger: str = "#E31A1A",
    raw: str = "#56577A",
    glow_a: float = 0.22,
    glow_b: float = 0.16,
    glow_c: float = 0.12,
) -> dict:
    """Curated Vision-style theme — white text, glass navy surfaces, vivid accents."""
    return {
        "label": label,
        "blurb": blurb,
        "swatch": f"linear-gradient(115deg, {blue} 0%, {accent} 42%, {accent_2} 72%, {cyan} 100%)",
        "bg": bg,
        "panel": panel,
        "panel_soft": panel_soft,
        "chip": chip,
        "text": "#FFFFFF",
        "muted": "#A0AEC0",
        "faint": "#718096",
        "border": "rgba(226, 232, 240, 0.14)",
        "accent": accent,
        "accent_2": accent_2,
        "blue": blue,
        "cyan": cyan,
        "green": green,
        "warn": warn,
        "danger": danger,
        "raw": raw,
        "glow_1": _rgba(accent, glow_a),
        "glow_2": _rgba(accent_2, glow_b),
        "glow_3": _rgba(blue, glow_c),
        "card": (
            "linear-gradient(127.09deg, "
            f"{_rgba(panel, 0.96)} 19.41%, {_rgba(bg, 0.55)} 76.65%)"
        ),
        "channels": {
            "temperature_c": accent_2,
            "humidity_pct": blue,
            "pressure_hpa": accent,
            "rainfall_mm": cyan,
            "wind_speed_ms": green,
            "wind_direction_deg": warn,
        },
    }


THEME_CHOICES = ("vision", "glacier", "aurora", "dusk", "graphite")

THEMES: dict[str, dict] = {
    "vision": _theme(
        label="Vision",
        blurb="Indigo glass — the original hub look",
        accent="#4318FF",
        accent_2="#9F7AEA",
        blue="#0075FF",
        cyan="#01B0F4",
        glow_a=0.24,
        glow_b=0.18,
        glow_c=0.14,
    ),
    "glacier": _theme(
        label="Glacier",
        blurb="Electric sky on deep navy",
        accent="#00B4FF",
        accent_2="#5CE1FF",
        blue="#2B6CFF",
        cyan="#00E5FF",
        chip="#14304A",
        panel="#0A1628",
        panel_soft="#122438",
        bg="#030A14",
        green="#00D68F",
    ),
    "aurora": _theme(
        label="Aurora",
        blurb="Emerald light over night blue",
        accent="#00C896",
        accent_2="#5EEFC0",
        blue="#3D8BFF",
        cyan="#2DE2C5",
        chip="#123528",
        panel="#0A1A16",
        panel_soft="#122820",
        bg="#030F0C",
        green="#00E8A0",
    ),
    "dusk": _theme(
        label="Dusk",
        blurb="Rose gold against indigo dusk",
        accent="#E86B9A",
        accent_2="#F0A0C0",
        blue="#6B7AFF",
        cyan="#FFB86C",
        chip="#2A1840",
        panel="#120E24",
        panel_soft="#1C1634",
        bg="#080612",
        green="#4ADE80",
        warn="#FFB86C",
    ),
    "graphite": _theme(
        label="Graphite",
        blurb="Cool steel with a bright edge",
        accent="#7B8CFF",
        accent_2="#B8C0FF",
        blue="#5B9FFF",
        cyan="#A8D8FF",
        chip="#222838",
        panel="#10141E",
        panel_soft="#181E2A",
        bg="#07090E",
        green="#34D399",
    ),
}

_THEME_ALIASES = {
    "arctic": "glacier",
    "radar": "aurora",
    "marine": "glacier",
    "solar": "dusk",
    "noir": "graphite",
    "ocean": "glacier",
    "forest": "aurora",
    "ember": "dusk",
    "slate": "graphite",
    "ink": "graphite",
}


def normalize_theme_id(theme_id: str | None = None) -> str:
    key = str(theme_id or "vision").strip().lower()
    key = _THEME_ALIASES.get(key, key)
    return key if key in THEMES else "vision"


def theme_tokens(theme_id: str | None = None) -> dict:
    return THEMES[normalize_theme_id(theme_id)]


def apply_theme(theme_id: str | None = None) -> dict:
    """Apply theme tokens to this module and any already-imported charts."""
    global BG, PANEL, PANEL_SOFT, BORDER, TEXT, MUTED, FAINT
    global ACCENT, REFERENCE, RAW, WARN, DANGER, SUCCESS, PURPLE, CYAN
    tokens = theme_tokens(theme_id)
    BG = tokens["bg"]
    PANEL = tokens["panel"]
    PANEL_SOFT = tokens["panel_soft"]
    BORDER = tokens["border"]
    TEXT = "#FFFFFF"
    MUTED = "#A0AEC0"
    FAINT = "#718096"
    ACCENT = tokens["accent"]
    REFERENCE = tokens["blue"]
    RAW = tokens["raw"]
    WARN = tokens["warn"]
    DANGER = tokens["danger"]
    SUCCESS = tokens["green"]
    PURPLE = tokens["accent_2"]
    CYAN = tokens["cyan"]
    CHANNEL_ACCENT.clear()
    CHANNEL_ACCENT.update(tokens["channels"])
    import sys

    charts_mod = sys.modules.get("app.charts")
    if charts_mod is not None:
        charts_mod.ACCENT = ACCENT
        charts_mod.REFERENCE = REFERENCE
        charts_mod.RAW = RAW
        charts_mod.MUTED = MUTED
        charts_mod.FAINT = FAINT
        charts_mod.TEXT = TEXT
        charts_mod.PANEL = PANEL
        charts_mod.BORDER = BORDER
        charts_mod.CHANNEL_ACCENT = CHANNEL_ACCENT
    return tokens


def _swatch_css() -> str:
    rules = []
    for key in THEME_CHOICES:
        swatch = THEMES[key]["swatch"]
        rules.append(
            f".st-key-theme_pick_{key} button {{\n"
            f"  background: {swatch} !important;\n"
            f"  background-image: {swatch} !important;\n"
            f"}}"
        )
    return "\n".join(rules)


def stylesheet(theme_id: str | None = None) -> str:
    """Full dashboard CSS with the active colour theme baked in."""
    t = apply_theme(theme_id)
    accent = t["accent"]
    accent2 = t["accent_2"]
    root = f"""
:root, html, body, .stApp, [data-testid="stAppViewContainer"] {{
  --v-bg: {t["bg"]};
  --v-panel: {t["panel"]};
  --v-chip: {t["chip"]};
  --v-blue: {t["blue"]};
  --v-purple: {accent};
  --v-accent: {accent};
  --v-lilac: {accent2};
  --v-cyan: {t["cyan"]};
  --v-green: {t["green"]};
  --v-warn: {t["warn"]};
  --v-danger: {t["danger"]};
  --v-text: #FFFFFF;
  --v-muted: #A0AEC0;
  --v-faint: #718096;
  --v-card: {t["card"]};
  --v-border: {t["border"]};
  --v-glow-1: {t["glow_1"]};
  --v-glow-2: {t["glow_2"]};
  --v-glow-3: {t["glow_3"]};
  --v-font: "Plus Jakarta Sans", Helvetica, Arial, sans-serif;
  --primary-color: {accent};
  --st-primary-color: {accent};
}}
"""
    body = CSS
    marker = '@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap");'
    if marker in body:
        body = body.replace(marker, marker + "\n" + root, 1)
    elif "<style>" in body:
        body = body.replace("<style>", "<style>" + root, 1)
    else:
        body = "<style>" + root + body + "</style>"
    body = body.replace(
        "</style>",
        _widget_accent_css(accent) + "\n" + _swatch_css() + "</style>",
        1,
    )
    return body


def _widget_accent_css(accent: str) -> str:
    """Hard-hex overrides for Streamlit checkbox / tag / radio / steppers."""
    return f"""
/* Hard accent (literal hex) — Streamlit ignores CSS variables on these widgets */
.stApp [data-baseweb="tag"],
.stApp span[data-baseweb="tag"],
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"],
.stApp [data-baseweb="select"] [data-baseweb="tag"] {{
  background: {accent} !important;
  background-color: {accent} !important;
  background-image: none !important;
  border-color: {accent} !important;
  color: #fff !important;
}}
.stApp [data-baseweb="tag"] *,
.stApp [data-baseweb="tag"] svg,
.stApp [data-baseweb="tag"] path {{
  color: #fff !important;
  fill: #fff !important;
  stroke: #fff !important;
}}
.stApp [data-testid="stCheckbox"] svg,
.stApp [data-testid="stCheckbox"] path,
.stApp [data-testid="stRadio"] svg,
.stApp [data-testid="stRadio"] path,
.stApp [data-testid="stRadio"] circle,
.stApp [data-baseweb="radio"] svg,
.stApp [data-baseweb="radio"] path,
.stApp [data-baseweb="radio"] circle,
.stApp [data-testid="stNumberInput"] button svg,
.stApp [data-testid="stNumberInput"] button path {{
  fill: {accent} !important;
  stroke: {accent} !important;
  color: {accent} !important;
}}
.stApp [data-testid="stCheckbox"] [data-testid="stCheckboxIcon"],
.stApp [data-baseweb="checkbox"][data-checked="true"] > div:first-child {{
  background-color: {accent} !important;
  border-color: {accent} !important;
}}
.stApp [data-testid="stCheckbox"] [data-testid="stCheckboxIcon"] svg,
.stApp [data-testid="stCheckbox"] [data-testid="stCheckboxIcon"] path {{
  fill: #fff !important;
  stroke: #fff !important;
}}
.stApp [data-baseweb="radio"] > div:first-child,
.stApp [data-baseweb="radio"] div[class] {{
  border-color: {accent} !important;
}}
input[type="checkbox"], input[type="radio"] {{
  accent-color: {accent} !important;
}}
.stButton > button[kind="primary"]:not([class*="theme_pick"]),
button[data-testid="baseButton-primary"]:not([class*="theme_pick"]) {{
  background: {accent} !important;
  background-color: {accent} !important;
  border-color: {accent} !important;
  color: #fff !important;
}}
"""


def paint_streamlit_controls(accent: str) -> str:
    """In-page JS+CSS (via st.html) so Streamlit Emotion primary cannot stick on blue."""
    safe = accent.replace("\\", "").replace("'", "").replace('"', "")[:16]
    return f"""
<style id="mh-accent-live">
  html, :root, body, .stApp, [data-testid="stAppViewContainer"] {{
    --st-primary-color: {safe} !important;
    --primary-color: {safe} !important;
    --v-accent: {safe} !important;
    --v-purple: {safe} !important;
  }}
  span[data-baseweb="tag"],
  [data-baseweb="tag"],
  div[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
    background: {safe} !important;
    background-color: {safe} !important;
    border-color: {safe} !important;
    color: #fff !important;
  }}
  [data-testid="stCheckbox"] svg,
  [data-testid="stCheckbox"] path,
  [data-testid="stRadio"] svg,
  [data-testid="stRadio"] path,
  [data-testid="stRadio"] circle,
  [data-testid="stNumberInput"] button svg,
  [data-testid="stNumberInput"] button path {{
    fill: {safe} !important;
    stroke: {safe} !important;
    color: {safe} !important;
  }}
  [data-testid="stCheckboxIcon"],
  [data-baseweb="checkbox"][data-checked="true"] > div:first-child {{
    background-color: {safe} !important;
    border-color: {safe} !important;
  }}
  input[type="checkbox"], input[type="radio"] {{
    accent-color: {safe} !important;
  }}
</style>
<script>
(function () {{
  const accent = "{safe}";
  const root = document;
  try {{
    root.documentElement.style.setProperty("--st-primary-color", accent, "important");
    root.documentElement.style.setProperty("--primary-color", accent, "important");
    if (root.body) {{
      root.body.style.setProperty("--st-primary-color", accent, "important");
      root.body.style.setProperty("--primary-color", accent, "important");
    }}
  }} catch (e) {{}}
  function paint(el, prop, value) {{
    try {{ el.style.setProperty(prop, value, "important"); }} catch (e) {{}}
  }}
  function paintFill(n, color) {{
    paint(n, "fill", color);
    paint(n, "stroke", color);
    paint(n, "color", color);
    try {{
      if (n.getAttribute && n.getAttribute("fill") && n.getAttribute("fill") !== "none") n.setAttribute("fill", color);
      if (n.getAttribute && n.getAttribute("stroke") && n.getAttribute("stroke") !== "none") n.setAttribute("stroke", color);
    }} catch (e) {{}}
  }}
  function apply() {{
    root.querySelectorAll('[data-baseweb="tag"]').forEach((el) => {{
      paint(el, "background", accent);
      paint(el, "background-color", accent);
      paint(el, "border-color", accent);
      paint(el, "color", "#ffffff");
      el.querySelectorAll("svg, path, span, p").forEach((n) => {{
        paintFill(n, "#ffffff");
        paint(n, "color", "#ffffff");
      }});
    }});
    root.querySelectorAll('[data-testid="stCheckbox"]').forEach((wrap) => {{
      const checked = !!wrap.querySelector("input:checked");
      wrap.querySelectorAll('[data-testid="stCheckboxIcon"]').forEach((el) => {{
        paint(el, "background-color", accent);
        paint(el, "border-color", accent);
      }});
      wrap.querySelectorAll("svg, path, circle, rect").forEach((n) => {{
        const icon = n.closest('[data-testid="stCheckboxIcon"]');
        paintFill(n, (icon || checked) ? "#ffffff" : accent);
      }});
      wrap.querySelectorAll('[data-baseweb="checkbox"] > div').forEach((box) => {{
        paint(box, "border-color", accent);
        if (checked) paint(box, "background-color", accent);
      }});
    }});
    root.querySelectorAll('[data-baseweb="checkbox"]').forEach((el) => {{
      const on = el.getAttribute("data-checked") === "true" || el.getAttribute("aria-checked") === "true";
      const box = el.querySelector(":scope > div") || el.querySelector("div");
      if (box) {{
        paint(box, "border-color", accent);
        if (on) paint(box, "background-color", accent);
      }}
      el.querySelectorAll("svg, path").forEach((n) => paintFill(n, on ? "#ffffff" : accent));
    }});
    root.querySelectorAll('[data-testid="stRadio"]').forEach((wrap) => {{
      wrap.querySelectorAll("svg, path, circle").forEach((n) => paintFill(n, accent));
    }});
    root.querySelectorAll('[data-testid="stNumberInput"] button').forEach((btn) => {{
      paint(btn, "color", accent);
      btn.querySelectorAll("svg, path").forEach((n) => paintFill(n, accent));
    }});
    root.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach((el) => {{
      paint(el, "accent-color", accent);
    }});
    root.querySelectorAll('.stButton > button[kind="primary"], button[data-testid="baseButton-primary"]').forEach((btn) => {{
      paint(btn, "background-color", accent);
      paint(btn, "background", accent);
      paint(btn, "border-color", accent);
      paint(btn, "color", "#ffffff");
    }});
    root.querySelectorAll('[data-testid="stPills"] button[aria-checked="true"], [data-testid="stBaseButton-pillsPrimary"]').forEach((btn) => {{
      paint(btn, "background-color", accent);
      paint(btn, "background", accent);
      paint(btn, "border-color", accent);
      paint(btn, "color", "#ffffff");
    }});
  }}
  apply();
  if (window.__mhAccentObs) {{
    try {{ window.__mhAccentObs.disconnect(); }} catch (e) {{}}
  }}
  let t = null;
  const obs = new MutationObserver(() => {{
    if (t) return;
    t = setTimeout(() => {{ t = null; apply(); }}, 50);
  }});
  window.__mhAccentObs = obs;
  if (root.body) obs.observe(root.body, {{ childList: true, subtree: true, attributes: true }});
  setTimeout(apply, 100);
  setTimeout(apply, 400);
  setTimeout(apply, 1000);
}})();
</script>
"""


_ICON = {
    "home": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M4 10.5 12 4l8 6.5V20a1 "
        "1 0 0 1-1 1h-5v-6H10v6H5a1 1 0 0 1-1-1z'/></svg>"
    ),
    "table": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round'><rect x='4' y='4' width='16' height='16' rx='2'/>"
        "<path d='M4 10h16M4 16h16M10 4v16'/></svg>"
    ),
    "chart": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M4 19V5M4 19h16'/>"
        "<path d='M8 15l4-5 3 3 5-7'/></svg>"
    ),
    "layers": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M12 3 3 8l9 5 9-5-9-5z'/>"
        "<path d='M3 12l9 5 9-5M3 16l9 5 9-5'/></svg>"
    ),
    "temp": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round'><path d='M10 14.5V6a2 2 0 1 1 4 0v8.5a3.5 3.5 0 1 1-4 0z'/>"
        "</svg>"
    ),
    "drop": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M12 3s6 7 6 11a6 6 0 1 1-12 "
        "0c0-4 6-11 6-11z'/></svg>"
    ),
    "wind": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round'><path d='M4 8h11a3 3 0 1 0-3-3M4 12h14a3 3 0 1 1-3 3M4 16h8'/>"
        "</svg>"
    ),
    "gauge": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round'><path d='M5 19a9 9 0 1 1 14 0'/><path d='M12 12l4-3'/></svg>"
    ),
    "cloud": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M7 18h10a4 4 0 0 0 "
        "0-8 5.5 5.5 0 0 0-10.4-1.5A3.5 3.5 0 0 0 7 18z'/></svg>"
    ),
}

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap");
/* Theme tokens are injected by stylesheet() — do not hardcode :root accents here. */

html, body, .stApp, p, h1, h2, h3, h4, div, label, button {
  font-family: var(--v-font);
}

/* Font colour is fixed — themes only change accents / surfaces */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stText"],
.stCheckbox label p,
.stRadio label p,
.stMultiSelect label p,
.stSelectbox label p,
.stNumberInput label p {
  color: #FFFFFF !important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp small,
.brand small,
.nav-kicker,
.crumb,
.topbar .sub,
.v-stat .k,
.v-stat .foot,
.sec p,
.page-intro-copy {
  color: #A0AEC0 !important;
}

[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
span[class*="material"] {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
  letter-spacing: 0 !important;
  line-height: 1;
}

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] {
  background: transparent !important;
  border: 0 !important;
  height: 0;
}

.stApp {
  background:
    radial-gradient(900px 540px at 78% -10%, var(--v-glow-1), transparent 62%),
    radial-gradient(700px 500px at 25% 35%, var(--v-glow-3), transparent 65%),
    radial-gradient(620px 420px at 10% 100%, var(--v-glow-2), transparent 58%),
    linear-gradient(145deg, var(--v-bg) 0%, color-mix(in srgb, var(--v-panel) 70%, #000) 58%, var(--v-bg) 100%);
  color: var(--v-text);
}

.block-container {
  padding: 0.45rem 1.35rem 1.6rem 1.9rem !important;
  max-width: 1440px;
}

section[data-testid="stSidebar"] {
  background: transparent !important;
  padding-top: 0 !important;
  padding-right: 12px !important;
}
section[data-testid="stSidebar"] > div {
  background: transparent !important;
  padding-top: 0 !important;
}
[data-testid="stSidebarHeader"] {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
  top: 0.55rem !important;
}
[data-testid="stSidebarContent"] {
  background: var(--v-card) !important;
  backdrop-filter: blur(120px);
  border: 1px solid var(--v-border);
  border-radius: 20px;
  margin: 56px 16px 14px 10px !important;
  padding: 34px 14px 16px !important;
  box-shadow: 0 20px 40px rgba(0,0,0,.18);
}
[data-testid="stMain"] {
  padding-left: 0.4rem;
}
[data-testid="stSidebarCollapseButton"] { color: var(--v-muted) !important; }

.brand {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 6px 16px;
  margin: 8px 0 4px;
  border-bottom: 1px solid rgba(226,232,240,.12);
}
.brand-mark {
  width: 32px; height: 32px; border-radius: 12px; flex: 0 0 32px;
  display: grid; place-items: center;
  background: linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
  color: #fff; font-weight: 700; font-size: .92rem; letter-spacing: -.04em;
}
.brand strong { display: block; color: #fff; font-size: .84rem; font-weight: 700; letter-spacing: .02em; }
.brand small { display: block; color: var(--v-muted); font-size: .62rem; font-weight: 500; margin-top: 1px; }

.nav-kicker {
  color: var(--v-muted);
  font-size: .65rem;
  font-weight: 700;
  letter-spacing: .12em;
  margin: 12px 8px 6px;
}

[data-testid="stSidebar"] .st-key-hub_project { margin: 0; }
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] {
  display: flex; flex-direction: column; gap: 6px;
  background: transparent; border: 0; padding: 0; box-shadow: none;
}
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"] {
  margin: 0; padding: 8px 10px; min-height: 46px;
  border: 0; border-radius: 15px;
  color: var(--v-muted); font-size: .82rem; font-weight: 500;
  background: transparent; box-shadow: none;
  display: flex; align-items: center; gap: 12px;
}
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"] > div:first-child { display: none; }
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"] p {
  margin: 0; color: inherit; font-size: inherit; font-weight: inherit;
}
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:hover {
  color: #fff; background: rgba(255,255,255,.03);
}
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:has(input:checked) {
  color: #fff;
  background: var(--v-chip);
}
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]::before {
  content: "";
  width: 30px; height: 30px; flex: 0 0 30px; border-radius: 8px;
  background-color: var(--v-chip);
  background-repeat: no-repeat; background-position: center; background-size: 15px;
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(1) label::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(1)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a0aec0' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-5v-6H10v6H5a1 1 0 0 1-1-1z'/></svg>");
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(2) label::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(2)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a0aec0' stroke-width='1.8'><rect x='4' y='4' width='16' height='16' rx='2'/><path d='M4 10h16M4 16h16M10 4v16'/></svg>");
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(3) label::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(3)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a0aec0' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 19V5M4 19h16'/><path d='M8 15l4-5 3 3 5-7'/></svg>");
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(4) label::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(4)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a0aec0' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M12 3 3 8l9 5 9-5-9-5z'/><path d='M3 12l9 5 9-5M3 16l9 5 9-5'/></svg>");
}
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:has(input:checked)::before {
  background-color: transparent;
  background-image: none;
  background: linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(1) label:has(input:checked)::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(1):has(input:checked)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-5v-6H10v6H5a1 1 0 0 1-1-1z'/></svg>"), linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
  background-repeat: no-repeat, no-repeat; background-position: center, center; background-size: 15px, 100% 100%;
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(2) label:has(input:checked)::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(2):has(input:checked)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8'><rect x='4' y='4' width='16' height='16' rx='2'/><path d='M4 10h16M4 16h16M10 4v16'/></svg>"), linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
  background-repeat: no-repeat, no-repeat; background-position: center, center; background-size: 15px, 100% 100%;
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(3) label:has(input:checked)::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(3):has(input:checked)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 19V5M4 19h16'/><path d='M8 15l4-5 3 3 5-7'/></svg>"), linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
  background-repeat: no-repeat, no-repeat; background-position: center, center; background-size: 15px, 100% 100%;
}
[data-testid="stSidebar"] .st-key-hub_project [role="radiogroup"] > :nth-child(4) label:has(input:checked)::before,
[data-testid="stSidebar"] .st-key-hub_project label[data-baseweb="radio"]:nth-of-type(4):has(input:checked)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M12 3 3 8l9 5 9-5-9-5z'/><path d='M3 12l9 5 9-5M3 16l9 5 9-5'/></svg>"), linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
  background-repeat: no-repeat, no-repeat; background-position: center, center; background-size: 15px, 100% 100%;
}

.side-meta {
  margin: 12px 4px 0;
  padding: 12px;
  border-radius: 15px;
  background: color-mix(in srgb, var(--v-chip) 88%, transparent);
  border: 1px solid var(--v-border);
}
.side-meta .k { color: var(--v-muted); font-size: .62rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.side-meta .v { color: #fff; font-size: .78rem; font-weight: 600; margin-top: 4px; }
.side-meta .s { color: var(--v-faint); font-size: .68rem; margin-top: 2px; }

[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: color-mix(in srgb, var(--v-chip) 70%, transparent) !important;
  border: 1px solid var(--v-border) !important;
  border-radius: 15px !important;
  margin-top: 12px;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary { color: var(--v-muted); font-size: .78rem; }
/* Sidebar actions: never use --v-text as background — global label CSS forces
   white span/p text, which made Reload UI / Run analysis unreadable. */
[data-testid="stSidebar"] .stButton button {
  width: 100%; margin-top: 10px; height: 38px;
  background: color-mix(in srgb, var(--v-chip) 92%, #fff) !important;
  color: #fff !important;
  border: 1px solid var(--v-border) !important;
  border-radius: 12px;
  font-weight: 700; font-size: .72rem; letter-spacing: .04em;
}
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span {
  color: #fff !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: color-mix(in srgb, var(--v-purple) 45%, var(--v-chip)) !important;
  color: #fff !important;
  border-color: color-mix(in srgb, var(--v-lilac) 70%, var(--v-border)) !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--v-purple) 0%, var(--v-lilac) 100%) !important;
  color: #fff !important;
  border: 0 !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"]:hover {
  filter: brightness(1.08);
}

.help-card {
  margin-top: 8px; padding: 10px 12px;
  border-radius: 12px; overflow: hidden; position: relative;
  background: var(--v-chip);
  border: 1px solid var(--v-border);
}
.help-card strong { display: block; color: var(--v-text); font-size: .78rem; font-weight: 700; }
.help-card small { display: block; color: var(--v-muted); font-size: .66rem; margin-top: 4px; line-height: 1.4; }

.topbar {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 12px; margin: 0 0 4px; padding: 0 2px 8px;
  border-bottom: 1px solid rgba(226,232,240,.08);
}
.crumb { color: var(--v-muted); font-size: .75rem; font-weight: 500; }
.crumb b { color: #fff; font-weight: 700; }
.topbar h1 {
  margin: 0; color: #fff; font-size: 1.22rem; font-weight: 700; letter-spacing: -.03em;
}
.topbar .sub { margin: 2px 0 0; color: var(--v-muted); font-size: .72rem; line-height: 1.35; }
.top-actions { display: flex; align-items: center; gap: 10px; }
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  height: 28px; padding: 0 10px; border-radius: 10px;
  font-size: .68rem; font-weight: 700; border: 1px solid var(--v-border);
  background: color-mix(in srgb, var(--v-chip) 88%, transparent);
}
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill-ok { color: var(--v-green); }
.pill-warn { color: var(--v-warn, #FFB547); }
.pill-idle { color: var(--v-muted); }
.stamp { color: var(--v-faint); font-size: .66rem; text-align: right; margin-top: 4px; }
.dataset-freshness {
  margin: 0 0 12px; padding: 10px 12px;
  border-radius: 14px; border: 1px solid var(--v-border);
  background: color-mix(in srgb, var(--v-chip) 88%, transparent);
}
.dataset-freshness .df-title {
  color: var(--v-muted); font-size: .62rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; margin: 0 0 8px;
}
.dataset-freshness .df-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 8px;
}
.dataset-freshness .df-item {
  min-width: 0; padding: 8px 10px; border-radius: 10px;
  background: color-mix(in srgb, var(--v-panel) 70%, transparent);
  border: 1px solid rgba(226,232,240,.06);
}
.dataset-freshness .df-name {
  display: block; color: #fff; font-size: .72rem; font-weight: 700; margin: 0 0 3px;
}
.dataset-freshness .df-when {
  display: block; font-size: .7rem; font-weight: 600; margin: 0;
}
.dataset-freshness .df-abs {
  display: block; color: var(--v-faint); font-size: .62rem; margin-top: 2px;
}
.dataset-freshness .df-ok { color: var(--v-green); }
.dataset-freshness .df-warn { color: var(--v-warn, #FFB547); }
.dataset-freshness .df-danger { color: var(--v-danger, #FF6B6B); }
.dataset-freshness .df-idle { color: var(--v-muted); }

.overview-alerts {
  display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px;
}
.alert-chip {
  display: inline-flex; align-items: center; max-width: min(100%, 420px);
  padding: 6px 11px; border-radius: 999px; font-size: .7rem; font-weight: 600;
  line-height: 1.35; border: 1px solid var(--v-border);
  background: color-mix(in srgb, var(--v-chip) 88%, transparent); color: var(--v-muted);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.alert-chip.alert-ok { color: var(--v-green); }
.alert-chip.alert-warn { color: var(--v-warn, #FFB547); }
.alert-chip.alert-danger { color: var(--v-danger, #FF6B6B); }
.side-analysis {
  margin: 8px 0 12px; color: var(--v-faint); font-size: .68rem; line-height: 1.35;
}

.v-stats {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px; margin-bottom: 10px;
}
.v-stat {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; min-height: 72px; padding: 14px 16px;
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 16px; backdrop-filter: blur(120px);
}
.v-stat .k { color: var(--v-muted); font-size: .7rem; font-weight: 500; margin: 0 0 3px; }
.v-stat .v { color: #fff; font-size: 1.08rem; font-weight: 700; margin: 0; letter-spacing: -.02em; }
.v-stat .v .u { color: var(--v-muted); font-size: .72rem; font-weight: 500; margin-left: 4px; }
.v-stat .foot { color: var(--v-faint); font-size: .64rem; margin-top: 3px; line-height: 1.3; }
.v-stat .tile-spark { margin-top: 6px; opacity: .9; }
.v-stat .tile-spark svg { display: block; max-width: 100%; height: 28px; }
.v-icon {
  width: 45px; height: 45px; flex: 0 0 45px; border-radius: 12px;
  display: grid; place-items: center; color: #fff;
  background: linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
}
.v-icon svg { width: 18px; height: 18px; }

.welcome {
  position: relative; overflow: hidden; min-height: 0;
  padding: 14px 16px 12px; border-radius: 16px; margin-bottom: 0;
  background:
    radial-gradient(280px 180px at 88% 20%, var(--v-glow-2), transparent 58%),
    radial-gradient(240px 200px at 70% 90%, var(--v-glow-1), transparent 60%),
    var(--v-card);
  border: 1px solid var(--v-border);
}
.welcome .k { color: var(--v-muted); font-size: .68rem; font-weight: 600; margin: 0 0 4px; letter-spacing: .04em; }
.welcome h2 { color: #fff; font-size: 1.18rem; font-weight: 700; margin: 0 0 6px; letter-spacing: -.03em; line-height: 1.25; }
.welcome p { color: var(--v-muted); font-size: .8rem; line-height: 1.45; max-width: none; margin: 0; }
.welcome .cta {
  display: inline-block; margin-top: 12px; padding: 7px 12px; border-radius: 10px;
  color: #fff; font-size: .74rem; font-weight: 700;
  background: rgba(255,255,255,.06); border: 1px solid rgba(226,232,240,.12);
}

.hero-row {
  display: grid; grid-template-columns: 1.35fr 0.82fr 0.82fr;
  gap: 12px; margin-bottom: 12px;
}
.chart-row {
  display: grid; grid-template-columns: 1.7fr 1fr;
  gap: 12px; margin-bottom: 12px;
}
.gauges { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.gauge-card {
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 20px; padding: 18px 16px 20px; text-align: center;
  backdrop-filter: blur(120px);
  overflow: visible;
}
.gauge-card .k { color: #fff; font-size: .95rem; font-weight: 700; margin: 0 0 4px; }
.gauge-card .s { color: var(--v-muted); font-size: .72rem; margin: 0 0 2px; }
.gauge-card .foot {
  color: var(--v-muted); font-size: .74rem; line-height: 1.45;
  margin-top: 8px; padding: 0 6px; min-height: 2.8em;
}
.gauge-card svg { overflow: visible; display: block; margin: 0 auto; width: 100%; max-width: 210px; height: auto; }

.kit-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px; margin: 0 0 16px;
}
.kit-grid-four { grid-template-columns: repeat(4, minmax(0, 1fr)); }

.chart-card {
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 20px; padding: 18px 8px 8px;
  backdrop-filter: blur(120px); margin-bottom: 16px;
}
.chart-head { display: flex; justify-content: space-between; align-items: baseline; padding: 0 16px 4px; }
.chart-head h3 { color: #fff; font-size: 1.05rem; font-weight: 700; margin: 0; }
.chart-head span { font-size: .76rem; font-weight: 600; }
.chart-head span.note { color: var(--v-muted); }
.chart-head span.warn { color: var(--v-warn); }

.activity-stats {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px;
  margin-top: 12px; padding: 16px 18px;
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 20px;
}
.activity-stats .item { display: flex; gap: 10px; align-items: center; }
.activity-stats .mini-icon {
  width: 32px; height: 32px; border-radius: 8px; display: grid; place-items: center; color: #fff;
  background: linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
}
.activity-stats .mini-icon svg { width: 14px; height: 14px; }
.activity-stats .k { color: var(--v-muted); font-size: .68rem; margin: 0; }
.activity-stats .v { color: #fff; font-size: .92rem; font-weight: 700; margin: 0; }

.days {
  display: grid; grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 8px; margin: 4px 0 6px;
}
.day {
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 14px; padding: 12px 10px; min-width: 0;
}
.day.today { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--v-blue) 50%, transparent); }
.day .d { color: #fff; font-size: .72rem; font-weight: 700; }
.day .date { color: var(--v-muted); font-size: .64rem; margin-top: 1px; }
.day .icon { margin: 8px 0 4px; color: #fff; }
.day .icon svg { width: 20px; height: 20px; }
.day .rain-main { color: #fff; font-size: 1.05rem; font-weight: 700; }
.day .rain-main span { color: var(--v-muted); font-size: .66rem; font-weight: 500; margin-left: 3px; }
.day .rain-hours { color: var(--v-muted); font-size: .66rem; font-weight: 600; margin-top: 4px; line-height: 1.3; min-height: 2.2em; }
.day .rain-hours.all-day { color: var(--v-cyan); }
.day .temps { color: var(--v-muted); font-size: .66rem; margin-top: 6px; }

.grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin-bottom: 12px;
}
.v-stat.full { min-height: 88px; }

.sec { margin: 4px 0 8px; }
.sec h3 { color: #fff; font-size: .95rem; font-weight: 700; margin: 0; letter-spacing: -.01em; text-transform: none; }
.sec p { color: var(--v-muted); font-size: .76rem; margin: 4px 0 0; line-height: 1.4; max-width: 64ch; }

.page-intro {
  background: transparent; border: 0; border-radius: 0;
  padding: 0 2px 8px; margin: 0 0 8px;
  border-bottom: 1px solid rgba(226,232,240,.06);
}
.page-intro-title {
  color: #fff; font-size: 1.02rem; font-weight: 700; margin: 0 0 3px;
  letter-spacing: -.03em; line-height: 1.2;
}
.page-intro-copy {
  color: var(--v-muted); font-size: .76rem; line-height: 1.4; max-width: 62ch; margin: 0;
}
.st-key-hub_nav .stButton > button {
  font-size: .72rem !important;
  letter-spacing: .01em !important;
}
.source-line, [data-testid="stCaptionContainer"] {
  max-width: 72ch;
}

.st-key-theme_swatches [data-testid="stHorizontalBlock"] { gap: 12px !important; }
.st-key-theme_swatches .stButton { margin: 0 !important; }
.st-key-theme_swatches .stButton > button {
  height: 44px !important;
  min-height: 44px !important;
  padding: 0 !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  color: transparent !important;
  font-size: 0 !important;
  box-shadow: 0 8px 20px rgba(0,0,0,.22) !important;
  overflow: hidden !important;
}
.st-key-theme_swatches .stButton > button p,
.st-key-theme_swatches .stButton > button span {
  color: transparent !important;
  font-size: 0 !important;
}
.st-key-theme_swatches .stButton > button:hover {
  filter: brightness(1.08) saturate(1.05);
  transform: translateY(-1px);
}

.panel {
  background: var(--v-card);
  border: 1px solid var(--v-border);
  border-radius: 18px;
  padding: 14px 16px 12px;
  margin: 0 0 10px;
  backdrop-filter: blur(120px);
}
.panel > .welcome {
  margin: 0;
  border: 0;
  background: transparent;
  padding: 0;
  min-height: 0;
}
.panel-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 12px; margin: 0 0 14px;
}
.panel-head h3 { color: #fff; font-size: .98rem; font-weight: 700; margin: 0; letter-spacing: -.01em; }
.panel-head p { color: var(--v-muted); font-size: .76rem; margin: 4px 0 0; line-height: 1.4; }
.panel-tag {
  flex: 0 0 auto; align-self: center;
  font-size: .68rem; font-weight: 700; letter-spacing: .04em;
  padding: 5px 10px; border-radius: 999px;
  border: 1px solid var(--v-border); color: var(--v-muted);
  background: color-mix(in srgb, var(--v-chip) 88%, transparent);
}
.panel-tag.up { color: var(--v-green); border-color: color-mix(in srgb, var(--v-green) 40%, transparent); background: color-mix(in srgb, var(--v-green) 12%, transparent); }
.panel-tag.down { color: var(--v-warn); border-color: color-mix(in srgb, var(--v-warn) 40%, transparent); background: color-mix(in srgb, var(--v-warn) 12%, transparent); }
.panel-tag.steady { color: var(--v-cyan); border-color: color-mix(in srgb, var(--v-cyan) 40%, transparent); background: color-mix(in srgb, var(--v-cyan) 12%, transparent); }

.compare-row {
  display: grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: stretch;
}
.compare-card {
  background: color-mix(in srgb, var(--v-chip) 78%, transparent);
  border: 1px solid var(--v-border);
  border-radius: 16px;
  padding: 16px 16px 14px;
}
.compare-card .era {
  color: var(--v-muted); font-size: .65rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; margin: 0 0 8px;
}
.compare-card .big {
  color: #fff; font-size: 1.55rem; font-weight: 700; letter-spacing: -.03em; margin: 0;
  line-height: 1.1;
}
.compare-card .big .u { color: var(--v-muted); font-size: .8rem; font-weight: 500; margin-left: 4px; }
.compare-card .meta { color: var(--v-faint); font-size: .72rem; margin-top: 8px; line-height: 1.4; }
.compare-arrow {
  display: grid; place-items: center; color: var(--v-muted); font-size: .9rem; font-weight: 700;
  min-width: 28px;
}

.milestone-rail {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px; margin: 0 0 8px;
}
.milestone {
  position: relative;
  background: color-mix(in srgb, var(--v-chip) 78%, transparent);
  border: 1px solid var(--v-border);
  border-radius: 16px;
  padding: 14px 14px 12px;
  min-height: 108px;
}
.milestone .step {
  color: var(--v-cyan); font-size: .62rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; margin: 0 0 6px;
}
.milestone .title { color: #fff; font-size: .84rem; font-weight: 700; margin: 0; line-height: 1.25; }
.milestone .value { color: #fff; font-size: 1.05rem; font-weight: 700; margin: 8px 0 0; letter-spacing: -.02em; }
.milestone .foot { color: var(--v-faint); font-size: .7rem; margin-top: 6px; line-height: 1.35; }

.band-row {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px; margin: 0 0 8px;
}
.band-card {
  background: color-mix(in srgb, var(--v-chip) 78%, transparent);
  border: 1px solid var(--v-border);
  border-radius: 16px;
  padding: 14px 14px 12px;
}
.band-card .k { color: var(--v-muted); font-size: .7rem; font-weight: 600; margin: 0; }
.band-card .v { color: #fff; font-size: 1.2rem; font-weight: 700; margin: 6px 0 0; }
.band-card .f { color: var(--v-faint); font-size: .68rem; margin-top: 4px; }
.band-card.good { border-color: color-mix(in srgb, var(--v-green) 40%, transparent); }
.band-card.fair { border-color: color-mix(in srgb, var(--v-blue) 40%, transparent); }
.band-card.weak { border-color: color-mix(in srgb, var(--v-warn) 40%, transparent); }
.band-card.timing { border-color: color-mix(in srgb, var(--v-cyan) 40%, transparent); }

.chart-split {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 0 0 8px;
}
.chart-split .panel { margin-bottom: 0; }

.source-grid {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px; margin: 0 0 16px;
}
.source-card {
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 20px; padding: 16px 18px; min-height: 118px;
}
.source-card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.source-role { color: var(--v-muted); font-size: .65rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.source-state { font-size: .68rem; font-weight: 700; padding: 3px 8px; border-radius: 8px; }
.source-ok { color: var(--v-green); background: color-mix(in srgb, var(--v-green) 14%, transparent); }
.source-idle { color: var(--v-muted); background: rgba(160,174,192,.1); }
.source-name { color: #fff; font-size: .95rem; font-weight: 700; }
.source-copy { color: var(--v-muted); font-size: .75rem; margin-top: 4px; line-height: 1.4; }

.source-line {
  color: var(--v-muted); font-size: .78rem; line-height: 1.5;
  padding: 14px 16px; border-radius: 20px; margin: 8px 0 0;
  background: var(--v-card); border: 1px solid var(--v-border);
}
.source-line strong { color: #fff; }

.insights { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 0 0 16px; }
.insight {
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 20px; padding: 16px 18px;
}
.insight .top { display: flex; justify-content: space-between; gap: 8px; }
.insight .k { color: var(--v-muted); font-size: .7rem; font-weight: 600; }
.insight .badge { color: var(--v-green); font-size: .68rem; font-weight: 700; }
.insight .title { color: #fff; font-size: .95rem; font-weight: 700; margin-top: 10px; }
.insight .copy { color: var(--v-muted); font-size: .75rem; margin-top: 4px; line-height: 1.4; }

.timeline {
  background: var(--v-card); border: 1px solid var(--v-border);
  border-radius: 20px; padding: 8px 8px 8px 6px;
}
.t-row { display: flex; gap: 12px; padding: 12px 12px; }
.t-icon {
  width: 32px; height: 32px; border-radius: 10px; flex: 0 0 32px;
  display: grid; place-items: center; color: #fff;
  background: linear-gradient(97.89deg, var(--v-purple) 0%, var(--v-lilac) 100%);
}
.t-icon svg { width: 14px; height: 14px; }
.t-title { color: #fff; font-size: .82rem; font-weight: 600; }
.t-copy { color: var(--v-muted); font-size: .72rem; margin-top: 2px; }

div[data-testid="stPlotlyChart"] {
  background: var(--v-card);
  border: 1px solid var(--v-border);
  border-radius: 20px;
  padding: 8px 4px 4px;
  backdrop-filter: blur(120px);
  margin-bottom: 10px;
}
.panel div[data-testid="stPlotlyChart"] {
  background: transparent;
  border: 0;
  border-radius: 12px;
  padding: 0;
  backdrop-filter: none;
  margin-bottom: 6px;
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--v-border); border-radius: 20px; overflow: hidden;
  margin-top: 4px;
}
.stAlert { border-radius: 16px; border: 1px solid var(--v-border); }
[data-testid="stExpander"] {
  background: var(--v-card) !important;
  border: 1px solid var(--v-border) !important;
  border-radius: 16px !important;
}
[data-testid="stExpander"] summary {
  position: relative;
  overflow: hidden;
}
[data-testid="stIconMaterial"] {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  width: 1.1rem;
  height: 1.1rem;
}
[data-testid="stMarkdownContainer"] { overflow: visible; }

.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: color-mix(in srgb, var(--v-chip) 90%, transparent);
  border: 1px solid var(--v-border);
  border-radius: 16px;
  padding: 5px;
  margin-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 12px;
  color: var(--v-muted);
  font-weight: 600;
  font-size: .88rem;
  padding: 10px 18px;
}
.stTabs [data-baseweb="tab"]:hover { color: #fff; }
.stTabs [aria-selected="true"] {
  color: #fff !important;
  background: var(--v-chip) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }
/* Section nav — underline tabs (custom buttons, not segmented_control) */
.st-key-hub_nav {
  margin: 0 0 12px !important;
  padding: 0 !important;
  border-bottom: 1px solid rgba(226, 232, 240, 0.1);
}
.st-key-hub_nav [data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: stretch !important;
}
.st-key-hub_nav [data-testid="stColumn"] {
  padding: 0 !important;
}
.st-key-hub_nav .stButton {
  margin: 0 !important;
  width: 100%;
}
.st-key-hub_nav .stButton > button {
  width: 100% !important;
  height: 38px !important;
  min-height: 38px !important;
  margin: 0 !important;
  padding: 0 10px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  background-image: none !important;
  color: var(--v-muted) !important;
  font-family: var(--v-font) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  box-shadow: none !important;
  outline: none !important;
  border-bottom: 2px solid transparent !important;
  transition: color .15s ease, border-color .15s ease, background .15s ease !important;
}
.st-key-hub_nav .stButton > button:hover {
  color: #fff !important;
  background: rgba(255, 255, 255, 0.03) !important;
  border-bottom-color: rgba(226, 232, 240, 0.22) !important;
}
.st-key-hub_nav .stButton > button:focus,
.st-key-hub_nav .stButton > button:focus-visible,
.st-key-hub_nav .stButton > button:active {
  outline: none !important;
  box-shadow: none !important;
  color: #fff !important;
  background: transparent !important;
}
.st-key-hub_nav .stButton > button[kind="primary"],
.st-key-hub_nav .stButton > button[data-testid="baseButton-primary"] {
  color: #fff !important;
  background: transparent !important;
  background-image: none !important;
  border-bottom: 2px solid var(--v-lilac) !important;
  font-weight: 700 !important;
}
.st-key-hub_nav .stButton > button[kind="primary"]:hover,
.st-key-hub_nav .stButton > button[data-testid="baseButton-primary"]:hover {
  color: #fff !important;
  background: color-mix(in srgb, var(--v-lilac) 14%, transparent) !important;
  border-bottom-color: var(--v-lilac) !important;
}
.st-key-hub_nav .stButton > button p,
.st-key-hub_nav .stButton > button span {
  color: inherit !important;
  font-size: inherit !important;
  font-weight: inherit !important;
}

[data-testid="stWidgetLabel"] p {
  color: var(--v-muted) !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
}
[data-testid="stCaptionContainer"] p { color: var(--v-faint) !important; font-size: .72rem !important; }

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div {
  border-radius: 12px !important;
}
[data-testid="stCustomComponentV1"] {
  background: var(--v-bg);
  border: 1px solid var(--v-border);
  border-radius: 20px;
  overflow: hidden;
  min-height: 560px;
}
[data-testid="stCustomComponentV1"] iframe {
  border: 0;
  border-radius: 20px;
}

.map-save-spacer { height: 1.85rem; }

.legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 8px; padding: 0 4px; }
.legend span { font-size: .75rem; color: var(--v-muted); display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 2px; display: inline-block; border-radius: 2px; }

@media (max-width: 1100px) {
  .v-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero-row, .chart-row, .gauges, .chart-split, .compare-row { grid-template-columns: 1fr; }
  .compare-arrow { display: none; }
  .kit-grid { grid-template-columns: 1fr 1fr; }
  .days { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .source-grid, .insights, .milestone-rail, .band-row { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 700px) {
  .v-stats, .days, .source-grid, .insights, .kit-grid, .milestone-rail, .band-row { grid-template-columns: 1fr; }
  .topbar { flex-direction: column; }
  .topbar h1 { font-size: 1.5rem; }
}

/* Streamlit form controls — force theme accent (beats config.toml primary) */
[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
  color: var(--v-muted) !important;
}
[data-testid="stCaptionContainer"] p {
  color: var(--v-faint) !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
  background-color: var(--v-chip) !important;
  color: var(--v-text) !important;
  border-color: var(--v-border) !important;
  caret-color: var(--v-purple) !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus,
div[data-baseweb="select"] > div:focus-within,
div[data-testid="stMultiSelect"] > div > div:focus-within {
  border-color: var(--v-purple) !important;
  box-shadow: 0 0 0 1px var(--v-purple) !important;
}
div[data-baseweb="select"] svg {
  color: var(--v-muted) !important;
  fill: var(--v-muted) !important;
}
/* Number +/- steppers */
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button svg,
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] svg,
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] svg {
  color: var(--v-purple) !important;
  fill: var(--v-purple) !important;
  stroke: var(--v-purple) !important;
}
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
ul[role="listbox"],
li[role="option"] {
  background-color: var(--v-panel) !important;
  color: var(--v-text) !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
  background-color: var(--v-chip) !important;
  color: var(--v-text) !important;
}
/* Multiselect / select tags */
.stApp [data-baseweb="tag"],
.stApp span[data-baseweb="tag"],
div[data-testid="stMultiSelect"] [data-baseweb="tag"],
div[data-baseweb="select"] [data-baseweb="tag"] {
  background: var(--v-purple) !important;
  background-color: var(--v-purple) !important;
  background-image: none !important;
  color: #ffffff !important;
  border: 0 !important;
}
.stApp [data-baseweb="tag"] span,
.stApp [data-baseweb="tag"] p,
.stApp [data-baseweb="tag"] svg,
.stApp [data-baseweb="tag"] path {
  color: #ffffff !important;
  fill: #ffffff !important;
  stroke: #ffffff !important;
}
/* Checkboxes / radios */
input[type="checkbox"],
input[type="radio"] {
  accent-color: var(--v-purple) !important;
}
.stApp [data-testid="stCheckbox"] svg,
.stApp [data-testid="stCheckbox"] path,
.stApp [data-testid="stRadio"] svg,
.stApp [data-testid="stRadio"] path {
  fill: var(--v-purple) !important;
  color: var(--v-purple) !important;
  stroke: var(--v-purple) !important;
}
.stApp [data-testid="stCheckbox"] [data-testid="stCheckboxIcon"],
.stApp [data-testid="stCheckbox"] label:has(input:checked) span,
.stApp [data-baseweb="checkbox"][data-checked="true"] > div:first-child {
  background-color: var(--v-purple) !important;
  border-color: var(--v-purple) !important;
  color: #ffffff !important;
}
.stApp [data-baseweb="checkbox"] > div:first-child {
  border-color: var(--v-purple) !important;
}
.stApp [data-testid="stCheckbox"] [data-testid="stCheckboxIcon"] svg,
.stApp [data-testid="stCheckbox"] [data-testid="stCheckboxIcon"] path {
  fill: #ffffff !important;
  stroke: #ffffff !important;
  color: #ffffff !important;
}
.stCheckbox label p,
.stMultiSelect label p,
.stSelectbox label p,
.stNumberInput label p,
.stRadio label p {
  color: var(--v-muted) !important;
}
/* Slider / progress accents if present */
[data-testid="stSlider"] [role="slider"],
[data-baseweb="slider"] [role="slider"] {
  background-color: var(--v-purple) !important;
}
[data-baseweb="slider"] div[class*="Tick"] ,
[data-testid="stSlider"] div[style*="background"] {
  background-color: var(--v-purple) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--v-purple) 0%, var(--v-lilac) 100%) !important;
  color: #fff !important;
  border: 0 !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
  background: color-mix(in srgb, var(--v-chip) 90%, transparent) !important;
  color: var(--v-text) !important;
  border: 1px solid var(--v-border) !important;
}
.stAlert, [data-testid="stAlert"] {
  background: var(--v-card) !important;
  color: var(--v-text) !important;
  border-color: var(--v-border) !important;
}
.st-key-hub_nav .stButton > button[kind="primary"],
.st-key-hub_nav .stButton > button[data-testid="baseButton-primary"] {
  background: transparent !important;
  background-image: none !important;
  color: var(--v-text) !important;
  border: 0 !important;
  border-bottom: 2px solid var(--v-lilac) !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
</style>
"""


def icon(name: str) -> str:
    return _ICON.get(name, _ICON["gauge"])


def sparkline(values: np.ndarray, color: str, width: int = 168, height: int = 34) -> str:
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 3:
        return ""
    if series.size > 90:
        series = series[:: int(np.ceil(series.size / 90))]
    lo, hi = float(series.min()), float(series.max())
    span = hi - lo
    pad = 3.0
    usable = height - 2 * pad
    ys = np.full(series.size, height / 2.0) if span <= 1e-9 else pad + usable * (1.0 - (series - lo) / span)
    xs = np.linspace(1, width - 1, series.size)
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    area = f"{xs[0]:.1f},{height} {line} {xs[-1]:.1f},{height}"
    uid = abs(hash((color, float(series[0]), float(series[-1]), series.size))) % 100000
    return (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" role="img">'
        f'<defs><linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        f'<polygon points="{area}" fill="url(#g{uid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )


def _icon_for(kicker: str) -> str:
    key = kicker.lower()
    if "temp" in key or "warm" in key:
        return "temp"
    if "humid" in key or "rain" in key or "wet" in key:
        return "drop"
    if "wind" in key:
        return "wind"
    if "pressure" in key or "source" in key or "area" in key:
        return "layers"
    if "sensor" in key or "usable" in key or "good" in key:
        return "gauge"
    if "reading" in key or "stored" in key:
        return "table"
    return "chart"


def tile(
    kicker: str,
    value: str,
    unit: str = "",
    foot: str = "",
    accent: str | None = None,
    spark: str = "",
) -> str:
    del accent
    unit_html = f'<span class="u">{html.escape(unit)}</span>' if unit else ""
    foot_html = f'<div class="foot">{html.escape(foot)}</div>' if foot else ""
    spark_html = f'<div class="tile-spark">{spark}</div>' if spark else ""
    icon_name = _icon_for(kicker)
    return (
        '<div class="v-stat full">'
        "<div>"
        f'<p class="k">{html.escape(kicker)}</p>'
        f'<p class="v">{html.escape(value)}{unit_html}</p>'
        f"{spark_html}"
        f"{foot_html}"
        "</div>"
        f'<div class="v-icon">{icon(icon_name)}</div>'
        "</div>"
    )


def stat_card(label: str, value: str, unit: str = "", footnote: str = "", icon_name: str = "gauge") -> str:
    unit_html = f'<span class="u">{html.escape(unit)}</span>' if unit else ""
    foot_html = f'<p class="foot">{html.escape(footnote)}</p>' if footnote else ""
    return (
        '<div class="v-stat">'
        "<div>"
        f'<p class="k">{html.escape(label)}</p>'
        f'<p class="v">{html.escape(value)}{unit_html}</p>'
        f"{foot_html}"
        "</div>"
        f'<div class="v-icon">{icon(icon_name)}</div>'
        "</div>"
    )


def grid(tiles: list[str]) -> str:
    return f'<div class="grid">{"".join(tiles)}</div>'


def section(title: str, note: str = "") -> str:
    note_html = f"<p>{html.escape(note)}</p>" if note else ""
    return f'<div class="sec"><h3>{html.escape(title)}</h3>{note_html}</div>'


def panel(title: str, body_html: str, note: str = "", tag: str = "", tag_class: str = "") -> str:
    """Card panel with optional status tag and HTML body."""
    note_html = f"<p>{html.escape(note)}</p>" if note else ""
    tag_html = (
        f'<span class="panel-tag {html.escape(tag_class)}">{html.escape(tag)}</span>' if tag else ""
    )
    return (
        '<div class="panel">'
        f'<div class="panel-head"><div><h3>{html.escape(title)}</h3>{note_html}</div>{tag_html}</div>'
        f"{body_html}"
        "</div>"
    )


def compare_pair(
    left_era: str,
    left_value: str,
    left_meta: str,
    right_era: str,
    right_value: str,
    right_meta: str,
) -> str:
    return (
        '<div class="compare-row">'
        f'<div class="compare-card"><div class="era">{html.escape(left_era)}</div>'
        f'<p class="big">{left_value}</p>'
        f'<div class="meta">{html.escape(left_meta)}</div></div>'
        '<div class="compare-arrow">→</div>'
        f'<div class="compare-card"><div class="era">{html.escape(right_era)}</div>'
        f'<p class="big">{right_value}</p>'
        f'<div class="meta">{html.escape(right_meta)}</div></div>'
        "</div>"
    )


def milestone(step: str, title: str, value: str, foot: str = "") -> str:
    foot_html = f'<div class="foot">{html.escape(foot)}</div>' if foot else ""
    return (
        '<div class="milestone">'
        f'<div class="step">{html.escape(step)}</div>'
        f'<div class="title">{html.escape(title)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f"{foot_html}"
        "</div>"
    )


def band_card(label: str, value: str, foot: str = "", kind: str = "") -> str:
    foot_html = f'<div class="f">{html.escape(foot)}</div>' if foot else ""
    cls = f"band-card {kind}".strip()
    return (
        f'<div class="{cls}">'
        f'<p class="k">{html.escape(label)}</p>'
        f'<p class="v">{html.escape(value)}</p>'
        f"{foot_html}"
        "</div>"
    )


def legend(entries: list[tuple[str, str]]) -> str:
    items = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        for label, color in entries
    )
    return f'<div class="legend">{items}</div>'


def welcome_card(kicker: str, title: str, body: str, cta: str = "") -> str:
    cta_html = f'<div class="cta">{html.escape(cta)}</div>' if cta else ""
    return (
        '<div class="welcome">'
        f'<p class="k">{html.escape(kicker)}</p>'
        f"<h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(body)}</p>"
        f"{cta_html}"
        "</div>"
    )


def semi_gauge(percent: float, center: str, caption: str, title: str = "", subtitle: str = "") -> str:
    pct = max(0.0, min(100.0, float(percent)))
    rest = max(0.0, 100.0 - pct)
    cx, cy, radius = 110.0, 138.0, 82.0
    theta = math.pi * (1.0 - pct / 100.0)
    cap_x = cx + radius * math.cos(theta)
    cap_y = cy - radius * math.sin(theta)
    title_html = f'<div class="k">{html.escape(title)}</div>' if title else ""
    sub_html = f'<div class="s">{html.escape(subtitle)}</div>' if subtitle else ""
    track = f"M {cx - radius:.1f} {cy:.1f} A {radius:.0f} {radius:.0f} 0 0 1 {cx + radius:.1f} {cy:.1f}"
    return (
        '<div class="gauge-card">'
        f"{title_html}{sub_html}"
        '<svg viewBox="0 0 220 168" width="210" height="160" overflow="visible" aria-hidden="true">'
        "<defs>"
        '<linearGradient id="rainWatchGrad" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{REFERENCE}"/>'
        f'<stop offset="100%" stop-color="{PURPLE}"/>'
        "</linearGradient>"
        "</defs>"
        f'<path d="{track}" fill="none" stroke="{PANEL_SOFT}" stroke-width="14" '
        'stroke-linecap="round" pathLength="100"/>'
        f'<path d="{track}" fill="none" stroke="url(#rainWatchGrad)" stroke-width="14" '
        'stroke-linecap="round" pathLength="100" '
        f'stroke-dasharray="{pct:.1f} {rest:.1f}"/>'
        f'<circle cx="{cap_x:.1f}" cy="{cap_y:.1f}" r="7" fill="{PURPLE}"/>'
        f'<text x="110" y="124" text-anchor="middle" fill="#fff" font-size="22" '
        f'font-weight="700" font-family="Plus Jakarta Sans, Helvetica, sans-serif">{html.escape(center)}</text>'
        "</svg>"
        f'<div class="foot">{html.escape(caption)}</div>'
        "</div>"
    )


def ring_gauge(center: str, title: str, caption: str, percent: float = 50.0) -> str:
    pct = max(0.0, min(100.0, float(percent))) / 100.0
    circumference = 2 * math.pi * 42
    dash = f"{circumference * pct:.1f} {circumference:.1f}"
    return (
        '<div class="gauge-card">'
        f'<div class="k">{html.escape(title)}</div>'
        '<svg viewBox="0 0 100 100" width="140" height="140">'
        f'<circle cx="50" cy="50" r="42" fill="none" stroke="{PANEL_SOFT}" stroke-width="8"/>'
        f'<circle cx="50" cy="50" r="42" fill="none" stroke="{SUCCESS}" stroke-width="8" '
        f'stroke-linecap="round" stroke-dasharray="{dash}" transform="rotate(-90 50 50)"/>'
        f'<text x="50" y="54" text-anchor="middle" fill="#fff" font-size="13" font-weight="700" '
        f'font-family="Plus Jakarta Sans, Helvetica, sans-serif">{html.escape(center)}</text>'
        "</svg>"
        f'<div class="foot">{html.escape(caption)}</div>'
        "</div>"
    )


def style_fig(
    fig: go.Figure,
    y_title: str = "",
    x_title: str = "",
    height: int = 340,
    show_legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Helvetica, Arial, sans-serif", color=MUTED, size=12),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=11),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=BORDER,
            font=dict(color=TEXT, family="Plus Jakarta Sans, Helvetica, Arial, sans-serif"),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color=FAINT, size=11)),
            showgrid=True,
            gridcolor="rgba(255,255,255,.06)",
            griddash="dot",
            zeroline=False,
            showline=False,
            tickfont=dict(color=MUTED, size=11),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color=FAINT, size=11)),
            showgrid=True,
            gridcolor="rgba(255,255,255,.06)",
            griddash="dot",
            zeroline=False,
            showline=False,
            tickfont=dict(color=MUTED, size=11),
        ),
    )
    return fig


def to_local(series: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(tz)
