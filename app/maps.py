"""Detailed satellite map and place search for station location."""

from __future__ import annotations

import json
import math

import requests

WATCH_COLORS = {
    "High": "#E31A1A",
    "Raised": "#FFB547",
    "Wet spell": "#E31A1A",
    "Rain likely": "#0075FF",
    "Humid": "#9F7AEA",
    "Routine": "#01B574",
    "Mostly dry": "#01B574",
}


def _place_label(properties: dict, fallback: str) -> str:
    parts = [
        properties.get("name") or properties.get("street") or fallback,
        properties.get("city")
        or properties.get("locality")
        or properties.get("district")
        or properties.get("county"),
        properties.get("state"),
        properties.get("country"),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        text = str(part).strip() if part else ""
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            unique.append(text)
    return ", ".join(unique[:4]) or fallback


def search_places(
    query: str,
    lat: float | None = None,
    lon: float | None = None,
    limit: int = 8,
) -> list[tuple[float, float, str]]:
    text = query.strip()
    if len(text) < 3:
        return []
    hits = _photon_places(text, lat, lon, limit)
    if hits:
        return hits
    return _nominatim_places(text, lat, lon, limit)


def search_place(
    query: str,
    lat: float | None = None,
    lon: float | None = None,
) -> tuple[float, float, str] | None:
    hits = search_places(query, lat, lon, limit=1)
    return hits[0] if hits else None


def _photon_places(
    text: str,
    lat: float | None,
    lon: float | None,
    limit: int,
) -> list[tuple[float, float, str]]:
    params: dict[str, str | int | float] = {"q": text, "limit": limit, "lang": "en"}
    if lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
        params["location_bias_scale"] = 4
    try:
        response = requests.get(
            "https://photon.komoot.io/api/",
            params=params,
            headers={"User-Agent": "MechatronWeatherHub/1.0"},
            timeout=6,
        )
        response.raise_for_status()
        features = response.json().get("features") or []
    except Exception:
        return []
    hits: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float]] = set()
    for feature in features:
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon_val, lat_val = float(coords[0]), float(coords[1])
        key = (round(lat_val, 5), round(lon_val, 5))
        if key in seen:
            continue
        seen.add(key)
        label = _place_label(feature.get("properties") or {}, text)
        hits.append((lat_val, lon_val, label))
    return hits


def _nominatim_places(
    text: str,
    lat: float | None,
    lon: float | None,
    limit: int,
) -> list[tuple[float, float, str]]:
    params: dict[str, str | int] = {
        "q": text,
        "format": "json",
        "limit": limit,
        "addressdetails": 1,
    }
    if lat is not None and lon is not None:
        # Prefer nearby results: viewbox around the station, then still allow worldwide.
        delta = 1.2
        params["viewbox"] = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
        params["bounded"] = 0
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "MechatronWeatherHub/1.0"},
            timeout=6,
        )
        response.raise_for_status()
        rows = response.json()
    except Exception:
        return []
    hits: list[tuple[float, float, str]] = []
    for row in rows:
        name = str(row.get("display_name") or text).split(",")
        short = ", ".join(part.strip() for part in name[:4] if part.strip())
        hits.append((float(row["lat"]), float(row["lon"]), short))
    return hits


def _station_popup(label: str, station: dict, weather: dict | None = None) -> str:
    lines = [
        f"<b>{label}</b>",
        str(station.get("site", "")),
        f"{float(station['latitude']):.5f}°N, {float(station['longitude']):.5f}°E",
    ]
    weather = weather or {}
    if weather.get("temp") is not None:
        lines.append(f"Now {float(weather['temp']):.1f} °C")
    if weather.get("rain_24h") is not None:
        lines.append(f"Rain next 24 h: {float(weather['rain_24h']):.1f} mm")
    if weather.get("condition"):
        lines.append(str(weather["condition"]))
    if weather.get("watch"):
        lines.append(f"Status: {weather['watch']}")
    return "<br>".join(part for part in lines if part)


def build_hub_map_html(
    projects: dict[str, dict],
    active_id: str,
    weather: dict | None = None,
    pending: tuple[float, float] | None = None,
    area_radius_km: float | None = None,
) -> str:
    active = projects[active_id]
    station = active["station"]
    lat = float(station["latitude"])
    lon = float(station["longitude"])
    weather = weather or {}
    color = WATCH_COLORS.get(str(weather.get("watch", "")), "#0075FF")
    view_lat, view_lon = (pending if pending is not None else (lat, lon))
    zoom = 15 if pending is not None else 14
    area_km = float(area_radius_km) if area_radius_km is not None else float(station.get("area_radius_km") or 12)
    radius_m = max(500.0, area_km * 1000.0)

    pins = [
        {
            "lat": lat,
            "lon": lon,
            "color": color,
            "radius": 9,
            "popup": _station_popup(str(active["label"]), station, weather),
        }
    ]
    for key, spec in projects.items():
        if key == active_id:
            continue
        other = spec["station"]
        pins.append(
            {
                "lat": float(other["latitude"]),
                "lon": float(other["longitude"]),
                "color": "#9F7AEA",
                "radius": 7,
                "popup": (
                    f"<b>{spec['label']}</b><br>{other.get('site', '')}<br>"
                    "Switch project in the sidebar to edit this pin."
                ),
            }
        )
    if pending is not None:
        plat, plon = pending
        pins.append(
            {
                "lat": plat,
                "lon": plon,
                "color": "#FFB547",
                "radius": 8,
                "popup": f"New pin<br>{plat:.5f}°N, {plon:.5f}°E<br>Not saved yet",
            }
        )

    pending_line = None if pending is None else [[lat, lon], [pending[0], pending[1]]]
    payload = {
        "lat": view_lat,
        "lon": view_lon,
        "zoom": zoom,
        "ringLat": lat,
        "ringLon": lon,
        "radiusM": radius_m,
        "areaKm": area_km,
        "color": color,
        "pins": pins,
        "line": pending_line,
    }
    if not math.isfinite(view_lat) or not math.isfinite(view_lon):
        payload["lat"], payload["lon"] = 25.26305, 92.20109

    spec = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.css"/>
  <style>
    html, body, #map {{ height: 100%; width: 100%; margin: 0; background: #070f24; }}
    .leaflet-container {{ background: #070f24; font-family: Plus Jakarta Sans, Helvetica, Arial, sans-serif; }}
    .leaflet-popup-content-wrapper, .leaflet-popup-tip {{
      background: #0B1437; color: #fff; border: 1px solid #1A1F37;
    }}
    .leaflet-control-attribution {{ background: rgba(11,20,55,.85); color: #A0AEC0; }}
    .leaflet-control-attribution a {{ color: #0075FF; }}
    .leaflet-control-layers {{
      background: #0B1437; color: #fff; border: 1px solid #1A1F37; border-radius: 10px;
    }}
    .leaflet-control-geocoder {{
      background: #0B1437;
      border: 1px solid #1A1F37;
      border-radius: 10px;
      color: #fff;
    }}
    .leaflet-control-geocoder-form input {{
      background: #0B1437;
      color: #fff;
      width: 280px;
      font-size: 13px;
    }}
    .leaflet-control-geocoder-alternatives {{
      background: #0B1437;
      color: #fff;
      border: 1px solid #1A1F37;
      width: 320px;
    }}
    .leaflet-control-geocoder-alternatives a {{ color: #fff; }}
    .leaflet-control-geocoder-alternatives .leaflet-control-geocoder-selected,
    .leaflet-control-geocoder-alternatives a:hover {{ background: #1A1F37; }}
    .leaflet-bar a {{ background: #0B1437; color: #fff; border-bottom-color: #1A1F37; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.js"></script>
  <script>
    const spec = {spec};
    const map = L.map("map", {{ zoomControl: true, scrollWheelZoom: true }})
      .setView([spec.lat, spec.lon], spec.zoom);
    const satellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
      {{
        attribution: "Tiles &copy; Esri",
        maxZoom: 18,
        maxNativeZoom: 16
      }}
    );
    const places = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}",
      {{ attribution: "Labels &copy; Esri", maxZoom: 18, maxNativeZoom: 15, pane: "overlayPane" }}
    );
    const roads = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{{z}}/{{y}}/{{x}}",
      {{ attribution: "Roads &copy; Esri", maxZoom: 18, maxNativeZoom: 15, pane: "overlayPane" }}
    );
    const satelliteView = L.layerGroup([satellite, roads, places]);
    const streets = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}",
      {{ attribution: "Tiles &copy; Esri", maxZoom: 18, maxNativeZoom: 17 }}
    );
    const voyager = L.tileLayer(
      "https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png",
      {{
        attribution: "&copy; OpenStreetMap &copy; CARTO",
        subdomains: "abcd",
        maxZoom: 18,
        maxNativeZoom: 18
      }}
    );
    streets.addTo(map);
    L.control.layers(
      {{ "Detailed streets": streets, "Satellite": satelliteView, "Town names": voyager }}
    ).addTo(map);
    L.control.scale({{ metric: true, imperial: false }}).addTo(map);
    L.Control.geocoder({{
      defaultMarkGeocode: true,
      collapsed: false,
      placeholder: "Search a place...",
      errorMessage: "No place found",
      suggestMinLength: 3,
      suggestTimeout: 250,
      geocoder: L.Control.Geocoder.photon({{
        geocodingQueryParams: {{ lat: spec.lat, lon: spec.lon, limit: 8 }}
      }})
    }}).on("markgeocode", function (event) {{
      const found = event.geocode;
      if (found.bbox && typeof found.bbox.pad === "function") {{
        map.fitBounds(found.bbox.pad(0.35), {{ maxZoom: 15 }});
      }} else if (found.center) {{
        map.setView(found.center, 15);
      }}
    }}).addTo(map);
    L.circle([spec.ringLat, spec.ringLon], {{
      radius: spec.radiusM,
      color: spec.color,
      weight: 2,
      fillColor: spec.color,
      fillOpacity: 0.12
    }}).bindPopup("Wider forecast area (about " + spec.areaKm + " km)").addTo(map);
    if (spec.line) {{
      L.polyline(spec.line, {{ color: "#FFB547", weight: 2, dashArray: "6 6" }}).addTo(map);
    }}
    spec.pins.forEach((pin) => {{
      L.circleMarker([pin.lat, pin.lon], {{
        radius: pin.radius,
        color: "#FFFFFF",
        weight: 2,
        fillColor: pin.color,
        fillOpacity: 1
      }}).bindPopup(pin.popup).addTo(map);
    }});
    function fit() {{ map.invalidateSize(); }}
    setTimeout(fit, 80);
    setTimeout(fit, 400);
    window.addEventListener("resize", fit);
  </script>
</body>
</html>
"""
