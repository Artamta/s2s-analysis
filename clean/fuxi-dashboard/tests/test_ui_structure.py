"""Focused source-level contracts for the lightweight TypeScript UI."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_forecast_map_places_coordinate_labels_outside_the_plot() -> None:
    forecast_map = read("src/components/ForecastMap.ts")
    assert "const AXIS_LEFT" in forecast_map
    assert "const AXIS_BOTTOM" in forecast_map
    assert 'class", "map-coordinate map-coordinate--longitude"' in forecast_map
    assert 'class", "map-coordinate map-coordinate--latitude"' in forecast_map
    assert "AXIS_LEFT + x" in forecast_map
    assert "this.mapHeight + 20" in forecast_map


def test_wind_streamlines_are_default_with_an_arrow_alternative() -> None:
    forecast_map = read("src/components/ForecastMap.ts")
    forecast_page = read("src/pages/forecast.ts")
    assert 'export type WindRenderingMode = "streamlines" | "arrows"' in forecast_map
    assert 'windRenderingMode: WindRenderingMode = "streamlines"' in forecast_map
    assert "traceStreamlineDirection(" in forecast_map
    assert "sampleWindVector(" in forecast_map
    assert 'class", "map-wind-streamline map-wind-vector"' in forecast_map
    assert 'data-wind-mode="streamlines" class="is-active"' in forecast_page
    assert 'data-wind-mode="arrows"' in forecast_page
    assert 'windModeControl.hidden = selectedProduct !== "wind850_anomaly"' in forecast_page
    assert "map.render(selectedProduct, week, product, windRenderingMode)" in forecast_page
