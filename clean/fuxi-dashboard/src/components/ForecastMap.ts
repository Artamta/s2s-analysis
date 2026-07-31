import { colorFor, formatValue } from "../lib/color";
import type {
  ForecastData,
  ForecastWeek,
  OutlineData,
  ProductDefinition,
  ProductKey,
} from "../types";

const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_SIZE = 620;
const LON_MIN = 59.25;
const LON_MAX = 99.75;
const LAT_MIN = -0.75;
const LAT_MAX = 39.75;

function svgElement<K extends keyof SVGElementTagNameMap>(
  tag: K,
): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NS, tag);
}

function mapX(longitude: number): number {
  return ((longitude - LON_MIN) / (LON_MAX - LON_MIN)) * MAP_SIZE;
}

function mapY(latitude: number): number {
  return ((LAT_MAX - latitude) / (LAT_MAX - LAT_MIN)) * MAP_SIZE;
}

type Coordinates = number[][] | number[][][] | number[][][][];

function ringPath(ring: number[][]): string {
  return ring
    .map(([longitude, latitude], index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${mapX(longitude).toFixed(2)},${mapY(latitude).toFixed(2)}`;
    })
    .join(" ")
    .concat(" Z");
}

function polygonPath(polygon: number[][][]): string {
  return polygon.map(ringPath).join(" ");
}

function geometryPath(type: string, coordinates: unknown): string {
  if (type === "Polygon") {
    return polygonPath(coordinates as number[][][]);
  }
  if (type === "MultiPolygon") {
    return (coordinates as number[][][][]).map(polygonPath).join(" ");
  }
  return "";
}

export class ForecastMap {
  private readonly container: HTMLElement;
  private readonly forecast: ForecastData;
  private readonly outline: OutlineData;
  private readonly tooltip: HTMLDivElement;

  constructor(
    container: HTMLElement,
    forecast: ForecastData,
    outline: OutlineData,
  ) {
    this.container = container;
    this.forecast = forecast;
    this.outline = outline;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "map-tooltip";
    this.tooltip.hidden = true;
    this.container.append(this.tooltip);
  }

  render(
    productKey: ProductKey,
    week: ForecastWeek,
    product: ProductDefinition,
  ): void {
    this.container.querySelector("svg")?.remove();
    const svg = svgElement("svg");
    svg.setAttribute("viewBox", `0 0 ${MAP_SIZE} ${MAP_SIZE}`);
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      `${product.label} map for Week ${week.week}, ${week.valid_start} through ${week.valid_end}`,
    );
    svg.classList.add("forecast-map");

    const ocean = svgElement("rect");
    ocean.setAttribute("width", String(MAP_SIZE));
    ocean.setAttribute("height", String(MAP_SIZE));
    ocean.setAttribute("class", "map-ocean");
    svg.append(ocean);
    this.addGraticules(svg);
    this.addCells(svg, productKey, week, product);
    this.addOutline(svg);
    this.container.prepend(svg);
  }

  private addGraticules(svg: SVGSVGElement): void {
    for (const longitude of [60, 70, 80, 90]) {
      const line = svgElement("line");
      line.setAttribute("x1", String(mapX(longitude)));
      line.setAttribute("x2", String(mapX(longitude)));
      line.setAttribute("y1", "0");
      line.setAttribute("y2", String(MAP_SIZE));
      line.setAttribute("class", "map-graticule");
      svg.append(line);

      const label = svgElement("text");
      label.setAttribute("x", String(mapX(longitude) + 5));
      label.setAttribute("y", String(MAP_SIZE - 10));
      label.setAttribute("class", "map-coordinate");
      label.textContent = `${longitude}°E`;
      svg.append(label);
    }
    for (const latitude of [0, 10, 20, 30, 40]) {
      const line = svgElement("line");
      line.setAttribute("x1", "0");
      line.setAttribute("x2", String(MAP_SIZE));
      line.setAttribute("y1", String(mapY(latitude)));
      line.setAttribute("y2", String(mapY(latitude)));
      line.setAttribute("class", "map-graticule");
      svg.append(line);

      if (latitude < 40) {
        const label = svgElement("text");
        label.setAttribute("x", "8");
        label.setAttribute("y", String(mapY(latitude) - 7));
        label.setAttribute("class", "map-coordinate");
        label.textContent = latitude === 0 ? "0°" : `${latitude}°N`;
        svg.append(label);
      }
    }
  }

  private addCells(
    svg: SVGSVGElement,
    productKey: ProductKey,
    week: ForecastWeek,
    product: ProductDefinition,
  ): void {
    const values = week.fields[productKey];
    const { latitude, longitude, india_mask: mask } = this.forecast.grid;
    const cellSize = mapX(LON_MIN + 1.5) - mapX(LON_MIN);

    latitude.forEach((lat, latIndex) => {
      longitude.forEach((lon, lonIndex) => {
        const flatIndex = latIndex * longitude.length + lonIndex;
        if (!mask[flatIndex]) return;
        const value = values[flatIndex];
        const cell = svgElement("rect");
        cell.setAttribute("x", String(mapX(lon - 0.75)));
        cell.setAttribute("y", String(mapY(lat + 0.75)));
        cell.setAttribute("width", String(cellSize + 0.3));
        cell.setAttribute("height", String(cellSize + 0.3));
        cell.setAttribute("fill", colorFor(value, product.legend));
        cell.setAttribute("class", "map-cell");
        cell.setAttribute("tabindex", "0");
        cell.setAttribute(
          "aria-label",
          `${lat.toFixed(1)} degrees north, ${lon.toFixed(1)} degrees east: ${formatValue(value, product.units)}`,
        );
        const show = (event: PointerEvent | FocusEvent): void => {
          this.tooltip.innerHTML = `
            <strong>${formatValue(value, product.units)}</strong>
            <span>${lat.toFixed(1)}°N · ${lon.toFixed(1)}°E</span>
            <small>Native 1.5° grid cell</small>
          `;
          this.tooltip.hidden = false;
          if (event instanceof PointerEvent) {
            const bounds = this.container.getBoundingClientRect();
            this.tooltip.style.left = `${event.clientX - bounds.left + 14}px`;
            this.tooltip.style.top = `${event.clientY - bounds.top + 14}px`;
          } else {
            this.tooltip.style.left = "16px";
            this.tooltip.style.top = "16px";
          }
        };
        cell.addEventListener("pointermove", show);
        cell.addEventListener("pointerenter", show);
        cell.addEventListener("focus", show);
        cell.addEventListener("pointerleave", () => {
          this.tooltip.hidden = true;
        });
        cell.addEventListener("blur", () => {
          this.tooltip.hidden = true;
        });
        svg.append(cell);
      });
    });
  }

  private addOutline(svg: SVGSVGElement): void {
    const geometry = this.outline.geometry;
    if (geometry.coordinates) {
      const path = svgElement("path");
      path.setAttribute(
        "d",
        geometryPath(geometry.type, geometry.coordinates as Coordinates),
      );
      path.setAttribute("class", "map-outline");
      path.setAttribute("fill-rule", "evenodd");
      svg.append(path);
    }
  }
}
