import { colorFor, formatValue } from "../lib/color";
import type {
  ForecastData,
  ForecastWeek,
  IndiaMapGeographyData,
  ProductDefinition,
  ProductKey,
} from "../types";

const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_WIDTH = 620;
const AXIS_LEFT = 42;
const AXIS_BOTTOM = 30;
const WIND_VECTOR_STRIDE = 3;
const WIND_REFERENCE_SPEED = 5;
const WIND_REFERENCE_LENGTH = 32;
const WIND_MINIMUM_SPEED = 0.2;
const STREAMLINE_SEED_SPACING = 52;
const STREAMLINE_STEP_LENGTH = 6;
const STREAMLINE_MAX_STEPS = 110;
const STREAMLINE_OCCUPANCY_SIZE = 18;

let forecastMapSequence = 0;

export type WindRenderingMode = "streamlines" | "arrows";

interface Point {
  x: number;
  y: number;
}

interface WindVector {
  u: number;
  v: number;
}

interface WindVectorGrid {
  latitude: number[];
  longitude: number[];
  u: number[];
  v: number[];
}

function svgElement<K extends keyof SVGElementTagNameMap>(
  tag: K,
): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NS, tag);
}

function graticuleValues(minimum: number, maximum: number): number[] {
  const span = maximum - minimum;
  const steps = [1, 2, 5, 10, 20, 30, 45, 90];
  const step = steps.find((candidate) => span / candidate <= 5) ?? 90;
  const first = Math.ceil(minimum / step) * step;
  const values: number[] = [];
  for (let value = first; value <= maximum; value += step) values.push(value);
  return values;
}

function interpolateGridValue(
  values: number[],
  row: number,
  column: number,
  rowCount: number,
  columnCount: number,
): number | undefined {
  if (
    rowCount < 2 ||
    columnCount < 2 ||
    row < 0 ||
    row > rowCount - 1 ||
    column < 0 ||
    column > columnCount - 1
  ) {
    return undefined;
  }
  const row0 = Math.min(rowCount - 2, Math.floor(row));
  const column0 = Math.min(columnCount - 2, Math.floor(column));
  const rowWeight = row - row0;
  const columnWeight = column - column0;
  const topLeft = values[row0 * columnCount + column0];
  const topRight = values[row0 * columnCount + column0 + 1];
  const bottomLeft = values[(row0 + 1) * columnCount + column0];
  const bottomRight = values[(row0 + 1) * columnCount + column0 + 1];
  if (![topLeft, topRight, bottomLeft, bottomRight].every(Number.isFinite)) {
    return undefined;
  }
  const top = topLeft * (1 - columnWeight) + topRight * columnWeight;
  const bottom = bottomLeft * (1 - columnWeight) + bottomRight * columnWeight;
  return top * (1 - rowWeight) + bottom * rowWeight;
}

function sampleWindVector(
  grid: WindVectorGrid,
  point: Point,
  mapHeight: number,
  longitudeMinimum: number,
  longitudeMaximum: number,
  latitudeMinimum: number,
  latitudeMaximum: number,
): WindVector | undefined {
  const longitude = longitudeMinimum +
    (point.x / MAP_WIDTH) * (longitudeMaximum - longitudeMinimum);
  const latitude = latitudeMaximum -
    (point.y / mapHeight) * (latitudeMaximum - latitudeMinimum);
  const longitudeStep =
    (grid.longitude.at(-1)! - grid.longitude[0]) / (grid.longitude.length - 1);
  const latitudeStep =
    (grid.latitude.at(-1)! - grid.latitude[0]) / (grid.latitude.length - 1);
  if (!Number.isFinite(longitudeStep) || !Number.isFinite(latitudeStep)) {
    return undefined;
  }
  const column = Math.max(
    0,
    Math.min(
      grid.longitude.length - 1,
      (longitude - grid.longitude[0]) / longitudeStep,
    ),
  );
  const row = Math.max(
    0,
    Math.min(
      grid.latitude.length - 1,
      (latitude - grid.latitude[0]) / latitudeStep,
    ),
  );
  const u = interpolateGridValue(
    grid.u,
    row,
    column,
    grid.latitude.length,
    grid.longitude.length,
  );
  const v = interpolateGridValue(
    grid.v,
    row,
    column,
    grid.latitude.length,
    grid.longitude.length,
  );
  return u === undefined || v === undefined ? undefined : { u, v };
}

function pointInsideMap(point: Point, mapHeight: number): boolean {
  return (
    point.x >= 0 &&
    point.x <= MAP_WIDTH &&
    point.y >= 0 &&
    point.y <= mapHeight
  );
}

function occupancyKey(point: Point): string {
  return `${Math.floor(point.x / STREAMLINE_OCCUPANCY_SIZE)}:${Math.floor(point.y / STREAMLINE_OCCUPANCY_SIZE)}`;
}

function traceStreamlineDirection(
  seed: Point,
  direction: -1 | 1,
  grid: WindVectorGrid,
  mapHeight: number,
  longitudeMinimum: number,
  longitudeMaximum: number,
  latitudeMinimum: number,
  latitudeMaximum: number,
  occupied: ReadonlySet<string>,
): Point[] {
  const points = [seed];
  const visited = new Set<string>([occupancyKey(seed)]);
  let previousCell = occupancyKey(seed);

  const projectedDirection = (point: Point): Point | undefined => {
    const vector = sampleWindVector(
      grid,
      point,
      mapHeight,
      longitudeMinimum,
      longitudeMaximum,
      latitudeMinimum,
      latitudeMaximum,
    );
    if (!vector) return undefined;
    const magnitude = Math.hypot(vector.u, vector.v);
    if (!Number.isFinite(magnitude) || magnitude < WIND_MINIMUM_SPEED) {
      return undefined;
    }
    const projectedU =
      vector.u * MAP_WIDTH / (longitudeMaximum - longitudeMinimum);
    const projectedV =
      vector.v * mapHeight / (latitudeMaximum - latitudeMinimum);
    const projectedMagnitude = Math.hypot(projectedU, projectedV);
    if (!Number.isFinite(projectedMagnitude) || projectedMagnitude === 0) {
      return undefined;
    }
    return {
      x: direction * projectedU / projectedMagnitude,
      y: direction * -projectedV / projectedMagnitude,
    };
  };

  for (let step = 0; step < STREAMLINE_MAX_STEPS; step += 1) {
    const current = points.at(-1)!;
    const initialDirection = projectedDirection(current);
    if (!initialDirection) break;
    const midpoint = {
      x: current.x + initialDirection.x * STREAMLINE_STEP_LENGTH / 2,
      y: current.y + initialDirection.y * STREAMLINE_STEP_LENGTH / 2,
    };
    if (!pointInsideMap(midpoint, mapHeight)) break;
    const midpointDirection = projectedDirection(midpoint);
    if (!midpointDirection) break;
    const next = {
      x: current.x + midpointDirection.x * STREAMLINE_STEP_LENGTH,
      y: current.y + midpointDirection.y * STREAMLINE_STEP_LENGTH,
    };
    if (!pointInsideMap(next, mapHeight)) break;

    const nextCell = occupancyKey(next);
    if (occupied.has(nextCell)) break;
    if (nextCell !== previousCell && visited.has(nextCell)) break;
    points.push(next);
    visited.add(nextCell);
    previousCell = nextCell;
  }
  return points;
}

function buildWindStreamlines(
  grid: WindVectorGrid,
  mapHeight: number,
  longitudeMinimum: number,
  longitudeMaximum: number,
  latitudeMinimum: number,
  latitudeMaximum: number,
): Point[][] {
  const streamlines: Point[][] = [];
  const occupied = new Set<string>();
  for (
    let y = STREAMLINE_SEED_SPACING / 2;
    y < mapHeight;
    y += STREAMLINE_SEED_SPACING
  ) {
    for (
      let x = STREAMLINE_SEED_SPACING / 2;
      x < MAP_WIDTH;
      x += STREAMLINE_SEED_SPACING
    ) {
      const seed = { x, y };
      if (occupied.has(occupancyKey(seed))) continue;
      const backward = traceStreamlineDirection(
        seed,
        -1,
        grid,
        mapHeight,
        longitudeMinimum,
        longitudeMaximum,
        latitudeMinimum,
        latitudeMaximum,
        occupied,
      );
      const forward = traceStreamlineDirection(
        seed,
        1,
        grid,
        mapHeight,
        longitudeMinimum,
        longitudeMaximum,
        latitudeMinimum,
        latitudeMaximum,
        occupied,
      );
      const points = backward.reverse().concat(forward.slice(1));
      if (points.length < 4) continue;
      streamlines.push(points);
      points.forEach((point) => occupied.add(occupancyKey(point)));
    }
  }
  return streamlines;
}

function pathData(points: Point[]): string {
  return points
    .map((point, index) =>
      `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(" ");
}

export class ForecastMap {
  private readonly container: HTMLElement;
  private readonly forecast: ForecastData;
  private readonly geography: IndiaMapGeographyData;
  private readonly tooltip: HTMLDivElement;
  private readonly longitudeMinimum: number;
  private readonly longitudeMaximum: number;
  private readonly latitudeMinimum: number;
  private readonly latitudeMaximum: number;
  private readonly mapHeight: number;
  private readonly instanceId: number;

  constructor(
    container: HTMLElement,
    forecast: ForecastData,
    geography: IndiaMapGeographyData,
  ) {
    this.container = container;
    this.forecast = forecast;
    this.geography = geography;
    const { latitude, longitude, spacing_degrees: spacing } = forecast.grid;
    this.longitudeMinimum = longitude[0] - spacing / 2;
    this.longitudeMaximum = longitude.at(-1)! + spacing / 2;
    this.latitudeMaximum = latitude[0] + spacing / 2;
    this.latitudeMinimum = latitude.at(-1)! - spacing / 2;
    const aspect =
      (this.latitudeMaximum - this.latitudeMinimum) /
      (this.longitudeMaximum - this.longitudeMinimum);
    this.mapHeight = Math.max(260, Math.min(760, Math.round(MAP_WIDTH * aspect)));
    this.instanceId = ++forecastMapSequence;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "map-tooltip";
    this.tooltip.hidden = true;
    this.container.append(this.tooltip);
  }

  render(
    productKey: ProductKey,
    week: ForecastWeek,
    product: ProductDefinition,
    windRenderingMode: WindRenderingMode = "streamlines",
  ): void {
    this.container.querySelector("svg")?.remove();
    const svg = svgElement("svg");
    svg.setAttribute(
      "viewBox",
      `0 0 ${AXIS_LEFT + MAP_WIDTH} ${this.mapHeight + AXIS_BOTTOM}`,
    );
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      `${product.label} map for Week ${week.week}, ${week.valid_start} through ${week.valid_end}`,
    );
    svg.classList.add("forecast-map");

    const definitions = svgElement("defs");
    const clipPath = svgElement("clipPath");
    const clipPathId = `forecast-map-plot-${this.instanceId}`;
    clipPath.setAttribute("id", clipPathId);
    clipPath.setAttribute("clipPathUnits", "userSpaceOnUse");
    const clipRectangle = svgElement("rect");
    clipRectangle.setAttribute("width", String(MAP_WIDTH));
    clipRectangle.setAttribute("height", String(this.mapHeight));
    clipPath.append(clipRectangle);
    definitions.append(clipPath);
    svg.append(definitions);

    const plot = svgElement("g");
    plot.setAttribute("class", "forecast-map__plot");
    plot.setAttribute("transform", `translate(${AXIS_LEFT} 0)`);
    plot.setAttribute("clip-path", `url(#${clipPathId})`);
    svg.append(plot);

    const ocean = svgElement("rect");
    ocean.setAttribute("width", String(MAP_WIDTH));
    ocean.setAttribute("height", String(this.mapHeight));
    ocean.setAttribute("class", "map-ocean");
    plot.append(ocean);
    this.addInterpolatedField(plot, productKey, week, product);
    this.addWorldBoundaries(plot);
    this.addGraticules(plot, svg);
    if (productKey === "wind850_anomaly") {
      if (windRenderingMode === "arrows") {
        this.addWindVectors(svg, plot, week);
      } else {
        this.addWindStreamlines(svg, plot, week);
      }
    }
    this.addOutline(plot);
    this.addIndiaAdmin(plot);
    this.addInteraction(svg, productKey, week, product);
    this.container.prepend(svg);
  }

  private addWindMarker(svg: SVGSVGElement, markerId: string): void {
    const definitions = svg.querySelector("defs");
    if (!definitions) throw new Error("SVG definitions are unavailable");
    const marker = svgElement("marker");
    marker.setAttribute("id", markerId);
    marker.setAttribute("viewBox", "0 0 8 8");
    marker.setAttribute("refX", "7");
    marker.setAttribute("refY", "4");
    marker.setAttribute("markerWidth", "5");
    marker.setAttribute("markerHeight", "5");
    marker.setAttribute("orient", "auto-start-reverse");
    const head = svgElement("path");
    head.setAttribute("d", "M 0 0 L 8 4 L 0 8 z");
    head.setAttribute("class", "map-wind-arrowhead");
    marker.append(head);
    definitions.append(marker);
  }

  private validWindVectors(week: ForecastWeek): WindVectorGrid | undefined {
    const vectors = week.vectors?.wind850_anomaly;
    if (!vectors) return undefined;
    const { latitude, longitude } = this.forecast.grid;
    const expected = latitude.length * longitude.length;
    if (vectors.u.length !== expected || vectors.v.length !== expected) {
      throw new Error(`wind vectors have an invalid size for Week ${week.week}`);
    }
    return { latitude, longitude, u: vectors.u, v: vectors.v };
  }

  private addWindVectors(
    svg: SVGSVGElement,
    plot: SVGGElement,
    week: ForecastWeek,
  ): void {
    const vectors = this.validWindVectors(week);
    if (!vectors) return;
    const { latitude, longitude } = vectors;
    const markerId = `forecast-map-wind-arrow-${this.instanceId}`;
    this.addWindMarker(svg, markerId);

    const group = svgElement("g");
    group.setAttribute("class", "map-wind-vectors");
    for (let row = 1; row < latitude.length; row += WIND_VECTOR_STRIDE) {
      for (let column = 1; column < longitude.length; column += WIND_VECTOR_STRIDE) {
        const index = row * longitude.length + column;
        const u = vectors.u[index];
        const v = vectors.v[index];
        const magnitude = Math.hypot(u, v);
        if (!Number.isFinite(magnitude) || magnitude < WIND_MINIMUM_SPEED) continue;
        const scale = Math.min(magnitude / WIND_REFERENCE_SPEED, 1.5);
        const length = WIND_REFERENCE_LENGTH * scale;
        const dx = (u / magnitude) * length;
        const dy = (-v / magnitude) * length;
        const x = this.mapX(longitude[column]);
        const y = this.mapY(latitude[row]);
        const line = svgElement("line");
        line.setAttribute("x1", String(x - dx / 2));
        line.setAttribute("y1", String(y - dy / 2));
        line.setAttribute("x2", String(x + dx / 2));
        line.setAttribute("y2", String(y + dy / 2));
        line.setAttribute("marker-end", `url(#${markerId})`);
        line.setAttribute("class", "map-wind-vector");
        group.append(line);
      }
    }
    plot.append(group);

    const key = svgElement("g");
    key.setAttribute("class", "map-wind-key");
    const background = svgElement("rect");
    background.setAttribute("x", "438");
    background.setAttribute("y", "18");
    background.setAttribute("width", "164");
    background.setAttribute("height", "42");
    const keyLine = svgElement("line");
    keyLine.setAttribute("x1", "452");
    keyLine.setAttribute("y1", "39");
    keyLine.setAttribute("x2", String(452 + WIND_REFERENCE_LENGTH));
    keyLine.setAttribute("y2", "39");
    keyLine.setAttribute("marker-end", `url(#${markerId})`);
    keyLine.setAttribute("class", "map-wind-vector");
    const keyLabel = svgElement("text");
    keyLabel.setAttribute("x", "494");
    keyLabel.setAttribute("y", "43");
    keyLabel.textContent = `${WIND_REFERENCE_SPEED} m s⁻¹ anomaly`;
    key.append(background, keyLine, keyLabel);
    plot.append(key);
  }

  private addWindStreamlines(
    svg: SVGSVGElement,
    plot: SVGGElement,
    week: ForecastWeek,
  ): void {
    const vectors = this.validWindVectors(week);
    if (!vectors) return;
    const markerId = `forecast-map-wind-streamline-${this.instanceId}`;
    this.addWindMarker(svg, markerId);
    const streamlines = buildWindStreamlines(
      vectors,
      this.mapHeight,
      this.longitudeMinimum,
      this.longitudeMaximum,
      this.latitudeMinimum,
      this.latitudeMaximum,
    );
    const group = svgElement("g");
    group.setAttribute("class", "map-wind-streamlines map-wind-vectors");
    for (const points of streamlines) {
      const path = svgElement("path");
      path.setAttribute("d", pathData(points));
      path.setAttribute("fill", "none");
      path.setAttribute("class", "map-wind-streamline map-wind-vector");
      group.append(path);

      const midpoint = Math.floor(points.length / 2);
      const start = points[Math.max(0, midpoint - 1)];
      const end = points[Math.min(points.length - 1, midpoint + 1)];
      const direction = svgElement("line");
      direction.setAttribute("x1", start.x.toFixed(1));
      direction.setAttribute("y1", start.y.toFixed(1));
      direction.setAttribute("x2", end.x.toFixed(1));
      direction.setAttribute("y2", end.y.toFixed(1));
      direction.setAttribute("marker-end", `url(#${markerId})`);
      direction.setAttribute(
        "class",
        "map-wind-streamline-direction map-wind-vector",
      );
      group.append(direction);
    }
    plot.append(group);

    const key = svgElement("g");
    key.setAttribute("class", "map-wind-key map-wind-key--streamlines");
    const background = svgElement("rect");
    background.setAttribute("x", "412");
    background.setAttribute("y", "18");
    background.setAttribute("width", "190");
    background.setAttribute("height", "42");
    const keyLine = svgElement("line");
    keyLine.setAttribute("x1", "426");
    keyLine.setAttribute("y1", "39");
    keyLine.setAttribute("x2", "462");
    keyLine.setAttribute("y2", "39");
    keyLine.setAttribute("marker-end", `url(#${markerId})`);
    keyLine.setAttribute("class", "map-wind-streamline-direction map-wind-vector");
    const keyLabel = svgElement("text");
    keyLabel.setAttribute("x", "474");
    keyLabel.setAttribute("y", "43");
    keyLabel.textContent = "Anomaly flow direction";
    key.append(background, keyLine, keyLabel);
    plot.append(key);
  }

  private addInterpolatedField(
    plot: SVGGElement,
    productKey: ProductKey,
    week: ForecastWeek,
    product: ProductDefinition,
  ): void {
    const values = week.fields[productKey];
    if (!values) throw new Error(`${productKey} is unavailable for Week ${week.week}`);
    const { latitude, longitude } = this.forecast.grid;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(2, (longitude.length - 1) * 6);
    canvas.height = Math.max(2, (latitude.length - 1) * 6);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas rendering is unavailable");

    const sample = (row: number, column: number): number => {
      const safeRow = Math.max(0, Math.min(latitude.length - 1, row));
      const safeColumn = Math.max(0, Math.min(longitude.length - 1, column));
      return values[safeRow * longitude.length + safeColumn];
    };

    for (let y = 0; y < canvas.height; y += 1) {
      const lat = this.latitudeMaximum -
        ((y + 0.5) / canvas.height) *
        (this.latitudeMaximum - this.latitudeMinimum);
      const row = (latitude[0] - lat) / this.forecast.grid.spacing_degrees;
      const row0 = Math.floor(row);
      const rowWeight = row - row0;
      for (let x = 0; x < canvas.width; x += 1) {
        const lon = this.longitudeMinimum +
          ((x + 0.5) / canvas.width) *
          (this.longitudeMaximum - this.longitudeMinimum);
        const column = (lon - longitude[0]) / this.forecast.grid.spacing_degrees;
        const column0 = Math.floor(column);
        const columnWeight = column - column0;
        const top =
          sample(row0, column0) * (1 - columnWeight) +
          sample(row0, column0 + 1) * columnWeight;
        const bottom =
          sample(row0 + 1, column0) * (1 - columnWeight) +
          sample(row0 + 1, column0 + 1) * columnWeight;
        const value = top * (1 - rowWeight) + bottom * rowWeight;
        context.fillStyle = colorFor(value, product.legend);
        context.fillRect(x, y, 1, 1);
      }
    }

    const image = svgElement("image");
    image.setAttribute("x", "0");
    image.setAttribute("y", "0");
    image.setAttribute("width", String(MAP_WIDTH));
    image.setAttribute("height", String(this.mapHeight));
    image.setAttribute("preserveAspectRatio", "none");
    image.setAttribute("href", canvas.toDataURL("image/png"));
    image.setAttribute("class", "map-raster");
    plot.append(image);
  }

  private addGraticules(plot: SVGGElement, svg: SVGSVGElement): void {
    const axes = svgElement("g");
    axes.setAttribute("class", "map-coordinate-axes");
    for (const longitude of graticuleValues(
      this.longitudeMinimum,
      this.longitudeMaximum,
    )) {
      const x = this.mapX(longitude);
      const line = svgElement("line");
      line.setAttribute("x1", String(x));
      line.setAttribute("x2", String(x));
      line.setAttribute("y1", "0");
      line.setAttribute("y2", String(this.mapHeight));
      line.setAttribute("class", "map-graticule");
      plot.append(line);

      const label = svgElement("text");
      label.setAttribute("x", String(AXIS_LEFT + x));
      label.setAttribute("y", String(this.mapHeight + 20));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", "map-coordinate map-coordinate--longitude");
      label.textContent = `${Math.abs(longitude)}°${longitude < 0 ? "W" : "E"}`;
      axes.append(label);
    }
    for (const latitude of graticuleValues(
      this.latitudeMinimum,
      this.latitudeMaximum,
    )) {
      const line = svgElement("line");
      line.setAttribute("x1", "0");
      line.setAttribute("x2", String(MAP_WIDTH));
      line.setAttribute("y1", String(this.mapY(latitude)));
      line.setAttribute("y2", String(this.mapY(latitude)));
      line.setAttribute("class", "map-graticule");
      plot.append(line);

      const label = svgElement("text");
      label.setAttribute("x", String(AXIS_LEFT - 8));
      label.setAttribute("y", String(this.mapY(latitude) + 3));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("class", "map-coordinate map-coordinate--latitude");
      label.textContent = `${Math.abs(latitude)}°${latitude < 0 ? "S" : "N"}`;
      axes.append(label);
    }
    svg.append(axes);
  }

  private addInteraction(
    svg: SVGSVGElement,
    productKey: ProductKey,
    week: ForecastWeek,
    product: ProductDefinition,
  ): void {
    const values = week.fields[productKey];
    if (!values) throw new Error(`${productKey} is unavailable for Week ${week.week}`);
    const { latitude, longitude } = this.forecast.grid;
    svg.setAttribute("tabindex", "0");
    svg.classList.add("forecast-map--interactive");
    const interactionSurface = svgElement("rect");
    interactionSurface.setAttribute("x", String(AXIS_LEFT));
    interactionSurface.setAttribute("y", "0");
    interactionSurface.setAttribute("width", String(MAP_WIDTH));
    interactionSurface.setAttribute("height", String(this.mapHeight));
    interactionSurface.setAttribute("fill", "transparent");
    interactionSurface.setAttribute("pointer-events", "all");
    interactionSurface.setAttribute("aria-hidden", "true");
    interactionSurface.setAttribute("class", "map-interaction-surface");
    svg.append(interactionSurface);
    const show = (event?: PointerEvent): void => {
      let latIndex = Math.floor(latitude.length / 2);
      let lonIndex = Math.floor(longitude.length / 2);
      if (event) {
        const bounds = interactionSurface.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
        const y = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
        const hoveredLongitude = this.longitudeMinimum +
          x * (this.longitudeMaximum - this.longitudeMinimum);
        const hoveredLatitude = this.latitudeMaximum -
          y * (this.latitudeMaximum - this.latitudeMinimum);
        lonIndex = Math.max(
          0,
          Math.min(
            longitude.length - 1,
            Math.round((hoveredLongitude - longitude[0]) / this.forecast.grid.spacing_degrees),
          ),
        );
        latIndex = Math.max(
          0,
          Math.min(
            latitude.length - 1,
            Math.round((latitude[0] - hoveredLatitude) / this.forecast.grid.spacing_degrees),
          ),
        );
      }
      const lat = latitude[latIndex];
      const lon = longitude[lonIndex];
      const value = values[latIndex * longitude.length + lonIndex];
      const vector = productKey === "wind850_anomaly"
        ? week.vectors?.wind850_anomaly
        : undefined;
      const vectorIndex = latIndex * longitude.length + lonIndex;
      const vectorU = vector?.u[vectorIndex];
      const vectorV = vector?.v[vectorIndex];
      const vectorDetail =
        typeof vectorU === "number" &&
        typeof vectorV === "number" &&
        Number.isFinite(vectorU) &&
        Number.isFinite(vectorV)
        ? `<small>U ${vectorU.toFixed(1)} · V ${vectorV.toFixed(1)} m s⁻¹</small>`
        : "";
      this.tooltip.innerHTML = `
        <strong>${formatValue(value, product.units)}</strong>
        <span>${lat.toFixed(1)}°N · ${lon.toFixed(1)}°E</span>
        ${vectorDetail}
        <small>Native ${this.forecast.grid.spacing_degrees}° grid cell</small>
      `;
      this.tooltip.hidden = false;
      svg.setAttribute(
        "aria-label",
        `${product.label}, Week ${week.week}; selected cell ${lat.toFixed(1)} degrees north, ${lon.toFixed(1)} degrees east: ${formatValue(value, product.units)}`,
      );
      if (event) {
        const bounds = this.container.getBoundingClientRect();
        this.tooltip.style.left = `${event.clientX - bounds.left + 14}px`;
        this.tooltip.style.top = `${event.clientY - bounds.top + 14}px`;
      } else {
        this.tooltip.style.left = "16px";
        this.tooltip.style.top = "16px";
      }
    };
    interactionSurface.addEventListener("pointermove", (event) => show(event));
    interactionSurface.addEventListener("pointerenter", (event) => show(event));
    svg.addEventListener("focus", () => show());
    interactionSurface.addEventListener("pointerleave", () => {
      this.tooltip.hidden = true;
    });
    svg.addEventListener("blur", () => {
      this.tooltip.hidden = true;
    });
  }

  private mapX(longitude: number): number {
    return (
      ((longitude - this.longitudeMinimum) /
        (this.longitudeMaximum - this.longitudeMinimum)) *
      MAP_WIDTH
    );
  }

  private mapY(latitude: number): number {
    return (
      ((this.latitudeMaximum - latitude) /
        (this.latitudeMaximum - this.latitudeMinimum)) *
      this.mapHeight
    );
  }

  private addWorldBoundaries(plot: SVGGElement): void {
    if (!this.geography.world_path) return;
    const path = svgElement("path");
    path.setAttribute("d", this.geography.world_path);
    path.setAttribute("class", "map-world-outline");
    path.setAttribute("fill-rule", "evenodd");
    plot.append(path);
  }

  private addOutline(plot: SVGGElement): void {
    if (!this.geography.india_outline_path) return;
    const path = svgElement("path");
    path.setAttribute("d", this.geography.india_outline_path);
    path.setAttribute("class", "map-outline");
    path.setAttribute("fill-rule", "evenodd");
    plot.append(path);
  }

  private addIndiaAdmin(plot: SVGGElement): void {
    if (!this.geography.india_admin_path) return;
    const path = svgElement("path");
    path.setAttribute("d", this.geography.india_admin_path);
    path.setAttribute("class", "map-india-admin");
    path.setAttribute("fill-rule", "evenodd");
    plot.append(path);
  }
}
