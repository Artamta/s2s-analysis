import { colorFor } from "../lib/color";
import type {
  GlobalForecastData,
  GlobalVariableKey,
  WorldCountriesData,
} from "../types";

const GRID_HEIGHT = 121;
const GRID_WIDTH = 240;
const FRAME_SIZE = GRID_HEIGHT * GRID_WIDTH;
const CONTOUR_LEVELS = [480, 500, 520, 540, 560, 580, 600];

interface MapRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MapSelection {
  latitude: number;
  longitude: number;
  latitudeIndex: number;
  longitudeIndex: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatMapValue(value: number, units: string): string {
  const decimals = units === "mm/day" ? (value < 10 ? 1 : 0) : 1;
  return `${value.toFixed(decimals)} ${units}`;
}

export class GlobalForecastMap {
  private readonly container: HTMLElement;
  private readonly data: GlobalForecastData;
  private readonly world: WorldCountriesData;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly tooltip: HTMLDivElement;
  private readonly resizeObserver: ResizeObserver;
  private fieldA: HTMLCanvasElement | null = null;
  private fieldB: HTMLCanvasElement | null = null;
  private cachedVariable: GlobalVariableKey | null = null;
  private cachedDay = -1;
  private contourPath: Path2D | null = null;
  private currentVariable: GlobalVariableKey = "precipitation";
  private currentDay = 0;
  private currentMix = 0;
  private mapRect: MapRect = { x: 0, y: 0, width: 1, height: 1 };
  private selection: MapSelection = {
    latitude: 28.5,
    longitude: 76.5,
    latitudeIndex: 41,
    longitudeIndex: 51,
  };

  constructor(
    container: HTMLElement,
    data: GlobalForecastData,
    world: WorldCountriesData,
    onSelect?: (selection: MapSelection) => void,
  ) {
    this.container = container;
    this.data = data;
    this.world = world;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "global-map-canvas";
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute(
      "aria-label",
      "Animated global experimental ensemble forecast",
    );
    const context = this.canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is not available");
    this.context = context;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "global-map-tooltip";
    this.tooltip.hidden = true;
    this.container.append(this.canvas, this.tooltip);

    this.canvas.addEventListener("pointermove", (event) =>
      this.showTooltip(event),
    );
    this.canvas.addEventListener("click", (event) => {
      const selection = this.selectionAtPointer(event);
      if (!selection) return;
      this.selection = selection;
      onSelect?.(selection);
      this.draw();
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.tooltip.hidden = true;
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
  }

  destroy(): void {
    this.resizeObserver.disconnect();
  }

  render(variable: GlobalVariableKey, day: number, mix = 0): void {
    const lastDay = this.data.metadata.issue.lead_days - 1;
    this.currentVariable = variable;
    this.currentDay = clamp(Math.round(day), 0, lastDay);
    this.currentMix = clamp(mix, 0, 1);
    if (
      this.cachedVariable !== this.currentVariable ||
      this.cachedDay !== this.currentDay
    ) {
      this.fieldA = this.buildFieldCanvas(this.currentVariable, this.currentDay);
      this.fieldB = this.buildFieldCanvas(
        this.currentVariable,
        Math.min(this.currentDay + 1, lastDay),
      );
      this.cachedVariable = this.currentVariable;
      this.cachedDay = this.currentDay;
      this.contourPath =
        this.currentVariable === "z500"
          ? this.buildContourPath(this.currentDay)
          : null;
    }
    this.draw();
  }

  private resize(): void {
    const bounds = this.container.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, Math.round(bounds.width));
    const height = Math.max(360, Math.round(bounds.height));
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);

    let mapWidth = width;
    let mapHeight = mapWidth / 2;
    if (mapHeight > height) {
      mapHeight = height;
      mapWidth = mapHeight * 2;
    }
    this.mapRect = {
      x: (width - mapWidth) / 2,
      y: (height - mapHeight) / 2,
      width: mapWidth,
      height: mapHeight,
    };
    this.contourPath =
      this.currentVariable === "z500"
        ? this.buildContourPath(this.currentDay)
        : null;
    this.draw();
  }

  private valueAt(
    variable: GlobalVariableKey,
    day: number,
    latitudeIndex: number,
    longitudeIndex: number,
  ): number {
    const definition = this.data.metadata.variables[variable];
    const encoded =
      this.data.fields[variable][
        day * FRAME_SIZE + latitudeIndex * GRID_WIDTH + longitudeIndex
      ];
    return encoded * definition.scale + definition.offset;
  }

  private buildFieldCanvas(
    variable: GlobalVariableKey,
    day: number,
  ): HTMLCanvasElement {
    const offscreen = document.createElement("canvas");
    offscreen.width = GRID_WIDTH;
    offscreen.height = GRID_HEIGHT;
    const context = offscreen.getContext("2d");
    if (!context) throw new Error("Offscreen Canvas 2D is not available");
    const image = context.createImageData(GRID_WIDTH, GRID_HEIGHT);
    const definition = this.data.metadata.variables[variable];

    for (let latitudeIndex = 0; latitudeIndex < GRID_HEIGHT; latitudeIndex += 1) {
      for (
        let displayLongitudeIndex = 0;
        displayLongitudeIndex < GRID_WIDTH;
        displayLongitudeIndex += 1
      ) {
        const sourceLongitudeIndex = (displayLongitudeIndex + 120) % GRID_WIDTH;
        const value = this.valueAt(
          variable,
          day,
          latitudeIndex,
          sourceLongitudeIndex,
        );
        const color = colorFor(value, definition.legend);
        const rgb = this.hexToRgb(color);
        const pixel =
          (latitudeIndex * GRID_WIDTH + displayLongitudeIndex) * 4;
        image.data[pixel] = rgb[0];
        image.data[pixel + 1] = rgb[1];
        image.data[pixel + 2] = rgb[2];
        image.data[pixel + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    return offscreen;
  }

  private hexToRgb(color: string): [number, number, number] {
    const normalized = color.startsWith("#") ? color.slice(1) : color;
    return [
      Number.parseInt(normalized.slice(0, 2), 16),
      Number.parseInt(normalized.slice(2, 4), 16),
      Number.parseInt(normalized.slice(4, 6), 16),
    ];
  }

  private draw(): void {
    const bounds = this.container.getBoundingClientRect();
    const width = Math.max(320, bounds.width);
    const height = Math.max(360, bounds.height);
    const { x, y, width: mapWidth, height: mapHeight } = this.mapRect;
    const context = this.context;
    context.clearRect(0, 0, width, height);

    const background = context.createRadialGradient(
      width * 0.52,
      height * 0.46,
      20,
      width * 0.52,
      height * 0.46,
      width * 0.72,
    );
    background.addColorStop(0, "#102e37");
    background.addColorStop(1, "#051015");
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    if (this.fieldA && this.fieldB) {
      context.save();
      context.beginPath();
      context.rect(x, y, mapWidth, mapHeight);
      context.clip();
      context.imageSmoothingEnabled = true;
      context.globalAlpha = 1;
      context.drawImage(this.fieldA, x, y, mapWidth, mapHeight);
      if (this.currentMix > 0) {
        context.globalAlpha = this.currentMix;
        context.drawImage(this.fieldB, x, y, mapWidth, mapHeight);
      }
      context.globalAlpha = 1;

      const shade = context.createLinearGradient(0, y, 0, y + mapHeight);
      shade.addColorStop(0, "rgb(1 11 17 / 28%)");
      shade.addColorStop(0.5, "rgb(1 11 17 / 0%)");
      shade.addColorStop(1, "rgb(1 11 17 / 24%)");
      context.fillStyle = shade;
      context.fillRect(x, y, mapWidth, mapHeight);
      context.restore();
    }

    this.drawGraticules();
    if (this.currentVariable === "z500") this.drawContours();
    this.drawCountries();
    this.drawSelection();

    context.strokeStyle = "rgb(194 225 221 / 30%)";
    context.lineWidth = 1;
    context.strokeRect(x + 0.5, y + 0.5, mapWidth - 1, mapHeight - 1);
  }

  private project(longitude: number, latitude: number): [number, number] {
    return [
      this.mapRect.x + ((longitude + 180) / 360) * this.mapRect.width,
      this.mapRect.y + ((90 - latitude) / 180) * this.mapRect.height,
    ];
  }

  private drawSelection(): void {
    const [x, y] = this.project(
      this.selection.longitude > 180
        ? this.selection.longitude - 360
        : this.selection.longitude,
      this.selection.latitude,
    );
    const context = this.context;
    context.save();
    context.shadowColor = "rgb(3 15 19 / 80%)";
    context.shadowBlur = 8;
    context.fillStyle = "#f7f3e5";
    context.beginPath();
    context.arc(x, y, 4.5, 0, Math.PI * 2);
    context.fill();
    context.shadowBlur = 0;
    context.strokeStyle = "#72c9b3";
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(x, y, 8, 0, Math.PI * 2);
    context.stroke();
    context.restore();
  }

  private drawGraticules(): void {
    const context = this.context;
    context.save();
    context.strokeStyle = "rgb(217 239 235 / 13%)";
    context.lineWidth = 1;
    context.setLineDash([2, 5]);
    for (let longitude = -150; longitude <= 150; longitude += 30) {
      const [x1, y1] = this.project(longitude, 90);
      const [, y2] = this.project(longitude, -90);
      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x1, y2);
      context.stroke();
    }
    for (let latitude = -60; latitude <= 60; latitude += 30) {
      const [x1, y1] = this.project(-180, latitude);
      const [x2] = this.project(180, latitude);
      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x2, y1);
      context.stroke();
    }
    context.restore();
  }

  private drawCountries(): void {
    const context = this.context;
    context.save();
    context.strokeStyle = "rgb(236 248 241 / 62%)";
    context.lineWidth = 0.75;
    context.lineJoin = "round";

    const drawRing = (ring: number[][]): void => {
      context.beginPath();
      let previousLongitude: number | null = null;
      ring.forEach(([longitude, latitude], index) => {
        const [x, y] = this.project(longitude, latitude);
        const split =
          previousLongitude !== null &&
          Math.abs(longitude - previousLongitude) > 180;
        if (index === 0 || split) context.moveTo(x, y);
        else context.lineTo(x, y);
        previousLongitude = longitude;
      });
      context.stroke();
    };

    this.world.features.forEach((feature) => {
      if (!feature.geometry) return;
      if (feature.geometry.type === "Polygon") {
        (feature.geometry.coordinates as number[][][]).forEach(drawRing);
      } else {
        (feature.geometry.coordinates as number[][][][]).forEach((polygon) =>
          polygon.forEach(drawRing),
        );
      }
    });
    context.restore();
  }

  private displayGridValue(day: number, x: number, y: number): number {
    const sourceLongitudeIndex = (x + 120) % GRID_WIDTH;
    return this.valueAt(this.currentVariable, day, y, sourceLongitudeIndex);
  }

  private drawContours(): void {
    if (!this.contourPath) {
      this.contourPath = this.buildContourPath(this.currentDay);
    }
    const context = this.context;
    context.save();
    context.strokeStyle = "rgb(248 249 229 / 65%)";
    context.lineWidth = 0.7;
    context.stroke(this.contourPath);
    context.restore();
  }

  private buildContourPath(day: number): Path2D {
    const path = new Path2D();
    const cellWidth = this.mapRect.width / GRID_WIDTH;
    const cellHeight = this.mapRect.height / (GRID_HEIGHT - 1);
    const pointOnEdge = (
      edge: number,
      x: number,
      y: number,
      level: number,
      tl: number,
      tr: number,
      br: number,
      bl: number,
    ): [number, number] => {
      const interpolate = (a: number, b: number): number =>
        clamp((level - a) / (b - a || 1), 0, 1);
      if (edge === 0) return [x + interpolate(tl, tr), y];
      if (edge === 1) return [x + 1, y + interpolate(tr, br)];
      if (edge === 2) return [x + interpolate(bl, br), y + 1];
      return [x, y + interpolate(tl, bl)];
    };
    const segments: Record<number, number[][]> = {
      1: [[3, 0]],
      2: [[0, 1]],
      3: [[3, 1]],
      4: [[1, 2]],
      5: [
        [3, 0],
        [1, 2],
      ],
      6: [[0, 2]],
      7: [[3, 2]],
      8: [[2, 3]],
      9: [[0, 2]],
      10: [
        [0, 1],
        [2, 3],
      ],
      11: [[1, 2]],
      12: [[1, 3]],
      13: [[0, 1]],
      14: [[3, 0]],
    };

    for (const level of CONTOUR_LEVELS) {
      for (let y = 0; y < GRID_HEIGHT - 1; y += 1) {
        for (let x = 0; x < GRID_WIDTH - 1; x += 1) {
          const tl = this.displayGridValue(day, x, y);
          const tr = this.displayGridValue(day, x + 1, y);
          const br = this.displayGridValue(day, x + 1, y + 1);
          const bl = this.displayGridValue(day, x, y + 1);
          const code =
            (tl >= level ? 1 : 0) |
            (tr >= level ? 2 : 0) |
            (br >= level ? 4 : 0) |
            (bl >= level ? 8 : 0);
          (segments[code] ?? []).forEach(([edgeA, edgeB]) => {
            const a = pointOnEdge(edgeA, x, y, level, tl, tr, br, bl);
            const b = pointOnEdge(edgeB, x, y, level, tl, tr, br, bl);
            path.moveTo(
              this.mapRect.x + a[0] * cellWidth,
              this.mapRect.y + a[1] * cellHeight,
            );
            path.lineTo(
              this.mapRect.x + b[0] * cellWidth,
              this.mapRect.y + b[1] * cellHeight,
            );
          });
        }
      }
    }
    return path;
  }

  private showTooltip(event: PointerEvent): void {
    const bounds = this.canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const selection = this.selectionAtPointer(event);
    if (!selection) {
      this.tooltip.hidden = true;
      return;
    }
    const definition =
      this.data.metadata.variables[this.currentVariable];
    const value = this.valueAt(
      this.currentVariable,
      this.currentDay,
      selection.latitudeIndex,
      selection.longitudeIndex,
    );
    const displayLongitude =
      selection.longitude > 180
        ? selection.longitude - 360
        : selection.longitude;
    const longitudeLabel =
      displayLongitude < 0
        ? `${Math.abs(displayLongitude).toFixed(1)}°W`
        : `${displayLongitude.toFixed(1)}°E`;
    const latitudeLabel =
      selection.latitude < 0
        ? `${Math.abs(selection.latitude).toFixed(1)}°S`
        : `${selection.latitude.toFixed(1)}°N`;
    this.tooltip.innerHTML = `
      <span>${definition.short_label}</span>
      <strong>${formatMapValue(value, definition.units)}</strong>
      <small>${latitudeLabel} · ${longitudeLabel} · Lead ${String(this.currentDay + 1).padStart(2, "0")} · Click to inspect</small>
    `;
    this.tooltip.hidden = false;
    this.tooltip.style.left = `${clamp(x + 18, 10, bounds.width - 205)}px`;
    this.tooltip.style.top = `${clamp(y + 18, 10, bounds.height - 105)}px`;
  }

  private selectionAtPointer(event: PointerEvent): MapSelection | null {
    const bounds = this.canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const rect = this.mapRect;
    if (
      x < rect.x ||
      x > rect.x + rect.width ||
      y < rect.y ||
      y > rect.y + rect.height
    ) {
      return null;
    }
    const longitude = ((x - rect.x) / rect.width) * 360 - 180;
    const latitude = 90 - ((y - rect.y) / rect.height) * 180;
    const sourceLongitude = longitude < 0 ? longitude + 360 : longitude;
    const longitudeIndex = Math.round(sourceLongitude / 1.5) % GRID_WIDTH;
    const latitudeIndex = clamp(
      Math.round((90 - latitude) / 1.5),
      0,
      GRID_HEIGHT - 1,
    );
    return {
      latitude: 90 - latitudeIndex * 1.5,
      longitude: longitudeIndex * 1.5,
      latitudeIndex,
      longitudeIndex,
    };
  }
}
