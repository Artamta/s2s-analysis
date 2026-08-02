import { colorFor } from "../lib/color";
import type {
  GlobalDisplayMode,
  GlobalForecastData,
  GlobalVariableKey,
  IndiaAdminData,
  WorldCountriesData,
} from "../types";

const GRID_HEIGHT = 121;
const GRID_WIDTH = 240;
const FRAME_SIZE = GRID_HEIGHT * GRID_WIDTH;
const Z500_LEVELS = [480, 500, 520, 540, 560, 580, 600];
const MSLP_LEVELS = [980, 990, 1000, 1010, 1020, 1030, 1040];

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
  country?: string;
  state?: string;
}

export interface MapOverlays {
  countryLabels: boolean;
  indiaStates: boolean;
  windVectors: boolean;
  pressureContours: boolean;
  z500Contours: boolean;
}

interface DragState {
  pointerId: number;
  x: number;
  y: number;
  panX: number;
  panY: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatMapValue(value: number, units: string): string {
  const decimals =
    units === "W/m²" || (units === "mm/day" && value >= 10) ? 0 : 1;
  return `${value.toFixed(decimals)} ${units}`;
}

function pointInRing(
  longitude: number,
  latitude: number,
  ring: number[][],
): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [x1, y1] = ring[index];
    const [x2, y2] = ring[previous];
    const crosses =
      y1 > latitude !== y2 > latitude &&
      longitude < ((x2 - x1) * (latitude - y1)) / (y2 - y1 || 1) + x1;
    if (crosses) inside = !inside;
  }
  return inside;
}

function pointInPolygon(
  longitude: number,
  latitude: number,
  polygon: number[][][],
): boolean {
  if (!polygon.length || !pointInRing(longitude, latitude, polygon[0])) {
    return false;
  }
  return !polygon.slice(1).some((ring) =>
    pointInRing(longitude, latitude, ring),
  );
}

function pointInGeometry(
  longitude: number,
  latitude: number,
  geometry: {
    type: "Polygon" | "MultiPolygon";
    coordinates: number[][][] | number[][][][];
  },
): boolean {
  if (geometry.type === "Polygon") {
    return pointInPolygon(
      longitude,
      latitude,
      geometry.coordinates as number[][][],
    );
  }
  return (geometry.coordinates as number[][][][]).some((polygon) =>
    pointInPolygon(longitude, latitude, polygon),
  );
}

export class GlobalForecastMap {
  private readonly container: HTMLElement;
  private readonly data: GlobalForecastData;
  private readonly world: WorldCountriesData;
  private readonly indiaAdmin: IndiaAdminData;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly tooltip: HTMLDivElement;
  private readonly resizeObserver: ResizeObserver;
  private readonly onSelect?: (selection: MapSelection) => void;
  private readonly onViewChange?: (zoom: number) => void;
  private fieldA: HTMLCanvasElement | null = null;
  private fieldB: HTMLCanvasElement | null = null;
  private cachedVariable: GlobalVariableKey | null = null;
  private cachedMode: GlobalDisplayMode | null = null;
  private cachedDay = -1;
  private z500ContourPath: Path2D | null = null;
  private pressureContourPath: Path2D | null = null;
  private currentVariable: GlobalVariableKey = "precipitation";
  private currentMode: GlobalDisplayMode = "absolute";
  private currentDay = 0;
  private currentMix = 0;
  private mapRect: MapRect = { x: 0, y: 0, width: 1, height: 1 };
  private zoom = 1;
  private panX = 0;
  private panY = 0;
  private drag: DragState | null = null;
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private pinchDistance: number | null = null;
  private dragMoved = false;
  private suppressClick = false;
  private overlays: MapOverlays = {
    countryLabels: true,
    indiaStates: false,
    windVectors: false,
    pressureContours: false,
    z500Contours: false,
  };
  private selection: MapSelection = {
    latitude: 28.5,
    longitude: 76.5,
    latitudeIndex: 41,
    longitudeIndex: 51,
    country: "India",
  };

  constructor(
    container: HTMLElement,
    data: GlobalForecastData,
    world: WorldCountriesData,
    indiaAdmin: IndiaAdminData,
    onSelect?: (selection: MapSelection) => void,
    onViewChange?: (zoom: number) => void,
  ) {
    this.container = container;
    this.data = data;
    this.world = world;
    this.indiaAdmin = indiaAdmin;
    this.onSelect = onSelect;
    this.onViewChange = onViewChange;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "global-map-canvas";
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute(
      "aria-label",
      "Zoomable animated global experimental ensemble forecast",
    );
    const context = this.canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is not available");
    this.context = context;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "global-map-tooltip";
    this.tooltip.hidden = true;
    this.container.append(this.canvas, this.tooltip);

    this.canvas.addEventListener("pointerdown", (event) =>
      this.startDrag(event),
    );
    this.canvas.addEventListener("pointermove", (event) =>
      this.movePointer(event),
    );
    this.canvas.addEventListener("pointerup", (event) => this.endDrag(event));
    this.canvas.addEventListener("pointercancel", (event) =>
      this.endDrag(event),
    );
    this.canvas.addEventListener("pointerleave", () => {
      if (!this.drag) this.tooltip.hidden = true;
    });
    this.canvas.addEventListener("click", (event) => {
      if (this.suppressClick) {
        this.suppressClick = false;
        return;
      }
      const selection = this.selectionAtPointer(event);
      if (!selection) return;
      this.selection = selection;
      this.onSelect?.(selection);
      this.draw();
    });
    this.canvas.addEventListener("dblclick", (event) => {
      event.preventDefault();
      this.zoomAt(event.clientX, event.clientY, 1.7);
    });
    this.canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.zoomAt(
          event.clientX,
          event.clientY,
          event.deltaY < 0 ? 1.18 : 1 / 1.18,
        );
      },
      { passive: false },
    );
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
  }

  destroy(): void {
    this.resizeObserver.disconnect();
  }

  render(
    variable: GlobalVariableKey,
    day: number,
    mix = 0,
    mode: GlobalDisplayMode = "absolute",
  ): void {
    if (!this.data.fields[variable]) return;
    const lastDay = this.data.metadata.issue.lead_days - 1;
    this.currentVariable = variable;
    this.currentMode =
      mode === "anomaly" && this.data.anomalies[variable]
        ? "anomaly"
        : "absolute";
    this.currentDay = clamp(Math.round(day), 0, lastDay);
    this.currentMix = clamp(mix, 0, 1);
    if (
      this.cachedVariable !== this.currentVariable ||
      this.cachedMode !== this.currentMode ||
      this.cachedDay !== this.currentDay
    ) {
      this.fieldA = this.buildFieldCanvas(
        this.currentVariable,
        this.currentDay,
        this.currentMode,
      );
      this.fieldB = this.buildFieldCanvas(
        this.currentVariable,
        Math.min(this.currentDay + 1, lastDay),
        this.currentMode,
      );
      this.cachedVariable = this.currentVariable;
      this.cachedMode = this.currentMode;
      this.cachedDay = this.currentDay;
      this.rebuildContours();
    }
    this.draw();
  }

  setOverlays(overlays: MapOverlays): void {
    this.overlays = overlays;
    this.rebuildContours();
    this.draw();
  }

  zoomBy(factor: number): void {
    const bounds = this.canvas.getBoundingClientRect();
    this.zoomAt(
      bounds.left + bounds.width / 2,
      bounds.top + bounds.height / 2,
      factor,
    );
  }

  resetView(): void {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.viewChanged();
  }

  focusIndia(): void {
    this.zoom = 4.1;
    const normalizedX = (80.5 + 180) / 360;
    const normalizedY = (90 - 22.5) / 180;
    this.panX = -(normalizedX - 0.5) * this.mapRect.width * this.zoom;
    this.panY = -(normalizedY - 0.5) * this.mapRect.height * this.zoom;
    this.selection = {
      latitude: 21.0,
      longitude: 78.0,
      latitudeIndex: 46,
      longitudeIndex: 52,
      country: "India",
    };
    this.onSelect?.(this.selection);
    this.constrainPan();
    this.viewChanged();
  }

  private resize(): void {
    const bounds = this.container.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, Math.round(bounds.width));
    const height = Math.max(160, Math.round(bounds.height));
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
    this.constrainPan();
    this.rebuildContours();
    this.draw();
  }

  private valueAt(
    variable: GlobalVariableKey,
    day: number,
    latitudeIndex: number,
    longitudeIndex: number,
  ): number {
    const definition = this.data.metadata.variables[variable];
    const field = this.data.fields[variable];
    if (!field) return Number.NaN;
    const encoded =
      field[
        day * FRAME_SIZE + latitudeIndex * GRID_WIDTH + longitudeIndex
      ];
    return encoded * definition.scale + definition.offset;
  }

  private displayValueAt(
    variable: GlobalVariableKey,
    day: number,
    latitudeIndex: number,
    longitudeIndex: number,
    mode: GlobalDisplayMode = this.currentMode,
  ): number {
    const definition = this.data.metadata.variables[variable];
    const anomalyDefinition = definition.anomaly;
    const anomaly = this.data.anomalies[variable];
    if (mode !== "anomaly" || !anomalyDefinition || !anomaly) {
      return this.valueAt(variable, day, latitudeIndex, longitudeIndex);
    }
    const encoded =
      anomaly[
        day * FRAME_SIZE + latitudeIndex * GRID_WIDTH + longitudeIndex
      ];
    return encoded * anomalyDefinition.scale + anomalyDefinition.offset;
  }

  private vectorValueAt(
    component: "u" | "v",
    day: number,
    latitudeIndex: number,
    longitudeIndex: number,
  ): number {
    const definition = this.data.metadata.variables.wind850.vector?.[component];
    const vector = this.data.vectors.wind850?.[component];
    if (!definition || !vector) return Number.NaN;
    const encoded =
      vector[
        day * FRAME_SIZE + latitudeIndex * GRID_WIDTH + longitudeIndex
      ];
    return encoded * definition.scale + definition.offset;
  }

  private buildFieldCanvas(
    variable: GlobalVariableKey,
    day: number,
    mode: GlobalDisplayMode,
  ): HTMLCanvasElement {
    const offscreen = document.createElement("canvas");
    offscreen.width = GRID_WIDTH;
    offscreen.height = GRID_HEIGHT;
    const context = offscreen.getContext("2d");
    if (!context) throw new Error("Offscreen Canvas 2D is not available");
    const image = context.createImageData(GRID_WIDTH, GRID_HEIGHT);
    const definition = this.data.metadata.variables[variable];
    const displayDefinition =
      mode === "anomaly" && definition.anomaly
        ? definition.anomaly
        : definition;

    for (let latitudeIndex = 0; latitudeIndex < GRID_HEIGHT; latitudeIndex += 1) {
      for (
        let displayLongitudeIndex = 0;
        displayLongitudeIndex < GRID_WIDTH;
        displayLongitudeIndex += 1
      ) {
        const sourceLongitudeIndex = (displayLongitudeIndex + 120) % GRID_WIDTH;
        const sourceIndex =
          latitudeIndex * GRID_WIDTH + sourceLongitudeIndex;
        const isMasked =
          definition.domain === "ocean" &&
          this.data.oceanMask[sourceIndex] !== 1;
        const color = isMasked
          ? "#102329"
          : colorFor(
              this.displayValueAt(
                variable,
                day,
                latitudeIndex,
                sourceLongitudeIndex,
                mode,
              ),
              displayDefinition.legend,
            );
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
    const height = Math.max(160, bounds.height);
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

    context.save();
    context.beginPath();
    context.rect(
      this.mapRect.x,
      this.mapRect.y,
      this.mapRect.width,
      this.mapRect.height,
    );
    context.clip();
    this.drawFields();
    this.drawGraticules();
    if (
      !this.drag &&
      (this.currentVariable === "z500" || this.overlays.z500Contours)
    ) {
      this.drawContour(this.z500ContourPath, "rgb(255 205 112 / 72%)");
    }
    if (!this.drag && this.overlays.pressureContours) {
      this.drawContour(
        this.pressureContourPath,
        "rgb(244 250 239 / 66%)",
        [3, 3],
      );
    }
    if (this.overlays.windVectors) this.drawWindVectors();
    this.drawCountries();
    if (this.overlays.indiaStates) this.drawIndiaAdmin();
    if (this.overlays.countryLabels) this.drawCountryLabels();
    this.drawSelection();
    context.restore();

    context.strokeStyle = "rgb(194 225 221 / 30%)";
    context.lineWidth = 1;
    context.strokeRect(
      this.mapRect.x + 0.5,
      this.mapRect.y + 0.5,
      this.mapRect.width - 1,
      this.mapRect.height - 1,
    );
  }

  private drawFields(): void {
    if (!this.fieldA || !this.fieldB) return;
    const [x, y] = this.project(-180, 90);
    const width = this.mapRect.width * this.zoom;
    const height = this.mapRect.height * this.zoom;
    const context = this.context;
    context.imageSmoothingEnabled = true;
    context.globalAlpha = 1;
    context.drawImage(this.fieldA, x, y, width, height);
    if (this.currentMix > 0) {
      context.globalAlpha = this.currentMix;
      context.drawImage(this.fieldB, x, y, width, height);
    }
    context.globalAlpha = 1;
    const shade = context.createLinearGradient(
      0,
      this.mapRect.y,
      0,
      this.mapRect.y + this.mapRect.height,
    );
    shade.addColorStop(0, "rgb(1 11 17 / 28%)");
    shade.addColorStop(0.5, "rgb(1 11 17 / 0%)");
    shade.addColorStop(1, "rgb(1 11 17 / 24%)");
    context.fillStyle = shade;
    context.fillRect(
      this.mapRect.x,
      this.mapRect.y,
      this.mapRect.width,
      this.mapRect.height,
    );
  }

  private project(longitude: number, latitude: number): [number, number] {
    const normalizedX = (longitude + 180) / 360;
    const normalizedY = (90 - latitude) / 180;
    return [
      this.mapRect.x +
        this.mapRect.width / 2 +
        (normalizedX - 0.5) * this.mapRect.width * this.zoom +
        this.panX,
      this.mapRect.y +
        this.mapRect.height / 2 +
        (normalizedY - 0.5) * this.mapRect.height * this.zoom +
        this.panY,
    ];
  }

  private unproject(x: number, y: number): [number, number] {
    const normalizedX =
      (x -
        this.mapRect.x -
        this.mapRect.width / 2 -
        this.panX) /
        (this.mapRect.width * this.zoom) +
      0.5;
    const normalizedY =
      (y -
        this.mapRect.y -
        this.mapRect.height / 2 -
        this.panY) /
        (this.mapRect.height * this.zoom) +
      0.5;
    return [normalizedX * 360 - 180, 90 - normalizedY * 180];
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

  private drawGeometry(
    geometry: {
      type: "Polygon" | "MultiPolygon";
      coordinates: number[][][] | number[][][][];
    },
    strokeStyle: string,
    lineWidth: number,
  ): void {
    const context = this.context;
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
    context.save();
    context.strokeStyle = strokeStyle;
    context.lineWidth = lineWidth;
    context.lineJoin = "round";
    if (geometry.type === "Polygon") {
      (geometry.coordinates as number[][][]).forEach(drawRing);
    } else {
      (geometry.coordinates as number[][][][]).forEach((polygon) =>
        polygon.forEach(drawRing),
      );
    }
    context.restore();
  }

  private drawCountries(): void {
    this.world.features.forEach((feature) => {
      if (!feature.geometry) return;
      this.drawGeometry(
        feature.geometry,
        "rgb(236 248 241 / 62%)",
        this.zoom > 2 ? 0.9 : 0.7,
      );
    });
  }

  private drawCountryLabels(): void {
    const rankLimit =
      this.zoom < 1.3 ? 2 : this.zoom < 2 ? 3 : this.zoom < 3 ? 5 : 9;
    const context = this.context;
    const occupied: Array<{ x: number; y: number; width: number; height: number }> =
      [];
    context.save();
    context.font = `${this.zoom >= 3 ? 9 : 8}px Inter, system-ui, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "rgb(239 247 241 / 62%)";
    context.shadowColor = "rgb(2 14 18 / 95%)";
    context.shadowBlur = 4;
    const candidates = this.world.features
      .filter((feature) => {
        const properties = feature.properties;
        return (
          properties &&
          Number(properties.LABELRANK ?? 9) <= rankLimit &&
          Number.isFinite(properties.LABEL_X) &&
          Number.isFinite(properties.LABEL_Y)
        );
      })
      .sort(
        (left, right) => {
          const rankDifference =
            Number(left.properties?.LABELRANK ?? 9) -
            Number(right.properties?.LABELRANK ?? 9);
          return (
            rankDifference ||
            Number(right.properties?.POP_EST ?? 0) -
              Number(left.properties?.POP_EST ?? 0)
          );
        },
      );
    (this.zoom < 1.3 ? candidates.slice(0, 18) : candidates).forEach(
      (feature) => {
        const properties = feature.properties!;
        const name =
          properties.NAME_EN ?? properties.NAME ?? properties.ADMIN ?? "";
        const [x, y] = this.project(
          Number(properties.LABEL_X),
          Number(properties.LABEL_Y),
        );
        if (
          x < this.mapRect.x + 8 ||
          x > this.mapRect.x + this.mapRect.width - 8 ||
          y < this.mapRect.y + 8 ||
          y > this.mapRect.y + this.mapRect.height - 8
        ) {
          return;
        }
        const width = context.measureText(name).width + 7;
        const box = { x: x - width / 2, y: y - 5, width, height: 10 };
        const collision = occupied.some(
          (other) =>
            box.x < other.x + other.width &&
            box.x + box.width > other.x &&
            box.y < other.y + other.height &&
            box.y + box.height > other.y,
        );
        if (collision) return;
        occupied.push(box);
        context.fillText(name, x, y);
      },
    );
    context.restore();
  }

  private drawIndiaAdmin(): void {
    this.indiaAdmin.features.forEach((feature) =>
      this.drawGeometry(
        feature.geometry,
        "rgb(255 218 133 / 78%)",
        this.zoom >= 3 ? 0.8 : 0.55,
      ),
    );
    if (this.zoom < 2.8) return;
    const context = this.context;
    context.save();
    context.font = "7px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "rgb(255 233 179 / 78%)";
    context.shadowColor = "rgb(3 15 19 / 96%)";
    context.shadowBlur = 4;
    this.indiaAdmin.features
      .filter((feature) => feature.properties.label)
      .forEach((feature) => {
        const [x, y] = this.project(
          feature.properties.label_longitude,
          feature.properties.label_latitude,
        );
        context.fillText(feature.properties.name, x, y);
      });
    context.restore();
  }

  private drawWindVectors(): void {
    if (!this.data.vectors.wind850) return;
    const context = this.context;
    const step = this.zoom >= 3 ? 4 : this.zoom >= 1.8 ? 6 : 10;
    const nextDay = Math.min(this.currentDay + 1, 41);
    context.save();
    context.strokeStyle = "rgb(244 248 234 / 67%)";
    context.fillStyle = "rgb(244 248 234 / 67%)";
    context.lineWidth = 0.8;
    for (
      let latitudeIndex = step;
      latitudeIndex < GRID_HEIGHT - step;
      latitudeIndex += step
    ) {
      const latitude = 90 - latitudeIndex * 1.5;
      if (Math.abs(latitude) > 78) continue;
      for (
        let displayLongitudeIndex = 0;
        displayLongitudeIndex < GRID_WIDTH;
        displayLongitudeIndex += step
      ) {
        const sourceLongitudeIndex =
          (displayLongitudeIndex + 120) % GRID_WIDTH;
        const uA = this.vectorValueAt(
          "u",
          this.currentDay,
          latitudeIndex,
          sourceLongitudeIndex,
        );
        const vA = this.vectorValueAt(
          "v",
          this.currentDay,
          latitudeIndex,
          sourceLongitudeIndex,
        );
        const uB = this.vectorValueAt(
          "u",
          nextDay,
          latitudeIndex,
          sourceLongitudeIndex,
        );
        const vB = this.vectorValueAt(
          "v",
          nextDay,
          latitudeIndex,
          sourceLongitudeIndex,
        );
        const u = uA * (1 - this.currentMix) + uB * this.currentMix;
        const v = vA * (1 - this.currentMix) + vB * this.currentMix;
        const speed = Math.hypot(u, v);
        if (!Number.isFinite(speed) || speed < 1) continue;
        const longitude = -180 + displayLongitudeIndex * 1.5;
        const [x, y] = this.project(longitude, latitude);
        if (
          x < this.mapRect.x ||
          x > this.mapRect.x + this.mapRect.width ||
          y < this.mapRect.y ||
          y > this.mapRect.y + this.mapRect.height
        ) {
          continue;
        }
        const length = clamp(4 + speed * 0.3, 5, 13);
        const dx = (u / speed) * length;
        const dy = (-v / speed) * length;
        const endX = x + dx;
        const endY = y + dy;
        const angle = Math.atan2(dy, dx);
        context.beginPath();
        context.moveTo(x - dx * 0.35, y - dy * 0.35);
        context.lineTo(endX, endY);
        context.stroke();
        context.beginPath();
        context.moveTo(endX, endY);
        context.lineTo(
          endX - Math.cos(angle - 0.55) * 3.3,
          endY - Math.sin(angle - 0.55) * 3.3,
        );
        context.lineTo(
          endX - Math.cos(angle + 0.55) * 3.3,
          endY - Math.sin(angle + 0.55) * 3.3,
        );
        context.closePath();
        context.fill();
      }
    }
    context.restore();
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

  private displayGridValue(
    variable: GlobalVariableKey,
    day: number,
    x: number,
    y: number,
  ): number {
    const sourceLongitudeIndex = (x + 120) % GRID_WIDTH;
    return this.valueAt(variable, day, y, sourceLongitudeIndex);
  }

  private drawContour(
    path: Path2D | null,
    color = "rgb(248 249 229 / 65%)",
    lineDash: number[] = [],
  ): void {
    if (!path) return;
    const context = this.context;
    context.save();
    context.strokeStyle = color;
    context.lineWidth = 0.7;
    context.setLineDash(lineDash);
    context.stroke(path);
    context.restore();
  }

  private rebuildContours(): void {
    this.z500ContourPath =
      (this.currentVariable === "z500" || this.overlays.z500Contours) &&
      this.data.fields.z500
        ? this.buildContourPath("z500", this.currentDay, Z500_LEVELS)
        : null;
    this.pressureContourPath =
      this.overlays.pressureContours && this.data.fields.mslp
        ? this.buildContourPath("mslp", this.currentDay, MSLP_LEVELS)
        : null;
  }

  private buildContourPath(
    variable: GlobalVariableKey,
    day: number,
    levels: number[],
  ): Path2D {
    const path = new Path2D();
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

    for (const level of levels) {
      for (let y = 0; y < GRID_HEIGHT - 1; y += 1) {
        for (let x = 0; x < GRID_WIDTH - 1; x += 1) {
          const tl = this.displayGridValue(variable, day, x, y);
          const tr = this.displayGridValue(variable, day, x + 1, y);
          const br = this.displayGridValue(variable, day, x + 1, y + 1);
          const bl = this.displayGridValue(variable, day, x, y + 1);
          const code =
            (tl >= level ? 1 : 0) |
            (tr >= level ? 2 : 0) |
            (br >= level ? 4 : 0) |
            (bl >= level ? 8 : 0);
          (segments[code] ?? []).forEach(([edgeA, edgeB]) => {
            const a = pointOnEdge(edgeA, x, y, level, tl, tr, br, bl);
            const b = pointOnEdge(edgeB, x, y, level, tl, tr, br, bl);
            const [ax, ay] = this.project(-180 + a[0] * 1.5, 90 - a[1] * 1.5);
            const [bx, by] = this.project(-180 + b[0] * 1.5, 90 - b[1] * 1.5);
            path.moveTo(ax, ay);
            path.lineTo(bx, by);
          });
        }
      }
    }
    return path;
  }

  private movePointer(event: PointerEvent): void {
    if (this.pointers.has(event.pointerId)) {
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }
    if (this.pointers.size >= 2) {
      const [first, second] = Array.from(this.pointers.values()).slice(0, 2);
      const distance = Math.hypot(second.x - first.x, second.y - first.y);
      if (this.pinchDistance && this.pinchDistance > 0) {
        this.zoomAt(
          (first.x + second.x) / 2,
          (first.y + second.y) / 2,
          distance / this.pinchDistance,
        );
      }
      this.pinchDistance = distance;
      this.suppressClick = true;
      this.tooltip.hidden = true;
      return;
    }
    if (this.drag && event.pointerId === this.drag.pointerId) {
      const deltaX = event.clientX - this.drag.x;
      const deltaY = event.clientY - this.drag.y;
      if (Math.hypot(deltaX, deltaY) > 3) this.dragMoved = true;
      this.panX = this.drag.panX + deltaX;
      this.panY = this.drag.panY + deltaY;
      this.constrainPan();
      this.draw();
      this.tooltip.hidden = true;
      return;
    }
    this.showTooltip(event);
  }

  private startDrag(event: PointerEvent): void {
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (this.pointers.size >= 2) {
      const [first, second] = Array.from(this.pointers.values()).slice(0, 2);
      this.pinchDistance = Math.hypot(second.x - first.x, second.y - first.y);
      this.drag = null;
      this.suppressClick = true;
      this.canvas.setPointerCapture(event.pointerId);
      return;
    }
    this.drag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      panX: this.panX,
      panY: this.panY,
    };
    this.dragMoved = false;
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.classList.add("is-dragging");
  }

  private endDrag(event: PointerEvent): void {
    this.pointers.delete(event.pointerId);
    if (this.pointers.size < 2) this.pinchDistance = null;
    if (!this.drag || event.pointerId !== this.drag.pointerId) {
      if (this.canvas.hasPointerCapture(event.pointerId)) {
        this.canvas.releasePointerCapture(event.pointerId);
      }
      this.rebuildContours();
      this.draw();
      return;
    }
    this.suppressClick = this.dragMoved;
    this.drag = null;
    this.canvas.classList.remove("is-dragging");
    this.rebuildContours();
    this.draw();
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
  }

  private zoomAt(clientX: number, clientY: number, factor: number): void {
    const bounds = this.canvas.getBoundingClientRect();
    const x = clientX - bounds.left;
    const y = clientY - bounds.top;
    const [longitude, latitude] = this.unproject(x, y);
    const nextZoom = clamp(this.zoom * factor, 1, 6);
    if (nextZoom === this.zoom) return;
    this.zoom = nextZoom;
    const normalizedX = (longitude + 180) / 360;
    const normalizedY = (90 - latitude) / 180;
    this.panX =
      x -
      this.mapRect.x -
      this.mapRect.width / 2 -
      (normalizedX - 0.5) * this.mapRect.width * this.zoom;
    this.panY =
      y -
      this.mapRect.y -
      this.mapRect.height / 2 -
      (normalizedY - 0.5) * this.mapRect.height * this.zoom;
    this.constrainPan();
    this.viewChanged();
  }

  private constrainPan(): void {
    const maximumX = ((this.zoom - 1) * this.mapRect.width) / 2;
    const maximumY = ((this.zoom - 1) * this.mapRect.height) / 2;
    this.panX = clamp(this.panX, -maximumX, maximumX);
    this.panY = clamp(this.panY, -maximumY, maximumY);
  }

  private viewChanged(): void {
    this.rebuildContours();
    this.draw();
    this.onViewChange?.(this.zoom);
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
    const definition = this.data.metadata.variables[this.currentVariable];
    const displayDefinition =
      this.currentMode === "anomaly" && definition.anomaly
        ? definition.anomaly
        : definition;
    const sourceIndex =
      selection.latitudeIndex * GRID_WIDTH + selection.longitudeIndex;
    const isMasked =
      definition.domain === "ocean" &&
      this.data.oceanMask[sourceIndex] !== 1;
    const value = this.displayValueAt(
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
      <span>${displayDefinition.short_label}</span>
      <strong>${isMasked ? "Open-ocean field" : formatMapValue(value, displayDefinition.units)}</strong>
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
    const [longitude, latitude] = this.unproject(x, y);
    if (
      longitude < -180 ||
      longitude > 180 ||
      latitude < -90 ||
      latitude > 90
    ) {
      return null;
    }
    const sourceLongitude = longitude < 0 ? longitude + 360 : longitude;
    const longitudeIndex = Math.round(sourceLongitude / 1.5) % GRID_WIDTH;
    const latitudeIndex = clamp(
      Math.round((90 - latitude) / 1.5),
      0,
      GRID_HEIGHT - 1,
    );
    const exactLatitude = 90 - latitudeIndex * 1.5;
    const exactLongitude = longitudeIndex * 1.5;
    const displayLongitude =
      exactLongitude > 180 ? exactLongitude - 360 : exactLongitude;
    const names = this.locationNames(displayLongitude, exactLatitude);
    return {
      latitude: exactLatitude,
      longitude: exactLongitude,
      latitudeIndex,
      longitudeIndex,
      ...names,
    };
  }

  private locationNames(
    longitude: number,
    latitude: number,
  ): { country?: string; state?: string } {
    let state: string | undefined;
    for (const feature of this.indiaAdmin.features) {
      if (pointInGeometry(longitude, latitude, feature.geometry)) {
        if (feature.properties.label) state = feature.properties.name;
        break;
      }
    }
    let country: string | undefined;
    for (const feature of this.world.features) {
      if (
        feature.geometry &&
        pointInGeometry(longitude, latitude, feature.geometry)
      ) {
        country =
          feature.properties?.NAME_EN ??
          feature.properties?.NAME ??
          feature.properties?.ADMIN;
        break;
      }
    }
    return { country, state };
  }
}
