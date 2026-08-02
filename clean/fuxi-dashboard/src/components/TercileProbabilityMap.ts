import type {
  ForecastData,
  IndiaMapGeographyData,
  RegionalOutlookData,
  RegionalOutlookWeek,
  TercileCategory,
} from "../types";

const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_SIZE = 620;
const LON_MIN = 59.25;
const LON_MAX = 99.75;
const LAT_MIN = -0.75;
const LAT_MAX = 39.75;
const VISUAL_GRID_SIZE = 162;

const CATEGORY_LABELS: Record<
  "rainfall" | "temperature",
  Record<TercileCategory, string>
> = {
  rainfall: {
    below_normal: "Below normal",
    near_normal: "Near normal",
    above_normal: "Above normal",
  },
  temperature: {
    below_normal: "Cooler than normal",
    near_normal: "Near normal",
    above_normal: "Warmer than normal",
  },
};

const CATEGORY_COLORS: Record<
  "rainfall" | "temperature",
  Record<TercileCategory, [number, number, number]>
> = {
  rainfall: {
    below_normal: [190, 91, 42],
    near_normal: [135, 143, 150],
    above_normal: [25, 125, 91],
  },
  temperature: {
    below_normal: [55, 105, 165],
    near_normal: [135, 143, 150],
    above_normal: [198, 67, 45],
  },
};

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

function dominantCategory(
  probabilities: Record<TercileCategory, number>,
): TercileCategory {
  const categories: TercileCategory[] = [
    "near_normal",
    "below_normal",
    "above_normal",
  ];
  return categories.reduce((best, category) =>
    probabilities[category] > probabilities[best] ? category : best,
  );
}

function probabilityColor(
  variable: "rainfall" | "temperature",
  category: TercileCategory,
  probability: number,
): string {
  const base = CATEGORY_COLORS[variable][category];
  const strength = Math.max(0, Math.min(1, (probability - 33) / 67));
  const whiteMix = 0.72 * (1 - strength);
  const channels = base.map((channel) =>
    Math.round(channel * (1 - whiteMix) + 255 * whiteMix),
  );
  return `rgb(${channels.join(" ")})`;
}

export class TercileProbabilityMap {
  private readonly container: HTMLElement;
  private readonly forecast: ForecastData;
  private readonly outlook: RegionalOutlookData;
  private readonly geography: IndiaMapGeographyData;
  private readonly tooltip: HTMLDivElement;

  constructor(
    container: HTMLElement,
    forecast: ForecastData,
    outlook: RegionalOutlookData,
    geography: IndiaMapGeographyData,
  ) {
    this.container = container;
    this.forecast = forecast;
    this.outlook = outlook;
    this.geography = geography;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "map-tooltip outlook-map-tooltip";
    this.tooltip.hidden = true;
    this.container.append(this.tooltip);
  }

  render(
    variable: "rainfall" | "temperature",
    week: RegionalOutlookWeek,
  ): void {
    this.container.querySelector("svg")?.remove();
    const svg = svgElement("svg");
    svg.setAttribute("viewBox", `0 0 ${MAP_SIZE} ${MAP_SIZE}`);
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      `${variable} most-likely tercile and raw ensemble probability for Week ${week.week}`,
    );
    svg.classList.add("forecast-map", "outlook-probability-map");

    const ocean = svgElement("rect");
    ocean.setAttribute("width", String(MAP_SIZE));
    ocean.setAttribute("height", String(MAP_SIZE));
    ocean.setAttribute("class", "map-ocean");
    svg.append(ocean);

    const definitions = svgElement("defs");
    const clip = svgElement("clipPath");
    clip.setAttribute("id", "outlook-india-clip");
    const clipPath = svgElement("path");
    clipPath.setAttribute("d", this.geography.india_outline_path);
    clipPath.setAttribute("fill-rule", "evenodd");
    clip.append(clipPath);
    definitions.append(clip);
    svg.append(definitions);

    this.addProbabilityField(svg, variable, week);
    this.addWorldBoundaries(svg);
    this.addGraticules(svg);
    this.addInteraction(svg, variable, week);
    this.addOutline(svg);
    this.addIndiaAdmin(svg);
    this.container.prepend(svg);
  }

  private addProbabilityField(
    svg: SVGSVGElement,
    variable: "rainfall" | "temperature",
    week: RegionalOutlookWeek,
  ): void {
    const fields = week.probability_fields[variable];
    const { latitude, longitude } = this.outlook.grid;
    const canvas = document.createElement("canvas");
    canvas.width = VISUAL_GRID_SIZE;
    canvas.height = VISUAL_GRID_SIZE;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas rendering is unavailable");

    const sample = (
      category: TercileCategory,
      row: number,
      column: number,
    ): number => {
      const safeRow = Math.max(0, Math.min(latitude.length - 1, row));
      const safeColumn = Math.max(0, Math.min(longitude.length - 1, column));
      return fields[category][safeRow * longitude.length + safeColumn];
    };

    const interpolate = (
      category: TercileCategory,
      row: number,
      column: number,
      rowWeight: number,
      columnWeight: number,
    ): number => {
      const top =
        sample(category, row, column) * (1 - columnWeight) +
        sample(category, row, column + 1) * columnWeight;
      const bottom =
        sample(category, row + 1, column) * (1 - columnWeight) +
        sample(category, row + 1, column + 1) * columnWeight;
      return top * (1 - rowWeight) + bottom * rowWeight;
    };

    for (let y = 0; y < VISUAL_GRID_SIZE; y += 1) {
      const lat = LAT_MAX - ((y + 0.5) / VISUAL_GRID_SIZE) * (LAT_MAX - LAT_MIN);
      const row = (latitude[0] - lat) / this.outlook.grid.spacing_degrees;
      const row0 = Math.floor(row);
      const rowWeight = row - row0;
      for (let x = 0; x < VISUAL_GRID_SIZE; x += 1) {
        const lon = LON_MIN + ((x + 0.5) / VISUAL_GRID_SIZE) * (LON_MAX - LON_MIN);
        const column = (lon - longitude[0]) / this.outlook.grid.spacing_degrees;
        const column0 = Math.floor(column);
        const columnWeight = column - column0;
        const probabilities: Record<TercileCategory, number> = {
          below_normal: interpolate(
            "below_normal", row0, column0, rowWeight, columnWeight,
          ),
          near_normal: interpolate(
            "near_normal", row0, column0, rowWeight, columnWeight,
          ),
          above_normal: interpolate(
            "above_normal", row0, column0, rowWeight, columnWeight,
          ),
        };
        const category = dominantCategory(probabilities);
        context.fillStyle = probabilityColor(
          variable,
          category,
          probabilities[category],
        );
        context.fillRect(x, y, 1, 1);
      }
    }

    const image = svgElement("image");
    image.setAttribute("x", "0");
    image.setAttribute("y", "0");
    image.setAttribute("width", String(MAP_SIZE));
    image.setAttribute("height", String(MAP_SIZE));
    image.setAttribute("preserveAspectRatio", "none");
    image.setAttribute("href", canvas.toDataURL("image/png"));
    image.setAttribute("clip-path", "url(#outlook-india-clip)");
    image.setAttribute("class", "map-raster");
    svg.append(image);
  }

  private addInteraction(
    svg: SVGSVGElement,
    variable: "rainfall" | "temperature",
    week: RegionalOutlookWeek,
  ): void {
    const fields = week.probability_fields[variable];
    const { latitude, longitude } = this.outlook.grid;
    svg.setAttribute("tabindex", "0");
    svg.classList.add("forecast-map--interactive");
    const show = (event?: PointerEvent): void => {
      let latIndex = Math.floor(latitude.length / 2);
      let lonIndex = Math.floor(longitude.length / 2);
      if (event) {
        const bounds = svg.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
        const y = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
        const hoveredLongitude = LON_MIN + x * (LON_MAX - LON_MIN);
        const hoveredLatitude = LAT_MAX - y * (LAT_MAX - LAT_MIN);
        lonIndex = Math.max(
          0,
          Math.min(
            longitude.length - 1,
            Math.round((hoveredLongitude - longitude[0]) / this.outlook.grid.spacing_degrees),
          ),
        );
        latIndex = Math.max(
          0,
          Math.min(
            latitude.length - 1,
            Math.round((latitude[0] - hoveredLatitude) / this.outlook.grid.spacing_degrees),
          ),
        );
      }
      const index = latIndex * longitude.length + lonIndex;
      const probabilities: Record<TercileCategory, number> = {
        below_normal: fields.below_normal[index],
        near_normal: fields.near_normal[index],
        above_normal: fields.above_normal[index],
      };
      const category = dominantCategory(probabilities);
      const lat = latitude[latIndex];
      const lon = longitude[lonIndex];
      const supported = this.forecast.grid.india_mask[index];
      this.tooltip.innerHTML = supported
        ? `
          <strong>${CATEGORY_LABELS[variable][category]} · ${probabilities[category]}%</strong>
          <span>Below ${probabilities.below_normal}% · Near ${probabilities.near_normal}% · Above ${probabilities.above_normal}%</span>
          <small>${lat.toFixed(1)}°N · ${lon.toFixed(1)}°E · native 1.5° cell</small>
        `
        : `
          <strong>Outside supported India cells</strong>
          <span>${lat.toFixed(1)}°N · ${lon.toFixed(1)}°E</span>
        `;
      this.tooltip.hidden = false;
      if (event) {
        const bounds = this.container.getBoundingClientRect();
        this.tooltip.style.left = `${event.clientX - bounds.left + 14}px`;
        this.tooltip.style.top = `${event.clientY - bounds.top + 14}px`;
      } else {
        this.tooltip.style.left = "16px";
        this.tooltip.style.top = "16px";
      }
    };
    svg.addEventListener("pointermove", (event) => show(event));
    svg.addEventListener("pointerenter", (event) => show(event));
    svg.addEventListener("focus", () => show());
    svg.addEventListener("pointerleave", () => {
      this.tooltip.hidden = true;
    });
    svg.addEventListener("blur", () => {
      this.tooltip.hidden = true;
    });
  }

  private addGraticules(svg: SVGSVGElement): void {
    for (const longitude of [70, 80, 90]) {
      const line = svgElement("line");
      line.setAttribute("x1", String(mapX(longitude)));
      line.setAttribute("x2", String(mapX(longitude)));
      line.setAttribute("y1", "0");
      line.setAttribute("y2", String(MAP_SIZE));
      line.setAttribute("class", "map-graticule");
      svg.append(line);
    }
    for (const latitude of [10, 20, 30]) {
      const line = svgElement("line");
      line.setAttribute("x1", "0");
      line.setAttribute("x2", String(MAP_SIZE));
      line.setAttribute("y1", String(mapY(latitude)));
      line.setAttribute("y2", String(mapY(latitude)));
      line.setAttribute("class", "map-graticule");
      svg.append(line);
    }
  }

  private addWorldBoundaries(svg: SVGSVGElement): void {
    const path = svgElement("path");
    path.setAttribute("d", this.geography.world_path);
    path.setAttribute("class", "map-world-outline");
    path.setAttribute("fill-rule", "evenodd");
    svg.append(path);
  }

  private addOutline(svg: SVGSVGElement): void {
    const path = svgElement("path");
    path.setAttribute("d", this.geography.india_outline_path);
    path.setAttribute("class", "map-outline");
    path.setAttribute("fill-rule", "evenodd");
    svg.append(path);
  }

  private addIndiaAdmin(svg: SVGSVGElement): void {
    const path = svgElement("path");
    path.setAttribute("d", this.geography.india_admin_path);
    path.setAttribute("class", "map-india-admin");
    path.setAttribute("fill-rule", "evenodd");
    svg.append(path);
  }
}
