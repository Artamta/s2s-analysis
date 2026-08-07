import {
  GlobalForecastMap,
  type MapOverlays,
  type MapSelection,
} from "../components/GlobalForecastMap";
import { createLegend } from "../components/Legend";
import type {
  AppData,
  GlobalDisplayMode,
  GlobalForecastData,
  GlobalVariableKey,
} from "../types";

const VARIABLE_ORDER: GlobalVariableKey[] = [
  "precipitation",
  "temperature",
  "z500",
  "wind850",
  "mslp",
  "sst",
  "tcwv",
  "olr",
];
const FAMILY_LABELS = {
  surface: "Surface",
  circulation: "Circulation",
  "ocean-convection": "Ocean & convection",
} as const;
const SPEEDS = [0.5, 1, 2];
const BASE_FRAME_MS = 1250;
const FRAME_SIZE = 121 * 240;

function friendlyDate(isoDate: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${isoDate}T12:00:00Z`));
}

function variableGlyph(variable: GlobalVariableKey): string {
  const glyphs: Record<GlobalVariableKey, string> = {
    precipitation: "◌",
    temperature: "°",
    z500: "≋",
    wind850: "↗",
    mslp: "P",
    sst: "≈",
    olr: "☁",
    tcwv: "≋",
  };
  return glyphs[variable];
}

function titleCase(value: string): string {
  return value
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function coordinateLabel(selection: MapSelection): string {
  const latitude =
    selection.latitude < 0
      ? `${Math.abs(selection.latitude).toFixed(1)}°S`
      : `${selection.latitude.toFixed(1)}°N`;
  const longitude =
    selection.longitude > 180
      ? `${(360 - selection.longitude).toFixed(1)}°W`
      : `${selection.longitude.toFixed(1)}°E`;
  return `${latitude} · ${longitude}`;
}

function pointLabel(selection: MapSelection): string {
  const place = selection.state
    ? `${titleCase(selection.state)}, India`
    : selection.country;
  return place
    ? `${place} · ${coordinateLabel(selection)}`
    : coordinateLabel(selection);
}

function leadGuidance(day: number): { period: string; guidance: string } {
  if (day < 14) {
    return {
      period: "Days 1–14 · weather → subseasonal",
      guidance:
        "Synoptic features remain useful, but local timing becomes less certain with lead.",
    };
  }
  if (day < 28) {
    return {
      period: "Weeks 3–4 · subseasonal range",
      guidance:
        "Read persistent large-scale patterns and ensemble agreement—not exact day-to-day weather.",
    };
  }
  return {
    period: "Weeks 5–6 · broad outlook",
    guidance:
      "Use only broad circulation and ensemble tendencies; local detail has low predictability.",
  };
}

function linePath(
  values: number[],
  x: (index: number) => number,
  y: (value: number) => number,
): string {
  return values
    .map(
      (value, index) =>
        `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(value).toFixed(1)}`,
    )
    .join(" ");
}

function renderPointChart(
  data: GlobalForecastData,
  variable: GlobalVariableKey,
  selection: MapSelection,
  selectedDay: number,
  mode: GlobalDisplayMode,
): string {
  const definition = data.metadata.variables[variable];
  const displayDefinition =
    mode === "anomaly" && definition.anomaly
      ? definition.anomaly
      : definition;
  const pointIndex =
    selection.latitudeIndex * 240 + selection.longitudeIndex;
  if (definition.domain === "ocean" && data.oceanMask[pointIndex] !== 1) {
    return `
      <div class="point-unavailable">
        <strong>Open-ocean field</strong>
        <p>Select a supported ocean grid cell to inspect the 42-day SST trace.</p>
      </div>
    `;
  }
  const field =
    mode === "anomaly" ? data.anomalies[variable] : data.fields[variable];
  const spreadField = data.spreads[variable];
  if (!field || !spreadField) {
    return `<div class="point-unavailable"><strong>Loading field…</strong></div>`;
  }
  const means = Array.from({ length: 42 }, (_, index) => {
    const encoded = field[index * FRAME_SIZE + pointIndex];
    return encoded * displayDefinition.scale + displayDefinition.offset;
  });
  const spreads = Array.from({ length: 42 }, (_, index) => {
    const encoded = spreadField[index * FRAME_SIZE + pointIndex];
    return encoded * definition.spread.scale + definition.spread.offset;
  });
  const lower = means.map((mean, index) =>
    mode === "absolute" &&
    (variable === "precipitation" || variable === "wind850")
      ? Math.max(0, mean - spreads[index])
      : mean - spreads[index],
  );
  const upper = means.map((mean, index) => mean + spreads[index]);
  let minimum = Math.min(...lower);
  let maximum = Math.max(...upper);
  const rawRange = Math.max(maximum - minimum, Math.abs(maximum) * 0.05, 1);
  const padding = rawRange * 0.1;
  minimum =
    mode === "absolute" &&
    (variable === "precipitation" || variable === "wind850")
      ? 0
      : minimum - padding;
  maximum += padding;

  const width = 320;
  const height = 116;
  const left = 8;
  const right = 8;
  const top = 8;
  const bottom = 15;
  const x = (index: number): number =>
    left + (index / 41) * (width - left - right);
  const y = (value: number): number =>
    top +
    ((maximum - value) / (maximum - minimum)) *
      (height - top - bottom);
  const band = [
    ...upper.map(
      (value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`,
    ),
    ...lower
      .map(
        (value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`,
      )
      .reverse(),
  ].join(" ");
  const weekLines = [7, 14, 21, 28, 35]
    .map(
      (index) =>
        `<line x1="${x(index - 0.5).toFixed(1)}" y1="${top}" x2="${x(index - 0.5).toFixed(1)}" y2="${height - bottom}" />`,
    )
    .join("");
  const mean = means[selectedDay];
  const spread = spreads[selectedDay];
  const decimals =
    displayDefinition.units === "W/m²" ||
    (variable === "precipitation" && mean >= 10)
      ? 0
      : 1;

  return `
    <div class="point-value">
      <strong>${mean.toFixed(decimals)}</strong>
      <span>± ${spread.toFixed(decimals)} ${displayDefinition.units}</span>
      <small>Lead ${String(selectedDay + 1).padStart(2, "0")} · ${mode === "anomaly" ? "mean anomaly" : "ensemble mean"} ±1σ</small>
    </div>
    <svg class="point-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="42-day ensemble mean and member spread at ${coordinateLabel(selection)}">
      <g class="point-chart__weeks">${weekLines}</g>
      <line class="point-chart__axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" />
      <polygon class="point-chart__band" points="${band}" />
      <path class="point-chart__mean" d="${linePath(means, x, y)}" />
      <line class="point-chart__selected" x1="${x(selectedDay).toFixed(1)}" y1="${top}" x2="${x(selectedDay).toFixed(1)}" y2="${height - bottom}" />
      <circle class="point-chart__dot" cx="${x(selectedDay).toFixed(1)}" cy="${y(mean).toFixed(1)}" r="3.2" />
      <g class="point-chart__labels">
        <text x="${left}" y="${height - 2}">D1</text>
        <text x="${x(20.5).toFixed(1)}" y="${height - 2}" text-anchor="middle">D21</text>
        <text x="${width - right}" y="${height - 2}" text-anchor="end">D42</text>
      </g>
    </svg>
  `;
}

export function renderGlobalPage(
  container: HTMLElement,
  data: AppData,
): () => void {
  const { global, world, indiaAdmin } = data;
  if (!global || !world || !indiaAdmin) {
    container.innerHTML = `
      <section class="blocked-state blocked-state--dark">
        <span class="eyebrow">Global data unavailable</span>
        <h1>The dated Global Demo package could not be loaded.</h1>
        <p>No partial map has been shown.</p>
      </section>
    `;
    return () => undefined;
  }
  const activeGlobal = global;
  if (global.metadata.validation.status === "failure") {
    container.innerHTML = `
      <section class="blocked-state blocked-state--dark">
        <span class="eyebrow">Global data gate closed</span>
        <h1>The global guidance is unavailable.</h1>
        <p>The viewer will not display a package that failed its publication checks.</p>
      </section>
    `;
    return () => undefined;
  }

  container.innerHTML = `
    <section class="global-stage" aria-label="Global 42-day forecast viewer">
      <div class="global-map-host" id="global-map"></div>
      <div class="global-stage-shade" aria-hidden="true"></div>

      <div class="global-title">
        <span class="global-eyebrow"><b>Dated demo</b> · Pinned dataset · not the current forecast</span>
        <h1>Global forecast <small>Demo</small></h1>
        <p>Initialized 28 Jul 2026 · 42-day guidance · 100 members</p>
      </div>

      <aside class="global-facts" aria-label="Forecast issue and integrity facts">
        <div><span>Initialized</span><strong>28 Jul 2026 · 00 UTC</strong></div>
        <div><span>Product</span><strong class="experimental-text">Pinned 28 Jul 2026 research demo</strong></div>
        <div><span>Integrity</span><strong class="integrity-ok" id="integrity-status">✓ Active layer SHA-256 verified</strong></div>
      </aside>

      <aside class="global-layer-dock" aria-label="Forecast fields">
        <div class="layer-dock-heading"><span>Shaded field · choose one</span><small id="layer-load-status">SHA-256 verified</small></div>
        <div id="global-layer-groups"></div>
        <div class="field-mode" role="group" aria-label="Absolute or anomaly display">
          <button id="mode-absolute" class="is-active" type="button">Absolute</button>
          <button id="mode-anomaly" type="button">Anomaly</button>
        </div>
        <p class="field-mode-note" id="field-mode-note">Anomaly available against the matched 2002–2021 model climate.</p>
        <div class="overlay-controls">
          <span>Overlays · combine freely</span>
          <label><input id="overlay-countries" type="checkbox" checked><i></i>Country names</label>
          <label><input id="overlay-india" type="checkbox"><i></i>India states · SoI ABDB</label>
          <label><input id="overlay-wind" type="checkbox"><i></i>850 hPa vectors</label>
          <label><input id="overlay-pressure" type="checkbox"><i></i>MSLP · 10 hPa dashed</label>
          <label><input id="overlay-z500" type="checkbox"><i></i>Z500 · 20 dam solid</label>
        </div>
      </aside>

      <div class="global-map-nav" aria-label="Map navigation">
        <button id="zoom-in" type="button" aria-label="Zoom in">+</button>
        <span id="zoom-level">100%</span>
        <button id="zoom-out" type="button" aria-label="Zoom out">−</button>
        <button id="india-focus" type="button">India</button>
        <button id="world-focus" type="button">World</button>
      </div>

      <aside class="global-inspector" aria-label="Selected-location 42-day ensemble trace">
        <div class="global-inspector__head">
          <span>Click map · 42-day point trace</span>
          <strong id="point-location">India · 28.5°N · 76.5°E</strong>
        </div>
        <div id="point-chart"></div>
        <div class="spread-key"><i></i><span>Mean</span><b></b><span>±1 population σ</span></div>
        <p class="spread-caution">Spread measures member disagreement, not calibrated confidence.</p>
        <div class="notice-card">
          <span>What to notice</span>
          <p id="notice-copy"></p>
        </div>
      </aside>

      <div class="global-legend-panel">
        <div>
          <span id="global-legend-label"></span>
          <strong id="global-legend-units"></strong>
        </div>
        <div id="global-legend"></div>
      </div>

      <div class="global-s2s-guide">
        <span id="lead-period"></span>
        <p id="lead-guidance"></p>
      </div>

      <div class="global-timeline">
        <button class="play-button" id="play-button" type="button" aria-label="Pause animation">
          <span aria-hidden="true">Ⅱ</span>
        </button>
        <div class="timeline-copy">
          <span id="lead-label">Lead 01</span>
          <strong id="valid-label">28 Jul 2026</strong>
        </div>
        <div class="timeline-track-wrap">
          <input id="day-slider" class="day-slider" type="range" min="0" max="41" value="0" step="1" aria-label="Forecast lead day">
          <div class="timeline-weeks" aria-hidden="true">
            <span>W1</span><span>W2</span><span>W3</span><span>W4</span><span>W5</span><span>W6</span>
          </div>
        </div>
        <button class="speed-button" id="speed-button" type="button" aria-label="Change animation speed">1×</button>
      </div>

      <p class="global-disclaimer">Dated 28 Jul 2026 demonstration · not updated with the current India forecast · broad guidance, not local weather · <a href="mailto:raj.ayush@students.iiserpune.ac.in">public contact</a></p>
    </section>
  `;

  let variable: GlobalVariableKey = "precipitation";
  let mode: GlobalDisplayMode = "absolute";
  let day = 0;
  let selection: MapSelection = {
    latitude: 28.5,
    longitude: 76.5,
    latitudeIndex: 41,
    longitudeIndex: 51,
    country: "India",
  };
  let overlays: MapOverlays = {
    countryLabels: true,
    indiaStates: false,
    windVectors: false,
    pressureContours: false,
    z500Contours: false,
  };
  let speedIndex = 1;
  let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let cycleStart = performance.now();
  let animationFrame = 0;
  const mapHost = container.querySelector<HTMLElement>("#global-map")!;
  const layerGroups =
    container.querySelector<HTMLDivElement>("#global-layer-groups")!;
  const slider = container.querySelector<HTMLInputElement>("#day-slider")!;
  const playButton =
    container.querySelector<HTMLButtonElement>("#play-button")!;
  const speedButton =
    container.querySelector<HTMLButtonElement>("#speed-button")!;
  const map = new GlobalForecastMap(
    mapHost,
    global,
    world,
    indiaAdmin,
    (next) => {
      selection = next;
      updateInspector();
    },
    (zoom) => {
      container.querySelector("#zoom-level")!.textContent =
        `${Math.round(zoom * 100)}%`;
    },
  );

  (
    ["surface", "circulation", "ocean-convection"] as const
  ).forEach((family) => {
    const group = document.createElement("div");
    group.className = "layer-family";
    group.innerHTML = `<span>${FAMILY_LABELS[family]}</span><div></div>`;
    const buttons = group.querySelector<HTMLDivElement>("div")!;
    VARIABLE_ORDER.filter(
      (key) => global.metadata.variables[key].family === family,
    ).forEach((key) => {
      const definition = global.metadata.variables[key];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "global-layer-button";
      button.dataset.variable = key;
      button.innerHTML = `
        <span class="layer-glyph" aria-hidden="true">${variableGlyph(key)}</span>
        <span><strong>${definition.short_label}</strong><small>${definition.units}</small></span>
      `;
      button.addEventListener("click", () => void selectVariable(key, button));
      buttons.append(button);
    });
    layerGroups.append(group);
  });

  async function selectVariable(
    key: GlobalVariableKey,
    button?: HTMLButtonElement,
  ): Promise<void> {
    if (
      key === variable &&
      activeGlobal.fields[key] &&
      activeGlobal.spreads[key]
    ) return;
    const status = container.querySelector("#layer-load-status")!;
    button?.classList.add("is-loading");
    status.textContent = "Verifying…";
    try {
      await activeGlobal.loadVariable(key);
      variable = key;
      if (mode === "anomaly" && !activeGlobal.metadata.variables[key].anomaly) {
        mode = "absolute";
      }
      if (key === "wind850") {
        overlays.windVectors = true;
        container.querySelector<HTMLInputElement>("#overlay-wind")!.checked =
          true;
      }
      if (key === "mslp") {
        overlays.pressureContours = true;
        container.querySelector<HTMLInputElement>(
          "#overlay-pressure",
        )!.checked = true;
      }
      map.setOverlays(overlays);
      cycleStart = performance.now();
      updateStatic();
      map.render(variable, day, 0, mode);
      status.textContent = "SHA-256 verified";
    } catch {
      status.textContent = "Layer unavailable";
    } finally {
      button?.classList.remove("is-loading");
    }
  }

  function updateInspector(): void {
    container.querySelector("#point-location")!.textContent =
      pointLabel(selection);
    container.querySelector("#point-chart")!.innerHTML = renderPointChart(
      activeGlobal,
      variable,
      selection,
      day,
      mode,
    );
    const definition = activeGlobal.metadata.variables[variable];
    container.querySelector("#notice-copy")!.textContent =
      mode === "anomaly" && definition.anomaly
        ? definition.anomaly.description
        : definition.interpretation;
  }

  function updateStatic(): void {
    const definition = activeGlobal.metadata.variables[variable];
    const displayDefinition =
      mode === "anomaly" && definition.anomaly
        ? definition.anomaly
        : definition;
    container
      .querySelectorAll<HTMLButtonElement>(".global-layer-button")
      .forEach((button) => {
        const active = button.dataset.variable === variable;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });
    container.querySelector("#lead-label")!.textContent =
      `Lead ${String(day + 1).padStart(2, "0")} / 42`;
    container.querySelector("#valid-label")!.textContent = friendlyDate(
      activeGlobal.metadata.valid_period_starts[day],
    );
    const guidance = leadGuidance(day);
    container.querySelector("#lead-period")!.textContent = guidance.period;
    container.querySelector("#lead-guidance")!.textContent =
      guidance.guidance;
    slider.value = String(day);
    slider.style.setProperty("--timeline-progress", `${(day / 41) * 100}%`);
    container.querySelector("#global-legend-label")!.textContent =
      displayDefinition.label;
    container.querySelector("#global-legend-units")!.textContent =
      displayDefinition.units;
    const legend = container.querySelector<HTMLDivElement>("#global-legend")!;
    legend.replaceChildren(createLegend(displayDefinition));
    const absoluteButton =
      container.querySelector<HTMLButtonElement>("#mode-absolute")!;
    const anomalyButton =
      container.querySelector<HTMLButtonElement>("#mode-anomaly")!;
    absoluteButton.classList.toggle("is-active", mode === "absolute");
    anomalyButton.classList.toggle("is-active", mode === "anomaly");
    const anomalyReady = Boolean(
      definition.anomaly && activeGlobal.anomalies[variable],
    );
    anomalyButton.disabled = !anomalyReady;
    anomalyButton.title = anomalyReady
      ? "Show departure from the exact-date, lead-matched model climate"
      : definition.anomaly
        ? "Loading matched anomaly field…"
      : "No matched global anomaly baseline is published for this field";
    container.querySelector("#field-mode-note")!.textContent =
      definition.anomaly && !anomalyReady
        ? "Mean map ready · loading ensemble spread and matched anomaly…"
        : definition.anomaly
        ? mode === "anomaly"
          ? "2002–2021 · exact 28 July initialization · lead matched"
          : "Anomaly available against the matched 2002–2021 model climate."
        : "Absolute field only · no matched global baseline published.";
    playButton.innerHTML = `<span aria-hidden="true">${playing ? "Ⅱ" : "▶"}</span>`;
    playButton.setAttribute(
      "aria-label",
      playing ? "Pause animation" : "Play animation",
    );
    speedButton.textContent = `${SPEEDS[speedIndex]}×`;
    updateInspector();
  }

  function animation(timestamp: number): void {
    if (playing) {
      const duration = BASE_FRAME_MS / SPEEDS[speedIndex];
      const elapsed = timestamp - cycleStart;
      if (elapsed >= duration) {
        const advance = Math.floor(elapsed / duration);
        day = (day + advance) % 42;
        cycleStart += advance * duration;
        updateStatic();
      }
      const linearMix = Math.min((timestamp - cycleStart) / duration, 1);
      const smoothMix = linearMix * linearMix * (3 - 2 * linearMix);
      map.render(variable, day, day === 41 ? 0 : smoothMix, mode);
    }
    animationFrame = window.requestAnimationFrame(animation);
  }

  playButton.addEventListener("click", () => {
    playing = !playing;
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0, mode);
  });
  speedButton.addEventListener("click", () => {
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    cycleStart = performance.now();
    updateStatic();
  });
  slider.addEventListener("input", () => {
    day = Number(slider.value);
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0, mode);
  });

  const countryInput =
    container.querySelector<HTMLInputElement>("#overlay-countries")!;
  const indiaInput =
    container.querySelector<HTMLInputElement>("#overlay-india")!;
  const windInput =
    container.querySelector<HTMLInputElement>("#overlay-wind")!;
  const pressureInput =
    container.querySelector<HTMLInputElement>("#overlay-pressure")!;
  const z500Input =
    container.querySelector<HTMLInputElement>("#overlay-z500")!;
  const updateOverlays = async (): Promise<void> => {
    overlays = {
      countryLabels: countryInput.checked,
      indiaStates: indiaInput.checked,
      windVectors: windInput.checked,
      pressureContours: pressureInput.checked,
      z500Contours: z500Input.checked,
    };
    if (overlays.windVectors) await activeGlobal.loadVariable("wind850");
    if (overlays.pressureContours) await activeGlobal.loadVariable("mslp");
    if (overlays.z500Contours) await activeGlobal.loadVariable("z500");
    map.setOverlays(overlays);
    map.render(variable, day, 0, mode);
  };
  [countryInput, indiaInput, windInput, pressureInput, z500Input].forEach(
    (input) => {
      input.addEventListener("change", () => void updateOverlays());
    },
  );
  container.querySelector("#mode-absolute")!.addEventListener("click", () => {
    mode = "absolute";
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0, mode);
  });
  container.querySelector("#mode-anomaly")!.addEventListener("click", () => {
    if (
      !activeGlobal.metadata.variables[variable].anomaly ||
      !activeGlobal.anomalies[variable]
    ) return;
    mode = "anomaly";
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0, mode);
  });
  container.querySelector("#zoom-in")!.addEventListener("click", () => {
    map.zoomBy(1.35);
  });
  container.querySelector("#zoom-out")!.addEventListener("click", () => {
    map.zoomBy(1 / 1.35);
  });
  container.querySelector("#india-focus")!.addEventListener("click", () => {
    indiaInput.checked = true;
    overlays.indiaStates = true;
    map.setOverlays(overlays);
    map.focusIndia();
  });
  container.querySelector("#world-focus")!.addEventListener("click", () => {
    map.resetView();
  });

  const keyboardHandler = (event: KeyboardEvent): void => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    day = Math.max(
      0,
      Math.min(41, day + (event.key === "ArrowRight" ? 1 : -1)),
    );
    playing = false;
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0, mode);
  };
  window.addEventListener("keydown", keyboardHandler);

  updateStatic();
  map.render(variable, day, 0, mode);
  const initialStatus = container.querySelector("#layer-load-status")!;
  initialStatus.textContent = "Loading diagnostics…";
  void activeGlobal.loadVariable("precipitation").then(
    () => {
      initialStatus.textContent = "SHA-256 verified";
      updateStatic();
      map.render(variable, day, 0, mode);
    },
    () => {
      initialStatus.textContent = "Mean ready · diagnostics unavailable";
    },
  );
  animationFrame = window.requestAnimationFrame(animation);

  return () => {
    window.cancelAnimationFrame(animationFrame);
    window.removeEventListener("keydown", keyboardHandler);
    map.destroy();
  };
}
