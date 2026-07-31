import {
  GlobalForecastMap,
  type MapSelection,
} from "../components/GlobalForecastMap";
import { createLegend } from "../components/Legend";
import type { AppData, GlobalVariableKey } from "../types";

const VARIABLE_ORDER: GlobalVariableKey[] = [
  "precipitation",
  "temperature",
  "z500",
];
const SPEEDS = [0.5, 1, 2];
const BASE_FRAME_MS = 1050;
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
  if (variable === "precipitation") return "◌";
  if (variable === "temperature") return "°";
  return "≋";
}

function pointLabel(selection: MapSelection): string {
  const latitude =
    selection.latitude < 0
      ? `${Math.abs(selection.latitude).toFixed(1)}°S`
      : `${selection.latitude.toFixed(1)}°N`;
  const longitude = selection.longitude > 180
    ? `${(360 - selection.longitude).toFixed(1)}°W`
    : `${selection.longitude.toFixed(1)}°E`;
  return `${latitude} · ${longitude}`;
}

function leadGuidance(day: number): { period: string; guidance: string } {
  if (day < 14) {
    return {
      period: "Days 1–14 · weather → subseasonal",
      guidance: "Synoptic features remain useful, but local timing becomes less certain with lead.",
    };
  }
  if (day < 28) {
    return {
      period: "Weeks 3–4 · subseasonal range",
      guidance: "Read persistent large-scale patterns and ensemble agreement—not exact day-to-day weather.",
    };
  }
  return {
    period: "Weeks 5–6 · broad outlook",
    guidance: "Use only broad circulation and ensemble tendencies; local detail has low predictability.",
  };
}

function noticeFor(variable: GlobalVariableKey): string {
  if (variable === "precipitation") {
    return "Watch rain belts, monsoon organization and storm-track shifts; isolated point peaks are less robust later.";
  }
  if (variable === "temperature") {
    return "Watch persistent warm or cool zones and hemispheric gradients, especially where members remain clustered.";
  }
  return "Watch broad ridges, troughs and planetary-wave evolution—the clearest circulation-scale S2S signal.";
}

function linePath(values: number[], x: (index: number) => number, y: (value: number) => number): string {
  return values
    .map((value, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(value).toFixed(1)}`)
    .join(" ");
}

function renderPointChart(
  data: AppData["global"],
  variable: GlobalVariableKey,
  selection: MapSelection,
  selectedDay: number,
): string {
  const definition = data.metadata.variables[variable];
  const pointIndex =
    selection.latitudeIndex * 240 + selection.longitudeIndex;
  const means = Array.from({ length: 42 }, (_, index) => {
    const encoded = data.fields[variable][index * FRAME_SIZE + pointIndex];
    return encoded * definition.scale + definition.offset;
  });
  const spreads = Array.from({ length: 42 }, (_, index) => {
    const encoded = data.spreads[variable][index * FRAME_SIZE + pointIndex];
    return encoded * definition.spread.scale + definition.spread.offset;
  });
  const lower = means.map((mean, index) =>
    variable === "precipitation"
      ? Math.max(0, mean - spreads[index])
      : mean - spreads[index],
  );
  const upper = means.map((mean, index) => mean + spreads[index]);
  let minimum = Math.min(...lower);
  let maximum = Math.max(...upper);
  const rawRange = Math.max(maximum - minimum, Math.abs(maximum) * 0.05, 1);
  const padding = rawRange * 0.1;
  minimum = variable === "precipitation" ? 0 : minimum - padding;
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
    top + ((maximum - value) / (maximum - minimum)) * (height - top - bottom);
  const band = [
    ...upper.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`),
    ...lower
      .map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`)
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
  const decimals = variable === "precipitation" && mean >= 10 ? 0 : 1;

  return `
    <div class="point-value">
      <strong>${mean.toFixed(decimals)}</strong>
      <span>± ${spread.toFixed(decimals)} ${definition.units}</span>
      <small>Lead ${String(selectedDay + 1).padStart(2, "0")} · ensemble mean ±1σ</small>
    </div>
    <svg class="point-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="42-day ensemble mean and member spread at ${pointLabel(selection)}">
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

export function renderGlobalPage(container: HTMLElement, data: AppData): () => void {
  const { global } = data;
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
        <span class="global-eyebrow">Experimental ensemble guidance · Issue 01</span>
        <h1>Atmosphere<br><em>in motion.</em></h1>
        <p>Follow rain, heat and mid-tropospheric circulation from the weather range into Weeks 5–6.</p>
      </div>

      <aside class="global-facts" aria-label="Forecast issue and integrity facts">
        <div><span>Initialized</span><strong>28 Jul 2026 · 00 UTC</strong></div>
        <div><span>Signal</span><strong>${global.metadata.issue.members}-member ensemble mean</strong></div>
        <div><span>Verification</span><strong>Awaiting future observations</strong></div>
        <div><span>Integrity</span><strong class="integrity-ok">✓ 6 files SHA-256 verified</strong></div>
      </aside>

      <div class="global-layer-dock" id="global-layer-dock" aria-label="Map layers"></div>

      <aside class="global-inspector" aria-label="Selected-location 42-day ensemble trace">
        <div class="global-inspector__head">
          <span>Click map · 42-day point trace</span>
          <strong id="point-location">28.5°N · 76.5°E</strong>
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

      <p class="global-disclaimer">Research product · Daily model fields · Long leads are broad guidance, not deterministic local forecasts</p>
    </section>
  `;

  let variable: GlobalVariableKey = "precipitation";
  let day = 0;
  let selection: MapSelection = {
    latitude: 28.5,
    longitude: 76.5,
    latitudeIndex: 41,
    longitudeIndex: 51,
  };
  let speedIndex = 1;
  let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let cycleStart = performance.now();
  let animationFrame = 0;
  const mapHost = container.querySelector<HTMLElement>("#global-map")!;
  const layerDock =
    container.querySelector<HTMLDivElement>("#global-layer-dock")!;
  const slider = container.querySelector<HTMLInputElement>("#day-slider")!;
  const playButton =
    container.querySelector<HTMLButtonElement>("#play-button")!;
  const speedButton =
    container.querySelector<HTMLButtonElement>("#speed-button")!;
  const map = new GlobalForecastMap(mapHost, global, data.world, (next) => {
    selection = next;
    updateInspector();
  });

  VARIABLE_ORDER.forEach((key) => {
    const definition = global.metadata.variables[key];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "global-layer-button";
    button.dataset.variable = key;
    button.innerHTML = `
      <span class="layer-glyph" aria-hidden="true">${variableGlyph(key)}</span>
      <span><strong>${definition.short_label}</strong><small>${definition.units}</small></span>
    `;
    button.addEventListener("click", () => {
      variable = key;
      cycleStart = performance.now();
      updateStatic();
      map.render(variable, day, 0);
    });
    layerDock.append(button);
  });

  function updateInspector(): void {
    container.querySelector("#point-location")!.textContent =
      pointLabel(selection);
    container.querySelector("#point-chart")!.innerHTML =
      renderPointChart(global, variable, selection, day);
    container.querySelector("#notice-copy")!.textContent =
      noticeFor(variable);
  }

  function updateStatic(): void {
    const definition = global.metadata.variables[variable];
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
      global.metadata.valid_period_starts[day],
    );
    const guidance = leadGuidance(day);
    container.querySelector("#lead-period")!.textContent = guidance.period;
    container.querySelector("#lead-guidance")!.textContent =
      guidance.guidance;
    slider.value = String(day);
    slider.style.setProperty("--timeline-progress", `${(day / 41) * 100}%`);
    container.querySelector("#global-legend-label")!.textContent =
      definition.label;
    container.querySelector("#global-legend-units")!.textContent =
      definition.units;
    const legend = container.querySelector<HTMLDivElement>("#global-legend")!;
    legend.replaceChildren(createLegend(definition));
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
      const mix = Math.min((timestamp - cycleStart) / duration, 1);
      map.render(variable, day, day === 41 ? 0 : mix);
    }
    animationFrame = window.requestAnimationFrame(animation);
  }

  playButton.addEventListener("click", () => {
    playing = !playing;
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0);
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
    map.render(variable, day, 0);
  });
  const keyboardHandler = (event: KeyboardEvent): void => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    day = Math.max(0, Math.min(41, day + (event.key === "ArrowRight" ? 1 : -1)));
    playing = false;
    cycleStart = performance.now();
    updateStatic();
    map.render(variable, day, 0);
  };
  window.addEventListener("keydown", keyboardHandler);

  updateStatic();
  map.render(variable, day, 0);
  animationFrame = window.requestAnimationFrame(animation);

  return () => {
    window.cancelAnimationFrame(animationFrame);
    window.removeEventListener("keydown", keyboardHandler);
    map.destroy();
  };
}
