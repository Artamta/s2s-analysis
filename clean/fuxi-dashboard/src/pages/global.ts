import { GlobalForecastMap } from "../components/GlobalForecastMap";
import { createLegend } from "../components/Legend";
import type { AppData, GlobalVariableKey } from "../types";

const VARIABLE_ORDER: GlobalVariableKey[] = [
  "precipitation",
  "temperature",
  "z500",
];
const SPEEDS = [0.5, 1, 2];
const BASE_FRAME_MS = 1050;

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
        <p>One global view. Three essential fields. Forty-two daily forecast periods.</p>
      </div>

      <aside class="global-facts" aria-label="Forecast issue facts">
        <div><span>Initialized</span><strong>28 Jul 2026 · 00 UTC</strong></div>
        <div><span>Signal</span><strong>${global.metadata.issue.members}-member mean</strong></div>
        <div><span>Resolution</span><strong>1.5° global grid</strong></div>
      </aside>

      <div class="global-layer-dock" id="global-layer-dock" aria-label="Map layers"></div>

      <div class="global-legend-panel">
        <div>
          <span id="global-legend-label"></span>
          <strong id="global-legend-units"></strong>
        </div>
        <div id="global-legend"></div>
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

      <p class="global-disclaimer">Research product · Daily fields · Visual transitions do not create additional forecast times</p>
    </section>
  `;

  let variable: GlobalVariableKey = "precipitation";
  let day = 0;
  let speedIndex = 1;
  let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let cycleStart = performance.now();
  let animationFrame = 0;
  const mapHost = container.querySelector<HTMLElement>("#global-map")!;
  const map = new GlobalForecastMap(mapHost, global, data.world);
  const layerDock =
    container.querySelector<HTMLDivElement>("#global-layer-dock")!;
  const slider = container.querySelector<HTMLInputElement>("#day-slider")!;
  const playButton =
    container.querySelector<HTMLButtonElement>("#play-button")!;
  const speedButton =
    container.querySelector<HTMLButtonElement>("#speed-button")!;

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
