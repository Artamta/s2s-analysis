import { TercileProbabilityMap } from "../components/TercileProbabilityMap";
import type {
  AppData,
  InitialConditionSourceId,
  RegionalOutlookWeek,
  RegionalVariableSummary,
  TercileCategory,
  TercileProbabilityRecord,
} from "../types";

type OutlookVariable = "rainfall" | "temperature";

const CATEGORY_COPY: Record<
  OutlookVariable,
  Record<TercileCategory | "mixed", string>
> = {
  rainfall: {
    below_normal: "Below normal",
    near_normal: "Near normal",
    above_normal: "Above normal",
    mixed: "Mixed signal",
  },
  temperature: {
    below_normal: "Cooler than normal",
    near_normal: "Near normal",
    above_normal: "Warmer than normal",
    mixed: "Mixed signal",
  },
};

function friendlyDate(value: string, includeYear = false): string {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  }).format(new Date(`${value.slice(0, 10)}T00:00:00Z`));
}

function signed(value: number, units: string): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(2)} ${units}`;
}

function sourceLink(id: InitialConditionSourceId, issue: string): string {
  return `./?source=${id}&issue=${issue}#outlook`;
}

function probabilityBar(probability: TercileProbabilityRecord): string {
  return `
    <div class="outlook-probability-bar" aria-label="Below ${probability.below_normal} percent, near ${probability.near_normal} percent, above ${probability.above_normal} percent">
      <i class="is-below" style="width:${probability.below_normal}%"></i>
      <i class="is-near" style="width:${probability.near_normal}%"></i>
      <i class="is-above" style="width:${probability.above_normal}%"></i>
    </div>
    <div class="outlook-probability-values">
      <span>B ${probability.below_normal}%</span>
      <span>N ${probability.near_normal}%</span>
      <span>A ${probability.above_normal}%</span>
    </div>
  `;
}

function regionCard(
  region: RegionalOutlookWeek["regions"][number],
  variable: OutlookVariable,
): string {
  const values: RegionalVariableSummary = region[variable];
  const probability = values.tercile_probability_percent;
  const isRainfall = variable === "rainfall";
  const mean = isRainfall ? values.weekly_mean_mm_day! : values.weekly_mean_deg_c!;
  const anomaly = isRainfall ? values.anomaly_mm_day! : values.anomaly_deg_c!;
  const spread = isRainfall
    ? values.ensemble_spread_mm_day!
    : values.ensemble_spread_deg_c!;
  const units = isRainfall ? "mm/day" : "°C";
  const cardClass = region.id === "all_india" ? " is-all-india" : "";
  return `
    <article class="outlook-region-card${cardClass}">
      <header>
        <span>${region.label}</span>
        <strong class="signal-${probability.dominant_category}">${CATEGORY_COPY[variable][probability.dominant_category]}</strong>
      </header>
      <div class="outlook-region-signal">
        <b>${probability.dominant_probability}%</b>
        <span>members in leading tercile</span>
      </div>
      ${probabilityBar(probability)}
      <dl>
        <div><dt>Weekly mean</dt><dd>${mean.toFixed(2)} ${units}</dd></div>
        <div><dt>Anomaly</dt><dd>${signed(anomaly, units)}</dd></div>
        <div><dt>Spread</dt><dd>${spread.toFixed(2)} ${units}</dd></div>
      </dl>
    </article>
  `;
}

function headline(
  week: RegionalOutlookWeek,
  variable: OutlookVariable,
): string {
  const regional = week.regions.filter((region) => region.id !== "all_india");
  const strongest = regional.reduce((best, candidate) =>
    candidate[variable].tercile_probability_percent.dominant_probability >
    best[variable].tercile_probability_percent.dominant_probability
      ? candidate
      : best,
  );
  const probability = strongest[variable].tercile_probability_percent;
  const label = CATEGORY_COPY[variable][probability.dominant_category].toLowerCase();
  return `${strongest.label} has the strongest regional ensemble signal: ${probability.dominant_probability}% of members favour ${label}.`;
}

function unavailablePage(container: HTMLElement, data: AppData): void {
  const { forecast, index } = data;
  if (!forecast || !index) return;
  const sourceId = forecast.issue.initial_condition_source.id;
  const source = index.initial_condition_sources.find((item) => item.id === sourceId)!;
  const issueId = forecast.issue.initialization.slice(0, 10).replaceAll("-", "");
  const recommended = source.issues.find((issue) => issue.regional_outlook);
  container.innerHTML = `
    <section class="outlook-page outlook-unavailable">
      <span class="outlook-eyebrow">India · Regional Outlook</span>
      <h1>Probabilistic guidance is withheld for this issue.</h1>
      <p>The ${friendlyDate(forecast.issue.initialization, true)} run contains ${forecast.issue.members} members. Tercile probabilities are published only for complete 100-member ensembles.</p>
      <div class="outlook-unavailable__actions">
        ${recommended ? `<a class="outlook-primary-action" href="${sourceLink(sourceId, recommended.id)}">Open ${friendlyDate(recommended.initialization, true)} · ${recommended.members} members</a>` : ""}
        <a href="./?source=${sourceId}&issue=${issueId}#india">Return to deterministic India maps</a>
      </div>
    </section>
  `;
}

export function renderOutlookPage(container: HTMLElement, data: AppData): void {
  const { forecast, validation, index, indiaGeography, regionalOutlook } = data;
  if (!forecast || !validation || !index || !indiaGeography) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Regional data unavailable</span>
        <h1>The regional outlook package could not be loaded.</h1>
        <p>No partial probability product has been shown.</p>
      </section>
    `;
    return;
  }
  if (!validation.presentation_allowed) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Publication gate closed</span>
        <h1>This regional outlook is not available for presentation.</h1>
      </section>
    `;
    return;
  }
  if (!regionalOutlook) {
    unavailablePage(container, data);
    return;
  }
  if (
    regionalOutlook.issue.initialization !== forecast.issue.initialization ||
    regionalOutlook.issue.source_id !== forecast.issue.initial_condition_source.id
  ) {
    throw new Error("Regional outlook does not match the selected forecast issue");
  }

  const sourceId = forecast.issue.initial_condition_source.id;
  const source = index.initial_condition_sources.find((item) => item.id === sourceId)!;
  const currentIssueId = forecast.issue.initialization.slice(0, 10).replaceAll("-", "");

  container.innerHTML = `
    <section class="outlook-page">
      <header class="outlook-hero">
        <div>
          <span class="outlook-eyebrow">India · Regional Outlook</span>
          <h1>Where the ensemble signal is strongest.</h1>
          <p>Most-likely rainfall and temperature terciles for All India and IMD's four broad homogeneous regions.</p>
        </div>
        <div class="outlook-hero__status">
          <span>Experimental</span>
          <strong>Raw 100-member guidance</strong>
          <small>Not calibrated operational probability</small>
        </div>
      </header>

      <section class="outlook-runbar" aria-label="Regional outlook run selection">
        <div class="outlook-source-tabs">
          ${index.initial_condition_sources.map((candidate) => `
            <a href="${sourceLink(candidate.id, candidate.default_issue)}" class="${candidate.id === sourceId ? "is-active" : ""}">
              <strong>${candidate.short_label}</strong><span>${candidate.category === "operational_proxy" ? "Operational proxy" : "Reanalysis reference"}</span>
            </a>
          `).join("")}
        </div>
        <label>
          <span>Initialization</span>
          <select id="outlook-date-select">
            ${source.issues.map((issue) => `<option value="${issue.id}" ${issue.id === currentIssueId ? "selected" : ""}>${friendlyDate(issue.initialization, true)} · ${issue.members} members${issue.regional_outlook ? "" : " · probabilities unavailable"}</option>`).join("")}
          </select>
        </label>
        <a class="outlook-map-link" href="./?source=${sourceId}&issue=${currentIssueId}#india">Open four-map India forecast →</a>
      </section>

      <section class="outlook-control-deck">
        <div class="outlook-variable-tabs" aria-label="Regional outlook variable">
          <button type="button" data-variable="rainfall" class="is-active">Rainfall</button>
          <button type="button" data-variable="temperature">Temperature</button>
        </div>
        <div class="outlook-week-tabs" aria-label="Forecast week">
          ${regionalOutlook.weeks.map((week) => `<button type="button" data-week="${week.week}" class="${week.week === 1 ? "is-active" : ""}">W${week.week}</button>`).join("")}
        </div>
      </section>

      <section class="outlook-headline" aria-live="polite">
        <span id="outlook-valid-period"></span>
        <strong id="outlook-headline-copy"></strong>
        <p>Probability is the fraction of ensemble members in each model-climatology tercile; it is not a forecast-skill score.</p>
      </section>

      <div class="outlook-grid">
        <section class="outlook-map-card">
          <header>
            <div><span>Most-likely tercile</span><h2 id="outlook-map-title"></h2></div>
            <small>Deeper colour = larger member fraction</small>
          </header>
          <div class="outlook-map-frame" id="outlook-map-frame"></div>
          <div class="outlook-map-legend" id="outlook-map-legend"></div>
          <p>Visual interpolation affects presentation only. Hover values report the native 1.5° probability cell.</p>
        </section>
        <section class="outlook-regions" aria-label="Regional forecast summaries">
          <header>
            <span>Area-weighted summaries</span>
            <h2>All India and four broad regions</h2>
          </header>
          <div class="outlook-region-list" id="outlook-region-list"></div>
        </section>
      </div>

      <footer class="outlook-method-strip">
        <div><span>Forecast sample</span><strong>100 stochastic members</strong></div>
        <div><span>Reference sample</span><strong>20 yearly ensemble means · 2002–2021</strong></div>
        <div><span>Area weighting</span><strong>cos(latitude) × regional land fraction</strong></div>
        <p>${regionalOutlook.region_definition.interpretation} Experimental research guidance; not an operational warning.</p>
      </footer>
    </section>
  `;

  let selectedVariable: OutlookVariable = "rainfall";
  let selectedWeek = 1;
  const mapFrame = container.querySelector<HTMLElement>("#outlook-map-frame")!;
  const map = new TercileProbabilityMap(
    mapFrame,
    forecast,
    regionalOutlook,
    indiaGeography,
  );

  container.querySelector<HTMLSelectElement>("#outlook-date-select")!.addEventListener("change", (event) => {
    const issue = (event.currentTarget as HTMLSelectElement).value;
    window.location.assign(sourceLink(sourceId, issue));
  });
  container.querySelectorAll<HTMLButtonElement>(".outlook-variable-tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      selectedVariable = button.dataset.variable as OutlookVariable;
      update();
    });
  });
  container.querySelectorAll<HTMLButtonElement>(".outlook-week-tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      selectedWeek = Number(button.dataset.week);
      update();
    });
  });

  function update(): void {
    const week = regionalOutlook!.weeks[selectedWeek - 1];
    container.querySelector<HTMLElement>(".outlook-page")!.dataset.variable =
      selectedVariable;
    container.querySelectorAll<HTMLButtonElement>(".outlook-variable-tabs button").forEach((button) => {
      const active = button.dataset.variable === selectedVariable;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    container.querySelectorAll<HTMLButtonElement>(".outlook-week-tabs button").forEach((button) => {
      const active = Number(button.dataset.week) === selectedWeek;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    container.querySelector("#outlook-valid-period")!.textContent =
      `Week ${week.week} · ${friendlyDate(week.valid_start)} – ${friendlyDate(week.valid_end, true)}`;
    container.querySelector("#outlook-headline-copy")!.textContent = headline(
      week,
      selectedVariable,
    );
    container.querySelector("#outlook-map-title")!.textContent =
      `${selectedVariable === "rainfall" ? "Rainfall" : "2 m temperature"} · Week ${week.week}`;
    container.querySelector("#outlook-map-legend")!.innerHTML = selectedVariable === "rainfall"
      ? `<span class="is-below">Below normal</span><span class="is-near">Near normal</span><span class="is-above">Above normal</span>`
      : `<span class="is-below is-temperature">Cooler</span><span class="is-near">Near normal</span><span class="is-above is-temperature">Warmer</span>`;
    container.querySelector("#outlook-region-list")!.innerHTML = week.regions
      .map((region) => regionCard(region, selectedVariable))
      .join("");
    map.render(selectedVariable, week);
  }

  update();
}
