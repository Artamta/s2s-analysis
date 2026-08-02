import { ForecastMap } from "../components/ForecastMap";
import { createLegend } from "../components/Legend";
import type {
  AppData,
  ForecastWeek,
  InitialConditionSourceId,
  ProductDefinition,
  ProductKey,
} from "../types";

const PRODUCT_ORDER: ProductKey[] = [
  "rainfall_total",
  "rainfall_anomaly",
  "temperature_mean",
  "temperature_anomaly",
];

const PRESENTATION_PRODUCTS: Record<ProductKey, ProductDefinition> = {
  rainfall_total: {
    label: "Weekly mean rainfall",
    short_label: "Rainfall",
    description: "Mean daily rainfall rate over each 7-day forecast week",
    units: "mm day⁻¹",
    baseline: null,
    legend: {
      boundaries: [0, 1, 2, 5, 10, 20, 40, 60],
      colors: [
        "#ffffff",
        "#b7ffb8",
        "#71f27b",
        "#24d13b",
        "#009a18",
        "#006b12",
        "#003d0c",
      ],
      under: "#ffffff",
      over: "#002807",
    },
  },
  rainfall_anomaly: {
    label: "Weekly mean rainfall anomaly",
    short_label: "Rainfall anomaly",
    description: "Difference from the model's typical rainfall for the same season and forecast lead",
    units: "mm day⁻¹",
    baseline: "Native reforecasts, 2002–2021",
    legend: {
      boundaries: [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20],
      colors: [
        "#ff5200",
        "#ff8e1d",
        "#ffca59",
        "#fff4a5",
        "#ffffff",
        "#c8c8e9",
        "#8c8cbf",
        "#6464a3",
        "#3c3c87",
      ],
      under: "#d70e00",
      over: "#00001e",
    },
  },
  temperature_mean: {
    label: "Weekly mean 2 m temperature",
    short_label: "Temperature",
    description: "Mean of seven daily-mean 2 m temperature forecasts",
    units: "°C",
    baseline: null,
    legend: {
      boundaries: [10, 14, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
      colors: [
        "#bebee2",
        "#fffacd",
        "#fff191",
        "#ffe271",
        "#ffca59",
        "#ffa635",
        "#ff8e1d",
        "#ff6a00",
        "#ff3a00",
        "#eb1800",
        "#c30400",
        "#9b0000",
        "#730000",
      ],
      under: "#9696c6",
      over: "#5f0000",
    },
  },
  temperature_anomaly: {
    label: "Weekly mean temperature anomaly",
    short_label: "Temperature anomaly",
    description: "Difference from the model's typical temperature for the same season and forecast lead",
    units: "°C",
    baseline: "Native reforecasts, 2002–2021",
    legend: {
      boundaries: [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
      colors: [
        "#383880",
        "#5a5a9c",
        "#8282b8",
        "#aaaad4",
        "#c8c8e9",
        "#dcdcf7",
        "#fffacd",
        "#ffa635",
        "#ff7605",
        "#ff5e00",
        "#ff2e00",
        "#c30400",
      ],
      under: "#10103a",
      over: "#730000",
    },
  },
};

function friendlyDate(isoDate: string, includeYear = false): string {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  }).format(new Date(`${isoDate}T12:00:00Z`));
}

function displayWeek(week: ForecastWeek, product: ProductKey): ForecastWeek {
  if (product !== "rainfall_total") return week;
  return {
    ...week,
    fields: {
      ...week.fields,
      rainfall_total: week.fields.rainfall_total.map((value) => value / 7),
    },
  };
}

export function renderForecastPage(container: HTMLElement, data: AppData): void {
  const { forecast, validation, index, indiaGeography } = data;
  if (!forecast || !validation || !index || !indiaGeography) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">India data unavailable</span>
        <h1>The India forecast package could not be loaded.</h1>
        <p>No partial map has been shown.</p>
      </section>
    `;
    return;
  }
  const activeForecast = forecast;
  const activeGeography = indiaGeography;
  if (!validation.presentation_allowed) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Publication gate closed</span>
        <h1>This forecast is not available for presentation.</h1>
        <p>A scientific or publication check failed, so no partial map is shown.</p>
      </section>
    `;
    return;
  }
  const sourceId = forecast.issue.initial_condition_source.id;
  const source = index.initial_condition_sources.find(
    (candidate) => candidate.id === sourceId,
  );
  if (!source) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Run unavailable</span>
        <h1>The selected initialization source is not registered.</h1>
        <p>No partial map has been shown.</p>
      </section>
    `;
    return;
  }
  const currentIssueId = forecast.issue.initialization.slice(0, 10).replaceAll("-", "");
  const sourceLink = (id: InitialConditionSourceId, issue: string): string =>
    `./?source=${id}&issue=${issue}#india`;
  const matchedSource = index.initial_condition_sources.find(
    (candidate) =>
      candidate.id !== sourceId &&
      candidate.issues.some((issue) => issue.id === currentIssueId),
  );
  const sourceBadge = sourceId === "gfs"
    ? "Experimental operational proxy"
    : "Delayed reference run";
  const pdfPath = forecast.issue.downloads.india_pdf;
  const pdfReady = typeof pdfPath === "string" && pdfPath.endsWith(".pdf");
  const pdfChecksum = forecast.issue.downloads.india_pdf_sha256;
  const pdfVersion = typeof pdfChecksum === "string" && pdfChecksum.length >= 12
    ? pdfChecksum.slice(0, 12)
    : encodeURIComponent(forecast.generated_at);
  const pdfUrl = pdfReady
    ? `./${pdfPath}?v=${pdfVersion}`
    : "";
  const initializationComparison = forecast.issue.initialization_comparison;
  const comparisonCard = initializationComparison
    ? `
      <section class="india-ic-comparison" aria-label="Matched GFS and ERA5 initialization comparison">
        <div>
          <span>Matched initialization sensitivity · Week 1</span>
          <strong>GFS − ERA5</strong>
          <p>This measures the effect of changing the initial-state source, not forecast skill. Observations are required before either source can be called better.</p>
        </div>
        <dl>
          <div><dt>India rainfall</dt><dd>${initializationComparison.week1_rainfall_gfs_minus_era5_mm_day >= 0 ? "+" : ""}${initializationComparison.week1_rainfall_gfs_minus_era5_mm_day.toFixed(2)} mm/day</dd></div>
          <div><dt>India temperature</dt><dd>${initializationComparison.week1_temperature_gfs_minus_era5_deg_c >= 0 ? "+" : ""}${initializationComparison.week1_temperature_gfs_minus_era5_deg_c.toFixed(2)} °C</dd></div>
        </dl>
      </section>`
    : "";

  container.innerHTML = `
    <section class="india-sheet">
      <header class="india-sheet__header">
        <div>
          <span class="india-sheet__brand">S2S RESEARCH · EXPERIMENTAL SUBSEASONAL FORECASTING</span>
          <h1 id="india-product-title">Weekly-mean rainfall anomaly</h1>
          <p>${friendlyDate(forecast.issue.initialization.slice(0, 10), true)} forecast start <i>•</i> ${forecast.issue.input_days.map((day) => friendlyDate(day)).join("–")} daily-mean inputs <i>•</i> ${forecast.issue.members}-member ensemble mean <i>•</i> Weeks 1–6</p>
        </div>
        <span class="india-experimental">${sourceBadge}</span>
      </header>

      <section class="india-run-console" aria-label="Forecast run selection and downloads">
        <div class="india-run-console__heading">
          <div>
            <span>Initial-condition source</span>
            <strong>${forecast.issue.initial_condition_source.label}</strong>
          </div>
          <p>${forecast.issue.initial_condition_source.description}</p>
        </div>
        <div class="india-source-tabs" role="navigation" aria-label="Initial-condition source">
          ${index.initial_condition_sources.map((candidate) => `
            <a href="${sourceLink(candidate.id, candidate.default_issue)}" class="${candidate.id === sourceId ? "is-active" : ""}" ${candidate.id === sourceId ? "aria-current=\"page\"" : ""}>
              <span>${candidate.short_label}</span>
              <small>${candidate.category === "operational_proxy" ? "Operational proxy" : "Reanalysis reference"}</small>
            </a>
          `).join("")}
        </div>
        <label class="india-date-select">
          <span>Initialization date</span>
          <select id="india-date-select" aria-label="Forecast initialization date">
            ${source.issues.map((issue) => `<option value="${issue.id}" ${issue.id === currentIssueId ? "selected" : ""}>${friendlyDate(issue.initialization.slice(0, 10), true)} · ${issue.members} members${issue.role === "rapid_prototype" ? " · limited ensemble" : ""}</option>`).join("")}
          </select>
        </label>
        <div class="india-run-actions" aria-label="Forecast downloads">
          <span>Forecast briefing</span>
          ${pdfReady
            ? `<a class="india-pdf-download" href="${pdfUrl}" download type="application/pdf">Download PDF</a>`
            : `<button class="india-pdf-download" type="button" disabled>PDF preparing · refresh shortly</button>`}
        </div>
        ${matchedSource ? `<a class="india-matched-run" href="${sourceLink(matchedSource.id, currentIssueId)}">Same ${friendlyDate(forecast.issue.initialization.slice(0, 10), true)} issue is available with ${matchedSource.short_label} initial conditions →</a>` : ""}
      </section>

      ${comparisonCard}

      <div class="india-toolbar">
        <div class="india-product-tabs" id="india-product-tabs" aria-label="India forecast field"></div>
        <div class="india-range-tabs" aria-label="Displayed forecast weeks">
          <button type="button" data-range="0" class="is-active">Weeks 1–4</button>
          <button type="button" data-range="2">Weeks 3–6</button>
        </div>
      </div>

      <div class="india-panel-grid" id="india-panel-grid"></div>
      <div class="india-shared-legend" id="india-shared-legend"></div>
      <p class="india-visual-note" id="india-visual-note"></p>

      <footer class="india-sheet__footer">
        <p id="india-baseline-note"></p>
        <span>Experimental S2S research guidance · Not an operational weather forecast or warning</span>
      </footer>
    </section>
  `;

  let selectedProduct: ProductKey = "rainfall_anomaly";
  let rangeStart = 0;
  const productTabs = container.querySelector<HTMLDivElement>("#india-product-tabs")!;
  const panelGrid = container.querySelector<HTMLDivElement>("#india-panel-grid")!;

  container.querySelector<HTMLSelectElement>("#india-date-select")!.addEventListener("change", (event) => {
    const issue = (event.currentTarget as HTMLSelectElement).value;
    const url = new URL(window.location.href);
    url.searchParams.delete("view");
    url.searchParams.set("source", sourceId);
    url.searchParams.set("issue", issue);
    url.hash = "india";
    window.location.assign(url);
  });

  PRODUCT_ORDER.forEach((productKey) => {
    const product = PRESENTATION_PRODUCTS[productKey];
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.product = productKey;
    button.innerHTML = `<span class="product-swatch product-swatch--${productKey}" aria-hidden="true"></span><strong>${product.short_label}</strong>`;
    button.addEventListener("click", () => {
      selectedProduct = productKey;
      update();
    });
    productTabs.append(button);
  });

  container
    .querySelectorAll<HTMLButtonElement>(".india-range-tabs button")
    .forEach((button) => {
      button.addEventListener("click", () => {
        rangeStart = Number(button.dataset.range);
        update();
      });
    });

  function update(): void {
    const product = PRESENTATION_PRODUCTS[selectedProduct];
    container
      .querySelectorAll<HTMLButtonElement>(".india-product-tabs button")
      .forEach((button) => {
        const active = button.dataset.product === selectedProduct;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });
    container
      .querySelectorAll<HTMLButtonElement>(".india-range-tabs button")
      .forEach((button) => {
        const active = Number(button.dataset.range) === rangeStart;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });

    container.querySelector("#india-product-title")!.textContent = product.label;
    panelGrid.replaceChildren();
    activeForecast.weeks.slice(rangeStart, rangeStart + 4).forEach((sourceWeek) => {
      const week = displayWeek(sourceWeek, selectedProduct);
      const panel = document.createElement("article");
      panel.className = "india-map-panel";
      panel.innerHTML = `
        <h2><strong>Week ${week.week}</strong><i>|</i>${friendlyDate(week.valid_start)} – ${friendlyDate(week.valid_end, true)}</h2>
        <div class="india-map-frame"></div>
      `;
      panelGrid.append(panel);
      const frame = panel.querySelector<HTMLDivElement>(".india-map-frame")!;
      const map = new ForecastMap(
        frame,
        activeForecast,
        activeGeography,
      );
      map.render(selectedProduct, week, product);
    });

    const legend = container.querySelector<HTMLDivElement>("#india-shared-legend")!;
    legend.style.setProperty("--legend-under", product.legend.under);
    legend.style.setProperty("--legend-over", product.legend.over);
    legend.replaceChildren(createLegend(product));
    container.querySelector("#india-visual-note")!.textContent =
      `${product.description.replace("100-member", `${activeForecast.issue.members}-member`)}. Visual-only bilinear interpolation 1.5°→0.25°; hover values remain native-grid values.`;
    container.querySelector("#india-baseline-note")!.textContent = product.baseline
      ? "Anomaly is the forecast minus the model's typical value for the same season and forecast lead, estimated from 2002–2021 reforecasts."
      : sourceId === "gfs"
        ? "Experimental initialization from operational analysis and short-range forecast proxy inputs."
        : "Delayed ERA5 reference initialization; not near-real-time operational guidance.";
  }

  update();
}
