import {
  ForecastMap,
  type WindRenderingMode,
} from "../components/ForecastMap";
import { createLegend } from "../components/Legend";
import {
  defaultIssueForSource,
  INITIAL_CONDITION_SOURCE_DISPLAY_ORDER,
  issueIsCurrent,
  productKeysForForecast,
  sourcesForDisplay,
} from "../lib/catalog";
import type {
  AppData,
  ForecastWeek,
  ForecastData,
  InitializationComparisonData,
  InitialConditionSourceId,
  ProductDefinition,
  ProductKey,
} from "../types";

const COMPARISON_PRODUCTS = {
  rainfall: {
    sourceKey: "rainfall_total" as ProductKey,
    differenceProduct: {
      label: "Rainfall difference",
      short_label: "Rainfall difference",
      description: "Pairwise ensemble-mean rainfall difference",
      units: "mm day⁻¹",
      baseline: null,
      legend: {
        boundaries: [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20],
        colors: [
          "#8b2b15", "#cf6135", "#efaa74", "#f8d8bd", "#f7f5ef",
          "#d8e8f0", "#a8cfdf", "#5fa4c1", "#26769f",
        ],
        under: "#5f180d",
        over: "#114c72",
      },
    } satisfies ProductDefinition,
  },
  temperature: {
    sourceKey: "temperature_mean" as ProductKey,
    differenceProduct: {
      label: "Temperature difference",
      short_label: "Temperature difference",
      description: "Pairwise ensemble-mean temperature difference",
      units: "°C",
      baseline: null,
      legend: {
        boundaries: [-6, -4, -3, -2, -1, 1, 2, 3, 4, 6],
        colors: [
          "#243b75", "#5279ad", "#8fb4d2", "#c9dce8", "#f7f5ef",
          "#f4d0b5", "#e99461", "#cc5739", "#8f271f",
        ],
        under: "#14234f",
        over: "#63140f",
      },
    } satisfies ProductDefinition,
  },
};

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
  wind850_anomaly: {
    label: "850 hPa wind anomaly",
    short_label: "850 hPa wind anomaly",
    description: "Shading shows the weekly wind-speed anomaly; the directional overlay follows the U/V vector-component anomaly",
    units: "m s⁻¹",
    baseline: "Native reforecasts, 2002–2021",
    legend: {
      boundaries: [-8, -6, -4, -2, -1, 1, 2, 4, 6, 8],
      colors: [
        "#6f321e",
        "#a8552f",
        "#d98b57",
        "#efc79f",
        "#f7f5ef",
        "#c9e2ee",
        "#83bdd4",
        "#3988b2",
        "#14567f",
      ],
      under: "#421c12",
      over: "#083653",
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
  const rainfall = week.fields.rainfall_total;
  if (!rainfall) return week;
  return {
    ...week,
    fields: {
      ...week.fields,
      rainfall_total: rainfall.map((value) => value / 7),
    },
  };
}

function renderInitializationComparison(
  container: HTMLElement,
  comparison: InitializationComparisonData,
  forecasts: Partial<Record<InitialConditionSourceId, ForecastData>>,
  geography: AppData["indiaGeography"],
): void {
  if (!geography || !comparison.pairs.length) return;
  const sourceOrder = INITIAL_CONDITION_SOURCE_DISPLAY_ORDER.filter(
    (source) => source in comparison.sources && forecasts[source],
  );
  if (sourceOrder.length < 2) return;
  const host = container.querySelector<HTMLDivElement>("#india-comparison-explorer");
  if (!host) return;
  let weekIndex = 0;
  let variable: keyof typeof COMPARISON_PRODUCTS = "rainfall";

  host.innerHTML = `
    <div class="india-comparison-toolbar">
      <div role="group" aria-label="Comparison variable">
        <button type="button" data-variable="rainfall" class="is-active">Rainfall</button>
        <button type="button" data-variable="temperature">Temperature</button>
      </div>
      <label>Forecast week
        <select aria-label="Comparison forecast week">
          ${Array.from({ length: 6 }, (_, index) => `<option value="${index}">Week ${index + 1}</option>`).join("")}
        </select>
      </label>
    </div>
    <h3>Ensemble means</h3>
    <div class="india-comparison-grid" data-comparison-sources></div>
    <h3>Pairwise differences and India metrics</h3>
    <div class="india-comparison-grid" data-comparison-pairs></div>
  `;
  const sourceGrid = host.querySelector<HTMLDivElement>("[data-comparison-sources]")!;
  const pairGrid = host.querySelector<HTMLDivElement>("[data-comparison-pairs]")!;

  const render = (): void => {
    const definition = COMPARISON_PRODUCTS[variable];
    sourceGrid.replaceChildren();
    sourceOrder.forEach((sourceId) => {
      const sourceForecast = forecasts[sourceId]!;
      const week = displayWeek(sourceForecast.weeks[weekIndex], definition.sourceKey);
      const panel = document.createElement("article");
      panel.className = "india-comparison-panel";
      panel.innerHTML = `<h4>${sourceForecast.issue.initial_condition_source.short_label}</h4><div class="india-map-frame"></div>`;
      sourceGrid.append(panel);
      new ForecastMap(
        panel.querySelector<HTMLDivElement>(".india-map-frame")!,
        sourceForecast,
        geography,
      ).render(
        definition.sourceKey,
        week,
        PRESENTATION_PRODUCTS[definition.sourceKey],
      );
    });

    pairGrid.replaceChildren();
    comparison.pairs.forEach((pair) => {
      const comparisonWeek = pair.weeks[weekIndex];
      const metrics = comparisonWeek[variable];
      const base = forecasts[pair.left_source_id]!;
      const fieldWeek: ForecastWeek = {
        week: comparisonWeek.week,
        valid_start: comparisonWeek.valid_start,
        valid_end: comparisonWeek.valid_end,
        fields: { [definition.sourceKey]: metrics.difference_field },
        summary: {},
      };
      const panel = document.createElement("article");
      panel.className = "india-comparison-panel";
      panel.innerHTML = `
        <h4>${pair.label}</h4>
        <div class="india-map-frame"></div>
        <dl>
          <div><dt>India mean</dt><dd>${metrics.area_weighted_mean_difference >= 0 ? "+" : ""}${metrics.area_weighted_mean_difference.toFixed(2)} ${definition.differenceProduct.units}</dd></div>
          <div><dt>Mean absolute difference</dt><dd>${metrics.mean_absolute_difference.toFixed(2)}</dd></div>
          <div><dt>Pattern correlation</dt><dd>${metrics.ensemble_mean_pattern_correlation.toFixed(2)}</dd></div>
          <div><dt>Spread · left / right</dt><dd>${metrics.left_mean_spread.toFixed(2)} / ${metrics.right_mean_spread.toFixed(2)}</dd></div>
        </dl>
      `;
      pairGrid.append(panel);
      new ForecastMap(
        panel.querySelector<HTMLDivElement>(".india-map-frame")!,
        base,
        geography,
      ).render(definition.sourceKey, fieldWeek, definition.differenceProduct);
    });
  };

  host.querySelectorAll<HTMLButtonElement>("[data-variable]").forEach((button) => {
    button.addEventListener("click", () => {
      variable = button.dataset.variable as keyof typeof COMPARISON_PRODUCTS;
      host.querySelectorAll<HTMLButtonElement>("[data-variable]").forEach((candidate) =>
        candidate.classList.toggle("is-active", candidate === button));
      render();
    });
  });
  host.querySelector<HTMLSelectElement>("select")!.addEventListener("change", (event) => {
    weekIndex = Number((event.currentTarget as HTMLSelectElement).value);
    render();
  });
  render();
}

export function renderForecastPage(container: HTMLElement, data: AppData): void {
  const {
    forecast,
    validation,
    index,
    indiaGeography,
    domainCatalog,
    activeDomain,
  } = data;
  if (!forecast || !validation || !index || !indiaGeography) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Forecast data unavailable</span>
        <h1>The selected domain forecast package could not be loaded.</h1>
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
  const catalogIssue = source.issues.find((issue) => issue.id === currentIssueId);
  if (!catalogIssue) {
    throw new Error("The loaded forecast is not present in the issue catalogue");
  }
  const isCurrent = issueIsCurrent(index, source, catalogIssue);
  const availableProducts = productKeysForForecast(
    forecast,
    catalogIssue.available_products,
  );
  if (availableProducts.length === 0) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Products unavailable</span>
        <h1>This issue has no publishable fields for the selected domain.</h1>
        <p>The catalogue remains available, but no partial map has been shown.</p>
        <a href="./#archive">Return to the forecast archive</a>
      </section>
    `;
    return;
  }
  const defaultProduct: ProductKey = availableProducts.includes("rainfall_anomaly")
    ? "rainfall_anomaly"
    : availableProducts[0];
  const sourceLink = (id: InitialConditionSourceId, issue: string): string =>
    `./?source=${id}&issue=${issue}${activeDomain && activeDomain.id !== "india" ? `&domain=${activeDomain.id}` : ""}#india`;
  const matchedSource = index.initial_condition_sources.find(
    (candidate) =>
      candidate.id !== sourceId &&
      candidate.issues.some((issue) => issue.id === currentIssueId),
  );
  const sourceBadge = isCurrent
    ? "Current · Experimental GFS proxy"
    : catalogIssue.members < 100
      ? "Archive · Limited experiment"
      : sourceId === "era5"
        ? "Archive · Delayed ERA5 reference"
        : "Archive · Complete ensemble";
  const pdfPath = forecast.issue.downloads?.domain_pdf ??
    forecast.issue.downloads?.india_pdf;
  const pdfReady = typeof pdfPath === "string" && pdfPath.endsWith(".pdf");
  const pdfChecksum = forecast.issue.downloads?.domain_pdf_sha256 ??
    forecast.issue.downloads?.india_pdf_sha256;
  const pdfVersion = typeof pdfChecksum === "string" && pdfChecksum.length >= 12
    ? pdfChecksum.slice(0, 12)
    : encodeURIComponent(forecast.generated_at);
  const pdfUrl = pdfReady
    ? `./${pdfPath}?v=${pdfVersion}`
    : "";
  const initializationComparison = data.initializationComparison;
  const validThrough = forecast.weeks.at(-1)?.valid_end ?? "";
  const archiveBanner = !isCurrent
    ? `
      <aside class="india-archive-banner" aria-label="Archived forecast notice">
        <div>
          <strong>${sourceId === "era5" ? "Delayed reference—not the current forecast" : "You are viewing an archived issue"}</strong>
          <p>${sourceId === "era5"
            ? "ERA5 initialization is retained for controlled reference experiments and normally arrives 5–7 days after real time."
            : catalogIssue.members < 100
            ? "This limited-member experiment is preserved for research transparency and must not be compared as a full operational ensemble."
              : "The current page always points to the newest validated GFS-proxy issue."}</p>
        </div>
        <a href="./#india">Return to current forecast →</a>
      </aside>
    `
    : "";
  const comparisonCard = activeDomain?.id !== "india"
    ? ""
    : initializationComparison
    ? `
      <section class="india-ic-comparison" aria-label="Matched initialization-source comparison">
        <div>
          <span>Matched initialization sensitivity</span>
          <strong>${Object.keys(initializationComparison.sources).map((id) => id.toUpperCase()).join(" · ")}</strong>
          <p>${initializationComparison.interpretation}</p>
        </div>
        <div id="india-comparison-explorer"></div>
      </section>`
    : "";

  container.innerHTML = `
    <section class="india-sheet">
      <header class="india-sheet__header">
        <div>
          <span class="india-sheet__brand">S2S RESEARCH · EXPERIMENTAL SUBSEASONAL FORECASTING</span>
          <h1 id="india-product-title">${PRESENTATION_PRODUCTS[defaultProduct].label}</h1>
          <p>${activeDomain?.label ?? "India"} <i>•</i> Initialized ${friendlyDate(forecast.issue.initialization.slice(0, 10), true)} <i>•</i> Valid through ${friendlyDate(validThrough, true)} <i>•</i> Weeks 1–6</p>
        </div>
        <span class="india-experimental">${sourceBadge}</span>
      </header>

      <section class="india-run-console" aria-label="Forecast run selection and downloads">
        ${domainCatalog && domainCatalog.domains.length > 1 ? `
          <label class="india-domain-select">Forecast domain
            <select aria-label="Forecast domain">
              ${domainCatalog.domains.map((domain) => `<option value="${domain.id}" ${domain.id === activeDomain?.id ? "selected" : ""}>${domain.label}</option>`).join("")}
            </select>
          </label>
        ` : ""}
        <div class="india-run-console__heading">
          <div>
            <span>${isCurrent ? "Current forecast" : "Selected archive issue"}</span>
            <strong>${forecast.issue.initial_condition_source.label}</strong>
          </div>
        </div>
        <div class="india-source-tabs" role="navigation" aria-label="Initial-condition source">
          ${sourcesForDisplay(index).map((candidate) => {
            const preferred = defaultIssueForSource(index, candidate);
            if (!preferred) return `<span class="is-pending" aria-disabled="true"><span>${candidate.short_label}</span><small>Pilot pending</small></span>`;
            return `
            <a href="${sourceLink(candidate.id, preferred.id)}" class="${candidate.id === sourceId ? "is-active" : ""}" ${candidate.id === sourceId ? "aria-current=\"page\"" : ""}>
              <span>${candidate.short_label}</span>
              <small>${candidate.category === "operational_proxy" ? "Current proxy" : candidate.category === "operational_initialization" ? "Experimental initialization" : "Delayed reference"}</small>
            </a>
          `;}).join("")}
        </div>
        <dl class="india-status-grid" aria-label="Issue status">
          <div><dt>Initialized</dt><dd>${friendlyDate(forecast.issue.initialization.slice(0, 10), true)} · 00 UTC</dd></div>
          <div><dt>Valid through</dt><dd>${friendlyDate(validThrough, true)}</dd></div>
          <div><dt>Ensemble</dt><dd>${forecast.issue.members} members</dd></div>
          <div><dt>Source</dt><dd>${sourceId === "gfs" ? "GFS proxy" : sourceId === "ifs" ? "IFS experimental" : "ERA5 delayed"}</dd></div>
        </dl>
        <div class="india-run-actions" aria-label="Forecast downloads">
          <span>Forecast briefing</span>
          ${pdfReady
            ? `<a class="india-pdf-download" href="${pdfUrl}" target="_blank" rel="noopener noreferrer" type="application/pdf">View briefing ↗</a><a class="india-pdf-download-secondary" href="${pdfUrl}" download type="application/pdf">Download PDF ↓</a>`
            : `<button class="india-pdf-download" type="button" disabled>PDF preparing · refresh shortly</button>`}
          <a href="./#archive">Browse archive</a>
        </div>
        ${matchedSource ? `<a class="india-matched-run" href="${sourceLink(matchedSource.id, currentIssueId)}">Same ${friendlyDate(forecast.issue.initialization.slice(0, 10), true)} issue is available with ${matchedSource.short_label} initial conditions →</a>` : ""}
      </section>

      ${archiveBanner}
      ${comparisonCard}

      <div class="india-toolbar">
        <div class="india-product-tabs" id="india-product-tabs" aria-label="India forecast field"></div>
        <div class="india-view-controls">
          <div class="india-wind-mode" id="india-wind-mode" role="group" aria-label="Wind direction display" hidden>
            <span>Wind display</span>
            <button type="button" data-wind-mode="streamlines" class="is-active">Streamlines</button>
            <button type="button" data-wind-mode="arrows">Arrows</button>
          </div>
          <div class="india-range-tabs" aria-label="Displayed forecast weeks">
            <button type="button" data-range="0" class="is-active">Weeks 1–4</button>
            <button type="button" data-range="2">Weeks 3–6</button>
          </div>
        </div>
      </div>

      <div class="india-panel-grid" id="india-panel-grid" aria-label="${activeDomain?.label ?? "India"} forecast maps"></div>
      <div class="india-shared-legend" id="india-shared-legend"></div>
      <p class="india-visual-note" id="india-visual-note"></p>

      <footer class="india-sheet__footer">
        <p id="india-baseline-note"></p>
        <span>Experimental S2S research guidance · Not an operational weather forecast or warning</span>
      </footer>
    </section>
  `;

  let selectedProduct: ProductKey = defaultProduct;
  let rangeStart = 0;
  let windRenderingMode: WindRenderingMode = "streamlines";
  const productTabs = container.querySelector<HTMLDivElement>("#india-product-tabs")!;
  const panelGrid = container.querySelector<HTMLDivElement>("#india-panel-grid")!;
  const windModeControl = container.querySelector<HTMLDivElement>("#india-wind-mode")!;

  container.querySelector<HTMLSelectElement>(".india-domain-select select")
    ?.addEventListener("change", (event) => {
      const parameters = new URLSearchParams(window.location.search);
      const domainId = (event.currentTarget as HTMLSelectElement).value;
      if (domainId === domainCatalog?.default_domain) parameters.delete("domain");
      else parameters.set("domain", domainId);
      parameters.delete("source");
      parameters.delete("issue");
      window.location.assign(`./?${parameters.toString()}#india`);
    });

  availableProducts.forEach((productKey) => {
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

  windModeControl
    .querySelectorAll<HTMLButtonElement>("button[data-wind-mode]")
    .forEach((button) => {
      button.addEventListener("click", () => {
        windRenderingMode = button.dataset.windMode as WindRenderingMode;
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
    windModeControl.hidden = selectedProduct !== "wind850_anomaly";
    windModeControl
      .querySelectorAll<HTMLButtonElement>("button[data-wind-mode]")
      .forEach((button) => {
        const active = button.dataset.windMode === windRenderingMode;
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
      map.render(selectedProduct, week, product, windRenderingMode);
    });

    const legend = container.querySelector<HTMLDivElement>("#india-shared-legend")!;
    legend.style.setProperty("--legend-under", product.legend.under);
    legend.style.setProperty("--legend-over", product.legend.over);
    legend.replaceChildren(createLegend(product));
    container.querySelector("#india-visual-note")!.textContent = selectedProduct === "wind850_anomaly"
      ? `${product.description}. ${windRenderingMode === "streamlines" ? "Streamlines trace the interpolated anomaly flow and include direction markers." : "Arrows sample every third native grid point and use one scale across all weeks."} Shading is visually interpolated 1.5°→0.25°; hover values remain native-grid values.`
      : `${product.description.replace("100-member", `${activeForecast.issue.members}-member`)}. Visual-only bilinear interpolation from the ${activeForecast.grid.spacing_degrees}° native grid; hover values remain native-grid values.`;
    container.querySelector("#india-baseline-note")!.textContent = product.baseline
      ? "Anomaly is the forecast minus the model's typical value for the same season and forecast lead, estimated from 2002–2021 reforecasts."
      : activeDomain?.id !== "india"
        ? "Configured-domain v1 provides raw rainfall and temperature maps only; anomalies, probabilities, regional summaries, and verification are withheld."
        : sourceId === "gfs"
        ? "Experimental initialization from operational analysis and short-range forecast proxy inputs."
        : "Delayed ERA5 reference initialization; not near-real-time operational guidance.";
  }

  update();
  if (initializationComparison && data.comparisonForecasts) {
    renderInitializationComparison(
      container,
      initializationComparison,
      data.comparisonForecasts,
      activeGeography,
    );
  }
}
