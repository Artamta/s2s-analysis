import { ForecastMap } from "../components/ForecastMap";
import { createLegend } from "../components/Legend";
import { formatValue } from "../lib/color";
import type { AppData, ProductKey } from "../types";

const PRODUCT_ORDER: ProductKey[] = [
  "rainfall_total",
  "rainfall_anomaly",
  "temperature_mean",
  "temperature_anomaly",
];

function friendlyDate(isoDate: string, includeYear = false): string {
  return new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  }).format(new Date(`${isoDate}T12:00:00Z`));
}

export function renderForecastPage(container: HTMLElement, data: AppData): void {
  const { forecast, validation } = data;
  if (!validation.presentation_allowed) {
    container.innerHTML = `
      <section class="blocked-state">
        <span class="eyebrow">Publication gate closed</span>
        <h1>This forecast is not available for presentation.</h1>
        <p>A scientific or publication validation check failed. Review the Data Validation page for details; no map has been rendered.</p>
        <a class="button-link" href="#validation">Open validation report</a>
      </section>
    `;
    return;
  }

  container.innerHTML = `
    <section class="page-intro forecast-intro">
      <div>
        <span class="eyebrow">Issue 01 · Research guidance</span>
        <h1>Six-week outlook<br><em>from 28 July 2026</em></h1>
      </div>
      <p class="intro-copy">A 100-member experimental FuXi-S2S view of rainfall and 2-metre temperature across India. Every map stays on the native 1.5° science grid.</p>
    </section>

    <aside class="experimental-note" aria-label="Experimental forecast warning">
      <span class="note-icon" aria-hidden="true">!</span>
      <div>
        <strong>Experimental GFS-proxy initialization</strong>
        <p>FuXi-S2S was trained with ERA5-style daily inputs. This run uses operational GFS proxy fields and has no matched GFS-initialized hindcast calibration. It is not an official warning.</p>
      </div>
    </aside>

    <section class="issue-strip" aria-label="Forecast issue details">
      <div><span>Initialized</span><strong>28 Jul 2026 · 00 UTC</strong></div>
      <div><span>Input days</span><strong>26–27 Jul · UTC means</strong></div>
      <div><span>Ensemble</span><strong>${forecast.issue.members} members</strong></div>
      <div><span>Horizon</span><strong>${forecast.issue.lead_days} days</strong></div>
    </section>

    <section class="forecast-workbench">
      <aside class="control-panel">
        <div class="control-section">
          <span class="control-label">Forecast week</span>
          <div class="week-selector" id="week-selector"></div>
        </div>
        <div class="control-section">
          <span class="control-label">Map layer</span>
          <div class="product-selector" id="product-selector"></div>
        </div>
        <div class="control-footnote">
          <span class="mini-rule"></span>
          <p>Hover or focus a native grid cell for its value. Legends are locked across all six weeks.</p>
        </div>
      </aside>

      <div class="map-column">
        <div class="map-heading">
          <div>
            <span class="map-kicker" id="map-kicker"></span>
            <h2 id="map-title"></h2>
          </div>
          <span class="native-badge">Native 1.5°</span>
        </div>
        <div class="map-frame" id="map-frame"></div>
        <div id="map-legend"></div>
        <p class="map-caption" id="map-caption"></p>
      </div>

      <aside class="reading-panel">
        <span class="control-label">India signal</span>
        <div class="headline-value" id="headline-value"></div>
        <p class="headline-description" id="headline-description"></p>
        <dl class="range-list">
          <div><dt>Supported minimum</dt><dd id="minimum-value"></dd></div>
          <div><dt>Supported maximum</dt><dd id="maximum-value"></dd></div>
          <div><dt>Native cells</dt><dd>${forecast.grid.supported_cell_count}</dd></div>
        </dl>
        <div class="baseline-card" id="baseline-card"></div>
      </aside>
    </section>

    <section class="science-note-grid">
      <article>
        <span>01 / Anomaly contract</span>
        <h3>Model climate, matched by lead</h3>
        <p>FuXi anomalies compare this forecast with 20 equally weighted native-reforecast yearly means. The 27 July model-state position is interpolated between 25 and 28 July.</p>
      </article>
      <article>
        <span>02 / Verification status</span>
        <h3>Scores wait for complete weeks</h3>
        <p>${forecast.issue.observation_verification.message}</p>
      </article>
      <article>
        <span>03 / Interpretation</span>
        <h3>Baselines stay separate</h3>
        <p>FuXi, IMD, and IMERG anomalies do not share a climatology. The Methods page explains which comparisons are scientifically safe.</p>
      </article>
    </section>
  `;

  let selectedWeek = 0;
  let selectedProduct: ProductKey = "rainfall_total";
  const weekSelector = container.querySelector<HTMLDivElement>("#week-selector")!;
  const productSelector =
    container.querySelector<HTMLDivElement>("#product-selector")!;
  const mapFrame = container.querySelector<HTMLDivElement>("#map-frame")!;
  const forecastMap = new ForecastMap(mapFrame, forecast, data.outline);

  forecast.weeks.forEach((week, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "week-button";
    button.dataset.week = String(index);
    button.innerHTML = `<strong>W${week.week}</strong><span>${friendlyDate(week.valid_start)}–${friendlyDate(week.valid_end)}</span>`;
    button.addEventListener("click", () => {
      selectedWeek = index;
      update();
    });
    weekSelector.append(button);
  });

  PRODUCT_ORDER.forEach((productKey) => {
    const product = forecast.products[productKey];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "product-button";
    button.dataset.product = productKey;
    button.innerHTML = `<span class="product-swatch product-swatch--${productKey}" aria-hidden="true"></span><span><strong>${product.short_label}</strong><small>${product.units}</small></span>`;
    button.addEventListener("click", () => {
      selectedProduct = productKey;
      update();
    });
    productSelector.append(button);
  });

  function update(): void {
    const week = forecast.weeks[selectedWeek];
    const product = forecast.products[selectedProduct];
    const summary = week.summary[selectedProduct];
    container.querySelectorAll<HTMLButtonElement>(".week-button").forEach((button) => {
      const active = Number(button.dataset.week) === selectedWeek;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    container
      .querySelectorAll<HTMLButtonElement>(".product-button")
      .forEach((button) => {
        const active = button.dataset.product === selectedProduct;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });
    container.querySelector("#map-kicker")!.textContent =
      `Week ${week.week} · ${friendlyDate(week.valid_start)}–${friendlyDate(week.valid_end, true)}`;
    container.querySelector("#map-title")!.textContent = product.label;
    container.querySelector("#headline-value")!.textContent = formatValue(
      summary.india_weighted_mean,
      product.units,
    );
    container.querySelector("#headline-description")!.textContent =
      "Area-weighted mean over supported India grid cells";
    container.querySelector("#minimum-value")!.textContent = formatValue(
      summary.india_minimum,
      product.units,
    );
    container.querySelector("#maximum-value")!.textContent = formatValue(
      summary.india_maximum,
      product.units,
    );
    const baselineCard = container.querySelector<HTMLDivElement>("#baseline-card")!;
    baselineCard.innerHTML = product.baseline
      ? `<span>Reference baseline</span><strong>${product.baseline}</strong><p>Forecast and climatology are matched at identical lead before subtraction.</p>`
      : `<span>Field definition</span><strong>${product.description}</strong><p>No anomaly baseline is applied to this layer.</p>`;
    const legend = container.querySelector<HTMLDivElement>("#map-legend")!;
    legend.replaceChildren(createLegend(product));
    container.querySelector("#map-caption")!.textContent =
      `${product.description}. Cell shading uses a fixed scientific scale; values are not visually interpolated.`;
    forecastMap.render(selectedProduct, week, product);
  }

  update();
}
