import "./styles/main.css";
import type {
  AppData,
  ForecastData,
  GlobalForecastData,
  GlobalMetadata,
  GlobalVariableKey,
  IndiaAdminData,
  IndiaMapGeographyData,
  IssueIndexData,
  RegionalOutlookData,
  ValidationData,
  WorldCountriesData,
} from "./types";
import {
  checksumVersionedPath,
  currentGfsIssue,
  findIssue,
  issueChecksum,
  issueHasRegionalOutlook,
} from "./lib/catalog";

type Route = "forecast" | "india" | "outlook" | "archive" | "models";

const ROUTES = new Set<Route>([
  "forecast",
  "india",
  "outlook",
  "archive",
  "models",
]);
let routeCleanup: (() => void) | undefined;

function currentRoute(): Route {
  const hash = window.location.hash.replace("#", "") as Route;
  if (new URLSearchParams(window.location.search).get("view") === "global") {
    return "forecast";
  }
  return ROUTES.has(hash) ? hash : "india";
}

async function fetchJson<T>(path: string, cache?: RequestCache): Promise<T> {
  const response = await fetch(path, cache ? { cache } : undefined);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchGlobalData(
  metadata: GlobalMetadata,
  preloadPrecipitation: boolean,
): Promise<GlobalForecastData> {
  const fetchBinary = async (
    path: string,
    sizeBytes: number,
    expectedSha256: string,
  ): Promise<ArrayBuffer> => {
    const response = await fetch(`./data/global/${path}`);
    if (!response.ok) {
      throw new Error(`${path} returned ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== sizeBytes) {
      throw new Error(`${path} has an unexpected byte length`);
    }
    if (await sha256Hex(buffer) !== expectedSha256) {
      throw new Error(`${path} failed its SHA-256 check`);
    }
    return buffer;
  };
  const fields: GlobalForecastData["fields"] = {};
  const anomalies: GlobalForecastData["anomalies"] = {};
  const spreads: GlobalForecastData["spreads"] = {};
  const vectors: GlobalForecastData["vectors"] = {};
  const meanLoading: Partial<Record<GlobalVariableKey, Promise<void>>> = {};
  const loading: Partial<Record<GlobalVariableKey, Promise<void>>> = {};
  const loadMean = (key: GlobalVariableKey): Promise<void> => {
    if (fields[key]) return Promise.resolve();
    if (meanLoading[key]) return meanLoading[key]!;
    const definition = metadata.variables[key];
    const promise = fetchBinary(
      definition.path,
      definition.size_bytes,
      definition.sha256,
    ).then((buffer) => {
      fields[key] = new Uint16Array(buffer);
    });
    meanLoading[key] = promise;
    return promise;
  };
  const loadVariable = (key: GlobalVariableKey): Promise<void> => {
    if (fields[key] && spreads[key]) return Promise.resolve();
    if (loading[key]) return loading[key]!;
    const promise = (async () => {
      const definition = metadata.variables[key];
      await loadMean(key);
      const tasks: Promise<void>[] = [
        fetchBinary(
          definition.spread.path,
          definition.spread.size_bytes,
          definition.spread.sha256,
        ).then((buffer) => {
          spreads[key] = new Uint16Array(buffer);
        }),
      ];
      if (definition.anomaly) {
        tasks.push(
          fetchBinary(
            definition.anomaly.path,
            definition.anomaly.size_bytes,
            definition.anomaly.sha256,
          ).then((buffer) => {
            anomalies[key] = new Uint16Array(buffer);
          }),
        );
      }
      if (definition.vector) {
        tasks.push(
          Promise.all([
            fetchBinary(
              definition.vector.u.path,
              definition.vector.u.size_bytes,
              definition.vector.u.sha256,
            ),
            fetchBinary(
              definition.vector.v.path,
              definition.vector.v.size_bytes,
              definition.vector.v.sha256,
            ),
          ]).then(([uBuffer, vBuffer]) => {
            vectors[key] = {
              u: new Uint16Array(uBuffer),
              v: new Uint16Array(vBuffer),
            };
          }),
        );
      }
      await Promise.all(tasks);
    })();
    loading[key] = promise;
    return promise;
  };
  const maskDefinition = metadata.grid.ocean_mask;
  const oceanMaskBuffer = await fetchBinary(
    maskDefinition.path,
    maskDefinition.size_bytes,
    maskDefinition.sha256,
  );
  const result: GlobalForecastData = {
    metadata,
    fields,
    anomalies,
    spreads,
    vectors,
    oceanMask: new Uint8Array(oceanMaskBuffer),
    loadVariable,
  };
  if (preloadPrecipitation) await loadMean("precipitation");
  return result;
}

async function loadData(): Promise<AppData> {
  const route = currentRoute();
  if (route === "models") return {};

  if (route === "archive") {
    return {
      index: await fetchJson<IssueIndexData>("./data/index.json", "no-store"),
    };
  }

  if (route === "forecast") {
    const [globalMetadata, world, indiaAdmin] = await Promise.all([
      fetchJson<GlobalMetadata>("./data/global/metadata.json"),
      fetchJson<WorldCountriesData>("./data/world-countries.geojson"),
      fetchJson<IndiaAdminData>("./data/india-admin.json"),
    ]);
    const global = await fetchGlobalData(globalMetadata, true);
    return { global, world, indiaAdmin };
  }

  const index = await fetchJson<IssueIndexData>("./data/index.json", "no-store");
  const parameters = new URLSearchParams(window.location.search);
  const selection = findIssue(
    index,
    parameters.get("source"),
    parameters.get("issue"),
  );
  if (!selection) throw new Error("No forecast issue is available");
  const { source, issue } = selection;
  const regionalPromise: Promise<RegionalOutlookData | undefined> =
    issueHasRegionalOutlook(index, issue)
      ? fetchJson<RegionalOutlookData>(checksumVersionedPath(
        `./data/${issue.regional_outlook}`,
        issueChecksum(issue, "regional_outlook"),
      ))
      : Promise.resolve(undefined);
  const [forecast, validation, indiaGeography, regionalOutlook] = await Promise.all([
    fetchJson<ForecastData>(checksumVersionedPath(
      `./data/${issue.forecast}`,
      issueChecksum(issue, "forecast"),
    )),
    fetchJson<ValidationData>(checksumVersionedPath(
      issue.validation ? `./data/${issue.validation}` : "./data/validation.json",
      issueChecksum(issue, "validation"),
    ), issue.validation ? undefined : "no-store"),
    fetchJson<IndiaMapGeographyData>("./data/india-map-geography.json"),
    regionalPromise,
  ]);
  if (forecast.issue.initial_condition_source.id !== source.id) {
    throw new Error("Forecast source does not match the selected initialization source");
  }
  return { index, forecast, validation, indiaGeography, regionalOutlook };
}

function shell(): string {
  return `
    <header class="site-header">
      <a class="brand" href="./#india" aria-label="S2S Research forecast home">
        <img class="brand-logo" src="./brand/s2s-research-mark.svg" alt="" />
        <span class="brand-copy">
          <strong>S2S Research</strong>
          <small>Experimental Subseasonal Forecasting</small>
        </span>
      </a>
      <nav class="site-nav" aria-label="Primary">
        <a href="./#india" data-route="india">Forecast</a>
        <a href="./#outlook" data-route="outlook">Regional</a>
        <a href="./?view=global#forecast" data-route="forecast">Global Demo</a>
        <a href="./#archive" data-route="archive">Archive</a>
        <a href="./#models" data-route="models">Models <small>S2S Lab</small></a>
      </nav>
      <div class="header-partner">
        <a
          class="institution-lockup"
          href="https://scdlds.ashoka.edu.in/"
          target="_blank"
          rel="noreferrer"
          aria-label="Safexpress Centre for Data, Learning and Decision Sciences at Ashoka University"
        >
          <img
            class="institution-lockup__centre"
            src="./brand/scdlds-centre.png"
            alt="Safexpress Centre for Data, Learning and Decision Sciences"
          />
          <span aria-hidden="true"></span>
          <img
            class="institution-lockup__ashoka"
            src="./brand/ashoka-university.png"
            alt="Ashoka University"
          />
        </a>
        <div class="header-status">
          <span class="live-dot" aria-hidden="true"></span>
          <span class="header-status__label">Research prototype</span>
        </div>
      </div>
    </header>
    <main id="content" tabindex="-1"></main>
    <footer class="site-footer" aria-label="About this forecast service">
      <div class="site-footer__brand">
        <strong>S2S Research</strong>
        <span>Safexpress Centre for Data, Learning and Decision Sciences · Ashoka University</span>
        <p>Experimental research guidance · Not an operational forecast, warning, or decision trigger.</p>
      </div>
      <div class="site-footer__section" id="about">
        <strong>About & methods</strong>
        <p>FuXi-S2S ensemble output is summarized as weekly India fields. Anomalies and raw tercile probabilities appear only when a seasonally matched model climatology is available.</p>
      </div>
      <div class="site-footer__section" id="schedule">
        <strong>Update schedule</strong>
        <p>GFS-proxy runs are targeted every Wednesday and Saturday after source-data checks. ERA5 is a delayed reference, normally 5–7 days behind.</p>
      </div>
      <div class="site-footer__section" id="limitations">
        <strong>Limitations & contact</strong>
        <p>Long leads describe broad ensemble tendencies, not local weather. Public contact: <a href="mailto:raj.ayush@students.iiserpune.ac.in">raj.ayush@students.iiserpune.ac.in</a> · <a href="https://scdlds.ashoka.edu.in/" target="_blank" rel="noreferrer">Centre website ↗</a></p>
      </div>
    </footer>
  `;
}

async function renderRoute(data: AppData): Promise<void> {
  const route = currentRoute();
  if (route === "forecast" && !data.global) {
    window.location.assign("./?view=global#forecast");
    return;
  }
  if ((route === "india" || route === "outlook") && !data.forecast) {
    window.location.assign("./#india");
    return;
  }
  routeCleanup?.();
  routeCleanup = undefined;
  const content = document.querySelector<HTMLElement>("#content")!;
  document.body.classList.toggle("global-route", route === "forecast");
  document.body.classList.toggle("india-route", route === "india");
  document.body.classList.toggle("outlook-route", route === "outlook");
  document.body.classList.toggle("archive-route", route === "archive");
  document.body.classList.toggle("models-route", route === "models");
  const statusDate = route === "forecast"
    ? data.global!.metadata.issue.initialization
    : data.forecast?.issue.initialization ??
      (data.index ? currentGfsIssue(data.index)?.initialization : undefined);
  const statusLabel = statusDate
    ? new Intl.DateTimeFormat("en-IN", {
      day: "2-digit", month: "short", year: "numeric", timeZone: "UTC",
    }).format(new Date(statusDate))
    : route === "archive" ? "Forecast archive" : "Research models";
  document.querySelector<HTMLElement>(".header-status__label")!.textContent = statusLabel;
  const regionalLink = document.querySelector<HTMLAnchorElement>(
    '.site-nav a[data-route="outlook"]',
  );
  if (regionalLink && data.index) {
    const selected = data.forecast
      ? findIssue(
        data.index,
        data.forecast.issue.initial_condition_source.id,
        data.forecast.issue.initialization.slice(0, 10).replaceAll("-", ""),
      )
      : undefined;
    const issue = selected?.issue ?? currentGfsIssue(data.index);
    regionalLink.hidden = Boolean(issue) && !issueHasRegionalOutlook(
      data.index,
      issue!,
      selected ? data.forecast : undefined,
    );
  }
  document
    .querySelectorAll<HTMLAnchorElement>(".site-nav a")
    .forEach((link) => {
      const active = link.dataset.route === route;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  content.replaceChildren();
  if (route === "forecast") {
    const { renderGlobalPage } = await import("./pages/global");
    routeCleanup = renderGlobalPage(content, data);
  }
  if (route === "india") {
    const { renderForecastPage } = await import("./pages/forecast");
    renderForecastPage(content, data);
  }
  if (route === "outlook") {
    const { renderOutlookPage } = await import("./pages/outlook");
    renderOutlookPage(content, data);
  }
  if (route === "archive") {
    const { renderArchivePage } = await import("./pages/archive");
    renderArchivePage(content, data);
  }
  if (route === "models") {
    const { renderModelsPage } = await import("./pages/models");
    renderModelsPage(content);
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}

function renderLoadFailure(error: unknown): void {
  const message = error instanceof Error ? error.message : "Unknown data error";
  document.querySelector<HTMLElement>("#content")!.innerHTML = `
    <section class="blocked-state">
      <span class="eyebrow">Data unavailable</span>
      <h1>The forecast package could not be loaded.</h1>
      <p>No partial or broken map has been shown. Check that the compact public data files were built and deployed together.</p>
      <details><summary>Technical detail</summary><code>${message}</code></details>
    </section>
  `;
}

async function start(): Promise<void> {
  const app = document.querySelector<HTMLDivElement>("#app")!;
  app.innerHTML = shell();
  try {
    const data = await loadData();
    await renderRoute(data);
    window.addEventListener("hashchange", () => {
      if (currentRoute() === "forecast" && !data.global) {
        window.location.assign("./?view=global#forecast");
        return;
      }
      if (
        (currentRoute() === "india" || currentRoute() === "outlook") &&
        !data.forecast
      ) {
        window.location.reload();
        return;
      }
      if (currentRoute() === "archive" && !data.index) {
        window.location.reload();
        return;
      }
      void renderRoute(data).catch(renderLoadFailure);
    });
  } catch (error) {
    renderLoadFailure(error);
  }
}

void start();
