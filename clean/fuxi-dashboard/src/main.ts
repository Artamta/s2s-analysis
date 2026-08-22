import "./styles/main.css";
import type {
  AppData,
  ForecastData,
  ForecastDomainCatalog,
  IndiaMapGeographyData,
  InitializationComparisonData,
  InitialConditionSourceId,
  IssueIndexData,
  ValidationData,
} from "./types";
import {
  checksumVersionedPath,
  currentGfsIssue,
  findIssue,
  issueChecksum,
} from "./lib/catalog";

type Route = "india" | "archive" | "briefing";

const ROUTES = new Set<Route>([
  "india",
  "archive",
  "briefing",
]);

function currentRoute(): Route {
  const hash = window.location.hash.replace("#", "") as Route;
  return ROUTES.has(hash) ? hash : "india";
}

async function fetchJson<T>(path: string, cache?: RequestCache): Promise<T> {
  const response = await fetch(path, cache ? { cache } : undefined);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function loadData(): Promise<AppData> {
  const route = currentRoute();
  const domainCatalog = await fetchJson<ForecastDomainCatalog>(
    "./data/domains.json",
    "no-store",
  );
  const parameters = new URLSearchParams(window.location.search);
  const activeDomain = domainCatalog.domains.find(
    (domain) => domain.id === parameters.get("domain"),
  ) ?? domainCatalog.domains.find(
    (domain) => domain.id === domainCatalog.default_domain,
  );
  if (!activeDomain) throw new Error("No forecast domain is configured");

  if (route === "briefing") {
    return { domainCatalog, activeDomain };
  }

  if (route === "archive") {
    return {
      index: await fetchJson<IssueIndexData>(
        `./data/${activeDomain.catalog}`,
        "no-store",
      ),
      domainCatalog,
      activeDomain,
    };
  }

  const index = await fetchJson<IssueIndexData>(
    `./data/${activeDomain.catalog}`,
    "no-store",
  );
  const selection = findIssue(
    index,
    parameters.get("source"),
    parameters.get("issue"),
  );
  if (!selection) throw new Error("No forecast issue is available");
  const { source, issue } = selection;
  const [forecast, validation, indiaGeography] = await Promise.all([
    fetchJson<ForecastData>(checksumVersionedPath(
      `./data/${issue.forecast}`,
      issueChecksum(issue, "forecast"),
    )),
    fetchJson<ValidationData>(checksumVersionedPath(
      issue.validation ? `./data/${issue.validation}` : "./data/validation.json",
      issueChecksum(issue, "validation"),
    ), issue.validation ? undefined : "no-store"),
    fetchJson<IndiaMapGeographyData>(`./data/${activeDomain.geography}`),
  ]);
  if (forecast.issue.initial_condition_source.id !== source.id) {
    throw new Error("Forecast source does not match the selected initialization source");
  }
  const comparisonPath = forecast.issue.initialization_comparison?.comparison;
  const comparedSourceIds =
    forecast.issue.initialization_comparison?.compared_source_ids;
  let initializationComparison: InitializationComparisonData | undefined;
  let comparisonForecasts:
    | Partial<Record<InitialConditionSourceId, ForecastData>>
    | undefined;
  if (
    activeDomain.id === "india" &&
    comparisonPath &&
    comparedSourceIds && comparedSourceIds.length >= 2
  ) {
    initializationComparison = await fetchJson<InitializationComparisonData>(
      `./data/${comparisonPath}`,
    );
    const entries = await Promise.all(
      Object.entries(initializationComparison.sources).map(async ([sourceId, record]) => [
        sourceId as InitialConditionSourceId,
        await fetchJson<ForecastData>(`./data/${record.public_forecast}`),
      ] as const),
    );
    comparisonForecasts = Object.fromEntries(entries) as Partial<
      Record<InitialConditionSourceId, ForecastData>
    >;
  }
  return {
    index,
    forecast,
    validation,
    indiaGeography,
    initializationComparison,
    comparisonForecasts,
    domainCatalog,
    activeDomain,
  };
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
        <a href="./#archive" data-route="archive">Archive</a>
        <a class="briefing-nav-link" href="./#briefing" data-route="briefing">
          <svg
            class="briefing-nav-link__icon"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path d="M7 2.75h6.6L18.25 7.4V21.25H7z" />
            <path d="M13.5 2.75V7.5h4.75M9.75 12h5.5M9.75 15.5h5.5" />
          </svg>
          <span>Briefing</span>
        </a>
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
        <p>AI Ensemble Model output is summarized as weekly India fields. Anomalies appear only when a seasonally matched model climatology is available.</p>
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
  if (route === "india" && !data.forecast) {
    window.location.assign("./#india");
    return;
  }
  const content = document.querySelector<HTMLElement>("#content")!;
  document.body.classList.toggle("india-route", route === "india");
  document.body.classList.toggle("archive-route", route === "archive");
  document.body.classList.toggle("briefing-route", route === "briefing");
  const statusDate = data.forecast?.issue.initialization ??
    (data.index ? currentGfsIssue(data.index)?.initialization : undefined);
  const statusLabel = route === "briefing"
    ? "Latest Thursday briefing"
    : statusDate
    ? new Intl.DateTimeFormat("en-IN", {
      day: "2-digit", month: "short", year: "numeric", timeZone: "UTC",
    }).format(new Date(statusDate))
    : "Forecast archive";
  document.querySelector<HTMLElement>(".header-status__label")!.textContent = statusLabel;
  document
    .querySelectorAll<HTMLAnchorElement>(".site-nav a")
    .forEach((link) => {
      const active = link.dataset.route === route;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  content.replaceChildren();
  if (route === "india") {
    const { renderForecastPage } = await import("./pages/forecast");
    renderForecastPage(content, data);
  }
  if (route === "archive") {
    const { renderArchivePage } = await import("./pages/archive");
    renderArchivePage(content, data);
  }
  if (route === "briefing") {
    const { renderBriefingPage } = await import("./pages/briefing");
    renderBriefingPage(content);
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
      if (currentRoute() === "india" && !data.forecast) {
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
