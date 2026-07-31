import "./styles/main.css";
import { renderForecastPage } from "./pages/forecast";
import { renderMethodsPage } from "./pages/methods";
import { renderValidationPage } from "./pages/validation";
import type { AppData } from "./types";

type Route = "forecast" | "validation" | "methods";

const ROUTES = new Set<Route>(["forecast", "validation", "methods"]);

function currentRoute(): Route {
  const hash = window.location.hash.replace("#", "") as Route;
  return ROUTES.has(hash) ? hash : "forecast";
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function loadData(): Promise<AppData> {
  const [forecast, validation, sources, formulas, outline] = await Promise.all([
    fetchJson<AppData["forecast"]>("./data/forecasts/20260728.json"),
    fetchJson<AppData["validation"]>("./data/validation.json"),
    fetchJson<AppData["sources"]>("./data/sources.json"),
    fetchJson<AppData["formulas"]>("./data/formulas.json"),
    fetchJson<AppData["outline"]>("./data/india-outline.json"),
  ]);
  return { forecast, validation, sources, formulas, outline };
}

function shell(): string {
  return `
    <header class="site-header">
      <a class="brand" href="#forecast" aria-label="FuXi-S2S India Forecast Lab home">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <span><strong>FuXi-S2S</strong><small>India Forecast Lab</small></span>
      </a>
      <nav class="site-nav" aria-label="Primary">
        <a href="#forecast" data-route="forecast">Forecast</a>
        <a href="#validation" data-route="validation">Data Validation</a>
        <a href="#methods" data-route="methods">Methods</a>
      </nav>
      <div class="header-status">
        <span class="live-dot" aria-hidden="true"></span>
        <span>28 Jul 2026</span>
      </div>
    </header>
    <main id="content" tabindex="-1"></main>
    <footer class="site-footer">
      <div>
        <strong>FuXi-S2S India Forecast Lab</strong>
        <span>Validated static prototype · Research use only</span>
      </div>
      <p>100 members · 42 days · Native 1.5° grid · No raw source data distributed</p>
    </footer>
  `;
}

function renderRoute(data: AppData): void {
  const route = currentRoute();
  const content = document.querySelector<HTMLElement>("#content")!;
  document
    .querySelectorAll<HTMLAnchorElement>(".site-nav a")
    .forEach((link) => {
      const active = link.dataset.route === route;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  content.replaceChildren();
  if (route === "forecast") renderForecastPage(content, data);
  if (route === "validation") renderValidationPage(content, data);
  if (route === "methods") renderMethodsPage(content, data);
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
    renderRoute(data);
    window.addEventListener("hashchange", () => renderRoute(data));
  } catch (error) {
    renderLoadFailure(error);
  }
}

void start();
