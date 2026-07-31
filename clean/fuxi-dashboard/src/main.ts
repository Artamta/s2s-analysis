import "./styles/main.css";
import { renderForecastPage } from "./pages/forecast";
import { renderGlobalPage } from "./pages/global";
import { renderMethodsPage } from "./pages/methods";
import { renderValidationPage } from "./pages/validation";
import type {
  AppData,
  GlobalForecastData,
  GlobalMetadata,
  GlobalVariableKey,
} from "./types";

type Route = "forecast" | "india" | "validation" | "methods";

const ROUTES = new Set<Route>(["forecast", "india", "validation", "methods"]);
let routeCleanup: (() => void) | undefined;

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

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchGlobalData(metadata: GlobalMetadata): Promise<GlobalForecastData> {
  const keys = Object.keys(metadata.variables) as GlobalVariableKey[];
  const entries = await Promise.all(
    keys.map(async (key) => {
      const definition = metadata.variables[key];
      const response = await fetch(`./data/global/${definition.path}`);
      if (!response.ok) {
        throw new Error(`${definition.path} returned ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== definition.size_bytes) {
        throw new Error(`${definition.path} has an unexpected byte length`);
      }
      if (await sha256Hex(buffer) !== definition.sha256) {
        throw new Error(`${definition.path} failed its SHA-256 check`);
      }
      return [key, new Uint16Array(buffer)] as const;
    }),
  );
  return { metadata, fields: Object.fromEntries(entries) } as GlobalForecastData;
}

async function loadData(): Promise<AppData> {
  const [forecast, globalMetadata, validation, sources, formulas, outline, world] =
    await Promise.all([
    fetchJson<AppData["forecast"]>("./data/forecasts/20260728.json"),
    fetchJson<GlobalMetadata>("./data/global/metadata.json"),
    fetchJson<AppData["validation"]>("./data/validation.json"),
    fetchJson<AppData["sources"]>("./data/sources.json"),
    fetchJson<AppData["formulas"]>("./data/formulas.json"),
    fetchJson<AppData["outline"]>("./data/india-outline.json"),
    fetchJson<AppData["world"]>("./data/world-countries.geojson"),
  ]);
  const global = await fetchGlobalData(globalMetadata);
  return { forecast, global, validation, sources, formulas, outline, world };
}

function shell(): string {
  return `
    <header class="site-header">
      <a class="brand" href="#forecast" aria-label="Atmosphere 42 home">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <span><strong>Atmosphere 42</strong><small>Global Outlook Lab</small></span>
      </a>
      <nav class="site-nav" aria-label="Primary">
        <a href="#forecast" data-route="forecast">Global</a>
        <a href="#india" data-route="india">India case</a>
        <a href="#validation" data-route="validation">Validation</a>
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
        <strong>Atmosphere 42</strong>
        <span>Validated static prototype · Research use only</span>
      </div>
      <p>100 members · 42 days · Native 1.5° grid · Natural Earth geography</p>
    </footer>
  `;
}

function renderRoute(data: AppData): void {
  const route = currentRoute();
  routeCleanup?.();
  routeCleanup = undefined;
  const content = document.querySelector<HTMLElement>("#content")!;
  document.body.classList.toggle("global-route", route === "forecast");
  document
    .querySelectorAll<HTMLAnchorElement>(".site-nav a")
    .forEach((link) => {
      const active = link.dataset.route === route;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  content.replaceChildren();
  if (route === "forecast") routeCleanup = renderGlobalPage(content, data);
  if (route === "india") renderForecastPage(content, data);
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
