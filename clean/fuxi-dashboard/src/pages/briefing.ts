import { FEATURED_REPORT } from "../lib/reports";

export function renderBriefingPage(container: HTMLElement): void {
  container.innerHTML = `
    <section class="briefing-page">
      <header class="briefing-hero">
        <span class="briefing-eyebrow">S2S Research · Thursday briefing</span>
        <h1>Latest forecast briefing</h1>
        <p>
          The latest India subseasonal outlook and the group responsible for
          its weekly scientific review and publication.
        </p>
      </header>

      <article class="briefing-feature" aria-labelledby="latest-briefing-title">
        <a
          class="briefing-preview"
          href="${FEATURED_REPORT.href}"
          target="_blank"
          rel="noopener noreferrer"
          type="application/pdf"
          aria-label="${FEATURED_REPORT.accessibleLabel}"
        >
          <img
            src="${FEATURED_REPORT.cover}"
            alt="First page of the ${FEATURED_REPORT.date} India S2S forecast briefing"
            decoding="async"
          />
          <span aria-hidden="true">Preview report ↗</span>
        </a>

        <div class="briefing-details">
          <span class="briefing-details__status">Latest briefing</span>
          <h2 id="latest-briefing-title">${FEATURED_REPORT.title}</h2>
          <p>
            Experimental Weeks 1–4 guidance covering rainfall, rainfall and
            temperature anomalies, and 850 hPa circulation over India.
          </p>

          <dl class="briefing-facts">
            <div><dt>Initialized</dt><dd>${FEATURED_REPORT.date}</dd></div>
            <div><dt>Valid period</dt><dd>${FEATURED_REPORT.validPeriod}</dd></div>
            <div><dt>Forecast range</dt><dd>${FEATURED_REPORT.forecastRange}</dd></div>
            <div><dt>Product</dt><dd>${FEATURED_REPORT.ensemble}</dd></div>
          </dl>

          <div class="briefing-actions">
            <a
              class="briefing-action briefing-action--primary"
              href="${FEATURED_REPORT.href}"
              target="_blank"
              rel="noopener noreferrer"
              type="application/pdf"
            >Read briefing ↗</a>
            <a
              class="briefing-action"
              href="${FEATURED_REPORT.href}"
              download
              type="application/pdf"
            >Download PDF ↓</a>
          </div>
          <small>${FEATURED_REPORT.details}</small>
        </div>
      </article>

      <aside class="briefing-team-link" aria-label="Forecast team">
        <div>
          <span>Forecast team</span>
          <h2>People behind the forecast</h2>
          <p>Meet the collaborators responsible for scientific review, interpretation, and publication.</p>
        </div>
        <a href="./#team">Meet the team <span aria-hidden="true">→</span></a>
      </aside>

      <aside class="briefing-context">
        <span>Research context</span>
        <p>
          Briefings and dashboard products are experimental research guidance
          for scientific review, not operational forecasts or warnings.
        </p>
      </aside>
    </section>
  `;
}
