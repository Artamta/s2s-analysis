import { FEATURED_REPORT } from "../lib/reports";

const TEAM_MEMBERS = [
  {
    name: "Sandeep Juneja",
    role: "Professor of Computer Science · Director, SCDLDS",
    affiliation: "SCDLDS · Ashoka University",
    image: "./team/sandeep-juneja.jpg",
    profile: "https://www.ashoka.edu.in/profile/sandeep-juneja/",
  },
  {
    name: "Parthasarathi Mukhopadhyay",
    role: "Academic Visitor, SCDLDS · Visiting Faculty",
    affiliation: "SCDLDS · Ashoka University · IISER Berhampur",
    image: "./team/parthasarathi-mukhopadhyay.jpg",
    profile: "https://www.ashoka.edu.in/profile/parthasarathi-mukhopadhyay/",
  },
  {
    name: "Manmeet Singh",
    role: "Project Lead · Academic Visitor, SCDLDS",
    affiliation: "SCDLDS · Ashoka University · The University of Texas at Austin",
    image: "./team/manmeet-singh.jpg",
    profile: "https://www.ashoka.edu.in/profile/manmeet-singh/",
  },
  {
    name: "Ayush Raj",
    role: "BS–MS Student",
    affiliation: "SCDLDS · IISER Pune",
    image: "./team/ayush-raj.jpg",
    profile: "https://github.com/Artamta",
  },
  {
    name: "Saptarishi Dhanuka",
    role: "Pre-doctoral Fellow, SCDLDS",
    affiliation: "SCDLDS · Ashoka University",
    image: "./team/saptarishi-dhanuka.jpg",
    profile: "https://www.ashoka.edu.in/profile/saptarishi-dhanuka/",
  },
] as const;

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

      <section class="briefing-team" aria-labelledby="briefing-team-title">
        <header class="briefing-section-heading">
          <div>
            <span>Forecast team</span>
            <h2 id="briefing-team-title">People behind the forecast</h2>
          </div>
          <p>Contributors to the scientific review, interpretation, and publication workflow.</p>
        </header>

        <div class="briefing-member-grid">
          ${TEAM_MEMBERS.map((member) => `
            <article class="briefing-member-card">
              <div class="briefing-member-card__portrait">
                <img
                  src="${member.image}"
                  alt="Portrait of ${member.name}"
                  loading="lazy"
                  decoding="async"
                />
              </div>
              <div class="briefing-member-card__body">
                <h3>${member.name}</h3>
                <p class="briefing-member-card__role">${member.role}</p>
                <p class="briefing-member-card__affiliation">${member.affiliation}</p>
                <a
                  href="${member.profile}"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Open ${member.name}'s public profile in a new tab"
                >Public profile <span aria-hidden="true">↗</span></a>
              </div>
            </article>
          `).join("")}
        </div>
      </section>

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
