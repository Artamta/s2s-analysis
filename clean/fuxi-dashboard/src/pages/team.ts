import { TEAM_MEMBERS } from "../lib/team";

export function renderTeamPage(container: HTMLElement): void {
  container.innerHTML = `
    <section class="briefing-page team-page">
      <header class="briefing-hero">
        <span class="briefing-eyebrow">Forecast team</span>
        <h1>People behind the forecast</h1>
        <p>
          The researchers and collaborators contributing to the scientific
          review, interpretation, and publication of this forecast guidance.
        </p>
      </header>

      <section class="briefing-team" aria-label="Forecast team members">
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
                <h2>${member.name}</h2>
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
    </section>
  `;
}
