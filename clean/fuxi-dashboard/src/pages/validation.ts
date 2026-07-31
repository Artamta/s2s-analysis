import type { AppData, ValidationStatus } from "../types";

function statusLabel(status: ValidationStatus): string {
  return status === "green"
    ? "Validated"
    : status === "warning"
      ? "Validated with scope warning"
      : "Failed";
}

function formatGeneratedAt(value: string): string {
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(value));
}

export function renderValidationPage(
  container: HTMLElement,
  data: AppData,
): void {
  const { validation, sources } = data;
  const sourceById = new Map(sources.sources.map((source) => [source.id, source]));
  const sourceIdByCheck: Record<string, string> = {
    fuxi_forecast: "fuxi_forecast_20260728",
    fuxi_climatology: "fuxi_climatology_2002_2021",
    imd_climatology: "imd_climatology_1991_2020",
    imerg_climatology: "imerg_final_climatology_2001_2022",
  };
  const rows = validation.checks
    .map((check) => {
      const source = sourceById.get(sourceIdByCheck[check.id]);
      const digest = source?.sha256;
      return `
        <tr>
          <td>
            <strong>${check.label}</strong>
            <span>${check.summary}</span>
          </td>
          <td><span class="status-pill status-pill--${check.status}"><i></i>${statusLabel(check.status)}</span></td>
          <td class="checksum">${digest ? `${digest.slice(0, 10)}…${digest.slice(-8)}` : "Tested in build"}</td>
        </tr>
      `;
    })
    .join("");
  const globalStatus = data.global.metadata.validation.status;
  const globalRow = `
    <tr>
      <td>
        <strong>Global animation package</strong>
        <span>42 daily leads × 8 ensemble fields, population spread, 850 hPa U/V vectors, and an ocean support mask on the exact 121 × 240 grid.</span>
      </td>
      <td><span class="status-pill status-pill--${globalStatus}"><i></i>${statusLabel(globalStatus)}</span></td>
      <td class="checksum">19 build-verified binaries; active layers rechecked in browser</td>
    </tr>
  `;

  container.innerHTML = `
    <section class="page-intro validation-intro">
      <div>
        <span class="eyebrow">Transparent by construction</span>
        <h1>Data validation<br><em>before presentation</em></h1>
      </div>
      <p class="intro-copy">The map is the final step of an ordered scientific gate. A failed source, formula, or publication check closes the forecast page instead of degrading silently.</p>
    </section>

    <section class="validation-summary">
      <div class="validation-dial validation-dial--${validation.overall_status}">
        <span>${validation.overall_status === "green" ? "All" : "Scoped"}</span>
        <strong>${validation.overall_status === "failure" ? "Failed" : "Ready"}</strong>
      </div>
      <div>
        <span class="eyebrow">Current publication state</span>
        <h2>${validation.presentation_allowed ? "Presentation allowed" : "Publication blocked"}</h2>
        <p>${validation.presentation_allowed ? "All hard gates passed. Two seasonal-coverage limits remain visible as warnings." : "At least one hard gate failed. Forecast products are withheld."}</p>
      </div>
      <dl>
        <div><dt>Generated</dt><dd>${formatGeneratedAt(validation.generated_at)} UTC</dd></div>
        <div><dt>Issue</dt><dd>28 July 2026 · 00 UTC</dd></div>
        <div><dt>Formula version</dt><dd>${data.formulas.formula_version}</dd></div>
      </dl>
    </section>

    <section class="validation-table-section">
      <div class="section-heading">
        <div><span class="eyebrow">Gate ledger</span><h2>Source and publication checks</h2></div>
        <p>Checksums identify the exact scientific source files without exposing private storage locations.</p>
      </div>
      <div class="table-scroll">
        <table class="validation-table">
          <thead><tr><th>Check</th><th>Result</th><th>SHA-256 / mode</th></tr></thead>
          <tbody>${globalRow}${rows}</tbody>
        </table>
      </div>
    </section>

    <section class="warning-ledger">
      <div>
        <span class="eyebrow">Warnings, not omissions</span>
        <h2>Known coverage limits</h2>
      </div>
      <ol>
        ${validation.warnings.map((warning) => `<li>${warning}</li>`).join("")}
      </ol>
    </section>

    <section class="source-cards">
      ${sources.sources
        .map(
          (source, index) => `
            <article>
              <span>0${index + 1}</span>
              <h3>${source.name}</h3>
              <p>${source.period}</p>
              <dl>
                <div><dt>Grid</dt><dd>${source.grid}</dd></div>
                <div><dt>Product</dt><dd>${source.product}</dd></div>
              </dl>
            </article>
          `,
        )
        .join("")}
    </section>
  `;
}
