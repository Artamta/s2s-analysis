import type { AppData } from "../types";

export function renderMethodsPage(container: HTMLElement, data: AppData): void {
  const alignment = data.forecast.issue.climatology_alignment;
  container.innerHTML = `
    <section class="page-intro methods-intro">
      <div>
        <span class="eyebrow">Methods · Version ${data.formulas.formula_version}</span>
        <h1>How to read<br><em>this experiment</em></h1>
      </div>
      <p class="intro-copy">A compact explanation of what was run, how weekly fields were calculated, and where comparisons must stop.</p>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">01</span>
      <div>
        <span class="eyebrow">The model</span>
        <h2>What FuXi-S2S is</h2>
        <p class="large-copy">FuXi-S2S is a data-driven subseasonal forecast model. This run makes 100 stochastic realizations for 42 daily periods, then summarizes them on a 1.5° India-domain grid.</p>
        <p>FuXi-S2S expects ERA5-style daily atmospheric inputs. For this near-real-time experiment, operational GFS analyses and short precipitation/radiation forecasts were assembled into two complete UTC daily proxy inputs. That input-domain mismatch is why the product remains explicitly experimental.</p>
      </div>
    </section>

    <section class="method-steps">
      <article>
        <span>02 / Rainfall</span>
        <h3>From rate to seven days</h3>
        <code>daily rainfall = TP × 24</code>
        <code>weekly total = Σ day 1…7</code>
        <p>TP begins in millimetres per hour. Seven daily values are summed for the accumulation map; dividing that total by seven gives the rate used for rainfall anomalies.</p>
      </article>
      <article>
        <span>03 / Temperature</span>
        <h3>Absolute and weekly mean</h3>
        <code>°C = Kelvin − 273.15</code>
        <code>weekly mean = mean(day 1…7)</code>
        <p>Temperature is converted member-by-member before weekly averaging. The displayed field is the arithmetic mean across all 100 members.</p>
      </article>
      <article>
        <span>04 / Calendar match</span>
        <h3>Between native slots</h3>
        <code>clim = (1 − w)L + wR</code>
        <p>The ${alignment.target_model_state_calendar_day.slice(2)} July model state lies between native ${alignment.left_slot.slice(2)} and ${alignment.right_slot.slice(2)} July reforecast slots. The right-slot weight is ${alignment.right_weight.toFixed(3)}. Each year is interpolated first; the 20 annual means are then weighted equally.</p>
      </article>
    </section>

    <section class="baseline-section">
      <div class="section-heading">
        <div><span class="eyebrow">Three baselines, three questions</span><h2>Anomaly does not mean one universal reference</h2></div>
        <p>Keeping these baselines named prevents plausible-looking but scientifically invalid comparisons.</p>
      </div>
      <div class="baseline-grid">
        <article class="baseline-grid__fuxi">
          <span>Forecast climate</span>
          <h3>FuXi · 2002–2021</h3>
          <p>Native FuXi reforecasts, lead-matched. Use for the rainfall and temperature anomaly maps on this site.</p>
        </article>
        <article class="baseline-grid__imd">
          <span>Gauge climate</span>
          <h3>IMD · 1991–2020</h3>
          <p>Native daily gauge climatology. Use only for IMD observation anomalies and appropriately remapped verification.</p>
        </article>
        <article class="baseline-grid__imerg">
          <span>Satellite climate</span>
          <h3>IMERG Final · 2001–2022</h3>
          <p>Audited fixed Final V07B climate. IMERG Late is a separate observation stream, not the climatology.</p>
        </article>
      </div>
    </section>

    <section class="comparison-guide">
      <div>
        <span class="eyebrow">Comparison guide</span>
        <h2>What is safe to compare?</h2>
      </div>
      <div class="comparison-list">
        <article><span class="comparison-icon comparison-icon--yes">✓</span><div><h3>Forecast weeks with each other</h3><p>Yes—when using the same layer and its fixed legend.</p></div></article>
        <article><span class="comparison-icon comparison-icon--yes">✓</span><div><h3>Forecast with matched FuXi climate</h3><p>Yes—that subtraction defines the two anomaly layers.</p></div></article>
        <article><span class="comparison-icon comparison-icon--wait">…</span><div><h3>Forecast with observations</h3><p>Only after the complete valid period, target grid, mask, and timing contract are available.</p></div></article>
        <article><span class="comparison-icon comparison-icon--no">×</span><div><h3>FuXi, IMD, and IMERG anomaly values as one baseline</h3><p>No—their climatologies answer different questions and must stay named.</p></div></article>
      </div>
    </section>

    <section class="license-note">
      <span>Research use</span>
      <div>
        <h2>Experimental guidance, not an operational warning</h2>
        <p>This prototype publishes compact derived guidance under the applicable FuXi-S2S model and upstream-data research conditions. It does not redistribute model weights, full forecasts, GFS inputs, FuXi reforecasts, or raw IMD/IMERG grids. Consult the original model and data-provider licenses before reuse.</p>
      </div>
    </section>
  `;
}
