import type { AppData } from "../types";

export function renderMethodsPage(container: HTMLElement, data: AppData): void {
  container.innerHTML = `
    <section class="page-intro methods-intro">
      <div>
        <span class="eyebrow">Methods · Version ${data.formulas.formula_version}</span>
        <h1>Clarity before<br><em>complexity</em></h1>
      </div>
      <p class="intro-copy">What the global animation shows, how the fields were converted, and where this experimental guidance must stop.</p>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">01</span>
      <div>
        <span class="eyebrow">Forecast provenance</span>
        <h2>One initialization, two validated views</h2>
        <p class="large-copy">Atmosphere 42 presents the ensemble mean of 100 stochastic realizations for 42 daily periods on the native 1.5° global grid. The India case adds separately validated weekly products and anomalies.</p>
        <p>The global animation is an independently sampled 100-member companion ensemble generated from the same frozen initialization input. Its individual stochastic draws are not the deleted raw members behind the earlier India case, so the two ensemble means can differ slightly.</p>
        <p>The underlying research model is FuXi-S2S, a data-driven subseasonal forecast system. Its name is kept here for reproducibility while the forecast interface uses neutral product language.</p>
        <p>FuXi-S2S expects ERA5-style daily atmospheric inputs. For this near-real-time experiment, operational GFS analyses and short precipitation/radiation forecasts were assembled into two complete UTC daily proxy inputs. That input-domain mismatch is why the product remains explicitly experimental.</p>
      </div>
    </section>

    <section class="method-steps">
      <article>
        <span>02 / Global precipitation</span>
        <h3>From rate to one day</h3>
        <code>daily rainfall = TP × 24</code>
        <p>TP begins in millimetres per hour. The global viewer converts each daily-mean forecast rate to a 24-hour total. The India case additionally sums seven days for each weekly accumulation.</p>
      </article>
      <article>
        <span>03 / Global temperature</span>
        <h3>Kelvin to Celsius</h3>
        <code>°C = Kelvin − 273.15</code>
        <p>The displayed global field is the arithmetic mean across all 100 members. The colour legend is fixed through the complete animation, so changes between days remain visually comparable.</p>
      </article>
      <article>
        <span>04 / Mid-troposphere</span>
        <h3>Geopotential to height</h3>
        <code>Z500 (dam) = geopotential ÷ 9.80665 ÷ 10</code>
        <p>The 500 hPa field is shown as geopotential height in decametres. Contours make large-scale ridges, troughs, and circulation changes easier to follow.</p>
      </article>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">05</span>
      <div>
        <span class="eyebrow">Animation contract</span>
        <h2>Daily evidence, smooth presentation</h2>
        <p class="large-copy">Only 42 daily model fields exist. The viewer cross-fades neighbouring maps for visual continuity; it does not claim additional hourly forecast information.</p>
        <p>Hover values always come from the labelled daily field. The browser checks the size and SHA-256 digest of each compact binary file before rendering it.</p>
      </div>
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
        <p>This prototype publishes compact derived guidance under the applicable FuXi-S2S model and upstream-data research conditions. It does not redistribute model weights, full forecasts, GFS inputs, FuXi reforecasts, or raw IMD/IMERG grids. Global boundary geography is from the public-domain Natural Earth 1:110m dataset. Consult the original model and data-provider licenses before reuse.</p>
      </div>
    </section>
  `;
}
