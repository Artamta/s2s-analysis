import type { AppData } from "../types";

export function renderMethodsPage(container: HTMLElement, data: AppData): void {
  container.innerHTML = `
    <section class="page-intro methods-intro">
      <div>
        <span class="eyebrow">Methods · Version ${data.formulas.formula_version}</span>
        <h1>Clarity before<br><em>complexity</em></h1>
      </div>
      <p class="intro-copy">What the experimental global animation shows, how the fields were converted, and where interpretation must stop.</p>
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
      <article>
        <span>05 / Lower-tropospheric flow</span>
        <h3>850 hPa wind</h3>
        <code>speed = √(U² + V²)</code>
        <p>Shading is the arithmetic ensemble mean of member wind speed. Arrows use the separately averaged U and V components, making monsoon and tropical inflow easier to follow.</p>
      </article>
      <article>
        <span>06 / Pressure</span>
        <h3>Pa to hPa</h3>
        <code>MSLP (hPa) = pressure ÷ 100</code>
        <p>Pressure is available as a base field and a contour overlay. Exact low and high centres should not be interpreted deterministically at long leads.</p>
      </article>
      <article>
        <span>07 / Ocean boundary</span>
        <h3>Sea-surface temperature</h3>
        <code>°C = Kelvin − 273.15</code>
        <p>SST is masked to a fixed, stable open-ocean support: Natural Earth ocean cells that remain within the physical −3 to 45 °C range through all 42 ensemble-mean leads. Absolute SST is shown; a scientifically matched SST anomaly is not yet published.</p>
      </article>
      <article>
        <span>08 / Tropical convection context</span>
        <h3>Outgoing longwave radiation</h3>
        <code>OLR = − top net thermal radiation</code>
        <p>Lower tropical OLR often accompanies higher cloud tops and deep convection. This is useful MJO context, but it is not a filtered OLR anomaly or an MJO phase/index.</p>
      </article>
      <article>
        <span>09 / Atmospheric moisture</span>
        <h3>Total-column water vapour</h3>
        <code>TCWV = vertically integrated water vapour (kg/m²)</code>
        <p>TCWV highlights broad tropical moisture reservoirs and transport pathways. Physically invalid negative experimental values are clipped to zero before the 100-member mean and spread are calculated. Combine moisture shading with 850 hPa vectors to read inflow.</p>
      </article>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">10</span>
      <div>
        <span class="eyebrow">Animation contract</span>
        <h2>Daily evidence, smooth presentation</h2>
        <p class="large-copy">Only 42 daily model fields exist. The viewer cross-fades neighbouring maps for visual continuity; it does not claim additional hourly forecast information.</p>
        <p>Hover values always come from the labelled daily field. Clicking the map opens the exact nearest 1.5° grid cell through all 42 leads.</p>
      </div>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">11</span>
      <div>
        <span class="eyebrow">Uncertainty and S2S interpretation</span>
        <h2>Member disagreement, not a confidence score</h2>
        <p class="large-copy">The point trace shows the 100-member arithmetic mean with a band of ±1 population standard deviation (<code>ddof=0</code>) at every daily lead.</p>
        <p>A wider band means the members disagree more at that grid cell. It is not a calibrated probability interval and does not by itself measure forecast skill. In Days 1–14, synoptic detail can still be informative; in Weeks 3–4, emphasize persistent large-scale patterns; in Weeks 5–6, interpret only broad ensemble and circulation tendencies.</p>
        <p>Verification metrics remain unpublished until every observation needed for the valid period is available on the identical grid, mask, and timing contract. The browser checks the size and SHA-256 digest of every selected mean, spread, and vector file before rendering it.</p>
      </div>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">12</span>
      <div>
        <span class="eyebrow">Geography and interaction</span>
        <h2>World context, India-standard focus</h2>
        <p class="large-copy">Country outlines and label points use Natural Earth at global scale. The India-focus overlay is a simplified display derivative of the Survey of India Administrative Boundary Database state/UT layer.</p>
        <p>Use the India button or zoom controls to reveal state and union-territory names. The simplified web derivative retains the complete supplied depiction but is not a legal or cadastral boundary product. The source standard and product catalogue are described by the <a href="https://onlinemaps.surveyofindia.gov.in/AboutPortal.aspx">Survey of India Online Maps Portal</a>.</p>
      </div>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">13</span>
      <div>
        <span class="eyebrow">MJO publication gate</span>
        <h2>Context now; formal index later</h2>
        <p class="large-copy">A defensible MJO product needs equatorial OLR, U850 and U200 anomalies, temporal filtering, a matched daily climatology, and projection onto documented EOF patterns.</p>
        <p>The current viewer therefore exposes OLR and 850 hPa flow without calling either an MJO index. A Wheeler–Hendon-style phase-space product will be added only after those climatology and reproducibility contracts pass.</p>
      </div>
    </section>

    <section class="methods-lead">
      <span class="chapter-number">14</span>
      <div>
        <span class="eyebrow">Global anomaly gate</span>
        <h2>Possible, but not from the India-only climate</h2>
        <p class="large-copy">The available 2002–2021 lead-matched model climatology covers the 27 × 27 India grid for rainfall and temperature. It cannot be stretched across the globe or exchanged between variables.</p>
        <p>The global explorer therefore publishes absolute fields only. Its controls permit one shaded field plus multiple compatible contour and vector overlays. Global anomalies will unlock after a 20-year, 121 × 240, lead- and calendar-matched climatology passes the same publication checks.</p>
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
