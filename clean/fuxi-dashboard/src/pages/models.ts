interface ModelFact {
  label: string;
  value: string;
}

function factList(facts: ModelFact[]): string {
  return facts
    .map(({ label, value }) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
}

export function renderModelsPage(container: HTMLElement): void {
  container.innerHTML = `
    <section class="models-page">
      <header class="models-hero">
        <div>
          <span class="models-eyebrow">Research model catalogue</span>
          <h1>Two AI weather models.<br /><em>Different roles.</em></h1>
          <p>FuXi-S2S supplies the current experimental forecast maps. NeuralGCM is now part of the research programme as a historical precipitation benchmark.</p>
        </div>
        <aside aria-label="Model availability summary">
          <span>Current website coverage</span>
          <strong>FuXi · forecast guidance</strong>
          <strong>NeuralGCM · research archive</strong>
          <small>Model roles are kept separate throughout the site.</small>
        </aside>
      </header>

      <div class="model-catalogue">
        <article class="model-card model-card--active">
          <header>
            <div class="model-card__number">01</div>
            <div>
              <span class="model-status model-status--live"><i></i> Current forecast model</span>
              <h2>FuXi-S2S</h2>
              <p>Data-driven global subseasonal forecasting</p>
            </div>
          </header>
          <p class="model-card__summary">The model behind the India forecast, regional outlook, and dated Global Demo. Near-real-time guidance uses GFS proxy inputs; matched ERA5 runs remain delayed research references.</p>
          <dl>${factList([
            { label: "Forecast horizon", value: "42 days" },
            { label: "Website fields", value: "Rainfall and 2 m temperature" },
            { label: "Current ensemble", value: "Up to 100 members" },
            { label: "Anomaly reference", value: "Native 2002–2021 reforecasts" },
          ])}</dl>
          <a class="model-card__action" href="./#india">Open current forecast <span aria-hidden="true">→</span></a>
        </article>

        <article class="model-card model-card--research">
          <header>
            <div class="model-card__number">02</div>
            <div>
              <span class="model-status model-status--archive"><i></i> Added to research archive</span>
              <h2>NeuralGCM</h2>
              <p>Hybrid neural general circulation model</p>
            </div>
          </header>
          <p class="model-card__summary">A frozen stochastic-precipitation checkpoint is being evaluated on the common India benchmark. It is presented as a historical model archive, not as a live 2026 forecast source.</p>
          <dl>${factList([
            { label: "Forecast horizon", value: "42 days" },
            { label: "Available field", value: "Precipitation only" },
            { label: "Research ensemble", value: "10 stochastic members" },
            { label: "Archive coverage", value: "621 cases · 2020–2025" },
          ])}</dl>
          <div class="model-card__notice">
            <strong>Why there is no temperature tab</strong>
            <p>The public precipitation checkpoint does not provide 2 m temperature. The website will not infer or relabel an unsupported field.</p>
          </div>
        </article>
      </div>

      <section class="model-comparison" aria-labelledby="model-comparison-title">
        <header>
          <span>Like-for-like boundaries</span>
          <h2 id="model-comparison-title">What can be compared</h2>
          <p>Only overlapping, definition-matched products belong in a model comparison.</p>
        </header>
        <div class="model-comparison__table" role="table" aria-label="FuXi and NeuralGCM comparison coverage">
          <div class="model-comparison__row model-comparison__row--head" role="row">
            <span role="columnheader">Product</span>
            <span role="columnheader">FuXi-S2S</span>
            <span role="columnheader">NeuralGCM</span>
            <span role="columnheader">Website status</span>
          </div>
          <div class="model-comparison__row" role="row">
            <strong role="cell">Weekly rainfall</strong>
            <span role="cell" class="is-available">Available</span>
            <span role="cell" class="is-available">Available</span>
            <span role="cell">Historical evaluation in preparation</span>
          </div>
          <div class="model-comparison__row" role="row">
            <strong role="cell">2 m temperature</strong>
            <span role="cell" class="is-available">Available</span>
            <span role="cell" class="is-unavailable">Not provided</span>
            <span role="cell">FuXi only</span>
          </div>
          <div class="model-comparison__row" role="row">
            <strong role="cell">Current guidance</strong>
            <span role="cell" class="is-available">Published</span>
            <span role="cell" class="is-unavailable">Not operational</span>
            <span role="cell">FuXi only</span>
          </div>
        </div>
      </section>

      <section class="neuralgcm-method" aria-labelledby="neuralgcm-method-title">
        <header>
          <span>NeuralGCM benchmark contract</span>
          <h2 id="neuralgcm-method-title">A reproducible historical experiment</h2>
        </header>
        <ol>
          <li><span>01</span><div><strong>Initialize without future leakage</strong><p>ERA5 atmosphere at 00 UTC, with SST and sea ice taken from the previous day and persisted through the rollout.</p></div></li>
          <li><span>02</span><div><strong>Run the frozen checkpoint</strong><p>NeuralGCM v1 stochastic precipitation at 2.8° native resolution, sampled with ten fixed ensemble seeds.</p></div></li>
          <li><span>03</span><div><strong>Evaluate on common support</strong><p>Daily precipitation is conservatively remapped to the shared 1.5° India grid for matched historical verification.</p></div></li>
        </ol>
        <p class="neuralgcm-method__footnote">Archive results remain research evidence until the full metric and publication gates are complete. They are not weather warnings or operational guidance.</p>
      </section>
    </section>
  `;
}
