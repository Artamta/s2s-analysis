import type { LegendDefinition } from "../types";

interface LegendProduct {
  units: string;
  legend: LegendDefinition;
}

export function createLegend(product: LegendProduct): HTMLElement {
  const container = document.createElement("div");
  container.className = "legend";
  container.setAttribute("aria-label", `Fixed legend in ${product.units}`);

  const bar = document.createElement("div");
  bar.className = "legend__bar";
  for (const color of product.legend.colors) {
    const segment = document.createElement("span");
    segment.style.backgroundColor = color;
    bar.append(segment);
  }

  const ticks = document.createElement("div");
  ticks.className = "legend__ticks";
  product.legend.boundaries.forEach((boundary, index) => {
    const tick = document.createElement("span");
    tick.textContent = String(boundary);
    const denominator = product.legend.boundaries.length - 1;
    tick.style.left = `${(index / denominator) * 100}%`;
    ticks.append(tick);
  });

  const caption = document.createElement("div");
  caption.className = "legend__caption";
  caption.innerHTML = `<span>Fixed scale</span><strong>${product.units}</strong>`;
  container.append(bar, ticks, caption);
  return container;
}
