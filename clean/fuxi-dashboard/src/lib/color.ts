import type { LegendDefinition } from "../types";

export function colorFor(value: number, legend: LegendDefinition): string {
  const { boundaries, colors, under, over } = legend;
  if (value < boundaries[0]) return under;
  if (value >= boundaries[boundaries.length - 1]) return over;
  for (let index = 0; index < boundaries.length - 1; index += 1) {
    if (value >= boundaries[index] && value < boundaries[index + 1]) {
      return colors[index];
    }
  }
  return colors[colors.length - 1];
}

export function formatValue(value: number, units: string): string {
  const absolute = Math.abs(value);
  const digits = absolute >= 100 ? 0 : absolute >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units}`;
}
