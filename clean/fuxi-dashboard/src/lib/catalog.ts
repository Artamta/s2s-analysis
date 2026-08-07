import type {
  AvailableProductKey,
  ForecastData,
  InitialConditionSourceId,
  IssueIndexData,
  ProductKey,
} from "../types";

export type CatalogSource = IssueIndexData["initial_condition_sources"][number];
export type CatalogIssue = CatalogSource["issues"][number];

const PRODUCT_KEYS: ProductKey[] = [
  "rainfall_total",
  "rainfall_anomaly",
  "temperature_mean",
  "temperature_anomaly",
];

function newest(issues: CatalogIssue[]): CatalogIssue | undefined {
  return [...issues].sort((left, right) =>
    right.initialization.localeCompare(left.initialization),
  )[0];
}

export function sourceById(
  index: IssueIndexData,
  id: InitialConditionSourceId,
): CatalogSource | undefined {
  return index.initial_condition_sources.find((source) => source.id === id);
}

export function currentGfsIssue(index: IssueIndexData): CatalogIssue | undefined {
  const gfs = sourceById(index, "gfs");
  if (!gfs) return undefined;
  const complete = gfs.issues.filter(
    (issue) =>
      issue.members === 100 &&
      issue.role === "operational_experimental" &&
      issueInteractiveAvailable(issue),
  );
  const pointer = index.current;
  if (pointer?.source_id === "gfs") {
    const pointed = complete.find((issue) => issue.id === pointer.issue_id);
    if (pointed) return pointed;
  }
  const declaredCurrent = complete.find((issue) =>
    issue.presentation === "current" ||
    (typeof issue.presentation === "object" &&
      issue.presentation.section === "current"),
  );
  if (declaredCurrent) return declaredCurrent;
  const latestId =
    index.operational_status?.latest_successful_issue ??
    index.latest_successful_issue;
  return complete.find((issue) => issue.id === latestId) ?? newest(complete);
}

export function latestReferenceIssue(
  index: IssueIndexData,
): CatalogIssue | undefined {
  const era5 = sourceById(index, "era5");
  if (!era5) return undefined;
  const pointer = index.latest_reference;
  if (pointer?.source_id === "era5") {
    const pointed = era5.issues.find((issue) => issue.id === pointer.issue_id);
    if (pointed && pointed.members === 100) return pointed;
  }
  return newest(
    era5.issues.filter(
      (issue) =>
        issue.members === 100 &&
        issue.role === "reference" &&
        issueInteractiveAvailable(issue),
    ),
  );
}

export function defaultIssueForSource(
  index: IssueIndexData,
  source: CatalogSource,
): CatalogIssue | undefined {
  if (source.id === "gfs") return currentGfsIssue(index);
  if (source.id === "era5") return latestReferenceIssue(index);
  return source.issues.find((issue) => issue.id === source.default_issue) ??
    newest(source.issues);
}

export function findIssue(
  index: IssueIndexData,
  sourceId?: string | null,
  issueId?: string | null,
): { source: CatalogSource; issue: CatalogIssue } | undefined {
  const requestedSource = index.initial_condition_sources.find(
    (source) => source.id === sourceId,
  );
  const source = requestedSource ?? sourceById(index, "gfs") ??
    index.initial_condition_sources[0];
  if (!source) return undefined;
  const requestedIssue = issueId
    ? source.issues.find((issue) => issue.id === issueId)
    : undefined;
  const issue = requestedIssue ?? defaultIssueForSource(index, source);
  return issue ? { source, issue } : undefined;
}

export function productsForIssue(
  index: IssueIndexData,
  issue: CatalogIssue,
  forecast?: ForecastData,
): Set<AvailableProductKey> {
  const declared = issue.available_products ?? forecast?.issue.available_products;
  if (declared) {
    const products = new Set<AvailableProductKey>(declared);
    if (
      (issue.capabilities?.regional_probabilities ??
        forecast?.issue.capabilities?.regional_probabilities) &&
      issue.regional_outlook
    ) {
      products.add("regional_probabilities");
    }
    if (
      issue.capabilities?.pdf ??
      issue.capabilities?.india_pdf ??
      forecast?.issue.capabilities?.pdf ??
      forecast?.issue.capabilities?.india_pdf
    ) {
      products.add("india_pdf");
    }
    return products;
  }
  const forecastProducts = forecast
    ? PRODUCT_KEYS.filter((key) => Boolean(forecast.products[key]))
    : [];
  if (forecastProducts.length > 0) {
    const products = new Set<AvailableProductKey>(forecastProducts);
    if (issue.regional_outlook) products.add("regional_probabilities");
    if (forecast?.issue.downloads?.india_pdf) products.add("india_pdf");
    return products;
  }
  // Schema v1/v2 catalogues predate per-issue capability fields. Their issue
  // exports always contain the four India fields, so this is a legacy-only
  // compatibility path. Newer catalogues must declare their products.
  if (index.schema_version <= 2) {
    const products = new Set<AvailableProductKey>(PRODUCT_KEYS);
    if (issue.regional_outlook) products.add("regional_probabilities");
    if (issue.pdf) products.add("india_pdf");
    return products;
  }
  return new Set<AvailableProductKey>();
}

export function productKeysForForecast(
  forecast: ForecastData,
  catalogProducts?: AvailableProductKey[],
): ProductKey[] {
  const declared = catalogProducts ?? forecast.issue.available_products;
  return PRODUCT_KEYS.filter(
    (key) =>
      Boolean(forecast.products[key]) &&
      Boolean(forecast.weeks[0]?.fields[key]) &&
      (!declared || declared.includes(key)),
  );
}

export function issueHasRegionalOutlook(
  index: IssueIndexData,
  issue: CatalogIssue,
  forecast?: ForecastData,
): boolean {
  const capability =
    issue.capabilities?.regional_probabilities ??
    forecast?.issue.capabilities?.regional_probabilities;
  if (capability !== undefined) return capability && Boolean(issue.regional_outlook);
  return (
    Boolean(issue.regional_outlook) &&
    productsForIssue(index, issue, forecast).has("regional_probabilities")
  );
}

export function issueChecksum(
  issue: CatalogIssue,
  asset: "forecast" | "regional_outlook" | "validation" | "pdf",
): string | undefined {
  const v3Checksums = {
    forecast: issue.checksums?.forecast_sha256,
    regional_outlook: issue.checksums?.regional_sha256,
    validation: issue.checksums?.validation_sha256,
    pdf: issue.checksums?.pdf_sha256,
  };
  const legacyChecksums = {
    forecast: issue.checksums?.forecast ?? issue.forecast_sha256,
    regional_outlook:
      issue.checksums?.regional_outlook ?? issue.regional_outlook_sha256,
    validation: issue.checksums?.validation ?? issue.validation_sha256,
    pdf: issue.checksums?.pdf ?? issue.pdf_sha256,
  };
  return v3Checksums[asset] ?? legacyChecksums[asset];
}

export function checksumVersionedPath(
  path: string,
  checksum?: string,
): string {
  if (!checksum) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}v=${encodeURIComponent(checksum.slice(0, 16))}`;
}

export function issueIsCurrent(
  index: IssueIndexData,
  source: CatalogSource,
  issue: CatalogIssue,
): boolean {
  return source.id === "gfs" && currentGfsIssue(index)?.id === issue.id;
}

export function issueClass(issue: CatalogIssue): "complete" | "limited" {
  if (typeof issue.presentation === "object") {
    return issue.presentation.member_class === "production" ? "complete" : "limited";
  }
  return issue.members === 100 ? "complete" : "limited";
}

export function issueInteractiveAvailable(issue: CatalogIssue): boolean {
  return issue.archive?.interactive_available ?? (issue.archive_available !== false);
}

export function issuePdfAvailable(issue: CatalogIssue): boolean {
  return issue.archive?.pdf_available ??
    issue.capabilities?.pdf ??
    issue.capabilities?.india_pdf ??
    Boolean(issue.pdf);
}

export function nextWednesdayOrSaturday(initialization: string): Date {
  const next = new Date(initialization);
  next.setUTCHours(0, 0, 0, 0);
  do {
    next.setUTCDate(next.getUTCDate() + 1);
  } while (next.getUTCDay() !== 3 && next.getUTCDay() !== 6);
  return next;
}
