export type ValidationStatus = "green" | "warning" | "failure";

export type ProductKey =
  | "rainfall_total"
  | "rainfall_anomaly"
  | "temperature_mean"
  | "temperature_anomaly";

export type GlobalVariableKey =
  | "precipitation"
  | "temperature"
  | "z500"
  | "wind850"
  | "mslp"
  | "sst"
  | "olr"
  | "tcwv";

export interface LegendDefinition {
  boundaries: number[];
  colors: string[];
  under: string;
  over: string;
}

export interface ProductDefinition {
  label: string;
  short_label: string;
  description: string;
  units: string;
  baseline: string | null;
  legend: LegendDefinition;
}

export interface FieldSummary {
  india_weighted_mean: number;
  india_minimum: number;
  india_maximum: number;
}

export interface ForecastWeek {
  week: number;
  valid_start: string;
  valid_end: string;
  fields: Record<ProductKey, number[]>;
  summary: Record<ProductKey, FieldSummary>;
}

export interface ForecastData {
  schema_version: number;
  generated_at: string;
  issue: {
    initialization: string;
    information_cutoff: string;
    model_state_time: string;
    input_days: string[];
    input_source: string;
    members: number;
    lead_days: number;
    status: "green" | "warning";
    scientific_status: string;
    climatology_alignment: {
      target_model_state_calendar_day: string;
      left_slot: string;
      right_slot: string;
      right_weight: number;
      operation_order: string;
    };
    hindcast_years: number[];
    observation_verification: {
      status: string;
      message: string;
    };
  };
  grid: {
    shape: [number, number];
    spacing_degrees: number;
    latitude: number[];
    longitude: number[];
    india_mask: boolean[];
    supported_cell_count: number;
    value_order: string;
  };
  products: Record<ProductKey, ProductDefinition>;
  diagnostics: {
    rainfall_weekly_mean_max_mm_day: number;
    temperature_weekly_mean_max_deg_c: number;
  };
  weeks: ForecastWeek[];
}

export interface ValidationCheck {
  id: string;
  label: string;
  group: string;
  status: ValidationStatus;
  summary: string;
  details: Record<string, unknown>;
}

export interface ValidationData {
  schema_version: number;
  generated_at: string;
  overall_status: ValidationStatus;
  presentation_allowed: boolean;
  checks: ValidationCheck[];
  warnings: string[];
  verification_metrics?: {
    status: string;
    reason: string;
  };
}

export interface SourceEntry {
  id: string;
  name: string;
  product: string;
  period: string;
  grid: string;
  units: Record<string, string>;
  sha256: string;
  validation_status: ValidationStatus;
}

export interface SourcesData {
  schema_version: number;
  generated_at: string;
  sources: SourceEntry[];
  observation_identity: Record<string, string>;
}

export interface FormulaDefinition {
  expression: string;
  description: string;
}

export interface FormulasData {
  schema_version: number;
  formula_version: string;
  definitions: Record<string, FormulaDefinition>;
  statistics: Record<string, string>;
  baseline_separation: Record<string, string>;
}

export interface OutlineData {
  schema_version: number;
  description: string;
  geometry: {
    type: string;
    coordinates?: unknown;
    geometries?: Array<{
      type: string;
      coordinates: unknown;
    }>;
  };
}

export interface GlobalBinaryDefinition {
  path: string;
  sha256: string;
  size_bytes: number;
  dtype: "uint16-little-endian";
  offset: number;
  scale: number;
  minimum: number;
  maximum: number;
  frame_ranges: Array<{
    minimum: number;
    maximum: number;
  }>;
}

export interface GlobalVariableDefinition extends GlobalBinaryDefinition {
  label: string;
  short_label: string;
  units: string;
  description: string;
  family: "surface" | "circulation" | "ocean-convection";
  interpretation: string;
  domain?: "global" | "ocean";
  vector?: {
    u: GlobalBinaryDefinition;
    v: GlobalBinaryDefinition;
    statistic: string;
  };
  spread: GlobalSpreadDefinition;
  legend: LegendDefinition;
}

export interface GlobalSpreadDefinition extends GlobalBinaryDefinition {
  offset: 0;
  frame_area_means: number[];
  statistic: string;
}

export interface GlobalMetadata {
  schema_version: number;
  generated_at: string;
  issue: {
    initialization: string;
    members: number;
    lead_days: number;
    status: "experimental";
    public_label: string;
    input_description: string;
    ensemble_relation: string;
    display_interpolation: string;
  };
  grid: {
    shape: [number, number];
    spacing_degrees: number;
    latitude_first: number;
    latitude_last: number;
    longitude_first: number;
    longitude_last: number;
    value_order: string;
    ocean_mask: {
      path: string;
      sha256: string;
      size_bytes: number;
      dtype: "uint8";
      meaning: string;
    };
  };
  valid_period_starts: string[];
  variables: Record<GlobalVariableKey, GlobalVariableDefinition>;
  validation: {
    status: ValidationStatus;
    checks: string[];
  };
}

export interface GlobalForecastData {
  metadata: GlobalMetadata;
  fields: Partial<Record<GlobalVariableKey, Uint16Array>>;
  spreads: Partial<Record<GlobalVariableKey, Uint16Array>>;
  vectors: Partial<
    Record<GlobalVariableKey, { u: Uint16Array; v: Uint16Array }>
  >;
  oceanMask: Uint8Array;
  loadVariable: (key: GlobalVariableKey) => Promise<void>;
}

export interface WorldCountriesData {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties?: {
      NAME?: string;
      NAME_EN?: string;
      ADMIN?: string;
      LABEL_X?: number;
      LABEL_Y?: number;
      LABELRANK?: number;
      MIN_LABEL?: number;
      MAX_LABEL?: number;
      POP_EST?: number;
    };
    geometry: {
      type: "Polygon" | "MultiPolygon";
      coordinates: number[][][] | number[][][][];
    } | null;
  }>;
}

export interface IndiaAdminData {
  type: "FeatureCollection";
  name: string;
  description: string;
  source: {
    name: string;
    product: string;
    source_sha256: string;
    display_note: string;
  };
  features: Array<{
    type: "Feature";
    properties: {
      name: string;
      label: boolean;
      label_longitude: number;
      label_latitude: number;
    };
    geometry: {
      type: "Polygon" | "MultiPolygon";
      coordinates: number[][][] | number[][][][];
    };
  }>;
}

export interface AppData {
  forecast: ForecastData;
  global: GlobalForecastData;
  validation: ValidationData;
  sources: SourcesData;
  formulas: FormulasData;
  outline: OutlineData;
  world: WorldCountriesData;
  indiaAdmin: IndiaAdminData;
}
