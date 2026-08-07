export type ValidationStatus = "green" | "warning" | "failure";
export type InitialConditionSourceId = "gfs" | "era5";

export type ProductKey =
  | "rainfall_total"
  | "rainfall_anomaly"
  | "temperature_mean"
  | "temperature_anomaly";

export type AvailableProductKey =
  | ProductKey
  | "regional_probabilities"
  | "india_pdf";

export interface IssueCapabilities {
  raw_fields?: boolean;
  anomalies?: boolean;
  regional_probabilities?: boolean;
  regional_probabilities_eligible?: boolean;
  pdf?: boolean;
  india_pdf?: boolean;
}

export interface IssuePointer {
  source_id: InitialConditionSourceId;
  issue_id: string;
  initialization: string;
  members: number;
  forecast: string;
  forecast_sha256?: string;
  valid_through?: string;
  published_at?: string;
  available_products?: AvailableProductKey[];
  capabilities?: IssueCapabilities;
  regional_outlook?: string;
  pdf?: string;
  pdf_available?: boolean;
}

export type GlobalVariableKey =
  | "precipitation"
  | "temperature"
  | "z500"
  | "wind850"
  | "mslp"
  | "sst"
  | "olr"
  | "tcwv";

export type GlobalDisplayMode = "absolute" | "anomaly";

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
  fields: Partial<Record<ProductKey, number[]>>;
  summary: Partial<Record<ProductKey, FieldSummary>>;
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
    initial_condition_source: {
      id: InitialConditionSourceId;
      label: string;
      short_label: string;
      category: "operational_proxy" | "reanalysis_reference";
      availability: "near_real_time" | "delayed_reference";
      description: string;
    };
    available_products?: AvailableProductKey[];
    capabilities?: IssueCapabilities;
    downloads?: {
      compact_json?: string;
      india_pdf?: string;
      india_pdf_sha256?: string;
    };
    climatology_alignment?: {
      status?: "available" | "unavailable_outside_jjas";
      target_model_state_calendar_day: string;
      left_slot?: string;
      right_slot?: string;
      right_weight?: number;
      operation_order?: string;
      available_slot_start?: string;
      available_slot_end?: string;
      message?: string;
    };
    hindcast_years?: number[];
    observation_verification: {
      status: string;
      message: string;
    };
    initialization_comparison?: {
      status: "initialization_sensitivity_only";
      counterpart_source_id: InitialConditionSourceId;
      comparison: string;
      week1_rainfall_gfs_minus_era5_mm_day: number;
      week1_temperature_gfs_minus_era5_deg_c: number;
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
  products: Partial<Record<ProductKey, ProductDefinition>>;
  diagnostics: {
    rainfall_weekly_mean_max_mm_day: number;
    temperature_weekly_mean_max_deg_c: number;
  };
  weeks: ForecastWeek[];
}

export type TercileCategory =
  | "below_normal"
  | "near_normal"
  | "above_normal";

export interface TercileProbabilityRecord {
  below_normal: number;
  near_normal: number;
  above_normal: number;
  dominant_category: TercileCategory | "mixed";
  dominant_probability: number;
}

export interface RegionalVariableSummary {
  weekly_mean_mm_day?: number;
  anomaly_mm_day?: number;
  ensemble_spread_mm_day?: number;
  weekly_mean_deg_c?: number;
  anomaly_deg_c?: number;
  ensemble_spread_deg_c?: number;
  tercile_probability_percent: TercileProbabilityRecord;
}

export interface RegionalOutlookWeek {
  week: number;
  valid_start: string;
  valid_end: string;
  probability_fields: Record<
    "rainfall" | "temperature",
    Record<TercileCategory, number[]>
  >;
  regions: Array<{
    id: string;
    label: string;
    short_label: string;
    rainfall: RegionalVariableSummary;
    temperature: RegionalVariableSummary;
  }>;
}

export interface RegionalOutlookData {
  schema_version: 1;
  generated_at: string;
  issue: {
    initialization: string;
    source_id: InitialConditionSourceId;
    source_label: string;
    members: 100;
    lead_days: 42;
    status: "experimental";
    probability_type: "raw_ensemble_tercile_probability";
    calibration:
      | "uncalibrated_gfs_proxy"
      | "raw_ensemble_reanalysis_reference";
    forecast_sha256: string;
    climatology_sha256: string;
    hindcast_years: number[];
  };
  grid: {
    shape: [27, 27];
    spacing_degrees: 1.5;
    latitude: number[];
    longitude: number[];
    value_order: string;
  };
  region_definition: {
    name: string;
    geometry_source: IndiaAdminData["source"];
    geometry_sha256: string;
    aggregation: string;
    reference: string;
    interpretation: string;
    excluded_geometry_features: string[];
    regions: Array<{
      id: string;
      label: string;
      short_label: string;
      states_and_union_territories: string[];
      equivalent_native_grid_cells: number;
    }>;
  };
  probability_definition: {
    below_normal: string;
    near_normal: string;
    above_normal: string;
    terciles: string;
    warning: string;
  };
  weeks: RegionalOutlookWeek[];
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
  anomaly?: GlobalAnomalyDefinition;
  legend: LegendDefinition;
}

export interface GlobalAnomalyDefinition extends GlobalBinaryDefinition {
  label: string;
  short_label: string;
  units: string;
  description: string;
  baseline: {
    name: string;
    source_file: string;
    source_sha256: string;
    initialization_slot: string;
    hindcast_years: number[];
    years: 20;
    native_members_per_year: 51;
    lead_days: 42;
    weighting: string;
  };
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
  anomalies: Partial<Record<GlobalVariableKey, Uint16Array>>;
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

export interface IndiaMapGeographyData {
  schema_version: 1;
  description: string;
  view_box: [0, 0, 620, 620];
  world_path: string;
  india_outline_path: string;
  india_admin_path: string;
  sources: {
    world_countries_sha256: string;
    india_outline_sha256: string;
    india_admin_sha256: string;
  };
}

export interface IssueIndexData {
  schema_version: number;
  default_view: "india";
  default_source: InitialConditionSourceId;
  latest_successful_issue: string;
  current?: IssuePointer;
  latest_reference?: IssuePointer;
  publication?: {
    cadence_days_utc: string[];
    target_time_utc: string;
    last_successful_at?: string;
    next_expected_at?: string;
  };
  operations?: {
    status: "on_schedule" | "delayed" | "stale";
    last_successful_at?: string;
    next_expected_at?: string;
    stale_after?: string;
  };
  retention?: {
    interactive_days: number;
    pdf_days: number;
    metadata: "indefinite";
  };
  cache?: {
    catalog: "no-store";
    assets: "checksum-versioned";
  };
  published_at?: string;
  operational_status?: {
    latest_successful_issue?: string;
    latest_success_at?: string;
    next_update?: string;
    next_update_at?: string;
    stale?: boolean;
    status?: "healthy" | "delayed" | "stale";
  };
  initial_condition_sources: Array<{
    id: InitialConditionSourceId;
    label: string;
    short_label: string;
    category: "operational_proxy" | "reanalysis_reference";
    status: "experimental" | "reference";
    description: string;
    default_issue: string;
    issues: Array<{
      id: string;
      initialization: string;
      members: number;
      status: "green" | "warning";
      role: "operational_experimental" | "reference" | "rapid_prototype";
      forecast: string;
      regional_outlook?: string;
      validation?: string;
      pdf?: string;
      published_at?: string;
      available_products?: AvailableProductKey[];
      capabilities?: IssueCapabilities;
      presentation?:
        | "current"
        | "archive"
        | "limited_experiment"
        | "delayed_reference"
        | {
          section: "current" | "archive";
          label: string;
          member_class: "production" | "limited";
        };
      archive_available?: boolean;
      archive?: {
        interactive_available: boolean;
        pdf_available: boolean;
        interactive_until?: string;
        pdf_until?: string;
        metadata: "indefinite";
      };
      valid_through?: string;
      checksums?: {
        forecast?: string;
        regional_outlook?: string;
        validation?: string;
        pdf?: string;
        forecast_sha256?: string;
        regional_sha256?: string;
        validation_sha256?: string;
        pdf_sha256?: string;
      };
      forecast_sha256?: string;
      regional_outlook_sha256?: string;
      validation_sha256?: string;
      pdf_sha256?: string;
    }>;
  }>;
  available_issues: Array<{
    id: string;
    initialization: string;
    status: "green" | "warning";
    forecast: string;
  }>;
}

export interface AppData {
  index?: IssueIndexData;
  forecast?: ForecastData;
  validation?: ValidationData;
  indiaGeography?: IndiaMapGeographyData;
  regionalOutlook?: RegionalOutlookData;
  global?: GlobalForecastData;
  world?: WorldCountriesData;
  indiaAdmin?: IndiaAdminData;
}
