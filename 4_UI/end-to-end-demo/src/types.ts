export type Decision = "propose" | "requires_human_review" | "leave_unchanged";

export interface RepairOperation {
  operation: string;
  selector: string;
  attribute_name: string | null;
  css_property: string | null;
  new_value: string | null;
}

export interface Suggestion {
  finding_id: string;
  rule_id: string;
  impact: string | null;
  target: string[];
  help?: string;
  decision?: Decision;
  rationale?: string;
  expected_resolution?: string;
  operations?: RepairOperation[];
  inspected_visual_elements?: Array<{
    source: string;
    selector: string;
    tag: string;
    text: string;
    bounds: { x: number; y: number; width: number; height: number } | null;
    foreground_rgb: number[] | null;
    background_rgb: number[] | null;
    contrast_ratio: number | null;
    required_contrast_ratio: number | null;
    contrast_failure: boolean;
    contrast_failure_source: string | null;
  }>;
  confidence?: number;
  requires_human_review?: boolean;
  human_review_reasons?: string[];
  validation_steps?: string[];
  model_evidence?: {
    evidence_kind?: "trained_prediction" | "measured_visual";
    graph_view: "a11y-tree" | "rendered-visual";
    architecture: string;
    detector_id: string;
    probability: number | null;
    threshold: number | null;
    routing_status: string;
    routing_confidence: number;
    evidence: { selector?: string; visual?: Record<string, unknown> };
  };
  model?: string;
  response_id?: string | null;
  usage?: Record<string, unknown>;
  api_trace?: {
    request: {
      method: string;
      endpoint: string;
      api_mode: string;
      model: string;
      system_prompt: string;
      user_prompt: Record<string, unknown>;
      response_format: string;
    };
    response: Record<string, unknown>;
  };
  generation_status: "completed" | "failed";
  error?: string;
}

export interface ModelRule {
  rule_id: string;
  probability: number;
  threshold: number;
  predicted_fail: boolean;
  node_index: number;
  wcag_ids?: string[];
}

export interface ModelRun {
  view: "a11y-tree" | "rendered-visual";
  architecture: "mlp" | "graphsage" | "gat";
  axe_used_for_prediction: boolean;
  node_count: number;
  edge_count: number;
  checkpoint_sha256: string;
  feature_contract?: Record<string, unknown>;
  rules: ModelRule[];
  findings: Array<Record<string, unknown>>;
}

export interface VisualElement {
  snapshot_node_id: string;
  selector: string;
  tag: string;
  text: string;
  bounds: { x: number; y: number; width: number; height: number };
  visible: boolean;
  in_viewport: boolean;
  clipped: boolean;
  visual: {
    foreground_rgb?: number[];
    background_rgb?: number[];
    contrast_ratio: number;
    required_contrast_ratio: number;
    contrast_deficit?: number;
    font_size?: number;
    font_weight?: number;
    opacity?: number;
    has_direct_text?: boolean;
  };
  numeric_contrast_failure: boolean;
  contrast_failure: boolean;
  contrast_failure_source?: string | null;
}

export interface VisualEvidence {
  source: string;
  contrast_highlight_policy: string;
  viewport: { width?: number; height?: number };
  canvas: { width: number; height: number };
  element_count: number;
  contrast_failure_count: number;
  elements: VisualElement[];
  contrast_failures: VisualElement[];
}

export interface SuggestionResult {
  schema_version: number;
  status: "completed" | "partial" | "scan_failed";
  run_id: string;
  source_url: string;
  final_url: string;
  screenshot_url?: string;
  violation_count: number;
  affected_node_count: number;
  violations_by_impact: Record<string, number>;
  suggestion_count: number;
  suggestions: Suggestion[];
  visual_evidence?: VisualEvidence | null;
  specialist: {
    architectures: string[];
    training_artifacts: string;
    fusion_policy: string;
    finding_count: number;
    model_runs: ModelRun[];
  };
  application_api: Record<string, unknown>;
  safety: string;
}

export interface ProgressEvent {
  event_id: string;
  label: string;
  status: "running" | "completed" | "failed" | "skipped";
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  details: Record<string, unknown>;
}

export interface Job {
  job_id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed";
  error?: string;
  progress?: {
    current_stage: string;
    label: string;
    completed?: number;
    total?: number;
    events: ProgressEvent[];
  };
  links?: { self: string; result: string; artifacts: string };
}
