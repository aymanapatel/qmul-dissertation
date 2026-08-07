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
  confidence?: number;
  requires_human_review?: boolean;
  human_review_reasons?: string[];
  validation_steps?: string[];
  model_evidence?: {
    graph_view: "a11y-tree" | "rendered-visual";
    architecture: string;
    detector_id: string;
    probability: number;
    threshold: number;
    routing_status: string;
    routing_confidence: number;
    evidence: { selector?: string; visual?: Record<string, unknown> };
  };
  model?: string;
  response_id?: string | null;
  usage?: Record<string, unknown>;
  generation_status: "completed" | "failed";
  error?: string;
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
  specialist: {
    architecture: string;
    training_artifacts: string;
    fusion_policy: string;
    finding_count: number;
    model_runs: Array<{
      view: string;
      architecture: string;
      axe_used_for_prediction: boolean;
      node_count: number;
      edge_count: number;
      checkpoint_sha256: string;
    }>;
  };
  safety: string;
}

export interface Job {
  job_id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed";
  error?: string;
}
