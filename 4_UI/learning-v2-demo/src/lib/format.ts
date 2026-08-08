import type { RepairOperation, Suggestion } from "../types";

export const architectureLabel = (value: string) =>
  value === "mlp" ? "MLP" : value === "gat" ? "GAT" : "GraphSAGE";

export const viewLabel = (value: string) =>
  value === "a11y-tree" ? "Accessibility tree" : "Rendered visual";

export function decisionLabel(suggestion: Suggestion): string {
  if (suggestion.generation_status === "failed") return "Generation failed";
  if (suggestion.decision === "requires_human_review") return "Human review";
  if (suggestion.decision === "leave_unchanged") return "Leave unchanged";
  return "Suggested change";
}

export function operationText(operation: RepairOperation): string {
  if (operation.operation === "set_attribute") {
    return `Set ${operation.attribute_name}="${operation.new_value}"`;
  }
  if (operation.operation === "remove_attribute") return `Remove ${operation.attribute_name}`;
  if (operation.operation === "set_style_property") {
    return `Set ${operation.css_property}: ${operation.new_value}`;
  }
  if (operation.operation === "insert_label_before") return `Insert label “${operation.new_value}”`;
  if (operation.operation === "replace_text") return `Replace text with “${operation.new_value}”`;
  return operation.operation.replaceAll("_", " ");
}
