export type Impact = "minor" | "moderate" | "serious" | "critical" | null;

export interface AxeNode {
  any?: unknown[];
  all?: unknown[];
  none?: unknown[];
  impact: Impact;
  html: string;
  target: string[];
  failureSummary?: string;
}

export interface AxeViolation {
  id: string;
  impact: Impact;
  tags: string[];
  description: string;
  help: string;
  helpUrl: string;
  nodes: AxeNode[];
}

export interface AxeReport {
  testEngine: {
    name: string;
    version: string;
  };
  testRunner: {
    name: string;
  };
  testEnvironment: {
    userAgent: string;
    windowWidth: number;
    windowHeight: number;
    orientationAngle?: number;
    orientationType?: string;
  };
  timestamp: string;
  url: string;
  toolOptions?: unknown;
  violations: AxeViolation[];
}

export interface SummaryReport {
  total_pages: number;
  total_violations: number;
  by_rule: Record<string, number>;
  by_page: Array<{
    page_index: number;
    url: string;
    violations: number;
  }>;
  scrapped_first?: string;
}

export interface IssueNode {
  key: string;
  ruleId: string;
  impact: Impact;
  help: string;
  description: string;
  helpUrl: string;
  tags: string[];
  target: string[];
  selector: string;
  html: string;
  failureSummary: string;
  index: number;
}

export interface IssueMarker extends IssueNode {
  matched: boolean;
  rect: DOMRectReadOnly | null;
}
