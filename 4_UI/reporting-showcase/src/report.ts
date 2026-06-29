import type { AxeReport, IssueNode } from "./types";

export function flattenViolations(report: AxeReport): IssueNode[] {
  return report.violations.flatMap((violation) =>
    violation.nodes.map((node, index) => ({
      key: `${violation.id}-${index}-${node.target.join("|")}`,
      ruleId: violation.id,
      impact: node.impact ?? violation.impact,
      help: violation.help,
      description: violation.description,
      helpUrl: violation.helpUrl,
      tags: violation.tags,
      target: node.target,
      selector: formatSelector(node.target),
      html: node.html,
      failureSummary: node.failureSummary ?? "No failure summary provided.",
      index,
    })),
  );
}

export function groupByRule(issues: IssueNode[]) {
  return issues.reduce<Record<string, IssueNode[]>>((groups, issue) => {
    groups[issue.ruleId] = groups[issue.ruleId] ?? [];
    groups[issue.ruleId].push(issue);
    return groups;
  }, {});
}

export function formatSelector(target: string[]) {
  return target.join(" ");
}

export function selectorCandidates(target: string[]) {
  const candidates = [formatSelector(target), ...target.slice().reverse()];
  return Array.from(new Set(candidates.filter(Boolean)));
}

export function resolveIssueElement(document: Document, issue: IssueNode) {
  for (const selector of selectorCandidates(issue.target)) {
    try {
      const element = document.querySelector(selector);
      if (isMeasurableElement(element)) {
        return element;
      }
    } catch {
      continue;
    }
  }

  return null;
}

function isMeasurableElement(element: Element | null): element is Element {
  return Boolean(element && "getBoundingClientRect" in element);
}

export function prepareHtmlSnapshot(html: string) {
  const baseTag = '<base href="https://www.bt.com/" />';
  const withoutScripts = html.replaceAll("<script", "<script type=\"application/x-blocked\"");

  if (withoutScripts.includes("<head>")) {
    return withoutScripts.replace("<head>", `<head>${baseTag}`);
  }

  return `${baseTag}${withoutScripts}`;
}

export function shortHtml(html: string) {
  return html.replace(/\s+/g, " ").trim();
}
