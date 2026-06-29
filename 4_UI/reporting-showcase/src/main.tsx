import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AlertTriangle, ArrowUpDown, ExternalLink, Eye, FileInput, Filter, RefreshCw, Search, Target } from "lucide-react";
import { flattenViolations, groupByRule, prepareHtmlSnapshot, resolveIssueElement, shortHtml } from "./report";
import type { AxeReport, Impact, IssueMarker, IssueNode, SummaryReport } from "./types";
import "./styles.css";

const discoveredHtmlFiles = Object.keys(import.meta.glob("/sites/**/*.html")).sort();
const discoveredReportFiles = Object.keys(import.meta.glob("/sites/**/page-*.json"));

interface SitePage {
  label: string;
  htmlPath: string;
  reportPath: string;
  summaryPath: string;
}

function buildSitePages(): SitePage[] {
  return discoveredHtmlFiles.map((htmlPath) => {
    const dir = htmlPath.substring(0, htmlPath.lastIndexOf("/"));
    const filename = htmlPath.substring(htmlPath.lastIndexOf("/") + 1);
    const pageNum = filename.replace(".html", "");
    const matchingReport = discoveredReportFiles.find((r) => {
      const rDir = r.substring(0, r.lastIndexOf("/"));
      const rFile = r.substring(r.lastIndexOf("/") + 1);
      return rDir === dir && rFile.startsWith(`page-${pageNum}_`);
    });
    const normalized = htmlPath.replace(/^\//, "");
    const reportPath = matchingReport ? matchingReport.replace(/^\//, "") : "";
    const summaryPath = `${dir.replace(/^\//, "")}/summary.json`;
    return { label: `${dir.split("/").pop()}/${filename}`, htmlPath: normalized, reportPath, summaryPath };
  });
}

const sitePages = buildSitePages();
const defaultPage = sitePages[0];
const defaultHtmlPath = defaultPage.htmlPath;
const defaultReportPath = defaultPage.reportPath;
const defaultSummaryPath = defaultPage.summaryPath;
const impactOrder: Array<Exclude<Impact, null>> = ["critical", "serious", "moderate", "minor"];

type LoadState =
  | { status: "loading"; message: string }
  | { status: "ready"; html: string; report: AxeReport; summary: SummaryReport }
  | { status: "error"; message: string };

function App() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const markerTimerRef = useRef<number | null>(null);
  const [htmlPath, setHtmlPath] = useState(defaultHtmlPath);
  const [reportPath, setReportPath] = useState(defaultReportPath);
  const [summaryPath, setSummaryPath] = useState(defaultSummaryPath);
  const [loadState, setLoadState] = useState<LoadState>({ status: "loading", message: "Loading default report" });
  const [markers, setMarkers] = useState<IssueMarker[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [ruleFilter, setRuleFilter] = useState("all");
  const [impactFilter, setImpactFilter] = useState<Impact | "all">("all");
  const [query, setQuery] = useState("");

  const loadInputs = useCallback(async () => {
    setLoadState({ status: "loading", message: "Loading report inputs" });
    setMarkers([]);
    setSelectedKey(null);

    try {
      const [htmlResponse, reportResponse, summaryResponse] = await Promise.all([
        fetchProjectFile(htmlPath),
        fetchProjectFile(reportPath),
        summaryPath.trim() ? fetchProjectFile(summaryPath) : Promise.resolve(null),
      ]);

      const html = await htmlResponse.text();
      const report = (await reportResponse.json()) as AxeReport;
      const summary = summaryResponse
        ? ((await summaryResponse.json()) as SummaryReport)
        : createSummaryFromReport(report);

      setLoadState({ status: "ready", html, report, summary });
    } catch (error) {
      setLoadState({
        status: "error",
        message: error instanceof Error ? error.message : "Unable to load report inputs",
      });
    }
  }, [htmlPath, reportPath, summaryPath]);

  const selectPage = useCallback(
    (page: SitePage) => {
      setHtmlPath(page.htmlPath);
      setReportPath(page.reportPath);
      setSummaryPath(page.summaryPath);
    },
    [],
  );

  useEffect(() => {
    void loadInputs();
  }, [htmlPath, reportPath, summaryPath]);

  const report = loadState.status === "ready" ? loadState.report : null;
  const summary = loadState.status === "ready" ? loadState.summary : null;
  const issues = useMemo(() => (report ? flattenViolations(report) : []), [report]);
  const htmlSnapshot = useMemo(() => (loadState.status === "ready" ? prepareHtmlSnapshot(loadState.html) : ""), [loadState]);
  const rules = useMemo(() => Object.keys(groupByRule(issues)).sort(), [issues]);
  const impacts = useMemo(() => {
    const present = new Set(issues.map((issue) => issue.impact).filter(Boolean) as Array<Exclude<Impact, null>>);
    return impactOrder.filter((impact) => present.has(impact));
  }, [issues]);

  const filteredIssues = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return issues.filter((issue) => {
      const matchesRule = ruleFilter === "all" || issue.ruleId === ruleFilter;
      const matchesImpact = impactFilter === "all" || issue.impact === impactFilter;
      const matchesQuery =
        normalizedQuery.length === 0 ||
        `${issue.ruleId} ${issue.selector} ${issue.help} ${issue.failureSummary} ${issue.html}`
          .toLowerCase()
          .includes(normalizedQuery);

      return matchesRule && matchesImpact && matchesQuery;
    });
  }, [impactFilter, issues, query, ruleFilter]);

  const filteredKeys = useMemo(() => new Set(filteredIssues.map((issue) => issue.key)), [filteredIssues]);
  const visibleMarkers = markers.filter((marker) => filteredKeys.has(marker.key) && marker.rect);
  const selectedIssue = issues.find((issue) => issue.key === selectedKey) ?? filteredIssues[0] ?? null;
  const selectedMarker = markers.find((marker) => marker.key === selectedIssue?.key);
  const matchedCount = markers.filter((marker) => marker.matched).length;

  const refreshMarkers = useCallback(() => {
    const iframe = iframeRef.current;
    const document = iframe?.contentDocument;
    if (!document || issues.length === 0) {
      return;
    }

    const nextMarkers = issues.map((issue) => {
      const element = resolveIssueElement(document, issue);
      const rect = element?.getBoundingClientRect() ?? null;
      const hasVisibleBox = Boolean(rect && rect.width > 0 && rect.height > 0);

      return {
        ...issue,
        matched: Boolean(element),
        rect: hasVisibleBox ? rect : null,
      };
    });

    setMarkers(nextMarkers);
  }, [issues]);

  const scheduleMarkerRefresh = useCallback(() => {
    refreshMarkers();
    window.requestAnimationFrame(refreshMarkers);

    if (markerTimerRef.current) {
      window.clearTimeout(markerTimerRef.current);
    }

    markerTimerRef.current = window.setTimeout(refreshMarkers, 350);
  }, [refreshMarkers]);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe || loadState.status !== "ready") {
      return undefined;
    }

    const handleLoad = () => {
      scheduleMarkerRefresh();
      iframe.contentWindow?.addEventListener("scroll", refreshMarkers, { passive: true });
      iframe.contentWindow?.addEventListener("resize", refreshMarkers);
    };

    iframe.addEventListener("load", handleLoad);
    window.addEventListener("resize", refreshMarkers);
    scheduleMarkerRefresh();

    return () => {
      iframe.removeEventListener("load", handleLoad);
      window.removeEventListener("resize", refreshMarkers);
      iframe.contentWindow?.removeEventListener("scroll", refreshMarkers);
      iframe.contentWindow?.removeEventListener("resize", refreshMarkers);

      if (markerTimerRef.current) {
        window.clearTimeout(markerTimerRef.current);
      }
    };
  }, [loadState.status, refreshMarkers, scheduleMarkerRefresh]);

  useEffect(() => {
    if (filteredIssues.length === 0) {
      setSelectedKey(null);
      return;
    }

    if (!selectedKey || !filteredKeys.has(selectedKey)) {
      setSelectedKey(filteredIssues[0].key);
    }
  }, [filteredIssues, filteredKeys, selectedKey]);

  const selectIssue = useCallback(
    (issue: IssueNode) => {
      setSelectedKey(issue.key);

      const document = iframeRef.current?.contentDocument;
      if (!document) {
        return;
      }

      const element = resolveIssueElement(document, issue);
      element?.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
      window.setTimeout(refreshMarkers, 350);
    },
    [refreshMarkers],
  );

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Accessibility report controls">
        <header className="sidebar-header">
          <div>
            <p className="eyebrow">axe-core visual report</p>
            <h1>Accessibility issue map</h1>
          </div>
          <span className="score" aria-label={`${summary?.total_violations ?? issues.length} violations`}>
            {summary?.total_violations ?? issues.length}
          </span>
        </header>

        <section className="input-panel" aria-label="Report input paths">
          <label className="field">
            <span>
              <FileInput size={14} aria-hidden="true" />
              HTML input
            </span>
            <select value={htmlPath} onChange={(event) => {
              const page = sitePages.find((p) => p.htmlPath === event.target.value);
              if (page) selectPage(page);
            }}>
              {sitePages.map((page) => (
                <option key={page.htmlPath} value={page.htmlPath}>
                  {page.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>
              <FileInput size={14} aria-hidden="true" />
              Axe report
            </span>
            <input value={reportPath} onChange={(event) => setReportPath(event.target.value)} />
          </label>
          <label className="field">
            <span>
              <FileInput size={14} aria-hidden="true" />
              Summary
            </span>
            <input value={summaryPath} onChange={(event) => setSummaryPath(event.target.value)} />
          </label>
          <button className="load-button" type="button" onClick={() => void loadInputs()}>
            <RefreshCw size={14} aria-hidden="true" />
            Load inputs
          </button>
          {loadState.status !== "ready" && <p className={`load-message ${loadState.status}`}>{loadState.message}</p>}
        </section>

        <section className="metrics" aria-label="Report summary">
          <Metric label="Rules" value={rules.length.toString()} />
          <Metric label="Matched" value={`${matchedCount}/${issues.length}`} />
          <Metric
            label="Viewport"
            value={report ? `${report.testEnvironment.windowWidth}x${report.testEnvironment.windowHeight}` : "-"}
          />
        </section>

        <section className="controls" aria-label="Issue filters">
          <label className="field">
            <span>
              <Filter size={14} aria-hidden="true" />
              Rule
            </span>
            <select value={ruleFilter} onChange={(event) => setRuleFilter(event.target.value)}>
              <option value="all">All rules</option>
              {rules.map((rule) => (
                <option key={rule} value={rule}>
                  {rule} ({summary?.by_rule[rule] ?? 0})
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>
              <AlertTriangle size={14} aria-hidden="true" />
              Impact
            </span>
            <select value={impactFilter ?? "all"} onChange={(event) => setImpactFilter(event.target.value as Impact | "all")}>
              <option value="all">All impacts</option>
              {impacts.map((impact) => (
                <option key={impact} value={impact}>
                  {impact}
                </option>
              ))}
            </select>
          </label>

          <label className="field search-field">
            <span>
              <Search size={14} aria-hidden="true" />
              Search
            </span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Selector, rule, summary" />
          </label>
        </section>

        <section className="issue-list" aria-label="Accessibility issues">
          <div className="list-heading">
            <span>{filteredIssues.length} visible issues</span>
            <button type="button" onClick={scheduleMarkerRefresh}>
              <RefreshCw size={14} aria-hidden="true" />
              Refresh
            </button>
          </div>

          {filteredIssues.map((issue) => (
            <button
              className={`issue-row ${selectedKey === issue.key ? "selected" : ""}`}
              key={issue.key}
              type="button"
              onClick={() => selectIssue(issue)}
            >
              <span className={`impact-dot ${issue.impact ?? "unknown"}`} aria-hidden="true" />
              <span className="issue-copy">
                <strong>{issue.ruleId}</strong>
                <span>{issue.selector}</span>
              </span>
            </button>
          ))}
        </section>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Input</p>
            <h2>{htmlPath}</h2>
          </div>
          <div className="report-meta">
            <span>{report ? `${report.testEngine.name} ${report.testEngine.version}` : "No report loaded"}</span>
            <span>{report ? new Date(report.timestamp).toLocaleString() : "-"}</span>
          </div>
        </header>

        <div className="browser-frame">
          {loadState.status === "ready" ? (
            <>
              <iframe
                key={`${htmlPath}:${reportPath}`}
                ref={iframeRef}
                title="Rendered accessibility fixture"
                sandbox="allow-same-origin"
                srcDoc={htmlSnapshot}
                onLoad={scheduleMarkerRefresh}
              />
              <div className="overlay" aria-hidden="true">
                {visibleMarkers.map((marker) => (
                  <Marker
                    key={marker.key}
                    marker={marker}
                    selected={selectedKey === marker.key}
                    onSelect={() => selectIssue(marker)}
                  />
                ))}
              </div>
            </>
          ) : (
            <div className="frame-state">{loadState.message}</div>
          )}
        </div>
      </main>

      <aside className="inspector" aria-label="Selected issue details">
        {selectedIssue ? (
          <IssueDetails issue={selectedIssue} marker={selectedMarker} markersResolved={markers.length > 0} />
        ) : (
          <div className="empty-state">No issue selected.</div>
        )}
      </aside>
    </div>
  );
}

async function fetchProjectFile(filePath: string) {
  const normalizedPath = normalizeProjectPath(filePath);
  const response = await fetch(normalizedPath);

  if (!response.ok) {
    throw new Error(`Could not load ${filePath}: ${response.status} ${response.statusText}`);
  }

  return response;
}

function normalizeProjectPath(filePath: string) {
  const trimmed = filePath.trim();

  if (/^https?:\/\//.test(trimmed)) {
    return trimmed;
  }

  return `/${trimmed.replace(/^\.?\//, "")}`;
}

function createSummaryFromReport(report: AxeReport): SummaryReport {
  const byRule = Object.fromEntries(report.violations.map((violation) => [violation.id, violation.nodes.length]));

  return {
    total_pages: 1,
    total_violations: report.violations.reduce((total, violation) => total + violation.nodes.length, 0),
    by_rule: byRule,
    by_page: [
      {
        page_index: 0,
        url: report.url,
        violations: report.violations.reduce((total, violation) => total + violation.nodes.length, 0),
      },
    ],
  };
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Marker({ marker, selected, onSelect }: { marker: IssueMarker; selected: boolean; onSelect: () => void }) {
  if (!marker.rect) {
    return null;
  }

  const style = {
    left: marker.rect.left,
    top: marker.rect.top,
    width: Math.max(marker.rect.width, 18),
    height: Math.max(marker.rect.height, 18),
  };

  return (
    <button
      type="button"
      className={`marker ${marker.impact ?? "unknown"} ${selected ? "selected" : ""}`}
      style={style}
      onClick={onSelect}
      title={`${marker.ruleId}: ${marker.help}`}
    >
      <span>{marker.index + 1}</span>
    </button>
  );
}

function IssueDetails({
  issue,
  marker,
  markersResolved,
}: {
  issue: IssueNode;
  marker: IssueMarker | undefined;
  markersResolved: boolean;
}) {
  const matchLabel = !markersResolved
    ? "Resolving selector"
    : marker?.matched
      ? marker.rect
        ? "Visible marker"
        : "Matched off-screen/hidden"
      : "Selector not found";
  const matchClass = marker?.matched ? "matched" : markersResolved ? "unmatched" : "pending";

  return (
    <div className="details">
      <div className="details-header">
        <span className={`impact-pill ${issue.impact ?? "unknown"}`}>{issue.impact ?? "unknown"}</span>
        <span className={`match ${matchClass}`}>
          <Target size={14} aria-hidden="true" />
          {matchLabel}
        </span>
      </div>

      <h2>{issue.help}</h2>
      <p>{issue.description}</p>

      <dl>
        <div>
          <dt>Rule</dt>
          <dd>{issue.ruleId}</dd>
        </div>
        <div>
          <dt>Selector</dt>
          <dd>{issue.selector}</dd>
        </div>
      </dl>

      <section>
        <h3>
          <ArrowUpDown size={16} aria-hidden="true" />
          Failure summary
        </h3>
        <pre>{issue.failureSummary}</pre>
      </section>

      <section>
        <h3>
          <Eye size={16} aria-hidden="true" />
          Source excerpt
        </h3>
        <code>{shortHtml(issue.html)}</code>
      </section>

      <a className="help-link" href={issue.helpUrl} target="_blank" rel="noreferrer">
        Open axe rule reference
        <ExternalLink size={14} aria-hidden="true" />
      </a>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
